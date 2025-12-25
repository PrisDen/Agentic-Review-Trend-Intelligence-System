"""
Semantic Normalization Agent.

Converts raw review text into clean, intent-focused statements
using LLM-based extraction and classification.
"""

import json
import logging
from typing import List, Optional
import google.generativeai as genai

from src.models.review import Review
from src.models.statement import Statement

logger = logging.getLogger(__name__)


# LLM Prompts (as designed in normalization_agent_design.md)
SYSTEM_PROMPT = """You are a product analytics assistant that extracts structured feedback from app reviews.

Your task:
1. Read the user's review text
2. Extract distinct complaints, requests, or feedback items
3. Convert each item into a clean, concise statement (3-10 words)
4. Classify each as "issue", "request", or "feedback"
5. Remove emotional language and preserve only the core concern

Rules:
- "issue" = Something is broken, wrong, or causing problems
- "request" = User wants a new feature or capability
- "feedback" = Neutral observation or praise (not actionable)
- Remove app name, emojis, profanity, and filler words
- Preserve technical specifics (e.g., "crashes on login" not "app broken")
- If review has no actionable content, return empty array
- One review may produce 0-5 statements
- Assign confidence based on how explicit the complaint is
- When multiple complaints are causally related (X happens WHEN/DURING Y), combine them into a single statement

Output valid JSON only."""


def _construct_user_prompt(review: Review) -> str:
    """Construct user prompt from review."""
    return f"""Review Text: "{review.text}"
Rating: {review.rating}/5
Date: {review.date}

Extract normalized statements as JSON:
{{
  "statements": [
    {{
      "normalized_statement": "...",
      "type": "issue|request|feedback",
      "confidence": "high|medium|low"
    }}
  ]
}}"""


class SemanticNormalizationAgent:
    """
    Converts raw reviews into normalized statements.
    
    Uses LLM (Gemini) to:
    1. Extract distinct concerns from review text
    2. Normalize to clean, concise statements
    3. Classify as issue/request/feedback
    4. Assign confidence scores
    """
    
    def __init__(
        self,
        api_key: str,
        model_name: str = "gemini-1.5-flash",
        temperature: float = 0.0,
        min_confidence: str = "medium",
        max_retries: int = 3,
        timeout_seconds: int = 30
    ):
        """
        Initialize normalization agent.
        
        Args:
            api_key: Gemini API key
            model_name: Gemini model to use
            temperature: LLM temperature (0.0 for deterministic)
            min_confidence: Minimum confidence level to return ("high", "medium", "low")
            max_retries: Number of retries on API failure
            timeout_seconds: API request timeout
        """
        self.model_name = model_name
        self.temperature = temperature
        self.min_confidence = min_confidence
        self.max_retries = max_retries
        self.timeout_seconds = timeout_seconds
        
        # Configure Gemini
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(
            model_name=model_name,
            generation_config={
                "temperature": temperature,
                "response_mime_type": "application/json"
            },
            system_instruction=SYSTEM_PROMPT
        )
        
        logger.info(f"Initialized SemanticNormalizationAgent with model={model_name}, temp={temperature}")
    
    def normalize(self, review: Review) -> List[Statement]:
        """
        Convert raw review into normalized statements.
        
        Args:
            review: Raw review object
        
        Returns:
            List of normalized statements (may be empty for non-actionable reviews)
        """
        # Handle empty review text
        if not review.text or len(review.text.strip()) == 0:
            logger.debug(f"Empty review text for {review.review_id}")
            return []
        
        # Construct prompt
        user_prompt = _construct_user_prompt(review)
        
        # Call LLM with retry logic
        for attempt in range(self.max_retries):
            try:
                response = self.model.generate_content(user_prompt)
                statements = self._parse_llm_response(response.text, review.review_id)
                
                # Filter by confidence if configured
                if self.min_confidence != "low":
                    statements = self._filter_by_confidence(statements)
                
                logger.debug(f"Normalized {review.review_id}: {len(statements)} statements extracted")
                return statements
                
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse LLM JSON response (attempt {attempt + 1}): {e}")
                if attempt == self.max_retries - 1:
                    logger.warning(f"Max retries reached for {review.review_id}, returning empty statements")
                    return []
                    
            except Exception as e:
                logger.error(f"LLM API error (attempt {attempt + 1}): {e}")
                if attempt == self.max_retries - 1:
                    logger.warning(f"Max retries reached for {review.review_id}, returning empty statements")
                    return []
        
        return []
    
    def _parse_llm_response(self, response_text: str, review_id: str) -> List[Statement]:
        """
        Parse LLM JSON response into Statement objects.
        
        Args:
            response_text: Raw JSON string from LLM
            review_id: Review ID for logging
        
        Returns:
            List of Statement objects
        
        Raises:
            json.JSONDecodeError: If response is not valid JSON
        """
        data = json.loads(response_text)
        
        # Validate structure
        if "statements" not in data:
            logger.warning(f"LLM response missing 'statements' field for {review_id}")
            return []
        
        statements = []
        for item in data["statements"]:
            try:
                statement = Statement(
                    normalized_statement=item["normalized_statement"],
                    type=item["type"],
                    confidence=item["confidence"]
                )
                statements.append(statement)
            except (KeyError, ValueError) as e:
                logger.warning(f"Invalid statement in LLM response for {review_id}: {e}")
                continue
        
        return statements
    
    def _filter_by_confidence(self, statements: List[Statement]) -> List[Statement]:
        """
        Filter statements by minimum confidence level.
        
        Confidence hierarchy: low < medium < high
        """
        confidence_order = {"low": 0, "medium": 1, "high": 2}
        min_level = confidence_order[self.min_confidence]
        
        filtered = [
            s for s in statements
            if confidence_order[s.confidence] >= min_level
        ]
        
        if len(filtered) < len(statements):
            logger.debug(f"Filtered {len(statements) - len(filtered)} low-confidence statements")
        
        return filtered


# Design Rationale and Trade-offs:
#
# 1. Why temperature=0.0?
#    - Deterministic output: same review → same statements
#    - Prevents topic registry churn from stochastic variations
#    - Per pulsegen.md emphasis on determinism
#    - Trade-off: Less creative extractions, but consistency more important
#
# 2. Why retry logic with exponential backoff NOT implemented?
#    - Simple retry (3x) is sufficient for transient failures
#    - Exponential backoff adds complexity for marginal benefit
#    - Daily batch processing tolerates occasional failures
#    - Trade-off: May hit rate limits faster, but unlikely with our volume
#
# 3. Why return empty list on failure instead of raising exception?
#    - Graceful degradation: one bad review doesn't kill entire day
#    - Aligns with pipeline design (continue on non-critical failures)
#    - Trade-off: Silent data loss if many reviews fail, mitigated by logging
#
# 4. Why filter by confidence after LLM call?
#    - LLM always extracts all statements (completeness)
#    - Filtering is a deployment decision, not extraction decision
#    - Allows tuning confidence threshold without re-running normalization
#    - Trade-off: Wasted LLM tokens on low-confidence extractions
#
# 5. Why Gemini instead of OpenAI?
#    - Gemini Flash is faster and cheaper for this use case
#    - JSON mode built-in (response_mime_type)
#    - Easy to swap to GPT-4 if needed (just change model_name)
#    - Trade-off: None, both models perform well on extraction tasks
#
# 6. Why system_instruction instead of prepending to each prompt?
#    - Cleaner prompt construction
#    - Gemini optimizes system instructions separately
#    - Reduces token usage (system prompt not repeated per call)
#    - Trade-off: Requires Gemini 1.5+, not backwards compatible
