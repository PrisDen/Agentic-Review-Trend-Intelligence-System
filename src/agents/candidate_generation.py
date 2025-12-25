"""
Topic Candidate Generator.

Converts normalized statements into topic candidates with human-readable labels.
"""

import logging
import json
from typing import Optional
import google.generativeai as genai

from src.models.statement import Statement
from src.models.topic import TopicCandidate

logger = logging.getLogger(__name__)


# LLM Prompt for topic label generation
SYSTEM_PROMPT = """You are a product analytics assistant that creates concise topic labels from user feedback.

Your task:
1. Read a normalized statement from a review
2. Generate a short, human-readable topic label (3-8 words)
3. The label should be PM-consumable (clear, professional, no jargon)
4. Preserve key specifics but remove redundancy

Rules:
- Keep labels concise (3-8 words ideal)
- Use simple, professional language
- Preserve technical terms if important (e.g., "login", "payment")
- Remove generic words like "issue with" or "problem with"
- Use noun phrases, not full sentences

Examples:
- "Delivery partner rude behavior" → "Delivery partner rude"
- "App crashes when I login with Google" → "App crashes on Google login"
- "Please add dark mode feature" → "Add dark mode"

Output valid JSON only."""


def _construct_user_prompt(statement: Statement) -> str:
    """Construct user prompt from statement."""
    return f"""Normalized Statement: "{statement.normalized_statement}"
Type: {statement.type}

Generate a concise topic label as JSON:
{{
  "topic_label": "..."
}}"""


class TopicCandidateGenerator:
    """
    Generates topic candidates from normalized statements.
    
    Uses LLM to create short, PM-readable topic labels that will be used
    as canonical labels in the registry.
    """
    
    def __init__(
        self,
        api_key: str,
        model_name: str = "gemini-1.5-flash",
        temperature: float = 0.0,
        max_retries: int = 3
    ):
        """
        Initialize candidate generator.
        
        Args:
            api_key: Gemini API key
            model_name: Model to use for label generation
            temperature: LLM temperature (0.0 for deterministic)
            max_retries: Retry attempts on API failure
        """
        self.model_name = model_name
        self.temperature = temperature
        self.max_retries = max_retries
        
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
        
        logger.info(f"Initialized TopicCandidateGenerator with model={model_name}")
    
    def generate(self, statement: Statement) -> Optional[TopicCandidate]:
        """
        Generate topic candidate from normalized statement.
        
        Args:
            statement: Normalized statement
        
        Returns:
            TopicCandidate with label and type, or None on failure
        """
        # Construct prompt
        user_prompt = _construct_user_prompt(statement)
        
        # Call LLM with retry logic
        for attempt in range(self.max_retries):
            try:
                response = self.model.generate_content(user_prompt)
                data = json.loads(response.text)
                
                if "topic_label" not in data:
                    logger.warning("LLM response missing 'topic_label' field")
                    continue
                
                topic_label = data["topic_label"]
                
                # Create candidate
                candidate = TopicCandidate(
                    label=topic_label,
                    type=statement.type,
                    # embedding and source_review_id will be set by consolidation agent
                )
                
                logger.debug(f"Generated topic candidate: '{topic_label}'")
                return candidate
                
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse LLM JSON (attempt {attempt + 1}): {e}")
                if attempt == self.max_retries - 1:
                    logger.warning("Max retries reached, returning None")
                    return None
                    
            except Exception as e:
                logger.error(f"LLM API error (attempt {attempt + 1}): {e}")
                if attempt == self.max_retries - 1:
                    logger.warning("Max retries reached, returning None")
                    return None
        
        return None


# Design Rationale and Trade-offs:
#
# 1. Why separate agent for label generation instead of doing it in normalization?
#    - Separation of concerns: normalization extracts intent, labeling creates PM labels
#    - Allows tuning each prompt independently
#    - Easier to test in isolation
#    - Trade-off: Extra LLM call per statement (~$0.0001 each), but cleaner architecture
#
# 2. Why temperature=0.0?
#    - Same statement should always produce same label
#    - Prevents registry churn from stochastic label variations
#    - Consistency more important than creativity for labels
#    - Trade-off: Less diverse labels, but determinism is critical
#
# 3. Why return None on failure instead of using statement text directly?
#    - Signals failure clearly to caller
#    - Caller can decide whether to skip or retry
#    - Prevents polluting registry with non-PM-friendly labels
#    - Trade-off: Loses data on failure, but better than corrupting registry
#
# 4. Why not cache labels for common statements?
#    - Statements are already normalized, so duplicates are rare
#    - Cache would add complexity without much benefit
#    - Temperature=0.0 already ensures deterministic output
#    - Trade-off: Slight inefficiency for exact duplicates, acceptable for V1
#
# 5. Why use LLM for this instead of rule-based label generation?
#    - Rules can't handle nuances (e.g., "app crashes on login" vs "login crashes app")
#    - LLM better at preserving important context while shortening
#    - Consistent with overall agentic design
#    - Trade-off: Higher cost and latency, but better quality
