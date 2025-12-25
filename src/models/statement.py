"""
Statement data model.

Represents a normalized, intent-focused statement extracted from a review.
"""

from dataclasses import dataclass


@dataclass
class Statement:
    """
    A clean, intent-focused statement extracted from a raw review.
    Output of Semantic Normalization Agent.
    """
    normalized_statement: str  # Clean statement (3-10 words ideal)
    type: str  # "issue", "request", or "feedback"
    confidence: str  # "high", "medium", or "low"
    
    def __post_init__(self):
        # Validate type
        if self.type not in ("issue", "request", "feedback"):
            raise ValueError(
                f"Invalid type: {self.type}. Must be 'issue', 'request', or 'feedback'"
            )
        
        # Validate confidence
        if self.confidence not in ("high", "medium", "low"):
            raise ValueError(
                f"Invalid confidence: {self.confidence}. Must be 'high', 'medium', or 'low'"
            )


# Design Rationale and Trade-offs:
#
# 1. Why separate type and confidence?
#    - Type: Semantic classification (issue/request/feedback)
#    - Confidence: LLM certainty in the extraction
#    - Allows filtering by confidence while preserving type info
#    - Trade-off: Two fields instead of one, but clearer semantics
#
# 2. Why confidence as string instead of float?
#    - Simpler for LLM to output ("high" vs 0.85)
#    - Easier for humans to interpret in logs
#    - Three levels sufficient for filtering decisions
#    - Trade-off: Less granular than 0-1 float, but adequate
#
# 3. Why no source_review_id field?
#    - Statements are ephemeral (not persisted in V1)
#    - Traceability handled at processing level, not data level
#    - Trade-off: Harder to debug individual extractions, acceptable for V1
