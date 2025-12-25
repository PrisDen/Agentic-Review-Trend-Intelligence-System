"""
Review data model.

Represents raw review from Google Play Store ingestion.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class Review:
    """
    Raw review from Google Play Store.
    Minimal fields needed for normalization.
    """
    review_id: str  # Unique identifier for the review
    text: str  # Raw review text
    rating: int  # 1-5 star rating
    date: str  # YYYY-MM-DD format
    author: Optional[str] = None  # Optional author identifier
    
    def __post_init__(self):
        # Validate rating
        if not (1 <= self.rating <= 5):
            raise ValueError(f"Invalid rating: {self.rating}. Must be 1-5")


# Design Rationale and Trade-offs:
#
# 1. Why minimal fields?
#    - Only store what's needed for normalization (text, rating as context)
#    - Keeps memory footprint small during batch processing
#    - Trade-off: Can't do author-specific analysis, but not required per spec
#
# 2. Why string date instead of datetime?
#    - Consistency with topic registry date format
#    - Simple JSON serialization
#    - Trade-off: Need to parse for date arithmetic, but rarely needed
#
# 3. Why rating as int instead of float?
#    - Google Play Store uses integer ratings (1-5)
#    - Simpler validation logic
#    - Trade-off: None, matches actual data format
