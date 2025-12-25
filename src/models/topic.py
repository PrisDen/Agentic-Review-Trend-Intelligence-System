"""
Topic data model.

Represents both topic candidates (from candidate generation)
and canonical topics (in the registry).
"""

from dataclasses import dataclass, field
from typing import List, Optional
from datetime import datetime
import uuid


@dataclass
class TopicCandidate:
    """
    A potential topic extracted from a normalized statement.
    Used during consolidation to decide MERGE or CREATE.
    """
    label: str  # Human-readable topic label (e.g., "Delivery partner rude")
    type: str  # "issue", "request", or "feedback"
    embedding: Optional[List[float]] = None  # Semantic vector for similarity search
    source_review_id: Optional[str] = None  # Traceability to original review
    
    def __post_init__(self):
        # Validate type
        if self.type not in ("issue", "request", "feedback"):
            raise ValueError(f"Invalid type: {self.type}. Must be 'issue', 'request', or 'feedback'")


@dataclass
class CanonicalTopic:
    """
    A canonical topic in the registry.
    Represents the single source of truth for a semantic concept.
    """
    topic_id: str
    canonical_label: str
    type: str  # "issue", "request", or "feedback"
    aliases: List[str] = field(default_factory=list)
    embedding: List[float] = field(default_factory=list)
    created_on: str = ""  # YYYY-MM-DD format
    last_seen: str = ""  # YYYY-MM-DD format
    total_mentions: int = 0
    metadata: dict = field(default_factory=dict)
    
    @classmethod
    def from_dict(cls, data: dict) -> "CanonicalTopic":
        """Create CanonicalTopic from JSON dict."""
        return cls(
            topic_id=data["topic_id"],
            canonical_label=data["canonical_label"],
            type=data["type"],
            aliases=data.get("aliases", []),
            embedding=data.get("embedding", []),
            created_on=data.get("created_on", ""),
            last_seen=data.get("last_seen", ""),
            total_mentions=data.get("total_mentions", 0),
            metadata=data.get("metadata", {})
        )
    
    def to_dict(self) -> dict:
        """Convert to JSON-serializable dict."""
        return {
            "topic_id": self.topic_id,
            "canonical_label": self.canonical_label,
            "type": self.type,
            "aliases": self.aliases,
            "embedding": self.embedding,
            "created_on": self.created_on,
            "last_seen": self.last_seen,
            "total_mentions": self.total_mentions,
            "metadata": self.metadata
        }


# Design Rationale and Trade-offs:
#
# 1. Why dataclasses instead of plain dicts?
#    - Type safety: Prevents field name typos
#    - IDE autocomplete: Improves developer experience
#    - Minimal boilerplate: dataclass decorator handles __init__, __repr__
#    - Trade-off: Slightly more verbose than dicts, but much safer
#
# 2. Why separate TopicCandidate and CanonicalTopic?
#    - Candidates are ephemeral (exist during processing only)
#    - Canonical topics are persistent (stored in registry)
#    - Different field requirements (candidates don't have aliases/mentions)
#    - Clear semantic distinction in code
#
# 3. Why store embedding as List[float] instead of numpy array?
#    - JSON serialization: Lists serialize directly, numpy needs conversion
#    - Simplicity: Avoid numpy dependency for data models
#    - Trade-off: Slightly slower for numerical operations, but acceptable
#
# 4. Why string dates (YYYY-MM-DD) instead of datetime objects?
#    - JSON serialization: Strings serialize directly
#    - Registry persistence: No timezone complexity
#    - Trade-off: Need to parse for date arithmetic, but parsing is trivial
