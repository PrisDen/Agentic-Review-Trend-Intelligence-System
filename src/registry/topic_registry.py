"""
Topic Registry - Single source of truth for canonical topics.

Manages topic creation, alias tracking, similarity search, and persistence.
"""

import json
import os
import shutil
import logging
from typing import List, Optional, Tuple, Dict
from datetime import datetime
import uuid

from src.models.topic import CanonicalTopic

logger = logging.getLogger(__name__)


class TopicRegistry:
    """
    Single source of truth for all canonical topics.
    
    Prevents topic explosion through:
    - Semantic deduplication (embedding similarity)
    - Alias accumulation (related phrasings map to same topic)
    - Exact label matching (fast path for known topics)
    """
    
    def __init__(self, registry_path: str):
        """
        Initialize registry from disk or create new empty registry.
        
        Args:
            registry_path: Path to topic_registry.json file
        """
        self.registry_path = registry_path
        self.topics: Dict[str, CanonicalTopic] = {}  # topic_id -> CanonicalTopic
        self.version = "1.0.0"
        self.embedding_model = "text-embedding-3-small"  # Default, can be overridden
        self.embedding_dimensions = 1536
        self.last_updated = datetime.utcnow().isoformat() + "Z"
        
        # Load existing registry if it exists
        if os.path.exists(registry_path):
            self._load()
        else:
            logger.info(f"No existing registry found at {registry_path}, initializing empty registry")
    
    def _load(self) -> None:
        """Load registry from disk."""
        try:
            with open(self.registry_path, 'r') as f:
                data = json.load(f)
            
            # Handle legacy format (list of topics) vs new format (dict with metadata)
            if isinstance(data, list):
                # Legacy format: just a list of topics
                topics_list = data
            else:
                # New format: dict with version, metadata, and topics
                self.version = data.get("version", "1.0.0")
                self.embedding_model = data.get("embedding_model", "text-embedding-3-small")
                self.embedding_dimensions = data.get("embedding_dimensions", 1536)
                self.last_updated = data.get("last_updated", datetime.utcnow().isoformat() + "Z")
                topics_list = data.get("topics", [])
            
            # Rebuild topics dict
            self.topics = {}
            for topic_data in topics_list:
                topic = CanonicalTopic.from_dict(topic_data)
                self.topics[topic.topic_id] = topic
            
            logger.info(f"Loaded {len(self.topics)} topics from registry")
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse registry JSON: {e}")
            self._try_restore_from_backup()
        except Exception as e:
            logger.error(f"Failed to load registry: {e}")
            self._try_restore_from_backup()
    
    def _try_restore_from_backup(self) -> None:
        """Attempt to restore from backup file if main registry is corrupted."""
        backup_path = f"{self.registry_path}.backup"
        if os.path.exists(backup_path):
            logger.warning(f"Attempting to restore from backup: {backup_path}")
            try:
                shutil.copy(backup_path, self.registry_path)
                self._load()
                logger.info("Successfully restored from backup")
            except Exception as e:
                logger.error(f"Backup restoration failed: {e}. Starting with empty registry.")
                self.topics = {}
        else:
            logger.warning("No backup file found. Starting with empty registry.")
            self.topics = {}
    
    def add_topic(
        self,
        canonical_label: str,
        topic_type: str,
        embedding: List[float],
        created_on: str,
        description: str = ""
    ) -> str:
        """
        Create a new canonical topic.
        
        Args:
            canonical_label: Human-readable PM-facing label
            topic_type: "issue", "request", or "feedback"
            embedding: Semantic vector for similarity search
            created_on: Creation date in YYYY-MM-DD format
            description: Optional long-form explanation
        
        Returns:
            topic_id (UUID) of newly created topic
        
        Raises:
            ValueError: If canonical_label already exists or is invalid
        """
        # Check for duplicate label
        existing_id = self.find_by_label(canonical_label)
        if existing_id:
            raise ValueError(
                f"Topic '{canonical_label}' already exists with ID: {existing_id}"
            )
        
        # Validate label
        if not self._validate_label(canonical_label):
            raise ValueError(f"Invalid label: '{canonical_label}'")
        
        # Validate embedding
        if not self._validate_embedding(embedding):
            raise ValueError(f"Invalid embedding (expected {self.embedding_dimensions} dimensions)")
        
        # Generate new topic ID
        topic_id = str(uuid.uuid4())
        
        # Create topic
        topic = CanonicalTopic(
            topic_id=topic_id,
            canonical_label=canonical_label,
            type=topic_type,
            aliases=[],
            embedding=embedding,
            created_on=created_on,
            last_seen=created_on,
            total_mentions=1,  # First mention is creation
            metadata={"description": description} if description else {}
        )
        
        self.topics[topic_id] = topic
        logger.info(f"Created new topic: {topic_id} - '{canonical_label}'")
        
        return topic_id
    
    def add_alias(self, topic_id: str, alias: str) -> None:
        """
        Add an alternative phrasing to an existing topic.
        Idempotent: Adding same alias twice is a no-op.
        
        Args:
            topic_id: Target topic UUID
            alias: Alternative label to add
        
        Raises:
            ValueError: If topic_id doesn't exist or alias conflicts with another topic
        """
        if topic_id not in self.topics:
            raise ValueError(f"Topic not found: {topic_id}")
        
        # Check for conflicts: alias must not exist in any other topic
        alias_lower = alias.lower()
        for tid, topic in self.topics.items():
            # Check canonical label
            if topic.canonical_label.lower() == alias_lower and tid != topic_id:
                raise ValueError(
                    f"Alias '{alias}' conflicts with canonical label of topic {tid}"
                )
            # Check existing aliases
            if alias_lower in [a.lower() for a in topic.aliases] and tid != topic_id:
                raise ValueError(
                    f"Alias '{alias}' already exists in topic {tid}"
                )
        
        # Add alias if not already present (case-insensitive check)
        topic = self.topics[topic_id]
        if alias_lower not in [a.lower() for a in topic.aliases]:
            topic.aliases.append(alias)
            logger.info(f"Added alias '{alias}' to topic {topic_id}")
    
    def get_topic(self, topic_id: str) -> Optional[CanonicalTopic]:
        """Retrieve topic by ID. Returns None if not found."""
        return self.topics.get(topic_id)
    
    def find_by_label(self, label: str) -> Optional[str]:
        """
        Find topic_id by exact canonical_label or alias match (case-insensitive).
        
        Args:
            label: Label to search for
        
        Returns:
            topic_id if match found, else None
        """
        label_lower = label.lower()
        
        for topic_id, topic in self.topics.items():
            # Check canonical label
            if topic.canonical_label.lower() == label_lower:
                return topic_id
            
            # Check aliases
            if label_lower in [a.lower() for a in topic.aliases]:
                return topic_id
        
        return None
    
    def find_similar(
        self,
        embedding: List[float],
        top_k: int = 5,
        min_similarity: float = 0.70
    ) -> List[Tuple[str, float]]:
        """
        Find top-k most similar topics by cosine similarity.
        
        Args:
            embedding: Query embedding vector
            top_k: Number of similar topics to return
            min_similarity: Minimum similarity threshold (0-1)
        
        Returns:
            List of (topic_id, similarity_score) tuples, sorted descending by similarity
        """
        similarities = []
        
        for topic_id, topic in self.topics.items():
            if not topic.embedding:
                continue
            
            # Compute cosine similarity
            similarity = self._cosine_similarity(embedding, topic.embedding)
            
            if similarity >= min_similarity:
                similarities.append((topic_id, similarity))
        
        # Sort by similarity descending and return top-k
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]
    
    def update_last_seen(self, topic_id: str, date: str) -> None:
        """
        Update last_seen date and increment total_mentions.
        
        Args:
            topic_id: Target topic UUID
            date: Date in YYYY-MM-DD format
        """
        if topic_id not in self.topics:
            raise ValueError(f"Topic not found: {topic_id}")
        
        topic = self.topics[topic_id]
        topic.last_seen = date
        topic.total_mentions += 1
    
    def get_all_topics(self) -> List[CanonicalTopic]:
        """Return all topics, sorted by creation date."""
        topics_list = list(self.topics.values())
        topics_list.sort(key=lambda t: t.created_on)
        return topics_list
    
    def save(self) -> None:
        """
        Persist registry to disk with atomic write pattern.
        Creates backup before write.
        """
        # Update last_updated timestamp
        self.last_updated = datetime.utcnow().isoformat() + "Z"
        
        # Create backup if registry file exists
        if os.path.exists(self.registry_path):
            backup_path = f"{self.registry_path}.backup"
            shutil.copy(self.registry_path, backup_path)
            logger.debug(f"Created backup: {backup_path}")
        
        # Prepare data
        data = {
            "version": self.version,
            "embedding_model": self.embedding_model,
            "embedding_dimensions": self.embedding_dimensions,
            "last_updated": self.last_updated,
            "topics": [topic.to_dict() for topic in self.topics.values()]
        }
        
        # Atomic write: write to temp file, then rename
        temp_path = f"{self.registry_path}.tmp"
        try:
            with open(temp_path, 'w') as f:
                json.dump(data, f, indent=2)
            
            # Atomic rename
            os.rename(temp_path, self.registry_path)
            logger.info(f"Registry saved: {len(self.topics)} topics")
            
        except Exception as e:
            logger.error(f"Failed to save registry: {e}")
            # Clean up temp file if it exists
            if os.path.exists(temp_path):
                os.remove(temp_path)
            raise
    
    def _validate_label(self, label: str) -> bool:
        """Ensure label is PM-consumable."""
        # Must be 3-50 characters
        if len(label) < 3 or len(label) > 50:
            return False
        
        # Must not be all uppercase (shouting)
        if label.isupper():
            return False
        
        return True
    
    def _validate_embedding(self, embedding: List[float]) -> bool:
        """Ensure embedding is non-zero and correct dimension."""
        # Check dimension
        if len(embedding) != self.embedding_dimensions:
            return False
        
        # Check not all zeros (API failure indicator)
        if sum(abs(x) for x in embedding) < 0.01:
            return False
        
        return True
    
    @staticmethod
    def _cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
        """
        Compute cosine similarity between two vectors.
        
        Returns:
            Similarity score in range [0, 1] (assuming positive embeddings)
        """
        # Dot product
        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        
        # Magnitudes
        mag1 = sum(a * a for a in vec1) ** 0.5
        mag2 = sum(b * b for b in vec2) ** 0.5
        
        # Avoid division by zero
        if mag1 == 0 or mag2 == 0:
            return 0.0
        
        return dot_product / (mag1 * mag2)


# Design Rationale and Trade-offs:
#
# 1. Why store topics as dict (topic_id -> CanonicalTopic) instead of list?
#    - O(1) lookup by ID instead of O(n) linear search
#    - Simplifies get_topic(), update_last_seen()
#    - Trade-off: Slightly more memory, but negligible for <100k topics
#
# 2. Why atomic write pattern (temp file + rename)?
#    - Prevents corruption if process crashes during write
#    - OS guarantees rename is atomic on most filesystems
#    - Trade-off: Requires 2x disk space temporarily, but registry is small
#
# 3. Why backup-and-restore for corrupted files?
#    - Registry is critical - cannot continue without it
#    - Backup provides one-level undo for corruption
#    - Trade-off: Only one backup level (could do rotating backups in V2)
#
# 4. Why case-insensitive label/alias matching?
#    - Users may write "delivery guy rude" vs "Delivery Guy Rude"
#    - Prevents duplicate topics differing only in case
#    - Trade-off: Canonical label preserves original case for PM readability
#
# 5. Why cosine similarity instead of Euclidean distance?
#    - Cosine is direction-based, robust to magnitude differences
#    - Standard for text embeddings (OpenAI, Sentence-BERT)
#    - Trade-off: Slightly slower than dot product, but more accurate
#
# 6. Why no numpy dependency?
#    - Keeps model layer lightweight
#    - Pure Python cosine similarity is fast enough for <10k topics
#    - Trade-off: 10x slower than numpy, but acceptable for our scale
