"""
Basic unit tests for Topic Registry.
Verifies core functionality before moving to next component.
"""

import pytest
import json
import os
import tempfile
from src.models.topic import TopicCandidate, CanonicalTopic
from src.registry.topic_registry import TopicRegistry


def test_topic_candidate_validation():
    """Test TopicCandidate type validation."""
    # Valid types
    candidate = TopicCandidate(label="Test issue", type="issue")
    assert candidate.type == "issue"
    
    # Invalid type should raise ValueError
    with pytest.raises(ValueError):
        TopicCandidate(label="Test", type="invalid_type")


def test_canonical_topic_serialization():
    """Test CanonicalTopic to/from dict conversion."""
    topic = CanonicalTopic(
        topic_id="test-123",
        canonical_label="Test Topic",
        type="issue",
        aliases=["alias1", "alias2"],
        embedding=[0.1, 0.2, 0.3],
        created_on="2024-06-01",
        last_seen="2024-06-10",
        total_mentions=5
    )
    
    # Convert to dict and back
    topic_dict = topic.to_dict()
    restored = CanonicalTopic.from_dict(topic_dict)
    
    assert restored.topic_id == topic.topic_id
    assert restored.canonical_label == topic.canonical_label
    assert restored.aliases == topic.aliases
    assert restored.total_mentions == 5


def test_registry_initialization():
    """Test creating new registry."""
    with tempfile.TemporaryDirectory() as tmpdir:
        registry_path = os.path.join(tmpdir, "test_registry.json")
        registry = TopicRegistry(registry_path)
        
        assert len(registry.topics) == 0
        assert registry.version == "1.0.0"


def test_add_topic():
    """Test adding new topics to registry."""
    with tempfile.TemporaryDirectory() as tmpdir:
        registry_path = os.path.join(tmpdir, "test_registry.json")
        registry = TopicRegistry(registry_path)
        
        # Add first topic
        topic_id = registry.add_topic(
            canonical_label="Delivery partner rude",
            topic_type="issue",
            embedding=[0.1] * 1536,  # Valid 1536-dim embedding
            created_on="2024-06-01"
        )
        
        assert topic_id in registry.topics
        assert registry.topics[topic_id].canonical_label == "Delivery partner rude"
        assert registry.topics[topic_id].total_mentions == 1


def test_duplicate_label_prevention():
    """Test that adding duplicate labels raises error."""
    with tempfile.TemporaryDirectory() as tmpdir:
        registry_path = os.path.join(tmpdir, "test_registry.json")
        registry = TopicRegistry(registry_path)
        
        # Add first topic
        registry.add_topic(
            canonical_label="Delivery partner rude",
            topic_type="issue",
            embedding=[0.1] * 1536,
            created_on="2024-06-01"
        )
        
        # Try to add duplicate - should raise ValueError
        with pytest.raises(ValueError, match="already exists"):
            registry.add_topic(
                canonical_label="Delivery partner rude",
                topic_type="issue",
                embedding=[0.2] * 1536,
                created_on="2024-06-02"
            )


def test_add_alias():
    """Test adding aliases to topics."""
    with tempfile.TemporaryDirectory() as tmpdir:
        registry_path = os.path.join(tmpdir, "test_registry.json")
        registry = TopicRegistry(registry_path)
        
        # Add topic
        topic_id = registry.add_topic(
            canonical_label="Delivery partner rude",
            topic_type="issue",
            embedding=[0.1] * 1536,
            created_on="2024-06-01"
        )
        
        # Add alias
        registry.add_alias(topic_id, "delivery guy rude")
        
        assert "delivery guy rude" in registry.topics[topic_id].aliases


def test_find_by_label():
    """Test finding topics by exact label match."""
    with tempfile.TemporaryDirectory() as tmpdir:
        registry_path = os.path.join(tmpdir, "test_registry.json")
        registry = TopicRegistry(registry_path)
        
        # Add topic
        topic_id = registry.add_topic(
            canonical_label="Delivery partner rude",
            topic_type="issue",
            embedding=[0.1] * 1536,
            created_on="2024-06-01"
        )
        registry.add_alias(topic_id, "delivery guy rude")
        
        # Find by canonical label (case-insensitive)
        found_id = registry.find_by_label("delivery partner rude")
        assert found_id == topic_id
        
        # Find by alias
        found_id = registry.find_by_label("delivery guy rude")
        assert found_id == topic_id
        
        # Not found
        found_id = registry.find_by_label("nonexistent topic")
        assert found_id is None


def test_find_similar():
    """Test similarity search."""
    with tempfile.TemporaryDirectory() as tmpdir:
        registry_path = os.path.join(tmpdir, "test_registry.json")
        registry = TopicRegistry(registry_path)
        
        # Add topics with different embeddings
        topic1_id = registry.add_topic(
            canonical_label="Topic A",
            topic_type="issue",
            embedding=[1.0] + [0.0] * 1535,  # Vector pointing in direction 0
            created_on="2024-06-01"
        )
        
        topic2_id = registry.add_topic(
            canonical_label="Topic B",
            topic_type="issue",
            embedding=[0.9] + [0.1] * 1535,  # Similar to Topic A
            created_on="2024-06-02"
        )
        
        topic3_id = registry.add_topic(
            canonical_label="Topic C",
            topic_type="issue",
            embedding=[0.0] * 1535 + [1.0],  # Very different
            created_on="2024-06-03"
        )
        
        # Search for similar topics
        query_embedding = [0.95] + [0.05] * 1535  # Very similar to Topic A
        similar = registry.find_similar(query_embedding, top_k=2, min_similarity=0.70)
        
        # Should return Topic A and B, sorted by similarity
        assert len(similar) == 2
        assert similar[0][0] in [topic1_id, topic2_id]  # One of the similar ones first


def test_save_and_load():
    """Test registry persistence."""
    with tempfile.TemporaryDirectory() as tmpdir:
        registry_path = os.path.join(tmpdir, "test_registry.json")
        
        # Create registry and add topics
        registry1 = TopicRegistry(registry_path)
        topic_id = registry1.add_topic(
            canonical_label="Test Topic",
            topic_type="issue",
            embedding=[0.1] * 1536,
            created_on="2024-06-01"
        )
        registry1.add_alias(topic_id, "test alias")
        registry1.save()
        
        # Load in new registry instance
        registry2 = TopicRegistry(registry_path)
        
        assert len(registry2.topics) == 1
        assert topic_id in registry2.topics
        assert "test alias" in registry2.topics[topic_id].aliases


def test_update_last_seen():
    """Test updating last_seen and total_mentions."""
    with tempfile.TemporaryDirectory() as tmpdir:
        registry_path = os.path.join(tmpdir, "test_registry.json")
        registry = TopicRegistry(registry_path)
        
        topic_id = registry.add_topic(
            canonical_label="Test Topic",
            topic_type="issue",
            embedding=[0.1] * 1536,
            created_on="2024-06-01"
        )
        
        # Initial state
        assert registry.topics[topic_id].total_mentions == 1
        assert registry.topics[topic_id].last_seen == "2024-06-01"
        
        # Update
        registry.update_last_seen(topic_id, "2024-06-15")
        
        assert registry.topics[topic_id].total_mentions == 2
        assert registry.topics[topic_id].last_seen == "2024-06-15"


# Run tests if executed directly
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
