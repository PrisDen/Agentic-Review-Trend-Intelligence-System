"""
Unit tests for Semantic Normalization Agent.

Note: These tests use mocked LLM responses to avoid API costs.
Integration tests with real LLM are in a separate file.
"""

import pytest
import json
from unittest.mock import Mock, patch, MagicMock
from src.models.review import Review
from src.models.statement import Statement
from src.agents.normalization import SemanticNormalizationAgent


@pytest.fixture
def mock_agent():
    """Create agent with mocked Gemini API."""
    with patch('src.agents.normalization.genai'):
        agent = SemanticNormalizationAgent(
            api_key="test-key",
            model_name="gemini-1.5-flash",
            temperature=0.0,
            min_confidence="medium"
        )
        return agent


def test_empty_review_handling(mock_agent):
    """Test that empty reviews return empty statements."""
    review = Review(
        review_id="test-1",
        text="",
        rating=3,
        date="2024-06-01"
    )
    
    statements = mock_agent.normalize(review)
    assert statements == []


def test_parse_llm_response(mock_agent):
    """Test parsing valid LLM JSON response."""
    llm_response = json.dumps({
        "statements": [
            {
                "normalized_statement": "Delivery partner rude",
                "type": "issue",
                "confidence": "high"
            },
            {
                "normalized_statement": "Food arrived cold",
                "type": "issue",
                "confidence": "high"
            }
        ]
    })
    
    statements = mock_agent._parse_llm_response(llm_response, "test-1")
    
    assert len(statements) == 2
    assert statements[0].normalized_statement == "Delivery partner rude"
    assert statements[0].type == "issue"
    assert statements[0].confidence == "high"


def test_parse_llm_response_missing_statements_field(mock_agent):
    """Test handling of malformed LLM response."""
    llm_response = json.dumps({"error": "something went wrong"})
    
    statements = mock_agent._parse_llm_response(llm_response, "test-1")
    assert statements == []


def test_parse_llm_response_invalid_statement(mock_agent):
    """Test handling of invalid statement in response."""
    llm_response = json.dumps({
        "statements": [
            {
                "normalized_statement": "Valid statement",
                "type": "issue",
                "confidence": "high"
            },
            {
                "normalized_statement": "Invalid statement",
                "type": "invalid_type",  # Invalid type
                "confidence": "high"
            }
        ]
    })
    
    statements = mock_agent._parse_llm_response(llm_response, "test-1")
    
    # Should skip invalid statement
    assert len(statements) == 1
    assert statements[0].normalized_statement == "Valid statement"


def test_confidence_filtering(mock_agent):
    """Test filtering statements by confidence level."""
    statements = [
        Statement("Statement 1", "issue", "high"),
        Statement("Statement 2", "issue", "medium"),
        Statement("Statement 3", "issue", "low")
    ]
    
    # Agent configured with min_confidence="medium"
    filtered = mock_agent._filter_by_confidence(statements)
    
    assert len(filtered) == 2  # Should keep high and medium
    assert filtered[0].confidence in ["high", "medium"]
    assert filtered[1].confidence in ["high", "medium"]


def test_normalize_with_mocked_llm():
    """Test full normalize() flow with mocked LLM."""
    # Create mock LLM response
    mock_response = MagicMock()
    mock_response.text = json.dumps({
        "statements": [
            {
                "normalized_statement": "App crashes on login",
                "type": "issue",
                "confidence": "high"
            }
        ]
    })
    
    # Patch Gemini API
    with patch('src.agents.normalization.genai') as mock_genai:
        mock_model = MagicMock()
        mock_model.generate_content.return_value = mock_response
        mock_genai.GenerativeModel.return_value = mock_model
        
        # Create agent
        agent = SemanticNormalizationAgent(
            api_key="test-key",
            temperature=0.0
        )
        
        # Test normalization
        review = Review(
            review_id="test-1",
            text="The app keeps crashing whenever I try to login!",
            rating=1,
            date="2024-06-01"
        )
        
        statements = agent.normalize(review)
        
        assert len(statements) == 1
        assert statements[0].normalized_statement == "App crashes on login"
        assert statements[0].type == "issue"


def test_json_decode_error_retry():
    """Test retry logic on JSON parsing errors."""
    # Mock LLM to return invalid JSON twice, then valid JSON
    mock_responses = [
        "invalid json{{{",  # First attempt
        "still invalid",    # Second attempt
        json.dumps({"statements": []})  # Third attempt succeeds
    ]
    
    with patch('src.agents.normalization.genai') as mock_genai:
        mock_model = MagicMock()
        mock_model.generate_content.side_effect = [
            MagicMock(text=resp) for resp in mock_responses
        ]
        mock_genai.GenerativeModel.return_value = mock_model
        
        agent = SemanticNormalizationAgent(
            api_key="test-key",
            max_retries=3
        )
        
        review = Review(
            review_id="test-1",
            text="Test review",
            rating=3,
            date="2024-06-01"
        )
        
        statements = agent.normalize(review)
        
        # Should succeed on third attempt
        assert statements == []
        assert mock_model.generate_content.call_count == 3


# Run tests if executed directly
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
