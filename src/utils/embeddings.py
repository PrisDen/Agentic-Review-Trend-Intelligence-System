"""
Embeddings utility.

Generate embeddings and compute semantic similarity.
"""

import logging
from typing import List
import google.generativeai as genai

logger = logging.getLogger(__name__)


class EmbeddingGenerator:
    """
    Generates text embeddings for semantic similarity search.
    
    Uses Google's text-embedding models for consistency with Gemini ecosystem.
    """
    
    def __init__(
        self,
        api_key: str,
        model_name: str = "models/text-embedding-004",
        embedding_dimensions: int = 768
    ):
        """
        Initialize embedding generator.
        
        Args:
            api_key: Google API key
            model_name: Embedding model to use
            embedding_dimensions: Output dimensions (768 for text-embedding-004)
        """
        self.model_name = model_name
        self.embedding_dimensions = embedding_dimensions
        
        # Configure Gemini
        genai.configure(api_key=api_key)
        
        logger.info(f"Initialized EmbeddingGenerator with model={model_name}, dims={embedding_dimensions}")
    
    def generate(self, text: str) -> List[float]:
        """
        Generate embedding for a single text.
        
        Args:
            text: Input text to embed
        
        Returns:
            Embedding vector as list of floats
        
        Raises:
            ValueError: If embedding generation fails or returns invalid dimensions
        """
        if not text or len(text.strip()) == 0:
            raise ValueError("Cannot generate embedding for empty text")
        
        try:
            result = genai.embed_content(
                model=self.model_name,
                content=text,
                task_type="semantic_similarity"
            )
            
            embedding = result['embedding']
            
            # Validate dimensions
            if len(embedding) != self.embedding_dimensions:
                raise ValueError(
                    f"Expected {self.embedding_dimensions} dimensions, got {len(embedding)}"
                )
            
            # Validate non-zero (API failure check)
            if sum(abs(x) for x in embedding) < 0.01:
                raise ValueError("Embedding is all zeros (possible API failure)")
            
            return embedding
            
        except Exception as e:
            logger.error(f"Failed to generate embedding: {e}")
            raise
    
    def generate_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for multiple texts.
        
        Args:
            texts: List of input texts
        
        Returns:
            List of embedding vectors
        
        Note: Currently processes sequentially. Can be optimized with batching in V2.
        """
        embeddings = []
        for text in texts:
            try:
                embedding = self.generate(text)
                embeddings.append(embedding)
            except ValueError as e:
                logger.warning(f"Skipping text due to embedding error: {e}")
                # Append None for failed embeddings to maintain index alignment
                embeddings.append(None)
        
        return embeddings


# Design Rationale and Trade-offs:
#
# 1. Why Google's text-embedding-004 instead of OpenAI's?
#    - Consistency with Gemini ecosystem (same API key)
#    - Lower cost: $0.00001 per 1k chars vs OpenAI's $0.00002
#    - 768 dimensions sufficient for our use case (vs 1536 for OpenAI)
#    - Trade-off: Slightly lower quality than text-embedding-3-large, but adequate
#
# 2. Why task_type="semantic_similarity"?
#    - Optimizes embeddings for cosine similarity comparisons
#    - Other options: "retrieval_query", "retrieval_document", "classification"
#    - semantic_similarity is best for topic deduplication
#    - Trade-off: Less optimal for other use cases, but we only do similarity
#
# 3. Why validate non-zero embeddings?
#    - API can silently fail and return zero vectors
#    - Zero vectors have undefined similarity with any other vector
#    - Better to fail fast than corrupt registry with invalid embeddings
#    - Trade-off: Stricter validation might reject rare edge cases
#
# 4. Why not batch embeddings in V1?
#    - Google's API supports batching, but adds complexity
#    - Sequential processing is simpler and easier to debug
#    - For <1000 embeddings/day, latency difference is ~30 seconds
#    - Trade-off: Slower, but acceptable for daily batch processing
#
# 5. Why return None for failed embeddings in batch instead of raising?
#    - Partial success better than complete failure
#    - Caller can filter out None values
#    - Maintains index alignment with input texts
#    - Trade-off: Silent failures if caller doesn't check for None
