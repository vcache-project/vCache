import unittest
import pytest
import os

from vectorq.config import VectorQConfig
from vectorq.vectorq_core.cache.embedding_engine.embedding_engine import (
    EmbeddingEngine,
    EmbeddingEngineType,
)

# Define test parameters once at the module level
EMBEDDING_ENGINE_PARAMS = [
    (EmbeddingEngineType.LANGCHAIN, "sentence-transformers/all-MiniLM-L6-v2"),
    pytest.param(
        EmbeddingEngineType.OPENAI,
        "text-embedding-ada-002",
        marks=pytest.mark.skipif(
            not os.environ.get("OPENAI_API_KEY"),
            reason="OPENAI_API_KEY environment variable not set",
        ),
    ),
]


class TestEmbeddingEngineStrategy:
    """Test all embedding engine strategies using parameterization."""

    @pytest.mark.parametrize(
        "embedding_engine_type, embedding_engine_model_name", EMBEDDING_ENGINE_PARAMS
    )
    def test_get_embedding(self, embedding_engine_type, embedding_engine_model_name):
        """Test getting embeddings from different embedding engines."""
        config = VectorQConfig(
            embedding_engine_type=embedding_engine_type,
            embedding_engine_model_name=embedding_engine_model_name,
        )

        engine = EmbeddingEngine(vectorq_config=config)
        text = "This is a test embedding."
        embedding = engine.get_embedding(text)

        # Verify the embedding has the expected format
        assert isinstance(embedding, list)
        assert len(embedding) > 0
        assert isinstance(embedding[0], float)

        # Test with a different text
        different_text = "This is a completely different text for embedding."
        different_embedding = engine.get_embedding(different_text)

        # Verify the different text produces a different embedding
        assert isinstance(different_embedding, list)
        assert len(different_embedding) > 0
        assert isinstance(different_embedding[0], float)

        # In a vector space, different texts should generally have different embeddings
        # This is a simple check, not a comprehensive one
        # We're just checking if at least one value is different
        has_different_values = False
        for i in range(min(len(embedding), len(different_embedding))):
            if abs(embedding[i] - different_embedding[i]) > 1e-6:
                has_different_values = True
                break
        assert (
            has_different_values
        ), "Different texts should produce different embeddings"

    @pytest.mark.parametrize(
        "embedding_engine_type, embedding_engine_model_name", EMBEDDING_ENGINE_PARAMS
    )
    def test_embedding_dimensions_consistent(
        self, embedding_engine_type, embedding_engine_model_name
    ):
        """Test that embeddings from the same engine have consistent dimensions."""
        config = VectorQConfig(
            embedding_engine_type=embedding_engine_type,
            embedding_engine_model_name=embedding_engine_model_name,
        )

        engine = EmbeddingEngine(vectorq_config=config)
        text1 = "First text for embedding."
        text2 = "Second text for embedding."

        embedding1 = engine.get_embedding(text1)
        embedding2 = engine.get_embedding(text2)

        # Verify both embeddings have the same dimension
        assert len(embedding1) == len(
            embedding2
        ), "Embeddings should have consistent dimensions"

    @pytest.mark.parametrize(
        "embedding_engine_type, embedding_engine_model_name", EMBEDDING_ENGINE_PARAMS
    )
    def test_similar_texts_have_similar_embeddings(
        self, embedding_engine_type, embedding_engine_model_name
    ):
        """Test that similar texts have more similar embeddings than dissimilar texts."""
        config = VectorQConfig(
            embedding_engine_type=embedding_engine_type,
            embedding_engine_model_name=embedding_engine_model_name,
        )

        engine = EmbeddingEngine(vectorq_config=config)

        text1 = "The cat sat on the mat."
        similar_text = "A cat was sitting on a mat."
        different_text = "Quantum physics explores the fundamental nature of reality."

        embedding1 = engine.get_embedding(text1)
        similar_embedding = engine.get_embedding(similar_text)
        different_embedding = engine.get_embedding(different_text)

        # Calculate cosine similarity (dot product of normalized vectors)
        def cosine_similarity(vec1, vec2):
            # Normalize vectors
            mag1 = sum(x * x for x in vec1) ** 0.5
            mag2 = sum(x * x for x in vec2) ** 0.5

            # Dot product of normalized vectors
            dot_product = sum(
                a * b
                for a, b in zip((x / mag1 for x in vec1), (y / mag2 for y in vec2))
            )

            return dot_product

        sim_to_similar = cosine_similarity(embedding1, similar_embedding)
        sim_to_different = cosine_similarity(embedding1, different_embedding)

        # Similar texts should have higher cosine similarity than different texts
        assert (
            sim_to_similar > sim_to_different
        ), "Similar texts should have more similar embeddings"


if __name__ == "__main__":
    unittest.main()
