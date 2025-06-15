import unittest

import pytest

from vcache.vcache_core.cache.embedding_store.embedding_metadata_storage.embedding_metadata_obj import (
    EmbeddingMetadataObj,
)
from vcache.vcache_core.cache.embedding_store.vector_db import (
    HNSWLibVectorDB,
    SimilarityMetricType,
)

VECTOR_DB_PARAMS = [
    (HNSWLibVectorDB, SimilarityMetricType.COSINE),
    (HNSWLibVectorDB, SimilarityMetricType.EUCLIDEAN),
]


class TestVectorDBStrategy:
    """Test all vector database strategies using parameterization."""

    @pytest.mark.parametrize(
        "vector_db_class, similarity_metric_type",
        VECTOR_DB_PARAMS,
    )
    def test_add_and_get_knn(self, vector_db_class, similarity_metric_type):
        """Test adding embeddings and retrieving nearest neighbors."""
        vector_db = vector_db_class(similarity_metric_type=similarity_metric_type)

        # Test with a single embedding
        embedding = [0.1, 0.2, 0.3]
        metadata1 = EmbeddingMetadataObj(response="test response 1")
        id1 = vector_db.add(embedding=embedding, metadata=metadata1)

        knn = vector_db.get_knn(embedding=embedding, k=1)
        assert len(knn) == 1
        assert abs(knn[0][0] - 1.0) < 1e-5  # Should be a perfect match
        assert knn[0][1] == id1

        # Test with multiple embeddings
        metadata2 = EmbeddingMetadataObj(response="test response 2")
        metadata3 = EmbeddingMetadataObj(response="test response 3")
        vector_db.add(embedding=[0.2, 0.3, 0.4], metadata=metadata2)
        vector_db.add(embedding=[0.3, 0.4, 0.5], metadata=metadata3)

        # Verify we get all embeddings when k is large enough
        knn = vector_db.get_knn(embedding=embedding, k=3)
        assert len(knn) == 3

        # Verify k limiting works
        knn = vector_db.get_knn(embedding=embedding, k=2)
        assert len(knn) == 2

    @pytest.mark.parametrize(
        "vector_db_class, similarity_metric_type",
        VECTOR_DB_PARAMS,
    )
    def test_remove(self, vector_db_class, similarity_metric_type):
        """Test removing embeddings from the vector database."""
        vector_db = vector_db_class(similarity_metric_type=similarity_metric_type)

        # Add multiple embeddings
        metadata1 = EmbeddingMetadataObj(response="test response 1")
        metadata2 = EmbeddingMetadataObj(response="test response 2")
        id1 = vector_db.add(embedding=[0.1, 0.2, 0.3], metadata=metadata1)
        id2 = vector_db.add(embedding=[0.2, 0.3, 0.4], metadata=metadata2)

        # Verify both exist
        knn = vector_db.get_knn(embedding=[0.1, 0.2, 0.3], k=2)
        assert len(knn) == 2

        # Remove one embedding
        vector_db.remove(embedding_id=id1)

        # Verify only one remains
        knn = vector_db.get_knn(embedding=[0.1, 0.2, 0.3], k=2)
        assert len(knn) == 1
        assert knn[0][1] == id2

    @pytest.mark.parametrize(
        "vector_db_class, similarity_metric_type",
        VECTOR_DB_PARAMS,
    )
    def test_reset(self, vector_db_class, similarity_metric_type):
        """Test resetting the vector database."""
        vector_db = vector_db_class(similarity_metric_type=similarity_metric_type)

        # Add multiple embeddings
        metadata1 = EmbeddingMetadataObj(response="test response 1")
        metadata2 = EmbeddingMetadataObj(response="test response 2")
        metadata3 = EmbeddingMetadataObj(response="test response 3")
        vector_db.add(embedding=[0.1, 0.2, 0.3], metadata=metadata1)
        vector_db.add(embedding=[0.2, 0.3, 0.4], metadata=metadata2)
        vector_db.add(embedding=[0.3, 0.4, 0.5], metadata=metadata3)

        # Verify embeddings exist
        knn = vector_db.get_knn(embedding=[0.1, 0.2, 0.3], k=3)
        assert len(knn) == 3

        # Reset the database
        vector_db.reset()

        # Verify no embeddings remain
        knn = vector_db.get_knn(embedding=[0.1, 0.2, 0.3], k=3)
        assert len(knn) == 0

    @pytest.mark.parametrize(
        "vector_db_class, similarity_metric_type",
        VECTOR_DB_PARAMS,
    )
    def test_metadata_operations(self, vector_db_class, similarity_metric_type):
        """Test metadata operations of the vector database."""
        vector_db = vector_db_class(similarity_metric_type=similarity_metric_type)

        # Add embedding with metadata (embedding_id will be set automatically)
        metadata = EmbeddingMetadataObj(response="test response")
        embedding_id = vector_db.add(embedding=[0.1, 0.2, 0.3], metadata=metadata)

        # Test get metadata
        retrieved_metadata = vector_db.get_metadata(embedding_id)
        assert retrieved_metadata.response == "test response"
        assert (
            retrieved_metadata.embedding_id == embedding_id
        )  # Should be set automatically

        # Test update metadata
        updated_metadata = EmbeddingMetadataObj(response="updated response")
        vector_db.update_metadata(embedding_id, updated_metadata)

        retrieved_metadata = vector_db.get_metadata(embedding_id)
        assert retrieved_metadata.response == "updated response"

        # Test get all metadata objects
        all_metadata = vector_db.get_all_embedding_metadata_objects()
        assert len(all_metadata) == 1
        assert all_metadata[0].response == "updated response"


if __name__ == "__main__":
    unittest.main()
