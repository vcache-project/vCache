import threading
from typing import List

import faiss
import numpy as np

from vcache.vcache_core.cache.embedding_store.vector_db.vector_db import (
    SimilarityMetricType,
    VectorDB,
)


class FAISSVectorDB(VectorDB):
    """A vector database implementation using FAISS.

    This class provides a thread-safe vector database that stores embeddings and
    performs k-nearest neighbor searches using the FAISS library. It supports
    both cosine similarity and L2 (Euclidean) distance.

    Attributes:
        similarity_metric_type (SimilarityMetricType): The metric for measuring similarity.
        index (faiss.Index): The underlying FAISS index.
    """

    def __init__(
        self, similarity_metric_type: SimilarityMetricType = SimilarityMetricType.COSINE
    ):
        self.similarity_metric_type = similarity_metric_type
        self.__next_embedding_id = 0
        self.index = None
        self._operation_lock = threading.RLock()

    def transform_similarity_score(
        self, similarity_score: float, metric_type: str
    ) -> float:
        """Transform a raw score from FAISS to a normalized similarity score.

        For cosine similarity, FAISS's IndexFlatIP returns the inner product, which
        is already the similarity, so no transformation is needed. For L2 distance,
        the raw distance is converted to a similarity score.

        Args:
            similarity_score (float): The raw score from the FAISS index.
            metric_type (str): The similarity metric used ('cosine' or 'euclidean').

        Returns:
            float: The transformed similarity score, normalized to [0, 1].
        """
        match metric_type:
            case "cosine":
                return similarity_score
            case "euclidean":
                return 1 - similarity_score
            case _:
                raise ValueError(f"Invalid similarity metric type: {metric_type}")

    def add(self, embedding: List[float]) -> int:
        """Add an embedding to the database, initializing the index if needed.

        This method is thread-safe. For cosine similarity, embeddings are
        L2-normalized before being added to the index.

        Args:
            embedding (List[float]): The embedding vector to add.

        Returns:
            int: The unique ID assigned to the added embedding.
        """
        with self._operation_lock:
            if self.index is None:
                self._init_vector_store(len(embedding))

            # Atomic ID generation and assignment
            embedding_id = self.__next_embedding_id
            ids = np.array([embedding_id], dtype=np.int64)
            embedding_array = np.array([embedding], dtype=np.float32)
            metric_type = self.similarity_metric_type.value
            # Normalize the embedding vector if the metric type is cosine
            if metric_type == "cosine":
                faiss.normalize_L2(embedding_array)
            self.index.add_with_ids(embedding_array, ids)
            self.__next_embedding_id += 1

            return embedding_id

    def remove(self, embedding_id: int) -> int:
        """Remove an embedding from the database by its ID.

        This method is thread-safe.

        Args:
            embedding_id (int): The ID of the embedding to remove.

        Returns:
            int: The ID of the removed embedding.
        """
        with self._operation_lock:
            if self.index is None:
                raise ValueError("Index is not initialized")
            id_array = np.array([embedding_id], dtype=np.int64)
            self.index.remove_ids(
                faiss.IDSelectorBatch(id_array.size, faiss.swig_ptr(id_array))
            )
            return embedding_id

    def get_knn(self, embedding: List[float], k: int) -> List[tuple[float, int]]:
        """Find the k-nearest neighbors for a given embedding.

        This method is thread-safe. For cosine similarity, the query embedding is
        L2-normalized before searching. Invalid IDs (-1) are filtered from results.

        Args:
            embedding (List[float]): The query embedding.
            k (int): The number of nearest neighbors to return.

        Returns:
            List[tuple[float, int]]: List of (similarity_score, embedding_id) tuples.
        """
        with self._operation_lock:
            if self.index is None:
                return []
            if self.index.ntotal == 0:
                return []
            k_ = min(k, self.index.ntotal)
            embedding_array = np.array([embedding], dtype=np.float32)
            metric_type = self.similarity_metric_type.value
            # Normalize the embedding vector if the metric type is cosine
            if metric_type == "cosine":
                faiss.normalize_L2(embedding_array)
            similarities, ids = self.index.search(embedding_array, k_)
            similarity_scores = [
                self.transform_similarity_score(sim, metric_type)
                for sim in similarities[0]
            ]
            id_list = [int(id) for id in ids[0] if id != -1]  # Filter out invalid IDs
            return list(zip(similarity_scores[: len(id_list)], id_list))

    def reset(self) -> None:
        """Clear all embeddings from the database and reset the index."""
        with self._operation_lock:
            self.index = None
            self.__next_embedding_id = 0

    def _init_vector_store(self, embedding_dim: int):
        """Initialize the FAISS index.

        This method selects the appropriate FAISS index type based on the
        similarity metric (IndexFlatIP for cosine, IndexFlatL2 for Euclidean)
        and wraps it with an IDMap to support custom embedding IDs. It must be
        called within a locked context.

        Args:
            embedding_dim (int): The dimension of the embedding vectors.
        """
        metric_type = self.similarity_metric_type.value
        match metric_type:
            case "cosine":
                # Use IndexFlatIP for cosine similarity (inner product after normalization)
                self.index = faiss.IndexFlatIP(embedding_dim)
            case "euclidean":
                # Use IndexFlatL2 for euclidean distance
                self.index = faiss.IndexFlatL2(embedding_dim)
            case _:
                raise ValueError(f"Invalid similarity metric type: {metric_type}")

        # Wrap with IDMap to support custom IDs
        self.index = faiss.IndexIDMap(self.index)

    def is_empty(self) -> bool:
        """Check if the database contains any embeddings.

        This method is thread-safe.

        Returns:
            bool: True if the database has no embeddings, False otherwise.
        """
        with self._operation_lock:
            if self.index is None:
                return True
            return self.index.ntotal == 0
