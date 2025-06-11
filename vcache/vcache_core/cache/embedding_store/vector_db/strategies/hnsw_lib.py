import threading
from typing import List

import hnswlib

from vcache.vcache_core.cache.embedding_store.vector_db.vector_db import (
    SimilarityMetricType,
    VectorDB,
)

"""
Run 'sudo apt-get install build-essential' on Linux Debian/Ubuntu to install the build-essential package
"""


class HNSWLibVectorDB(VectorDB):
    """A vector database implementation using HNSWLib.

    This class provides a thread-safe vector database that stores embeddings and
    performs k-nearest neighbor searches using the HNSWLib library.

    Attributes:
        embedding_count (int): The current number of embeddings in the database.
        similarity_metric_type (SimilarityMetricType): The metric for measuring similarity.
        index (hnswlib.Index): The underlying HNSWLib index.
    """

    def __init__(
        self,
        similarity_metric_type: SimilarityMetricType = SimilarityMetricType.COSINE,
        max_capacity: int = 100000,
    ):
        self.embedding_count = 0
        self.__next_embedding_id = 0
        self.similarity_metric_type = similarity_metric_type
        self.space = None
        self.dim = None
        self.max_elements = max_capacity
        self.ef_construction = None
        self.M = None
        self.ef = None
        self.index = None
        self._operation_lock = threading.RLock()

    def add(self, embedding: List[float]) -> int:
        """Add an embedding to the database, initializing the index if needed.

        This method is thread-safe.

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
            self.index.add_items(embedding, embedding_id)
            self.embedding_count += 1
            self.__next_embedding_id += 1

            return embedding_id

    def remove(self, embedding_id: int) -> int:
        """Mark an embedding as deleted in the database.

        This method is thread-safe. Note that HNSWLib only marks items for
        deletion; it does not immediately reclaim the space.

        Args:
            embedding_id (int): The ID of the embedding to remove.

        Returns:
            int: The ID of the removed embedding.
        """
        with self._operation_lock:
            if self.index is None:
                raise ValueError("Index is not initialized")
            self.index.mark_deleted(embedding_id)
            self.embedding_count -= 1
            return embedding_id

    def get_knn(self, embedding: List[float], k: int) -> List[tuple[float, int]]:
        """Find the k-nearest neighbors for a given embedding.

        This method is thread-safe. If the database is empty or `k` is larger
        than the number of items, it returns the maximum number of neighbors possible.

        Args:
            embedding (List[float]): The query embedding.
            k (int): The number of nearest neighbors to return.

        Returns:
            List[tuple[float, int]]: List of (similarity_score, embedding_id) tuples.
        """
        with self._operation_lock:
            if self.index is None:
                return []
            k_ = min(k, self.embedding_count)
            if k_ == 0:
                return []
            ids, similarities = self.index.knn_query(embedding, k=k_)
            metric_type = self.similarity_metric_type.value
            similarity_scores = [
                self.transform_similarity_score(sim, metric_type)
                for sim in similarities[0]
            ]
            id_list = [int(id) for id in ids[0]]
            return list(zip(similarity_scores, id_list))

    def reset(self) -> None:
        """Clear all embeddings from the database."""
        with self._operation_lock:
            if self.dim is None:
                return
            self._init_vector_store(self.dim)
            self.embedding_count = 0
            self.__next_embedding_id = 0

    def _init_vector_store(self, embedding_dim: int):
        """Initialize the HNSWLib index.

        This method configures the index with the specified space ('cosine' or 'l2')
        and dimensions. It must be called within a locked context.

        Args:
            embedding_dim (int): The dimension of the embedding vectors.
        """
        metric_type = self.similarity_metric_type.value
        match metric_type:
            case "cosine":
                self.space = "cosine"
            case "euclidean":
                self.space = "l2"
            case _:
                raise ValueError(f"Invalid similarity metric type: {metric_type}")
        self.dim = embedding_dim
        self.ef_construction = 350
        self.M = 52
        self.ef = 400
        self.index = hnswlib.Index(space=self.space, dim=self.dim)
        self.index.init_index(
            max_elements=self.max_elements,
            ef_construction=self.ef_construction,
            M=self.M,
        )
        self.index.set_ef(self.ef)

    def is_empty(self) -> bool:
        """Check if the database contains any embeddings.

        This method is thread-safe.

        Returns:
            bool: True if the database has no embeddings, False otherwise.
        """
        with self._operation_lock:
            return self.embedding_count == 0
