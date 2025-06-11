from typing import List
import threading

import faiss
import numpy as np

from vcache.vcache_core.cache.embedding_store.vector_db.vector_db import (
    SimilarityMetricType,
    VectorDB,
)


class FAISSVectorDB(VectorDB):
    def __init__(
        self, similarity_metric_type: SimilarityMetricType = SimilarityMetricType.COSINE
    ):
        self.similarity_metric_type = similarity_metric_type
        self.__next_embedding_id = 0
        self.index = None
        # REVIEW COMMENT: Consider using threading.Lock() instead of RLock() for better performance
        # RLock allows recursive locking which may mask potential issues and is slower
        self._operation_lock = threading.RLock()

    def transform_similarity_score(
        self, similarity_score: float, metric_type: str
    ) -> float:
        # Override the default transform_similarity_score method
        match metric_type:
            case "cosine":
                return similarity_score
            case "euclidean":
                return 1 - similarity_score
            case _:
                raise ValueError(f"Invalid similarity metric type: {metric_type}")

    def add(self, embedding: List[float]) -> int:
        """
        Thread-safe addition of embedding to the vector database.
        
        Args:
            embedding: List[float] - The embedding vector to add
            
        Returns:
            int - The unique ID assigned to the embedding
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
        """
        Thread-safe removal of embedding from the vector database.
        
        Args:
            embedding_id: int - The ID of the embedding to remove
            
        Returns:
            int - The ID of the removed embedding
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
        """
        Thread-safe k-nearest neighbors search.
        
        Args:
            embedding: List[float] - The query embedding
            k: int - Number of nearest neighbors to return
            
        Returns:
            List[tuple[float, int]] - List of (similarity_score, embedding_id) tuples
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
            # REVIEW COMMENT: Potential bug - filtering IDs after transforming similarities
            # can cause index mismatch. Filter both arrays together or use enumerate.
            similarity_scores = [
                self.transform_similarity_score(sim, metric_type) for sim in similarities[0]
            ]
            id_list = [int(id) for id in ids[0] if id != -1]  # Filter out invalid IDs
            return list(zip(similarity_scores[:len(id_list)], id_list))

    def reset(self) -> None:
        """
        Thread-safe reset of the vector database.
        """
        with self._operation_lock:
            # REVIEW COMMENT: Setting index to None loses dimension info.
            # Consider storing dim separately or reinitializing with previous dim
            self.index = None
            self.__next_embedding_id = 0

    def _init_vector_store(self, embedding_dim: int):
        """
        Initialize the vector store. Should be called within a lock context.
        
        Args:
            embedding_dim: int - The dimension of the embedding vectors
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
        """
        Thread-safe check if the vector database is empty.
        
        Returns:
            bool - True if the database is empty, False otherwise
        """
        with self._operation_lock:
            if self.index is None:
                return True
            return self.index.ntotal == 0
