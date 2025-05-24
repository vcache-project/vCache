from typing import List

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
        if self.index is None:
            self._init_vector_store(len(embedding))
        id = self.__next_embedding_id
        ids = np.array([id], dtype=np.int64)
        embedding_array = np.array([embedding], dtype=np.float32)
        metric_type = self.similarity_metric_type.value
        # Normalize the embedding vector if the metric type is cosine
        if metric_type == "cosine":
            faiss.normalize_L2(embedding_array)
        self.index.add_with_ids(embedding_array, ids)
        self.__next_embedding_id += 1
        return id

    def remove(self, embedding_id: int) -> int:
        if self.index is None:
            raise ValueError("Index is not initialized")
        id_array = np.array([embedding_id], dtype=np.int64)
        self.index.remove_ids(
            faiss.IDSelectorBatch(id_array.size, faiss.swig_ptr(id_array))
        )
        return embedding_id

    def get_knn(self, embedding: List[float], k: int) -> List[tuple[float, int]]:
        if self.index is None:
            raise ValueError("Index is not initialized")
        if self.index.ntotal == 0:
            return []
        k_ = min(k, self.index.ntotal)
        query_vector = np.array([embedding], dtype=np.float32)
        metric_type = self.similarity_metric_type.value
        # Normalize the query vector if the metric type is cosine
        if metric_type == "cosine":
            faiss.normalize_L2(query_vector)
        distances, indices = self.index.search(query_vector, k_)
        # Filter out results where index is -1 (deleted embeddings)
        filtered_results = [
            (distances[0][i], indices[0][i])
            for i in range(len(indices[0]))
            if indices[0][i] != -1
        ]
        return [
            (self.transform_similarity_score(dist, metric_type), int(idx))
            for dist, idx in filtered_results
        ]

    def reset(self) -> None:
        if self.index is not None:
            dim = self.index.d
            self._init_vector_store(dim)
        self.__next_embedding_id = 0

    def _init_vector_store(self, embedding_dim: int):
        metric_type = self.similarity_metric_type.value
        match metric_type:
            case "cosine":
                faiss_metric = faiss.METRIC_INNER_PRODUCT
            case "euclidean":
                faiss_metric = faiss.METRIC_L2
            case _:
                raise ValueError(f"Invalid similarity metric type: {metric_type}")
        self.index = faiss.index_factory(embedding_dim, "IDMap,Flat", faiss_metric)

    def is_empty(self) -> bool:
        return self.index.ntotal == 0
