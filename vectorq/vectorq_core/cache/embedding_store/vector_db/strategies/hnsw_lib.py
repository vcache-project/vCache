from typing import List

import hnswlib

from vectorq.vectorq_core.cache.embedding_store.vector_db.vector_db import (
    SimilarityMetricType,
    VectorDB,
)

"""
Run 'sudo apt-get install build-essential' on Linux Debian/Ubuntu to install the build-essential package
"""


class HNSWLibVectorDB(VectorDB):
    def __init__(
        self,
        similarity_metric_type: SimilarityMetricType = SimilarityMetricType.COSINE,
        max_capacity: int = 100000,
    ):
        self.similarity_metric_type = similarity_metric_type
        self.space = None
        self.dim = None
        self.ef_construction = None
        self.M = None
        self.ef = None
        self.index = None
        self.embedding_id_to_question_idx = {}
        self.evicted_qu_id = []
        self.max_capacity = max_capacity

    def add(self, embedding: List[float], insert_id, question_idx: int = -1):
        if self.index is None:
            self._init_vector_store(len(embedding))    
        self.index.add_items(embedding, insert_id, replace_deleted=True)
        self.embedding_id_to_question_idx[insert_id] = question_idx
        return True

    def get_evicted_ids(self) -> List[int]:
        return self.evicted_qu_id
    
    def remove(self, embedding_id: int) -> int:
        if self.index is None:
            raise ValueError("Index is not initialized")
        self.evicted_qu_id.append(self.embedding_id_to_question_idx[embedding_id])
        self.index.mark_deleted(embedding_id)
        return embedding_id

    def get_knn(self, embedding: List[float], k: int) -> List[tuple[float, int]]:
        if self.index is None:
            raise ValueError("Index is not initialized")
        # k_ = min(k, self.embedding_count) ## need to patch this
        if k == 0:
            return []
        ids, similarities = self.index.knn_query(embedding, k=k)
        metric_type = self.similarity_metric_type.value
        similarity_scores = [self.transform_similarity_score(sim, metric_type) for sim in similarities[0]]
        id_list = [int(id) for id in ids[0]]
        question_idx_list = [self.embedding_id_to_question_idx[id] for id in id_list]
        return list(zip(similarity_scores, id_list, question_idx_list))

    def reset(self) -> None:
        if self.dim is None:
            return
        self.evicted_qu_id = []
        self.embedding_id_to_question_idx = {}
        self._init_vector_store(self.dim)


    def _init_vector_store(self, embedding_dim: int):
        metric_type = self.similarity_metric_type.value
        match metric_type:
            case "cosine":
                self.space = "cosine"
            case "euclidean":
                self.space = "l2"
            case _:
                raise ValueError(f"Invalid similarity metric type: {metric_type}")
        self.dim = embedding_dim
        self.max_elements = self.max_capacity
        print(self.max_elements)
        self.ef_construction = 350
        self.M = 52
        self.ef = 400
        self.index = hnswlib.Index(space=self.space, dim=self.dim)
        self.index.init_index(max_elements=self.max_elements, ef_construction=self.ef_construction, M=self.M, allow_replace_deleted=True)
        self.index.set_ef(self.ef)
