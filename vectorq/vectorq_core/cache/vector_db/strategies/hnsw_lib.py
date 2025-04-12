import hnswlib
from typing import List, TYPE_CHECKING
from vectorq.vectorq_core.cache.vector_db.strategy import VectorDBStrategy

if TYPE_CHECKING:
    from vectorq.vectorq_core.cache.vector_db.embedding_metadata_storage.embedding_metadata_obj import EmbeddingMetadataObj
    from vectorq.config import VectorQConfig
    
'''
Run 'sudo apt-get install build-essential' on Linux Debian/Ubuntu to install the build-essential package
'''
    
class HNSWLib(VectorDBStrategy):
    
    def __init__(self, vectorq_config: "VectorQConfig"):
        super().__init__(vectorq_config)
        self.embedding_count = 0
        self.__next_embedding_id = 0
        
        self.space = None
        self.dim = None
        self.max_elements = None
        self.ef_construction = None
        self.M = None
        self.ef = None
        self.index = None
    
    def add(self, embedding: List[float]) -> int:
        if self.index is None:
            self._init_vector_store(len(embedding))
        self.index.add_items(embedding, self.__next_embedding_id)
        self.embedding_count += 1
        self.__next_embedding_id += 1
        return self.__next_embedding_id - 1
    
    def remove(self, embedding_id: int) -> int:
        if self.index is None:
            raise ValueError("Index is not initialized")
        self.index.mark_deleted(embedding_id)
        self.embedding_count -= 1
        return embedding_id
    
    def get_knn(self, embedding: List[float], k: int) -> List[tuple[float, int]]:
        if self.index is None:
            raise ValueError("Index is not initialized")
        k_ = min(k, self.embedding_count)
        if k_ == 0:
            return []
        ids, similarities = self.index.knn_query(embedding, k=k_)
        metric_type = self.vectorq_config._vector_db_similarity_metric_type.value
        similarity_scores = [self.transform_similarity_score(sim, metric_type) for sim in similarities[0]]
        id_list = [int(id) for id in ids[0]]
        return list(zip(similarity_scores, id_list))
    
    def reset(self) -> None:
        if self.dim is None:
            return
        self._init_vector_store(self.dim)
        self.embedding_count = 0
        self.__next_embedding_id = 0
    
    def _init_vector_store(self, embedding_dim: int):
        metric_type = self.vectorq_config._vector_db_similarity_metric_type.value
        match metric_type:
            case "cosine":
                self.space = "cosine"
            case "euclidean":
                self.space = "l2"
            case _:
                raise ValueError(f"Invalid similarity metric type: {metric_type}")
        self.dim = embedding_dim
        self.max_elements = self.vectorq_config.max_capacity
        self.ef_construction = 350
        self.M = 52
        self.ef = 400
        self.index = hnswlib.Index(space=self.space, dim=self.dim)
        self.index.init_index(max_elements=self.max_elements, ef_construction=self.ef_construction, M=self.M)
        self.index.set_ef(self.ef)
