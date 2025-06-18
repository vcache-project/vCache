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
    """
    HNSWLib-based vector database implementation for efficient similarity search.
    """

    def __init__(
        self,
        similarity_metric_type: SimilarityMetricType = SimilarityMetricType.COSINE,
        max_capacity: int = 100000,
    ):
        """Initializes the HNSWLib vector database.

        Args:
            similarity_metric_type (SimilarityMetricType): The similarity metric
                to use for comparisons.
            max_capacity (int): The maximum number of vectors the database can store.
        """
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

    def add(self, embedding: List[float]) -> int:
        """Adds an embedding vector to the database.

        Args:
            embedding (List[float]): The embedding vector to add.

        Returns:
            int: The unique ID assigned to the added embedding.
        """
        if self.index is None:
            self._init_vector_store(len(embedding))
        self.index.add_items(embedding, self.__next_embedding_id)
        self.embedding_count += 1
        self.__next_embedding_id += 1
        return self.__next_embedding_id - 1

    def remove(self, embedding_id: int) -> int:
        """Marks an embedding for deletion in the database.

        Note:
            HNSWLib does not physically remove data, but marks it as deleted.

        Args:
            embedding_id (int): The ID of the embedding to remove.

        Returns:
            int: The ID of the removed embedding.

        Raises:
            ValueError: If the index has not been initialized.
        """
        if self.index is None:
            raise ValueError("Index is not initialized")
        self.index.mark_deleted(embedding_id)
        self.embedding_count -= 1
        return embedding_id

    def get_knn(self, embedding: List[float], k: int) -> List[tuple[float, int]]:
        """Gets k-nearest neighbors for a given embedding.

        Args:
            embedding (List[float]): The query embedding vector.
            k (int): The number of nearest neighbors to return.

        Returns:
            List[tuple[float, int]]: A list of tuples containing similarity
            scores and embedding IDs.
        """
        if self.index is None:
            return []
        k_ = min(k, self.embedding_count)
        if k_ == 0:
            return []
        ids, similarities = self.index.knn_query(embedding, k=k_)
        metric_type = self.similarity_metric_type.value
        similarity_scores = [
            self.transform_similarity_score(sim, metric_type) for sim in similarities[0]
        ]
        id_list = [int(id) for id in ids[0]]
        return list(zip(similarity_scores, id_list))

    def reset(self) -> None:
        """Resets the vector database to an empty state."""
        if self.dim is None:
            return
        self._init_vector_store(self.dim)
        self.embedding_count = 0
        self.__next_embedding_id = 0

    def _init_vector_store(self, embedding_dim: int):
        """Initializes the HNSWLib index.

        Args:
            embedding_dim (int): The dimension of the embedding vectors.

        Raises:
            ValueError: If the similarity metric type is invalid.
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
        """Checks if the vector database is empty.

        Returns:
            bool: True if the database contains no embeddings, False otherwise.
        """
        return self.embedding_count == 0

    def size(self) -> int:
        """Gets the number of embeddings in the vector database.

        Returns:
            int: The number of embeddings in the vector database.
        """
        return self.embedding_count
