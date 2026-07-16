import json
import os
from typing import List

import hnswlib

from vcache.vcache_core.cache.embedding_store.vector_db.vector_db import (
    SimilarityMetricType,
    VectorDB,
)

"""
Run 'sudo apt-get install build-essential' on Linux Debian/Ubuntu to install the build-essential package
"""


class PersistentHNSWLibVectorDB(VectorDB):
    """
    HNSWLib-based vector database implementation that persists its index to
    disk, so it survives process restarts.

    This is a standalone implementation (not a subclass of `HNSWLibVectorDB`)
    so it does not depend on, or risk altering, that class's internals.
    hnswlib's own `save_index`/`load_index` only capture the graph and
    vectors, so a small JSON sidecar file next to the index file stores the
    remaining bookkeeping (dimension, space, and id counters) needed to
    resume exactly where the cache left off.
    """

    def __init__(
        self,
        persist_path: str,
        similarity_metric_type: SimilarityMetricType = SimilarityMetricType.COSINE,
        max_capacity: int = 100000,
    ):
        """Initializes the persistent HNSWLib vector database.

        Args:
            persist_path (str): Path to the file used to store the hnswlib
                index. A sidecar file at `persist_path + ".meta.json"` stores
                the remaining bookkeeping. If either file already exists, the
                database is restored from disk.
            similarity_metric_type (SimilarityMetricType): The similarity metric
                to use for comparisons.
            max_capacity (int): The maximum number of vectors the database can store.
        """
        self.persist_path = persist_path
        self._meta_path = persist_path + ".meta.json"
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

        if os.path.exists(self._meta_path) and os.path.exists(persist_path):
            self._load_from_disk()

    def add(self, embedding: List[float]) -> int:
        """Adds an embedding vector to the database and persists it to disk.

        Args:
            embedding (List[float]): The embedding vector to add.

        Returns:
            int: The unique ID assigned to the added embedding.
        """
        if self.index is None:
            self._init_vector_store(len(embedding))
        id = self.__next_embedding_id
        self.index.add_items(embedding, id)
        self.embedding_count += 1
        self.__next_embedding_id += 1
        self._save_to_disk()
        return id

    def remove(self, embedding_id: int) -> int:
        """Marks an embedding for deletion and persists the change to disk.

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
        self._save_to_disk()
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
        """Resets the vector database to an empty state and persists it."""
        if self.dim is not None:
            self._init_vector_store(self.dim)
        self.embedding_count = 0
        self.__next_embedding_id = 0
        self._save_to_disk()

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

    def _save_to_disk(self) -> None:
        """Persists the hnswlib index and its bookkeeping sidecar to disk."""
        self.index.save_index(self.persist_path)
        with open(self._meta_path, "w") as f:
            json.dump(
                {
                    "dim": self.dim,
                    "space": self.space,
                    "max_elements": self.max_elements,
                    "ef_construction": self.ef_construction,
                    "M": self.M,
                    "ef": self.ef,
                    "embedding_count": self.embedding_count,
                    "next_embedding_id": self.__next_embedding_id,
                },
                f,
            )

    def _load_from_disk(self) -> None:
        """Restores the hnswlib index and bookkeeping from disk."""
        with open(self._meta_path, "r") as f:
            meta = json.load(f)
        self.dim = meta["dim"]
        self.space = meta["space"]
        self.max_elements = meta["max_elements"]
        self.ef_construction = meta["ef_construction"]
        self.M = meta["M"]
        self.ef = meta["ef"]
        self.embedding_count = meta["embedding_count"]
        self.__next_embedding_id = meta["next_embedding_id"]
        self.index = hnswlib.Index(space=self.space, dim=self.dim)
        self.index.load_index(self.persist_path, max_elements=self.max_elements)
        self.index.set_ef(self.ef)
