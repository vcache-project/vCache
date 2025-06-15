from typing import Dict, List

import hnswlib

from vcache.vcache_core.cache.embedding_store.embedding_metadata_storage.embedding_metadata_obj import (
    EmbeddingMetadataObj,
)
from vcache.vcache_core.cache.embedding_store.vector_db.vector_db import (
    SimilarityMetricType,
    VectorDB,
)

"""
Run 'sudo apt-get install build-essential' on Linux Debian/Ubuntu to install the build-essential package
"""


class HNSWLibVectorDB(VectorDB):
    """
    HNSWLib-based vector database implementation that stores both embeddings and metadata.
    """

    def __init__(
        self,
        similarity_metric_type: SimilarityMetricType = SimilarityMetricType.COSINE,
        max_capacity: int = 100000,
    ):
        """
        Initialize HNSWLib vector database.

        Args:
            similarity_metric_type: The similarity metric to use for comparisons.
            max_capacity: Maximum number of vectors the database can store.
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
        self.metadata_storage: Dict[int, EmbeddingMetadataObj] = {}

    def add(self, embedding: List[float], metadata: EmbeddingMetadataObj) -> int:
        """
        Add an embedding vector and its metadata to the database.

        Args:
            embedding: The embedding vector to add.
            metadata: The metadata object associated with the embedding.

        Returns:
            The unique ID assigned to the added embedding.
        """
        if self.index is None:
            self._init_vector_store(len(embedding))

        embedding_id = self.__next_embedding_id
        self.index.add_items(embedding, embedding_id)

        # Automatically set the embedding_id in the metadata
        metadata.embedding_id = embedding_id
        self.metadata_storage[embedding_id] = metadata

        self.embedding_count += 1
        self.__next_embedding_id += 1
        return embedding_id

    def remove(self, embedding_id: int) -> int:
        """
        Remove an embedding and its metadata from the database.

        Args:
            embedding_id: The ID of the embedding to remove.

        Returns:
            The ID of the removed embedding.

        Raises:
            ValueError: If the index is not initialized or embedding not found.
        """
        if self.index is None:
            raise ValueError("Index is not initialized")
        if embedding_id not in self.metadata_storage:
            raise ValueError(f"Embedding with ID {embedding_id} not found")

        self.index.mark_deleted(embedding_id)
        del self.metadata_storage[embedding_id]
        self.embedding_count -= 1
        return embedding_id

    def get_knn(self, embedding: List[float], k: int) -> List[tuple[float, int]]:
        """
        Get k-nearest neighbors for the given embedding.

        Args:
            embedding: The query embedding vector.
            k: The number of nearest neighbors to return.

        Returns:
            List of tuples containing similarity scores and embedding IDs.
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

        # Filter out deleted embeddings (those not in metadata_storage)
        results = []
        for score, embedding_id in zip(similarity_scores, id_list):
            if embedding_id in self.metadata_storage:
                results.append((score, embedding_id))

        return results

    def get_metadata(self, embedding_id: int) -> EmbeddingMetadataObj:
        """
        Get metadata for a specific embedding.

        Args:
            embedding_id: The ID of the embedding to get metadata for.

        Returns:
            The metadata object for the embedding.
        """
        if embedding_id not in self.metadata_storage:
            raise ValueError(f"Metadata for embedding ID {embedding_id} not found")
        return self.metadata_storage[embedding_id]

    def update_metadata(
        self, embedding_id: int, metadata: EmbeddingMetadataObj
    ) -> EmbeddingMetadataObj:
        """
        Update metadata for a specific embedding.

        Args:
            embedding_id: The ID of the embedding to update metadata for.
            metadata: The new metadata object.

        Returns:
            The updated metadata object.
        """
        if embedding_id not in self.metadata_storage:
            raise ValueError(f"Metadata for embedding ID {embedding_id} not found")

        self.metadata_storage[embedding_id] = metadata
        self.metadata_storage[embedding_id].embedding_id = embedding_id
        return metadata

    def get_all_embedding_metadata_objects(self) -> List[EmbeddingMetadataObj]:
        """
        Get all embedding metadata objects in the database.

        Returns:
            A list of all embedding metadata objects.
        """
        return list(self.metadata_storage.values())

    def reset(self) -> None:
        """
        Reset the vector database to empty state.
        """
        if self.dim is None:
            return
        self._init_vector_store(self.dim)
        self.embedding_count = 0
        self.__next_embedding_id = 0
        self.metadata_storage.clear()

    def _init_vector_store(self, embedding_dim: int):
        """
        Initialize the HNSWLib index with the given embedding dimension.

        Args:
            embedding_dim: The dimension of the embedding vectors.

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
        """
        Check if the vector database is empty.

        Returns:
            True if the database contains no embeddings, False otherwise.
        """
        return self.embedding_count == 0
