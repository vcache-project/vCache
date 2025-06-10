from typing import List

import chromadb

from vcache.vcache_core.cache.embedding_store.vector_db.vector_db import (
    SimilarityMetricType,
    VectorDB,
)


class ChromaVectorDB(VectorDB):
    """
    ChromaDB-based vector database implementation for efficient similarity search.
    """

    def __init__(
        self, similarity_metric_type: SimilarityMetricType = SimilarityMetricType.COSINE
    ):
        """
        Initialize ChromaDB vector database.

        Args:
            similarity_metric_type: The similarity metric to use for comparisons.
        """
        self.__next_embedding_id = 0
        self.collection = None
        self.client = None
        self.similarity_metric_type = similarity_metric_type

    def add(self, embedding: List[float]) -> int:
        """
        Add an embedding vector to the database.

        Args:
            embedding: The embedding vector to add.

        Returns:
            The unique ID assigned to the added embedding.
        """
        if self.collection is None:
            self._init_vector_store(len(embedding))
        id = self.__next_embedding_id
        self.collection.add(embeddings=[embedding], ids=[str(id)])
        self.__next_embedding_id += 1
        return id

    def remove(self, embedding_id: int) -> int:
        """
        Remove an embedding from the database.

        Args:
            embedding_id: The ID of the embedding to remove.

        Returns:
            The ID of the removed embedding.

        Raises:
            ValueError: If the collection is not initialized.
        """
        if self.collection is None:
            raise ValueError("Collection is not initialized")
        self.collection.delete(ids=[str(embedding_id)])
        return embedding_id

    def get_knn(self, embedding: List[float], k: int) -> List[tuple[float, int]]:
        """
        Get k-nearest neighbors for the given embedding.

        Args:
            embedding: The query embedding vector.
            k: The number of nearest neighbors to return.

        Returns:
            List of tuples containing similarity scores and embedding IDs.

        Raises:
            ValueError: If the collection is not initialized.
        """
        if self.collection is None:
            raise ValueError("Collection is not initialized")
        if self.collection.count() == 0:
            return []
        k_ = min(k, self.collection.count())
        results = self.collection.query(
            query_embeddings=[embedding], n_results=k_, include=["distances"]
        )
        distances = results.get("distances", [[]])[0]
        ids = results.get("ids", [[]])[0]
        return [
            (
                self.transform_similarity_score(
                    float(dist), self.similarity_metric_type.value
                ),
                int(idx),
            )
            for dist, idx in zip(distances, ids)
        ]

    def reset(self) -> None:
        """
        Reset the vector database to empty state.
        """
        if self.collection is not None:
            self.collection.delete(ids=self.collection.get()["ids"])
        self.__next_embedding_id = 0

    def _init_vector_store(self, embedding_dim: int):
        """
        Initialize the ChromaDB collection with the given embedding dimension.

        Args:
            embedding_dim: The dimension of the embedding vectors.

        Raises:
            ValueError: If the similarity metric type is invalid.
        """
        self.client = chromadb.Client()
        collection_name = f"vcache_collection_{id(self)}"
        metric_type = self.similarity_metric_type.value
        match metric_type:
            case "cosine":
                space = "cosine"
            case "euclidean":
                space = "l2"
            case _:
                raise ValueError(f"Invalid similarity metric type: {metric_type}")
        self.collection = self.client.create_collection(
            name=collection_name,
            metadata={"dimension": embedding_dim, "hnsw:space": space},
            get_or_create=True,
        )

    def is_empty(self) -> bool:
        """
        Check if the vector database is empty.

        Returns:
            True if the database contains no embeddings, False otherwise.
        """
        return self.collection.count() == 0
