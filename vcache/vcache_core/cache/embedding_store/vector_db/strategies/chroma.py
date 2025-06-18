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
        """Initializes the ChromaDB vector database.

        Args:
            similarity_metric_type (SimilarityMetricType): The similarity metric
                to use for comparisons.
        """
        self.__next_embedding_id = 0
        self.collection = None
        self.client = None
        self.similarity_metric_type = similarity_metric_type

    def add(self, embedding: List[float]) -> int:
        """Adds an embedding vector to the database.

        Args:
            embedding (List[float]): The embedding vector to add.

        Returns:
            int: The unique ID assigned to the added embedding.
        """
        if self.collection is None:
            self._init_vector_store(len(embedding))
        id = self.__next_embedding_id
        self.collection.add(embeddings=[embedding], ids=[str(id)])
        self.__next_embedding_id += 1
        return id

    def remove(self, embedding_id: int) -> int:
        """Removes an embedding from the database.

        Args:
            embedding_id (int): The ID of the embedding to remove.

        Returns:
            int: The ID of the removed embedding.

        Raises:
            ValueError: If the collection has not been initialized.
        """
        if self.collection is None:
            raise ValueError("Collection is not initialized")
        self.collection.delete(ids=[str(embedding_id)])
        return embedding_id

    def get_knn(self, embedding: List[float], k: int) -> List[tuple[float, int]]:
        """Gets k-nearest neighbors for a given embedding.

        Args:
            embedding (List[float]): The query embedding vector.
            k (int): The number of nearest neighbors to return.

        Returns:
            List[tuple[float, int]]: A list of tuples containing similarity
            scores and embedding IDs.

        Raises:
            ValueError: If the collection has not been initialized.
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
        """Resets the vector database to an empty state."""
        if self.collection is not None:
            self.collection.delete(ids=self.collection.get()["ids"])
        self.__next_embedding_id = 0

    def _init_vector_store(self, embedding_dim: int):
        """Initializes the ChromaDB collection.

        Args:
            embedding_dim (int): The dimension of the embedding vectors.

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
        """Checks if the vector database is empty.

        Returns:
            bool: True if the database contains no embeddings, False otherwise.
        """
        return self.collection.count() == 0

    def size(self) -> int:
        """Gets the number of embeddings in the vector database.

        Returns:
            int: The number of embeddings in the vector database.
        """
        return self.collection.count()
