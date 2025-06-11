import threading
from typing import List

import chromadb

from vcache.vcache_core.cache.embedding_store.vector_db.vector_db import (
    SimilarityMetricType,
    VectorDB,
)


class ChromaVectorDB(VectorDB):
    """A vector database implementation using ChromaDB.

    This class provides a thread-safe vector database that stores embeddings and
    performs k-nearest neighbor searches using the ChromaDB library.

    Attributes:
        collection (chromadb.Collection): The ChromaDB collection object.
        client (chromadb.Client): The ChromaDB client instance.
        similarity_metric_type (SimilarityMetricType): The metric for measuring similarity.
    """

    def __init__(
        self, similarity_metric_type: SimilarityMetricType = SimilarityMetricType.COSINE
    ):
        self.__next_embedding_id = 0
        self.collection = None
        self.client = None
        self.similarity_metric_type = similarity_metric_type
        self._operation_lock = threading.RLock()

    def add(self, embedding: List[float]) -> int:
        """Add an embedding to the database, initializing the collection if needed.

        This method is thread-safe.

        Args:
            embedding (List[float]): The embedding vector to add.

        Returns:
            int: The unique ID assigned to the added embedding.
        """
        with self._operation_lock:
            if self.collection is None:
                self._init_vector_store(len(embedding))

            # Atomic ID generation and assignment
            embedding_id = self.__next_embedding_id
            self.collection.add(embeddings=[embedding], ids=[str(embedding_id)])
            self.__next_embedding_id += 1

            return embedding_id

    def remove(self, embedding_id: int) -> int:
        """Remove an embedding from the database by its ID.

        This method is thread-safe.

        Args:
            embedding_id (int): The ID of the embedding to remove.

        Returns:
            int: The ID of the removed embedding.
        """
        with self._operation_lock:
            if self.collection is None:
                raise ValueError("Collection is not initialized")
            self.collection.delete(ids=[str(embedding_id)])
            return embedding_id

    def get_knn(self, embedding: List[float], k: int) -> List[tuple[float, int]]:
        """Find the k-nearest neighbors for a given embedding.

        This method is thread-safe. If `k` is larger than the number of items,
        it returns the maximum number of neighbors possible.

        Args:
            embedding (List[float]): The query embedding.
            k (int): The number of nearest neighbors to return.

        Returns:
            List[tuple[float, int]]: List of (similarity_score, embedding_id) tuples.
        """
        with self._operation_lock:
            if self.collection is None:
                # Initialize the store with the dimension of the query embedding
                self._init_vector_store(len(embedding))
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
        """Clear all embeddings from the collection."""
        with self._operation_lock:
            if self.collection is not None:
                self.collection.delete(ids=self.collection.get()["ids"])
            self.__next_embedding_id = 0

    def _init_vector_store(self, embedding_dim: int):
        """Initialize the ChromaDB client and collection.

        This method creates a ChromaDB client and a new collection with a unique
        name. It configures the collection's metadata to use the specified
        similarity metric ('cosine' or 'l2'). This method should be called
        within a locked context.

        Args:
            embedding_dim (int): The dimension of the embedding vectors.
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
            metadata={"hnsw:space": space},
            get_or_create=True,
        )

    def is_empty(self) -> bool:
        """Check if the collection contains any embeddings.

        This method is thread-safe.

        Returns:
            bool: True if the collection has no embeddings, False otherwise.
        """
        with self._operation_lock:
            if self.collection is None:
                return True
            return self.collection.count() == 0
