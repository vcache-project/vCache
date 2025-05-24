from typing import List

from vcache.vcache_core.cache.embedding_store.embedding_metadata_storage import (
    EmbeddingMetadataStorage,
)
from vcache.vcache_core.cache.embedding_store.embedding_metadata_storage.embedding_metadata_obj import (
    EmbeddingMetadataObj,
)
from vcache.vcache_core.cache.embedding_store.vector_db.vector_db import VectorDB


class EmbeddingStore:
    def __init__(
        self,
        vector_db: VectorDB,
        embedding_metadata_storage: EmbeddingMetadataStorage,
    ):
        self.vector_db = vector_db
        self.embedding_metadata_storage = embedding_metadata_storage

    def add_embedding(self, embedding: List[float], response: str) -> int:
        embedding_id = self.vector_db.add(embedding)
        metadata = EmbeddingMetadataObj(
            embedding_id=embedding_id,
            response=response,
        )
        self.embedding_metadata_storage.add_metadata(
            embedding_id=embedding_id, metadata=metadata
        )
        return embedding_id

    def remove(self, embedding_id: int) -> int:
        self.embedding_metadata_storage.remove_metadata(embedding_id)
        return self.vector_db.remove(embedding_id)

    def get_knn(self, embedding: List[float], k: int) -> List[tuple[float, int]]:
        return self.vector_db.get_knn(embedding, k)

    def reset(self) -> None:
        self.embedding_metadata_storage.flush()
        return self.vector_db.reset()

    def calculate_storage_consumption(self) -> int:
        # TODO: Add metadata logic
        return -1

    def get_metadata(self, embedding_id: int) -> "EmbeddingMetadataObj":
        return self.embedding_metadata_storage.get_metadata(embedding_id)

    def update_metadata(
        self, embedding_id: int, metadata: "EmbeddingMetadataObj"
    ) -> "EmbeddingMetadataObj":
        return self.embedding_metadata_storage.update_metadata(embedding_id, metadata)

    def is_empty(self) -> bool:
        return self.vector_db.is_empty()
