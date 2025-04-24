from typing import TYPE_CHECKING, List

from vectorq.vectorq_core.cache.embedding_store.embedding_metadata_storage import (
    EmbeddingMetadataStorage,
)
from vectorq.vectorq_core.cache.embedding_store.embedding_metadata_storage.embedding_metadata_obj import (
    EmbeddingMetadataObj,
)
from vectorq.vectorq_core.cache.embedding_store.vector_db.vector_db import VectorDB

if TYPE_CHECKING:
    from vectorq.config import VectorQConfig


class EmbeddingStore:
    def __init__(self, vectorq_config: "VectorQConfig"):
        self.vector_db: "VectorDB" = vectorq_config.vector_db
        self.embedding_metadata_storage: "EmbeddingMetadataStorage" = (
            vectorq_config.embedding_metadata_storage
        )

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
        return self.embedding_metadata_storage.update(embedding_id, metadata)

    def is_empty(self) -> bool:
        return self.vector_db.is_empty()
