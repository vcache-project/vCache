from typing import Any, Dict, List, Optional

from vcache.vcache_core.cache.embedding_store.embedding_metadata_storage.embedding_metadata_obj import (
    EmbeddingMetadataObj,
)
from vcache.vcache_core.cache.embedding_store.embedding_metadata_storage.embedding_metadata_storage import (
    EmbeddingMetadataStorage,
)


class InMemoryEmbeddingMetadataStorage(EmbeddingMetadataStorage):
    def __init__(self):
        self.metadata_storage: Dict[int, "EmbeddingMetadataObj"] = {}

    def add_metadata(
        self, embedding_id: int, metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        self.metadata_storage[embedding_id] = metadata
        return embedding_id

    def get_metadata(self, embedding_id: int) -> Optional[Dict[str, Any]]:
        if embedding_id not in self.metadata_storage:
            raise ValueError(
                f"Embedding metadata for embedding id {embedding_id} not found"
            )
        else:
            return self.metadata_storage[embedding_id]

    def update_metadata(
        self, embedding_id: int, metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        if embedding_id not in self.metadata_storage:
            raise ValueError(
                f"Embedding metadata for embedding id {embedding_id} not found"
            )
        else:
            self.metadata_storage[embedding_id] = metadata
            return metadata

    def remove_metadata(self, embedding_id: int) -> bool:
        if embedding_id in self.metadata_storage:
            del self.metadata_storage[embedding_id]
            return True
        return False

    def flush(self) -> None:
        self.metadata_storage = {}

    def get_all_embedding_metadata_objects(self) -> List[EmbeddingMetadataObj]:
        return list(self.metadata_storage.values())
