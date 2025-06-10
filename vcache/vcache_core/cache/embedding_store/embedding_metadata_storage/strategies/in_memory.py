import threading
from typing import Dict, List, Tuple

from vcache.vcache_core.cache.embedding_store.embedding_metadata_storage.embedding_metadata_obj import (
    EmbeddingMetadataObj,
)
from vcache.vcache_core.cache.embedding_store.embedding_metadata_storage.embedding_metadata_storage import (
    EmbeddingMetadataStorage,
)


class InMemoryEmbeddingMetadataStorage(EmbeddingMetadataStorage):
    def __init__(self):
        self.metadata_storage: Dict[int, EmbeddingMetadataObj] = {}
        self._store_lock = threading.RLock()
        self._entry_locks: Dict[int, threading.Lock] = {}

    def _get_entry_lock(self, embedding_id: int) -> threading.Lock:
        with self._store_lock:
            if embedding_id not in self._entry_locks:
                self._entry_locks[embedding_id] = threading.Lock()
            return self._entry_locks[embedding_id]

    def add_metadata(self, embedding_id: int, metadata: EmbeddingMetadataObj) -> None:
        with self._store_lock:
            self.metadata_storage[embedding_id] = metadata
        return embedding_id

    def get_metadata(self, embedding_id: int) -> EmbeddingMetadataObj:
        with self._store_lock:
            if embedding_id not in self.metadata_storage:
                raise ValueError(
                    f"Embedding metadata for embedding id {embedding_id} not found"
                )
            return self.metadata_storage[embedding_id]

    def update_metadata(
        self, embedding_id: int, metadata: EmbeddingMetadataObj
    ) -> EmbeddingMetadataObj:
        with self._store_lock:
            if embedding_id not in self.metadata_storage:
                raise ValueError(
                    f"Embedding metadata for embedding id {embedding_id} not found"
                )
            self.metadata_storage[embedding_id] = metadata
            return metadata

    def add_observation(
        self, embedding_id: int, observation: Tuple[float, int]
    ) -> None:
        entry_lock = self._get_entry_lock(embedding_id)
        with entry_lock:
            metadata = self.get_metadata(embedding_id)
            metadata.observations.append(observation)
            self.update_metadata(embedding_id, metadata)

    def remove_metadata(self, embedding_id: int) -> bool:
        with self._store_lock:
            if embedding_id in self.metadata_storage:
                del self.metadata_storage[embedding_id]
                # Also remove the associated lock
                if embedding_id in self._entry_locks:
                    del self._entry_locks[embedding_id]
                return True
        return False

    def flush(self) -> None:
        with self._store_lock:
            self.metadata_storage = {}
            self._entry_locks = {}

    def get_all_embedding_metadata_objects(self) -> List[EmbeddingMetadataObj]:
        with self._store_lock:
            return list(self.metadata_storage.values())
