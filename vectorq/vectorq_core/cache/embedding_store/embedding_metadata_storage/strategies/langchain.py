from typing import Any, Dict, List, Optional

from vectorq.vectorq_core.cache.embedding_store.embedding_metadata_storage.embedding_metadata_obj import (
    EmbeddingMetadataObj,
)
from vectorq.vectorq_core.cache.embedding_store.embedding_metadata_storage.embedding_metadata_storage import (
    EmbeddingMetadataStorage,
)


class LangchainMetadataStorage(EmbeddingMetadataStorage):
    def __init__(self):
        # TODO
        pass

    def add_metadata(
        self, embedding_id: int, metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        # TODO
        pass

    def get_metadata(self, embedding_id: int) -> Optional[Dict[str, Any]]:
        # TODO
        pass

    def update(
        self, embedding_id: int, metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        # TODO
        pass

    def remove_metadata(self, embedding_id: int) -> bool:
        # TODO
        pass

    def flush(self) -> None:
        # TODO
        pass

    def get_all_embedding_metadata_objects(self) -> List[EmbeddingMetadataObj]:
        # TODO
        pass
