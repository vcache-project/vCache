import pickle
import sqlite3
import threading
from typing import List

from vcache.vcache_core.cache.embedding_store.embedding_metadata_storage.embedding_metadata_obj import (
    EmbeddingMetadataObj,
)
from vcache.vcache_core.cache.embedding_store.embedding_metadata_storage.embedding_metadata_storage import (
    EmbeddingMetadataStorage,
)


class SQLiteEmbeddingMetadataStorage(EmbeddingMetadataStorage):
    """
    SQLite-backed implementation of embedding metadata storage.

    Unlike `InMemoryEmbeddingMetadataStorage`, this implementation persists
    metadata to disk, so it survives process restarts. Each metadata object
    is stored as a pickled blob keyed by `embedding_id`, rather than mapped to
    individual columns, since `EmbeddingMetadataObj` mixes datetimes, optional
    floats, and tuples, and gains fields over time.
    """

    def __init__(self, db_path: str):
        """
        Initialize SQLite-backed embedding metadata storage.

        Args:
            db_path: Path to the SQLite database file. Created if it does
                not yet exist.
        """
        self.db_path = db_path
        self._lock = threading.Lock()
        self._connection = sqlite3.connect(db_path, check_same_thread=False)
        with self._lock:
            self._connection.execute(
                "CREATE TABLE IF NOT EXISTS embedding_metadata ("
                "embedding_id INTEGER PRIMARY KEY, data BLOB NOT NULL)"
            )
            self._connection.commit()

    def add_metadata(self, embedding_id: int, metadata: EmbeddingMetadataObj) -> int:
        """
        Add metadata for a specific embedding.

        Args:
            embedding_id: The id of the embedding to add the metadata for.
            metadata: The metadata to add to the embedding.

        Returns:
            The id of the embedding.
        """
        with self._lock:
            self._connection.execute(
                "INSERT OR REPLACE INTO embedding_metadata (embedding_id, data) "
                "VALUES (?, ?)",
                (embedding_id, pickle.dumps(metadata)),
            )
            self._connection.commit()
        return embedding_id

    def get_metadata(self, embedding_id: int) -> EmbeddingMetadataObj:
        """
        Get metadata for a specific embedding.

        Args:
            embedding_id: The id of the embedding to get the metadata for.

        Returns:
            The metadata of the embedding.

        Raises:
            ValueError: If embedding metadata is not found.
        """
        with self._lock:
            row = self._connection.execute(
                "SELECT data FROM embedding_metadata WHERE embedding_id = ?",
                (embedding_id,),
            ).fetchone()
        if row is None:
            raise ValueError(
                f"Embedding metadata for embedding id {embedding_id} not found"
            )
        return pickle.loads(row[0])

    def update_metadata(
        self, embedding_id: int, metadata: EmbeddingMetadataObj
    ) -> EmbeddingMetadataObj:
        """
        Update metadata for a specific embedding.

        Args:
            embedding_id: The id of the embedding to update the metadata for.
            metadata: The metadata to update the embedding with.

        Returns:
            The updated metadata of the embedding.

        Raises:
            ValueError: If embedding metadata is not found.
        """
        with self._lock:
            cursor = self._connection.execute(
                "UPDATE embedding_metadata SET data = ? WHERE embedding_id = ?",
                (pickle.dumps(metadata), embedding_id),
            )
            self._connection.commit()
            not_found = cursor.rowcount == 0
        if not_found:
            raise ValueError(
                f"Embedding metadata for embedding id {embedding_id} not found"
            )
        return metadata

    def remove_metadata(self, embedding_id: int) -> bool:
        """
        Remove metadata for a specific embedding.

        Args:
            embedding_id: The id of the embedding to remove metadata for.

        Returns:
            True if metadata was removed, False if not found.
        """
        with self._lock:
            cursor = self._connection.execute(
                "DELETE FROM embedding_metadata WHERE embedding_id = ?",
                (embedding_id,),
            )
            self._connection.commit()
            return cursor.rowcount > 0

    def flush(self) -> None:
        """
        Flush all metadata from storage.
        """
        with self._lock:
            self._connection.execute("DELETE FROM embedding_metadata")
            self._connection.commit()

    def get_all_embedding_metadata_objects(self) -> List[EmbeddingMetadataObj]:
        """
        Get all embedding metadata objects in storage.

        Returns:
            A list of all the embedding metadata objects in the storage.
        """
        with self._lock:
            rows = self._connection.execute(
                "SELECT data FROM embedding_metadata"
            ).fetchall()
        return [pickle.loads(row[0]) for row in rows]
