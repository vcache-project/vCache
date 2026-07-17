import json
import os
from typing import List

import hnswlib

from vcache.vcache_core.cache.embedding_store.vector_db.vector_db import (
    SimilarityMetricType,
    VectorDB,
)

"""
Run 'sudo apt-get install build-essential' on Linux Debian/Ubuntu to install the build-essential package
"""


class PersistentHNSWLibVectorDB(VectorDB):
    """
    HNSWLib-based vector database implementation that persists its index to
    disk, so it survives process restarts.

    Persistence uses a checkpoint + write-ahead log (WAL) scheme rather than
    rewriting the full hnswlib index on every mutation:

    - Each `add`/`remove` is appended as one line to a WAL file and fsync'd.
      This is cheap (O(1) per call) regardless of index size.
    - Every `checkpoint_interval` mutations (or on an explicit `checkpoint()`
      call), the in-memory index is serialized to a new, uniquely-named
      generation file and a manifest is written pointing at it. Both the
      generation file and the manifest are written to a temporary path first
      and then atomically renamed into place via `os.replace`, so a crash
      mid-write can never leave a partially-written file visible under its
      final name.
    - The manifest is the single source of truth for "what generation is
      current" and carries all bookkeeping (dimension, space, id counters)
      that used to live in a separate sidecar file, so the index and its
      metadata can never desynchronize: either the manifest rename completed
      and both are visible together, or it didn't and both are still the
      previous, consistent generation.
    - On startup, the last checkpointed generation is loaded and then any WAL
      entries recorded after that checkpoint are replayed, recovering
      mutations that happened after the last checkpoint but before a crash.

    This is a standalone implementation (not a subclass of `HNSWLibVectorDB`)
    so it does not depend on, or risk altering, that class's internals.

    Note: this scheme makes on-disk state crash-safe for a single writer. It
    does not coordinate concurrent writers across multiple processes; running
    more than one process against the same `persist_path` concurrently is not
    supported.
    """

    def __init__(
        self,
        persist_path: str,
        similarity_metric_type: SimilarityMetricType = SimilarityMetricType.COSINE,
        max_capacity: int = 100000,
        checkpoint_interval: int = 100,
    ):
        """Initializes the persistent HNSWLib vector database.

        Args:
            persist_path (str): Base path used to derive the on-disk files:
                a manifest at `persist_path + ".manifest.json"`, a WAL at
                `persist_path + ".wal"`, and index generation snapshots at
                `persist_path + ".g<N>"`. If a manifest already exists, the
                database is restored from disk.
            similarity_metric_type (SimilarityMetricType): The similarity metric
                to use for comparisons.
            max_capacity (int): The maximum number of vectors the database can store.
            checkpoint_interval (int): Number of mutations to accumulate in the
                WAL before automatically checkpointing (serializing the full
                index and clearing the WAL). Lower values bound the amount of
                WAL replay work needed after a crash at the cost of more
                frequent full-index writes; higher values do the opposite.
        """
        self.persist_path = persist_path
        self._manifest_path = persist_path + ".manifest.json"
        self._wal_path = persist_path + ".wal"
        self.checkpoint_interval = checkpoint_interval

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

        self._generation = 0
        self._pending_wal_entries = 0

        if os.path.exists(self._manifest_path):
            self._load_from_disk()
        elif os.path.exists(self._wal_path):
            # No checkpoint has ever completed, but a WAL from a previous
            # (possibly crashed) instance exists: replay it from scratch.
            self._replay_wal()

    def add(self, embedding: List[float]) -> int:
        """Adds an embedding vector to the database and durably logs it.

        Args:
            embedding (List[float]): The embedding vector to add.

        Returns:
            int: The unique ID assigned to the added embedding.
        """
        if self.index is None:
            self._init_vector_store(len(embedding))
        id = self.__next_embedding_id
        self.index.add_items(embedding, id)
        self.embedding_count += 1
        self.__next_embedding_id += 1
        self._append_wal({"op": "add", "id": id, "embedding": list(embedding)})
        return id

    def remove(self, embedding_id: int) -> int:
        """Marks an embedding for deletion and durably logs the change.

        Note:
            HNSWLib does not physically remove data, but marks it as deleted.

        Args:
            embedding_id (int): The ID of the embedding to remove.

        Returns:
            int: The ID of the removed embedding.

        Raises:
            ValueError: If the index has not been initialized.
        """
        if self.index is None:
            raise ValueError("Index is not initialized")
        self.index.mark_deleted(embedding_id)
        self.embedding_count -= 1
        self._append_wal({"op": "remove", "id": embedding_id})
        return embedding_id

    def get_knn(self, embedding: List[float], k: int) -> List[tuple[float, int]]:
        """Gets k-nearest neighbors for a given embedding.

        Args:
            embedding (List[float]): The query embedding vector.
            k (int): The number of nearest neighbors to return.

        Returns:
            List[tuple[float, int]]: A list of tuples containing similarity
            scores and embedding IDs.
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
        return list(zip(similarity_scores, id_list))

    def reset(self) -> None:
        """Resets the vector database to an empty state and checkpoints it."""
        if self.dim is not None:
            self._init_vector_store(self.dim)
        self.embedding_count = 0
        self.__next_embedding_id = 0
        self.checkpoint()

    def checkpoint(self) -> None:
        """Forces a full, atomic checkpoint of the index and manifest.

        Serializes the current in-memory index to a new generation file,
        atomically publishes a manifest pointing at it, and then clears the
        WAL, since every mutation up to this point is now captured in the
        checkpoint itself. Both writes go through a temp-file-plus-rename so
        a crash mid-checkpoint leaves the previous, still-consistent
        generation visible.
        """
        previous_generation = self._generation
        self._generation += 1

        if self.index is not None:
            index_path = self._generation_path(self._generation)
            tmp_index_path = index_path + ".tmp"
            self.index.save_index(tmp_index_path)
            self._fsync_file(tmp_index_path)
            os.replace(tmp_index_path, index_path)

        self._write_manifest()
        self._truncate_wal()
        self._pending_wal_entries = 0

        if self.index is not None:
            old_index_path = self._generation_path(previous_generation)
            if previous_generation != self._generation and os.path.exists(
                old_index_path
            ):
                os.remove(old_index_path)

    def _init_vector_store(self, embedding_dim: int):
        """Initializes the HNSWLib index.

        Args:
            embedding_dim (int): The dimension of the embedding vectors.

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
        """Checks if the vector database is empty.

        Returns:
            bool: True if the database contains no embeddings, False otherwise.
        """
        return self.embedding_count == 0

    def size(self) -> int:
        """Gets the number of embeddings in the vector database.

        Returns:
            int: The number of embeddings in the vector database.
        """
        return self.embedding_count

    def _generation_path(self, generation: int) -> str:
        """Path of the index snapshot file for a given generation number."""
        return f"{self.persist_path}.g{generation}"

    def _append_wal(self, entry: dict) -> None:
        """Durably appends one mutation record to the WAL, checkpointing if
        the configured interval has been reached."""
        with open(self._wal_path, "a") as f:
            f.write(json.dumps(entry) + "\n")
            f.flush()
            os.fsync(f.fileno())
        self._pending_wal_entries += 1
        if self._pending_wal_entries >= self.checkpoint_interval:
            self.checkpoint()

    def _truncate_wal(self) -> None:
        """Atomically clears the WAL, e.g. after its entries are captured in
        a checkpoint."""
        tmp_wal_path = self._wal_path + ".tmp"
        open(tmp_wal_path, "w").close()
        os.replace(tmp_wal_path, self._wal_path)

    def _write_manifest(self) -> None:
        """Atomically publishes a manifest pointing at the current
        generation. This is the single commit point: once this rename
        completes, the new generation (and the bookkeeping alongside it) is
        the durable state; until then, the previous manifest is still
        authoritative."""
        manifest = {
            "generation": self._generation,
            "dim": self.dim,
            "space": self.space,
            "max_elements": self.max_elements,
            "ef_construction": self.ef_construction,
            "M": self.M,
            "ef": self.ef,
            "embedding_count": self.embedding_count,
            "next_embedding_id": self.__next_embedding_id,
        }
        tmp_manifest_path = self._manifest_path + ".tmp"
        with open(tmp_manifest_path, "w") as f:
            json.dump(manifest, f)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_manifest_path, self._manifest_path)

    def _fsync_file(self, path: str) -> None:
        """Flushes a file's contents to durable storage."""
        fd = os.open(path, os.O_RDWR)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)

    def _load_from_disk(self) -> None:
        """Restores state from the last manifest, then replays any WAL
        entries recorded after that checkpoint."""
        with open(self._manifest_path, "r") as f:
            manifest = json.load(f)
        self._generation = manifest["generation"]
        self.dim = manifest["dim"]
        self.space = manifest["space"]
        self.max_elements = manifest["max_elements"]
        self.ef_construction = manifest["ef_construction"]
        self.M = manifest["M"]
        self.ef = manifest["ef"]
        self.embedding_count = manifest["embedding_count"]
        self.__next_embedding_id = manifest["next_embedding_id"]

        index_path = self._generation_path(self._generation)
        if self.dim is not None and os.path.exists(index_path):
            self.index = hnswlib.Index(space=self.space, dim=self.dim)
            self.index.load_index(index_path, max_elements=self.max_elements)
            self.index.set_ef(self.ef)

        self._replay_wal()

    def _replay_wal(self) -> None:
        """Re-applies mutations recorded in the WAL since the last
        checkpoint, recovering work that a crash interrupted before it could
        be checkpointed."""
        if not os.path.exists(self._wal_path):
            return
        with open(self._wal_path, "r") as f:
            lines = [line for line in f.read().splitlines() if line]
        if not lines:
            return

        for line in lines:
            entry = json.loads(line)
            if entry["op"] == "add":
                if self.index is None:
                    self._init_vector_store(len(entry["embedding"]))
                self.index.add_items(entry["embedding"], entry["id"])
                self.embedding_count += 1
                self.__next_embedding_id = max(
                    self.__next_embedding_id, entry["id"] + 1
                )
            elif entry["op"] == "remove":
                self.index.mark_deleted(entry["id"])
                self.embedding_count -= 1

        # The replayed mutations are now reflected in-memory; fold them into
        # a fresh checkpoint so a subsequent restart doesn't need to replay
        # them again.
        self.checkpoint()
