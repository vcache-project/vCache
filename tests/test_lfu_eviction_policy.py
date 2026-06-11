"""
Unit tests for LFUEvictionPolicy.
Run with: pytest tests/test_lfu_eviction_policy.py -v
"""
import sys
import types
from datetime import datetime, timezone, timedelta

for _n in ["vllm", "vllm.config", "vllm.entrypoints", "vllm.entrypoints.llm",
           "vllm.platforms", "vllm.platforms.cuda", "vllm.utils"]:
    sys.modules.setdefault(_n, types.ModuleType(_n))

import pytest
from vcache.vcache_core.cache.embedding_store.embedding_metadata_storage.embedding_metadata_obj import EmbeddingMetadataObj
from vcache.vcache_core.cache.eviction_policy.lfu_eviction_policy import LFUEvictionPolicy


def make_meta(embedding_id, access_count=0, last_accessed=None):
    m = EmbeddingMetadataObj(embedding_id=embedding_id, response="r")
    m.access_count = access_count
    m.last_accessed = last_accessed or datetime.min.replace(tzinfo=timezone.utc)
    return m


class TestConstruction:
    def test_custom_max_size(self):
        p = LFUEvictionPolicy(max_size=500)
        assert p.max_size == 500

    def test_default_watermark(self):
        p = LFUEvictionPolicy(max_size=100)
        assert p.watermark == 0.95

    def test_custom_params(self):
        p = LFUEvictionPolicy(max_size=500, watermark=0.8, eviction_percentage=0.2)
        assert p.eviction_percentage == 0.2

    def test_str_contains_class_name(self):
        assert "LFUEvictionPolicy" in str(LFUEvictionPolicy(max_size=10))


class TestUpdateEvictionMetadata:
    def test_first_access_sets_count_to_1(self):
        p = LFUEvictionPolicy(max_size=100)
        m = EmbeddingMetadataObj(embedding_id=1, response="r")
        p.update_eviction_metadata(m)
        assert m.access_count == 1

    def test_repeated_access_increments(self):
        p = LFUEvictionPolicy(max_size=100)
        m = EmbeddingMetadataObj(embedding_id=1, response="r")
        for _ in range(5):
            p.update_eviction_metadata(m)
        assert m.access_count == 5

    def test_updates_last_accessed(self):
        p = LFUEvictionPolicy(max_size=100)
        m = EmbeddingMetadataObj(embedding_id=1, response="r")
        before = datetime.now(timezone.utc)
        p.update_eviction_metadata(m)
        assert m.last_accessed >= before

    def test_access_count_starts_from_zero(self):
        p = LFUEvictionPolicy(max_size=100)
        m = EmbeddingMetadataObj(embedding_id=1, response="r")
        p.update_eviction_metadata(m)
        p.update_eviction_metadata(m)
        assert m.access_count == 2


class TestSelectVictims:
    def test_empty_returns_empty(self):
        p = LFUEvictionPolicy(max_size=100)
        assert p.select_victims([]) == []

    def test_evicts_least_frequent(self):
        p = LFUEvictionPolicy(max_size=100, eviction_percentage=0.1)
        metadata = [make_meta(i, access_count=i) for i in range(20)]
        victims = p.select_victims(metadata)
        assert set(victims) == set(range(10))

    def test_lru_tiebreak(self):
        p = LFUEvictionPolicy(max_size=10, eviction_percentage=0.1)
        now = datetime.now(timezone.utc)
        old = make_meta(1, access_count=1, last_accessed=now - timedelta(hours=2))
        new = make_meta(2, access_count=1, last_accessed=now)
        victims = p.select_victims([old, new])
        assert 1 in victims

    def test_never_accessed_evicted_first(self):
        p = LFUEvictionPolicy(max_size=10, eviction_percentage=0.1)
        cold = make_meta(1, access_count=0)
        hot  = make_meta(2, access_count=99)
        victims = p.select_victims([cold, hot])
        assert 1 in victims
        assert 2 not in victims

    def test_high_frequency_survives(self):
        p = LFUEvictionPolicy(max_size=100, eviction_percentage=0.1)
        popular = make_meta(999, access_count=1000)
        cold = [make_meta(i, access_count=0) for i in range(20)]
        victims = p.select_victims(cold + [popular])
        assert 999 not in victims

    def test_num_victims_respects_eviction_percentage(self):
        p = LFUEvictionPolicy(max_size=100, eviction_percentage=0.2)
        metadata = [make_meta(i) for i in range(50)]
        victims = p.select_victims(metadata)
        assert len(victims) == 20

    def test_returns_embedding_ids(self):
        p = LFUEvictionPolicy(max_size=10, eviction_percentage=0.1)
        metadata = [make_meta(i) for i in range(10)]
        victims = p.select_victims(metadata)
        assert all(isinstance(v, int) for v in victims)

    def test_update_then_select_keeps_hot_entry(self):
        p = LFUEvictionPolicy(max_size=100, eviction_percentage=0.1)
        hot = EmbeddingMetadataObj(embedding_id=99, response="r")
        for _ in range(50):
            p.update_eviction_metadata(hot)
        cold = [make_meta(i, access_count=0) for i in range(20)]
        victims = p.select_victims(cold + [hot])
        assert 99 not in victims
