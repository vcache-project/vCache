import json
import shutil
import tempfile
import unittest

import pandas as pd

from benchmarks.benchmark import Benchmark
from vcache import VCache, VCacheConfig, VerifiedDecisionPolicy
from vcache.inference_engine.strategies.benchmark import BenchmarkInferenceEngine
from vcache.vcache_core.cache.embedding_engine.strategies.benchmark import (
    BenchmarkEmbeddingEngine,
)
from vcache.vcache_core.cache.embedding_store.embedding_metadata_storage import (
    InMemoryEmbeddingMetadataStorage,
)
from vcache.vcache_core.cache.embedding_store.vector_db import (
    HNSWLibVectorDB,
    SimilarityMetricType,
)
from vcache.vcache_core.cache.eviction_policy.strategies.mru import MRUEvictionPolicy
from vcache.vcache_core.similarity_evaluator.strategies.benchmark_comparison import (
    BenchmarkComparisonSimilarityEvaluator,
)

# Column-name prefixes used to key the synthetic pre-computed dataset rows
# consumed by `Benchmark.run_benchmark_loop`. These don't need to match any
# real EmbeddingModel/LargeLanguageModel enum value.
EMBEDDING_MODEL = ("emb", "TestEmbeddingModel", "float32", 8)
LLM_MODEL = ("resp", "TestLLM", "float16", None)
NUM_SAMPLES = 10


def _build_synthetic_data_entries():
    """Builds a small, fully offline, pre-computed dataset for the benchmark loop.

    Each entry supplies its own embedding and label response, so the run
    never needs network access or an API key (mirrors how
    `BenchmarkInferenceEngine`/`BenchmarkEmbeddingEngine` are used for the
    non-custom-dataset baselines in `benchmarks/benchmark.py`).
    """
    entries = []
    for i in range(NUM_SAMPLES):
        embedding = [0.0] * 8
        embedding[i % 8] = 1.0
        embedding[(i + 1) % 8] = 0.5
        entries.append(
            {
                "prompt": f"synthetic prompt {i}",
                f"{EMBEDDING_MODEL[0]}_lat": 0.01,
                f"{LLM_MODEL[0]}_lat": 0.02,
                LLM_MODEL[0]: f"synthetic response {i}",
                EMBEDDING_MODEL[0]: embedding,
                "id_set": i % 3,
            }
        )
    return entries


class TestBenchmarkSmoke(unittest.TestCase):
    """Fully offline smoke test for the benchmark suite's metrics pipeline.

    Runs a tiny, deterministic benchmark (no network, no API key) through the
    same code path production runs use, to catch regressions in the
    resource/throughput instrumentation on every commit without paying for a
    full-scale, live benchmark run.
    """

    def setUp(self):
        self.output_dir = tempfile.mkdtemp()

        eviction_policy = MRUEvictionPolicy(
            max_size=1000, watermark=0.99, eviction_percentage=0.1
        )
        vcache_config = VCacheConfig(
            inference_engine=BenchmarkInferenceEngine(),
            embedding_engine=BenchmarkEmbeddingEngine(),
            vector_db=HNSWLibVectorDB(
                similarity_metric_type=SimilarityMetricType.COSINE,
                max_capacity=1000,
            ),
            embedding_metadata_storage=InMemoryEmbeddingMetadataStorage(),
            similarity_evaluator=BenchmarkComparisonSimilarityEvaluator(),
            eviction_policy=eviction_policy,
        )
        self.eviction_policy = eviction_policy
        vcache = VCache(vcache_config, VerifiedDecisionPolicy(delta=0.5))

        self.benchmark = Benchmark(vcache)
        self.benchmark.filepath = "synthetic"
        self.benchmark.embedding_model = EMBEDDING_MODEL
        self.benchmark.llm_model = LLM_MODEL
        self.benchmark.timestamp = "smoketest"
        self.benchmark.threshold = None
        self.benchmark.delta = 0.5
        self.benchmark.is_static_threshold = False
        self.benchmark.output_folder_path = self.output_dir
        self.benchmark.eviction_policy = eviction_policy
        self.benchmark.stats_set_up()

    def tearDown(self):
        self.eviction_policy.shutdown()
        shutil.rmtree(self.output_dir, ignore_errors=True)

    def test_benchmark_run_populates_resource_and_throughput_metrics(self):
        data_entries = _build_synthetic_data_entries()

        self.benchmark.run_benchmark_loop(data_entries, max_samples=NUM_SAMPLES)

        self.assertEqual(len(self.benchmark.cache_hit_list), NUM_SAMPLES)
        self.assertEqual(len(self.benchmark.cpu_percent_list), NUM_SAMPLES)
        self.assertEqual(len(self.benchmark.memory_mb_list), NUM_SAMPLES)
        self.assertEqual(len(self.benchmark.gpu_util_list), NUM_SAMPLES)
        for gpu_util in self.benchmark.gpu_util_list:
            self.assertTrue(gpu_util is None or 0.0 <= gpu_util <= 100.0)

        self.assertIsNotNone(self.benchmark.elapsed_time_sec)
        self.assertGreater(self.benchmark.elapsed_time_sec, 0.0)

        self.benchmark.dump_results_to_json()
        self.benchmark.dump_results_to_csv()

        json_path = f"{self.output_dir}/results_smoketest.json"
        with open(json_path) as f:
            data = json.load(f)

        self.assertEqual(len(data["cpu_percent_list"]), NUM_SAMPLES)
        self.assertEqual(len(data["memory_mb_list"]), NUM_SAMPLES)
        self.assertEqual(len(data["gpu_util_list"]), NUM_SAMPLES)
        self.assertIsInstance(data["peak_memory_mb"], float)
        self.assertGreater(data["throughput_qps"], 0.0)
        self.assertGreaterEqual(data["throughput_tps"], 0.0)

        csv_path = f"{self.output_dir}/results_smoketest.csv"
        df = pd.read_csv(csv_path)
        self.assertEqual(len(df), NUM_SAMPLES)
        for column in ("cpu_percent", "memory_mb", "gpu_util_percent"):
            self.assertIn(column, df.columns)


if __name__ == "__main__":
    unittest.main()
