"""

This script is designed to benchmark the performance of vCache against several baselines.
It evaluates cache hit rates, accuracy, latency, and other metrics across different configurations.

The primary configuration is done by modifying the global variables in the Benchmark Config section:

1.  `RUN_COMBINATIONS`: This is the most important setting. It's a list of tuples, where each tuple
    defines a complete benchmark scenario to run. Each tuple contains:
    - `EmbeddingModel`: The embedding model to use (e.g., `EmbeddingModel.GTE`).
    - `LargeLanguageModel`: The large language model to use (e.g., `LargeLanguageModel.GPT_4O_MINI`).
    - `Dataset`: The dataset for the benchmark. The string values correspond to Hugging Face dataset
        repository IDs (e.g., 'vCache/SemBenchmarkSearchQueries'). These datasets will be automatically
        downloaded and cached by the `datasets` library on the first run.
    - `GeneratePlotsOnly`: Set to `GeneratePlotsOnly.YES` to skip the benchmark and only regenerate
        plots from existing results.
    - `SimilarityEvaluator`: The strategy for comparing semantic similarity (e.g., `StringComparisonSimilarityEvaluator`,
        `BenchmarkComparisonSimilarityEvaluator`).
    - `EvictionPolicy`: The cache eviction policy to use (e.g., `SCUEvictionPolicy`).
    - `int`: The maximum number of samples to process

2.  `BASELINES_TO_RUN`: A list to specify which caching strategies to evaluate. Every baseline is run
    for every run combination. Comment out or remove baselines you don't want to run. Available baselines
    include `VCacheLocal`, `GPTCache`, `BerkeleyEmbedding`, etc.

3.  `STATIC_THRESHOLDS`: A list of floating-point values for the similarity thresholds used by static policies
    like GPTCache and BerkeleyEmbedding. The benchmark will run once for each threshold in this list.

4.  `DELTAS`: A list of floating-point values for the `delta` parameter used by dynamic policies
    like vCache. The benchmark will run once for each delta in this list.

Additional configuration variables:

5.  `CONFIDENCE_INTERVALS_ITERATIONS`: Number of iterations to run each configuration for calculating
    confidence intervals in statistical analysis.

6.  `DISABLE_PROGRESS_BAR`: Set to `True` to disable the progress bar during benchmark execution.

7.  `KEEP_SPLIT`: Determines how many samples to keep from the dataset for evaluation. This controls
    the size of the test set used in the benchmark.

8.  `MAX_VECTOR_DB_CAPACITY`: Maximum capacity for the vector database.

9.  `PLOT_FONT_SIZE`: Font size used in generated plots and visualizations.

"""

import json
import logging
import os
import time
import unittest
from datetime import datetime
from enum import Enum
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from datasets import load_dataset
from tqdm import tqdm

from benchmarks._plotter_combined import generate_combined_plots
from benchmarks._plotter_individual import generate_individual_plots
from benchmarks.common.comparison import (
    answers_have_same_meaning_llm,
    answers_have_same_meaning_static,
)
from vcache.config import VCacheConfig
from vcache.inference_engine.strategies.benchmark import (
    BenchmarkInferenceEngine,
)
from vcache.inference_engine.strategies.open_ai import OpenAIInferenceEngine
from vcache.main import VCache
from vcache.vcache_core.cache.embedding_engine.strategies.benchmark import (
    BenchmarkEmbeddingEngine,
)
from vcache.vcache_core.cache.embedding_engine.strategies.open_ai import (
    OpenAIEmbeddingEngine,
)
from vcache.vcache_core.cache.embedding_store.embedding_metadata_storage import (
    InMemoryEmbeddingMetadataStorage,
)
from vcache.vcache_core.cache.embedding_store.embedding_metadata_storage.embedding_metadata_obj import (
    EmbeddingMetadataObj,
)
from vcache.vcache_core.cache.embedding_store.vector_db import (
    HNSWLibVectorDB,
    SimilarityMetricType,
)
from vcache.vcache_core.cache.eviction_policy.eviction_policy import EvictionPolicy
from vcache.vcache_core.cache.eviction_policy.strategies.scu import SCUEvictionPolicy
from vcache.vcache_core.similarity_evaluator import SimilarityEvaluator
from vcache.vcache_core.similarity_evaluator.strategies.benchmark_comparison import (
    BenchmarkComparisonSimilarityEvaluator,
)
from vcache.vcache_core.similarity_evaluator.strategies.string_comparison import (
    StringComparisonSimilarityEvaluator,
)
from vcache.vcache_policy.strategies.benchmark_iid_verified import (
    BenchmarkVerifiedIIDDecisionPolicy,
)
from vcache.vcache_policy.strategies.benchmark_sigmoid_only import (
    SigmoidOnlyDecisionPolicy,
)
from vcache.vcache_policy.strategies.benchmark_sigmoid_probability import (
    SigmoidProbabilityDecisionPolicy,
)
from vcache.vcache_policy.strategies.benchmark_static import (
    BenchmarkStaticDecisionPolicy,
)
from vcache.vcache_policy.strategies.benchmark_verified_global import (
    BenchmarkVerifiedGlobalDecisionPolicy,
)
from vcache.vcache_policy.strategies.verified import (
    VerifiedDecisionPolicy,
)
from vcache.vcache_policy.vcache_policy import VCachePolicy

repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
results_dir = os.path.join(repo_root, "benchmarks", "results")
if not os.path.exists(results_dir):
    os.makedirs(results_dir)
logging.basicConfig(
    filename=os.path.join(results_dir, "benchmark.log"),
    level=logging.INFO,
    format="%(asctime)s %(levelname)s:%(message)s",
)

########################################################################################################################
### Available Classes ##################################################################################################
########################################################################################################################


class EmbeddingModel(Enum):
    """Enumeration of available embedding models for benchmarking.

    Each enum value contains a tuple with:
    - Column name prefix in datasets
    - Model display name
    - Data type (float32/float16)
    - Embedding dimension
    """

    GTE = ("emb_gte", "GteLargeENv1_5", "float32", 1024)
    GTE_FT = ("emb_gte_ft", "GteLargeENv1_5", "float32", 1024)
    E5_MISTRAL_7B = ("emb_e5_mistral_7b", "E5_Mistral_7B_Instruct", "float16", 4096)
    E5_LARGE_V2 = ("emb_e5_large_v2", "E5_Large_v2", "float16", 512)
    E5_LARGE_V2_FT = ("emb_e5_large_v2_ft", "E5_Large_v2", "float16", 512)
    OPENAI_TEXT_EMBEDDING_SMALL = (
        "emb_openai_text_embedding_small",
        "text-embedding-3-small",
        "float16",
        1536,
    )


class LargeLanguageModel(Enum):
    """Enumeration of available large language models for benchmarking.

    Each enum value contains a tuple with:
    - Column name prefix in datasets
    - Model display name
    - Data type (float16)
    - Context length (None for variable)
    """

    LLAMA_3_8B = ("response_llama_3_8b", "Llama_3_8B_Instruct", "float16", None)
    LLAMA_3_70B = ("response_llama_3_70b", "Llama_3_70B_Instruct", "float16", None)
    GPT_4O_MINI = ("response_gpt-4o-mini", "GPT-4o-mini", "float16", None)
    GPT_4O_NANO = ("response_gpt-4.1-nano", "GPT-4.1-nano", "float16", None)
    GPT_4_1 = ("response_gpt-4.1", "gpt-4.1-2025-04-14", "float16", None)


class Baseline(Enum):
    """Enumeration of available caching baselines for comparison.

    Each baseline represents a different caching strategy:
    - GPTCache: Static threshold-based caching
    - VCacheLocal: vCache with local threshold adaptation
    - VCacheGlobal: vCache with global threshold adaptation
    - BerkeleyEmbedding: Fine-tuned embeddings with static threshold
    - VCacheBerkeleyEmbedding: vCache with fine-tuned embeddings
    - IID: Independent and Identically Distributed threshold policy
    - SigmoidProbability: Sigmoid probability-based threshold policy
    - SigmoidOnly: Sigmoid only-based threshold policy
    """

    GPTCache = "GPTCache"
    VCacheLocal = "vCacheLocal"
    VCacheGlobal = "vCacheGlobal"
    BerkeleyEmbedding = "BerkeleyEmbedding"
    VCacheBerkeleyEmbedding = "VCacheBerkeleyEmbedding"
    IID = "iid"
    SigmoidProbability = "SigmoidProbability"
    SigmoidOnly = "SigmoidOnly"


class Dataset(Enum):
    """Enumeration of available datasets for benchmarking.

    Supports both HuggingFace datasets (with repository IDs) and custom datasets
    (with relative paths from benchmarks/your_datasets/).
    """

    SEM_BENCHMARK_CLASSIFICATION = "vCache/SemBenchmarkClassification"
    SEM_BENCHMARK_ARENA = "vCache/SemBenchmarkLmArena"
    SEM_BENCHMARK_SEARCH_QUERIES = "vCache/SemBenchmarkSearchQueries"
    # Example for custom dataset. The path is relative to 'benchmarks/your_datasets/'
    CUSTOM_EXAMPLE = "your_datasets/your_custom_dataset.parquet"


class GeneratePlotsOnly(Enum):
    """Enumeration for controlling whether to run benchmarks or only generate plots.

    YES: Skip benchmark execution and only generate plots from existing results
    NO: Run full benchmark and generate plots
    """

    YES = True
    NO = False


########################################################################################################################
### Benchmark Config ###################################################################################################
########################################################################################################################

CONFIDENCE_INTERVALS_ITERATIONS: int = 1
DISABLE_PROGRESS_BAR: bool = False
KEEP_SPLIT: int = 100
MAX_VECTOR_DB_CAPACITY: int = 150000
PLOT_FONT_SIZE: int = 50

RUN_COMBINATIONS: List[
    Tuple[
        EmbeddingModel,
        LargeLanguageModel,
        Dataset,
        GeneratePlotsOnly,
        SimilarityEvaluator,
        EvictionPolicy,
        int,
    ]
] = [
    (
        EmbeddingModel.E5_LARGE_V2,
        LargeLanguageModel.GPT_4O_MINI,
        Dataset.SEM_BENCHMARK_ARENA,
        GeneratePlotsOnly.YES,
        BenchmarkComparisonSimilarityEvaluator(),
        SCUEvictionPolicy(max_size=100000, watermark=0.99, eviction_percentage=0.1),
        60000,
    ),
    (
        EmbeddingModel.GTE,
        LargeLanguageModel.LLAMA_3_8B,
        Dataset.SEM_BENCHMARK_CLASSIFICATION,
        GeneratePlotsOnly.YES,
        StringComparisonSimilarityEvaluator(),
        SCUEvictionPolicy(max_size=100000, watermark=0.99, eviction_percentage=0.1),
        45000,
    ),
]

BASELINES_TO_RUN: List[Baseline] = [
    Baseline.SigmoidProbability,
    Baseline.SigmoidOnly,
]

STATIC_THRESHOLDS: List[float] = [0.80, 0.83, 0.86, 0.89, 0.92, 0.95, 0.97, 0.98, 0.99]

DELTAS: List[float] = [0.01, 0.015, 0.02, 0.025, 0.03, 0.035, 0.04, 0.05, 0.06, 0.07]


########################################################################################################################
### Benchmark Class ####################################################################################################
########################################################################################################################
class Benchmark(unittest.TestCase):
    """Main benchmark class for evaluating vCache performance against baselines.

    This class extends unittest.TestCase to leverage testing infrastructure while
    providing comprehensive benchmarking capabilities. It handles dataset loading,
    cache evaluation, statistics collection, and result serialization.

    The benchmark evaluates caching strategies by comparing cache hits/misses,
    accuracy (true/false positives/negatives), and latency between direct inference
    and cached inference across different datasets and model configurations.

    Attributes:
        vcache: The vCache instance being benchmarked
        embedding_model: Tuple containing embedding model configuration
        llm_model: Tuple containing LLM model configuration
        filepath: Path to the dataset file
        output_folder_path: Directory for saving results
        timestamp: Timestamp for result file naming
        threshold: Static threshold value (for static policies)
        delta: Delta parameter (for dynamic policies)
        is_static_threshold: Whether using static or dynamic threshold
        eviction_policy: Cache eviction policy instance
        is_custom_dataset: Whether using custom dataset format
    """

    def __init__(self, vcache: VCache):
        super().__init__()
        self.vcache: VCache = vcache
        self.embedding_model: Tuple[str, str, str, int] = None
        self.llm_model: Tuple[str, str, str, int] = None
        self.filepath: str = None
        self.output_folder_path: str = None
        self.timestamp: str = None
        self.threshold: float = None
        self.delta: float = None
        self.is_static_threshold: bool = None
        self.eviction_policy: EvictionPolicy = None
        self.is_custom_dataset: bool = False

    def stats_set_up(self):
        """Initialize statistics tracking lists and create output directory.

        Sets up all the necessary data structures for tracking benchmark metrics
        including cache hits/misses, true/false positives/negatives, latency
        measurements, and advanced statistics from the caching policy.

        Note:
            This method must be called before running the benchmark to ensure
            proper statistics collection.
        """
        self.cache_hit_list: List[int] = []
        self.cache_miss_list: List[int] = []
        self.tp_list: List[int] = []
        self.fp_list: List[int] = []
        self.tn_list: List[int] = []
        self.fn_list: List[int] = []
        self.latency_direct_list: List[float] = []
        self.latency_vcache_list: List[float] = []
        self.observations_dict: Dict[str, Dict[str, float]] = {}
        self.gammas_dict: Dict[str, float] = {}
        self.t_hats_dict: Dict[str, float] = {}
        self.t_primes_dict: Dict[str, float] = {}
        self.var_ts_dict: Dict[str, float] = {}

        if self.output_folder_path and not os.path.exists(self.output_folder_path):
            os.makedirs(self.output_folder_path)

    def run_benchmark_loop_custom(self, data_entries: List[Dict], max_samples: int):
        """Run benchmark loop for custom datasets with live inference.

        This method processes custom datasets that only contain prompts, requiring
        live inference calls to both the embedding and language models. It compares
        direct inference (no cache) against vCache inference for each prompt.

        Args:
            data_entries: List of dictionaries containing dataset entries. Each entry
                must have a 'prompt' key.
            max_samples: Maximum number of samples to process from the dataset.

        Note:
            Custom datasets use live model calls, so this method will make actual
            API calls to embedding and inference engines. Ensure proper API keys
            and rate limits are configured.
        """
        logging.info("Running benchmark loop for custom dataset")
        pbar = tqdm(
            total=min(max_samples, len(data_entries)),
            desc="Processing entries",
            disable=DISABLE_PROGRESS_BAR,
        )

        for idx, data_entry in enumerate(data_entries):
            if idx >= max_samples:
                break

            # 1) Get Data
            prompt: str = data_entry["prompt"]

            # 2.1) Direct Inference (No Cache) - Live call
            start_time = time.time()

            label_response = self.vcache.vcache_config.inference_engine.create(prompt)
            latency_direct = time.time() - start_time

            # 2.2) vCache Inference (With Cache)
            (
                is_cache_hit,
                cache_response,
                response_metadata,
                nn_metadata,
                latency_vcache,
            ) = self.get_vcache_answer_custom(prompt=prompt)

            # This is important for the async logic
            time.sleep(0.002)

            # 3) Update Stats
            self.update_stats(
                is_cache_hit=is_cache_hit,
                label_response=label_response,
                cache_response=cache_response,
                label_id_set=-1,  # Custom datasets don't have id_set
                response_metadata=response_metadata,
                nn_metadata=nn_metadata,
                latency_direct=latency_direct,
                latency_vcache=latency_vcache,
            )

            pbar.update(1)

        pbar.close()

    def run_benchmark_loop(self, data_entries: List[Dict], max_samples: int):
        """Run benchmark loop for pre-computed datasets from HuggingFace.

        This method processes datasets that contain pre-computed embeddings and
        responses, avoiding the need for live model calls. It extracts embeddings
        and responses from the dataset and uses them to simulate the caching
        behavior.

        Args:
            data_entries: List of dictionaries containing dataset entries. Each entry
                must contain prompt, pre-computed embeddings, responses, and latency
                measurements.
            max_samples: Maximum number of samples to process from the dataset.

        Note:
            This method uses pre-computed embeddings and responses from the dataset,
            making it much faster than the custom dataset approach but limiting it
            to specific model combinations available in the dataset.
        """
        logging.info("Running benchmark loop")
        pbar = tqdm(
            total=min(max_samples, len(data_entries)),
            desc="Processing entries",
            disable=DISABLE_PROGRESS_BAR,
        )
        logging.info(f"data_entries: {data_entries}")

        for idx, data_entry in enumerate(data_entries):
            if idx >= max_samples:
                break

            # 1) Get Data
            prompt: str = data_entry["prompt"]
            system_prompt: str = data_entry.get("output_format", "")

            emb_generation_latency: float = float(
                data_entry[self.embedding_model[0] + "_lat"]
            )

            llm_generation_latency: float = float(
                data_entry[self.llm_model[0] + "_lat"]
            )

            # 2.1) Direct Inference (No Cache)
            label_response: str = data_entry[self.llm_model[0]]
            latency_direct: float = llm_generation_latency

            # 2.2) vCache Inference (With Cache)
            candidate_embedding: List[float] = data_entry[self.embedding_model[0]]

            label_id_set: int = data_entry.get("id_set", -1)
            if label_id_set == -1:
                label_id_set: int = data_entry.get("ID_Set", -1)

            (
                is_cache_hit,
                cache_response,
                response_metadata,
                nn_metadata,
                latency_vcache_logic,
            ) = self.get_vcache_answer(
                prompt=prompt,
                candidate_embedding=candidate_embedding,
                label_response=label_response,
                system_prompt=system_prompt,
                id_set=label_id_set,
            )
            latency_vcache: float = latency_vcache_logic + emb_generation_latency
            if not is_cache_hit:
                latency_vcache += llm_generation_latency

            # This is important for the async logic
            time.sleep(0.002)

            # 3) Update Stats
            self.update_stats(
                is_cache_hit=is_cache_hit,
                label_response=label_response,
                cache_response=cache_response,
                label_id_set=label_id_set,
                response_metadata=response_metadata,
                nn_metadata=nn_metadata,
                latency_direct=latency_direct,
                latency_vcache=latency_vcache,
            )

            pbar.update(1)

        pbar.close()

    def test_run_benchmark(self, max_samples):
        """Main benchmark execution method that loads data and runs evaluation.

        This method serves as the main entry point for benchmark execution. It
        determines whether to use custom or pre-computed datasets, loads the
        appropriate data, runs the benchmark loop, and generates results.

        Args:
            max_samples: Maximum number of samples to process from the dataset.

        Raises:
            ValueError: If required parameters (filepath, embedding_model, llm_model)
                are not set.
            FileNotFoundError: If the specified dataset file cannot be found.
            Exception: For any other errors during benchmark execution.

        Note:
            Results are automatically saved to JSON and plots are generated upon
            successful completion.
        """
        if not self.filepath or not self.embedding_model or not self.llm_model:
            raise ValueError(
                f"Required parameters not set: filepath: {self.filepath}, embedding_model: {self.embedding_model}, or llm_model: {self.llm_model}"
            )

        try:
            if self.is_custom_dataset:
                logging.info(f"Loading custom dataset: {self.filepath}")
                if self.filepath.endswith(".csv"):
                    df = pd.read_csv(self.filepath)
                elif self.filepath.endswith(".parquet"):
                    df = pd.read_parquet(self.filepath)
                else:
                    raise ValueError(
                        f"Unsupported file format (not .csv or .parquet) for custom dataset: {self.filepath}"
                    )
                data_iterator = df.to_dict("records")
                self.run_benchmark_loop_custom(data_iterator, max_samples)
            elif "/" in self.filepath:
                logging.info(f"Loading Hugging Face dataset: {self.filepath}")
                data_iterator = load_dataset(
                    self.filepath, split=f"train[:{max_samples}]"
                )
                self.run_benchmark_loop(data_iterator, max_samples)

        except FileNotFoundError as e:
            logging.error(f"Benchmark dataset file not found: {e}")
            return
        except Exception as e:
            logging.error(f"Error processing benchmark: {e}")
            return

        self.dump_results_to_json()
        generate_individual_plots(
            self,
            font_size=PLOT_FONT_SIZE,
            is_static=self.is_static_threshold,
            parameter=self.threshold if self.is_static_threshold else self.delta,
        )

    ########################################################################################################################
    ### Class Helper Functions #############################################################################################
    ########################################################################################################################
    def update_stats(
        self,
        is_cache_hit: bool,
        label_response: str,
        cache_response: str,
        label_id_set: int,
        response_metadata: EmbeddingMetadataObj,
        nn_metadata: EmbeddingMetadataObj,
        latency_direct: float,
        latency_vcache: float,
    ):
        """Update benchmark statistics with results from a single inference.

        This method processes the results of a single inference request and updates
        the appropriate statistics tracking lists. It handles both cache hits and
        misses, calculating true/false positives/negatives based on response
        correctness.

        Args:
            is_cache_hit: Whether the request resulted in a cache hit.
            label_response: The ground truth response for the prompt.
            cache_response: The response returned from the cache (if cache hit).
            label_id_set: The ground truth ID set for the prompt (-1 if not available).
            response_metadata: Metadata object for the cache response.
            nn_metadata: Metadata object for the nearest neighbor in cache.
            latency_direct: Latency for direct inference without cache.
            latency_vcache: Latency for vCache inference including cache logic.

        Note:
            The method uses different correctness evaluation strategies based on
            whether ID sets are available or if it's a custom dataset requiring
            LLM-based evaluation.
        """
        if is_cache_hit:  # If cache hit, the actual response is the nearest neighbor response (cache_response == nn_response)
            self.cache_hit_list.append(1)
            self.cache_miss_list.append(0)

            equality_check_with_id_set: bool = label_id_set != -1
            if equality_check_with_id_set:
                cache_response_correct: bool = label_id_set == response_metadata.id_set
            elif self.is_custom_dataset:
                cache_response_correct: bool = answers_have_same_meaning_llm(
                    label_response, cache_response
                )
            else:
                cache_response_correct: bool = answers_have_same_meaning_static(
                    label_response, cache_response
                )

            if cache_response_correct:
                self.tp_list.append(1)
                self.fp_list.append(0)
            else:
                self.fp_list.append(1)
                self.tp_list.append(0)
            self.fn_list.append(0)
            self.tn_list.append(0)
        else:  # If cache miss, the actual response is the label response
            self.cache_miss_list.append(1)
            self.cache_hit_list.append(0)

            equality_check_with_id_set: bool = label_id_set != -1
            if equality_check_with_id_set:
                nn_response_correct: bool = label_id_set == nn_metadata.id_set
            elif self.is_custom_dataset:
                nn_response_correct: bool = answers_have_same_meaning_llm(
                    label_response, nn_metadata.response
                )
            else:
                nn_response_correct: bool = answers_have_same_meaning_static(
                    label_response, nn_metadata.response
                )

            if nn_response_correct:
                self.fn_list.append(1)
                self.tn_list.append(0)
            else:
                self.tn_list.append(1)
                self.fn_list.append(0)
            self.tp_list.append(0)
            self.fp_list.append(0)

        self.latency_direct_list.append(latency_direct)
        self.latency_vcache_list.append(latency_vcache)

    def get_vcache_answer(
        self,
        prompt: str,
        candidate_embedding: List[float],
        label_response: str,
        system_prompt: str,
        id_set: int,
    ) -> Tuple[bool, str, EmbeddingMetadataObj, EmbeddingMetadataObj, float]:
        """Get vCache response for pre-computed datasets with embedding injection.

        This method simulates vCache inference by injecting pre-computed embeddings
        and responses into the vCache engines, then measuring the cache decision
        and response retrieval performance.

        Args:
            prompt: The input prompt for inference.
            candidate_embedding: Pre-computed embedding vector for the prompt.
            label_response: Ground truth response to inject into inference engine.
            system_prompt: System prompt for structured outputs.
            id_set: ID set for the prompt (used for correctness evaluation).

        Returns:
            Tuple containing:
            - is_cache_hit: Whether the request resulted in a cache hit
            - cache_response: The response returned by vCache
            - response_metadata: Metadata for the cache response
            - nn_metadata: Metadata for the nearest neighbor
            - latency_vcache_logic: Time spent in vCache logic (excluding model calls)

        Note:
            This method handles various embedding formats (string, tensor, numpy array)
            and converts them to the appropriate list format for vCache processing.
        """
        if isinstance(candidate_embedding, str):
            try:
                candidate_embedding = json.loads(candidate_embedding)
            except json.JSONDecodeError:
                print("Error loading embedding from string")
                import ast

                candidate_embedding = ast.literal_eval(candidate_embedding)

        if isinstance(candidate_embedding, torch.Tensor):
            candidate_embedding = candidate_embedding.tolist()
        elif isinstance(candidate_embedding, np.ndarray):
            candidate_embedding = candidate_embedding.tolist()

        if isinstance(candidate_embedding, list):
            candidate_embedding = [
                float(val) if hasattr(val, "__float__") else val
                for val in candidate_embedding
            ]

        self.vcache.vcache_config.embedding_engine.set_next_embedding(
            candidate_embedding
        )
        self.vcache.vcache_config.inference_engine.set_next_response(label_response)

        latency_vcache_logic: float = time.time()
        try:
            is_cache_hit, cache_response, response_metadata, nn_metadata = (
                self.vcache.infer_with_cache_info(
                    prompt=prompt,
                    system_prompt=system_prompt,
                    id_set=id_set,
                )
            )
        except Exception as e:
            logging.error(
                "Error getting vCache answer. Check vCache logs for more details."
            )
            raise e

        latency_vcache_logic = time.time() - latency_vcache_logic
        return (
            is_cache_hit,
            cache_response,
            response_metadata,
            nn_metadata,
            latency_vcache_logic,
        )

    def get_vcache_answer_custom(
        self, prompt: str
    ) -> Tuple[bool, str, EmbeddingMetadataObj, EmbeddingMetadataObj, float]:
        """Get vCache response for custom datasets with live inference.

        This method performs live vCache inference for custom datasets, making
        actual calls to embedding and inference engines without pre-computed
        values.

        Args:
            prompt: The input prompt for inference.

        Returns:
            Tuple containing:
            - is_cache_hit: Whether the request resulted in a cache hit
            - cache_response: The response returned by vCache
            - response_metadata: Metadata for the cache response
            - nn_metadata: Metadata for the nearest neighbor
            - latency_vcache_logic: Time spent in vCache logic (excluding model calls)

        Note:
            This method makes live API calls and may incur costs and latency
            depending on the configured engines.
        """
        latency_vcache_logic: float = time.time()
        try:
            (
                is_cache_hit,
                cache_response,
                response_metadata,
                nn_metadata,
            ) = self.vcache.infer_with_cache_info(prompt=prompt)
        except Exception as e:
            logging.error(
                "Error getting vCache answer. Check vCache logs for more details."
            )
            raise e

        latency_vcache_logic = time.time() - latency_vcache_logic

        return (
            is_cache_hit,
            cache_response,
            response_metadata,
            nn_metadata,
            latency_vcache_logic,
        )

    def dump_results_to_json(self):
        """Serialize benchmark results to JSON file.

        This method collects all benchmark statistics, configuration parameters,
        and internal vCache state (observations, Bayesian parameters) and saves
        them to a JSON file for later analysis and plotting.

        The output includes:
        - Configuration parameters (models, thresholds, policies)
        - Performance metrics (cache hits, accuracy, latency)
        - Internal vCache statistics (observations, policy parameters)
        - Global statistics (if available from the policy)

        The JSON file is saved in the output folder with a timestamp-based filename.

        Raises:
            Exception: If there are issues accessing vCache internal state or
                writing to the output file.
        """
        observations_dict = {}
        gammas_dict = {}
        t_hats_dict = {}
        t_primes_dict = {}
        var_ts_dict = {}

        metadata_objects: List[EmbeddingMetadataObj] = (
            self.vcache.vcache_config.embedding_metadata_storage.get_all_embedding_metadata_objects()
        )

        for metadata_object in metadata_objects:
            observations_dict[metadata_object.embedding_id] = (
                metadata_object.observations
            )
            gammas_dict[metadata_object.embedding_id] = metadata_object.gamma
            t_hats_dict[metadata_object.embedding_id] = metadata_object.t_hat
            t_primes_dict[metadata_object.embedding_id] = metadata_object.t_prime
            var_ts_dict[metadata_object.embedding_id] = metadata_object.var_t

        self.observations_dict = observations_dict
        self.gammas_dict = gammas_dict
        self.t_hats_dict = t_hats_dict
        self.t_primes_dict = t_primes_dict
        self.var_ts_dict = var_ts_dict

        try:
            global_observations_dict = self.vcache.vcache_policy.global_observations
            global_gamma = self.vcache.vcache_policy.bayesian.global_gamma
            global_t_hat = self.vcache.vcache_policy.bayesian.global_t_hat
            global_t_prime = self.vcache.vcache_policy.bayesian.global_t_prime
            global_var_t = self.vcache.vcache_policy.bayesian.global_var_t
        except Exception:
            global_observations_dict = {}
            global_gamma = None
            global_t_hat = None
            global_t_prime = None
            global_var_t = None

        data = {
            "config": {
                "filepath": self.filepath,
                "embedding_model": self.embedding_model,
                "llm_model": self.llm_model,
                "eviction_policy": str(self.eviction_policy),
                "is_static_threshold": self.is_static_threshold,
                "threshold": self.threshold,
                "delta": self.delta,
            },
            "cache_hit_list": self.cache_hit_list,
            "cache_miss_list": self.cache_miss_list,
            "tp_list": self.tp_list,
            "fp_list": self.fp_list,
            "tn_list": self.tn_list,
            "fn_list": self.fn_list,
            "latency_direct_list": self.latency_direct_list,
            "latency_vectorq_list": self.latency_vcache_list,
            "observations_dict": self.observations_dict,
            "gammas_dict": self.gammas_dict,
            "t_hats_dict": self.t_hats_dict,
            "t_primes_dict": self.t_primes_dict,
            "var_ts_dict": self.var_ts_dict,
            "global_observations_dict": global_observations_dict,
            "global_gamma": global_gamma,
            "global_t_hat": global_t_hat,
            "global_t_prime": global_t_prime,
            "global_var_t": global_var_t,
        }

        filepath = self.output_folder_path + f"/results_{self.timestamp}.json"
        with open(filepath, "w") as json_file:
            json.dump(data, json_file, indent=4)
        logging.info(f"Results successfully dumped to {filepath}")


########################################################################################################################
### Helper #############################################################################################################
########################################################################################################################


def __run_baseline(
    vcache_policy: VCachePolicy,
    path: str,
    dataset_file: str,
    embedding_model: Tuple[str, str, str, int],
    llm_model: Tuple[str, str, str, int],
    timestamp: str,
    delta: float,
    threshold: float,
    similarity_evaluator: SimilarityEvaluator,
    eviction_policy: EvictionPolicy,
    max_samples: int,
    is_custom_dataset: bool = False,
):
    """Run a single baseline benchmark configuration.

    This helper function creates a vCache instance with the specified configuration
    and runs a complete benchmark evaluation. It handles both custom datasets
    (requiring live inference) and pre-computed datasets.

    Args:
        vcache_policy: The caching policy to evaluate (e.g., VerifiedDecisionPolicy).
        path: Output directory path for saving results.
        dataset_file: Path to the dataset file or HuggingFace dataset ID.
        embedding_model: Tuple containing embedding model configuration.
        llm_model: Tuple containing LLM model configuration.
        timestamp: Timestamp string for result file naming.
        delta: Delta parameter for dynamic policies (-1 if not applicable).
        threshold: Threshold parameter for static policies (-1 if not applicable).
        similarity_evaluator: Strategy for evaluating response similarity.
        eviction_policy: Cache eviction policy instance.
        max_samples: Maximum number of samples to process.
        is_custom_dataset: Whether using custom dataset format requiring live inference.

    Note:
        This function creates different vCache configurations based on whether
        it's processing custom datasets (using OpenAI engines) or pre-computed
        datasets (using benchmark engines).
    """
    if is_custom_dataset:
        llm_model_name = llm_model[1].lower()
        embedding_model_name = embedding_model[1].lower()

        vcache_config: VCacheConfig = VCacheConfig(
            inference_engine=OpenAIInferenceEngine(model_name=llm_model_name),
            embedding_engine=OpenAIEmbeddingEngine(model_name=embedding_model_name),
            vector_db=HNSWLibVectorDB(
                similarity_metric_type=SimilarityMetricType.COSINE,
                max_capacity=MAX_VECTOR_DB_CAPACITY,
            ),
            embedding_metadata_storage=InMemoryEmbeddingMetadataStorage(),
            similarity_evaluator=similarity_evaluator,
            eviction_policy=eviction_policy,
        )
    else:
        vcache_config: VCacheConfig = VCacheConfig(
            inference_engine=BenchmarkInferenceEngine(),
            embedding_engine=BenchmarkEmbeddingEngine(),
            vector_db=HNSWLibVectorDB(
                similarity_metric_type=SimilarityMetricType.COSINE,
                max_capacity=MAX_VECTOR_DB_CAPACITY,
            ),
            embedding_metadata_storage=InMemoryEmbeddingMetadataStorage(),
            similarity_evaluator=similarity_evaluator,
            eviction_policy=eviction_policy,
        )

    vcache: VCache = VCache(vcache_config, vcache_policy)

    benchmark = Benchmark(vcache)
    benchmark.filepath = dataset_file
    benchmark.embedding_model = embedding_model
    benchmark.llm_model = llm_model
    benchmark.timestamp = timestamp
    benchmark.threshold = threshold if threshold != -1 else None
    benchmark.delta = delta if delta != -1 else None
    benchmark.is_static_threshold = threshold != -1
    benchmark.output_folder_path = path
    benchmark.eviction_policy = eviction_policy
    benchmark.is_custom_dataset = is_custom_dataset

    benchmark.stats_set_up()

    try:
        benchmark.test_run_benchmark(max_samples)
    except Exception as e:
        logging.error(f"Error running benchmark: {e}")


########################################################################################################################
### Main ###############################################################################################################
########################################################################################################################


def main():
    """Main function that orchestrates the complete benchmarking process.

    This function serves as the entry point for the benchmarking system. It:
    1. Sets up the benchmarking environment and output directories
    2. Iterates through all configured RUN_COMBINATIONS
    3. For each combination, runs all specified baselines with their parameter sweeps
    4. Generates combined plots comparing all baselines
    5. Logs timing information and completion status

    The function handles multiple baselines including:
    - vCache Local: Dynamic local threshold adaptation
    - vCache Global: Dynamic global threshold adaptation
    - Berkeley Embedding: Fine-tuned embeddings with static thresholds
    - vCache + Berkeley Embedding: vCache with fine-tuned embeddings
    - IID: Independent and Identically Distributed threshold policy
    - GPTCache: Static threshold-based caching

    Note:
        Configuration is controlled through global variables in the "Benchmark Config"
        section. Modify RUN_COMBINATIONS, BASELINES_TO_RUN, and parameter lists to
        customize the benchmarking process.
    """
    benchmarks_dir = os.path.dirname(os.path.abspath(__file__))

    custom_datasets_dir = os.path.join(benchmarks_dir, "your_datasets")
    if not os.path.exists(custom_datasets_dir):
        os.makedirs(custom_datasets_dir, exist_ok=True)
        logging.info(f"Created directory: {custom_datasets_dir}")

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")

    for (
        embedding_model,
        llm_model,
        dataset,
        generate_plots_only,
        similarity_evaluator,
        eviction_policy,
        max_samples,
    ) in RUN_COMBINATIONS:
        try:
            dataset_value: str = dataset.value
            is_custom_dataset: bool = dataset_value.startswith("your_datasets")

            if is_custom_dataset:
                # The path in the enum is relative to 'benchmarks/data/'
                dataset_path = os.path.join(benchmarks_dir, dataset_value)
                dataset_name = os.path.basename(dataset_path)
                if not os.path.exists(dataset_path):
                    logging.warning(f"Custom dataset file not found: {dataset_path}")
                    continue
            else:
                dataset_name = dataset_value
                if "/" in dataset_name:  # HuggingFace dataset
                    dataset_path = dataset_name
                    logging.info(f"Using Hugging Face dataset: {dataset_path}")
                else:
                    logging.warning(
                        f"Dataset {dataset_name} not found. Please check the dataset path."
                    )
                    continue

            logging.info(
                f"\nRunning benchmark for dataset: {dataset_name}, embedding model: {embedding_model.value[1]}, LLM model: {llm_model.value[1]}\n"
            )
            start_time_llm_model = time.time()

            #####################################################
            ### Baseline: vCache Local
            if (
                Baseline.VCacheLocal in BASELINES_TO_RUN
                and not generate_plots_only.value
            ):
                for delta in DELTAS:
                    for i in range(0, CONFIDENCE_INTERVALS_ITERATIONS):
                        path = os.path.join(
                            results_dir,
                            dataset_name,
                            embedding_model.value[1],
                            llm_model.value[1],
                            f"vcache_local_{delta}_run_{i + 1}",
                        )
                        if os.path.exists(path) and os.listdir(path):
                            continue

                        logging.info(
                            f"vCache Local: Using dynamic threshold with delta: {delta}. Run {i + 1} of {CONFIDENCE_INTERVALS_ITERATIONS}"
                        )

                        __run_baseline(
                            vcache_policy=VerifiedDecisionPolicy(delta=delta),
                            path=path,
                            dataset_file=dataset_path,
                            embedding_model=embedding_model.value,
                            llm_model=llm_model.value,
                            timestamp=timestamp,
                            delta=delta,
                            threshold=-1,
                            similarity_evaluator=similarity_evaluator,
                            eviction_policy=eviction_policy,
                            max_samples=max_samples,
                            is_custom_dataset=is_custom_dataset,
                        )

            #####################################################
            ### Baseline: vCache Global
            if (
                Baseline.VCacheGlobal in BASELINES_TO_RUN
                and not generate_plots_only.value
            ):
                for delta in DELTAS:
                    path = os.path.join(
                        results_dir,
                        dataset_name,
                        embedding_model.value[1],
                        llm_model.value[1],
                        f"vcache_global_{delta}",
                    )
                    if os.path.exists(path) and os.listdir(path):
                        continue

                    logging.info(
                        f"vCache Global: Using dynamic threshold with delta: {delta}. Run {i + 1} of {CONFIDENCE_INTERVALS_ITERATIONS}"
                    )

                    __run_baseline(
                        vcache_policy=BenchmarkVerifiedGlobalDecisionPolicy(
                            delta=delta
                        ),
                        path=path,
                        dataset_file=dataset_path,
                        embedding_model=embedding_model.value,
                        llm_model=llm_model.value,
                        timestamp=timestamp,
                        delta=delta,
                        threshold=-1,
                        similarity_evaluator=similarity_evaluator,
                        eviction_policy=eviction_policy,
                        max_samples=max_samples,
                        is_custom_dataset=is_custom_dataset,
                    )

            #####################################################
            ### Baseline: Berkeley Embedding
            if (
                Baseline.BerkeleyEmbedding in BASELINES_TO_RUN
                and not generate_plots_only.value
            ):
                for threshold in STATIC_THRESHOLDS:
                    if embedding_model == EmbeddingModel.E5_MISTRAL_7B:
                        logging.info(
                            f"Skipping Berkeley Embedding for {embedding_model.value[1]}. No fine-tuned Berkeley Embedding for this model."
                        )
                        continue

                    if embedding_model == EmbeddingModel.GTE:
                        berkeley_embedding_model = EmbeddingModel.GTE_FT
                    elif embedding_model == EmbeddingModel.E5_LARGE_V2:
                        berkeley_embedding_model = EmbeddingModel.E5_LARGE_V2_FT
                    else:
                        logging.info(
                            f"Skipping Berkeley Embedding for {embedding_model.value[1]}. No fine-tuned Berkeley Embedding for this model."
                        )
                        continue

                    path = os.path.join(
                        results_dir,
                        dataset_name,
                        berkeley_embedding_model.value[1],
                        llm_model.value[1],
                        f"berkeley_embedding_{threshold}",
                    )
                    if os.path.exists(path) and os.listdir(path):
                        continue

                    logging.info(
                        f"Berkeley Embedding: Using static threshold: {threshold}"
                    )

                    __run_baseline(
                        vcache_policy=BenchmarkStaticDecisionPolicy(
                            threshold=threshold
                        ),
                        path=path,
                        dataset_file=dataset_path,
                        embedding_model=berkeley_embedding_model.value,
                        llm_model=llm_model.value,
                        timestamp=timestamp,
                        delta=-1,
                        threshold=threshold,
                        similarity_evaluator=similarity_evaluator,
                        eviction_policy=eviction_policy,
                        max_samples=max_samples,
                        is_custom_dataset=is_custom_dataset,
                    )

            #####################################################
            ### Baseline: vCache + Berkeley Embedding
            if (
                Baseline.VCacheBerkeleyEmbedding in BASELINES_TO_RUN
                and not generate_plots_only.value
            ):
                for delta in DELTAS:
                    for i in range(0, CONFIDENCE_INTERVALS_ITERATIONS):
                        if embedding_model == EmbeddingModel.E5_MISTRAL_7B:
                            logging.info(
                                f"Skipping Berkeley Embedding for {embedding_model.value[1]}. No fine-tuned Berkeley Embedding for this model."
                            )
                            continue

                        if embedding_model == EmbeddingModel.GTE:
                            berkeley_embedding_model = EmbeddingModel.GTE_FT
                        elif embedding_model == EmbeddingModel.E5_LARGE_V2:
                            berkeley_embedding_model = EmbeddingModel.E5_LARGE_V2_FT
                        else:
                            logging.info(
                                f"Skipping Berkeley Embedding for {embedding_model.value[1]}. No fine-tuned Berkeley Embedding for this model."
                            )
                            continue

                        path = os.path.join(
                            results_dir,
                            dataset_name,
                            berkeley_embedding_model.value[1],
                            llm_model.value[1],
                            f"vcache_berkeley_embedding_{delta}_run_{i + 1}",
                        )
                        if os.path.exists(path) and os.listdir(path):
                            continue

                        logging.info(
                            f"vCache + Berkeley Embedding: Using dynamic threshold with delta: {delta}. Run {i + 1} of {CONFIDENCE_INTERVALS_ITERATIONS}"
                        )

                        __run_baseline(
                            vcache_policy=VerifiedDecisionPolicy(delta=delta),
                            path=path,
                            dataset_file=dataset_path,
                            embedding_model=berkeley_embedding_model.value,
                            llm_model=llm_model.value,
                            timestamp=timestamp,
                            delta=delta,
                            threshold=-1,
                            similarity_evaluator=similarity_evaluator,
                            eviction_policy=eviction_policy,
                            max_samples=max_samples,
                            is_custom_dataset=is_custom_dataset,
                        )

            #####################################################
            ### Baseline: IID Local
            if Baseline.IID in BASELINES_TO_RUN and not generate_plots_only.value:
                for delta in DELTAS:
                    for i in range(0, CONFIDENCE_INTERVALS_ITERATIONS):
                        path = os.path.join(
                            results_dir,
                            dataset_name,
                            embedding_model.value[1],
                            llm_model.value[1],
                            f"iid_local_{delta}_run_{i + 1}",
                        )
                        if os.path.exists(path) and os.listdir(path):
                            continue

                        logging.info(
                            f"IID: Using delta {delta}. Run {i + 1} of {CONFIDENCE_INTERVALS_ITERATIONS}"
                        )

                        __run_baseline(
                            vcache_policy=BenchmarkVerifiedIIDDecisionPolicy(
                                delta=delta
                            ),
                            path=path,
                            dataset_file=dataset_path,
                            embedding_model=embedding_model.value,
                            llm_model=llm_model.value,
                            timestamp=timestamp,
                            delta=delta,
                            threshold=-1,
                            similarity_evaluator=similarity_evaluator,
                            eviction_policy=eviction_policy,
                            max_samples=max_samples,
                            is_custom_dataset=is_custom_dataset,
                        )

            #####################################################
            ### Baseline: GPTCache
            if Baseline.GPTCache in BASELINES_TO_RUN and not generate_plots_only.value:
                for threshold in STATIC_THRESHOLDS:
                    path = os.path.join(
                        results_dir,
                        dataset_name,
                        embedding_model.value[1],
                        llm_model.value[1],
                        f"gptcache_{threshold}",
                    )
                    if os.path.exists(path) and os.listdir(path):
                        continue

                    logging.info(f"GPTCache: Using static threshold: {threshold}")

                    __run_baseline(
                        vcache_policy=BenchmarkStaticDecisionPolicy(
                            threshold=threshold
                        ),
                        path=path,
                        dataset_file=dataset_path,
                        embedding_model=embedding_model.value,
                        llm_model=llm_model.value,
                        timestamp=timestamp,
                        delta=-1,
                        threshold=threshold,
                        similarity_evaluator=similarity_evaluator,
                        eviction_policy=eviction_policy,
                        max_samples=max_samples,
                        is_custom_dataset=is_custom_dataset,
                    )

            #####################################################
            ### Baseline: Sigmoid Probability
            if (
                Baseline.SigmoidProbability in BASELINES_TO_RUN
                and not generate_plots_only.value
            ):
                for delta in DELTAS:
                    for i in range(0, CONFIDENCE_INTERVALS_ITERATIONS):
                        path = os.path.join(
                            results_dir,
                            dataset_name,
                            embedding_model.value[1],
                            llm_model.value[1],
                            f"sigmoid_probability_{delta}_run_{i + 1}",
                        )
                        if os.path.exists(path) and os.listdir(path):
                            continue

                        logging.info(
                            f"Sigmoid Probability: Using dynamic threshold with delta: {delta}. Run {i + 1} of {CONFIDENCE_INTERVALS_ITERATIONS}"
                        )

                        __run_baseline(
                            vcache_policy=SigmoidProbabilityDecisionPolicy(delta=delta),
                            path=path,
                            dataset_file=dataset_path,
                            embedding_model=embedding_model.value,
                            llm_model=llm_model.value,
                            timestamp=timestamp,
                            delta=delta,
                            threshold=-1,
                            similarity_evaluator=similarity_evaluator,
                            eviction_policy=eviction_policy,
                            max_samples=max_samples,
                            is_custom_dataset=is_custom_dataset,
                        )

            #####################################################
            ### Baseline: Sigmoid Only
            if (
                Baseline.SigmoidOnly in BASELINES_TO_RUN
                and not generate_plots_only.value
            ):
                for delta in DELTAS:
                    for i in range(0, CONFIDENCE_INTERVALS_ITERATIONS):
                        path = os.path.join(
                            results_dir,
                            dataset_name,
                            embedding_model.value[1],
                            llm_model.value[1],
                            f"sigmoid_only_{delta}_run_{i + 1}",
                        )
                        if os.path.exists(path) and os.listdir(path):
                            continue

                        logging.info(
                            f"Sigmoid Only: Using dynamic threshold with delta: {delta}. Run {i + 1} of {CONFIDENCE_INTERVALS_ITERATIONS}"
                        )

                        __run_baseline(
                            vcache_policy=SigmoidOnlyDecisionPolicy(delta=delta),
                            path=path,
                            dataset_file=dataset_path,
                            embedding_model=embedding_model.value,
                            llm_model=llm_model.value,
                            timestamp=timestamp,
                            delta=delta,
                            threshold=-1,
                            similarity_evaluator=similarity_evaluator,
                            eviction_policy=eviction_policy,
                            max_samples=max_samples,
                            is_custom_dataset=is_custom_dataset,
                        )

            #####################################################
            generate_combined_plots(
                dataset=dataset_name,
                embedding_model_name=embedding_model.value[1],
                llm_model_name=llm_model.value[1],
                results_dir=results_dir,
                timestamp=timestamp,
                font_size=PLOT_FONT_SIZE,
                keep_split=KEEP_SPLIT,
            )

            end_time_embedding_model = time.time()
            logging.info(
                f"LLM Model Time: {(end_time_embedding_model - start_time_llm_model) / 60:.2f} minutes, {(end_time_embedding_model - start_time_llm_model) / 3600:.4f} hours"
            )

        except Exception:
            logging.error(
                f"Error running benchmark. Combination {embedding_model.value[1]} {llm_model.value[1]} {dataset.value} failed."
            )

    total_time = (
        time.time()
        - time.mktime(datetime.strptime(timestamp, "%Y-%m-%d_%H-%M").timetuple())
    ) / 3600
    logging.info(f"All benchmarks completed in {total_time:.2f} hours!")


if __name__ == "__main__":
    main()
