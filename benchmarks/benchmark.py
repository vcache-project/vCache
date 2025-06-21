import json
import logging
import os
import time
import unittest
from datetime import datetime
from enum import Enum
from typing import Dict, List, Tuple

import ijson
import numpy as np
import torch
from datasets import load_dataset
from tqdm import tqdm

from benchmarks._plotter_combined import generate_combined_plots
from benchmarks._plotter_individual import generate_individual_plots
from benchmarks.common.comparison import answers_have_same_meaning_static
from vcache.config import VCacheConfig
from vcache.inference_engine.strategies.benchmark import (
    BenchmarkInferenceEngine,
)
from vcache.main import VCache
from vcache.vcache_core.cache.embedding_engine.strategies.benchmark import (
    BenchmarkEmbeddingEngine,
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
    GTE = ("emb_gte", "GteLargeENv1_5", "float32", 1024)
    GTE_FT = ("emb_gte_ft", "GteLargeENv1_5", "float32", 1024)
    E5_MISTRAL_7B = ("emb_e5_mistral_7b", "E5_Mistral_7B_Instruct", "float16", 4096)
    E5_LARGE_V2 = ("emb_e5_large_v2", "E5_Large_v2", "float16", 512)
    E5_LARGE_V2_FT = ("emb_e5_large_v2_ft", "E5_Large_v2", "float16", 512)


class LargeLanguageModel(Enum):
    LLAMA_3_8B = ("response_llama_3_8b", "Llama_3_8B_Instruct", "float16", None)
    LLAMA_3_70B = ("response_llama_3_70b", "Llama_3_70B_Instruct", "float16", None)
    GPT_4O_MINI = ("response_gpt-4o-mini", "GPT-4o-mini", "float16", None)
    GPT_4O_NANO = ("response_gpt-4.1-nano", "GPT-4.1-nano", "float16", None)


class Baseline(Enum):
    GPTCache = "GPTCache"
    VCacheLocal = "vCacheLocal"
    VCacheGlobal = "vCacheGlobal"
    BerkeleyEmbedding = "BerkeleyEmbedding"
    VCacheBerkeleyEmbedding = "VCacheBerkeleyEmbedding"
    IID = "iid"


class Dataset(Enum):
    SEM_BENCHMARK_CLASSIFICATION = "vCache/SemBenchmarkClassification"
    SEM_BENCHMARK_ARENA = "vCache/SemBenchmarkLmArena"
    SEM_BENCHMARK_SEARCH_QUERIES = "vCache/SemBenchmarkSearchQueries"
    CUSTOM = "datasets/"


class GeneratePlotsOnly(Enum):
    YES = True
    NO = False


########################################################################################################################
### Benchmark Config ###################################################################################################
########################################################################################################################

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

CONFIDENCE_INTERVALS_ITERATIONS: int = 1
DISABLE_PROGRESS_BAR: bool = False
KEEP_SPLIT: int = 100
MAX_VECTOR_DB_CAPACITY: int = 150000
PLOT_FONT_SIZE: int = 50

RUN_COMBINATIONS: List[
    Tuple[EmbeddingModel, LargeLanguageModel, Dataset, GeneratePlotsOnly]
] = [
    (
        EmbeddingModel.GTE,
        LargeLanguageModel.GPT_4O_MINI,
        Dataset.SEM_BENCHMARK_ARENA,
        GeneratePlotsOnly.NO,
        BenchmarkComparisonSimilarityEvaluator(),
        SCUEvictionPolicy(max_size=6000, watermark=0.99, eviction_percentage=0.1),
        60000,
    ),
    (
        EmbeddingModel.E5_LARGE_V2,
        LargeLanguageModel.GPT_4O_MINI,
        Dataset.SEM_BENCHMARK_SEARCH_QUERIES,
        GeneratePlotsOnly.NO,
        BenchmarkComparisonSimilarityEvaluator(),
        SCUEvictionPolicy(max_size=15000, watermark=0.99, eviction_percentage=0.1),
        150000,
    ),
    (
        EmbeddingModel.GTE,
        LargeLanguageModel.LLAMA_3_8B,
        Dataset.SEM_BENCHMARK_CLASSIFICATION,
        GeneratePlotsOnly.NO,
        StringComparisonSimilarityEvaluator(),
        SCUEvictionPolicy(max_size=4500, watermark=0.99, eviction_percentage=0.1),
        45000,
    ),
]

BASELINES_TO_RUN: List[Baseline] = [
    Baseline.VCacheLocal,
    Baseline.IID,
    Baseline.GPTCache,
    Baseline.BerkeleyEmbedding,
    Baseline.VCacheBerkeleyEmbedding,
]

STATIC_THRESHOLDS: List[float] = [0.98]
# STATIC_THRESHOLDS: List[float] = [
#     0.80,
#     0.81,
#     0.82,
#     0.83,
#     0.84,
#     0.85,
#     0.86,
#     0.87,
#     0.88,
#     0.89,
#     0.90,
#     0.91,
#     0.92,
#     0.93,
#     0.94,
#     0.95,
#     0.96,
#     0.97,
#     0.98,
#     0.99
# ]

DELTAS: List[float] = [0.05]
# DELTAS: List[float] = [
#     0.01, 0.015, 0.02, 0.025, 0.03, 0.035, 0.04, 0.05, 0.06, 0.07
# ]


########################################################################################################################
### Benchmark Class ####################################################################################################
########################################################################################################################
class Benchmark(unittest.TestCase):
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

    def stats_set_up(self):
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

    def run_benchmark_loop(self, data_entries, max_samples):
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
        if not self.filepath or not self.embedding_model or not self.llm_model:
            raise ValueError(
                f"Required parameters not set: filepath: {self.filepath}, embedding_model: {self.embedding_model}, or llm_model: {self.llm_model}"
            )

        try:
            if "/" in self.filepath:
                logging.info(f"Loading Hugging Face dataset: {self.filepath}")
                data_iterator = load_dataset(
                    self.filepath, split=f"train[:{max_samples}]"
                )
                self.run_benchmark_loop(data_iterator, max_samples)
            else:
                logging.info(f"Loading local dataset: {self.filepath}")
                with open(self.filepath, "rb") as file:
                    data_entries = ijson.items(file, "item")
                    self.run_benchmark_loop(data_entries, max_samples)

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
        if is_cache_hit:  # If cache hit, the actual response is the nearest neighbor response (cache_response == nn_response)
            self.cache_hit_list.append(1)
            self.cache_miss_list.append(0)

            equality_check_with_id_set: bool = label_id_set != -1
            if equality_check_with_id_set:
                cache_response_correct: bool = label_id_set == response_metadata.id_set
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
        """
        Returns: Tuple[bool, str, EmbeddingMetadataObj, EmbeddingMetadataObj, float] - [is_cache_hit, cache_response, response_metadata, nn_metadata, latency_vcache_logic]
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

    def dump_results_to_json(self):
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
):
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

    benchmark.stats_set_up()
    try:
        benchmark.test_run_benchmark(max_samples)
    except Exception as e:
        logging.error(f"Error running benchmark: {e}")


########################################################################################################################
### Main ###############################################################################################################
########################################################################################################################


def main():
    benchmarks_dir = os.path.dirname(os.path.abspath(__file__))
    datasets_dir = os.path.join(benchmarks_dir, "data", "large_scale")
    if not os.path.exists(datasets_dir):
        os.makedirs(datasets_dir, exist_ok=True)
        logging.info(f"Created directory: {datasets_dir}")

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
            dataset_name = dataset.value
            dataset_path = ""
            if "/" in dataset_name:
                dataset_path = dataset_name
                logging.info(f"Using Hugging Face dataset: {dataset_path}")
            else:
                dataset_path = os.path.join(datasets_dir, f"{dataset_name}.json")
                logging.info(f"Using local dataset: {dataset_path}")

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
                            f"Using dynamic threshold with delta: {delta}. Run {i + 1} of {CONFIDENCE_INTERVALS_ITERATIONS}"
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
                        f"Using dynamic threshold with delta: {delta}. Run {i + 1} of {CONFIDENCE_INTERVALS_ITERATIONS}"
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

                    logging.info(f"Using static threshold: {threshold}")

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
                            f"Using dynamic threshold with delta: {delta}. Run {i + 1} of {CONFIDENCE_INTERVALS_ITERATIONS}"
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
                            f"Using IID local threshold with delta: {delta}. Run {i + 1} of {CONFIDENCE_INTERVALS_ITERATIONS}"
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

                    logging.info(f"Using static threshold: {threshold}")

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
