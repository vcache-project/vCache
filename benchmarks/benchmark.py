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
from tqdm import tqdm

from benchmarks._plotter_combined import generate_combined_plots
from benchmarks._plotter_individual import generate_individual_plots
from benchmarks.common.comparison import answers_have_same_meaning_static
from vcache.config import VCacheConfig
from vcache.inference_engine.strategies.benchmark import (
    BenchmarkInferenceEngine,
)
from vcache.main import vCache
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
from vcache.vcache_core.similarity_evaluator.strategies.llm_comparison import (
    LLMComparisonSimilarityEvaluator,
)
from vcache.vcache_core.similarity_evaluator.strategies.string_comparison import (
    StringComparisonSimilarityEvaluator,
)
from vcache.vcache_policy.strategies.dynamic_global_threshold import (
    DynamicGlobalThresholdPolicy,
)
from vcache.vcache_policy.strategies.dynamic_local_threshold import (
    DynamicLocalThresholdPolicy,
)
from vcache.vcache_policy.strategies.iid_local_threshold import (
    IIDLocalThresholdPolicy,
)
from vcache.vcache_policy.strategies.static_global_threshold import (
    StaticGlobalThresholdPolicy,
)
from vcache.vcache_policy.vcache_policy import vCachePolicy

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
    SEM_BENCHMARK_CLASSIFICATION = "benchmark_classification"
    SEM_BENCHMARK_ARENA = "benchmark_arena"
    SEM_BENCHMARK_SEARCH_QUERIES = "sem_benchmark_search_queries_150k"
    AMAZON_INSTANT_VIDEO = "amazon_instant_video"
    COMMONSENSE_QA = "commonsense_qa"
    ECOMMERCE_DATASET = "ecommerce_dataset"


class GenerateResultsOnly(Enum):
    YES = True
    NO = False


########################################################################################################################
### Benchmark Config ###################################################################################################
########################################################################################################################


MAX_SAMPLES: int = 60000
CONFIDENCE_INTERVALS_ITERATIONS: int = 5
IS_LLM_JUDGE_BENCHMARK: bool = False
DISABLE_PROGRESS_BAR: bool = True
KEEP_SPLIT: int = 100

RUN_COMBINATIONS: List[
    Tuple[EmbeddingModel, LargeLanguageModel, Dataset, GenerateResultsOnly]
] = [
    (
        EmbeddingModel.GTE,
        LargeLanguageModel.LLAMA_3_8B,
        Dataset.SEM_BENCHMARK_SEARCH_QUERIES,
        GenerateResultsOnly.YES,
    ),
    (
        EmbeddingModel.GTE,
        LargeLanguageModel.GPT_4O_MINI,
        Dataset.SEM_BENCHMARK_ARENA,
        GenerateResultsOnly.YES,
    ),
    (
        EmbeddingModel.E5_LARGE_V2,
        LargeLanguageModel.GPT_4O_MINI,
        Dataset.SEM_BENCHMARK_ARENA,
        GenerateResultsOnly.YES,
    ),
    (
        EmbeddingModel.E5_LARGE_V2,
        LargeLanguageModel.LLAMA_3_8B,
        Dataset.SEM_BENCHMARK_CLASSIFICATION,
        GenerateResultsOnly.YES,
    ),
    (
        EmbeddingModel.GTE,
        LargeLanguageModel.LLAMA_3_8B,
        Dataset.SEM_BENCHMARK_CLASSIFICATION,
        GenerateResultsOnly.YES,
    ),
    (
        EmbeddingModel.GTE,
        LargeLanguageModel.LLAMA_3_70B,
        Dataset.SEM_BENCHMARK_CLASSIFICATION,
        GenerateResultsOnly.YES,
    ),
]

BASELINES_TO_RUN: List[Baseline] = [
    # Baseline.IID,
    # Baseline.GPTCache,
    # Baseline.VCacheLocal,
    # Baseline.BerkeleyEmbedding,
    # Baseline.VCacheBerkeleyEmbedding,
]

DATASETS_TO_RUN: List[str] = [Dataset.SEM_BENCHMARK_SEARCH_QUERIES]

STATIC_THRESHOLDS: List[float] = [
    0.80,
    0.81,
    0.82,
    0.83,
    0.84,
    0.85,
    0.86,
    0.87,
    0.88,
    0.89,
    0.90,
    0.91,
    0.92,
    0.93,
    0.94,
    0.95,
    0.96,
    0.97,
    0.98,
]

DELTAS: List[float] = [0.01, 0.015, 0.02, 0.025, 0.03, 0.035, 0.04, 0.05, 0.06, 0.07]

MAX_VECTOR_DB_CAPACITY: int = 100000
PLOT_FONT_SIZE: int = 50


########################################################################################################################
### Benchmark Class ####################################################################################################
########################################################################################################################
class Benchmark(unittest.TestCase):
    def __init__(self, vectorq: vCache):
        super().__init__()
        self.vectorq: vCache = vectorq
        self.embedding_model: Tuple[str, str, str, int] = None
        self.llm_model: Tuple[str, str, str, int] = None
        self.filepath: str = None
        self.output_folder_path: str = None
        self.timestamp: str = None
        self.threshold: float = None
        self.delta: float = None
        self.is_static_threshold: bool = None

    def stats_set_up(self):
        self.cache_hit_list: List[int] = []
        self.cache_miss_list: List[int] = []
        self.tp_list: List[int] = []
        self.fp_list: List[int] = []
        self.tn_list: List[int] = []
        self.fn_list: List[int] = []
        self.latency_direct_list: List[float] = []
        self.latency_vectorq_list: List[float] = []
        self.observations_dict: Dict[str, Dict[str, float]] = {}
        self.gammas_dict: Dict[str, float] = {}
        self.t_hats_dict: Dict[str, float] = {}
        self.t_primes_dict: Dict[str, float] = {}
        self.var_ts_dict: Dict[str, float] = {}

        if self.output_folder_path and not os.path.exists(self.output_folder_path):
            os.makedirs(self.output_folder_path)

    def test_run_benchmark(self):
        if not self.filepath or not self.embedding_model or not self.llm_model:
            raise ValueError(
                f"Required parameters not set: filepath: {self.filepath}, embedding_model: {self.embedding_model}, or llm_model: {self.llm_model}"
            )

        try:
            with open(self.filepath, "rb") as file:
                data_entries = ijson.items(file, "item")

                pbar = tqdm(
                    total=MAX_SAMPLES,
                    desc="Processing entries",
                    disable=DISABLE_PROGRESS_BAR,
                )
                for idx, data_entry in enumerate(data_entries):
                    if idx >= MAX_SAMPLES:
                        break

                    # 1) Get Data
                    task = data_entry["task"]
                    system_prompt = data_entry.get("output_format", "")
                    review_text = data_entry.get("text", "")

                    # TODO Hacked: Needs to be fixed in benchmark arena json file
                    if self.embedding_model[0] == "emb_e5_large_v2_ft":
                        emb_generation_latency: float = float(
                            data_entry["emb_e5_large_v2_lat"]
                        )
                    else:
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
                    candidate_embedding: List[float] = data_entry[
                        self.embedding_model[0]
                    ]
                    is_cache_hit, cache_response, nn_response, latency_vectorq_logic = (
                        self.get_vectorQ_answer(
                            task=task,
                            review_text=review_text,
                            candidate_embedding=candidate_embedding,
                            label_response=label_response,
                            system_prompt=system_prompt,
                        )
                    )
                    latency_vectorq: float = (
                        latency_vectorq_logic + emb_generation_latency
                    )
                    if not is_cache_hit:
                        latency_vectorq += llm_generation_latency

                    # 3) Update Stats
                    self.update_stats(
                        is_cache_hit=is_cache_hit,
                        label_response=label_response,
                        cache_response=cache_response,
                        nn_response=nn_response,
                        latency_direct=latency_direct,
                        latency_vectorq=latency_vectorq,
                    )

                    pbar.update(1)

                pbar.close()

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
        nn_response: str,
        latency_direct: float,
        latency_vectorq: float,
    ):
        if is_cache_hit:  # If cache hit, the actual response is the nearest neighbor response (cache_response == nn_response)
            self.cache_hit_list.append(1)
            self.cache_miss_list.append(0)
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
            nn_response_correct: bool = answers_have_same_meaning_static(
                label_response, nn_response
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
        self.latency_vectorq_list.append(latency_vectorq)

    def get_vectorQ_answer(
        self,
        task: str,
        review_text: str,
        candidate_embedding: List[float],
        label_response: str,
        system_prompt: str,
    ) -> Tuple[bool, str, str, float]:
        """
        Returns: Tuple[bool, str, str, float] - [is_cache_hit, cache_response, nn_response, latency_vectorq_logic]
        """
        if isinstance(candidate_embedding, torch.Tensor):
            candidate_embedding = candidate_embedding.tolist()
        elif isinstance(candidate_embedding, np.ndarray):
            candidate_embedding = candidate_embedding.tolist()

        if isinstance(candidate_embedding, list):
            candidate_embedding = [
                float(val) if hasattr(val, "__float__") else val
                for val in candidate_embedding
            ]

        self.vectorq.vectorq_config.embedding_engine.set_next_embedding(
            candidate_embedding
        )
        self.vectorq.vectorq_config.inference_engine.set_next_response(label_response)

        vectorQ_prompt = f"{task} {review_text}"
        latency_vectorq_logic: float = time.time()
        try:
            is_cache_hit, cache_response, nn_response = (
                self.vectorq.infer_with_cache_info(
                    prompt=vectorQ_prompt,
                    system_prompt=system_prompt,
                )
            )
        except Exception as e:
            logging.error(
                "Error getting vCache answer. Check vCache logs for more details."
            )
            raise e

        latency_vectorq_logic = time.time() - latency_vectorq_logic
        return is_cache_hit, cache_response, nn_response, latency_vectorq_logic

    def dump_results_to_json(self):
        observations_dict = {}
        gammas_dict = {}
        t_hats_dict = {}
        t_primes_dict = {}
        var_ts_dict = {}

        metadata_objects: List[EmbeddingMetadataObj] = (
            self.vectorq.vectorq_config.embedding_metadata_storage.get_all_embedding_metadata_objects()
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
            global_observations_dict = (
                self.vectorq.vectorq_policy.bayesian.global_observations
            )
            global_gamma = self.vectorq.vectorq_policy.bayesian.global_gamma
            global_t_hat = self.vectorq.vectorq_policy.bayesian.global_t_hat
            global_t_prime = self.vectorq.vectorq_policy.bayesian.global_t_prime
            global_var_t = self.vectorq.vectorq_policy.bayesian.global_var_t
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
            "latency_vectorq_list": self.latency_vectorq_list,
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
    vectorq_policy: vCachePolicy,
    path: str,
    dataset_file: str,
    embedding_model: Tuple[str, str, str, int],
    llm_model: Tuple[str, str, str, int],
    timestamp: str,
    delta: float,
    threshold: float,
):
    if IS_LLM_JUDGE_BENCHMARK:
        similarity_evaluator = LLMComparisonSimilarityEvaluator()
    else:
        similarity_evaluator = StringComparisonSimilarityEvaluator()

    vectorq_config: VCacheConfig = VCacheConfig(
        inference_engine=BenchmarkInferenceEngine(),
        embedding_engine=BenchmarkEmbeddingEngine(),
        vector_db=HNSWLibVectorDB(
            similarity_metric_type=SimilarityMetricType.COSINE,
            max_capacity=MAX_VECTOR_DB_CAPACITY,
        ),
        embedding_metadata_storage=InMemoryEmbeddingMetadataStorage(),
        similarity_evaluator=similarity_evaluator,
    )
    vectorQ: vCache = vCache(vectorq_config, vectorq_policy)

    benchmark = Benchmark(vectorQ)
    benchmark.filepath = dataset_file
    benchmark.embedding_model = embedding_model
    benchmark.llm_model = llm_model
    benchmark.timestamp = timestamp
    benchmark.threshold = threshold if threshold != -1 else None
    benchmark.delta = delta if delta != -1 else None
    benchmark.is_static_threshold = threshold != -1
    benchmark.output_folder_path = path

    benchmark.stats_set_up()
    try:
        benchmark.test_run_benchmark()
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

    for embedding_model, llm_model, dataset, generate_results_only in RUN_COMBINATIONS:
        try:
            dataset_file = os.path.join(datasets_dir, f"{dataset.value}.json")
            logging.info(
                f"Running benchmark for dataset: {dataset}, embedding model: {embedding_model.value[1]}, LLM model: {llm_model.value[1]}\n"
            )
            start_time_llm_model = time.time()

            #####################################################
            ### Baseline: vCache Local
            if Baseline.VCacheLocal in BASELINES_TO_RUN and not generate_results_only:
                for delta in DELTAS:
                    for i in range(0, CONFIDENCE_INTERVALS_ITERATIONS):
                        path = os.path.join(
                            results_dir,
                            dataset.value,
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
                            vectorq_policy=DynamicLocalThresholdPolicy(delta=delta),
                            path=path,
                            dataset_file=dataset_file,
                            embedding_model=embedding_model.value,
                            llm_model=llm_model.value,
                            timestamp=timestamp,
                            delta=delta,
                            threshold=-1,
                        )

            #####################################################
            ### Baseline: vCache Global
            if Baseline.VCacheGlobal in BASELINES_TO_RUN and not generate_results_only:
                for delta in DELTAS:
                    path = os.path.join(
                        results_dir,
                        dataset.value,
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
                        vectorq_policy=DynamicGlobalThresholdPolicy(delta=delta),
                        path=path,
                        dataset_file=dataset_file,
                        embedding_model=embedding_model.value,
                        llm_model=llm_model.value,
                        timestamp=timestamp,
                        delta=delta,
                        threshold=-1,
                    )

            #####################################################
            ### Baseline: Berkeley Embedding
            if (
                Baseline.BerkeleyEmbedding in BASELINES_TO_RUN
                and not generate_results_only
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
                        dataset.value,
                        berkeley_embedding_model.value[1],
                        llm_model.value[1],
                        f"berkeley_embedding_{threshold}",
                    )
                    if os.path.exists(path) and os.listdir(path):
                        continue

                    logging.info(f"Using static threshold: {threshold}")

                    __run_baseline(
                        vectorq_policy=StaticGlobalThresholdPolicy(threshold=threshold),
                        path=path,
                        dataset_file=dataset_file,
                        embedding_model=berkeley_embedding_model.value,
                        llm_model=llm_model.value,
                        timestamp=timestamp,
                        delta=-1,
                        threshold=threshold,
                    )

            #####################################################
            ### Baseline: vCache + Berkeley Embedding
            if (
                Baseline.VCacheBerkeleyEmbedding in BASELINES_TO_RUN
                and not generate_results_only
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
                            dataset.value,
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
                            vectorq_policy=DynamicLocalThresholdPolicy(delta=delta),
                            path=path,
                            dataset_file=dataset_file,
                            embedding_model=berkeley_embedding_model.value,
                            llm_model=llm_model.value,
                            timestamp=timestamp,
                            delta=delta,
                            threshold=-1,
                        )

            #####################################################
            ### Baseline: IID Local
            if Baseline.IID in BASELINES_TO_RUN and not generate_results_only:
                for delta in DELTAS:
                    for i in range(0, CONFIDENCE_INTERVALS_ITERATIONS):
                        path = os.path.join(
                            results_dir,
                            dataset.value,
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
                            vectorq_policy=IIDLocalThresholdPolicy(delta=delta),
                            path=path,
                            dataset_file=dataset_file,
                            embedding_model=embedding_model.value,
                            llm_model=llm_model.value,
                            timestamp=timestamp,
                            delta=delta,
                            threshold=-1,
                        )

            #####################################################
            ### Baseline: GPTCache
            if Baseline.GPTCache in BASELINES_TO_RUN and not generate_results_only:
                for threshold in STATIC_THRESHOLDS:
                    path = os.path.join(
                        results_dir,
                        dataset.value,
                        embedding_model.value[1],
                        llm_model.value[1],
                        f"gptcache_{threshold}",
                    )
                    if os.path.exists(path) and os.listdir(path):
                        continue

                    logging.info(f"Using static threshold: {threshold}")

                    __run_baseline(
                        vectorq_policy=StaticGlobalThresholdPolicy(threshold=threshold),
                        path=path,
                        dataset_file=dataset_file,
                        embedding_model=embedding_model.value,
                        llm_model=llm_model.value,
                        timestamp=timestamp,
                        delta=-1,
                        threshold=threshold,
                    )

            #####################################################
            generate_combined_plots(
                dataset=dataset.value,
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
