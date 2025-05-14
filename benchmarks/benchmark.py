import json
import logging
import os
import time
import unittest
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import ijson
import numpy as np
import torch
from tqdm import tqdm

from benchmarks._plotter_combined import generate_combined_plots
from benchmarks._plotter_individual import generate_individual_plots
from benchmarks.common.comparison import answers_have_same_meaning_static
from vectorq.config import VectorQConfig
from vectorq.inference_engine.strategies.benchmark import (
    BenchmarkInferenceEngine,
)
from vectorq.main import VectorQ
from vectorq.vectorq_core.cache.embedding_engine.strategies.benchmark import (
    BenchmarkEmbeddingEngine,
)
from vectorq.vectorq_core.cache.embedding_store.embedding_metadata_storage import (
    InMemoryEmbeddingMetadataStorage,
)
from vectorq.vectorq_core.cache.embedding_store.embedding_metadata_storage.embedding_metadata_obj import (
    EmbeddingMetadataObj,
)
from vectorq.vectorq_core.cache.embedding_store.vector_db import (
    HNSWLibVectorDB,
    SimilarityMetricType,
)
from vectorq.vectorq_core.similarity_evaluator.strategies.llm_comparison import (
    LLMComparisonSimilarityEvaluator,
)
from vectorq.vectorq_core.similarity_evaluator.strategies.string_comparison import (
    StringComparisonSimilarityEvaluator,
)
from vectorq.vectorq_policy.strategies.dynamic_global_threshold import (
    DynamicGlobalThresholdPolicy,
)
from vectorq.vectorq_policy.strategies.dynamic_local_threshold import (
    DynamicLocalThresholdPolicy,
)
from vectorq.vectorq_policy.strategies.static_global_threshold import (
    StaticGlobalThresholdPolicy,
)
from vectorq.vectorq_policy.vectorq_policy import VectorQPolicy

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
### Parameters #########################################################################################################
########################################################################################################################

# Benchmark Config
MAX_SAMPLES: int = 60000
CONFIDENCE_INTERVALS_ITERATIONS: int = 4
IS_LLM_JUDGE_BENCHMARK: bool = True

EMBEDDING_MODEL_1 = (
    "e5_large_v2",
    "E5_Large_v2",
    "float32",
    1024,
)  # 'Alibaba-NLP/gte-large-en-v1.5'
EMBEDDING_MODEL_1_FT = (
    "e5_large_v2_ft",
    "E5_Large_v2_FT",
    "float32",
    1024,
)
EMBEDDING_MODEL_2 = (
    "emb_gte",
    "E5_Mistral_7B_Instruct",
    "float16",
    4096,
)  # 'intfloat/e5-mistral-7b-instruct'
EMBEDDING_MODEL_2_FT = (
    "emb_gte_ft",
    "E5_Mistral_7B_Instruct",
    "float16",
    4096,
)  # 'intfloat/e5-mistral-7b-instruct'
LARGE_LANGUAGE_MODEL_1 = (
    "response_gpt-4o-mini",
    "Llama_3_8B_Instruct",
    "float16",
    None,
)  # 'meta-llama/Meta-Llama-3-8B-Instruct'
LARGE_LANGUAGE_MODEL_2 = (
    "response_2",
    "Llama_3_70B_Instruct",
    "float16",
    None,
)  # 'meta-llama/Meta-Llama-3-70B-Instruct'
SIMILARITY_STRATEGY = (
    "string_comparison",
    "embedding_comparison",
    "llm_judge_comparison",
)

DATASETS: List[str] = [
    "amazon_instant_video.json",
    "commonsense_qa.json",
    "ecommerce_dataset.json",
    "semantic_prompt_cache_benchmark.json",
    "sem_benchmark_search_queries.json"
]
DATASETS_TO_EXCLUDE: List[str] = [DATASETS[0], DATASETS[1], DATASETS[2], DATASETS[3]]

embedding_models: List[Tuple[str, str, str, int]] = [
    EMBEDDING_MODEL_1,
    EMBEDDING_MODEL_2,
]
llm_models: List[Tuple[str, str, str, int]] = [
    LARGE_LANGUAGE_MODEL_1,
    #LARGE_LANGUAGE_MODEL_2,
]
candidate_strategy: str = SIMILARITY_STRATEGY[0]

static_thresholds = np.array(
    [0.84, 0.86, 0.89, 0.90, 0.91, 0.92, 0.93, 0.94, 0.95, 0.96, 0.97, 0.98, 0.985, 0.99, 0.995, 0.9975, 1.0]
)
deltas = np.array([0.01, 0.0125, 0.015, 0.0175, 0.02, 0.025,0.03, 0.035, 0.04, 0.05, 0.06, 0.07, 0.08]) #np.array([0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1])

# VectorQ Config
MAX_VECTOR_DB_CAPACITY: int = 100000
PLOT_FONT_SIZE: int = 24

SYSTEM_TYPES: List[str] = ["static", "dynamic_local", "dynamic_global", "berkeley_embedding", "vcache_berkeley_embedding", "all"]
SYSTEM_TYPE: str = SYSTEM_TYPES[5]


########################################################################################################################
### Benchmark Class ####################################################################################################
########################################################################################################################
class Benchmark(unittest.TestCase):
    def __init__(self, vectorq: VectorQ):
        super().__init__()
        self.vectorq: VectorQ = vectorq
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
            extra_time = 0
            with open(self.filepath, "rb") as file:
                data_entries = ijson.items(file, "item")

                pbar = tqdm(total=MAX_SAMPLES, desc="Processing entries", disable=True)
                for idx, data_entry in enumerate(data_entries):
                    if idx >= MAX_SAMPLES:
                        break

                    # 1) Get Data
                    task = data_entry["task"]
                    system_prompt = data_entry.get("output_format", "")
                    review_text = data_entry.get("text", "")

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
                    try:
                        label_set_id = data_entry["ID_Set"]
                    except:
                        label_set_id = data_entry["id_set"]

                    # 2.2) VectorQ Inference (With Cache)
                    candidate_embedding: List[float] = data_entry[
                        self.embedding_model[0]
                    ]
                    # has to return cache_response_set_id and nn_response_set_id (this has to be stored in the EmbeddingMetadataObj of the NN)
                    is_cache_hit, cache_response, nn_response, latency_vectorq_logic, cache_response_set_id, nn_response_set_id = (
                        self.get_vectorQ_answer(
                            task=task,
                            review_text=review_text,
                            candidate_embedding=candidate_embedding,
                            label_response=label_response,
                            system_prompt=system_prompt,
                            label_set_id=label_set_id,
                        )
                    )
                    latency_vectorq: float = (
                        latency_vectorq_logic + emb_generation_latency
                    )
                    if not is_cache_hit:
                        latency_vectorq += llm_generation_latency

                    # 3) Update Stats
                    # Pass the two set ids
                    extra_time += self.update_stats(
                        task=task,
                        is_cache_hit=is_cache_hit,
                        label_response=label_response,
                        cache_response=cache_response,
                        nn_response=nn_response,
                        latency_direct=latency_direct,
                        latency_vectorq=latency_vectorq,
                        label_set_id=label_set_id,
                        cache_response_set_id=cache_response_set_id,
                        nn_response_set_id=nn_response_set_id,
                    )

                    pbar.update(1)

                pbar.close()

            print(f"Total extra time: {extra_time}")

        except Exception as e:
            logging.error(f"Error processing benchmark: {e}")
            raise e

        self.dump_results_to_json()
        generate_individual_plots(
            self,
            font_size=PLOT_FONT_SIZE,
            is_static=self.is_static_threshold,
            parameter=self.threshold if self.is_static_threshold else self.delta,
        )
        return extra_time

    ########################################################################################################################
    ### Class Helper Functions #############################################################################################
    ########################################################################################################################
    def update_stats(
        self,
        task: str,
        is_cache_hit: bool,
        label_response: str,
        cache_response: str,
        nn_response: str,
        latency_direct: float,
        latency_vectorq: float,
        label_set_id: str,
        cache_response_set_id: str,
        nn_response_set_id: str,
    ):
        start_time = time.time()
        if is_cache_hit:  # If cache hit, the actual response is the nearest neighbor response (cache_response == nn_response) 
            self.cache_hit_list.append(1)
            self.cache_miss_list.append(0)
            if IS_LLM_JUDGE_BENCHMARK:
                #cache_response_correct: bool = answers_have_same_meaning_static(
                #    label_response, cache_response
                #)
                cache_response_correct: bool = label_set_id == cache_response_set_id
            else:
                cache_response_correct: bool = answers_have_same_meaning_static(
                    label_response, cache_response
                )
            #print(f"\n\n>Task: {task}")
            #print(f"> Cached response: \n{cache_response} \n")
            #print(f"> NN response: \n{nn_response} \n")
            #print(f"> Label response: \n{label_response} \n")
            #print(f"> Cache correct: {cache_response_correct}")
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
            if IS_LLM_JUDGE_BENCHMARK:
                #cache_response_correct: bool = answers_have_same_meaning_llm(
                #    label_response, nn_response
                #)
                cache_response_correct: bool = label_set_id == nn_response_set_id
            else:
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
        return time.time() - start_time

    def get_vectorQ_answer(
        self,
        task: str,
        review_text: str,
        candidate_embedding: List[float],
        label_response: str,
        system_prompt: str,
        label_set_id: str,
    ) -> Tuple[bool, str, str, float, str, str]:
        """
        Returns: Tuple[bool, str, str, float] - [is_cache_hit, cache_response, nn_response, latency_vectorq_logic]
        """
        if isinstance(candidate_embedding, torch.Tensor):
            candidate_embedding = candidate_embedding.tolist()
        elif isinstance(candidate_embedding, np.ndarray):
            candidate_embedding = candidate_embedding.tolist()
        elif isinstance(candidate_embedding, str):
            # Handle case where embedding is a string representation of a list
            try:
                # Remove brackets and split by commas
                if candidate_embedding.startswith('[') and candidate_embedding.endswith(']'):
                    candidate_embedding = candidate_embedding[1:-1]
                candidate_embedding = [float(x.strip()) for x in candidate_embedding.split(',')]
            except Exception as e:
                logging.error(f"Error converting string embedding to list: {e}")
                raise ValueError(f"Could not convert string embedding to list of floats: {candidate_embedding[:100]}...")

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
            is_cache_hit, cache_response, nn_response, cache_response_set_id, nn_response_set_id = (
                self.vectorq.infer_with_cache_info(
                    prompt=vectorQ_prompt,
                    system_prompt=system_prompt,
                    set_id=label_set_id,
                )
            )
        except Exception as e:
            logging.error(
                "Error getting VectorQ answer. Check VectorQ logs for more details."
            )
            raise e

        latency_vectorq_logic = time.time() - latency_vectorq_logic
        return is_cache_hit, cache_response, nn_response, latency_vectorq_logic, cache_response_set_id, nn_response_set_id

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
                "is_static_threshold": bool(self.is_static_threshold),
                "threshold": float(self.threshold) if self.threshold is not None else None,
                "delta": float(self.delta) if self.delta is not None else None,
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
        print(f"Results successfully dumped to {filepath}")


########################################################################################################################
### Helper #############################################################################################################
########################################################################################################################


def __run_baseline(
    vectorq_policy: VectorQPolicy,
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

    vectorq_config: VectorQConfig = VectorQConfig(
        inference_engine=BenchmarkInferenceEngine(),
        embedding_engine=BenchmarkEmbeddingEngine(),
        vector_db=HNSWLibVectorDB(
            similarity_metric_type=SimilarityMetricType.COSINE,
            max_capacity=MAX_VECTOR_DB_CAPACITY,
        ),
        embedding_metadata_storage=InMemoryEmbeddingMetadataStorage(),
        similarity_evaluator=similarity_evaluator,
    )
    vectorQ: VectorQ = VectorQ(vectorq_config, vectorq_policy)

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
    extra_time = benchmark.test_run_benchmark()
    return extra_time


########################################################################################################################
### Main ###############################################################################################################
########################################################################################################################


def main():
    benchmarks_dir = os.path.dirname(os.path.abspath(__file__))
    datasets_dir = os.path.join(benchmarks_dir, "data", "large_scale")
    if not os.path.exists(datasets_dir):
        os.makedirs(datasets_dir, exist_ok=True)
        print(f"Created directory: {datasets_dir}")

    datasets_dir = datasets_dir + "/"
    print(f"Datasets directory: {datasets_dir}")
    datasets = [
        f.split(".")[0]
        for f in os.listdir(datasets_dir)
        if (f.endswith(".json") and (f not in DATASETS_TO_EXCLUDE))
    ]
    print(f"Datasets to be processed: {datasets}")

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")

    for dataset in datasets:
        dataset_file = f"{datasets_dir}{dataset}.json"
        logging.info(f"Running benchmark for dataset: {dataset}\n\n\n")
        start_time_dataset = time.time()
        total_extra_time = 0
        
        for embedding_model in embedding_models:
            logging.info(
                f"Running benchmark for dataset: {dataset}, embedding model: {embedding_model[1]}\n\n"
            )
            start_time_embedding_model = time.time()
            embedding_model_extra_time = 0
            for llm_model in llm_models:
                logging.info(
                    f"Running benchmark for dataset: {dataset}, embedding model: {embedding_model[1]}, LLM model: {llm_model[1]}\n"
                )
                start_time_llm_model = time.time()
                llm_model_extra_time = 0

                # Baseline 1) Dynamic thresholds (VectorQ, Local)
                if SYSTEM_TYPE in ["dynamic_local", "all"]:
                    for delta in deltas:
                        for i in range(0, CONFIDENCE_INTERVALS_ITERATIONS):
                            path = os.path.join(
                                results_dir,
                                dataset,
                                embedding_model[1],
                                llm_model[1],
                                f"vectorq_local_{delta}_run_{i + 1}",
                            )
                            if os.path.exists(path) and os.listdir(path):
                                continue

                            logging.info(
                                f"Using dynamic threshold with delta: {delta}. Run {i + 1} of {CONFIDENCE_INTERVALS_ITERATIONS}"
                            )

                            extra_time = __run_baseline(
                                vectorq_policy=DynamicLocalThresholdPolicy(delta=delta),
                                path=path,
                                dataset_file=dataset_file,
                                embedding_model=embedding_model,
                                llm_model=llm_model,
                                timestamp=timestamp,
                                delta=delta,
                                threshold=-1,
                            )
                            llm_model_extra_time += extra_time
                # Baseline 2) Dynamic thresholds (VectorQ, Global)
                """
                if SYSTEM_TYPE in ["dynamic_global", "all"]:
                    for delta in deltas:
                        for i in range(0, CONFIDENCE_INTERVALS_ITERATIONS):
                            path = os.path.join(
                                results_dir,
                                dataset,
                                embedding_model[1],
                                llm_model[1],
                                f"vectorq_global_{delta}_run_{i + 1}",
                            )
                            if os.path.exists(path) and os.listdir(path):
                                continue

                            logging.info(
                                f"Using dynamic threshold with delta: {delta}. Run {i + 1} of {CONFIDENCE_INTERVALS_ITERATIONS}"
                            )

                            extra_time = __run_baseline(
                                vectorq_policy=DynamicGlobalThresholdPolicy(
                                    delta=delta
                                ),
                                path=path,
                                dataset_file=dataset_file,
                                embedding_model=embedding_model,
                                llm_model=llm_model,
                                timestamp=timestamp,
                                delta=delta,
                                threshold=-1,
                            )
                            llm_model_extra_time += extra_time
                """

                # Baseline 3) Static thresholds
                if SYSTEM_TYPE in ["static", "all"]:
                    for threshold in static_thresholds:
                        path = os.path.join(
                            results_dir,
                            dataset,
                            embedding_model[1],
                            llm_model[1],
                            f"static_{threshold}",
                        )
                        if os.path.exists(path) and os.listdir(path):
                            continue

                        logging.info(f"Using static threshold: {threshold}")

                        extra_time = __run_baseline(
                            vectorq_policy=StaticGlobalThresholdPolicy(
                                threshold=threshold
                            ),
                            path=path,
                            dataset_file=dataset_file,
                            embedding_model=embedding_model,
                            llm_model=llm_model,
                            timestamp=timestamp,
                            delta=-1,
                            threshold=threshold,
                        )
                        llm_model_extra_time += extra_time
                        
                # Baseline 4) Berkely Embedding
                if SYSTEM_TYPE in ["berkeley_embedding", "all"]:
                    for threshold in static_thresholds:
                        if embedding_model[0] == "emb_gte":
                            berkeley_embedding_model = EMBEDDING_MODEL_2_FT
                        elif embedding_model[0] == "emb_e5_large_v2" or embedding_model[0] == "e5_large_v2":
                            berkeley_embedding_model = EMBEDDING_MODEL_1_FT
                        else:
                            print(f"Skipping {embedding_model[0]} for berkeley embedding. Not supported.")
                            continue
                        
                        path = os.path.join(
                            results_dir,
                            dataset,
                            embedding_model[1],
                            llm_model[1],
                            f"berkeley_embedding_{threshold}",
                        )
                        if os.path.exists(path) and os.listdir(path):
                            continue

                        logging.info(f"Using static threshold for berkeley embedding: {threshold}")

                        extra_time = __run_baseline(
                            vectorq_policy=StaticGlobalThresholdPolicy(
                                threshold=threshold
                            ),
                            path=path,
                            dataset_file=dataset_file,
                            embedding_model=berkeley_embedding_model,
                            llm_model=llm_model,
                            timestamp=timestamp,
                            delta=-1,
                            threshold=threshold,
                        )
                        llm_model_extra_time += extra_time
                        
                # Baseline 5) VectorQ with Berkely Embedding
                if SYSTEM_TYPE in ["vcache_berkeley_embedding", "all"]:
                    for delta in deltas:
                        for i in range(0, CONFIDENCE_INTERVALS_ITERATIONS):
                            if embedding_model[0] == "emb_gte":
                                berkeley_embedding_model = EMBEDDING_MODEL_2_FT
                            elif embedding_model[0] == "emb_e5_large_v2" or embedding_model[0] == "e5_large_v2":
                                berkeley_embedding_model = EMBEDDING_MODEL_1_FT
                            else:
                                print(f"Skipping {embedding_model[0]} for berkeley embedding. Not supported.")
                                continue
                        
                            path = os.path.join(
                                results_dir,
                                dataset,
                                embedding_model[1],
                                llm_model[1],
                                f"vcache_berkeley_embedding_{delta}_run_{i + 1}",
                            )
                            if os.path.exists(path) and os.listdir(path):
                                continue

                            logging.info(
                                f"Using dynamic threshold with berkeley embedding and delta: {delta}. Run {i + 1} of {CONFIDENCE_INTERVALS_ITERATIONS}"
                            )

                            extra_time = __run_baseline(
                                vectorq_policy=DynamicLocalThresholdPolicy(delta=delta),
                                path=path,
                                dataset_file=dataset_file,
                                embedding_model=berkeley_embedding_model,
                                llm_model=llm_model,
                                timestamp=timestamp,
                                delta=delta,
                                threshold=-1,
                            )
                            llm_model_extra_time += extra_time

                if SYSTEM_TYPE == "all":
                    generate_combined_plots(
                        dataset=dataset,
                        embedding_model_name=embedding_model[1],
                        llm_model_name=llm_model[1],
                        results_dir=results_dir,
                        timestamp=timestamp,
                        font_size=PLOT_FONT_SIZE,
                    )
                embedding_model_extra_time += llm_model_extra_time
                end_time_llm_model = time.time() - llm_model_extra_time
                logging.info(
                    f"LLM Model Time: {(end_time_llm_model - start_time_llm_model) / 60:.2f} minutes, {(end_time_llm_model - start_time_llm_model) / 3600:.4f} hours"
                )
            total_extra_time += embedding_model_extra_time
            end_time_embedding_model = time.time() - embedding_model_extra_time
            logging.info(
                f"Embedding Model Time: {(end_time_embedding_model - start_time_embedding_model) / 60:.2f} minutes, {(end_time_embedding_model - start_time_embedding_model) / 3600:.4f} hours"
            )
        end_time_dataset = time.time() - total_extra_time
        logging.info(
            f"Dataset Time: {(end_time_dataset - start_time_dataset) / 60:.2f} minutes, {(end_time_dataset - start_time_dataset) / 3600:.4f} hours"
        )

    print("All benchmarks completed!")


if __name__ == "__main__":
    main()
