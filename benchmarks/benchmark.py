import asyncio
import json
import logging
import os
import time
import unittest
from datetime import datetime
from typing import Dict, List, Tuple

import ijson
import numpy as np
import torch
from tqdm import tqdm

from benchmarks._plotter_combined import (
    plot_cache_hit_latency_vs_size_comparison,
    plot_duration_comparison,
    plot_duration_vs_error_rate,
    plot_hit_rate_vs_error,
    plot_hit_rate_vs_latency,
    plot_precision_vs_recall,
    plot_roc_curve,
)
from benchmarks._plotter_individual import (
    plot_accuracy,
    plot_bayesian_decision_boundary,
    plot_cache_hit_latency_vs_size,
    plot_cache_size,
    plot_combined_thresholds_and_posteriors,
    plot_duration_trend,
    plot_error_rate_absolute,
    plot_error_rate_relative,
    plot_precision,
    plot_recall,
    plot_reuse_rate,
)
from benchmarks.common.comparison import answers_have_same_meaning_static
from vectorq.config import VectorQConfig
from vectorq.main import VectorQ, VectorQBenchmark
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
from vectorq.vectorq_core.similarity_evaluator.strategies.string_comparison import (
    StringComparisonSimilarityEvaluator,
)
from vectorq.vectorq_core.vectorq_policy.strategies.bayesian import (
    VectorQBayesianPolicy,
)

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
MAX_SAMPLES = 20000
CONFIDENCE_INTERVALS_ITERATIONS = 1
EMBEDDING_MODEL_1 = (
    "embedding_1",
    "GteLargeENv1_5",
    "float32",
    1024,
)  # 'Alibaba-NLP/gte-large-en-v1.5'
EMBEDDING_MODEL_2 = (
    "embedding_2",
    "E5_Mistral_7B_Instruct",
    "float16",
    4096,
)  # 'intfloat/e5-mistral-7b-instruct'
LARGE_LANGUAGE_MODEL_1 = (
    "response_1",
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

embedding_models = [EMBEDDING_MODEL_1, EMBEDDING_MODEL_2]
llm_models = [LARGE_LANGUAGE_MODEL_1, LARGE_LANGUAGE_MODEL_2]
candidate_strategy = SIMILARITY_STRATEGY[0]

static_thresholds = np.array(
    [0.74, 0.76, 0.78, 0.8, 0.825, 0.85, 0.875, 0.9, 0.92, 0.94, 0.96]
)
deltas = np.array([0.025, 0.05, 0.075, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35])

# VectorQ Config
MAX_VECTOR_DB_CAPACITY = 100000

THRESHOLD_TYPES = ["static", "dynamic", "both"]
THRESHOLD_TYPE = THRESHOLD_TYPES[1]


########################################################################################################################
### Benchmark Class ####################################################################################################
########################################################################################################################
class Benchmark(unittest.TestCase):
    def __init__(self, max_samples, vectorq: VectorQ):
        super().__init__()
        self.max_samples = max_samples
        self.vectorq: VectorQ = vectorq
        self.is_dynamic_threshold = None
        self.threshold = None
        self.rnd_num_lb = None
        self.rnd_num_ub = None
        self.delta = None
        self.embedding_model = None
        self.llm_model = None
        self.filepath = None
        self.output_folder_path = None
        self.timestamp = None
        self.field_to_extract = "text"
        self.step_size = 1
        self.current_data_entry = None
        self.tasks = set()
        self.task = None
        self.output_format = None
        self.candidate_strategy = None

    def setUp(self):
        """Initialize VectorQ with the necessary configuration."""
        # Initialize counters and result stores first
        self.correct_answers = 0
        self.incorrect_answers = 0
        self.total_reused = 0
        self.total_duration_direct = 0
        self.total_duration_vectorq = 0
        self.incorrect_answers_in_step_size = 0
        self.true_positive_counter = 0
        self.false_positive_counter = 0
        self.true_negative_counter = 0
        self.false_negative_counter = 0
        self.cache_size_mb = 0

        # Lists to store results
        self.sample_sizes = []
        self.error_rates_relative_to_reused_answers = []
        self.error_rates_relative_to_step_size = []
        self.error_rates_absolute = []
        self.total_reused_list = []
        self.relative_reuse_rates = []
        self.inference_time_direct_step_size = []
        self.inference_time_vectorq_step_size = []  # latency of vectorQ/GPTCache request
        self.total_duration_direct_list = []
        self.total_duration_vectorq_list = []
        self.answers_reused = []
        self.true_positive_list = []
        self.false_positive_list = []
        self.true_negative_list = []
        self.false_negative_list = []
        self.precision_list = []
        self.recall_list = []
        self.accuracy_list = []
        self.cache_size_list = []
        self.observations: Dict[int, List[Tuple[float, int]]] = {}
        self.gammas: Dict[int, float] = {}
        self.correct_x: Dict[int, List[float]] = {}
        self.incorrect_x: Dict[int, List[float]] = {}
        self.correct_y: Dict[int, List[float]] = {}
        self.incorrect_y: Dict[int, List[float]] = {}
        self.posteriors: Dict[int, List[float]] = {}

        if self.output_folder_path and not os.path.exists(self.output_folder_path):
            os.makedirs(self.output_folder_path)

    async def test_run_benchmark(self):
        if not self.filepath or not self.embedding_model or not self.llm_model:
            raise ValueError(
                f"Required parameters not set: filepath: {self.filepath}, embedding_model: {self.embedding_model}, or llm_model: {self.llm_model}"
            )

        try:
            with open(self.filepath, "rb") as file:
                data_entries = ijson.items(file, "item")

                pbar = tqdm(total=self.max_samples, desc="Processing entries")
                for idx, data_entry in enumerate(data_entries):
                    if idx >= self.max_samples:
                        break

                    self.task = data_entry["task"]
                    self.tasks.add(self.task)
                    self.output_format = data_entry["output_format"]
                    self.current_data_entry = data_entry
                    review_text = data_entry[self.field_to_extract]

                    emb_generation_latency = float(
                        data_entry[self.embedding_model[0] + "_lat"]
                    )
                    llm_generation_latency = float(
                        data_entry[self.llm_model[0] + "_lat"]
                    )

                    start_time_direct = time.time()
                    direct_answer = data_entry[self.llm_model[0]]
                    duration_direct = (
                        time.time() - start_time_direct + llm_generation_latency
                    )
                    self.total_duration_direct += duration_direct

                    row_embedding = data_entry[self.embedding_model[0]]
                    start_time_vectorq = time.time()
                    vectorQ_answer, cache_hit = await self.get_vectorQ_answer(
                        review_text, row_embedding
                    )
                    duration_vectorq = (
                        time.time() - start_time_vectorq + emb_generation_latency
                    )
                    if not cache_hit:
                        duration_vectorq += llm_generation_latency
                    self.total_duration_vectorq += duration_vectorq

                    if cache_hit:
                        self.total_reused += 1

                    if cache_hit:
                        if answers_have_same_meaning_static(
                            self.task, direct_answer, vectorQ_answer
                        ):
                            self.true_positive_counter += 1  # Correctly reused
                        else:
                            self.false_positive_counter += 1  # Incorrectly reused
                    else:
                        if answers_have_same_meaning_static(
                            self.task, direct_answer, vectorQ_answer
                        ):
                            self.false_negative_counter += (
                                1  # Should have reused but didn't
                            )
                        else:
                            self.true_negative_counter += 1  # Correctly didn't reuse

                    if not cache_hit or answers_have_same_meaning_static(
                        self.task, direct_answer, vectorQ_answer
                    ):
                        self.correct_answers += 1
                    else:
                        self.incorrect_answers += 1
                        self.incorrect_answers_in_step_size += 1

                    self.append_results(
                        idx, duration_direct, duration_vectorq, cache_hit, direct_answer
                    )
                    self.incorrect_answers_in_step_size = 0

                    pbar.update(1)

                pbar.close()

        except Exception as e:
            logging.error(f"Error processing benchmark: {e}")
            raise e

        self.current_data_entry = None

        plot_error_rate_relative(self)
        plot_error_rate_absolute(self)
        plot_reuse_rate(self)
        plot_duration_trend(self)
        plot_precision(self)
        plot_recall(self)
        plot_accuracy(self)
        plot_cache_size(self)
        plot_cache_hit_latency_vs_size(self)

        self.dump_results_to_json()
        plot_combined_thresholds_and_posteriors(self)
        plot_bayesian_decision_boundary(self)

    ########################################################################################################################
    ### Class Helper Functions #############################################################################################
    ########################################################################################################################

    def append_results(
        self, idx, duration_direct, duration_vectorq, answer_reused, direct_answer
    ):
        error_rate_relative = (
            (self.incorrect_answers / self.total_reused * 100)
            if self.total_reused > 0
            else 0
        )
        error_rate_absolute = (self.incorrect_answers / idx * 100) if idx > 0 else 0
        total_reused_rate = (self.total_reused / idx * 100) if idx > 0 else 0

        if len(self.total_reused_list) == 0:
            prev_reused = 0
        else:
            prev_reused = self.total_reused_list[-1]
        reused_this_interval = self.total_reused - prev_reused
        relative_reuse_rate = (
            (reused_this_interval / self.step_size) * 100 if self.step_size > 0 else 0
        )
        relative_error_rate_to_step_size = (
            (self.incorrect_answers_in_step_size / reused_this_interval) * 100
            if reused_this_interval > 0
            else 0
        )

        self.sample_sizes.append(idx)
        self.error_rates_relative_to_reused_answers.append(error_rate_relative)
        self.error_rates_relative_to_step_size.append(relative_error_rate_to_step_size)
        self.error_rates_absolute.append(error_rate_absolute)
        self.total_reused_list.append(self.total_reused)
        self.relative_reuse_rates.append(relative_reuse_rate)
        self.inference_time_direct_step_size.append(duration_direct)
        self.inference_time_vectorq_step_size.append(duration_vectorq)
        self.total_duration_direct_list.append(self.total_duration_direct)
        self.total_duration_vectorq_list.append(self.total_duration_vectorq)
        self.answers_reused.append(answer_reused)
        self.true_positive_list.append(self.true_positive_counter)
        self.false_positive_list.append(self.false_positive_counter)
        self.true_negative_list.append(self.true_negative_counter)
        self.false_negative_list.append(self.false_negative_counter)

        precision = (
            self.true_positive_counter
            / (self.true_positive_counter + self.false_positive_counter)
            if (self.true_positive_counter + self.false_positive_counter) > 0
            else 0
        )
        recall = (
            self.true_positive_counter
            / (self.true_positive_counter + self.false_negative_counter)
            if (self.true_positive_counter + self.false_negative_counter) > 0
            else 0
        )
        accuracy = (
            (self.true_positive_counter + self.true_negative_counter)
            / (
                self.true_positive_counter
                + self.false_positive_counter
                + self.true_negative_counter
                + self.false_negative_counter
            )
            if (
                self.true_positive_counter
                + self.false_positive_counter
                + self.true_negative_counter
                + self.false_negative_counter
            )
            > 0
            else 0
        )
        self.precision_list.append(precision)
        self.recall_list.append(recall)
        self.accuracy_list.append(accuracy)

        # Update cache size if this was a cache miss (not reused -> got added to cache)
        if not answer_reused:
            embedding_dim = self.embedding_model[3]
            embedding_precision = self.embedding_model[2]

            entry_size_mb = calculate_cache_entry_size_mb(
                embedding_dim, embedding_precision, direct_answer
            )
            self.cache_size_mb += entry_size_mb
        self.cache_size_list.append(self.cache_size_mb)

        if idx == 0:
            logging.info(
                f"VectorQ Config (Client) | Task: {self.task}, Output Format: {self.output_format}, Embedding Model: {self.embedding_model[1]}, LLM Model: {self.llm_model[1]}, Is Dynamic Threshold: {self.is_dynamic_threshold}, Threshold: {self.threshold}, Rnd_num_ub: {self.rnd_num_ub}"
            )

        if (idx + 1) % 500 == 0:
            logging.info(
                f"Sample Size: {idx + 1}, Total Reused: {self.total_reused}, Incorrect Answers: {self.incorrect_answers}, Absolute Error Rate: {error_rate_absolute:.2f}%, Relative Error Rate (Reused Answers): {error_rate_relative:.2f}%, Relative Error Rate (Step Size): {relative_error_rate_to_step_size:.2f}%, Total Reused Rate: {total_reused_rate:.2f}%, Relative Reused Rate (Step Size): {relative_reuse_rate:.2f}%, Cache Size: {self.cache_size_mb:.2f} MB"
            )

    async def get_vectorQ_answer(self, review_text, row_embedding):
        try:
            row_embedding = self.current_data_entry[self.embedding_model[0]]
        except json.JSONDecodeError as e:
            logging.error(
                f"Could not find embedding in the dataset for {self.embedding_model[0]}"
            )
            raise e

        try:
            candidate_response = self.current_data_entry[self.llm_model[0]]
        except json.JSONDecodeError as e:
            logging.error(
                f"Could not find LLM response in the dataset for {self.llm_model[0]}"
            )
            raise e

        if isinstance(row_embedding, torch.Tensor):
            row_embedding = row_embedding.tolist()
        elif isinstance(row_embedding, np.ndarray):
            row_embedding = row_embedding.tolist()

        # Ensure all values are standard Python floats, not Decimal objects
        if isinstance(row_embedding, list):
            row_embedding = [
                float(val) if hasattr(val, "__float__") else val
                for val in row_embedding
            ]

        vectorQ_benchmark = VectorQBenchmark(
            candidate_embedding=row_embedding, candidate_response=candidate_response
        )

        vectorQ_prompt = f"{self.task} {review_text}"
        try:
            vectorQ_response, cache_hit = await self.vectorq.create(
                prompt=vectorQ_prompt,
                output_format=self.output_format,
                benchmark=vectorQ_benchmark,
            )
        except Exception as e:
            logging.error(
                "Error getting VectorQ answer. Check VectorQ logs for more details."
            )
            raise e

        return vectorQ_response, cache_hit

    def dump_results_to_json(self):
        # VectorQ Bayesian Policy ########################
        observations = {}
        gammas = {}
        ##################################################

        # VectorQ Heuristic Policy #######################
        correct_x = {}
        incorrect_x = {}
        correct_y = {}
        incorrect_y = {}
        posteriors = {}

        metadata_objects: List[EmbeddingMetadataObj] = (
            self.vectorq.core.cache.get_all_embedding_metadata_objects()
        )

        for metadata_object in metadata_objects:
            observations[metadata_object.embedding_id] = metadata_object.observations
            gammas[metadata_object.embedding_id] = metadata_object.gamma

            correct_x[metadata_object.embedding_id] = (
                metadata_object.correct_similarities
            )
            incorrect_x[metadata_object.embedding_id] = (
                metadata_object.incorrect_similarities
            )
            correct_y[metadata_object.embedding_id] = [
                1 for _ in metadata_object.correct_similarities
            ]
            incorrect_y[metadata_object.embedding_id] = [
                0 for _ in metadata_object.incorrect_similarities
            ]
            posteriors[metadata_object.embedding_id] = metadata_object.posteriors

        self.observations = observations
        self.gammas = gammas

        self.correct_x = correct_x
        self.incorrect_x = incorrect_x
        self.correct_y = correct_y
        self.incorrect_y = incorrect_y
        self.posteriors = posteriors

        data = {
            "config": {
                "filepath": self.filepath,
                "tasks": list(self.tasks),
                "embedding_model": self.embedding_model,
                "is_dynamic_threshold": self.is_dynamic_threshold,
                "threshold": self.threshold,
                "rnd_num_lb": self.rnd_num_lb,
                "rnd_num_ub": self.rnd_num_ub,
                "delta": self.delta,
            },
            "sample_sizes": self.sample_sizes,
            "error_rates_relative_to_reused_answers": self.error_rates_relative_to_reused_answers,
            "error_rates_relative_to_step_size": self.error_rates_relative_to_step_size,
            "error_rates_absolute": self.error_rates_absolute,
            "total_reused_list": self.total_reused_list,
            "relative_reuse_rates": self.relative_reuse_rates,
            "inference_time_direct_step_size": self.inference_time_direct_step_size,
            "inference_time_vectorq_step_size": self.inference_time_vectorq_step_size,
            "total_duration_direct_list": self.total_duration_direct_list,
            "total_duration_vectorq_list": self.total_duration_vectorq_list,
            "answers_reused": self.answers_reused,
            "observations": self.observations,
            "gammas": self.gammas,
            "correct_x": self.correct_x,
            "incorrect_x": self.incorrect_x,
            "correct_y": self.correct_y,
            "incorrect_y": self.incorrect_y,
            "posteriors": self.posteriors,
            "true_positive_list": self.true_positive_list,
            "false_positive_list": self.false_positive_list,
            "true_negative_list": self.true_negative_list,
            "false_negative_list": self.false_negative_list,
            "precision_list": self.precision_list,
            "precision": self.precision_list[-1],
            "recall_list": self.recall_list,
            "recall": self.recall_list[-1],
            "accuracy_list": self.accuracy_list,
            "accuracy": self.accuracy_list[-1],
            "cache_size_list": self.cache_size_list,
            "cache_size": self.cache_size_list[-1],
        }

        filepath = self.output_folder_path + f"/results_{self.timestamp}.json"
        with open(filepath, "w") as json_file:
            json.dump(data, json_file, indent=4)
        print(f"Results successfully dumped to {filepath}")


########################################################################################################################
### Helper Functions ###################################################################################################
########################################################################################################################


def calculate_cache_entry_size_mb(embedding_dim, embedding_precision, cached_response):
    bytes_per_float = (
        2 if embedding_precision == "float16" else 4
    )  # float16 = 2 bytes, float32 = 4 bytes
    embedding_size_bytes = embedding_dim * bytes_per_float
    response_size_bytes = len(cached_response.encode("utf-8"))
    posterior = np.linspace(0, 1, 500)
    metadata_size_bytes = posterior.size * posterior.itemsize
    total_size_mb = (
        embedding_size_bytes + response_size_bytes + metadata_size_bytes
    ) / (1024 * 1024)
    return total_size_mb


def compare_static_vs_dynamic(
    dataset, embedding_model_name, llm_model_name, timestamp, results_dir=None
):
    # Use the global results_dir if none is provided
    if results_dir is None:
        results_dir = os.path.join(repo_root, "results")
    print(
        f"\n\nComparing static vs dynamic thresholds for {dataset}, {embedding_model_name}, {llm_model_name}"
    )
    plot_hit_rate_vs_error(
        dataset, embedding_model_name, llm_model_name, timestamp, results_dir
    )
    plot_precision_vs_recall(
        dataset, embedding_model_name, llm_model_name, timestamp, results_dir
    )
    plot_duration_comparison(
        dataset, embedding_model_name, llm_model_name, timestamp, results_dir
    )
    plot_duration_vs_error_rate(
        dataset, embedding_model_name, llm_model_name, timestamp, results_dir
    )
    plot_cache_hit_latency_vs_size_comparison(
        dataset, embedding_model_name, llm_model_name, timestamp, results_dir
    )
    plot_hit_rate_vs_latency(
        dataset, embedding_model_name, llm_model_name, timestamp, results_dir
    )
    plot_roc_curve(
        dataset, embedding_model_name, llm_model_name, timestamp, results_dir
    )


########################################################################################################################
### Main ###############################################################################################################
########################################################################################################################


async def main():
    benchmarks_dir = os.path.dirname(os.path.abspath(__file__))
    datasets_dir = os.path.join(benchmarks_dir, "data", "large_scale")
    if not os.path.exists(datasets_dir):
        os.makedirs(datasets_dir, exist_ok=True)
        print(f"Created directory: {datasets_dir}")

    datasets_dir = datasets_dir + "/"
    datasets = [
        f.split(".")[0]
        for f in os.listdir(datasets_dir)
        if (f.endswith(".json") and (f.startswith("sem") or f.startswith("ama")))
    ]
    print(f"Datasets to be processed: {datasets}")

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")

    for dataset in datasets:
        dataset_file = f"{datasets_dir}{dataset}.json"
        logging.info(f"Running benchmark for dataset: {dataset}\n\n\n")
        start_time_dataset = time.time()

        for embedding_model in embedding_models:
            logging.info(
                f"Running benchmark for dataset: {dataset}, embedding model: {embedding_model[1]}\n\n"
            )
            start_time_embedding_model = time.time()

            for llm_model in llm_models:
                logging.info(
                    f"Running benchmark for dataset: {dataset}, embedding model: {embedding_model[1]}, LLM model: {llm_model[1]}\n"
                )
                start_time_llm_model = time.time()

                # Static thresholds
                if THRESHOLD_TYPE in ["static", "both"]:
                    for threshold in static_thresholds:
                        print(f"Using static threshold: {threshold}")

                        config = VectorQConfig(
                            enable_cache=True,
                            is_static_threshold=True,
                            static_threshold=threshold,
                            vector_db=HNSWLibVectorDB(
                                similarity_metric_type=SimilarityMetricType.COSINE,
                                max_capacity=MAX_VECTOR_DB_CAPACITY,
                            ),
                            embedding_metadata_storage=InMemoryEmbeddingMetadataStorage(),
                            similarity_evaluator=StringComparisonSimilarityEvaluator(),
                        )
                        vectorQ: VectorQ = VectorQ(config)

                        benchmark = Benchmark(MAX_SAMPLES, vectorQ)
                        # Set required parameters
                        benchmark.filepath = dataset_file
                        benchmark.embedding_model = embedding_model
                        benchmark.llm_model = llm_model
                        benchmark.is_dynamic_threshold = False
                        benchmark.threshold = round(threshold, 2)
                        benchmark.timestamp = timestamp
                        benchmark.candidate_strategy = candidate_strategy
                        benchmark.output_folder_path = os.path.join(
                            results_dir,
                            dataset,
                            embedding_model[1],
                            llm_model[1],
                            f"static_{threshold}",
                        )

                        # Run the benchmark
                        benchmark.setUp()
                        await benchmark.test_run_benchmark()
                        await vectorQ.shutdown()

                # Dynamic thresholds (VectorQ)
                if THRESHOLD_TYPE in ["dynamic", "both"]:
                    for delta in deltas:
                        for i in range(0, CONFIDENCE_INTERVALS_ITERATIONS):
                            print(
                                f"Using dynamic threshold with delta: {delta}. Run {i + 1} of {CONFIDENCE_INTERVALS_ITERATIONS}"
                            )

                            config = VectorQConfig(
                                enable_cache=True,
                                is_static_threshold=False,
                                vector_db=HNSWLibVectorDB(
                                    similarity_metric_type=SimilarityMetricType.COSINE,
                                    max_capacity=MAX_VECTOR_DB_CAPACITY,
                                ),
                                embedding_metadata_storage=InMemoryEmbeddingMetadataStorage(),
                                similarity_evaluator=StringComparisonSimilarityEvaluator(),
                                vectorq_policy=VectorQBayesianPolicy(delta=delta),
                            )
                            vectorQ: VectorQ = VectorQ(config)

                            benchmark = Benchmark(MAX_SAMPLES, vectorQ)
                            # Set required parameters
                            benchmark.filepath = dataset_file
                            benchmark.embedding_model = embedding_model
                            benchmark.llm_model = llm_model
                            benchmark.is_dynamic_threshold = True
                            benchmark.threshold = None
                            benchmark.delta = delta
                            benchmark.timestamp = timestamp
                            benchmark.candidate_strategy = candidate_strategy
                            benchmark.output_folder_path = os.path.join(
                                results_dir,
                                dataset,
                                embedding_model[1],
                                llm_model[1],
                                f"vectorq_{delta}_run_{i + 1}",
                            )

                            # Run the benchmark
                            benchmark.setUp()
                            await benchmark.test_run_benchmark()
                            await vectorQ.shutdown()

                if THRESHOLD_TYPE == "both":
                    compare_static_vs_dynamic(
                        dataset, embedding_model[1], llm_model[1], timestamp
                    )

                end_time_llm_model = time.time()
                logging.info(
                    f"LLM Model Time: {(end_time_llm_model - start_time_llm_model) / 60:.2f} minutes, {(end_time_llm_model - start_time_llm_model) / 3600:.4f} hours"
                )
            end_time_embedding_model = time.time()
            logging.info(
                f"Embedding Model Time: {(end_time_embedding_model - start_time_embedding_model) / 60:.2f} minutes, {(end_time_embedding_model - start_time_embedding_model) / 3600:.4f} hours"
            )
        end_time_dataset = time.time()
        logging.info(
            f"Dataset Time: {(end_time_dataset - start_time_dataset) / 60:.2f} minutes, {(end_time_dataset - start_time_dataset) / 3600:.4f} hours"
        )

    print("All benchmarks completed!")


if __name__ == "__main__":
    asyncio.run(main())
