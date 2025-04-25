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

from benchmarks._plotter_combined import generate_combined_plots
from benchmarks._plotter_individual import generate_individual_plots
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
MAX_SAMPLES: int = 2000
CONFIDENCE_INTERVALS_ITERATIONS: int = 1
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

DATASETS: List[str] = [
    "amazon_instant_video.json",
    "commonsense_qa.json",
    "ecommerce_dataset.json",
    "semantic_prompt_cache_benchmark.json",
]
DATASETS_TO_EXCLUDE: List[str] = [DATASETS[1]]

embedding_models: List[Tuple[str, str, str, int]] = [
    EMBEDDING_MODEL_1,
    EMBEDDING_MODEL_2,
]
llm_models: List[Tuple[str, str, str, int]] = [
    LARGE_LANGUAGE_MODEL_1,
    LARGE_LANGUAGE_MODEL_2,
]
candidate_strategy: str = SIMILARITY_STRATEGY[0]

static_thresholds = np.array(
    [0.74, 0.76, 0.78, 0.8, 0.825, 0.85, 0.875, 0.9, 0.92, 0.94, 0.96]
)
deltas = np.array([0.025, 0.05, 0.075, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35])

# VectorQ Config
MAX_VECTOR_DB_CAPACITY: int = 100000
PLOT_FONT_SIZE: int = 24

THRESHOLD_TYPES: List[str] = ["static", "dynamic", "both"]
THRESHOLD_TYPE: str = THRESHOLD_TYPES[1]


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
        self.cache_hit_counter: int = 0
        self.cache_hit_acc_list: List[int] = []
        self.cache_miss_counter: int = 0
        self.cache_miss_acc_list: List[int] = []
        self.true_positive_counter: int = 0
        self.true_positive_acc_list: List[int] = []
        self.false_positive_counter: int = 0
        self.false_positive_acc_list: List[int] = []
        self.true_negative_counter: int = 0
        self.true_negative_acc_list: List[int] = []
        self.false_negative_counter: int = 0
        self.false_negative_acc_list: List[int] = []
        self.latency_direct_list: List[float] = []
        self.latency_vectorq_list: List[float] = []
        self.observations_dict: Dict[str, Dict[str, float]] = {}
        self.gammas_dict: Dict[str, float] = {}

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

                pbar = tqdm(total=MAX_SAMPLES, desc="Processing entries")
                for idx, data_entry in enumerate(data_entries):
                    if idx >= MAX_SAMPLES:
                        break

                    task = data_entry["task"]
                    output_format = data_entry["output_format"]
                    review_text = data_entry["text"]

                    emb_generation_latency: float = float(
                        data_entry[self.embedding_model[0] + "_lat"]
                    )
                    llm_generation_latency: float = float(
                        data_entry[self.llm_model[0] + "_lat"]
                    )

                    direct_answer: str = data_entry[self.llm_model[0]]
                    latency_direct: float = llm_generation_latency

                    row_embedding: List[float] = data_entry[self.embedding_model[0]]
                    start_time_vectorq: float = time.time()
                    vectorQ_answer, cache_hit = await self.get_vectorQ_answer(
                        data_entry, task, review_text, row_embedding, output_format
                    )
                    latency_vectorq: float = (
                        time.time() - start_time_vectorq + emb_generation_latency
                    )

                    if not cache_hit:
                        latency_vectorq += llm_generation_latency

                    if cache_hit:
                        self.cache_hit_counter += 1
                        if answers_have_same_meaning_static(
                            task, direct_answer, vectorQ_answer
                        ):
                            self.true_positive_counter += 1  # Correctly reused
                        else:
                            self.false_positive_counter += 1  # Incorrectly reused
                    else:
                        self.cache_miss_counter += 1
                        if answers_have_same_meaning_static(
                            task, direct_answer, vectorQ_answer
                        ):
                            self.false_negative_counter += (
                                1  # Should've reused but didn't
                            )
                        else:
                            self.true_negative_counter += 1  # Correctly didn't reuse

                    self.append_results(latency_direct, latency_vectorq)
                    pbar.update(1)

                pbar.close()

        except Exception as e:
            logging.error(f"Error processing benchmark: {e}")
            raise e

        self.dump_results_to_json()
        generate_individual_plots(self, font_size=PLOT_FONT_SIZE)

    ########################################################################################################################
    ### Class Helper Functions #############################################################################################
    ########################################################################################################################

    def append_results(self, latency_direct: float, latency_vectorq: float):
        self.cache_hit_acc_list.append(self.cache_hit_counter)
        self.cache_miss_acc_list.append(self.cache_miss_counter)
        self.true_positive_acc_list.append(self.true_positive_counter)
        self.false_positive_acc_list.append(self.false_positive_counter)
        self.true_negative_acc_list.append(self.true_negative_counter)
        self.false_negative_acc_list.append(self.false_negative_counter)
        self.latency_direct_list.append(latency_direct)
        self.latency_vectorq_list.append(latency_vectorq)

    async def get_vectorQ_answer(
        self,
        data_entry: Dict,
        task: str,
        review_text: str,
        row_embedding: List[float],
        output_format: str,
    ):
        try:
            row_embedding = data_entry[self.embedding_model[0]]
        except json.JSONDecodeError as e:
            logging.error(
                f"Could not find embedding in the dataset for {self.embedding_model[0]}"
            )
            raise e

        try:
            candidate_response = data_entry[self.llm_model[0]]
        except json.JSONDecodeError as e:
            logging.error(
                f"Could not find LLM response in the dataset for {self.llm_model[0]}"
            )
            raise e

        if isinstance(row_embedding, torch.Tensor):
            row_embedding = row_embedding.tolist()
        elif isinstance(row_embedding, np.ndarray):
            row_embedding = row_embedding.tolist()

        if isinstance(row_embedding, list):
            row_embedding = [
                float(val) if hasattr(val, "__float__") else val
                for val in row_embedding
            ]

        vectorQ_benchmark = VectorQBenchmark(
            candidate_embedding=row_embedding, candidate_response=candidate_response
        )

        vectorQ_prompt = f"{task} {review_text}"
        try:
            vectorQ_response, cache_hit = await self.vectorq.create(
                prompt=vectorQ_prompt,
                output_format=output_format,
                benchmark=vectorQ_benchmark,
            )
        except Exception as e:
            logging.error(
                "Error getting VectorQ answer. Check VectorQ logs for more details."
            )
            raise e

        return vectorQ_response, cache_hit

    def dump_results_to_json(self):
        observations_dict = {}
        gammas_dict = {}

        metadata_objects: List[EmbeddingMetadataObj] = (
            self.vectorq.core.cache.get_all_embedding_metadata_objects()
        )

        for metadata_object in metadata_objects:
            observations_dict[metadata_object.embedding_id] = (
                metadata_object.observations
            )
            gammas_dict[metadata_object.embedding_id] = metadata_object.gamma

        self.observations_dict = observations_dict
        self.gammas_dict = gammas_dict

        data = {
            "config": {
                "filepath": self.filepath,
                "embedding_model": self.embedding_model,
                "is_static_threshold": self.is_static_threshold,
                "threshold": self.threshold,
                "delta": self.delta,
            },
            "cache_hit_acc_list": self.cache_hit_acc_list,
            "cache_miss_acc_list": self.cache_miss_acc_list,
            "true_positive_acc_list": self.true_positive_acc_list,
            "false_positive_acc_list": self.false_positive_acc_list,
            "true_negative_acc_list": self.true_negative_acc_list,
            "false_negative_acc_list": self.false_negative_acc_list,
            "latency_direct_list": self.latency_direct_list,
            "latency_vectorq_list": self.latency_vectorq_list,
            "observations_dict": self.observations_dict,
            "gammas_dict": self.gammas_dict,
        }

        filepath = self.output_folder_path + f"/results_{self.timestamp}.json"
        with open(filepath, "w") as json_file:
            json.dump(data, json_file, indent=4)
        print(f"Results successfully dumped to {filepath}")


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
        if (f.endswith(".json") and (f not in DATASETS_TO_EXCLUDE))
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

                        benchmark = Benchmark(vectorQ)
                        # Set required parameters
                        benchmark.filepath = dataset_file
                        benchmark.embedding_model = embedding_model
                        benchmark.llm_model = llm_model
                        benchmark.timestamp = timestamp
                        benchmark.threshold = threshold
                        benchmark.delta = -1
                        benchmark.is_static_threshold = True
                        benchmark.output_folder_path = os.path.join(
                            results_dir,
                            dataset,
                            embedding_model[1],
                            llm_model[1],
                            f"static_{threshold}",
                        )

                        # Run the benchmark
                        benchmark.stats_set_up()
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

                            benchmark = Benchmark(vectorQ)
                            # Set required parameters
                            benchmark.filepath = dataset_file
                            benchmark.embedding_model = embedding_model
                            benchmark.llm_model = llm_model
                            benchmark.timestamp = timestamp
                            benchmark.threshold = -1
                            benchmark.delta = delta
                            benchmark.is_static_threshold = False
                            benchmark.output_folder_path = os.path.join(
                                results_dir,
                                dataset,
                                embedding_model[1],
                                llm_model[1],
                                f"vectorq_{delta}_run_{i + 1}",
                            )

                            # Run the benchmark
                            benchmark.stats_set_up()
                            await benchmark.test_run_benchmark()
                            await vectorQ.shutdown()

                if THRESHOLD_TYPE == "both":
                    generate_combined_plots(
                        dataset=dataset,
                        embedding_model_name=embedding_model[1],
                        llm_model_name=llm_model[1],
                        results_dir=results_dir,
                        timestamp=timestamp,
                        font_size=PLOT_FONT_SIZE,
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
