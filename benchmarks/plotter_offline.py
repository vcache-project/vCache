# import json
# import os
# from datetime import datetime

# from benchmarks._plotter_combined_old import (
#     plot_cache_hit_latency_vs_size_comparison,
#     plot_duration_comparison,
#     plot_duration_vs_error_rate,
#     plot_hit_rate_vs_error,
#     plot_hit_rate_vs_latency,
#     plot_precision_vs_recall,
#     plot_roc_curve,
# )
# from benchmarks._plotter_individual_old import (
#     plot_accuracy,
#     plot_cache_hit_latency_vs_size,
#     plot_cache_size,
#     plot_duration_trend,
#     plot_error_rate_absolute,
#     plot_error_rate_relative,
#     plot_precision,
#     plot_recall,
#     plot_reuse_rate,
# )
# from benchmarks.benchmark_old import Benchmark


# ########################################################################################################################
# ########################################################################################################################
# def _generate_combination_plots(
#     dataset, embedding_model_name, llm_model_name, timestamp, results_dir
# ):
#     plot_roc_curve(
#         dataset, embedding_model_name, llm_model_name, timestamp, results_dir, FONT_SIZE
#     )
#     plot_hit_rate_vs_error(
#         dataset, embedding_model_name, llm_model_name, timestamp, results_dir, FONT_SIZE
#     )
#     plot_precision_vs_recall(
#         dataset, embedding_model_name, llm_model_name, timestamp, results_dir, FONT_SIZE
#     )
#     plot_duration_comparison(
#         dataset, embedding_model_name, llm_model_name, timestamp, results_dir, FONT_SIZE
#     )
#     plot_duration_vs_error_rate(
#         dataset, embedding_model_name, llm_model_name, timestamp, results_dir, FONT_SIZE
#     )
#     plot_cache_hit_latency_vs_size_comparison(
#         dataset, embedding_model_name, llm_model_name, timestamp, results_dir, FONT_SIZE
#     )
#     plot_hit_rate_vs_latency(
#         dataset, embedding_model_name, llm_model_name, timestamp, results_dir, FONT_SIZE
#     )


# def _generate_individual_plots(
#     dataset, emb_model, llm_model, llm_model_path, results_dir, timestamp
# ):
#     static_dirs = [
#         d
#         for d in os.listdir(llm_model_path)
#         if d.startswith("static_") and os.path.isdir(os.path.join(llm_model_path, d))
#     ]
#     vectorq_dirs = [
#         d
#         for d in os.listdir(llm_model_path)
#         if d.startswith("vectorq_") and os.path.isdir(os.path.join(llm_model_path, d))
#     ]

#     # Process static threshold directories
#     for static_dir in static_dirs:
#         static_path = os.path.join(llm_model_path, static_dir)
#         result_files = [
#             f
#             for f in os.listdir(static_path)
#             if f.startswith("results_") and f.endswith(".json")
#         ]

#         if not result_files:
#             print(f"No result files found in {static_path}, skipping...")
#             continue

#         # For each result file in the static threshold directory
#         for result_file in result_files:
#             json_path = os.path.join(static_path, result_file)

#             try:
#                 with open(json_path, "r") as f:
#                     data = json.load(f)
#                 benchmark_object = _create_benchmark_object(
#                     json_path,
#                     data,
#                     dataset,
#                     emb_model,
#                     llm_model,
#                     is_dynamic=False,
#                     results_dir=results_dir,
#                     timestamp=timestamp,
#                 )
#                 _generate_plots_for_individual_benchmark(
#                     benchmark_object, is_dynamic=False
#                 )
#             except Exception as e:
#                 print(f"        Error processing {json_path}: {str(e)}")

#     # Process dynamic threshold (vectorq) directories
#     for vectorq_dir in vectorq_dirs:
#         vectorq_path = os.path.join(llm_model_path, vectorq_dir)
#         result_files = [
#             f
#             for f in os.listdir(vectorq_path)
#             if f.startswith("results_") and f.endswith(".json")
#         ]

#         if not result_files:
#             print(f"No result files found in {vectorq_path}, skipping...")
#             continue

#         # For each result file in the vectorq directory
#         for result_file in result_files:
#             json_path = os.path.join(vectorq_path, result_file)

#             try:
#                 with open(json_path, "r") as f:
#                     data = json.load(f)
#                 benchmark_object = _create_benchmark_object(
#                     json_path,
#                     data,
#                     dataset,
#                     emb_model,
#                     llm_model,
#                     is_dynamic=True,
#                     results_dir=results_dir,
#                     timestamp=timestamp,
#                 )
#                 _generate_plots_for_individual_benchmark(
#                     benchmark_object, is_dynamic=True
#                 )
#             except Exception as e:
#                 print(f"        Error processing {json_path}: {str(e)}")


# def _create_benchmark_object(
#     json_path, data, dataset, emb_model, llm_model, is_dynamic, results_dir, timestamp
# ):
#     benchmark_object = Benchmark(MAX_SAMPLES)

#     benchmark_object.is_dynamic_threshold = is_dynamic

#     if is_dynamic:
#         benchmark_object.threshold = None
#         benchmark_object.rnd_num_ub = data["config"]["rnd_num_ub"]
#         dir_name = f"vectorq_{benchmark_object.rnd_num_ub}"
#     else:
#         benchmark_object.threshold = data["config"]["threshold"]
#         benchmark_object.rnd_num_ub = None
#         dir_name = f"static_{benchmark_object.threshold}"

#     benchmark_object.embedding_model = emb_model
#     benchmark_object.llm_model = llm_model
#     benchmark_object.filepath = json_path

#     output_dir = f"processed_{results_dir}{dataset}/{emb_model}/{llm_model}/{dir_name}/"
#     os.makedirs(output_dir, exist_ok=True)
#     benchmark_object.output_folder_path = output_dir
#     benchmark_object.timestamp = timestamp
#     benchmark_object.step_size = data.get("config", {}).get("step_size", 1)

#     benchmark_object.sample_sizes = data["sample_sizes"]
#     benchmark_object.error_rates_relative_to_reused_answers = data[
#         "error_rates_relative_to_reused_answers"
#     ]
#     benchmark_object.error_rates_relative_to_step_size = data.get(
#         "error_rates_relative_to_step_size"
#     )
#     benchmark_object.error_rates_absolute = data["error_rates_absolute"]
#     benchmark_object.total_reused_list = data["total_reused_list"]
#     benchmark_object.relative_reuse_rates = data.get("relative_reuse_rates")
#     benchmark_object.inference_time_direct_step_size = data.get(
#         "inference_time_direct_step_size"
#     )
#     benchmark_object.inference_time_vectorq_step_size = data.get(
#         "inference_time_vectorq_step_size"
#     )
#     benchmark_object.total_duration_direct_list = data["total_duration_direct_list"]
#     benchmark_object.total_duration_vectorq_list = data["total_duration_vectorq_list"]
#     benchmark_object.answers_reused = data["answers_reused"]

#     benchmark_object.true_positive_list = data.get("true_positive_list")
#     benchmark_object.false_positive_list = data.get("false_positive_list")
#     benchmark_object.true_negative_list = data.get("true_negative_list")
#     benchmark_object.false_negative_list = data.get("false_negative_list")
#     benchmark_object.precision_list = data.get("precision_list")
#     benchmark_object.recall_list = data.get("recall_list")
#     benchmark_object.accuracy_list = data.get("accuracy_list")
#     benchmark_object.cache_size_list = data.get("cache_size_list")

#     return benchmark_object


# def _generate_plots_for_individual_benchmark(
#     benchmark_object: Benchmark, is_dynamic: bool
# ):
#     plot_cache_hit_latency_vs_size(benchmark_object, FONT_SIZE)
#     plot_error_rate_relative(benchmark_object, FONT_SIZE)
#     plot_error_rate_absolute(benchmark_object, FONT_SIZE)
#     plot_reuse_rate(benchmark_object, FONT_SIZE)
#     plot_duration_trend(benchmark_object, FONT_SIZE)
#     plot_precision(benchmark_object, FONT_SIZE)
#     plot_recall(benchmark_object, FONT_SIZE)
#     plot_accuracy(benchmark_object, FONT_SIZE)
#     plot_cache_size(benchmark_object, FONT_SIZE)


# ########################################################################################################################
# ########################################################################################################################
# FONT_SIZE = 24
# MAX_SAMPLES = 10
# results_dir = "results/"

# EMBEDDING_MODEL_1 = (
#     "embedding_1",
#     "GteLargeENv1_5",
#     "float32",
#     1024,
# )  # 'Alibaba-NLP/gte-large-en-v1.5'
# EMBEDDING_MODEL_2 = (
#     "embedding_2",
#     "E5_Mistral_7B_Instruct",
#     "float16",
#     4096,
# )  # 'intfloat/e5-mistral-7b-instruct'
# LARGE_LANGUAGE_MODEL_1 = (
#     "response_1",
#     "Llama_3_8B_Instruct",
#     "float16",
#     None,
# )  # 'meta-llama/Meta-Llama-3-8B-Instruct'
# LARGE_LANGUAGE_MODEL_2 = (
#     "response_2",
#     "Llama_3_70B_Instruct",
#     "float16",
#     None,
# )  # 'meta-llama/Meta-Llama-3-70B-Instruct'
# DATASET_NAMES = [
#     "semantic_prompt_cache_benchmark",
#     "ecommerce_dataset",
#     "amazon_instant_video",
# ]

# if __name__ == "__main__":
#     datasets = [DATASET_NAMES[0], DATASET_NAMES[1], DATASET_NAMES[2]]
#     embedding_models = [EMBEDDING_MODEL_1[1], EMBEDDING_MODEL_2[1]]
#     llm_models = [LARGE_LANGUAGE_MODEL_1[1], LARGE_LANGUAGE_MODEL_2[1]]
#     GENERATE_INDIVIDUAL_PLOTS = False
#     GENERATE_COMBINATION_PLOTS = True

#     # Iterate through each dataset
#     for dataset in datasets:
#         dataset_path = os.path.join(results_dir, dataset)
#         if not os.path.exists(dataset_path):
#             print(f"Dataset path not found: {dataset_path}, skipping...")
#             continue
#         print(f"Processing dataset: {dataset}")

#         for emb_model in embedding_models:
#             emb_model_path = os.path.join(dataset_path, emb_model)
#             if not os.path.exists(emb_model_path):
#                 print(
#                     f"  Embedding model path not found: {emb_model_path}, skipping..."
#                 )
#                 continue
#             print(f"  Processing embedding model: {emb_model}")

#             for llm_model in llm_models:
#                 llm_model_path = os.path.join(emb_model_path, llm_model)
#                 if not os.path.exists(llm_model_path):
#                     print(
#                         f"    LLM model path not found: {llm_model_path}, skipping..."
#                     )
#                     continue
#                 print(f"    Processing LLM model: {llm_model}")

#                 timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
#                 if GENERATE_INDIVIDUAL_PLOTS:
#                     _generate_individual_plots(
#                         dataset,
#                         emb_model,
#                         llm_model,
#                         llm_model_path,
#                         results_dir,
#                         timestamp,
#                     )
#                 if GENERATE_COMBINATION_PLOTS:
#                     _generate_combination_plots(
#                         dataset, emb_model, llm_model, timestamp, results_dir
#                     )
#         print("\n")
#     print("Plot generation complete!")
