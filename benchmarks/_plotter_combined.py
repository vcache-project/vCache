import os
import json

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from benchmarks.benchmark import Benchmark
    
def _get_result_files(
    dataset, embedding_model_name, llm_model_name, results_dir
):
    base_dir = f"{results_dir}{dataset}/{embedding_model_name}/{llm_model_name}/"

    if not os.path.exists(base_dir):
        print(f"No results found in {base_dir}")
        return [], []

    static_files = []
    dynamic_files = []

    for d in os.listdir(base_dir):
        # Process static threshold directories
        if d.startswith("static_") and os.path.isdir(os.path.join(base_dir, d)):
            dir_path = os.path.join(base_dir, d)
            for file in os.listdir(dir_path):
                if file.startswith("results_") and file.endswith(".json"):
                    static_files.append(os.path.join(dir_path, file))

        # Process vectorq (dynamic threshold) directories
        elif d.startswith("vectorq_") and os.path.isdir(os.path.join(base_dir, d)):
            dir_path = os.path.join(base_dir, d)
            for file in os.listdir(dir_path):
                if file.startswith("results_") and file.endswith(".json"):
                    dynamic_files.append(os.path.join(dir_path, file))

    return static_files, dynamic_files
    
def generate_combined_plots(benchmark: "Benchmark", font_size: int, dataset: str, results_dir: str):
    static_files, dynamic_files = _get_result_files(
        dataset=dataset,
        embedding_model_name=benchmark.embedding_model,
        llm_model_name=benchmark.llm_model,
        results_dir=results_dir
    )
    
    if not static_files and not dynamic_files:
        print(
            f"No folders found for {dataset}, {benchmark.embedding_model}, {benchmark.llm_model}\n"
            f"in {results_dir}"
        )
        return
    
    for static_file_path in static_files:
        with open(static_file_path, "r") as f:
            data = json.load(f)
            plot_roc(benchmark=benchmark, font_size=font_size)
            plot_precision_vs_recall(benchmark=benchmark, font_size=font_size)
            plot_avg_latency(benchmark=benchmark, font_size=font_size)
            plot_cache_hit_vs_error_rate(benchmark=benchmark, font_size=font_size)
            plot_delta_accuracy(benchmark=benchmark, font_size=font_size)

    for dynamic_file_path in dynamic_files:
        with open(dynamic_file_path, "r") as f:
            data = json.load(f)        
            plot_roc(benchmark=benchmark, font_size=font_size)
            plot_precision_vs_recall(benchmark=benchmark, font_size=font_size)
            plot_avg_latency(benchmark=benchmark, font_size=font_size)
            plot_cache_hit_vs_error_rate(benchmark=benchmark, font_size=font_size)
            plot_delta_accuracy(benchmark=benchmark, font_size=font_size)
    
def plot_roc(benchmark: "Benchmark", font_size: int):
    
    filename = (
            benchmark.output_folder_path + f"/roc_{benchmark.timestamp}.pdf"
        )

def plot_precision_vs_recall(benchmark: "Benchmark", font_size: int):
    
    filename = (
            benchmark.output_folder_path + f"/precision_vs_recall_{benchmark.timestamp}.pdf"
        )

def plot_avg_latency(benchmark: "Benchmark", font_size: int):
    
    filename = (
            benchmark.output_folder_path + f"/avg_latency_{benchmark.timestamp}.pdf"
        )

def plot_cache_hit_vs_error_rate(benchmark: "Benchmark", font_size: int):
    
    filename = (
            benchmark.output_folder_path + f"/cache_hit_vs_error_rate_{benchmark.timestamp}.pdf"
        )

def plot_delta_accuracy(benchmark: "Benchmark", font_size: int):
    
    filename = (
            benchmark.output_folder_path + f"/delta_accuracy_{benchmark.timestamp}.pdf"
        )
