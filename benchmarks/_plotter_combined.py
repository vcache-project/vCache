import json
import os
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from benchmarks._plotter_helper import (
    compute_avg_latency_score,
    compute_cache_hit_rate_cumulative_list,
    compute_cache_hit_rate_score,
    compute_error_rate_cumulative_list,
    compute_error_rate_score,
    compute_false_positive_rate_score,
    compute_precision_score,
    compute_recall_score,
    convert_to_dataframe_from_json_file,
)


def __get_result_files(results_dir: str):
    if not os.path.exists(results_dir):
        print(f"No results found in {results_dir}")
        return [], []

    static_files: List[str] = []
    vectorq_local_files: List[str] = []
    vectorq_global_files: List[str] = []

    for d in os.listdir(results_dir):
        # Process static threshold directories
        if d.startswith("static_") and os.path.isdir(os.path.join(results_dir, d)):
            dir_path: str = os.path.join(results_dir, d)
            for file in os.listdir(dir_path):
                if file.startswith("results_") and file.endswith(".json"):
                    static_files.append(os.path.join(dir_path, file))

        # Process vectorq (embedding specific threshold) directories
        elif d.startswith("vectorq_local") and os.path.isdir(
            os.path.join(results_dir, d)
        ):
            dir_path: str = os.path.join(results_dir, d)
            for file in os.listdir(dir_path):
                if file.startswith("results_") and file.endswith(".json"):
                    vectorq_local_files.append(os.path.join(dir_path, file))

        # Process vectorq (global threshold) directories
        elif d.startswith("vectorq_global") and os.path.isdir(
            os.path.join(results_dir, d)
        ):
            dir_path: str = os.path.join(results_dir, d)
            for file in os.listdir(dir_path):
                if file.startswith("results_") and file.endswith(".json"):
                    vectorq_global_files.append(os.path.join(dir_path, file))

    return static_files, vectorq_local_files, vectorq_global_files


def generate_combined_plots(
    dataset: str,
    embedding_model_name: str,
    llm_model_name: str,
    results_dir: str,
    timestamp: str,
    font_size: int,
):
    results_dir: str = (
        f"{results_dir}/{dataset}/{embedding_model_name}/{llm_model_name}/"
    )

    static_files, vectorq_local_files, vectorq_global_files = __get_result_files(
        results_dir
    )

    if not static_files and not vectorq_local_files and not vectorq_global_files:
        print(
            f"No folders found for {dataset}, {embedding_model_name}, {llm_model_name}\n"
            f"in {results_dir}"
        )
        return

    static_data_frames: Dict[float, pd.DataFrame] = {}
    for static_file_path in static_files:
        with open(static_file_path, "r") as f:
            data: Any = json.load(f)
            dataframe, _ = convert_to_dataframe_from_json_file(data)
            threshold: float = data["config"]["threshold"]
            static_data_frames[threshold] = dataframe

    vectorq_local_data_frames: Dict[float, pd.DataFrame] = {}
    for dynamic_file_path in vectorq_local_files:
        with open(dynamic_file_path, "r") as f:
            data: Any = json.load(f)
            dataframe, _ = convert_to_dataframe_from_json_file(data)
            delta: float = data["config"]["delta"]
            vectorq_local_data_frames[delta] = dataframe

    vectorq_global_data_frames: Dict[float, pd.DataFrame] = {}
    for global_file_path in vectorq_global_files:
        with open(global_file_path, "r") as f:
            data: Any = json.load(f)
            dataframe, _ = convert_to_dataframe_from_json_file(data)
            delta: float = data["config"]["delta"]
            vectorq_global_data_frames[delta] = dataframe

    __plot_roc(
        static_data_frames=static_data_frames,
        vectorq_local_data_frames=vectorq_local_data_frames,
        vectorq_global_data_frames=vectorq_global_data_frames,
        results_dir=results_dir,
        timestamp=timestamp,
        font_size=font_size,
    )
    __plot_precision_vs_recall(
        static_data_frames=static_data_frames,
        vectorq_local_data_frames=vectorq_local_data_frames,
        vectorq_global_data_frames=vectorq_global_data_frames,
        results_dir=results_dir,
        timestamp=timestamp,
        font_size=font_size,
    )
    __plot_avg_latency_vs_error_rate(
        static_data_frames=static_data_frames,
        vectorq_local_data_frames=vectorq_local_data_frames,
        vectorq_global_data_frames=vectorq_global_data_frames,
        results_dir=results_dir,
        timestamp=timestamp,
        font_size=font_size,
    )
    __plot_cache_hit_vs_error_rate(
        static_data_frames=static_data_frames,
        vectorq_local_data_frames=vectorq_local_data_frames,
        vectorq_global_data_frames=vectorq_global_data_frames,
        results_dir=results_dir,
        timestamp=timestamp,
        font_size=font_size,
    )
    __plot_cache_hit_vs_error_rate_vs_sample_size(
        static_data_frames=static_data_frames,
        vectorq_local_data_frames=vectorq_local_data_frames,
        vectorq_global_data_frames=vectorq_global_data_frames,
        results_dir=results_dir,
        timestamp=timestamp,
        font_size=font_size,
    )
    __plot_delta_accuracy(
        vectorq_local_data_frames=vectorq_local_data_frames,
        vectorq_global_data_frames=vectorq_global_data_frames,
        results_dir=results_dir,
        timestamp=timestamp,
        font_size=font_size,
    )


def __plot_roc(
    static_data_frames: Dict[float, pd.DataFrame],
    vectorq_local_data_frames: Dict[float, pd.DataFrame],
    vectorq_global_data_frames: Dict[float, pd.DataFrame],
    results_dir: str,
    timestamp: str,
    font_size: int,
):
    plt.figure(figsize=(12, 10))

    static_thresholds = sorted(static_data_frames.keys())
    static_tpr_values = []
    static_fpr_values = []

    for threshold in static_thresholds:
        df = static_data_frames[threshold]

        tpr = compute_recall_score(tp=df["tp_list"], fn=df["fn_list"])

        fpr = compute_false_positive_rate_score(fp=df["fp_list"], tn=df["tn_list"])

        static_tpr_values.append(tpr)
        static_fpr_values.append(fpr)

    if static_thresholds:
        plt.plot(
            static_fpr_values,
            static_tpr_values,
            "o-",
            color="blue",
            linewidth=2,
            label="GPTCache",
            markersize=8,
        )

        for i, threshold in enumerate(static_thresholds):
            if i == 0 or i == len(static_thresholds) - 1:
                label = f"{threshold:.2f}"
            else:
                label = None
            plt.annotate(
                label,
                (static_fpr_values[i], static_tpr_values[i]),
                textcoords="offset points",
                xytext=(0, 10),
                ha="center",
                fontsize=font_size - 4,
            )

    vectorq_local_deltas = sorted(vectorq_local_data_frames.keys())
    vectorq_local_tpr_values = []
    vectorq_local_fpr_values = []

    for delta in vectorq_local_deltas:
        df = vectorq_local_data_frames[delta]

        tpr = compute_recall_score(tp=df["tp_list"], fn=df["fn_list"])

        fpr = compute_false_positive_rate_score(fp=df["fp_list"], tn=df["tn_list"])

        vectorq_local_tpr_values.append(tpr)
        vectorq_local_fpr_values.append(fpr)

    if vectorq_local_deltas:
        plt.plot(
            vectorq_local_fpr_values,
            vectorq_local_tpr_values,
            "o-",
            color="green",
            linewidth=2,
            label="VectorQ (Local)",
            markersize=8,
        )

        for i, delta in enumerate(vectorq_local_deltas):
            if i == 0 or i == len(vectorq_local_deltas) - 1:
                label = f"{delta:.2f}"
            else:
                label = None
            plt.annotate(
                label,
                (vectorq_local_fpr_values[i], vectorq_local_tpr_values[i]),
                textcoords="offset points",
                xytext=(0, 10),
                ha="center",
                fontsize=font_size - 4,
            )

    vectorq_global_deltas = sorted(vectorq_global_data_frames.keys())
    vectorq_global_tpr_values = []
    vectorq_global_fpr_values = []

    for delta in vectorq_global_deltas:
        df = vectorq_global_data_frames[delta]

        tpr = compute_recall_score(tp=df["tp_list"], fn=df["fn_list"])

        fpr = compute_false_positive_rate_score(fp=df["fp_list"], tn=df["tn_list"])

        vectorq_global_tpr_values.append(tpr)
        vectorq_global_fpr_values.append(fpr)

    if vectorq_global_deltas:
        plt.plot(
            vectorq_global_fpr_values,
            vectorq_global_tpr_values,
            "o-",
            color="red",
            linewidth=2,
            label="VectorQ (Global)",
            markersize=8,
        )

        for i, delta in enumerate(vectorq_global_deltas):
            if i == 0 or i == len(vectorq_global_deltas) - 1:
                label = f"{delta:.2f}"
            else:
                label = None
            plt.annotate(
                label,
                (vectorq_global_fpr_values[i], vectorq_global_tpr_values[i]),
                textcoords="offset points",
                xytext=(0, 10),
                ha="center",
                fontsize=font_size - 4,
            )

    plt.plot([0, 1], [0, 1], "k--", label="Random classifier")

    plt.xlabel("False Positive Rate", fontsize=font_size)
    plt.ylabel("True Positive Rate", fontsize=font_size)
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.legend(loc="lower right", fontsize=font_size - 2)
    plt.tick_params(axis="both", labelsize=font_size - 2)

    plt.xlim(0, 1)
    plt.ylim(0, 1)

    filename = results_dir + f"/roc_{timestamp}.pdf"
    plt.savefig(filename, format="pdf", bbox_inches="tight")
    plt.close()


def __plot_precision_vs_recall(
    static_data_frames: Dict[float, pd.DataFrame],
    vectorq_local_data_frames: Dict[float, pd.DataFrame],
    vectorq_global_data_frames: Dict[float, pd.DataFrame],
    results_dir: str,
    timestamp: str,
    font_size: int,
):
    plt.figure(figsize=(12, 10))

    static_thresholds = sorted(static_data_frames.keys())
    static_precision_values = []
    static_recall_values = []

    for threshold in static_thresholds:
        df = static_data_frames[threshold]
        precision = compute_precision_score(tp=df["tp_list"], fp=df["fp_list"])
        recall = compute_recall_score(tp=df["tp_list"], fn=df["fn_list"])

        static_precision_values.append(precision)
        static_recall_values.append(recall)

    if static_thresholds:
        plt.plot(
            static_recall_values,
            static_precision_values,
            "o-",
            color="blue",
            linewidth=2,
            label="GPTCache",
            markersize=8,
        )

        for i, threshold in enumerate(static_thresholds):
            if i == 0 or i == len(static_thresholds) - 1:
                label = f"{threshold:.2f}"
            else:
                label = None
            plt.annotate(
                label,
                (static_recall_values[i], static_precision_values[i]),
                textcoords="offset points",
                xytext=(0, 10),
                ha="center",
                fontsize=font_size - 4,
            )

    vectorq_local_deltas = sorted(vectorq_local_data_frames.keys())
    vectorq_local_precision_values = []
    vectorq_local_recall_values = []

    for delta in vectorq_local_deltas:
        df = vectorq_local_data_frames[delta]
        precision = compute_precision_score(tp=df["tp_list"], fp=df["fp_list"])
        recall = compute_recall_score(tp=df["tp_list"], fn=df["fn_list"])

        vectorq_local_precision_values.append(precision)
        vectorq_local_recall_values.append(recall)

    if vectorq_local_deltas:
        plt.plot(
            vectorq_local_recall_values,
            vectorq_local_precision_values,
            "o-",
            color="green",
            linewidth=2,
            label="VectorQ (Local)",
            markersize=8,
        )

        for i, delta in enumerate(vectorq_local_deltas):
            if i == 0 or i == len(vectorq_local_deltas) - 1:
                label = f"{delta:.2f}"
            else:
                label = None
            plt.annotate(
                label,
                (vectorq_local_recall_values[i], vectorq_local_precision_values[i]),
                textcoords="offset points",
                xytext=(0, 10),
                ha="center",
                fontsize=font_size - 4,
            )

    vectorq_global_deltas = sorted(vectorq_global_data_frames.keys())
    vectorq_global_precision_values = []
    vectorq_global_recall_values = []

    for delta in vectorq_global_deltas:
        df = vectorq_global_data_frames[delta]
        precision = compute_precision_score(tp=df["tp_list"], fp=df["fp_list"])
        recall = compute_recall_score(tp=df["tp_list"], fn=df["fn_list"])

        vectorq_global_precision_values.append(precision)
        vectorq_global_recall_values.append(recall)

    if vectorq_global_deltas:
        plt.plot(
            vectorq_global_recall_values,
            vectorq_global_precision_values,
            "o-",
            color="red",
            linewidth=2,
            label="VectorQ (Global)",
            markersize=8,
        )

        for i, delta in enumerate(vectorq_global_deltas):
            if i == 0 or i == len(vectorq_global_deltas) - 1:
                label = f"{delta:.2f}"
            else:
                label = None
            plt.annotate(
                label,
                (vectorq_global_recall_values[i], vectorq_global_precision_values[i]),
                textcoords="offset points",
                xytext=(0, 10),
                ha="center",
                fontsize=font_size - 4,
            )

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("Recall", fontsize=font_size)
    plt.ylabel("Precision", fontsize=font_size)
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.legend(loc="best", fontsize=font_size - 2)
    plt.tick_params(axis="both", labelsize=font_size - 2)

    filename = results_dir + f"/precision_vs_recall_{timestamp}.pdf"
    plt.savefig(filename, format="pdf", bbox_inches="tight")
    plt.close()


def __plot_avg_latency_vs_error_rate(
    static_data_frames: Dict[float, pd.DataFrame],
    vectorq_local_data_frames: Dict[float, pd.DataFrame],
    vectorq_global_data_frames: Dict[float, pd.DataFrame],
    results_dir: str,
    timestamp: str,
    font_size: int,
):
    plt.figure(figsize=(12, 10))

    static_thresholds = sorted(static_data_frames.keys())
    static_error_rates = []
    static_latencies = []

    for threshold in static_thresholds:
        df = static_data_frames[threshold]

        error_rate = compute_error_rate_score(fp=df["fp_list"])

        avg_latency = compute_avg_latency_score(latency_list=df["latency_vectorq_list"])
        avg_latency = avg_latency / 60.0

        static_error_rates.append(error_rate)
        static_latencies.append(avg_latency)

    if static_thresholds:
        plt.plot(
            static_error_rates,
            static_latencies,
            "o-",
            color="blue",
            linewidth=2,
            label="GPTCache",
            markersize=8,
        )

        for i, threshold in enumerate(static_thresholds):
            if i == 0 or i == len(static_thresholds) - 1:
                label = f"{threshold:.2f}"
                plt.annotate(
                    label,
                    (static_error_rates[i], static_latencies[i]),
                    textcoords="offset points",
                    xytext=(0, 10),
                    ha="center",
                    fontsize=font_size - 4,
                )

    vectorq_local_deltas = sorted(vectorq_local_data_frames.keys())
    vectorq_local_error_rates = []
    vectorq_local_latencies = []

    for delta in vectorq_local_deltas:
        df = vectorq_local_data_frames[delta]

        error_rate = compute_error_rate_score(fp=df["fp_list"])

        avg_latency = compute_avg_latency_score(latency_list=df["latency_vectorq_list"])

        vectorq_local_error_rates.append(error_rate)
        vectorq_local_latencies.append(avg_latency)

    if vectorq_local_deltas:
        plt.plot(
            vectorq_local_latencies,
            vectorq_local_error_rates,
            "o-",
            color="green",
            linewidth=2,
            label="VectorQ (Local)",
            markersize=8,
        )

        for i, delta in enumerate(vectorq_local_deltas):
            if i == 0 or i == len(vectorq_local_deltas) - 1:
                label = f"{delta:.2f}"
                plt.annotate(
                    label,
                    (vectorq_local_error_rates[i], vectorq_local_latencies[i]),
                    textcoords="offset points",
                    xytext=(0, 10),
                    ha="center",
                    fontsize=font_size - 4,
                )

    vectorq_global_deltas = sorted(vectorq_global_data_frames.keys())
    vectorq_global_error_rates = []
    vectorq_global_latencies = []

    for delta in vectorq_global_deltas:
        df = vectorq_global_data_frames[delta]

        error_rate = compute_error_rate_score(fp=df["fp_list"])

        avg_latency = compute_avg_latency_score(latency_list=df["latency_vectorq_list"])

        vectorq_global_error_rates.append(error_rate)
        vectorq_global_latencies.append(avg_latency)

    if vectorq_global_deltas:
        plt.plot(
            vectorq_global_latencies,
            vectorq_global_error_rates,
            "o-",
            color="red",
            linewidth=2,
            label="VectorQ (Global)",
            markersize=8,
        )

        for i, delta in enumerate(vectorq_global_deltas):
            if i == 0 or i == len(vectorq_global_deltas) - 1:
                label = f"{delta:.2f}"
                plt.annotate(
                    label,
                    (vectorq_global_error_rates[i], vectorq_global_latencies[i]),
                    textcoords="offset points",
                    xytext=(0, 10),
                    ha="center",
                    fontsize=font_size - 4,
                )

    plt.xlabel("Average Latency (sec)", fontsize=font_size)
    plt.ylabel("Error Rate", fontsize=font_size)
    plt.xlim(left=0.0)
    plt.ylim(bottom=0.0)
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.legend(loc="best", fontsize=font_size - 2)
    plt.tick_params(axis="both", labelsize=font_size - 2)

    filename = results_dir + f"/avg_latency_vs_error_rate_{timestamp}.pdf"
    plt.savefig(filename, format="pdf", bbox_inches="tight")
    plt.close()


def __plot_cache_hit_vs_error_rate(
    static_data_frames: Dict[float, pd.DataFrame],
    vectorq_local_data_frames: Dict[float, pd.DataFrame],
    vectorq_global_data_frames: Dict[float, pd.DataFrame],
    results_dir: str,
    timestamp: str,
    font_size: int,
):
    plt.figure(figsize=(12, 10))

    static_thresholds = sorted(static_data_frames.keys())
    static_cache_hit_rates = []
    static_error_rates = []

    for threshold in static_thresholds:
        df = static_data_frames[threshold]

        cache_hit_rate = compute_cache_hit_rate_score(
            cache_hit_list=df["cache_hit_list"]
        )

        error_rate = compute_error_rate_score(fp=df["fp_list"])

        static_cache_hit_rates.append(cache_hit_rate)
        static_error_rates.append(error_rate)

    if static_thresholds:
        plt.plot(
            static_error_rates,
            static_cache_hit_rates,
            "o-",
            color="blue",
            linewidth=2,
            label="GPTCache",
            markersize=8,
        )

        for i, threshold in enumerate(static_thresholds):
            if i == 0 or i == len(static_thresholds) - 1:
                label = f"{threshold:.2f}"
            else:
                label = None
            plt.annotate(
                label,
                (static_error_rates[i], static_cache_hit_rates[i]),
                textcoords="offset points",
                xytext=(0, 10),
                ha="center",
                fontsize=font_size - 4,
            )

    vectorq_local_deltas = sorted(vectorq_local_data_frames.keys())
    vectorq_local_cache_hit_rates = []
    vectorq_local_error_rates = []

    for delta in vectorq_local_deltas:
        df = vectorq_local_data_frames[delta]

        cache_hit_rate = compute_cache_hit_rate_score(
            cache_hit_list=df["cache_hit_list"]
        )

        error_rate = compute_error_rate_score(fp=df["fp_list"])

        vectorq_local_cache_hit_rates.append(cache_hit_rate)
        vectorq_local_error_rates.append(error_rate)

    if vectorq_local_deltas:
        plt.plot(
            vectorq_local_error_rates,
            vectorq_local_cache_hit_rates,
            "o-",
            color="green",
            linewidth=2,
            label="VectorQ (Local)",
            markersize=8,
        )

        for i, delta in enumerate(vectorq_local_deltas):
            if i == 0 or i == len(vectorq_local_deltas) - 1:
                label = f"{delta:.2f}"
            else:
                label = None
            plt.annotate(
                label,
                (vectorq_local_error_rates[i], vectorq_local_cache_hit_rates[i]),
                textcoords="offset points",
                xytext=(0, 10),
                ha="center",
                fontsize=font_size - 4,
            )

    vectorq_global_deltas = sorted(vectorq_global_data_frames.keys())
    vectorq_global_cache_hit_rates = []
    vectorq_global_error_rates = []

    for delta in vectorq_global_deltas:
        df = vectorq_global_data_frames[delta]

        cache_hit_rate = compute_cache_hit_rate_score(
            cache_hit_list=df["cache_hit_list"]
        )

        error_rate = compute_error_rate_score(fp=df["fp_list"])

        vectorq_global_cache_hit_rates.append(cache_hit_rate)
        vectorq_global_error_rates.append(error_rate)

    if vectorq_global_deltas:
        plt.plot(
            vectorq_global_error_rates,
            vectorq_global_cache_hit_rates,
            "o-",
            color="red",
            linewidth=2,
            label="VectorQ (Global)",
            markersize=8,
        )

        for i, delta in enumerate(vectorq_global_deltas):
            if i == 0 or i == len(vectorq_global_deltas) - 1:
                label = f"{delta:.2f}"
            else:
                label = None
            plt.annotate(
                label,
                (vectorq_global_error_rates[i], vectorq_global_cache_hit_rates[i]),
                textcoords="offset points",
                xytext=(0, 10),
                ha="center",
                fontsize=font_size - 4,
            )

    plt.xlabel("Error Rate", fontsize=font_size)
    plt.ylabel("Cache Hit Rate", fontsize=font_size)
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.legend(loc="best", fontsize=font_size - 2)
    plt.tick_params(axis="both", labelsize=font_size - 2)
    plt.xlim(0, 1)
    plt.ylim(0, 1)

    filename = results_dir + f"/cache_hit_vs_error_rate_{timestamp}.pdf"
    plt.savefig(filename, format="pdf", bbox_inches="tight")
    plt.close()


def __plot_cache_hit_vs_error_rate_vs_sample_size(
    static_data_frames: Dict[float, pd.DataFrame],
    vectorq_local_data_frames: Dict[float, pd.DataFrame],
    vectorq_global_data_frames: Dict[float, pd.DataFrame],
    results_dir: str,
    timestamp: str,
    font_size: int,
):
    target_deltas = [0.01, 0.02]

    # Baseline 1) VectorQ (Local)
    vectorq_local_error_rates = []

    for delta in target_deltas:
        df = vectorq_local_data_frames[delta]
        error_rate = compute_error_rate_score(fp=df["fp_list"])
        vectorq_local_error_rates.append(error_rate)

    # Baseline 3) Static thresholds
    # Find the static thresholds that match the VectorQ (Local) error rates
    static_thresholds = sorted(static_data_frames.keys())
    static_error_rates = []

    for threshold in static_thresholds:
        df = static_data_frames[threshold]
        error_rate = compute_error_rate_score(fp=df["fp_list"])
        static_error_rates.append(error_rate)

    matched_static_thresholds = []

    for _, target_error_rate in enumerate(vectorq_local_error_rates):
        closest_idx = min(
            range(len(static_error_rates)),
            key=lambda j: abs(static_error_rates[j] - target_error_rate),
        )

        matched_static_thresholds.append(static_thresholds[closest_idx])

    # Plot the results
    for i, delta in enumerate(target_deltas):
        df_vectorq_local = vectorq_local_data_frames[delta]
        df_vectorq_global = vectorq_global_data_frames[delta]
        static_threshold = matched_static_thresholds[i]
        df_static = static_data_frames[static_threshold]

        vectorq_local_error_rates = compute_error_rate_cumulative_list(
            fp=df_vectorq_local["fp_list"]
        )
        vectorq_global_error_rates = compute_error_rate_cumulative_list(
            fp=df_vectorq_global["fp_list"]
        )
        static_error_rates = compute_error_rate_cumulative_list(fp=df_static["fp_list"])

        vectorq_local_cache_hit_rates = compute_cache_hit_rate_cumulative_list(
            cache_hit_list=df_vectorq_local["cache_hit_list"]
        )
        vectorq_global_cache_hit_rates = compute_cache_hit_rate_cumulative_list(
            cache_hit_list=df_vectorq_global["cache_hit_list"]
        )
        static_cache_hit_rates = compute_cache_hit_rate_cumulative_list(
            cache_hit_list=df_static["cache_hit_list"]
        )

        sample_sizes = np.arange(1, len(vectorq_local_error_rates) + 1)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))

        # Plot 1: Error Rate vs Sample Size
        ax1.plot(
            sample_sizes,
            vectorq_local_error_rates,
            "-",
            color="green",
            linewidth=2,
            label=f"VectorQ Local (δ={delta:.2f})",
        )

        ax1.plot(
            sample_sizes,
            vectorq_global_error_rates,
            "-",
            color="red",
            linewidth=2,
            label=f"VectorQ Global (δ={delta:.2f})",
        )

        ax1.plot(
            sample_sizes,
            static_error_rates,
            "-",
            color="blue",
            linewidth=2,
            label=f"GPTCache (t={static_threshold:.2f})",
        )

        ax1.set_xlabel("Sample Size", fontsize=font_size)
        ax1.set_ylabel("Cumulative Error Rate (%)", fontsize=font_size)
        ax1.grid(True, linestyle="--", alpha=0.7)
        ax1.legend(fontsize=font_size - 2)
        ax1.tick_params(axis="both", labelsize=font_size - 2)

        # Plot 2: Cache Hit Rate vs Sample Size
        ax2.plot(
            sample_sizes,
            vectorq_local_cache_hit_rates,
            "-",
            color="green",
            linewidth=2,
            label=f"VectorQ Local (δ={delta:.2f})",
        )

        ax2.plot(
            sample_sizes,
            vectorq_global_cache_hit_rates,
            "-",
            color="red",
            linewidth=2,
            label=f"VectorQ Global (δ={delta:.2f})",
        )

        ax2.plot(
            sample_sizes,
            static_cache_hit_rates,
            "-",
            color="blue",
            linewidth=2,
            label=f"GPTCache (t={static_threshold:.2f})",
        )

        ax2.set_xlabel("Sample Size", fontsize=font_size)
        ax2.set_ylabel("Cumulative Cache Hit Rate (%)", fontsize=font_size)
        ax2.grid(True, linestyle="--", alpha=0.7)
        ax2.legend(fontsize=font_size - 2)
        ax2.tick_params(axis="both", labelsize=font_size - 2)

        # Adjust layout and save
        plt.tight_layout()
        filename = (
            results_dir
            + f"/cache_hit_vs_error_rate_vs_sample_size_delta_{delta:.2f}_{timestamp}.pdf"
        )
        plt.savefig(filename, format="pdf", bbox_inches="tight")
        plt.close()


def __plot_delta_accuracy(
    vectorq_local_data_frames: Dict[float, pd.DataFrame],
    vectorq_global_data_frames: Dict[float, pd.DataFrame],
    results_dir: str,
    timestamp: str,
    font_size: int,
):
    plt.figure(figsize=(16, 10))

    vectorq_local_deltas = sorted(vectorq_local_data_frames.keys())

    if vectorq_local_deltas:
        error_rates = []
        delta_labels = []

        for delta in vectorq_local_deltas:
            df = vectorq_local_data_frames[delta]

            error_rate = compute_error_rate_score(fp=df["fp_list"])

            error_rates.append(error_rate)
            delta_labels.append(f"{delta:.3f}")

        x_pos = np.arange(len(vectorq_local_deltas))
        bar_width = 0.8

        plt.bar(
            x_pos, error_rates, bar_width, color="skyblue", label="Achieved Error Rate"
        )

        for i, delta in enumerate(vectorq_local_deltas):
            plt.hlines(
                y=delta,
                xmin=i - bar_width / 2,
                xmax=i + bar_width / 2,
                colors="red",
                linestyles="dashed",
                linewidth=2,
            )

        custom_lines = [
            Line2D([0], [0], color="red", linestyle="dashed", lw=2),
            Line2D([0], [0], color="skyblue", lw=4),
        ]
        plt.legend(
            custom_lines,
            ["Delta (Upper Bound)", "Achieved Error Rate"],
            fontsize=font_size - 2,
        )

        plt.xlabel("Delta Values", fontsize=font_size)
        plt.ylabel("Error Rate", fontsize=font_size)
        plt.xticks(x_pos, delta_labels, fontsize=font_size - 2)
        plt.yticks(fontsize=font_size - 2)

        for i, err in enumerate(error_rates):
            plt.text(
                x_pos[i],
                err + 0.003,
                f"{err:.3f}",
                ha="center",
                va="bottom",
                fontsize=font_size - 2,
            )

            plt.text(
                x_pos[i],
                vectorq_local_deltas[i] + 0.002,
                "",
                ha="center",
                va="bottom",
                fontsize=font_size - 2,
                color="red",
            )

        all_values = error_rates + vectorq_local_deltas
        if all_values:
            y_min = 0
            y_max = max(all_values) * 1.15
            plt.ylim(y_min, y_max)

    filename = results_dir + f"/delta_accuracy_{timestamp}.pdf"
    plt.savefig(filename, format="pdf", bbox_inches="tight")
    plt.close()
