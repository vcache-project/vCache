import json
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import pandas as pd

from benchmarks._plotter_helper import (
    compute_accuracy_acc_list,
    compute_avg_latency_score,
    compute_cache_hit_rate_acc_list,
    compute_cache_hit_rate_score,
    compute_cache_miss_rate_score,
    compute_duration_acc_list,
    compute_error_rate_acc_list,
    compute_f1_score_acc_list,
    compute_precision_acc_list,
    compute_recall_acc_list,
    convert_to_dataframe_from_benchmark,
)

if TYPE_CHECKING:
    from benchmarks.benchmark import Benchmark


def generate_individual_plots(benchmark: "Benchmark", font_size: int):
    df, _ = convert_to_dataframe_from_benchmark(benchmark)

    __plot_accuracy_precision_recall_f1_score(
        benchmark=benchmark, df=df, font_size=font_size
    )
    __plot_error_rate_cache_hit_rate_duration_avg_latency(
        benchmark=benchmark, df=df, font_size=font_size
    )
    __plot_avg_latency_cache_hit_rate_cache_miss_rate(benchmark=benchmark, df=df)


def __plot_accuracy_precision_recall_f1_score(
    benchmark: "Benchmark", df: pd.DataFrame, font_size: int
):
    accuracy_acc_list = compute_accuracy_acc_list(
        tp=df["true_positive_acc_list"],
        fp=df["false_positive_acc_list"],
        tn=df["true_negative_acc_list"],
        fn=df["false_negative_acc_list"],
    )
    precision_acc_list = compute_precision_acc_list(
        tp=df["true_positive_acc_list"], fp=df["false_positive_acc_list"]
    )
    recall_acc_list = compute_recall_acc_list(
        tp=df["true_positive_acc_list"], fn=df["false_negative_acc_list"]
    )
    f1_score_acc_list = compute_f1_score_acc_list(
        tp=df["true_positive_acc_list"],
        fp=df["false_positive_acc_list"],
        fn=df["false_negative_acc_list"],
    )

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    sample_index = range(1, len(accuracy_acc_list) + 1)

    # Plot Accuracy
    axes[0, 0].plot(sample_index, accuracy_acc_list, "b-", linewidth=2)
    axes[0, 0].set_title("Accuracy", fontsize=font_size)
    axes[0, 0].set_xlabel("Samples", fontsize=font_size)
    axes[0, 0].set_ylabel("Accuracy", fontsize=font_size)
    axes[0, 0].grid(True)
    axes[0, 0].tick_params(axis="both", labelsize=font_size - 2)
    axes[0, 0].set_ylim(0, 1)

    # Plot Precision
    axes[0, 1].plot(sample_index, precision_acc_list, "r-", linewidth=2)
    axes[0, 1].set_title("Precision", fontsize=font_size)
    axes[0, 1].set_xlabel("Samples", fontsize=font_size)
    axes[0, 1].set_ylabel("Precision", fontsize=font_size)
    axes[0, 1].grid(True)
    axes[0, 1].tick_params(axis="both", labelsize=font_size - 2)
    axes[0, 1].set_ylim(0, 1)

    # Plot Recall
    axes[1, 0].plot(sample_index, recall_acc_list, "g-", linewidth=2)
    axes[1, 0].set_title("Recall", fontsize=font_size)
    axes[1, 0].set_xlabel("Samples", fontsize=font_size)
    axes[1, 0].set_ylabel("Recall", fontsize=font_size)
    axes[1, 0].grid(True)
    axes[1, 0].tick_params(axis="both", labelsize=font_size - 2)
    axes[1, 0].set_ylim(0, 1)

    # Plot F1 Score
    axes[1, 1].plot(sample_index, f1_score_acc_list, "y-", linewidth=2)
    axes[1, 1].set_title("F1 Score", fontsize=font_size)
    axes[1, 1].set_xlabel("Samples", fontsize=font_size)
    axes[1, 1].set_ylabel("F1 Score", fontsize=font_size)
    axes[1, 1].grid(True)
    axes[1, 1].tick_params(axis="both", labelsize=font_size - 2)
    axes[1, 1].set_ylim(0, 1)

    plt.tight_layout(rect=[0, 0, 1, 0.95])

    filename = (
        benchmark.output_folder_path
        + f"/accuracy_precision_recall_f1_score_{benchmark.timestamp}.pdf"
    )
    plt.savefig(filename, format="pdf")
    plt.close()


def __plot_error_rate_cache_hit_rate_duration_avg_latency(
    benchmark: "Benchmark", df: pd.DataFrame, font_size: int
):
    error_rate_acc_list = compute_error_rate_acc_list(
        tp=df["true_positive_acc_list"],
        fp=df["false_positive_acc_list"],
        tn=df["true_negative_acc_list"],
        fn=df["false_negative_acc_list"],
    )
    cache_hit_rate_acc_list = compute_cache_hit_rate_acc_list(
        cache_hit_list=df["cache_hit_acc_list"],
        cache_miss_list=df["cache_miss_acc_list"],
    )
    duration_vectorq_acc_list = compute_duration_acc_list(
        latency_list=df["latency_vectorq_list"]
    )
    duration_direct_acc_list = compute_duration_acc_list(
        latency_list=df["latency_direct_list"]
    )

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    sample_index = range(1, len(error_rate_acc_list) + 1)

    # Plot Error Rate
    axes[0, 0].plot(sample_index, error_rate_acc_list, "r-", linewidth=2)
    axes[0, 0].set_title("Error Rate", fontsize=font_size)
    axes[0, 0].set_xlabel("Samples", fontsize=font_size)
    axes[0, 0].set_ylabel("Error Rate", fontsize=font_size)
    axes[0, 0].grid(True)
    axes[0, 0].tick_params(axis="both", labelsize=font_size - 2)
    axes[0, 0].set_ylim(0, 1)

    # Plot Cache Hit Rate
    axes[0, 1].plot(sample_index, cache_hit_rate_acc_list, "g-", linewidth=2)
    axes[0, 1].set_title("Cache Hit Rate", fontsize=font_size)
    axes[0, 1].set_xlabel("Samples", fontsize=font_size)
    axes[0, 1].set_ylabel("Hit Rate", fontsize=font_size)
    axes[0, 1].grid(True)
    axes[0, 1].tick_params(axis="both", labelsize=font_size - 2)
    axes[0, 1].set_ylim(0, 1)

    # Plot VectorQ Duration
    duration_vectorq_minutes = [d / 60.0 for d in duration_vectorq_acc_list]
    axes[1, 0].plot(sample_index, duration_vectorq_minutes, "b-", linewidth=2)
    axes[1, 0].set_title("VectorQ Cumulative Duration", fontsize=font_size)
    axes[1, 0].set_xlabel("Samples", fontsize=font_size)
    axes[1, 0].set_ylabel("Duration (min)", fontsize=font_size)
    axes[1, 0].grid(True)
    axes[1, 0].tick_params(axis="both", labelsize=font_size - 2)

    # Plot Direct Duration
    duration_direct_minutes = [d / 60.0 for d in duration_direct_acc_list]
    axes[1, 1].plot(sample_index, duration_direct_minutes, "m-", linewidth=2)
    axes[1, 1].set_title("Direct Cumulative Duration", fontsize=font_size)
    axes[1, 1].set_xlabel("Samples", fontsize=font_size)
    axes[1, 1].set_ylabel("Duration (min)", fontsize=font_size)
    axes[1, 1].grid(True)
    axes[1, 1].tick_params(axis="both", labelsize=font_size - 2)

    plt.tight_layout(rect=[0, 0, 1, 0.95])

    filename = (
        benchmark.output_folder_path
        + f"/error_rate_cache_hit_rate_duration_avg_latency_{benchmark.timestamp}.pdf"
    )
    plt.savefig(filename, format="pdf")
    plt.close()


def __plot_avg_latency_cache_hit_rate_cache_miss_rate(
    benchmark: "Benchmark", df: pd.DataFrame
):
    avg_latency_vectorq_overall = compute_avg_latency_score(
        latency_list=df["latency_vectorq_list"]
    )
    avg_latency_direct_overall = compute_avg_latency_score(
        latency_list=df["latency_direct_list"]
    )

    cache_hit_mask = df["cache_hit_acc_list"] > 0
    cache_miss_mask = df["cache_miss_acc_list"] > 0

    latency_vectorq_cache_hit_list = df.loc[cache_hit_mask, "latency_vectorq_list"]
    latency_vectorq_cache_miss_list = df.loc[cache_miss_mask, "latency_vectorq_list"]

    avg_latency_vectorq_cache_hit = 0
    if not latency_vectorq_cache_hit_list.empty:
        avg_latency_vectorq_cache_hit = compute_avg_latency_score(
            latency_list=latency_vectorq_cache_hit_list
        )

    avg_latency_vectorq_cache_miss = 0
    if not latency_vectorq_cache_miss_list.empty:
        avg_latency_vectorq_cache_miss = compute_avg_latency_score(
            latency_list=latency_vectorq_cache_miss_list
        )

    latency_direct_cache_hit_list = df.loc[cache_hit_mask, "latency_direct_list"]
    latency_direct_cache_miss_list = df.loc[cache_miss_mask, "latency_direct_list"]

    avg_latency_direct_cache_hit = 0
    if not latency_direct_cache_hit_list.empty:
        avg_latency_direct_cache_hit = compute_avg_latency_score(
            latency_list=latency_direct_cache_hit_list
        )

    avg_latency_direct_cache_miss = 0
    if not latency_direct_cache_miss_list.empty:
        avg_latency_direct_cache_miss = compute_avg_latency_score(
            latency_list=latency_direct_cache_miss_list
        )

    cache_hit_rate_vectorq = compute_cache_hit_rate_score(
        cache_hit_list_acc=df["cache_hit_acc_list"],
        cache_miss_list_acc=df["cache_miss_acc_list"],
    )
    cache_miss_rate_vectorq = compute_cache_miss_rate_score(
        cache_miss_list_acc=df["cache_miss_acc_list"],
        cache_hit_list_acc=df["cache_hit_acc_list"],
    )

    statistics = {
        "avg_latency": {
            "cache": {
                "overall": float(avg_latency_vectorq_overall),
                "cache_hit": float(avg_latency_vectorq_cache_hit),
                "cache_miss": float(avg_latency_vectorq_cache_miss),
            },
            "direct": {
                "overall": float(avg_latency_direct_overall)
            },
            "difference": {
                "overall": float(
                    avg_latency_direct_overall - avg_latency_vectorq_overall
                ),
                "cache_hit": float(
                    avg_latency_direct_cache_hit - avg_latency_vectorq_cache_hit
                ),
                "cache_miss": float(
                    avg_latency_direct_cache_miss - avg_latency_vectorq_cache_miss
                ),
            },
            "ratio": {
                "overall": float(
                    avg_latency_direct_overall / avg_latency_vectorq_overall
                )
                if avg_latency_vectorq_overall > 0
                else "N/A",
                "cache_hit": float(
                    avg_latency_direct_cache_hit / avg_latency_vectorq_cache_hit
                )
                if avg_latency_vectorq_cache_hit > 0
                else "N/A",
                "cache_miss": float(
                    avg_latency_direct_cache_miss / avg_latency_vectorq_cache_miss
                )
                if avg_latency_vectorq_cache_miss > 0
                else "N/A",
            },
        },
        "cache": {
            "hit_rate": float(cache_hit_rate_vectorq),
            "miss_rate": float(cache_miss_rate_vectorq),
            "total_samples": int(len(df["latency_vectorq_list"])),
            "hits": int(df["cache_hit_acc_list"].iloc[-1]),
            "misses": int(df["cache_miss_acc_list"].iloc[-1]),
        },
    }

    filename = benchmark.output_folder_path + f"/statistics_{benchmark.timestamp}.json"

    with open(filename, "w") as f:
        json.dump(statistics, f, indent=4)
