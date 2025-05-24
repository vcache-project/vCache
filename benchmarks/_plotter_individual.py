import json
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats

from benchmarks._plotter_helper import (
    compute_accuracy_cumulative_list,
    compute_accuracy_score,
    compute_avg_latency_score,
    compute_cache_hit_rate_cumulative_list,
    compute_cache_hit_rate_score,
    compute_duration_cumulative_list,
    compute_duration_score,
    compute_error_rate_cumulative_list,
    compute_error_rate_score,
    compute_f1_score_cumulative_list,
    compute_f1_score_score,
    compute_precision_cumulative_list,
    compute_precision_score,
    compute_recall_cumulative_list,
    compute_recall_score,
    convert_to_dataframe_from_benchmark,
)

if TYPE_CHECKING:
    from benchmarks.benchmark import Benchmark


def generate_individual_plots(
    benchmark: "Benchmark", font_size: int, is_static: bool, parameter: float
):
    df, _ = convert_to_dataframe_from_benchmark(benchmark)

    __plot_accuracy_precision_recall_f1_score(
        benchmark=benchmark, df=df, font_size=font_size
    )
    __plot_error_rate_cache_hit_rate_duration_avg_latency(
        benchmark=benchmark,
        df=df,
        font_size=font_size,
        is_static=is_static,
        parameter=parameter,
    )
    __plot_avg_latency_cache_hit_rate_cache_miss_rate(benchmark=benchmark, df=df)


def __plot_accuracy_precision_recall_f1_score(
    benchmark: "Benchmark", df: pd.DataFrame, font_size: int
):
    accuracy_acc_list = compute_accuracy_cumulative_list(
        tp=df["tp_list"],
        fp=df["fp_list"],
        tn=df["tn_list"],
        fn=df["fn_list"],
    )
    precision_acc_list = compute_precision_cumulative_list(
        tp=df["tp_list"], fp=df["fp_list"]
    )
    recall_acc_list = compute_recall_cumulative_list(tp=df["tp_list"], fn=df["fn_list"])
    f1_score_acc_list = compute_f1_score_cumulative_list(
        tp=df["tp_list"],
        fp=df["fp_list"],
        fn=df["fn_list"],
    )

    fig, axes = plt.subplots(2, 2, figsize=(25, 20))
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
    benchmark: "Benchmark",
    df: pd.DataFrame,
    font_size: int,
    is_static: bool,
    parameter: float,
):
    error_rate_acc_list = compute_error_rate_cumulative_list(fp=df["fp_list"])
    error_rate_acc_list = [rate * 100 for rate in error_rate_acc_list]
    cache_hit_rate_acc_list = compute_cache_hit_rate_cumulative_list(
        cache_hit_list=df["cache_hit_list"]
    )
    cache_hit_rate_acc_list = [rate * 100 for rate in cache_hit_rate_acc_list]

    duration_vcache_acc_list = compute_duration_cumulative_list(
        latency_list=df["latency_vectorq_list"]
    )
    duration_direct_acc_list = compute_duration_cumulative_list(
        latency_list=df["latency_direct_list"]
    )

    fig, axes = plt.subplots(2, 2, figsize=(25, 20))
    sample_index = range(1, len(error_rate_acc_list) + 1)
    sample_index_array = list(sample_index)

    # Plot Error Rate
    axes[0, 0].plot(sample_index, error_rate_acc_list, "r-", linewidth=2)
    if is_static:
        axes[0, 0].set_title(f"Error Rate (T = {parameter})", fontsize=font_size)
    else:
        axes[0, 0].set_title(f"Error Rate (δ = {parameter})", fontsize=font_size)
    axes[0, 0].set_xlabel("Samples", fontsize=font_size)
    axes[0, 0].set_ylabel("Error Rate (%)", fontsize=font_size)
    axes[0, 0].grid(True)
    axes[0, 0].tick_params(axis="both", labelsize=font_size - 2)

    # Plot Cache Hit Rate
    axes[0, 1].plot(sample_index, cache_hit_rate_acc_list, "g-", linewidth=2)
    if is_static:
        axes[0, 1].set_title(f"Cache Hit Rate (T = {parameter})", fontsize=font_size)
    else:
        axes[0, 1].set_title(f"Cache Hit Rate (δ = {parameter})", fontsize=font_size)
    axes[0, 1].set_xlabel("Samples", fontsize=font_size)
    axes[0, 1].set_ylabel("Hit Rate (%)", fontsize=font_size)
    axes[0, 1].grid(True)
    axes[0, 1].tick_params(axis="both", labelsize=font_size - 2)

    # Plot vCache and DirectDuration
    duration_vcache_minutes = [d / 60.0 for d in duration_vcache_acc_list]
    duration_direct_minutes = [d / 60.0 for d in duration_direct_acc_list]
    axes[1, 0].plot(
        sample_index, duration_vcache_minutes, "b-", linewidth=2, label="vCache"
    )
    axes[1, 0].plot(
        sample_index, duration_direct_minutes, "m-", linewidth=2, label="Direct"
    )
    axes[1, 0].set_title("Duration Comparison", fontsize=font_size)
    axes[1, 0].set_xlabel("Samples", fontsize=font_size)
    axes[1, 0].set_ylabel("Duration (min)", fontsize=font_size)
    axes[1, 0].grid(True)
    axes[1, 0].tick_params(axis="both", labelsize=font_size - 2)
    axes[1, 0].legend(fontsize=font_size - 2)

    # Plot vCache and Direct Latency with regression lines
    slope_vq, intercept_vq, r_value_vq, p_value_vq, std_err_vq = stats.linregress(
        sample_index_array, df["latency_vectorq_list"]
    )
    line_vq = [slope_vq * x + intercept_vq for x in sample_index_array]
    axes[1, 1].plot(sample_index, line_vq, "b--", linewidth=2, label="vCache")

    slope_dir, intercept_dir, r_value_dir, p_value_dir, std_err_dir = stats.linregress(
        sample_index_array, df["latency_direct_list"]
    )
    line_dir = [slope_dir * x + intercept_dir for x in sample_index_array]
    axes[1, 1].plot(sample_index, line_dir, "m--", linewidth=2, label="Direct")

    axes[1, 1].set_title("Latency Comparison", fontsize=font_size)
    axes[1, 1].set_xlabel("Samples", fontsize=font_size)
    axes[1, 1].set_ylabel("Latency (s)", fontsize=font_size)
    axes[1, 1].grid(True)
    axes[1, 1].tick_params(axis="both", labelsize=font_size - 2)
    axes[1, 1].legend(fontsize=font_size - 4)

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
    avg_latency_vcache_overall = compute_avg_latency_score(
        latency_list=df["latency_vectorq_list"]
    )
    avg_latency_direct_overall = compute_avg_latency_score(
        latency_list=df["latency_direct_list"]
    )

    cache_hit_mask = df["cache_hit_list"] > 0
    cache_miss_mask = df["cache_miss_list"] > 0

    latency_vcache_cache_hit_list = df.loc[cache_hit_mask, "latency_vectorq_list"]
    latency_vcache_cache_miss_list = df.loc[cache_miss_mask, "latency_vectorq_list"]

    avg_latency_vcache_cache_hit = 0
    if not latency_vcache_cache_hit_list.empty:
        avg_latency_vcache_cache_hit = compute_avg_latency_score(
            latency_list=latency_vcache_cache_hit_list
        )

    avg_latency_vcache_cache_miss = 0
    if not latency_vcache_cache_miss_list.empty:
        avg_latency_vcache_cache_miss = compute_avg_latency_score(
            latency_list=latency_vcache_cache_miss_list
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

    cache_hit_rate_vcache = compute_cache_hit_rate_score(
        cache_hit_list=df["cache_hit_list"]
    )
    cache_miss_rate_vcache = 1 - cache_hit_rate_vcache

    error_rate_vcache = compute_error_rate_score(fp=df["fp_list"])

    duration_vcache = compute_duration_score(latency_list=df["latency_vectorq_list"])

    duration_direct = compute_duration_score(latency_list=df["latency_direct_list"])

    accuracy_vcache = compute_accuracy_score(
        tp=df["tp_list"],
        fp=df["fp_list"],
        tn=df["tn_list"],
        fn=df["fn_list"],
    )
    precision_vcache = compute_precision_score(
        tp=df["tp_list"],
        fp=df["fp_list"],
    )
    recall_vcache = compute_recall_score(
        tp=df["tp_list"],
        fn=df["fn_list"],
    )
    f1_score_vcache = compute_f1_score_score(
        tp=df["tp_list"],
        fp=df["fp_list"],
        fn=df["fn_list"],
    )

    statistics = {
        "avg_latency": {
            "cache": {
                "overall": float(avg_latency_vcache_overall),
                "cache_hit": float(avg_latency_vcache_cache_hit),
                "cache_miss": float(avg_latency_vcache_cache_miss),
            },
            "direct": {"overall": float(avg_latency_direct_overall)},
            "difference": {
                "overall": float(
                    avg_latency_direct_overall - avg_latency_vcache_overall
                ),
                "cache_hit": float(
                    avg_latency_direct_cache_hit - avg_latency_vcache_cache_hit
                ),
                "cache_miss": float(
                    avg_latency_direct_cache_miss - avg_latency_vcache_cache_miss
                ),
            },
            "ratio": {
                "overall": float(
                    avg_latency_direct_overall / avg_latency_vcache_overall
                )
                if avg_latency_vcache_overall > 0
                else "N/A",
                "cache_hit": float(
                    avg_latency_direct_cache_hit / avg_latency_vcache_cache_hit
                )
                if avg_latency_vcache_cache_hit > 0
                else "N/A",
                "cache_miss": float(
                    avg_latency_direct_cache_miss / avg_latency_vcache_cache_miss
                )
                if avg_latency_vcache_cache_miss > 0
                else "N/A",
            },
        },
        "cache": {
            "hit_rate": float(cache_hit_rate_vcache),
            "miss_rate": float(cache_miss_rate_vcache),
            "total_samples": int(len(df["latency_vectorq_list"])),
            "hits": int(df["cache_hit_list"].iloc[-1]),
            "misses": int(df["cache_miss_list"].iloc[-1]),
            "error_rate": float(error_rate_vcache),
        },
        "duration": {
            "vectorq": float(duration_vcache),
            "direct": float(duration_direct),
        },
        "statistics": {
            "accuracy": float(accuracy_vcache),
            "precision": float(precision_vcache),
            "recall": float(recall_vcache),
            "f1_score": float(f1_score_vcache),
        },
    }

    filename = benchmark.output_folder_path + f"/statistics_{benchmark.timestamp}.json"

    with open(filename, "w") as f:
        json.dump(statistics, f, indent=4)
