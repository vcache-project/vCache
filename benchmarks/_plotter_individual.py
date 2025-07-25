import json
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats

from benchmarks._plotter_helper import (
    compute_accuracy_cumulative_list,
    compute_accuracy_score,
    compute_avg_input_tokens_score,
    compute_avg_latency_score,
    compute_avg_output_tokens_score,
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
    compute_total_tokens_score,
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
    __plot_token_usage_analysis(
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
    axes[1, 0].legend(fontsize=font_size - 4, loc='upper left')

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
    axes[1, 1].legend(fontsize=font_size - 6, loc='upper left')

    plt.tight_layout(rect=[0, 0, 1, 0.96], pad=2.0)

    filename = (
        benchmark.output_folder_path
        + f"/error_rate_cache_hit_rate_duration_avg_latency_{benchmark.timestamp}.pdf"
    )
    plt.savefig(filename, format="pdf")
    plt.close()


def __plot_token_usage_analysis(
    benchmark: "Benchmark",
    df: pd.DataFrame,
    font_size: int,
    is_static: bool,
    parameter: float,
):
    """Plot comprehensive token usage analysis including cumulative and distribution plots."""
    
    # Check if token data is available
    if "input_tokens_list" not in df.columns or "output_tokens_list" not in df.columns:
        print("Token data not available in dataset - skipping token usage plot")
        return
    
    # Calculate cumulative token usage
    input_tokens_cumulative = df["input_tokens_list"].cumsum()
    output_tokens_cumulative = df["output_tokens_list"].cumsum()
    total_tokens_cumulative = input_tokens_cumulative + output_tokens_cumulative
    
    # Calculate average tokens per request (rolling average)
    window_size = max(10, len(df) // 20)  # Adaptive window size
    input_tokens_avg = df["input_tokens_list"].rolling(window=window_size, min_periods=1).mean()
    output_tokens_avg = df["output_tokens_list"].rolling(window=window_size, min_periods=1).mean()
    total_tokens_avg = input_tokens_avg + output_tokens_avg
    
    # Separate tokens by cache hit/miss
    cache_hit_mask = df["cache_hit_list"] > 0
    cache_miss_mask = df["cache_miss_list"] > 0
    
    input_tokens_hit = df.loc[cache_hit_mask, "input_tokens_list"]
    input_tokens_miss = df.loc[cache_miss_mask, "input_tokens_list"]
    output_tokens_hit = df.loc[cache_hit_mask, "output_tokens_list"]
    output_tokens_miss = df.loc[cache_miss_mask, "output_tokens_list"]
    
    # Create the plot with 2x3 subplots - increased figure size and better spacing
    fig, axes = plt.subplots(2, 3, figsize=(36, 24))
    sample_index = range(1, len(df) + 1)
    
    # Plot 1: Cumulative Token Usage
    axes[0, 0].plot(sample_index, input_tokens_cumulative, "b-", linewidth=2, label="Input Tokens")
    axes[0, 0].plot(sample_index, output_tokens_cumulative, "r-", linewidth=2, label="Output Tokens")
    axes[0, 0].plot(sample_index, total_tokens_cumulative, "g-", linewidth=2, label="Total Tokens")
    
    if is_static:
        axes[0, 0].set_title(f"Cum Tok Usage (T = {parameter})", fontsize=font_size)
    else:
        axes[0, 0].set_title(f"Cum Tok Usage (δ = {parameter})", fontsize=font_size)
    axes[0, 0].set_xlabel("Samples", fontsize=font_size)
    axes[0, 0].set_ylabel("Cum Tokens", fontsize=font_size)
    axes[0, 0].grid(True)
    axes[0, 0].tick_params(axis="both", labelsize=font_size - 4)
    axes[0, 0].legend(fontsize=font_size - 6, loc='upper left')
    
    # Plot 2: Rolling Average Token Usage
    axes[0, 1].plot(sample_index, input_tokens_avg, "b-", linewidth=2, label="Avg Input")
    axes[0, 1].plot(sample_index, output_tokens_avg, "r-", linewidth=2, label="Avg Output")
    axes[0, 1].plot(sample_index, total_tokens_avg, "g-", linewidth=2, label="Avg Total")
    
    if is_static:
        axes[0, 1].set_title(f"Rolling Avg Tok Usage (T = {parameter})", fontsize=font_size)
    else:
        axes[0, 1].set_title(f"Rolling Avg Tok Usage (δ = {parameter})", fontsize=font_size)
    axes[0, 1].set_xlabel("Samples", fontsize=font_size)
    axes[0, 1].set_ylabel("Avg Tok/Request", fontsize=font_size)
    axes[0, 1].grid(True)
    axes[0, 1].tick_params(axis="both", labelsize=font_size - 4)
    axes[0, 1].legend(fontsize=font_size - 6, loc='upper left')
    
    # Plot 3: Token Distribution by Cache Status (Box Plot)
    token_data = []
    labels = []
    
    if len(input_tokens_hit) > 0:
        token_data.append(input_tokens_hit)
        labels.append("Input\n(Hit)")
    if len(input_tokens_miss) > 0:
        token_data.append(input_tokens_miss)
        labels.append("Input\n(Miss)")
    if len(output_tokens_hit) > 0:
        token_data.append(output_tokens_hit)
        labels.append("Output\n(Hit)")
    if len(output_tokens_miss) > 0:
        token_data.append(output_tokens_miss)
        labels.append("Output\n(Miss)")
    
    if token_data:
        box_plot = axes[0, 2].boxplot(token_data, labels=labels, patch_artist=True)
        colors = ['lightblue', 'lightcoral', 'lightgreen', 'lightyellow']
        for patch, color in zip(box_plot['boxes'], colors[:len(token_data)]):
            patch.set_facecolor(color)
    
    axes[0, 2].set_title("Tok Distr by Cache Status", fontsize=font_size)
    axes[0, 2].set_ylabel("Tokens", fontsize=font_size)
    axes[0, 2].tick_params(axis="both", labelsize=font_size - 4)
    axes[0, 2].grid(True, axis='y')
    
    # Plot 4: Input vs Output Token Scatter
    colors = ['red' if hit else 'blue' for hit in df["cache_hit_list"]]
    scatter = axes[1, 0].scatter(df["input_tokens_list"], df["output_tokens_list"], 
                               c=colors, alpha=0.6, s=20)
    axes[1, 0].set_title("In vs Out Toks", fontsize=font_size)
    axes[1, 0].set_xlabel("In Toks", fontsize=font_size)
    axes[1, 0].set_ylabel("Out Toks", fontsize=font_size)
    axes[1, 0].grid(True)
    axes[1, 0].tick_params(axis="both", labelsize=font_size - 4)
    
    # Add legend for scatter plot
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor='red', label='Cache Hit'),
                      Patch(facecolor='blue', label='Cache Miss')]
    axes[1, 0].legend(handles=legend_elements, fontsize=font_size - 6, loc='upper left')
    
    # Plot 5: Token Efficiency Over Time
    token_efficiency = df["output_tokens_list"] / (df["input_tokens_list"] + 1)  # +1 to avoid division by zero
    efficiency_avg = token_efficiency.rolling(window=window_size, min_periods=1).mean()
    
    axes[1, 1].plot(sample_index, token_efficiency, "gray", alpha=0.3, linewidth=1, label="Efficiency")
    axes[1, 1].plot(sample_index, efficiency_avg, "purple", linewidth=2, label="Avg Efficiency")
    axes[1, 1].set_title("Tok Efficiency (Out/In Ratio)", fontsize=font_size)
    axes[1, 1].set_xlabel("Samples", fontsize=font_size)
    axes[1, 1].set_ylabel("Out/In Tok Ratio", fontsize=font_size)
    axes[1, 1].grid(True)
    axes[1, 1].tick_params(axis="both", labelsize=font_size - 4)
    axes[1, 1].legend(fontsize=font_size - 6, loc='upper right')
    
    # Plot 6: Token Summary Statistics
    axes[1, 2].axis('off')  # Turn off axis for text display
    
    # Calculate summary statistics
    avg_input = compute_avg_input_tokens_score(df["input_tokens_list"])
    avg_output = compute_avg_output_tokens_score(df["output_tokens_list"])
    avg_total = compute_total_tokens_score(df["input_tokens_list"], df["output_tokens_list"])
    
    total_input = df["input_tokens_list"].sum()
    total_output = df["output_tokens_list"].sum()
    total_all = total_input + total_output
    
    # Cache hit/miss token averages
    avg_input_hit = input_tokens_hit.mean() if len(input_tokens_hit) > 0 else 0
    avg_input_miss = input_tokens_miss.mean() if len(input_tokens_miss) > 0 else 0
    avg_output_hit = output_tokens_hit.mean() if len(output_tokens_hit) > 0 else 0
    avg_output_miss = output_tokens_miss.mean() if len(output_tokens_miss) > 0 else 0
    
    # Create summary text
    summary_text = f"""TOKEN USAGE SUMMARY

Average per Request:
• Input Tokens: {avg_input:.1f}
• Output Tokens: {avg_output:.1f}  
• Total Tokens: {avg_total:.1f}

Total Usage:
• Total Input: {total_input:,}
• Total Output: {total_output:,}
• Total All: {total_all:,}

Cache Impact:
• Input Hit Avg: {avg_input_hit:.1f}
• Input Miss Avg: {avg_input_miss:.1f}
• Output Hit Avg: {avg_output_hit:.1f}
• Output Miss Avg: {avg_output_miss:.1f}

Token Efficiency:
• Avg Ratio: {(avg_output/avg_input):.2f}
• Hit Ratio: {(avg_output_hit/avg_input_hit if avg_input_hit > 0 else 0):.2f}
• Miss Ratio: {(avg_output_miss/avg_input_miss if avg_input_miss > 0 else 0):.2f}"""
    
    axes[1, 2].text(0.05, 0.95, summary_text, transform=axes[1, 2].transAxes,
                    fontsize=font_size - 8, verticalalignment='top', 
                    fontfamily='monospace', bbox=dict(boxstyle="round,pad=0.3", 
                    facecolor="lightgray", alpha=0.8))
    
    plt.tight_layout(rect=[0, 0, 1, 0.96], pad=3.0, h_pad=4.0, w_pad=3.0)
    
    filename = (
        benchmark.output_folder_path
        + f"/token_usage_analysis_{benchmark.timestamp}.pdf"
    )
    plt.savefig(filename, format="pdf", bbox_inches="tight")
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

    # Token statistics (if available)
    token_stats = {}
    if "input_tokens_list" in df.columns and "output_tokens_list" in df.columns:
        avg_input_tokens = compute_avg_input_tokens_score(df["input_tokens_list"])
        avg_output_tokens = compute_avg_output_tokens_score(df["output_tokens_list"])
        avg_total_tokens = compute_total_tokens_score(df["input_tokens_list"], df["output_tokens_list"])
        
        total_input_tokens = int(df["input_tokens_list"].sum())
        total_output_tokens = int(df["output_tokens_list"].sum())
        total_all_tokens = total_input_tokens + total_output_tokens
        
        # Cache hit/miss token analysis
        cache_hit_mask = df["cache_hit_list"] > 0
        cache_miss_mask = df["cache_miss_list"] > 0
        
        input_tokens_hit = df.loc[cache_hit_mask, "input_tokens_list"]
        input_tokens_miss = df.loc[cache_miss_mask, "input_tokens_list"]
        output_tokens_hit = df.loc[cache_hit_mask, "output_tokens_list"]
        output_tokens_miss = df.loc[cache_miss_mask, "output_tokens_list"]
        
        token_stats = {
            "average_per_request": {
                "input_tokens": float(avg_input_tokens),
                "output_tokens": float(avg_output_tokens),
                "total_tokens": float(avg_total_tokens),
            },
            "total_usage": {
                "input_tokens": total_input_tokens,
                "output_tokens": total_output_tokens,
                "total_tokens": total_all_tokens,
            },
            "cache_impact": {
                "input_hit_avg": float(input_tokens_hit.mean()) if len(input_tokens_hit) > 0 else 0.0,
                "input_miss_avg": float(input_tokens_miss.mean()) if len(input_tokens_miss) > 0 else 0.0,
                "output_hit_avg": float(output_tokens_hit.mean()) if len(output_tokens_hit) > 0 else 0.0,
                "output_miss_avg": float(output_tokens_miss.mean()) if len(output_tokens_miss) > 0 else 0.0,
            },
            "efficiency": {
                "output_input_ratio": float(avg_output_tokens / avg_input_tokens) if avg_input_tokens > 0 else 0.0,
                "hit_ratio": float(output_tokens_hit.mean() / input_tokens_hit.mean()) if len(input_tokens_hit) > 0 and input_tokens_hit.mean() > 0 else 0.0,
                "miss_ratio": float(output_tokens_miss.mean() / input_tokens_miss.mean()) if len(input_tokens_miss) > 0 and input_tokens_miss.mean() > 0 else 0.0,
            }
        }

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
        "tokens": token_stats,
    }

    filename = benchmark.output_folder_path + f"/statistics_{benchmark.timestamp}.json"

    with open(filename, "w") as f:
        json.dump(statistics, f, indent=4)
