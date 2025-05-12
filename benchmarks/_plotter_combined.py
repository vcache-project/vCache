import json
import os
import re
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


# Module-level private helper functions
def __wilson_score_interval(successes: float, n: float, z: float = 1.96):
    """Computes the Wilson score interval for a binomial proportion."""
    if n == 0:
        return 0.0, 0.0

    p_hat = successes / n

    # Ensure p_hat is within [0, 1] to avoid domain errors with sqrt if successes > n or < 0
    p_hat = max(0.0, min(1.0, p_hat))

    denominator = 1 + z**2 / n
    center = (p_hat + z**2 / (2 * n)) / denominator

    sqrt_term_numerator = p_hat * (1 - p_hat) + (z**2 / (4 * n))
    # Ensure the term under the square root is non-negative
    if sqrt_term_numerator < 0:
        sqrt_term_numerator = 0

    width = (z * np.sqrt(sqrt_term_numerator / n)) / denominator

    lower_bound = center - width
    upper_bound = center + width

    return max(0.0, lower_bound), min(1.0, upper_bound)


def __collect_run_dirs_by_prefix_and_key(
    results_dir_path: str, dir_prefix_to_match: str
):
    """
    Collects full paths to run directories based on a prefix and extracts a normalized key.
    The key is typically a threshold or delta value formatted as a string.
    Example: results_dir_path contains "gptcache_0.8_run_1", "gptcache_0.8_run_2".
    If dir_prefix_to_match is "gptcache_", it will map {"0.8": [path_to_run1, path_to_run2]}.
    """
    mapping: Dict[str, List[str]] = {}
    if not os.path.exists(results_dir_path):
        return mapping

    for dir_name in os.listdir(results_dir_path):
        full_dir_path = os.path.join(results_dir_path, dir_name)
        if not dir_name.startswith(dir_prefix_to_match) or not os.path.isdir(
            full_dir_path
        ):
            continue

        name_part_after_prefix = dir_name[
            len(dir_prefix_to_match) :
        ]  # e.g., "0.8", "0.01_run_1"

        # Attempt to match a floating point number at the beginning of the remaining part.
        # This should capture values like "0.8", "0.01", "0.955" etc.
        # It handles cases like "0.8_run_1" by only taking "0.8".
        match = re.match(r"([0-9]+\.?[0-9]*|[0-9]*\.?[0-9]+)", name_part_after_prefix)

        if match:
            key_str_from_dir = match.group(1)
            try:
                float_val = float(key_str_from_dir)
                # Normalize the string representation to match keys from data_frames
                # e.g., 0.8 -> "0.8", 0.0 -> "0", 0.010 -> "0.01"
                normalized_key_str = f"{float_val:.3f}".rstrip("0").rstrip(".")
                if not normalized_key_str and float_val == 0:  # Handles 0.0 becoming ""
                    normalized_key_str = "0"
                elif (
                    not normalized_key_str and "." in key_str_from_dir
                ):  # e.g. if input was "0."
                    normalized_key_str = key_str_from_dir.rstrip("0").rstrip(".")
                    if not normalized_key_str:
                        normalized_key_str = "0"

                mapping.setdefault(normalized_key_str, []).append(full_dir_path)
            except ValueError:
                # print(f"Warning: Could not parse float from '{key_str_from_dir}' in dir '{dir_name}'")
                continue
        # else:
        # print(f"Warning: No numeric key found in '{name_part_after_prefix}' for dir '{dir_name}'")

    return mapping


def __draw_confidence_series(
    x_means: List[float],
    y_means: List[float],
    x_lower_errors: List[float],
    x_upper_errors: List[float],
    y_lower_errors: List[float],
    y_upper_errors: List[float],
    is_multi_run: List[bool],
    color: str,
    label: str,
    marker_size: int,
    line_style: str = "-",
    line_width: int = 3,
    error_bar_alpha: float = 0.6,
    error_bar_capsize: int = 4,
    error_bar_elinewidth: float = 1.5,
):
    """Plots a series with optional confidence interval error bars."""
    if not x_means:
        return

    plt.plot(
        x_means,
        y_means,
        line_style,
        color=color,
        linewidth=line_width,
        label=label,
        zorder=3,
    )

    for i in range(len(x_means)):
        xm, ym = x_means[i], y_means[i]
        is_multi = is_multi_run[i]

        current_marker_size = 5 if is_multi else marker_size
        plt.plot(xm, ym, "o", color=color, markersize=current_marker_size, zorder=10)

        if is_multi:
            # Prepare error lists for errorbar, ensuring they are 2xN arrays
            # xerr should be [[lower_errors], [upper_errors]]
            current_x_err = None
            if (
                x_lower_errors[i] > 1e-9 or x_upper_errors[i] > 1e-9
            ):  # Check for non-zero error
                current_x_err = [[x_lower_errors[i]], [x_upper_errors[i]]]

            current_y_err = None
            if y_lower_errors[i] > 1e-9 or y_upper_errors[i] > 1e-9:
                current_y_err = [[y_lower_errors[i]], [y_upper_errors[i]]]

            if current_x_err or current_y_err:  # Only plot if there is some error
                plt.errorbar(
                    xm,
                    ym,
                    xerr=current_x_err,
                    yerr=current_y_err,
                    fmt="none",
                    ecolor=color,
                    elinewidth=error_bar_elinewidth,
                    capsize=error_bar_capsize,
                    alpha=error_bar_alpha,
                    zorder=5,
                )


# Aggregation functions for different metrics
def __aggregate_stats_for_roc(run_dirs: List[str]):
    all_tp_values = []
    all_fn_values = []
    all_fp_values = []
    all_tn_values = []

    for rd_path in run_dirs:
        for file_name in os.listdir(rd_path):
            if file_name.startswith("results_") and file_name.endswith(".json"):
                file_path = os.path.join(rd_path, file_name)
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    df, _ = convert_to_dataframe_from_json_file(data)
                    all_tp_values.extend(df["tp_list"])
                    all_fn_values.extend(df["fn_list"])
                    all_fp_values.extend(df["fp_list"])
                    all_tn_values.extend(df["tn_list"])
                except Exception:
                    # print(f"Warning: Error reading/processing {file_path} for ROC stats: {e}")
                    continue

    total_tp = np.sum(all_tp_values)
    total_fn = np.sum(all_fn_values)
    total_fp = np.sum(all_fp_values)
    total_tn = np.sum(all_tn_values)

    tpr_successes = total_tp
    tpr_n = total_tp + total_fn
    if tpr_n == 0:
        tpr_mean, tpr_ci_low, tpr_ci_up = 0.0, 0.0, 0.0
    else:
        tpr_mean = tpr_successes / tpr_n
        tpr_ci_low, tpr_ci_up = __wilson_score_interval(tpr_successes, tpr_n)

    fpr_successes = total_fp
    fpr_n = total_fp + total_tn
    if fpr_n == 0:
        fpr_mean, fpr_ci_low, fpr_ci_up = 0.0, 0.0, 0.0
    else:
        fpr_mean = fpr_successes / fpr_n
        fpr_ci_low, fpr_ci_up = __wilson_score_interval(fpr_successes, fpr_n)

    return tpr_mean, tpr_ci_low, tpr_ci_up, fpr_mean, fpr_ci_low, fpr_ci_up


def __aggregate_stats_for_latency_error(run_dirs: List[str], z: float = 1.96):
    all_fp_values = []
    all_latency_values = []
    num_samples_for_fp_calc = 0

    for rd_path in run_dirs:
        for file_name in os.listdir(rd_path):
            if file_name.startswith("results_") and file_name.endswith(".json"):
                file_path = os.path.join(rd_path, file_name)
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    df, _ = convert_to_dataframe_from_json_file(data)
                    all_fp_values.extend(df["fp_list"])
                    all_latency_values.extend(df["latency_vectorq_list"])
                    num_samples_for_fp_calc += len(df["fp_list"])
                except Exception:
                    # print(f"Warning: Error reading/processing {file_path} for Latency/Error stats: {e}")
                    continue

    total_fp = np.sum(all_fp_values)
    if num_samples_for_fp_calc == 0:
        err_mean, err_ci_low, err_ci_up = 0.0, 0.0, 0.0
    else:
        err_mean = total_fp / num_samples_for_fp_calc
        err_ci_low, err_ci_up = __wilson_score_interval(
            total_fp, num_samples_for_fp_calc
        )

    if not all_latency_values:
        lat_mean, lat_ci_low, lat_ci_up = 0.0, 0.0, 0.0
    else:
        latencies_np = np.array(all_latency_values)
        lat_mean = np.mean(latencies_np)
        num_lat_samples = len(latencies_np)
        if num_lat_samples <= 1:
            lat_ci_low, lat_ci_up = lat_mean, lat_mean
        else:
            lat_std_dev = np.std(latencies_np, ddof=1)
            lat_sem = lat_std_dev / np.sqrt(num_lat_samples)
            ci_half_width = z * lat_sem
            lat_ci_low = lat_mean - ci_half_width
            lat_ci_up = lat_mean + ci_half_width

    return err_mean, err_ci_low, err_ci_up, lat_mean, lat_ci_low, lat_ci_up


def __aggregate_stats_for_cache_hit_error_rate(run_dirs: List[str]):
    total_samples = 0
    total_cache_hits = 0
    total_fp = 0

    for rd in run_dirs:
        for f_name in os.listdir(rd):
            if not (f_name.startswith("results_") and f_name.endswith(".json")):
                continue
            file_path = os.path.join(rd, f_name)
            try:
                with open(file_path, "r", encoding="utf-8") as fp_file:
                    data = json.load(fp_file)
                df, _ = convert_to_dataframe_from_json_file(data)
                total_samples += len(df)
                total_cache_hits += int(np.sum(df["cache_hit_list"]))
                total_fp += int(np.sum(df["fp_list"]))
            except Exception:
                # print(f"Warning: Error reading/processing {file_path} for CacheHit/Error stats: {e}")
                continue

    if total_samples == 0:
        return {  # Return structure with zeros if no data
            "cache_hit_rate_mean": 0.0,
            "cache_hit_rate_ci_low": 0.0,
            "cache_hit_rate_ci_up": 0.0,
            "error_rate_mean": 0.0,
            "error_rate_ci_low": 0.0,
            "error_rate_ci_up": 0.0,
        }

    ch_mean = total_cache_hits / total_samples
    ch_ci_low, ch_ci_up = __wilson_score_interval(total_cache_hits, total_samples)

    er_mean = total_fp / total_samples
    er_ci_low, er_ci_up = __wilson_score_interval(total_fp, total_samples)

    return {
        "cache_hit_rate_mean": ch_mean,
        "cache_hit_rate_ci_low": ch_ci_low,
        "cache_hit_rate_ci_up": ch_ci_up,
        "error_rate_mean": er_mean,
        "error_rate_ci_low": er_ci_low,
        "error_rate_ci_up": er_ci_up,
    }


# End of new helper functions


def __get_result_files(results_dir: str):
    if not os.path.exists(results_dir):
        print(f"No results found in {results_dir}")
        return [], [], [], [], []

    gptcache_files: List[str] = []
    vcache_local_files: List[str] = []
    vcache_global_files: List[str] = []
    berkeley_embedding_files: List[str] = []
    vcache_berkeley_embedding_files: List[str] = []

    for d in os.listdir(results_dir):
        # Process GPTCache (static threshold) directories
        if d.startswith("gptcache_") and os.path.isdir(os.path.join(results_dir, d)):
            dir_path: str = os.path.join(results_dir, d)
            for file in os.listdir(dir_path):
                if file.startswith("results_") and file.endswith(".json"):
                    gptcache_files.append(os.path.join(dir_path, file))

        # Process vCache local directories
        elif d.startswith("vcache_local_") and os.path.isdir(
            os.path.join(results_dir, d)
        ):
            dir_path: str = os.path.join(results_dir, d)
            for file in os.listdir(dir_path):
                if file.startswith("results_") and file.endswith(".json"):
                    vcache_local_files.append(os.path.join(dir_path, file))

        # Process vCache global directories
        elif d.startswith("vcache_global_") and os.path.isdir(
            os.path.join(results_dir, d)
        ):
            dir_path: str = os.path.join(results_dir, d)
            for file in os.listdir(dir_path):
                if file.startswith("results_") and file.endswith(".json"):
                    vcache_global_files.append(os.path.join(dir_path, file))

        # Process Fine-tuned Embedding directories
        elif d.startswith("berkeley_embedding_") and os.path.isdir(
            os.path.join(results_dir, d)
        ):
            dir_path: str = os.path.join(results_dir, d)
            for file in os.listdir(dir_path):
                if file.startswith("results_") and file.endswith(".json"):
                    berkeley_embedding_files.append(os.path.join(dir_path, file))

        # Process vCache Fine-tuned Embedding directories
        elif d.startswith("vcache_berkeley_embedding_") and os.path.isdir(
            os.path.join(results_dir, d)
        ):
            dir_path: str = os.path.join(results_dir, d)
            for file in os.listdir(dir_path):
                if file.startswith("results_") and file.endswith(".json"):
                    vcache_berkeley_embedding_files.append(os.path.join(dir_path, file))

    return (
        gptcache_files,
        vcache_local_files,
        vcache_global_files,
        berkeley_embedding_files,
        vcache_berkeley_embedding_files,
    )


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

    (
        gptcache_files,
        vcache_local_files,
        vcache_global_files,
        berkeley_embedding_files,
        vcache_berkeley_embedding_files,
    ) = __get_result_files(results_dir)

    if (
        not gptcache_files
        and not vcache_local_files
        and not vcache_global_files
        and not berkeley_embedding_files
        and not vcache_berkeley_embedding_files
    ):
        print(
            f"No folders found for {dataset}, {embedding_model_name}, {llm_model_name}\n"
            f"in {results_dir}"
        )
        return

    ############################################################
    ### Baseline: GPTCache
    gptcache_data_frames: Dict[float, pd.DataFrame] = {}
    for gptcache_file_path in gptcache_files:
        with open(gptcache_file_path, "r") as f:
            data: Any = json.load(f)
            dataframe, _ = convert_to_dataframe_from_json_file(data)
            threshold: float = data["config"]["threshold"]
            gptcache_data_frames[threshold] = dataframe

    ############################################################
    ### Baseline: vCache Local
    vcache_local_data_frames: Dict[float, pd.DataFrame] = {}
    for vcache_local_file_path in vcache_local_files:
        with open(vcache_local_file_path, "r") as f:
            data: Any = json.load(f)
            dataframe, _ = convert_to_dataframe_from_json_file(data)
            delta: float = data["config"]["delta"]
            vcache_local_data_frames[delta] = dataframe

    ############################################################
    ### Baseline: vCache Global
    vcache_global_data_frames: Dict[float, pd.DataFrame] = {}
    for vcache_global_file_path in vcache_global_files:
        with open(vcache_global_file_path, "r") as f:
            try:
                data: Any = json.load(f)
                dataframe, _ = convert_to_dataframe_from_json_file(data)
                delta: float = data["config"]["delta"]
                vcache_global_data_frames[delta] = dataframe
            except Exception as e:
                print(f"Error loading {vcache_global_file_path}: {e}")
                continue

    ############################################################
    ### Baseline: Fine-tuned Embedding
    berkeley_embedding_data_frames: Dict[float, pd.DataFrame] = {}
    for berkeley_embedding_file_path in berkeley_embedding_files:
        with open(berkeley_embedding_file_path, "r") as f:
            try:
                data: Any = json.load(f)
                dataframe, _ = convert_to_dataframe_from_json_file(data)
                threshold: float = data["config"]["threshold"]
                berkeley_embedding_data_frames[threshold] = dataframe
            except Exception as e:
                print(f"Error loading {berkeley_embedding_file_path}: {e}")
                continue

    ############################################################
    ### vCache + Fine-tuned Embedding
    vcache_berkeley_embedding_data_frames: Dict[float, pd.DataFrame] = {}
    for vcache_berkeley_embedding_file_path in vcache_berkeley_embedding_files:
        with open(vcache_berkeley_embedding_file_path, "r") as f:
            try:
                data: Any = json.load(f)
                dataframe, _ = convert_to_dataframe_from_json_file(data)
                delta: float = data["config"]["delta"]
                vcache_berkeley_embedding_data_frames[delta] = dataframe
            except Exception as e:
                print(f"Error loading {vcache_berkeley_embedding_file_path}: {e}")
                continue

    __plot_legend(
        gptcache_data_frames=gptcache_data_frames,
        vcache_local_data_frames=vcache_local_data_frames,
        vcache_global_data_frames=vcache_global_data_frames,
        berkeley_embedding_data_frames=berkeley_embedding_data_frames,
        vcache_berkeley_embedding_data_frames=vcache_berkeley_embedding_data_frames,
        results_dir=results_dir,
        timestamp=timestamp,
        font_size=font_size,
    )

    try:
        __plot_roc(
            gptcache_data_frames=gptcache_data_frames,
            vcache_local_data_frames=vcache_local_data_frames,
            vcache_global_data_frames=vcache_global_data_frames,
            berkeley_embedding_data_frames=berkeley_embedding_data_frames,
            vcache_berkeley_embedding_data_frames=vcache_berkeley_embedding_data_frames,
            results_dir=results_dir,
            timestamp=timestamp,
            font_size=font_size,
        )
    except Exception as e:
        print(f"Error plotting ROC: {e}")

    try:
        __plot_precision_vs_recall(
            gptcache_data_frames=gptcache_data_frames,
            vcache_local_data_frames=vcache_local_data_frames,
            vcache_global_data_frames=vcache_global_data_frames,
            berkeley_embedding_data_frames=berkeley_embedding_data_frames,
            vcache_berkeley_embedding_data_frames=vcache_berkeley_embedding_data_frames,
            results_dir=results_dir,
            timestamp=timestamp,
            font_size=font_size,
        )
    except Exception as e:
        print(f"Error plotting precision vs recall: {e}")

    try:
        __plot_avg_latency_vs_error_rate(
            gptcache_data_frames=gptcache_data_frames,
            vcache_local_data_frames=vcache_local_data_frames,
            vcache_global_data_frames=vcache_global_data_frames,
            berkeley_embedding_data_frames=berkeley_embedding_data_frames,
            vcache_berkeley_embedding_data_frames=vcache_berkeley_embedding_data_frames,
            results_dir=results_dir,
            timestamp=timestamp,
            font_size=font_size,
        )
    except Exception as e:
        print(f"Error plotting avg latency vs error rate: {e}")

    try:
        __plot_cache_hit_vs_error_rate(
            gptcache_data_frames=gptcache_data_frames,
            vcache_local_data_frames=vcache_local_data_frames,
            vcache_global_data_frames=vcache_global_data_frames,
            berkeley_embedding_data_frames=berkeley_embedding_data_frames,
            vcache_berkeley_embedding_data_frames=vcache_berkeley_embedding_data_frames,
            results_dir=results_dir,
            timestamp=timestamp,
            font_size=font_size,
        )
    except Exception as e:
        print(f"Error plotting cache hit vs error rate: {e}")

    # try:
    __plot_cache_hit_vs_error_rate_vs_sample_size(
        gptcache_data_frames=gptcache_data_frames,
        vcache_local_data_frames=vcache_local_data_frames,
        vcache_global_data_frames=vcache_global_data_frames,
        berkeley_embedding_data_frames=berkeley_embedding_data_frames,
        vcache_berkeley_embedding_data_frames=vcache_berkeley_embedding_data_frames,
        results_dir=results_dir,
        timestamp=timestamp,
        font_size=font_size,
    )
    # except Exception as e:
    #     print(f"Error plotting cache hit vs error rate vs sample size: {e}")

    try:
        __plot_delta_accuracy(
            vcache_local_data_frames=vcache_local_data_frames,
            vcache_global_data_frames=vcache_global_data_frames,
            vcache_berkeley_embedding_data_frames=vcache_berkeley_embedding_data_frames,
            results_dir=results_dir,
            timestamp=timestamp,
            font_size=font_size,
        )
    except Exception as e:
        print(f"Error plotting delta accuracy: {e}")


def __plot_legend(
    gptcache_data_frames: Dict[float, pd.DataFrame],
    vcache_local_data_frames: Dict[float, pd.DataFrame],
    vcache_global_data_frames: Dict[float, pd.DataFrame],
    berkeley_embedding_data_frames: Dict[float, pd.DataFrame],
    vcache_berkeley_embedding_data_frames: Dict[float, pd.DataFrame],
    results_dir: str,
    timestamp: str,
    font_size: int,
):
    figlegend = plt.figure(figsize=(12, 1.5))
    ax = figlegend.add_subplot(111)
    ax.axis("off")

    lines = []
    labels = []

    if gptcache_data_frames:
        lines.append(
            Line2D(
                [0],
                [0],
                color="#C23B48",
                linewidth=3,
                linestyle="-",
                marker="o",
                markersize=8,
            )
        )
        labels.append("GPTCache")

    if vcache_local_data_frames:
        lines.append(
            Line2D(
                [0],
                [0],
                color="#37A9EC",
                linewidth=3,
                linestyle="-",
                marker="o",
                markersize=8,
            )
        )
        labels.append("vCache")

    if vcache_global_data_frames:
        lines.append(
            Line2D(
                [0],
                [0],
                color="#8CBE94",
                linewidth=3,
                linestyle="-",
                marker="o",
                markersize=8,
            )
        )
        labels.append("vCache (Ablation)")

    if vcache_berkeley_embedding_data_frames:
        lines.append(
            Line2D(
                [0],
                [0],
                color="#EDBE24",
                linewidth=3,
                linestyle="-",
                marker="o",
                markersize=8,
            )
        )
        labels.append("vCache + Fine-tuned Embedding")

    if berkeley_embedding_data_frames:
        lines.append(
            Line2D(
                [0],
                [0],
                color="#3B686A",
                linewidth=3,
                linestyle="-",
                marker="o",
                markersize=8,
            )
        )
        labels.append("Fine-tuned Embedding")

    ax.legend(lines, labels, loc="center", ncol=2, fontsize=font_size, frameon=False)

    legend_filename = results_dir + "/legend.pdf"
    figlegend.savefig(
        legend_filename, format="pdf", bbox_inches="tight", transparent=True
    )
    plt.close(figlegend)

    lines.append(Line2D([0], [0], color="grey", linewidth=3, linestyle="--", alpha=0.7))
    labels.append("Random Classifier")

    lines.append(Line2D([0], [0], color="grey", linewidth=3, linestyle="-"))
    labels.append("No Cache")

    ax.legend(lines, labels, loc="center", ncol=3, fontsize=font_size, frameon=False)

    legend_filename = results_dir + "/legend_w_rnd_class.pdf"
    figlegend.savefig(
        legend_filename, format="pdf", bbox_inches="tight", transparent=True
    )
    plt.close(figlegend)


def __plot_roc(
    gptcache_data_frames: Dict[float, pd.DataFrame],
    vcache_local_data_frames: Dict[float, pd.DataFrame],
    vcache_global_data_frames: Dict[float, pd.DataFrame],
    berkeley_embedding_data_frames: Dict[float, pd.DataFrame],
    vcache_berkeley_embedding_data_frames: Dict[float, pd.DataFrame],
    results_dir: str,
    timestamp: str,
    font_size: int,
):
    plt.figure(figsize=(12, 10))

    # Collect all run directories once
    # base_results_dir = os.path.dirname(results_dir.rstrip('/')) # Get parent of timestamped dir if any
    # Or, if results_dir is the one containing gptcache_X.Y folders:
    # base_results_dir = results_dir # This should be correct based on how results_dir is constructed

    gptcache_run_dirs_map = __collect_run_dirs_by_prefix_and_key(
        results_dir, "gptcache_"
    )
    vcache_local_run_dirs_map = __collect_run_dirs_by_prefix_and_key(
        results_dir, "vcache_local_"
    )
    vcache_global_run_dirs_map = __collect_run_dirs_by_prefix_and_key(
        results_dir, "vcache_global_"
    )
    berkeley_embedding_run_dirs_map = __collect_run_dirs_by_prefix_and_key(
        results_dir, "berkeley_embedding_"
    )
    vcache_berkeley_embedding_run_dirs_map = __collect_run_dirs_by_prefix_and_key(
        results_dir, "vcache_berkeley_embedding_"
    )

    plt.plot(
        [0, 1],
        [0, 1],
        "--",
        color="grey",
        alpha=0.7,
        linewidth=3,
        label="Random Classifier",
    )

    # Helper to prepare data for a series
    def prepare_roc_series_data(data_frames, run_dirs_map_for_series):
        thresholds_or_deltas = sorted(data_frames.keys())
        tpr_means, fpr_means = [], []
        tpr_low_err, tpr_up_err = [], []
        fpr_low_err, fpr_up_err = [], []
        is_multi = []

        for key_val in thresholds_or_deltas:
            key_str = f"{key_val:.3f}".rstrip("0").rstrip(".")
            if not key_str and key_val == 0:
                key_str = "0"

            current_run_dirs = run_dirs_map_for_series.get(key_str, [])

            if len(current_run_dirs) > 1:  # Multi-run
                tpr_m, tpr_l, tpr_u, fpr_m, fpr_l, fpr_u = __aggregate_stats_for_roc(
                    current_run_dirs
                )
                tpr_means.append(tpr_m)
                fpr_means.append(fpr_m)
                tpr_low_err.append(tpr_m - tpr_l)
                tpr_up_err.append(tpr_u - tpr_m)
                fpr_low_err.append(fpr_m - fpr_l)
                fpr_up_err.append(fpr_u - fpr_m)
                is_multi.append(True)
            elif (
                data_frames
            ):  # Single run (or fallback to single file if no dirs found but df exists)
                df = data_frames[key_val]
                tpr_m = compute_recall_score(tp=df["tp_list"], fn=df["fn_list"])
                fpr_m = compute_false_positive_rate_score(
                    fp=df["fp_list"], tn=df["tn_list"]
                )
                tpr_means.append(tpr_m)
                fpr_means.append(fpr_m)
                tpr_low_err.append(0)
                tpr_up_err.append(0)
                fpr_low_err.append(0)
                fpr_up_err.append(0)
                is_multi.append(False)

        # Ensure errors are non-negative
        tpr_low_err = [max(0, err) for err in tpr_low_err]
        tpr_up_err = [max(0, err) for err in tpr_up_err]
        fpr_low_err = [max(0, err) for err in fpr_low_err]
        fpr_up_err = [max(0, err) for err in fpr_up_err]

        return (
            fpr_means,
            tpr_means,
            fpr_low_err,
            fpr_up_err,
            tpr_low_err,
            tpr_up_err,
            is_multi,
        )

    ############################################################
    ### Baseline: GPTCache
    if gptcache_data_frames:
        gpt_fpr, gpt_tpr, gpt_fpr_le, gpt_fpr_ue, gpt_tpr_le, gpt_tpr_ue, gpt_multi = (
            prepare_roc_series_data(gptcache_data_frames, gptcache_run_dirs_map)
        )
        __draw_confidence_series(
            gpt_fpr,
            gpt_tpr,
            gpt_fpr_le,
            gpt_fpr_ue,
            gpt_tpr_le,
            gpt_tpr_ue,
            gpt_multi,
            "#C23B48",
            "GPTCache",
            10,
        )

    ############################################################
    ### Baseline: vCache Local
    if vcache_local_data_frames:
        vl_fpr, vl_tpr, vl_fpr_le, vl_fpr_ue, vl_tpr_le, vl_tpr_ue, vl_multi = (
            prepare_roc_series_data(vcache_local_data_frames, vcache_local_run_dirs_map)
        )
        __draw_confidence_series(
            vl_fpr,
            vl_tpr,
            vl_fpr_le,
            vl_fpr_ue,
            vl_tpr_le,
            vl_tpr_ue,
            vl_multi,
            "#37A9EC",
            "vCache",
            10,
        )

    ############################################################
    ### Baseline: vCache Global
    if vcache_global_data_frames:
        vg_fpr, vg_tpr, vg_fpr_le, vg_fpr_ue, vg_tpr_le, vg_tpr_ue, vg_multi = (
            prepare_roc_series_data(
                vcache_global_data_frames, vcache_global_run_dirs_map
            )
        )
        __draw_confidence_series(
            vg_fpr,
            vg_tpr,
            vg_fpr_le,
            vg_fpr_ue,
            vg_tpr_le,
            vg_tpr_ue,
            vg_multi,
            "#8CBE94",
            "vCache (Ablation)",
            10,
        )

    ############################################################
    ### Baseline: Fine-tuned Embedding
    if berkeley_embedding_data_frames:
        be_fpr, be_tpr, be_fpr_le, be_fpr_ue, be_tpr_le, be_tpr_ue, be_multi = (
            prepare_roc_series_data(
                berkeley_embedding_data_frames, berkeley_embedding_run_dirs_map
            )
        )
        __draw_confidence_series(
            be_fpr,
            be_tpr,
            be_fpr_le,
            be_fpr_ue,
            be_tpr_le,
            be_tpr_ue,
            be_multi,
            "#3B686A",
            "Fine-tuned Embedding",
            10,
        )

    ############################################################
    ### vCache + Fine-tuned Embedding
    if vcache_berkeley_embedding_data_frames:
        vb_fpr, vb_tpr, vb_fpr_le, vb_fpr_ue, vb_tpr_le, vb_tpr_ue, vb_multi = (
            prepare_roc_series_data(
                vcache_berkeley_embedding_data_frames,
                vcache_berkeley_embedding_run_dirs_map,
            )
        )
        __draw_confidence_series(
            vb_fpr,
            vb_tpr,
            vb_fpr_le,
            vb_fpr_ue,
            vb_tpr_le,
            vb_tpr_ue,
            vb_multi,
            "#EDBE24",
            "vCache + Fine-tuned Embedding",
            10,
        )

    plt.xlabel("FPR", fontsize=font_size)
    plt.ylabel("TPR", fontsize=font_size)
    plt.tick_params(axis="both", labelsize=font_size - 2)

    plt.xlim(0, 1)
    plt.ylim(0, 1)
    yticks = plt.yticks()[0]
    if yticks[0] == 0.0:
        plt.yticks(yticks[1:])

    plt.gca().spines["top"].set_linewidth(1)
    plt.gca().spines["right"].set_linewidth(1)
    plt.gca().spines["bottom"].set_linewidth(1)
    plt.gca().spines["left"].set_linewidth(1)

    filename = results_dir + "/roc.pdf"
    plt.savefig(filename, format="pdf", bbox_inches="tight", transparent=True)
    plt.close()


def __plot_precision_vs_recall(
    gptcache_data_frames: Dict[float, pd.DataFrame],
    vcache_local_data_frames: Dict[float, pd.DataFrame],
    vcache_global_data_frames: Dict[float, pd.DataFrame],
    berkeley_embedding_data_frames: Dict[float, pd.DataFrame],
    vcache_berkeley_embedding_data_frames: Dict[float, pd.DataFrame],
    results_dir: str,
    timestamp: str,
    font_size: int,
):
    plt.figure(figsize=(12, 8))

    ############################################################
    ### Baseline: GPTCache
    gptcache_thresholds = sorted(gptcache_data_frames.keys())
    gptcache_precision_values = []
    gptcache_recall_values = []

    for threshold in gptcache_thresholds:
        df = gptcache_data_frames[threshold]
        precision = compute_precision_score(tp=df["tp_list"], fp=df["fp_list"])
        recall = compute_recall_score(tp=df["tp_list"], fn=df["fn_list"])

        gptcache_precision_values.append(precision)
        gptcache_recall_values.append(recall)

    if gptcache_thresholds:
        plt.plot(
            gptcache_recall_values,
            gptcache_precision_values,
            "o-",
            color="#C23B48",
            linewidth=3,
            label="GPTCache",
            markersize=8,
        )

        for i, threshold in enumerate(gptcache_thresholds):
            plt.annotate(
                text="",
                xy=(gptcache_recall_values[i], gptcache_precision_values[i]),
                textcoords="offset points",
                xytext=(0, 10),
                ha="center",
                fontsize=font_size - 4,
            )

    ############################################################
    ### Baseline: vCache Local
    vcache_local_deltas = sorted(vcache_local_data_frames.keys())
    vcache_local_precision_values = []
    vcache_local_recall_values = []

    for delta in vcache_local_deltas:
        df = vcache_local_data_frames[delta]
        precision = compute_precision_score(tp=df["tp_list"], fp=df["fp_list"])
        recall = compute_recall_score(tp=df["tp_list"], fn=df["fn_list"])

        vcache_local_precision_values.append(precision)
        vcache_local_recall_values.append(recall)

    if vcache_local_deltas:
        plt.plot(
            vcache_local_recall_values,
            vcache_local_precision_values,
            "o-",
            color="#37A9EC",
            linewidth=3,
            label="vCache",
            markersize=8,
        )

        for i, _ in enumerate(vcache_local_precision_values):
            plt.annotate(
                text="",
                xy=(vcache_local_recall_values[i], vcache_local_precision_values[i]),
                textcoords="offset points",
                xytext=(0, 10),
                ha="center",
                fontsize=font_size - 4,
            )

    ############################################################
    ### Baseline: vCache Global
    vcache_global_deltas = sorted(vcache_global_data_frames.keys())
    vcache_global_precision_values = []
    vcache_global_recall_values = []

    for delta in vcache_global_deltas:
        df = vcache_global_data_frames[delta]
        precision = compute_precision_score(tp=df["tp_list"], fp=df["fp_list"])
        recall = compute_recall_score(tp=df["tp_list"], fn=df["fn_list"])

        vcache_global_precision_values.append(precision)
        vcache_global_recall_values.append(recall)

    if vcache_global_deltas:
        plt.plot(
            vcache_global_recall_values,
            vcache_global_precision_values,
            "o-",
            color="#8CBE94",
            linewidth=3,
            label="vCache (Ablation)",
            markersize=8,
        )

        for i, delta in enumerate(vcache_global_deltas):
            plt.annotate(
                text="",
                xy=(vcache_global_recall_values[i], vcache_global_precision_values[i]),
                textcoords="offset points",
                xytext=(0, 10),
                ha="center",
                fontsize=font_size - 4,
            )

    ############################################################
    ### Baseline: Fine-tuned Embedding
    berkeley_embedding_thresholds = sorted(berkeley_embedding_data_frames.keys())
    berkeley_embedding_precision_values = []
    berkeley_embedding_recall_values = []

    for threshold in berkeley_embedding_thresholds:
        df = berkeley_embedding_data_frames[threshold]

        precision = compute_precision_score(tp=df["tp_list"], fp=df["fp_list"])
        recall = compute_recall_score(tp=df["tp_list"], fn=df["fn_list"])

        berkeley_embedding_precision_values.append(precision)
        berkeley_embedding_recall_values.append(recall)

    if berkeley_embedding_thresholds:
        plt.plot(
            berkeley_embedding_recall_values,
            berkeley_embedding_precision_values,
            "o-",
            color="#3B686A",
            linewidth=3,
            label="Fine-tuned Embedding",
            markersize=8,
        )

        for i, threshold in enumerate(berkeley_embedding_thresholds):
            plt.annotate(
                text="",
                xy=(
                    berkeley_embedding_recall_values[i],
                    berkeley_embedding_precision_values[i],
                ),
                textcoords="offset points",
                xytext=(0, 10),
                ha="center",
                fontsize=font_size - 4,
            )

    ############################################################
    ### vCache + Fine-tuned Embedding
    vcache_berkeley_embedding_thresholds = sorted(
        vcache_berkeley_embedding_data_frames.keys()
    )
    vcache_berkeley_embedding_precision_values = []
    vcache_berkeley_embedding_recall_values = []

    for delta in vcache_berkeley_embedding_thresholds:
        df = vcache_berkeley_embedding_data_frames[delta]

        precision = compute_precision_score(tp=df["tp_list"], fp=df["fp_list"])
        recall = compute_recall_score(tp=df["tp_list"], fn=df["fn_list"])

        vcache_berkeley_embedding_precision_values.append(precision)
        vcache_berkeley_embedding_recall_values.append(recall)

    if vcache_berkeley_embedding_thresholds:
        plt.plot(
            vcache_berkeley_embedding_recall_values,
            vcache_berkeley_embedding_precision_values,
            "o-",
            color="#EDBE24",
            linewidth=3,
            label="vCache + Fine-tuned Embedding",
            markersize=8,
        )

        for i, delta in enumerate(vcache_berkeley_embedding_thresholds):
            plt.annotate(
                text="",
                xy=(
                    vcache_berkeley_embedding_recall_values[i],
                    vcache_berkeley_embedding_precision_values[i],
                ),
                textcoords="offset points",
                xytext=(0, 10),
                ha="center",
                fontsize=font_size - 4,
            )

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel("Recall", fontsize=font_size)
    plt.ylabel("Precision", fontsize=font_size)
    plt.tick_params(axis="both", labelsize=font_size - 2)

    yticks = plt.yticks()[0]
    if yticks[0] == 0.0:
        plt.yticks(yticks[1:])

    plt.gca().spines["top"].set_linewidth(1)
    plt.gca().spines["right"].set_linewidth(1)
    plt.gca().spines["bottom"].set_linewidth(1)
    plt.gca().spines["left"].set_linewidth(1)

    filename = results_dir + "/precision_vs_recall.pdf"
    plt.savefig(filename, format="pdf", bbox_inches="tight", transparent=True)
    plt.close()


def __plot_avg_latency_vs_error_rate(
    gptcache_data_frames: Dict[float, pd.DataFrame],
    vcache_local_data_frames: Dict[float, pd.DataFrame],
    vcache_global_data_frames: Dict[float, pd.DataFrame],
    berkeley_embedding_data_frames: Dict[float, pd.DataFrame],
    vcache_berkeley_embedding_data_frames: Dict[float, pd.DataFrame],
    results_dir: str,
    timestamp: str,
    font_size: int,
):
    plt.figure(figsize=(12, 10))
    ERROR_RATE_UPPER_BOUND = 6  # 6%

    # Collect all run directories once
    gptcache_run_dirs_map = __collect_run_dirs_by_prefix_and_key(
        results_dir, "gptcache_"
    )
    vcache_local_run_dirs_map = __collect_run_dirs_by_prefix_and_key(
        results_dir, "vcache_local_"
    )
    vcache_global_run_dirs_map = __collect_run_dirs_by_prefix_and_key(
        results_dir, "vcache_global_"
    )
    berkeley_embedding_run_dirs_map = __collect_run_dirs_by_prefix_and_key(
        results_dir, "berkeley_embedding_"
    )
    vcache_berkeley_embedding_run_dirs_map = __collect_run_dirs_by_prefix_and_key(
        results_dir, "vcache_berkeley_embedding_"
    )

    avg_latency_no_cache = -1  # Initialize

    # Helper to prepare data for latency vs error rate
    def prepare_latency_error_series_data(
        data_frames, run_dirs_map_for_series, error_rate_bound
    ):
        keys = sorted(data_frames.keys())
        err_means, lat_means = [], []
        err_low_err, err_up_err = [], []
        lat_low_err, lat_up_err = [], []
        is_multi = []
        nonlocal avg_latency_no_cache  # To update from the first available df

        for key_val in keys:
            key_str = f"{key_val:.3f}".rstrip("0").rstrip(".")
            if not key_str and key_val == 0:
                key_str = "0"

            current_run_dirs = run_dirs_map_for_series.get(key_str, [])

            current_err_m, current_lat_m = -1, -1  # Init for filtering

            if len(current_run_dirs) > 1:  # Multi-run
                err_m, err_l, err_u, lat_m, lat_l, lat_u = (
                    __aggregate_stats_for_latency_error(current_run_dirs)
                )
                current_err_m = err_m * 100  # Convert to percentage
                current_lat_m = lat_m

                if current_err_m <= error_rate_bound:
                    err_means.append(current_err_m)
                    lat_means.append(current_lat_m)
                    err_low_err.append((err_m - err_l) * 100)
                    err_up_err.append((err_u - err_m) * 100)
                    lat_low_err.append(lat_m - lat_l)
                    lat_up_err.append(lat_u - lat_m)
                    is_multi.append(True)
            elif data_frames:  # Single run
                df = data_frames[key_val]
                err_m_single = compute_error_rate_score(fp=df["fp_list"]) * 100
                lat_m_single = compute_avg_latency_score(
                    latency_list=df["latency_vectorq_list"]
                )
                current_err_m = err_m_single
                current_lat_m = lat_m_single

                if current_err_m <= error_rate_bound:
                    err_means.append(current_err_m)
                    lat_means.append(current_lat_m)
                    err_low_err.append(0)
                    err_up_err.append(0)
                    lat_low_err.append(0)
                    lat_up_err.append(0)
                    is_multi.append(False)

            # Update avg_latency_no_cache from the first valid df processed
            if avg_latency_no_cache == -1 and data_frames and key_val in data_frames:
                df_for_no_cache = data_frames[key_val]
                if (
                    "latency_direct_list" in df_for_no_cache.columns
                    and not df_for_no_cache["latency_direct_list"].empty
                ):
                    avg_latency_no_cache = compute_avg_latency_score(
                        latency_list=df_for_no_cache["latency_direct_list"]
                    )

        # Ensure errors are non-negative
        err_low_err = [max(0, err) for err in err_low_err]
        err_up_err = [max(0, err) for err in err_up_err]
        lat_low_err = [max(0, err) for err in lat_low_err]
        lat_up_err = [max(0, err) for err in lat_up_err]

        return (
            lat_means,
            err_means,
            lat_low_err,
            lat_up_err,
            err_low_err,
            err_up_err,
            is_multi,
        )

    ############################################################
    ### Baseline: GPTCache
    if gptcache_data_frames:
        gpt_lat, gpt_err, gpt_lat_le, gpt_lat_ue, gpt_err_le, gpt_err_ue, gpt_multi = (
            prepare_latency_error_series_data(
                gptcache_data_frames, gptcache_run_dirs_map, ERROR_RATE_UPPER_BOUND
            )
        )
        # Plotting x=latency, y=error_rate
        __draw_confidence_series(
            gpt_lat,
            gpt_err,
            gpt_lat_le,
            gpt_lat_ue,
            gpt_err_le,
            gpt_err_ue,
            gpt_multi,
            "#C23B48",
            "GPTCache",
            8,
        )

    ############################################################
    ### Baseline: No Cache (plotted after at least one series to get avg_latency_no_cache)
    # This logic is kept similar to original, plotted as a vertical line.

    ############################################################
    ### Baseline: vCache Local
    if vcache_local_data_frames:
        vl_lat, vl_err, vl_lat_le, vl_lat_ue, vl_err_le, vl_err_ue, vl_multi = (
            prepare_latency_error_series_data(
                vcache_local_data_frames,
                vcache_local_run_dirs_map,
                ERROR_RATE_UPPER_BOUND,
            )
        )
        __draw_confidence_series(
            vl_lat,
            vl_err,
            vl_lat_le,
            vl_lat_ue,
            vl_err_le,
            vl_err_ue,
            vl_multi,
            "#37A9EC",
            "vCache",
            8,
        )

    ############################################################
    ### Baseline: vCache Global
    if vcache_global_data_frames:
        vg_lat, vg_err, vg_lat_le, vg_lat_ue, vg_err_le, vg_err_ue, vg_multi = (
            prepare_latency_error_series_data(
                vcache_global_data_frames,
                vcache_global_run_dirs_map,
                ERROR_RATE_UPPER_BOUND,
            )
        )
        __draw_confidence_series(
            vg_lat,
            vg_err,
            vg_lat_le,
            vg_lat_ue,
            vg_err_le,
            vg_err_ue,
            vg_multi,
            "#8CBE94",
            "vCache (Ablation)",
            8,
        )

    ############################################################
    ### Baseline: Fine-tuned Embedding
    if berkeley_embedding_data_frames:
        be_lat, be_err, be_lat_le, be_lat_ue, be_err_le, be_err_ue, be_multi = (
            prepare_latency_error_series_data(
                berkeley_embedding_data_frames,
                berkeley_embedding_run_dirs_map,
                ERROR_RATE_UPPER_BOUND,
            )
        )
        __draw_confidence_series(
            be_lat,
            be_err,
            be_lat_le,
            be_lat_ue,
            be_err_le,
            be_err_ue,
            be_multi,
            "#3B686A",
            "Fine-tuned Embedding",
            8,
        )

    ############################################################
    ### vCache + Fine-tuned Embedding
    if vcache_berkeley_embedding_data_frames:
        vb_lat, vb_err, vb_lat_le, vb_lat_ue, vb_err_le, vb_err_ue, vb_multi = (
            prepare_latency_error_series_data(
                vcache_berkeley_embedding_data_frames,
                vcache_berkeley_embedding_run_dirs_map,
                ERROR_RATE_UPPER_BOUND,
            )
        )
        __draw_confidence_series(
            vb_lat,
            vb_err,
            vb_lat_le,
            vb_lat_ue,
            vb_err_le,
            vb_err_ue,
            vb_multi,
            "#EDBE24",
            "vCache + Fine-tuned Embedding",
            8,
        )

    ############################################################
    ### Baseline: No Cache Plotting
    # Ensure avg_latency_no_cache has been set by one of the prepare_latency_error_series_data calls
    if avg_latency_no_cache != -1:  # Check if it was updated
        plt.axvline(
            x=avg_latency_no_cache,
            color="grey",
            linewidth=3,
            label="No Cache",
            zorder=1,  # Ensure it's behind most other elements
        )
        # Arrow annotation removed as it was empty and complex to replicate without specific text

    plt.xlabel("Average Latency (sec)", fontsize=font_size)
    plt.ylabel("Error Rate (%)", fontsize=font_size)
    plt.ylim(bottom=0.0)
    plt.tick_params(axis="both", labelsize=font_size - 2)

    yticks = plt.yticks()[0]
    if yticks[0] == 0.0:
        plt.yticks(yticks[1:])

    plt.gca().spines["top"].set_linewidth(1)
    plt.gca().spines["right"].set_linewidth(1)
    plt.gca().spines["bottom"].set_linewidth(1)
    plt.gca().spines["left"].set_linewidth(1)

    filename = results_dir + "/avg_latency_vs_error_rate.pdf"
    plt.savefig(filename, format="pdf", bbox_inches="tight", transparent=True)
    plt.close()


def __plot_cache_hit_vs_error_rate(
    gptcache_data_frames: Dict[float, pd.DataFrame],
    vcache_local_data_frames: Dict[float, pd.DataFrame],
    vcache_global_data_frames: Dict[float, pd.DataFrame],
    berkeley_embedding_data_frames: Dict[float, pd.DataFrame],
    vcache_berkeley_embedding_data_frames: Dict[float, pd.DataFrame],
    results_dir: str,
    timestamp: str,
    font_size: int,
):
    plt.figure(figsize=(12, 10))
    ERROR_RATE_UPPER_BOUND = 6  # 6%

    # Collect run directories for each type
    gptcache_run_dirs_map = __collect_run_dirs_by_prefix_and_key(
        results_dir, "gptcache_"
    )
    vcache_local_run_dirs_map = __collect_run_dirs_by_prefix_and_key(
        results_dir, "vcache_local_"
    )
    vcache_global_run_dirs_map = __collect_run_dirs_by_prefix_and_key(
        results_dir, "vcache_global_"
    )
    berkeley_embedding_run_dirs_map = __collect_run_dirs_by_prefix_and_key(
        results_dir, "berkeley_embedding_"
    )
    vcache_berkeley_embedding_run_dirs_map = __collect_run_dirs_by_prefix_and_key(
        results_dir, "vcache_berkeley_embedding_"
    )

    # Helper to prepare data for a series
    def prepare_cache_hit_error_series_data(
        data_frames, run_dirs_map_for_series, error_rate_bound
    ):
        keys = sorted(data_frames.keys())
        err_means, ch_means = [], []
        err_low_err, err_up_err = [], []
        ch_low_err, ch_up_err = [], []
        is_multi = []

        for key_val in keys:
            key_str = f"{key_val:.3f}".rstrip("0").rstrip(".")
            if not key_str and key_val == 0:
                key_str = "0"

            current_run_dirs = run_dirs_map_for_series.get(key_str, [])

            if len(current_run_dirs) > 1:  # Multi-run
                stats = __aggregate_stats_for_cache_hit_error_rate(current_run_dirs)
                if stats is None:
                    continue  # Should not happen with new return type

                er_m = stats["error_rate_mean"] * 100
                ch_m = stats["cache_hit_rate_mean"] * 100

                if er_m <= error_rate_bound:
                    err_means.append(er_m)
                    ch_means.append(ch_m)
                    # errors are relative to mean
                    err_low_err.append(
                        (stats["error_rate_mean"] - stats["error_rate_ci_low"]) * 100
                    )
                    err_up_err.append(
                        (stats["error_rate_ci_up"] - stats["error_rate_mean"]) * 100
                    )
                    ch_low_err.append(
                        (stats["cache_hit_rate_mean"] - stats["cache_hit_rate_ci_low"])
                        * 100
                    )
                    ch_up_err.append(
                        (stats["cache_hit_rate_ci_up"] - stats["cache_hit_rate_mean"])
                        * 100
                    )
                    is_multi.append(True)
            elif data_frames:  # Single run
                df = data_frames[key_val]
                er_m_single = compute_error_rate_score(fp=df["fp_list"]) * 100
                ch_m_single = (
                    compute_cache_hit_rate_score(cache_hit_list=df["cache_hit_list"])
                    * 100
                )

                if er_m_single <= error_rate_bound:
                    err_means.append(er_m_single)
                    ch_means.append(ch_m_single)
                    err_low_err.append(0)
                    err_up_err.append(0)
                    ch_low_err.append(0)
                    ch_up_err.append(0)
                    is_multi.append(False)

        # Ensure errors are non-negative
        err_low_err = [max(0, err) for err in err_low_err]
        err_up_err = [max(0, err) for err in err_up_err]
        ch_low_err = [max(0, err) for err in ch_low_err]
        ch_up_err = [max(0, err) for err in ch_up_err]

        return (
            err_means,
            ch_means,
            err_low_err,
            err_up_err,
            ch_low_err,
            ch_up_err,
            is_multi,
        )

    # GPTCache
    if gptcache_data_frames:
        gpt_err, gpt_ch, gpt_err_le, gpt_err_ue, gpt_ch_le, gpt_ch_ue, gpt_multi = (
            prepare_cache_hit_error_series_data(
                gptcache_data_frames, gptcache_run_dirs_map, ERROR_RATE_UPPER_BOUND
            )
        )
        __draw_confidence_series(
            gpt_err,
            gpt_ch,
            gpt_err_le,
            gpt_err_ue,
            gpt_ch_le,
            gpt_ch_ue,
            gpt_multi,
            "#C23B48",
            "GPTCache",
            10,
        )  # x=error, y=cache_hit

    # vCache Local
    if vcache_local_data_frames:
        vl_err, vl_ch, vl_err_le, vl_err_ue, vl_ch_le, vl_ch_ue, vl_multi = (
            prepare_cache_hit_error_series_data(
                vcache_local_data_frames,
                vcache_local_run_dirs_map,
                ERROR_RATE_UPPER_BOUND,
            )
        )
        __draw_confidence_series(
            vl_err,
            vl_ch,
            vl_err_le,
            vl_err_ue,
            vl_ch_le,
            vl_ch_ue,
            vl_multi,
            "#37A9EC",
            "vCache",
            8,
        )

    # vCache Global
    if vcache_global_data_frames:
        vg_err, vg_ch, vg_err_le, vg_err_ue, vg_ch_le, vg_ch_ue, vg_multi = (
            prepare_cache_hit_error_series_data(
                vcache_global_data_frames,
                vcache_global_run_dirs_map,
                ERROR_RATE_UPPER_BOUND,
            )
        )
        __draw_confidence_series(
            vg_err,
            vg_ch,
            vg_err_le,
            vg_err_ue,
            vg_ch_le,
            vg_ch_ue,
            vg_multi,
            "#8CBE94",
            "vCache (Ablation)",
            8,
        )

    # Fine-tuned Embedding
    if berkeley_embedding_data_frames:
        be_err, be_ch, be_err_le, be_err_ue, be_ch_le, be_ch_ue, be_multi = (
            prepare_cache_hit_error_series_data(
                berkeley_embedding_data_frames,
                berkeley_embedding_run_dirs_map,
                ERROR_RATE_UPPER_BOUND,
            )
        )
        __draw_confidence_series(
            be_err,
            be_ch,
            be_err_le,
            be_err_ue,
            be_ch_le,
            be_ch_ue,
            be_multi,
            "#3B686A",
            "Fine-tuned Embedding",
            8,
        )

    # vCache + Fine-tuned Embedding
    if vcache_berkeley_embedding_data_frames:
        vb_err, vb_ch, vb_err_le, vb_err_ue, vb_ch_le, vb_ch_ue, vb_multi = (
            prepare_cache_hit_error_series_data(
                vcache_berkeley_embedding_data_frames,
                vcache_berkeley_embedding_run_dirs_map,
                ERROR_RATE_UPPER_BOUND,
            )
        )
        __draw_confidence_series(
            vb_err,
            vb_ch,
            vb_err_le,
            vb_err_ue,
            vb_ch_le,
            vb_ch_ue,
            vb_multi,
            "#EDBE24",
            "vCache + Fine-tuned Embedding",
            8,
        )

    plt.xlabel("Error Rate (%)", fontsize=font_size)
    plt.ylabel("Cache Hit Rate (%)", fontsize=font_size)
    plt.tick_params(axis="both", labelsize=font_size - 2)

    yticks = plt.yticks()[0]
    if yticks[0] == 0.0:
        plt.yticks(yticks[1:])

    for spine in ["top", "right", "bottom", "left"]:
        plt.gca().spines[spine].set_linewidth(1)

    filename = results_dir + "/cache_hit_vs_error_rate.pdf"
    plt.savefig(filename, format="pdf", bbox_inches="tight", transparent=True)
    plt.close()


def __plot_cache_hit_vs_error_rate_vs_sample_size(
    gptcache_data_frames: Dict[float, pd.DataFrame],
    vcache_local_data_frames: Dict[float, pd.DataFrame],
    vcache_global_data_frames: Dict[float, pd.DataFrame],
    berkeley_embedding_data_frames: Dict[float, pd.DataFrame],
    vcache_berkeley_embedding_data_frames: Dict[float, pd.DataFrame],
    results_dir: str,
    timestamp: str,
    font_size: int,
):
    target_deltas = [0.015, 0.03]
    target_error_rates = [2, 3.5]

    for target_delta, target_error_rate in zip(target_deltas, target_error_rates):
        ############################################################
        ### Baseline: vCache Local
        vcache_local_df = vcache_local_data_frames[target_delta]
        vcache_local_error_rates = [
            rate * 100
            for rate in compute_error_rate_cumulative_list(
                fp=vcache_local_df["fp_list"]
            )
        ]
        vcache_local_cache_hit_rates = [
            rate * 100
            for rate in compute_cache_hit_rate_cumulative_list(
                cache_hit_list=vcache_local_df["cache_hit_list"]
            )
        ]

        ############################################################
        ### Baseline: vCache Global
        if vcache_global_data_frames:
            vcache_global_df = vcache_global_data_frames[target_delta]
            vcache_global_error_rates = [
                rate * 100
                for rate in compute_error_rate_cumulative_list(
                    fp=vcache_global_df["fp_list"]
                )
            ]
            vcache_global_cache_hit_rates = [
                rate * 100
                for rate in compute_cache_hit_rate_cumulative_list(
                    cache_hit_list=vcache_global_df["cache_hit_list"]
                )
            ]

        ############################################################
        ### Baseline: GPTCache
        if gptcache_data_frames:
            gptcache_thresholds = sorted(gptcache_data_frames.keys())
            gptcache_closest_error_rate_diff = float("inf")
            gptcache_df = None

            for threshold in gptcache_thresholds:
                df = gptcache_data_frames[threshold]
                error_rate = compute_error_rate_score(fp=df["fp_list"]) * 100
                error_rate_diff = abs(error_rate - target_error_rate)

                if error_rate_diff < gptcache_closest_error_rate_diff:
                    gptcache_closest_error_rate_diff = error_rate_diff
                    gptcache_df = df

            gptcache_error_rates = [
                rate * 100
                for rate in compute_error_rate_cumulative_list(
                    fp=gptcache_df["fp_list"]
                )
            ]
            gptcache_cache_hit_rates = [
                rate * 100
                for rate in compute_cache_hit_rate_cumulative_list(
                    cache_hit_list=gptcache_df["cache_hit_list"]
                )
            ]

        ############################################################
        ### Baseline: Berkeley Embedding
        if berkeley_embedding_data_frames:
            berkeley_embedding_thresholds = sorted(
                berkeley_embedding_data_frames.keys()
            )
            berkeley_embedding_closest_error_rate_diff = float("inf")
            berkeley_embedding_df = None

            for threshold in berkeley_embedding_thresholds:
                df = berkeley_embedding_data_frames[threshold]
                error_rate = compute_error_rate_score(fp=df["fp_list"]) * 100
                error_rate_diff = abs(error_rate - target_error_rate)

                if error_rate_diff < berkeley_embedding_closest_error_rate_diff:
                    berkeley_embedding_closest_error_rate_diff = error_rate_diff
                    berkeley_embedding_df = df

            berkeley_embedding_error_rates = [
                rate * 100
                for rate in compute_error_rate_cumulative_list(
                    fp=berkeley_embedding_df["fp_list"]
                )
            ]
            berkeley_embedding_cache_hit_rates = [
                rate * 100
                for rate in compute_cache_hit_rate_cumulative_list(
                    cache_hit_list=berkeley_embedding_df["cache_hit_list"]
                )
            ]

        ############################################################
        ### vCache + Berkeley Embedding
        if vcache_berkeley_embedding_data_frames:
            vcache_berkeley_embedding_df = vcache_berkeley_embedding_data_frames[
                target_delta
            ]
            vcache_berkeley_embedding_error_rates = [
                rate * 100
                for rate in compute_error_rate_cumulative_list(
                    fp=vcache_berkeley_embedding_df["fp_list"]
                )
            ]
            vcache_berkeley_embedding_cache_hit_rates = [
                rate * 100
                for rate in compute_cache_hit_rate_cumulative_list(
                    cache_hit_list=vcache_berkeley_embedding_df["cache_hit_list"]
                )
            ]

        sample_sizes = np.arange(1, len(vcache_local_error_rates) + 1)

        # Plot 1: Error rates vs sample size
        plt.figure(figsize=(12, 11))
        plt.plot(
            sample_sizes[::45],
            vcache_local_error_rates[::45],
            "-",
            color="#37A9EC",
            linewidth=4,
            label="vCache",
        )

        if vcache_global_data_frames:
            plt.plot(
                sample_sizes[::45],
                vcache_global_error_rates[::45],
                "-",
                color="#8CBE94",
                linewidth=4,
                label="vCache (Ablation)",
            )

        if gptcache_data_frames:
            plt.plot(
                sample_sizes[::45],
                gptcache_error_rates[::45],
                "-",
                color="#C23B48",
                linewidth=4,
                label="GPTCache",
            )

        if berkeley_embedding_data_frames:
            plt.plot(
                sample_sizes[::45],
                berkeley_embedding_error_rates[::45],
                "-",
                color="#3B686A",
                linewidth=4,
                label="Fine-tuned Embedding",
            )

        if vcache_berkeley_embedding_data_frames:
            plt.plot(
                sample_sizes[::45],
                vcache_berkeley_embedding_error_rates[::45],
                "-",
                color="#EDBE24",
                linewidth=4,
                label="vCache + Fine-tuned Embedding",
            )

        plt.xlabel("Sample Size", fontsize=font_size)
        plt.ylabel("Error Rate (%)", fontsize=font_size)

        plt.tick_params(axis="both", labelsize=font_size - 2)

        error_rate_filename = (
            results_dir + f"/error_rate_vs_sample_size_delta_{target_delta:.3f}.pdf"
        )
        plt.savefig(
            error_rate_filename, format="pdf", bbox_inches="tight", transparent=True
        )
        plt.close()

        # Plot 2: Cache hit rates vs sample size
        plt.figure(figsize=(12, 11))
        plt.plot(
            sample_sizes[::5],
            vcache_local_cache_hit_rates[::5],
            "-",
            color="#37A9EC",
            linewidth=4,
            label="vCache Local",
        )

        if vcache_global_data_frames:
            plt.plot(
                sample_sizes[::5],
                vcache_global_cache_hit_rates[::5],
                "-",
                color="#8CBE94",
                linewidth=4,
                label="vCache Global",
            )

        if gptcache_data_frames:
            plt.plot(
                sample_sizes[::5],
                gptcache_cache_hit_rates[::5],
                "-",
                color="#C23B48",
                linewidth=4,
                label="GPTCache",
            )

        if berkeley_embedding_data_frames:
            plt.plot(
                sample_sizes[::5],
                berkeley_embedding_cache_hit_rates[::5],
                "-",
                color="#3B686A",
                linewidth=4,
                label="Fine-tuned Embedding",
            )

        if vcache_berkeley_embedding_data_frames:
            plt.plot(
                sample_sizes[::5],
                vcache_berkeley_embedding_cache_hit_rates[::5],
                "-",
                color="#EDBE24",
                linewidth=4,
                label="vCache + Fine-tuned Embedding",
            )

        plt.xlabel("Sample Size", fontsize=font_size)
        plt.ylabel("Cache Hit Rate (%)", fontsize=font_size)
        plt.tick_params(axis="both", labelsize=font_size - 2)

        cache_hit_filename = (
            results_dir + f"/cache_hit_rate_vs_sample_size_delta_{target_delta:.3f}.pdf"
        )
        plt.savefig(
            cache_hit_filename, format="pdf", bbox_inches="tight", transparent=True
        )
        plt.close()


def __plot_delta_accuracy(
    vcache_local_data_frames: Dict[float, pd.DataFrame],
    vcache_global_data_frames: Dict[float, pd.DataFrame],
    vcache_berkeley_embedding_data_frames: Dict[float, pd.DataFrame],
    results_dir: str,
    timestamp: str,
    font_size: int,
):
    plt.figure(figsize=(12, 8))

    vcache_local_run_dirs_map = __collect_run_dirs_by_prefix_and_key(
        results_dir, "vcache_local_"
    )

    vcache_local_deltas_sorted = sorted(vcache_local_data_frames.keys())

    if vcache_local_deltas_sorted:
        error_rate_means = []
        error_rate_cis_lower_err = []  # error from mean to lower bound
        error_rate_cis_upper_err = []  # error from mean to upper bound
        delta_labels = []
        is_multi_run_list = []

        for delta_val in vcache_local_deltas_sorted:
            delta_key_str = f"{delta_val:.3f}".rstrip("0").rstrip(".")
            if not delta_key_str and delta_val == 0:
                delta_key_str = "0"

            current_run_dirs = vcache_local_run_dirs_map.get(delta_key_str, [])

            if len(current_run_dirs) > 1:  # Multi-run
                # We only need error rate stats from this function
                er_mean, er_ci_low, er_ci_up, _, _, _ = (
                    __aggregate_stats_for_latency_error(current_run_dirs)
                )
                error_rate_means.append(er_mean)
                error_rate_cis_lower_err.append(max(0, er_mean - er_ci_low))
                error_rate_cis_upper_err.append(max(0, er_ci_up - er_mean))
                is_multi_run_list.append(True)
            else:  # Single run
                df = vcache_local_data_frames[delta_val]
                er_mean_single = compute_error_rate_score(fp=df["fp_list"])
                error_rate_means.append(er_mean_single)
                error_rate_cis_lower_err.append(0)
                error_rate_cis_upper_err.append(0)
                is_multi_run_list.append(False)

            delta_labels.append(f".{int(delta_val * 1000):03d}")

        x_pos = np.arange(len(vcache_local_deltas_sorted))
        bar_width = 0.8

        # Prepare yerr for plt.bar. It needs to be a 2xN array for asymmetric error bars.
        actual_errors_for_bar = [
            list(z) for z in zip(error_rate_cis_lower_err, error_rate_cis_upper_err)
        ]
        # Only include error bars where is_multi_run is True, otherwise pass 0 for that bar
        y_errors_for_plot = []
        for i in range(len(is_multi_run_list)):
            if is_multi_run_list[i]:
                y_errors_for_plot.append(actual_errors_for_bar[i])
            else:
                y_errors_for_plot.append([0, 0])  # No error bar for single runs

        y_errors_for_plot_transposed = np.array(y_errors_for_plot).T

        plt.bar(
            x_pos,
            error_rate_means,
            bar_width,
            color="#37A9EC",
            label="Actual Error Rate",
            yerr=y_errors_for_plot_transposed,
            capsize=7.5,  # Add cap to error bars
        )

        for i, delta_target_val in enumerate(vcache_local_deltas_sorted):
            plt.hlines(
                y=delta_target_val,
                xmin=i - bar_width / 2,
                xmax=i + bar_width / 2,
                colors="#EDBE24",
                linestyles="dashed",
                linewidth=3,
            )

        custom_lines = [
            Line2D([0], [0], color="#EDBE24", linestyle="dashed", lw=4),
            Line2D([0], [0], color="#37A9EC", lw=4),
        ]
        plt.legend(
            custom_lines,
            ["$\delta$ Target", "Actual Error"],
            fontsize=font_size - 2,  # Adjusted for potentially more space needed
            handlelength=1.5,  # Adjusted handle length
            loc="upper left",  # Ensure legend doesn't overlap much
        )

        plt.xlabel("$\delta$ Values", fontsize=font_size)
        plt.xticks(rotation=45, ha="right")  # Rotate for better readability
        plt.ylabel("Error Rate", fontsize=font_size)
        plt.xticks(x_pos, delta_labels, fontsize=font_size)
        plt.yticks(fontsize=font_size - 2)

        yticks = plt.yticks()[0]
        # Ensure y-axis starts at 0, if 0 is not already the first tick.
        # And if y_ticks has values, check the first one.
        if len(yticks) > 0 and yticks[0] != 0.0:
            if 0.0 not in yticks:
                # Create new ticks that include 0 and maintain reasonable spacing
                new_yticks = np.linspace(0, yticks[-1], len(yticks))
                plt.yticks(new_yticks)
        elif not len(yticks):
            plt.yticks([0, 0.1, 0.2])  # Default if no ticks

        def format_tick(x, pos):
            if x < 1e-9 and x > -1e-9:  # handles almost zero
                return "0"
            s = f"{x:.3f}"  # Use .3f to get some precision
            parts = s.split(".")
            if len(parts) == 1:  # Integer or number like "0"
                return parts[0]
            integer_part, decimal_part = parts
            decimal_part = decimal_part.rstrip("0")
            if not decimal_part:  # was like .000
                return integer_part
            if integer_part == "0":  # like 0.123 -> .123
                return f".{decimal_part}"
            return f"{integer_part}.{decimal_part}"  # like 1.123

        formatter = plt.FuncFormatter(format_tick)
        plt.gca().yaxis.set_major_formatter(formatter)

        plt.gca().spines["top"].set_linewidth(1)
        plt.gca().spines["right"].set_linewidth(1)
        plt.gca().spines["bottom"].set_linewidth(1)
        plt.gca().spines["left"].set_linewidth(1)

        all_plot_values = (
            error_rate_means
            + [
                er + ci_u
                for er, ci_u in zip(error_rate_means, error_rate_cis_upper_err)
            ]
            + vcache_local_deltas_sorted
        )
        if all_plot_values:
            y_min = 0
            y_max = max(all_plot_values) * 1.15
            if y_max < 0.08:
                y_max = 0.08  # Ensure a minimum sensible y_max
            plt.ylim(y_min, y_max)
        else:
            plt.ylim(0, 0.08)  # Default if no values

    filename = results_dir + "/delta_accuracy.pdf"
    plt.savefig(filename, format="pdf", bbox_inches="tight", transparent=True)
    plt.close()
