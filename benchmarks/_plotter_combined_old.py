import json
import os

import matplotlib.pyplot as plt
import numpy as np
from adjustText import adjust_text
from scipy import stats
from sklearn.metrics import auc, roc_curve


def _get_result_files(
    dataset, embedding_model_name, llm_model_name, timestamp, results_dir
):
    """
    Get all result files for a given dataset, embedding model, and LLM model.
    If timestamp is provided, only get files for that timestamp, otherwise get all files.
    """
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


def plot_hit_rate_vs_error(
    dataset, embedding_model_name, llm_model_name, timestamp, results_dir, FONT_SIZE=20
):
    static_files, dynamic_files = _get_result_files(
        dataset, embedding_model_name, llm_model_name, timestamp, results_dir
    )

    if not static_files and not dynamic_files:
        print(
            f"No results found for {dataset}, {embedding_model_name}, {llm_model_name}"
        )
        return

    static_hit_rates = []
    static_error_rates = []
    static_thresholds = []

    dynamic_hit_rates = []
    dynamic_error_rates = []
    dynamic_rnd_ubs = []

    # Dictionary to group vectorq results by delta
    vectorq_results = {}  # {delta: {'hit_rates': [], 'error_rates': []}}

    for file_path in static_files:
        with open(file_path, "r") as f:
            data = json.load(f)

            total_samples = data["sample_sizes"][-1] + 1  # Add 1 because it's 0-indexed
            total_reused = data["total_reused_list"][-1]
            hit_rate = (total_reused / total_samples) * 100 if total_samples > 0 else 0
            error_rate = data["error_rates_absolute"][-1]
            threshold = data["config"]["threshold"]

            static_hit_rates.append(hit_rate)
            static_error_rates.append(error_rate)
            static_thresholds.append(threshold)

    # Find all vectorq files, and get the data by delta
    for file_path in dynamic_files:
        with open(file_path, "r") as f:
            data = json.load(f)

            total_samples = data["sample_sizes"][-1] + 1
            total_reused = data["total_reused_list"][-1]
            hit_rate = (total_reused / total_samples) * 100 if total_samples > 0 else 0
            error_rate = data["error_rates_absolute"][-1]
            delta = data["config"]["delta"]

            # Check if this is a vectorq_X directory
            if "vectorq_" in file_path:
                if delta not in vectorq_results:
                    vectorq_results[delta] = {"hit_rates": [], "error_rates": []}

                vectorq_results[delta]["hit_rates"].append(hit_rate)
                vectorq_results[delta]["error_rates"].append(error_rate)
            else:
                raise ValueError(f"Unknown file format: {file_path}")

    # For new runs with multiple files per delta, compute average and confidence intervals
    if vectorq_results:
        # For each delta, compute average and append to dynamic lists
        for delta, values in sorted(
            vectorq_results.items(), key=lambda x: float(x[0])
        ):
            hit_rates = values["hit_rates"]
            error_rates = values["error_rates"]

            # If we have multiple runs, compute mean and add to main lists
            if hit_rates:
                avg_hit_rate = np.mean(hit_rates)
                avg_error_rate = np.mean(error_rates)

                dynamic_hit_rates.append(avg_hit_rate)
                dynamic_error_rates.append(avg_error_rate)
                dynamic_rnd_ubs.append(delta)

    plt.rcParams.update({"font.size": FONT_SIZE})
    plt.figure(figsize=(10, 6.5))

    texts = []

    # Plot static threshold points
    if static_hit_rates:
        static_points = sorted(
            zip(static_hit_rates, static_error_rates, static_thresholds),
            key=lambda x: x[0],
        )
        static_hit_rates_sorted = [point[0] for point in static_points]
        static_error_rates_sorted = [point[1] for point in static_points]
        static_thresholds_sorted = [point[2] for point in static_points]

        plt.plot(
            static_hit_rates_sorted,
            static_error_rates_sorted,
            "b-",
            alpha=0.5,
            linewidth=3,
        )
        plt.scatter(
            static_hit_rates_sorted,
            static_error_rates_sorted,
            c="blue",
            marker="o",
            s=140,
            label="State of the Art",
        )

        # Add annotations for static thresholds
        for i, threshold in enumerate(static_thresholds_sorted):
            if i == 0 or i == len(static_thresholds_sorted) - 1:
                t = plt.text(
                    static_hit_rates_sorted[i],
                    static_error_rates_sorted[i],
                    f"{threshold}",
                    fontsize=20,
                    bbox=dict(
                        boxstyle="round,pad=0.3", fc="aliceblue", ec="blue", alpha=0.8
                    ),
                )
                texts.append(t)

    # Plot dynamic threshold points
    if dynamic_hit_rates:
        dynamic_points = sorted(
            zip(dynamic_hit_rates, dynamic_error_rates, dynamic_rnd_ubs),
            key=lambda x: x[0],
        )
        dynamic_hit_rates_sorted = [point[0] for point in dynamic_points]
        dynamic_error_rates_sorted = [point[1] for point in dynamic_points]
        dynamic_rnd_ubs_sorted = [point[2] for point in dynamic_points]

        plt.plot(
            dynamic_hit_rates_sorted,
            dynamic_error_rates_sorted,
            "r-",
            alpha=0.7,
            linewidth=3,
        )
        plt.scatter(
            dynamic_hit_rates_sorted,
            dynamic_error_rates_sorted,
            c="red",
            marker="^",
            s=140,
            label="VectorQ",
        )

        # Add confidence intervals if we have multiple runs
        if vectorq_results:
            # Prepare data for error bars
            xerr = []
            yerr = []

            for i, rnd_ub in enumerate(dynamic_rnd_ubs_sorted):
                if (
                    rnd_ub in vectorq_results
                    and len(vectorq_results[rnd_ub]["hit_rates"]) > 1
                ):
                    # Calculate 95% confidence interval
                    hit_rates = vectorq_results[rnd_ub]["hit_rates"]
                    error_rates = vectorq_results[rnd_ub]["error_rates"]

                    # Standard error of the mean
                    hit_rate_std = np.std(hit_rates, ddof=1)
                    error_rate_std = np.std(error_rates, ddof=1)

                    # 95% confidence interval factor (for small samples)
                    t_factor = stats.t.ppf(0.975, len(hit_rates) - 1)  # 95% CI

                    # Confidence interval
                    hit_rate_ci = t_factor * (hit_rate_std / np.sqrt(len(hit_rates)))
                    error_rate_ci = t_factor * (
                        error_rate_std / np.sqrt(len(error_rates))
                    )

                    xerr.append(hit_rate_ci)
                    yerr.append(error_rate_ci)
                else:
                    xerr.append(0)
                    yerr.append(0)

            plt.errorbar(
                dynamic_hit_rates_sorted,
                dynamic_error_rates_sorted,
                xerr=xerr,
                yerr=yerr,
                fmt="none",
                ecolor="orange",
                alpha=0.75,
                capsize=5,
                label="95% Confidence Interval",
            )

        # Add annotations for dynamic thresholds
        for i, rnd_ub in enumerate(dynamic_rnd_ubs_sorted):
            if i == 0 or i == len(dynamic_rnd_ubs_sorted) - 1:
                t = plt.text(
                    dynamic_hit_rates_sorted[i],
                    dynamic_error_rates_sorted[i],
                    f"{rnd_ub}",
                    fontsize=20,
                    bbox=dict(
                        boxstyle="round,pad=0.3", fc="mistyrose", ec="red", alpha=0.8
                    ),
                )
                texts.append(t)

    if texts:
        adjust_text(
            texts,
            expand_points=(3.5, 3.5),
            force_points=(3.5, 3.5),
        )

    plt.xlabel("Cache Hit Rate (%)", fontsize=FONT_SIZE)
    plt.ylabel("Error Rate (%)", fontsize=FONT_SIZE)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.xticks(fontsize=FONT_SIZE)
    plt.yticks(fontsize=FONT_SIZE)
    plt.ylim(bottom=0)
    plt.subplots_adjust(top=0.85)
    plt.legend(
        fontsize=FONT_SIZE,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.167),
        ncol=3,
        frameon=True,
        fancybox=True,
        columnspacing=1.0,
        handletextpad=0.4,
    )

    output_dir = (
        f"processed_{results_dir}{dataset}/{embedding_model_name}/{llm_model_name}/"
    )
    os.makedirs(output_dir, exist_ok=True)
    output_file = f"{output_dir}comparison_hit_rate_vs_error.pdf"
    plt.savefig(output_file, format="pdf", bbox_inches="tight")
    plt.close()
    print(f"     Hit rate vs error plot saved to {output_file}")


def plot_precision_vs_recall(
    dataset, embedding_model_name, llm_model_name, timestamp, results_dir, FONT_SIZE=20
):
    static_files, dynamic_files = _get_result_files(
        dataset, embedding_model_name, llm_model_name, timestamp, results_dir
    )

    if not static_files and not dynamic_files:
        print(
            f"No results found for {dataset}, {embedding_model_name}, {llm_model_name}"
        )
        return

    static_precisions = []
    static_recalls = []
    static_thresholds = []

    dynamic_precisions = []
    dynamic_recalls = []
    dynamic_rnd_ubs = []

    # Dictionary to group vectorq results by delta
    vectorq_results = {}  # {delta: {'precisions': [], 'recalls': []}}

    for file_path in static_files:
        with open(file_path, "r") as f:
            data = json.load(f)

            precision = data["precision"]
            recall = data["recall"]
            threshold = data["config"]["threshold"]

            static_precisions.append(precision)
            static_recalls.append(recall)
            static_thresholds.append(threshold)

    # Find all vectorq files, and get the data by delta
    for file_path in dynamic_files:
        with open(file_path, "r") as f:
            data = json.load(f)

            precision = data["precision"]
            recall = data["recall"]
            delta = data["config"]["delta"]

            # Check if this is a vectorq_X directory
            if "vectorq_" in file_path:
                if delta not in vectorq_results:
                    vectorq_results[delta] = {"precisions": [], "recalls": []}

                vectorq_results[delta]["precisions"].append(precision)
                vectorq_results[delta]["recalls"].append(recall)
            else:
                # Legacy support for old files
                dynamic_precisions.append(precision)
                dynamic_recalls.append(recall)
                dynamic_rnd_ubs.append(delta)

    # For new runs with multiple files per delta, compute average and confidence intervals
    if vectorq_results:
        # For each delta, compute average and append to dynamic lists
        for delta, values in sorted(
            vectorq_results.items(), key=lambda x: float(x[0])
        ):
            precisions = values["precisions"]
            recalls = values["recalls"]

            # If we have multiple runs, compute mean and add to main lists
            if precisions:
                avg_precision = np.mean(precisions)
                avg_recall = np.mean(recalls)

                dynamic_precisions.append(avg_precision)
                dynamic_recalls.append(avg_recall)
                dynamic_rnd_ubs.append(delta)

    plt.rcParams.update({"font.size": FONT_SIZE})
    plt.figure(figsize=(10, 6.5))

    texts = []

    # Plot static threshold points
    if static_recalls:
        static_points = sorted(
            zip(static_recalls, static_precisions, static_thresholds),
            key=lambda x: x[0],
        )
        static_recalls_sorted = [point[0] for point in static_points]
        static_precisions_sorted = [point[1] for point in static_points]
        static_thresholds_sorted = [point[2] for point in static_points]

        plt.plot(
            static_recalls_sorted,
            static_precisions_sorted,
            "b-",
            alpha=0.5,
            linewidth=3,
        )
        plt.scatter(
            static_recalls_sorted,
            static_precisions_sorted,
            c="blue",
            marker="o",
            s=140,
            label="State of the Art",
        )

        # Add annotations for static thresholds
        for i, threshold in enumerate(static_thresholds_sorted):
            if i == 0 or i == len(static_thresholds_sorted) - 1:
                # Add a y-offset to place labels below points instead of directly on them
                y_offset = -0.55  # Negative offset to move downward
                t = plt.text(
                    static_recalls_sorted[i],
                    static_precisions_sorted[i]
                    + y_offset,  # Add offset to y-coordinate
                    f"{threshold}",
                    fontsize=20,
                    bbox=dict(
                        boxstyle="round,pad=0.3", fc="aliceblue", ec="blue", alpha=0.8
                    ),
                )
                texts.append(t)

                # Add explicit annotation arrows directly from dot to label
                plt.annotate(
                    "",
                    xy=(static_recalls_sorted[i], static_precisions_sorted[i]),
                    xytext=(
                        static_recalls_sorted[i],
                        static_precisions_sorted[i] + y_offset,
                    ),
                    arrowprops=dict(arrowstyle="->", color="blue", lw=1.2, alpha=0.4),
                    zorder=0,
                )

    # Plot dynamic threshold points
    if dynamic_recalls:
        dynamic_points = sorted(
            zip(dynamic_recalls, dynamic_precisions, dynamic_rnd_ubs),
            key=lambda x: x[0],
        )
        dynamic_recalls_sorted = [point[0] for point in dynamic_points]
        dynamic_precisions_sorted = [point[1] for point in dynamic_points]
        dynamic_rnd_ubs_sorted = [point[2] for point in dynamic_points]

        plt.plot(
            dynamic_recalls_sorted,
            dynamic_precisions_sorted,
            "r-",
            alpha=0.7,
            linewidth=3,
        )
        plt.scatter(
            dynamic_recalls_sorted,
            dynamic_precisions_sorted,
            c="red",
            marker="^",
            s=140,
            label="VectorQ",
        )

        # Add confidence intervals if we have multiple runs
        if vectorq_results:
            # Prepare data for error bars
            xerr = []
            yerr = []

            for i, rnd_ub in enumerate(dynamic_rnd_ubs_sorted):
                if (
                    rnd_ub in vectorq_results
                    and len(vectorq_results[rnd_ub]["recalls"]) > 1
                ):
                    # Calculate 95% confidence interval
                    recalls = vectorq_results[rnd_ub]["recalls"]
                    precisions = vectorq_results[rnd_ub]["precisions"]

                    # Standard error of the mean
                    recall_std = np.std(recalls, ddof=1)
                    precision_std = np.std(precisions, ddof=1)

                    # 95% confidence interval factor (for small samples)
                    t_factor = stats.t.ppf(0.975, len(recalls) - 1)  # 95% CI

                    # Confidence interval
                    recall_ci = t_factor * (recall_std / np.sqrt(len(recalls)))
                    precision_ci = t_factor * (precision_std / np.sqrt(len(precisions)))

                    xerr.append(recall_ci)
                    yerr.append(precision_ci)
                else:
                    xerr.append(0)
                    yerr.append(0)

            plt.errorbar(
                dynamic_recalls_sorted,
                dynamic_precisions_sorted,
                xerr=xerr,
                yerr=yerr,
                fmt="none",
                ecolor="orange",
                alpha=0.75,
                capsize=5,
                label="95% Confidence Interval",
            )

        # Add annotations for dynamic thresholds
        for i, rnd_ub in enumerate(dynamic_rnd_ubs_sorted):
            if i == 0 or i == len(dynamic_rnd_ubs_sorted) - 1:
                y_offset = -0.55
                t = plt.text(
                    dynamic_recalls_sorted[i],
                    dynamic_precisions_sorted[i] + y_offset,
                    f"{rnd_ub}",
                    fontsize=20,
                    bbox=dict(
                        boxstyle="round,pad=0.3", fc="mistyrose", ec="red", alpha=0.8
                    ),
                )
                texts.append(t)

                # Add explicit annotation arrows directly from dot to label
                plt.annotate(
                    "",  # Empty text
                    xy=(
                        dynamic_recalls_sorted[i],
                        dynamic_precisions_sorted[i],
                    ),  # Start at the data point
                    xytext=(
                        dynamic_recalls_sorted[i],
                        dynamic_precisions_sorted[i] + y_offset,
                    ),  # End at the label location
                    arrowprops=dict(
                        arrowstyle="->", color="red", lw=1.2, alpha=0.7
                    ),  # Arrow style
                    zorder=0,  # Put arrows behind other elements
                )

    plt.xlabel("Recall", fontsize=FONT_SIZE)
    plt.ylabel("Precision", fontsize=FONT_SIZE)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.xlim(0, 1.05)
    plt.ylim(0, 1.05)
    plt.xticks(fontsize=FONT_SIZE)
    plt.yticks(fontsize=FONT_SIZE)
    plt.subplots_adjust(top=0.85)
    plt.legend(
        fontsize=FONT_SIZE,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.167),
        ncol=3,
        frameon=True,
        fancybox=True,
        columnspacing=1.0,
        handletextpad=0.4,
    )

    output_dir = (
        f"processed_{results_dir}{dataset}/{embedding_model_name}/{llm_model_name}/"
    )
    os.makedirs(output_dir, exist_ok=True)
    output_file = f"{output_dir}comparison_precision_vs_recall.pdf"
    plt.savefig(output_file, format="pdf", bbox_inches="tight")
    plt.close()
    print(f"     Precision vs recall plot saved to {output_file}")


def plot_duration_comparison(
    dataset, embedding_model_name, llm_model_name, timestamp, results_dir, FONT_SIZE=20
):
    static_files, dynamic_files = _get_result_files(
        dataset, embedding_model_name, llm_model_name, timestamp, results_dir
    )

    if not static_files and not dynamic_files:
        print(
            f"No results found for {dataset}, {embedding_model_name}, {llm_model_name}"
        )
        return

    static_durations = []
    static_thresholds = []

    # Dictionary to group vectorq results by delta
    vectorq_results = {}  # {delta: {'durations': []}}

    direct_duration = None

    for file_path in static_files:
        with open(file_path, "r") as f:
            data = json.load(f)

            # Get the threshold value
            threshold = data["config"]["threshold"]
            static_thresholds.append(threshold)

            # Get total duration for this run (convert to minutes)
            total_duration = data["total_duration_vectorq_list"][-1] / 60
            static_durations.append(total_duration)

            # Get direct duration (without cache) if not set yet
            if direct_duration is None:
                direct_duration = data["total_duration_direct_list"][-1] / 60

    # Process dynamic threshold files and group by delta
    for file_path in dynamic_files:
        with open(file_path, "r") as f:
            data = json.load(f)

            delta = data["config"]["delta"]
            total_duration = data["total_duration_vectorq_list"][-1] / 60

            if delta not in vectorq_results:
                vectorq_results[delta] = {"durations": []}

            vectorq_results[delta]["durations"].append(total_duration)

            if direct_duration is None:
                direct_duration = data["total_duration_direct_list"][-1] / 60

    # Calculate average durations for each delta
    dynamic_durations = []
    dynamic_rnd_ubs = []

    for delta, values in sorted(
        vectorq_results.items(), key=lambda x: float(x[0])
    ):
        if values["durations"]:
            avg_duration = np.mean(values["durations"])
            dynamic_durations.append(avg_duration)
            dynamic_rnd_ubs.append(delta)

    plt.rcParams.update({"font.size": FONT_SIZE})
    plt.figure(figsize=(16, 8))

    width = 0.6
    x_pos = 0
    bar_positions = []
    all_durations = []
    all_labels = []

    bar_positions.append(x_pos)
    all_durations.append(direct_duration)
    all_labels.append("No Cache")
    x_pos += width * 2

    sorted_static = sorted(
        zip(static_thresholds, static_durations),
        key=lambda x: float(x[0]),
        reverse=True,
    )
    static_thresholds = [t for t, _ in sorted_static]
    static_durations = [d for _, d in sorted_static]

    # Add static threshold bars
    for i, duration in enumerate(static_durations):
        bar_positions.append(x_pos)
        all_durations.append(duration)
        all_labels.append(f"{static_thresholds[i]}")
        x_pos += width

    x_pos += width

    for i, duration in enumerate(dynamic_durations):
        bar_positions.append(x_pos)
        all_durations.append(duration)
        all_labels.append(f"{dynamic_rnd_ubs[i]}")
        x_pos += width

    plt.bar(
        [bar_positions[0]], [all_durations[0]], width, color="gray", label="No Cache"
    )

    static_indices = range(1, len(static_durations) + 1)
    plt.bar(
        [bar_positions[i] for i in static_indices],
        [all_durations[i] for i in static_indices],
        width,
        color="blue",
        alpha=0.7,
        label="State of the Art",
    )

    dynamic_indices = range(len(static_durations) + 1, len(all_durations))
    plt.bar(
        [bar_positions[i] for i in dynamic_indices],
        [all_durations[i] for i in dynamic_indices],
        width,
        color="red",
        alpha=0.7,
        label="VectorQ",
    )

    plt.xlabel("Threshold / delta", fontsize=FONT_SIZE)
    plt.ylabel("Total Duration (minutes)", fontsize=FONT_SIZE)
    plt.xticks(bar_positions, all_labels, rotation=90, ha="right", fontsize=FONT_SIZE)
    plt.yticks(fontsize=FONT_SIZE)
    plt.ylim(bottom=0)
    plt.legend(fontsize=FONT_SIZE)
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()

    output_dir = (
        f"processed_{results_dir}{dataset}/{embedding_model_name}/{llm_model_name}/"
    )
    os.makedirs(output_dir, exist_ok=True)
    output_file = f"{output_dir}comparison_duration.pdf"
    plt.savefig(output_file, format="pdf", bbox_inches="tight")
    plt.close()
    print(f"     Duration comparison plot saved to {output_file}")


def plot_duration_vs_error_rate(
    dataset, embedding_model_name, llm_model_name, timestamp, results_dir, FONT_SIZE=20
):
    static_files, dynamic_files = _get_result_files(
        dataset, embedding_model_name, llm_model_name, timestamp, results_dir
    )

    if not static_files and not dynamic_files:
        print(
            f"No results found for {dataset}, {embedding_model_name}, {llm_model_name}"
        )
        return

    static_durations = []
    static_error_rates = []
    static_thresholds = []

    dynamic_durations = []
    dynamic_error_rates = []
    dynamic_rnd_ubs = []

    # Dictionary to group vectorq results by delta
    vectorq_results = {}  # {delta: {'durations': [], 'error_rates': []}}

    for file_path in static_files:
        with open(file_path, "r") as f:
            data = json.load(f)

            total_duration = data["total_duration_vectorq_list"][-1] / 60
            total_reused = data["total_reused_list"][-1]

            if total_reused > 0:
                error_rate = data["error_rates_absolute"][-1]
            else:
                error_rate = 0

            threshold = data["config"]["threshold"]

            static_durations.append(total_duration)
            static_error_rates.append(error_rate)
            static_thresholds.append(threshold)

    # Process dynamic threshold files
    for file_path in dynamic_files:
        with open(file_path, "r") as f:
            data = json.load(f)

            total_duration = data["total_duration_vectorq_list"][-1] / 60
            total_reused = data["total_reused_list"][-1]

            # Get error rate relative to reused answers
            if total_reused > 0:
                error_rate = data["error_rates_absolute"][-1]
            else:
                error_rate = 0

            delta = data["config"]["delta"]

            num_samples = data["sample_sizes"][-1] + 1

            # Group results by delta for confidence intervals
            if "vectorq_" in file_path:
                if delta not in vectorq_results:
                    vectorq_results[delta] = {
                        "durations": [],
                        "error_rates": [],
                        "samples": [],
                    }

                vectorq_results[delta]["durations"].append(total_duration)
                vectorq_results[delta]["error_rates"].append(error_rate)
                vectorq_results[delta]["samples"].append(num_samples)

    # For new runs with multiple files per delta, compute average
    if vectorq_results:
        for delta, values in sorted(
            vectorq_results.items(), key=lambda x: float(x[0])
        ):
            durations = values["durations"]
            error_rates = values["error_rates"]

            if durations:
                avg_duration = np.mean(durations)
                avg_error_rate = np.mean(error_rates)

                dynamic_durations.append(avg_duration)
                dynamic_error_rates.append(avg_error_rate)
                dynamic_rnd_ubs.append(delta)

    plt.rcParams.update({"font.size": FONT_SIZE})
    plt.figure(figsize=(10, 6.5))

    texts = []
    direct_duration = None

    # Look for direct inference duration in the first available file
    if static_files:
        with open(static_files[0], "r") as f:
            data = json.load(f)
            if (
                "total_duration_direct_list" in data
                and data["total_duration_direct_list"]
            ):
                # Convert seconds to minutes
                direct_duration = data["total_duration_direct_list"][-1] / 60
    elif dynamic_files:
        with open(dynamic_files[0], "r") as f:
            data = json.load(f)
            if (
                "total_duration_direct_list" in data
                and data["total_duration_direct_list"]
            ):
                # Convert seconds to minutes
                direct_duration = data["total_duration_direct_list"][-1] / 60

    if static_durations:
        static_points = sorted(
            zip(static_durations, static_error_rates, static_thresholds),
            key=lambda x: x[0],
        )
        static_durations_sorted = [point[0] for point in static_points]
        static_error_rates_sorted = [point[1] for point in static_points]
        static_thresholds_sorted = [point[2] for point in static_points]

        plt.plot(
            static_durations_sorted,
            static_error_rates_sorted,
            "b-",
            alpha=0.5,
            linewidth=3,
        )
        plt.scatter(
            static_durations_sorted,
            static_error_rates_sorted,
            c="blue",
            marker="o",
            s=140,
            label="State of the Art",
        )

        # Add annotations for static thresholds
        for i, threshold in enumerate(static_thresholds_sorted):
            if i == 0 or i == len(static_thresholds_sorted) - 1:
                t = plt.text(
                    static_durations_sorted[i],
                    static_error_rates_sorted[i],
                    f"{threshold}",
                    fontsize=20,
                    bbox=dict(
                        boxstyle="round,pad=0.3", fc="aliceblue", ec="blue", alpha=0.8
                    ),
                )
                texts.append(t)

    if dynamic_durations:
        dynamic_points = sorted(
            zip(dynamic_durations, dynamic_error_rates, dynamic_rnd_ubs),
            key=lambda x: x[0],
        )
        dynamic_durations_sorted = [point[0] for point in dynamic_points]
        dynamic_error_rates_sorted = [point[1] for point in dynamic_points]
        dynamic_rnd_ubs_sorted = [point[2] for point in dynamic_points]

        plt.plot(
            dynamic_durations_sorted,
            dynamic_error_rates_sorted,
            "r-",
            alpha=0.7,
            linewidth=3,
        )

        sizes = []

        for rnd_ub in dynamic_rnd_ubs_sorted:
            if rnd_ub in vectorq_results:
                # Use average sample size as the marker size
                avg_samples = np.mean(vectorq_results[rnd_ub]["samples"])
                size = max(80, min(200, avg_samples / 20))
                sizes.append(size)
            else:
                sizes.append(80)

        plt.scatter(
            dynamic_durations_sorted,
            dynamic_error_rates_sorted,
            c="red",
            marker="^",
            s=140,
            label="VectorQ",
        )

        # Add confidence intervals for dynamic thresholds
        if vectorq_results:
            # Prepare data for error bars
            xerr = []
            yerr = []

            for i, rnd_ub in enumerate(dynamic_rnd_ubs_sorted):
                if (
                    rnd_ub in vectorq_results
                    and len(vectorq_results[rnd_ub]["durations"]) > 1
                ):
                    # Calculate 95% confidence interval
                    durations = vectorq_results[rnd_ub]["durations"]
                    error_rates = vectorq_results[rnd_ub]["error_rates"]

                    # Standard error of the mean
                    duration_std = np.std(durations, ddof=1)
                    error_rate_std = np.std(error_rates, ddof=1)

                    # 95% confidence interval factor (for small samples)
                    t_factor = stats.t.ppf(0.975, len(durations) - 1)  # 95% CI

                    # Confidence interval
                    duration_ci = t_factor * (duration_std / np.sqrt(len(durations)))
                    error_rate_ci = t_factor * (
                        error_rate_std / np.sqrt(len(error_rates))
                    )

                    xerr.append(duration_ci)
                    yerr.append(error_rate_ci)
                else:
                    xerr.append(0)
                    yerr.append(0)

            plt.errorbar(
                dynamic_durations_sorted,
                dynamic_error_rates_sorted,
                xerr=xerr,
                yerr=yerr,
                fmt="none",
                ecolor="orange",
                alpha=0.75,
                capsize=5,
                label="95% Confidence Interval",
            )

    # Add vertical line for direct inference (no cache)
    if direct_duration is not None:
        ylim = plt.ylim()
        plt.axvline(x=direct_duration, color="grey", linestyle="--", linewidth=4)

        # Get current x-axis limits
        xlim = plt.xlim()
        right_margin = xlim[1] - direct_duration

        # Calculate text position, ensuring it stays within bounds
        # Check if we have enough space on the right
        if right_margin > 5:  # If we have enough space to the right
            text_x = direct_duration + 1
            text_va = "top"
            text_ha = "left"
            rotation = 90
        else:  # If we're too close to the right edge
            text_x = direct_duration - 1
            text_va = "top"
            text_ha = "right"
            rotation = 90

        plt.text(
            text_x,
            (ylim[1] - ylim[0]) * 0.9 + ylim[0],
            f"No Cache: {direct_duration:.1f} min",
            fontsize=FONT_SIZE - 2,
            rotation=rotation,
            va=text_va,
            ha=text_ha,
            color="grey",
        )

        if right_margin < 5:
            plt.xlim(xlim[0], direct_duration + 5)  # Add padding to the right

        plt.ylim(ylim)

        # Add annotations for dynamic thresholds
        for i, rnd_ub in enumerate(dynamic_rnd_ubs_sorted):
            if i == 0 or i == len(dynamic_rnd_ubs_sorted) - 1:
                t = plt.text(
                    dynamic_durations_sorted[i],
                    dynamic_error_rates_sorted[i],
                    f"{rnd_ub}",
                    fontsize=20,
                    bbox=dict(
                        boxstyle="round,pad=0.3", fc="mistyrose", ec="red", alpha=0.8
                    ),
                )
                texts.append(t)

    if texts:
        adjust_text(
            texts,
            expand_points=(3.5, 3.5),
            force_points=(3.5, 3.5),
        )

    plt.xlabel("Duration (minutes)", fontsize=FONT_SIZE)
    plt.ylabel("Error Rate (%)", fontsize=FONT_SIZE)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.xticks(fontsize=FONT_SIZE)
    plt.yticks(fontsize=FONT_SIZE)
    plt.ylim(bottom=0.0)
    plt.subplots_adjust(top=0.85)
    plt.legend(
        fontsize=FONT_SIZE,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.167),
        ncol=3,
        frameon=True,
        fancybox=True,
        columnspacing=1.0,
        handletextpad=0.4,
    )

    output_dir = (
        f"processed_{results_dir}{dataset}/{embedding_model_name}/{llm_model_name}/"
    )
    os.makedirs(output_dir, exist_ok=True)
    output_file = f"{output_dir}comparison_duration_vs_error_rate.pdf"
    plt.savefig(output_file, format="pdf", bbox_inches="tight")
    plt.close()
    print(f"     Duration vs Error Rate plot saved to {output_file}")


def plot_cache_hit_latency_vs_size_comparison(
    dataset, embedding_model_name, llm_model_name, timestamp, results_dir, FONT_SIZE=20
):
    static_files, dynamic_files = _get_result_files(
        dataset, embedding_model_name, llm_model_name, timestamp, results_dir
    )

    if not static_files and not dynamic_files:
        print(
            f"No results found for {dataset}, {embedding_model_name}, {llm_model_name}"
        )
        return

    # Create dictionaries to store data for each threshold
    static_data = {}  # {threshold: {'cache_sizes': [], 'hit_latencies': []}}
    dynamic_data = {}  # {delta: {'cache_sizes': [], 'hit_latencies': []}}

    # Process static threshold files
    for file_path in static_files:
        with open(file_path, "r") as f:
            data = json.load(f)

            threshold = data["config"]["threshold"]
            answers_reused = data["answers_reused"]
            cache_size_list = data["cache_size_list"]
            inference_times = data.get("inference_time_vectorq_step_size", [])

            if len(inference_times) == 0:
                continue

            cache_sizes = []
            hit_latencies = []

            # Collect points where we had cache hits
            for i, reused in enumerate(answers_reused):
                if i >= len(inference_times):
                    break
                if reused and i > 0:  # It's a cache hit and not the first element
                    cache_sizes.append(cache_size_list[i])
                    hit_latencies.append(inference_times[i])

            if len(cache_sizes) > 1:
                static_data[threshold] = {
                    "cache_sizes": cache_sizes,
                    "hit_latencies": hit_latencies,
                }

    # Process dynamic threshold files
    for file_path in dynamic_files:
        with open(file_path, "r") as f:
            data = json.load(f)

            delta = data["config"]["delta"]
            answers_reused = data["answers_reused"]
            cache_size_list = data["cache_size_list"]
            inference_times = data.get("inference_time_vectorq_step_size", [])

            if len(inference_times) == 0:
                continue

            cache_sizes = []
            hit_latencies = []

            # Collect points where we had cache hits
            for i, reused in enumerate(answers_reused):
                if i >= len(inference_times):
                    break
                if reused and i > 0:  # It's a cache hit and not the first element
                    cache_sizes.append(cache_size_list[i])
                    hit_latencies.append(inference_times[i])

            if len(cache_sizes) > 1:
                dynamic_data[delta] = {
                    "cache_sizes": cache_sizes,
                    "hit_latencies": hit_latencies,
                }

    if not static_data and not dynamic_data:
        print(
            f"No cache hit data found for {dataset}, {embedding_model_name}, {llm_model_name}"
        )
        return

    plt.figure(figsize=(12, 8))

    static_blues = plt.cm.Blues(np.linspace(0.5, 1.0, len(static_data)))
    for i, (threshold, values) in enumerate(sorted(static_data.items())):
        plt.scatter(
            values["cache_sizes"],
            values["hit_latencies"],
            color=static_blues[i],
            marker="o",
            s=140,
            label=f"Static {threshold}",
            alpha=0.7,
        )

    dynamic_reds = plt.cm.Reds(np.linspace(0.5, 1.0, len(dynamic_data)))
    for i, (delta, values) in enumerate(sorted(dynamic_data.items())):
        plt.scatter(
            values["cache_sizes"],
            values["hit_latencies"],
            color=dynamic_reds[i],
            marker="^",
            s=140,
            label=f"Dynamic {delta}",
            alpha=0.7,
        )

    # Fit trend lines for each group (static and dynamic)
    all_static_sizes = []
    all_static_latencies = []
    all_dynamic_sizes = []
    all_dynamic_latencies = []

    for values in static_data.values():
        all_static_sizes.extend(values["cache_sizes"])
        all_static_latencies.extend(values["hit_latencies"])

    for values in dynamic_data.values():
        all_dynamic_sizes.extend(values["cache_sizes"])
        all_dynamic_latencies.extend(values["hit_latencies"])

    # Fit static trend line if enough points
    if len(all_static_sizes) > 5:
        try:
            z_static = np.polyfit(all_static_sizes, all_static_latencies, 1)
            p_static = np.poly1d(z_static)
            x_range = np.linspace(min(all_static_sizes), max(all_static_sizes), 100)
            plt.plot(
                x_range,
                p_static(x_range),
                "b-",
                linewidth=3,
                label=f"Static Trend: {z_static[0]:.6f}x + {z_static[1]:.6f}",
            )
        except Exception:
            pass

    # Fit dynamic trend line if enough points
    if len(all_dynamic_sizes) > 5:
        try:
            z_dynamic = np.polyfit(all_dynamic_sizes, all_dynamic_latencies, 1)
            p_dynamic = np.poly1d(z_dynamic)
            x_range = np.linspace(min(all_dynamic_sizes), max(all_dynamic_sizes), 100)
            plt.plot(
                x_range,
                p_dynamic(x_range),
                "r-",
                linewidth=3,
                label=f"Dynamic Trend: {z_dynamic[0]:.6f}x + {z_dynamic[1]:.6f}",
            )
        except Exception:  # Replace bare except
            pass

    plt.xlabel("Cache Size (MB)")
    plt.ylabel("Cache Hit Latency (seconds)")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.ylim(bottom=0)

    handles, labels = plt.gca().get_legend_handles_labels()
    static_indices = [
        i
        for i, label in enumerate(labels)
        if "Static" in label and "Trend" not in label
    ]
    dynamic_indices = [
        i
        for i, label in enumerate(labels)
        if "Dynamic" in label and "Trend" not in label
    ]
    trend_indices = [i for i, label in enumerate(labels) if "Trend" in label]

    static_pairs = [(float(labels[i].split()[-1]), i) for i in static_indices]
    dynamic_pairs = [(float(labels[i].split()[-1]), i) for i in dynamic_indices]

    static_pairs.sort()
    dynamic_pairs.sort()

    ordered_indices = (
        [p[1] for p in static_pairs] + [p[1] for p in dynamic_pairs] + trend_indices
    )
    plt.legend(
        [handles[i] for i in ordered_indices],
        [labels[i] for i in ordered_indices],
        loc="best",
        ncol=2,
    )

    output_dir = (
        f"processed_{results_dir}{dataset}/{embedding_model_name}/{llm_model_name}/"
    )
    os.makedirs(output_dir, exist_ok=True)
    output_file = f"{output_dir}comparison_latency_vs_size.pdf"
    plt.savefig(output_file, format="pdf", bbox_inches="tight")
    plt.close()
    print(f"     Cache hit latency vs cache size plot saved to {output_file}")


def plot_hit_rate_vs_latency(
    dataset, embedding_model_name, llm_model_name, timestamp, results_dir, FONT_SIZE=20
):
    static_files, dynamic_files = _get_result_files(
        dataset, embedding_model_name, llm_model_name, timestamp, results_dir
    )

    if not static_files and not dynamic_files:
        print(
            f"No results found for {dataset}, {embedding_model_name}, {llm_model_name}"
        )
        return

    static_hit_rates = []
    static_latencies = []
    static_thresholds = []

    dynamic_hit_rates = []
    dynamic_latencies = []
    dynamic_rnd_ubs = []

    # Dictionary to group vectorq results by delta
    vectorq_results = {}

    # Process static threshold files
    for file_path in static_files:
        with open(file_path, "r") as f:
            data = json.load(f)

            total_samples = data["sample_sizes"][-1] + 1  # Add 1 because it's 0-indexed
            total_reused = data["total_reused_list"][-1]
            hit_rate = (total_reused / total_samples) * 100 if total_samples > 0 else 0

            answers_reused = data["answers_reused"]
            inference_times = data.get("inference_time_vectorq_step_size", [])

            if len(inference_times) == 0 or total_reused == 0:
                continue

            hit_latencies = []

            # Collect latencies for cache hits
            for i, reused in enumerate(answers_reused):
                if i >= len(inference_times):
                    break
                if reused and i > 0:  # It's a cache hit and not the first element
                    hit_latencies.append(inference_times[i])

            if hit_latencies:
                avg_latency = np.mean(hit_latencies)
                threshold = data["config"]["threshold"]

                static_hit_rates.append(hit_rate)
                static_latencies.append(avg_latency)
                static_thresholds.append(threshold)

    # Process dynamic threshold files
    for file_path in dynamic_files:
        with open(file_path, "r") as f:
            data = json.load(f)

            total_samples = data["sample_sizes"][-1] + 1
            total_reused = data["total_reused_list"][-1]
            hit_rate = (total_reused / total_samples) * 100 if total_samples > 0 else 0

            answers_reused = data["answers_reused"]
            inference_times = data.get("inference_time_vectorq_step_size", [])
            delta = data["config"]["delta"]

            if len(inference_times) == 0 or total_reused == 0:
                continue

            hit_latencies = []

            # Collect latencies for cache hits
            for i, reused in enumerate(answers_reused):
                if i >= len(inference_times):
                    break
                if reused and i > 0:  # It's a cache hit and not the first element
                    hit_latencies.append(inference_times[i])

            if hit_latencies:
                avg_latency = np.mean(hit_latencies)

                # Group results by delta for confidence intervals
                if "vectorq_" in file_path:
                    if delta not in vectorq_results:
                        vectorq_results[delta] = {"hit_rates": [], "latencies": []}

                    vectorq_results[delta]["hit_rates"].append(hit_rate)
                    vectorq_results[delta]["latencies"].append(avg_latency)
                else:
                    # Legacy support for old files
                    dynamic_hit_rates.append(hit_rate)
                    dynamic_latencies.append(avg_latency)
                    dynamic_rnd_ubs.append(delta)

    # For new runs with multiple files per delta, compute average and confidence intervals
    if vectorq_results:
        # For each delta, compute average and append to dynamic lists
        for delta, values in sorted(
            vectorq_results.items(), key=lambda x: float(x[0])
        ):
            hit_rates = values["hit_rates"]
            latencies = values["latencies"]

            if hit_rates and latencies:
                avg_hit_rate = np.mean(hit_rates)
                avg_latency = np.mean(latencies)

                dynamic_hit_rates.append(avg_hit_rate)
                dynamic_latencies.append(avg_latency)
                dynamic_rnd_ubs.append(delta)

    plt.rcParams.update({"font.size": FONT_SIZE})
    plt.figure(figsize=(10, 6.5))

    if static_hit_rates:
        static_points = sorted(
            zip(static_hit_rates, static_latencies, static_thresholds),
            key=lambda x: x[0],
        )
        static_hit_rates_sorted = [point[0] for point in static_points]
        static_latencies_sorted = [point[1] for point in static_points]

        plt.plot(
            static_hit_rates_sorted,
            static_latencies_sorted,
            "b-",
            alpha=0.5,
            linewidth=3,
        )
        plt.scatter(
            static_hit_rates_sorted,
            static_latencies_sorted,
            c="blue",
            marker="o",
            s=140,
            label="State of the Art",
        )

    if dynamic_hit_rates:
        dynamic_points = sorted(
            zip(dynamic_hit_rates, dynamic_latencies, dynamic_rnd_ubs),
            key=lambda x: x[0],
        )
        dynamic_hit_rates_sorted = [point[0] for point in dynamic_points]
        dynamic_latencies_sorted = [point[1] for point in dynamic_points]
        dynamic_rnd_ubs_sorted = [point[2] for point in dynamic_points]

        plt.plot(
            dynamic_hit_rates_sorted,
            dynamic_latencies_sorted,
            "r-",
            alpha=0.7,
            linewidth=3,
        )
        plt.scatter(
            dynamic_hit_rates_sorted,
            dynamic_latencies_sorted,
            c="red",
            marker="^",
            s=140,
            label="VectorQ",
        )

        # Add confidence intervals if we have multiple runs
        if vectorq_results:
            # Prepare data for error bars
            xerr = []
            yerr = []

            for i, rnd_ub in enumerate(dynamic_rnd_ubs_sorted):
                if (
                    rnd_ub in vectorq_results
                    and len(vectorq_results[rnd_ub]["hit_rates"]) > 1
                ):
                    # Calculate 95% confidence interval
                    hit_rates = vectorq_results[rnd_ub]["hit_rates"]
                    latencies = vectorq_results[rnd_ub]["latencies"]

                    # Standard error of the mean
                    hit_rate_std = np.std(hit_rates, ddof=1)
                    latency_std = np.std(latencies, ddof=1)

                    # 95% confidence interval factor (for small samples)
                    t_factor = stats.t.ppf(0.975, len(hit_rates) - 1)  # 95% CI

                    # Confidence interval
                    hit_rate_ci = t_factor * (hit_rate_std / np.sqrt(len(hit_rates)))
                    latency_ci = t_factor * (latency_std / np.sqrt(len(latencies)))

                    xerr.append(hit_rate_ci)
                    yerr.append(latency_ci)
                else:
                    xerr.append(0)
                    yerr.append(0)

            plt.errorbar(
                dynamic_hit_rates_sorted,
                dynamic_latencies_sorted,
                xerr=xerr,
                yerr=yerr,
                fmt="none",
                ecolor="orange",
                alpha=0.75,
                capsize=5,
                label="95% Confidence Interval",
            )

    plt.xlabel("Cache Hit Rate (%)", fontsize=FONT_SIZE)
    plt.ylabel("Avg. Cache Hit Latency (seconds)", fontsize=FONT_SIZE)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.xticks(fontsize=FONT_SIZE)
    plt.yticks(fontsize=FONT_SIZE)
    plt.ylim(bottom=0)
    plt.subplots_adjust(top=0.85)
    plt.legend(
        fontsize=FONT_SIZE,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.167),
        ncol=3,
        frameon=True,
        fancybox=True,
        columnspacing=1.0,
        handletextpad=0.4,
    )

    output_dir = (
        f"processed_{results_dir}{dataset}/{embedding_model_name}/{llm_model_name}/"
    )
    os.makedirs(output_dir, exist_ok=True)
    output_file = f"{output_dir}comparison_hit_rate_vs_latency.pdf"
    plt.savefig(output_file, format="pdf", bbox_inches="tight")
    plt.close()
    print(f"     Hit rate vs latency plot saved to {output_file}")


def plot_roc_curve(
    dataset, embedding_model_name, llm_model_name, timestamp, results_dir, FONT_SIZE=20
):
    static_files, dynamic_files = _get_result_files(
        dataset, embedding_model_name, llm_model_name, timestamp, results_dir
    )

    if not static_files and not dynamic_files:
        print(
            f"No results found for {dataset}, {embedding_model_name}, {llm_model_name}"
        )
        return

    static_y_true = []  # Ground truth labels
    static_y_scores = []  # Prediction scores/probabilities

    dynamic_y_true = []
    dynamic_y_scores = []

    # Process static threshold files - combine all into one dataset
    for file_path in static_files:
        with open(file_path, "r") as f:
            data = json.load(f)

            if (
                "true_positive_list" in data
                and "false_positive_list" in data
                and "true_negative_list" in data
                and "false_negative_list" in data
            ):
                tp = data["true_positive_list"][-1]
                fp = data["false_positive_list"][-1]
                tn = data["true_negative_list"][-1]
                fn = data["false_negative_list"][-1]

                # 1s for actual positives (TP + FN)
                y_true_positives = [1] * (tp + fn)
                # 0s for actual negatives (TN + FP)
                y_true_negatives = [0] * (tn + fp)

                y_score_positives = [1] * tp + [0] * fn
                y_score_negatives = [1] * fp + [0] * tn

                y_true = y_true_positives + y_true_negatives
                y_score = y_score_positives + y_score_negatives
                static_y_true.extend(y_true)
                static_y_scores.extend(y_score)

    # Process dynamic threshold files - combine all into one dataset
    for file_path in dynamic_files:
        with open(file_path, "r") as f:
            data = json.load(f)

            if (
                "true_positive_list" in data
                and "false_positive_list" in data
                and "true_negative_list" in data
                and "false_negative_list" in data
            ):
                tp = data["true_positive_list"][-1]
                fp = data["false_positive_list"][-1]
                tn = data["true_negative_list"][-1]
                fn = data["false_negative_list"][-1]

                # 1s for actual positives (TP + FN)
                y_true_positives = [1] * (tp + fn)
                # 0s for actual negatives (TN + FP)
                y_true_negatives = [0] * (tn + fp)

                y_score_positives = [1] * tp + [0] * fn
                y_score_negatives = [1] * fp + [0] * tn

                y_true = y_true_positives + y_true_negatives
                y_score = y_score_positives + y_score_negatives
                dynamic_y_true.extend(y_true)
                dynamic_y_scores.extend(y_score)

    plt.rcParams.update({"font.size": FONT_SIZE})
    plt.figure(figsize=(10, 8))
    plt.plot([0, 1], [0, 1], "k--", alpha=0.8, label="Random Classifier")

    if static_y_true and static_y_scores:
        fpr, tpr, thresholds = roc_curve(static_y_true, static_y_scores)
        roc_auc = auc(fpr, tpr)

        plt.plot(
            fpr,
            tpr,
            "b-",
            lw=3,
            alpha=0.7,
            label=f"State of the Art (AUC = {roc_auc:.3f})",
        )

    if dynamic_y_true and dynamic_y_scores:
        fpr, tpr, thresholds = roc_curve(dynamic_y_true, dynamic_y_scores)
        roc_auc = auc(fpr, tpr)

        plt.plot(
            fpr, tpr, "r-", lw=3, alpha=0.7, label=f"VectorQ (AUC = {roc_auc:.3f})"
        )

    plt.xlabel("False Positive Rate (1 - Specificity)", fontsize=FONT_SIZE)
    plt.ylabel("True Positive Rate (Sensitivity)", fontsize=FONT_SIZE)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xticks(fontsize=FONT_SIZE)
    plt.yticks(fontsize=FONT_SIZE)
    plt.subplots_adjust(top=0.85)
    plt.legend(
        fontsize=FONT_SIZE,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.167),
        ncol=3,
        frameon=True,
        fancybox=True,
        columnspacing=1.0,
        handletextpad=0.4,
    )

    output_dir = (
        f"processed_{results_dir}{dataset}/{embedding_model_name}/{llm_model_name}/"
    )
    os.makedirs(output_dir, exist_ok=True)
    output_file = f"{output_dir}comparison_roc_curve.pdf"
    plt.savefig(output_file, format="pdf", bbox_inches="tight")
    plt.close()
    print(f"     ROC curve comparison saved to {output_file}")
