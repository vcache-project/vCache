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

    #try:
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
    figlegend.savefig(legend_filename, format="pdf", bbox_inches="tight", transparent=True)
    plt.close(figlegend)

    lines.append(Line2D([0], [0], color="grey", linewidth=3, linestyle="--", alpha=0.7))
    labels.append("Random Classifier")

    lines.append(Line2D([0], [0], color="grey", linewidth=3, linestyle="-"))
    labels.append("No Cache")

    ax.legend(lines, labels, loc="center", ncol=3, fontsize=font_size, frameon=False)

    legend_filename = results_dir + "/legend_w_rnd_class.pdf"
    figlegend.savefig(legend_filename, format="pdf", bbox_inches="tight", transparent=True)
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

    plt.plot(
        [0, 1],
        [0, 1],
        "--",
        color="grey",
        alpha=0.7,
        linewidth=3,
        label="Random Classifier",
    )

    ############################################################
    ### Baseline: GPTCache
    gptcache_thresholds = sorted(gptcache_data_frames.keys())
    gptcache_tpr_values = []
    gptcache_fpr_values = []

    for threshold in gptcache_thresholds:
        df = gptcache_data_frames[threshold]

        tpr = compute_recall_score(tp=df["tp_list"], fn=df["fn_list"])
        fpr = compute_false_positive_rate_score(fp=df["fp_list"], tn=df["tn_list"])

        gptcache_tpr_values.append(tpr)
        gptcache_fpr_values.append(fpr)

    if gptcache_thresholds:
        plt.plot(
            gptcache_fpr_values,
            gptcache_tpr_values,
            "o-",
            color="#C23B48",
            linewidth=3,
            label="GPTCache",
            markersize=10,
        )

        for i, threshold in enumerate(gptcache_thresholds):
            plt.annotate(
                text="",
                xy=(gptcache_fpr_values[i], gptcache_tpr_values[i]),
                textcoords="offset points",
                xytext=(0, 10),
                ha="center",
                fontsize=font_size - 4,
            )

    ############################################################
    ### Baseline: vCache Local
    vcache_local_deltas = sorted(vcache_local_data_frames.keys())
    vcache_local_tpr_values = []
    vcache_local_fpr_values = []

    for delta in vcache_local_deltas:
        df = vcache_local_data_frames[delta]

        tpr = compute_recall_score(tp=df["tp_list"], fn=df["fn_list"])
        fpr = compute_false_positive_rate_score(fp=df["fp_list"], tn=df["tn_list"])

        vcache_local_tpr_values.append(tpr)
        vcache_local_fpr_values.append(fpr)

    if vcache_local_deltas:
        plt.plot(
            vcache_local_fpr_values,
            vcache_local_tpr_values,
            "o-",
            color="#8CBE94",
            linewidth=3,
            label="vCache",
            markersize=10,
        )

        for i, _ in enumerate(vcache_local_tpr_values):
            plt.annotate(
                text="",
                xy=(vcache_local_fpr_values[i], vcache_local_tpr_values[i]),
                textcoords="offset points",
                xytext=(0, 10),
                ha="center",
                fontsize=font_size - 4,
            )

    ############################################################
    ### Baseline: vCache Global
    vcache_global_deltas = sorted(vcache_global_data_frames.keys())
    vcache_global_tpr_values = []
    vcache_global_fpr_values = []

    for delta in vcache_global_deltas:
        df = vcache_global_data_frames[delta]

        tpr = compute_recall_score(tp=df["tp_list"], fn=df["fn_list"])
        fpr = compute_false_positive_rate_score(fp=df["fp_list"], tn=df["tn_list"])

        vcache_global_tpr_values.append(tpr)
        vcache_global_fpr_values.append(fpr)

    if vcache_global_deltas:
        plt.plot(
            vcache_global_fpr_values,
            vcache_global_tpr_values,
            "o-",
            color="#8CBE94",
            linewidth=3,
            label="vCache (Ablation)",
            markersize=10,
        )

        for i, delta in enumerate(vcache_global_deltas):
            plt.annotate(
                text="",
                xy=(vcache_global_fpr_values[i], vcache_global_tpr_values[i]),
                textcoords="offset points",
                xytext=(0, 10),
                ha="center",
                fontsize=font_size - 4,
            )

    ############################################################
    ### Baseline: Fine-tuned Embedding
    berkeley_embedding_thresholds = sorted(berkeley_embedding_data_frames.keys())
    berkeley_embedding_tpr_values = []
    berkeley_embedding_fpr_values = []

    for threshold in berkeley_embedding_thresholds:
        df = berkeley_embedding_data_frames[threshold]

        tpr = compute_recall_score(tp=df["tp_list"], fn=df["fn_list"])
        fpr = compute_false_positive_rate_score(fp=df["fp_list"], tn=df["tn_list"])

        berkeley_embedding_tpr_values.append(tpr)
        berkeley_embedding_fpr_values.append(fpr)

    if berkeley_embedding_thresholds:
        plt.plot(
            berkeley_embedding_fpr_values,
            berkeley_embedding_tpr_values,
            "o-",
            color="#3B686A",
            linewidth=3,
            label="Fine-tuned Embedding",
            markersize=10,
        )

        for i, threshold in enumerate(berkeley_embedding_thresholds):
            plt.annotate(
                text="",
                xy=(berkeley_embedding_fpr_values[i], berkeley_embedding_tpr_values[i]),
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
    vcache_berkeley_embedding_tpr_values = []
    vcache_berkeley_embedding_fpr_values = []

    for delta in vcache_berkeley_embedding_thresholds:
        df = vcache_berkeley_embedding_data_frames[delta]

        tpr = compute_recall_score(tp=df["tp_list"], fn=df["fn_list"])
        fpr = compute_false_positive_rate_score(fp=df["fp_list"], tn=df["tn_list"])

        vcache_berkeley_embedding_tpr_values.append(tpr)
        vcache_berkeley_embedding_fpr_values.append(fpr)

    if vcache_berkeley_embedding_thresholds:
        plt.plot(
            vcache_berkeley_embedding_fpr_values,
            vcache_berkeley_embedding_tpr_values,
            "o-",
            color="#EDBE24",
            linewidth=3,
            label="vCache + Fine-tuned Embedding",
            markersize=10,
        )

        for i, delta in enumerate(vcache_berkeley_embedding_thresholds):
            plt.annotate(
                text="",
                xy=(
                    vcache_berkeley_embedding_fpr_values[i],
                    vcache_berkeley_embedding_tpr_values[i],
                ),
                textcoords="offset points",
                xytext=(0, 10),
                ha="center",
                fontsize=font_size - 4,
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

    filename = results_dir + f"/roc.pdf"
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

    filename = results_dir + f"/precision_vs_recall.pdf"
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

    ############################################################
    ### Baseline: GPTCache
    gptcache_thresholds = sorted(gptcache_data_frames.keys())
    gptcache_error_rates = []
    gptcache_latencies = []
    
    avg_latency_no_cache = -1

    for threshold in gptcache_thresholds:
        df = gptcache_data_frames[threshold]

        error_rate = compute_error_rate_score(fp=df["fp_list"]) * 100
        avg_latency = compute_avg_latency_score(latency_list=df["latency_vectorq_list"])
        
        if error_rate <= ERROR_RATE_UPPER_BOUND:
            gptcache_error_rates.append(error_rate)
            gptcache_latencies.append(avg_latency)
            
        avg_latency_no_cache = compute_avg_latency_score(
            latency_list=df["latency_direct_list"]
        )

    if gptcache_thresholds and gptcache_error_rates:
        plt.plot(
            gptcache_latencies,
            gptcache_error_rates,
            "o-",
            color="#C23B48",
            linewidth=3,
            label="GPTCache",
            markersize=8,
        )

        for i, _ in enumerate(gptcache_error_rates):
            plt.annotate(
                text="",
                xy=(gptcache_error_rates[i], gptcache_latencies[i]),
                textcoords="offset points",
                xytext=(0, 10),
                ha="center",
                fontsize=font_size - 4,
            )

    ############################################################
    ### Baseline: No Cache
    if gptcache_thresholds and gptcache_error_rates:
        plt.axvline(
            x=avg_latency_no_cache,
            color="grey",
            linewidth=3,
            label="No Cache",
        )
        plt.annotate(
            "", 
            xy=(avg_latency_no_cache, 0),
            xytext=(avg_latency_no_cache, 1),
            xycoords=("data", "axes fraction"),
            textcoords=("data", "axes fraction"),
            arrowprops=dict(
                arrowstyle="->",
                linewidth=3,
                color="grey",
                mutation_scale=24,
            ),
            annotation_clip=True,
        )

    ############################################################
    ### Baseline: vCache Local
    vcache_local_deltas = sorted(vcache_local_data_frames.keys())
    vcache_local_error_rates = []
    vcache_local_latencies = []

    for delta in vcache_local_deltas:
        df = vcache_local_data_frames[delta]

        error_rate = compute_error_rate_score(fp=df["fp_list"]) * 100
        avg_latency = compute_avg_latency_score(latency_list=df["latency_vectorq_list"])
        
        if error_rate <= ERROR_RATE_UPPER_BOUND:
            vcache_local_error_rates.append(error_rate)
            vcache_local_latencies.append(avg_latency)

    if vcache_local_deltas and vcache_local_error_rates:
        plt.plot(
            vcache_local_latencies,
            vcache_local_error_rates,
            "o-",
            color="#37A9EC",
            linewidth=3,
            label="vCache",
            markersize=8,
        )

        for i, _ in enumerate(vcache_local_error_rates):
            plt.annotate(
                text="",
                xy=(vcache_local_error_rates[i], vcache_local_latencies[i]),
                textcoords="offset points",
                xytext=(0, 10),
                ha="center",
                fontsize=font_size - 4,
            )

    ############################################################
    ### Baseline: vCache Global
    vcache_global_deltas = sorted(vcache_global_data_frames.keys())
    vcache_global_error_rates = []
    vcache_global_latencies = []

    for delta in vcache_global_deltas:
        df = vcache_global_data_frames[delta]

        error_rate = compute_error_rate_score(fp=df["fp_list"]) * 100
        avg_latency = compute_avg_latency_score(latency_list=df["latency_vectorq_list"])

        if error_rate <= ERROR_RATE_UPPER_BOUND:
            vcache_global_error_rates.append(error_rate)
            vcache_global_latencies.append(avg_latency)

    if vcache_global_deltas and vcache_global_error_rates:
        plt.plot(
            vcache_global_latencies,
            vcache_global_error_rates,
            "o-",
            color="#8CBE94",
            linewidth=3,
            label="vCache (Ablation)",
            markersize=8,
        )

        for i, _ in enumerate(vcache_global_error_rates):
            plt.annotate(
                text="",
                xy=(vcache_global_error_rates[i], vcache_global_latencies[i]),
                textcoords="offset points",
                xytext=(0, 10),
                ha="center",
                fontsize=font_size - 4,
            )

    ############################################################
    ### Baseline: Fine-tuned Embedding
    berkeley_embedding_thresholds = sorted(berkeley_embedding_data_frames.keys())
    berkeley_embedding_error_rates = []
    berkeley_embedding_latencies = []

    for threshold in berkeley_embedding_thresholds:
        df = berkeley_embedding_data_frames[threshold]

        error_rate = compute_error_rate_score(fp=df["fp_list"]) * 100
        avg_latency = compute_avg_latency_score(latency_list=df["latency_vectorq_list"])
        
        if error_rate <= ERROR_RATE_UPPER_BOUND:
            berkeley_embedding_error_rates.append(error_rate)
            berkeley_embedding_latencies.append(avg_latency)

    if berkeley_embedding_thresholds and berkeley_embedding_error_rates:
        plt.plot(
            berkeley_embedding_latencies,
            berkeley_embedding_error_rates,
            "o-",
            color="#3B686A",
            linewidth=3,
            label="Fine-tuned Embedding",
            markersize=8,
        )

        for i, _ in enumerate(berkeley_embedding_error_rates):
            plt.annotate(
                text="",
                xy=(berkeley_embedding_error_rates[i], berkeley_embedding_latencies[i]),
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
    vcache_berkeley_embedding_error_rates = []
    vcache_berkeley_embedding_latencies = []

    for delta in vcache_berkeley_embedding_thresholds:
        df = vcache_berkeley_embedding_data_frames[delta]

        error_rate = compute_error_rate_score(fp=df["fp_list"]) * 100
        avg_latency = compute_avg_latency_score(latency_list=df["latency_vectorq_list"])
        
        if error_rate <= ERROR_RATE_UPPER_BOUND:
            vcache_berkeley_embedding_error_rates.append(error_rate)
            vcache_berkeley_embedding_latencies.append(avg_latency)

    if vcache_berkeley_embedding_thresholds and vcache_berkeley_embedding_error_rates:
        plt.plot(
            vcache_berkeley_embedding_latencies,
            vcache_berkeley_embedding_error_rates,
            "o-",
            color="#EDBE24",
            linewidth=3,
            label="vCache + Fine-tuned Embedding",
            markersize=8,
        )

        for i, _ in enumerate(vcache_berkeley_embedding_error_rates):
            plt.annotate(
                text="",
                xy=(
                    vcache_berkeley_embedding_error_rates[i],
                    vcache_berkeley_embedding_latencies[i],
                ),
                textcoords="offset points",
                xytext=(0, 10),
                ha="center",
                fontsize=font_size - 4,
            )

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

    filename = results_dir + f"/avg_latency_vs_error_rate.pdf"
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

    ############################################################
    ### Baseline: GPTCache
    gptcache_thresholds = sorted(gptcache_data_frames.keys())
    gptcache_cache_hit_rates = []
    gptcache_error_rates = []

    for threshold in gptcache_thresholds:
        df = gptcache_data_frames[threshold]

        cache_hit_rate = (
            compute_cache_hit_rate_score(cache_hit_list=df["cache_hit_list"]) * 100
        )
        error_rate = compute_error_rate_score(fp=df["fp_list"]) * 100

        if error_rate <= ERROR_RATE_UPPER_BOUND:
            gptcache_cache_hit_rates.append(cache_hit_rate)
            gptcache_error_rates.append(error_rate)

    if gptcache_thresholds and gptcache_error_rates:
        plt.plot(
            gptcache_error_rates,
            gptcache_cache_hit_rates,
            "o-",
            color="#C23B48",
            linewidth=3,
            label="GPTCache",
            markersize=10,
        )

        for i, _ in enumerate(gptcache_error_rates):
            plt.annotate(
                text="",
                xy=(gptcache_error_rates[i], gptcache_cache_hit_rates[i]),
                textcoords="offset points",
                xytext=(0, 10),
                ha="center",
                fontsize=font_size - 4,
            )

    ############################################################
    ### Baseline: vCache Local
    vcache_local_deltas = sorted(vcache_local_data_frames.keys())
    vcache_local_cache_hit_rates = []
    vcache_local_error_rates = []

    for delta in vcache_local_deltas:
        df = vcache_local_data_frames[delta]

        cache_hit_rate = (
            compute_cache_hit_rate_score(cache_hit_list=df["cache_hit_list"]) * 100
        )
        error_rate = compute_error_rate_score(fp=df["fp_list"]) * 100

        if error_rate <= ERROR_RATE_UPPER_BOUND:
            vcache_local_cache_hit_rates.append(cache_hit_rate)
            vcache_local_error_rates.append(error_rate)

    if vcache_local_deltas and vcache_local_error_rates:
        plt.plot(
            vcache_local_error_rates,
            vcache_local_cache_hit_rates,
            "o-",
            color="#37A9EC",
            linewidth=3,
            label="vCache",
            markersize=10,
        )

        for i, _ in enumerate(vcache_local_error_rates):
            plt.annotate(
                text="",
                xy=(vcache_local_error_rates[i], vcache_local_cache_hit_rates[i]),
                xytext=(0, 10),
                ha="center",
                fontsize=font_size - 4,
            )

    ############################################################
    ### Baseline: vCache Global
    vcache_global_deltas = sorted(vcache_global_data_frames.keys())
    vcache_global_cache_hit_rates = []
    vcache_global_error_rates = []

    for delta in vcache_global_deltas:
        df = vcache_global_data_frames[delta]

        cache_hit_rate = (
            compute_cache_hit_rate_score(cache_hit_list=df["cache_hit_list"]) * 100
        )
        error_rate = compute_error_rate_score(fp=df["fp_list"]) * 100

        if error_rate <= ERROR_RATE_UPPER_BOUND:
            vcache_global_cache_hit_rates.append(cache_hit_rate)
            vcache_global_error_rates.append(error_rate)

    if vcache_global_deltas and vcache_global_error_rates:
        plt.plot(
            vcache_global_error_rates,
            vcache_global_cache_hit_rates,
            "o-",
            color="#8CBE94",
            linewidth=3,
            label="vCache (Ablation)",
            markersize=10,
        )

        for i, _ in enumerate(vcache_global_error_rates):
            plt.annotate(
                text="",
                xy=(vcache_global_error_rates[i], vcache_global_cache_hit_rates[i]),
                textcoords="offset points",
                xytext=(0, 10),
                ha="center",
                fontsize=font_size - 4,
            )

    ############################################################
    ### Baseline: Fine-tuned Embedding
    berkeley_embedding_thresholds = sorted(berkeley_embedding_data_frames.keys())
    berkeley_embedding_cache_hit_rates = []
    berkeley_embedding_error_rates = []

    for threshold in berkeley_embedding_thresholds:
        df = berkeley_embedding_data_frames[threshold]

        cache_hit_rate = (
            compute_cache_hit_rate_score(cache_hit_list=df["cache_hit_list"]) * 100
        )
        error_rate = compute_error_rate_score(fp=df["fp_list"]) * 100

        if error_rate <= ERROR_RATE_UPPER_BOUND:
            berkeley_embedding_cache_hit_rates.append(cache_hit_rate)
            berkeley_embedding_error_rates.append(error_rate)

    if berkeley_embedding_thresholds and berkeley_embedding_error_rates:
        plt.plot(
            berkeley_embedding_error_rates,
            berkeley_embedding_cache_hit_rates,
            "o-",
            color="#3B686A",
            linewidth=3,
            label="Fine-tuned Embedding",
            markersize=8,
        )

        for i, _ in enumerate(berkeley_embedding_error_rates):
            plt.annotate(
                text="",
                xy=(
                    berkeley_embedding_error_rates[i],
                    berkeley_embedding_cache_hit_rates[i],
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
    vcache_berkeley_embedding_cache_hit_rates = []
    vcache_berkeley_embedding_error_rates = []

    for delta in vcache_berkeley_embedding_thresholds:
        df = vcache_berkeley_embedding_data_frames[delta]

        cache_hit_rate = (
            compute_cache_hit_rate_score(cache_hit_list=df["cache_hit_list"]) * 100
        )
        error_rate = compute_error_rate_score(fp=df["fp_list"]) * 100

        if error_rate <= ERROR_RATE_UPPER_BOUND:
            vcache_berkeley_embedding_cache_hit_rates.append(cache_hit_rate)
            vcache_berkeley_embedding_error_rates.append(error_rate)

    if vcache_berkeley_embedding_thresholds and vcache_berkeley_embedding_error_rates:
        plt.plot(
            vcache_berkeley_embedding_error_rates,
            vcache_berkeley_embedding_cache_hit_rates,
            "o-",
            color="#EDBE24",
            linewidth=3,
            label="vCache + Fine-tuned Embedding",
            markersize=8,
        )

        for i, _ in enumerate(vcache_berkeley_embedding_error_rates):
            plt.annotate(
                text="",
                xy=(
                    vcache_berkeley_embedding_error_rates[i],
                    vcache_berkeley_embedding_cache_hit_rates[i],
                ),
                textcoords="offset points",
                xytext=(0, 10),
                ha="center",
                fontsize=font_size - 4,
            )

    plt.xlabel("Error Rate (%)", fontsize=font_size)
    plt.ylabel("Cache Hit Rate (%)", fontsize=font_size)
    plt.tick_params(axis="both", labelsize=font_size - 2)

    yticks = plt.yticks()[0]
    if yticks[0] == 0.0:
        plt.yticks(yticks[1:])

    plt.gca().spines["top"].set_linewidth(1)
    plt.gca().spines["right"].set_linewidth(1)
    plt.gca().spines["bottom"].set_linewidth(1)
    plt.gca().spines["left"].set_linewidth(1)

    filename = results_dir + f"/cache_hit_vs_error_rate.pdf"
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
        vcache_local_error_rate = compute_error_rate_score(fp=vcache_local_df["fp_list"]) * 100
        vcache_local_error_rates = [rate * 100 for rate in compute_error_rate_cumulative_list(
            fp=vcache_local_df["fp_list"]
        )]
        vcache_local_cache_hit_rates = [rate * 100 for rate in compute_cache_hit_rate_cumulative_list(
            cache_hit_list=vcache_local_df["cache_hit_list"]
        )]
        
        ############################################################
        ### Baseline: vCache Global
        if vcache_global_data_frames:   
            vcache_global_df = vcache_global_data_frames[target_delta]
            vcache_global_error_rate = compute_error_rate_score(fp=vcache_global_df["fp_list"])
            vcache_global_error_rates = [rate * 100 for rate in compute_error_rate_cumulative_list(
                fp=vcache_global_df["fp_list"]
            )]
            vcache_global_cache_hit_rates = [rate * 100 for rate in compute_cache_hit_rate_cumulative_list(
                cache_hit_list=vcache_global_df["cache_hit_list"]
            )]
        
        ############################################################
        ### Baseline: GPTCache
        if gptcache_data_frames:
            gptcache_thresholds = sorted(gptcache_data_frames.keys())
            gptcache_closest_threshold = None
            gptcache_closest_error_rate_diff = float('inf')
            gptcache_df = None

            for threshold in gptcache_thresholds:
                df = gptcache_data_frames[threshold]
                error_rate = compute_error_rate_score(fp=df["fp_list"]) * 100
                error_rate_diff = abs(error_rate - target_error_rate)
                
                if error_rate_diff < gptcache_closest_error_rate_diff:
                    gptcache_closest_error_rate_diff = error_rate_diff
                    gptcache_closest_threshold = threshold
                    gptcache_df = df

            gptcache_error_rates = [rate * 100 for rate in compute_error_rate_cumulative_list(
                fp=gptcache_df["fp_list"]
            )]
            gptcache_cache_hit_rates = [rate * 100 for rate in compute_cache_hit_rate_cumulative_list(
                cache_hit_list=gptcache_df["cache_hit_list"]
            )]

        ############################################################
        ### Baseline: Berkeley Embedding
        if berkeley_embedding_data_frames:
            berkeley_embedding_thresholds = sorted(berkeley_embedding_data_frames.keys())
            berkeley_embedding_closest_threshold = None
            berkeley_embedding_closest_error_rate_diff = float('inf')
            berkeley_embedding_df = None
            
            for threshold in berkeley_embedding_thresholds:
                df = berkeley_embedding_data_frames[threshold]
                error_rate = compute_error_rate_score(fp=df["fp_list"]) * 100
                error_rate_diff = abs(error_rate - target_error_rate)
                
                if error_rate_diff < berkeley_embedding_closest_error_rate_diff:
                    berkeley_embedding_closest_error_rate_diff = error_rate_diff
                    berkeley_embedding_closest_threshold = threshold
                    berkeley_embedding_df = df
                    
            berkeley_embedding_error_rates = [rate * 100 for rate in compute_error_rate_cumulative_list(
                fp=berkeley_embedding_df["fp_list"]
            )]
            berkeley_embedding_cache_hit_rates = [rate * 100 for rate in compute_cache_hit_rate_cumulative_list(
                cache_hit_list=berkeley_embedding_df["cache_hit_list"]
            )]
            
        ############################################################
        ### vCache + Berkeley Embedding
        if vcache_berkeley_embedding_data_frames:
            vcache_berkeley_embedding_df = vcache_berkeley_embedding_data_frames[target_delta]
            vcache_berkeley_embedding_error_rate = compute_error_rate_score(fp=vcache_berkeley_embedding_df["fp_list"])
            vcache_berkeley_embedding_error_rates = [rate * 100 for rate in compute_error_rate_cumulative_list(
                fp=vcache_berkeley_embedding_df["fp_list"]
            )]
            vcache_berkeley_embedding_cache_hit_rates = [rate * 100 for rate in compute_cache_hit_rate_cumulative_list(
                cache_hit_list=vcache_berkeley_embedding_df["cache_hit_list"]
            )]

        sample_sizes = np.arange(1, len(vcache_local_error_rates) + 1)
        
        # Plot 1: Error rates vs sample size
        plt.figure(figsize=(12, 11))
        plt.plot(sample_sizes[::45], vcache_local_error_rates[::45], '-', color='#37A9EC', linewidth=4, label='vCache')
        
        if vcache_global_data_frames:
            plt.plot(sample_sizes[::45], vcache_global_error_rates[::45], '-', color='#8CBE94', linewidth=4, label='vCache (Ablation)')
            
        if gptcache_data_frames:
            plt.plot(sample_sizes[::45], gptcache_error_rates[::45], '-', color='#C23B48', linewidth=4, label='GPTCache')
            
        if berkeley_embedding_data_frames:
            plt.plot(sample_sizes[::45], berkeley_embedding_error_rates[::45], '-', color='#3B686A', linewidth=4, label='Fine-tuned Embedding')
      
        if vcache_berkeley_embedding_data_frames:
            plt.plot(sample_sizes[::45], vcache_berkeley_embedding_error_rates[::45], '-', color='#EDBE24', linewidth=4, label='vCache + Fine-tuned Embedding')
           
        plt.xlabel('Sample Size', fontsize=font_size)
        plt.ylabel('Error Rate (%)', fontsize=font_size)
    
        plt.tick_params(axis='both', labelsize=font_size-2)
        
        error_rate_filename = (
            results_dir
            + f"/error_rate_vs_sample_size_delta_{target_delta:.3f}.pdf"
        )
        plt.savefig(error_rate_filename, format="pdf", bbox_inches="tight", transparent=True)
        plt.close()
        
        # Plot 2: Cache hit rates vs sample size
        plt.figure(figsize=(12, 11))
        plt.plot(sample_sizes[::5], vcache_local_cache_hit_rates[::5], '-', color='#37A9EC', linewidth=4, label='vCache Local')
      
        if vcache_global_data_frames:
            plt.plot(sample_sizes[::5], vcache_global_cache_hit_rates[::5], '-', color='#8CBE94', linewidth=4, label='vCache Global')
        
        if gptcache_data_frames:
            plt.plot(sample_sizes[::5], gptcache_cache_hit_rates[::5], '-', color='#C23B48', linewidth=4, label='GPTCache')
          
        if berkeley_embedding_data_frames:
            plt.plot(sample_sizes[::5], berkeley_embedding_cache_hit_rates[::5], '-', color='#3B686A', linewidth=4, label='Fine-tuned Embedding')
            
        if vcache_berkeley_embedding_data_frames:
            plt.plot(sample_sizes[::5], vcache_berkeley_embedding_cache_hit_rates[::5], '-', color='#EDBE24', linewidth=4, label='vCache + Fine-tuned Embedding')
        
        plt.xlabel('Sample Size', fontsize=font_size)
        plt.ylabel('Cache Hit Rate (%)', fontsize=font_size)
        plt.tick_params(axis='both', labelsize=font_size-2)
        
        cache_hit_filename = (
            results_dir
            + f"/cache_hit_rate_vs_sample_size_delta_{target_delta:.3f}.pdf"
        )
        plt.savefig(cache_hit_filename, format="pdf", bbox_inches="tight", transparent=True)
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

    vcache_local_deltas = sorted(vcache_local_data_frames.keys())

    if vcache_local_deltas:
        error_rates = []
        delta_labels = []

        for delta in vcache_local_deltas:
            df = vcache_local_data_frames[delta]

            error_rate = compute_error_rate_score(fp=df["fp_list"])

            error_rates.append(error_rate)
            delta_labels.append(f".{int(delta*1000):03d}")

        x_pos = np.arange(len(vcache_local_deltas))
        bar_width = 0.8

        plt.bar(
            x_pos, error_rates, bar_width, color="#37A9EC", label="Actual Error Rate"
        )

        for i, delta in enumerate(vcache_local_deltas):
            plt.hlines(
                y=delta,
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
            fontsize=font_size - 6,
            handlelength=1.1
        )

        plt.xlabel("$\delta$ Values", fontsize=font_size)
        plt.xticks(rotation=40)
        plt.ylabel("Error Rate", fontsize=font_size)
        plt.xticks(x_pos, delta_labels, fontsize=font_size - 2)
        plt.yticks(fontsize=font_size - 2)

        yticks = plt.yticks()[0]
        if yticks[0] == 0.0:
            plt.yticks(yticks[1:])
        
        def format_tick(x, pos):
            if x <= 0:
                return '0'
            s = f'{x:.3f}'
            after_decimal = s.split('.')[1].rstrip('0')
            after_decimal = after_decimal.lstrip('0') if len(after_decimal.lstrip('0')) > 0 else after_decimal[-1]
            return f'.{after_decimal}'
            
        formatter = plt.FuncFormatter(format_tick)
        plt.gca().yaxis.set_major_formatter(formatter)

        plt.gca().spines["top"].set_linewidth(1)
        plt.gca().spines["right"].set_linewidth(1)
        plt.gca().spines["bottom"].set_linewidth(1)
        plt.gca().spines["left"].set_linewidth(1)

        for i, err in enumerate(error_rates):
            plt.text(
                x_pos[i],
                err + 0.003,
                #f"{err:.3f}",
                "",
                ha="center",
                va="bottom",
                fontsize=font_size - 2,
            )

            plt.text(
                x_pos[i],
                vcache_local_deltas[i] + 0.002,
                "",
                ha="center",
                va="bottom",
                fontsize=font_size - 2,
                color="#C23B48",
            )

        all_values = error_rates + vcache_local_deltas
        if all_values:
            y_min = 0
            y_max = max(all_values) * 1.15
            plt.ylim(y_min, y_max)

    filename = results_dir + f"/delta_accuracy.pdf"
    plt.savefig(filename, format="pdf", bbox_inches="tight", transparent=True)
    plt.close()
