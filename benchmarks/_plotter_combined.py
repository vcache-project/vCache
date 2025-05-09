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
                    
        # Process Berkeley embedding directories
        elif d.startswith("berkeley_embedding_") and os.path.isdir(
            os.path.join(results_dir, d)
        ):
            dir_path: str = os.path.join(results_dir, d)
            for file in os.listdir(dir_path):
                if file.startswith("results_") and file.endswith(".json"):
                    berkeley_embedding_files.append(os.path.join(dir_path, file))
                    
        # Process vCache Berkeley embedding directories
        elif d.startswith("vcache_berkeley_embedding_") and os.path.isdir(
            os.path.join(results_dir, d)
        ):
            dir_path: str = os.path.join(results_dir, d)
            for file in os.listdir(dir_path):
                if file.startswith("results_") and file.endswith(".json"):
                    vcache_berkeley_embedding_files.append(os.path.join(dir_path, file))

    return gptcache_files, vcache_local_files, vcache_global_files, berkeley_embedding_files, vcache_berkeley_embedding_files


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

    gptcache_files, vcache_local_files, vcache_global_files, berkeley_embedding_files, vcache_berkeley_embedding_files = __get_result_files(
        results_dir
    )

    if not gptcache_files and not vcache_local_files and not vcache_global_files and not berkeley_embedding_files and not vcache_berkeley_embedding_files:
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
    ### Baseline: Berkeley Embedding
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
    ### vCache + Berkeley Embedding
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
        
    try:
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
    except Exception as e:
        print(f"Error plotting cache hit vs error rate vs sample size: {e}")
        
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
            color="blue",
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
        if delta == 0.01:
            continue

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
            color="green",
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
            color="red",
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
    ### Baseline: Berkeley Embedding
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
            color="purple",
            linewidth=3,
            label="Berkeley Embedding",
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
    ### vCache + Berkeley Embedding
    vcache_berkeley_embedding_thresholds = sorted(vcache_berkeley_embedding_data_frames.keys())
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
            color="orange",
            linewidth=3,
            label="vCache + Berkeley Embedding",
            markersize=10,
        )
        
        for i, delta in enumerate(vcache_berkeley_embedding_thresholds):
            plt.annotate(
                text="",
                xy=(vcache_berkeley_embedding_fpr_values[i], vcache_berkeley_embedding_tpr_values[i]),
                textcoords="offset points",
                xytext=(0, 10),
                ha="center",
                fontsize=font_size - 4,
            )

    plt.xlabel("False Positive Rate", fontsize=font_size)
    plt.ylabel("True Positive Rate", fontsize=font_size)
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.legend(loc="lower right", fontsize=font_size - 2)
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

    filename = results_dir + f"/roc_{timestamp}.pdf"
    plt.savefig(filename, format="pdf", transparent=True)
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
    plt.figure(figsize=(12, 10))

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
            color="blue",
            linewidth=2,
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
        if delta == 0.01:
            continue

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
            color="green",
            linewidth=2,
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
            color="red",
            linewidth=2,
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
    ### Baseline: Berkeley Embedding
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
            color="purple",
            linewidth=2,
            label="Berkeley Embedding",
            markersize=8,
        )
        
        for i, threshold in enumerate(berkeley_embedding_thresholds):
            plt.annotate(
                text="",
                xy=(berkeley_embedding_recall_values[i], berkeley_embedding_precision_values[i]),
                textcoords="offset points",
                xytext=(0, 10),
                ha="center",
                fontsize=font_size - 4,
            )

    ############################################################
    ### vCache + Berkeley Embedding
    vcache_berkeley_embedding_thresholds = sorted(vcache_berkeley_embedding_data_frames.keys())
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
            color="orange",
            linewidth=2,
            label="vCache + Berkeley Embedding",
            markersize=8,
        )
        
        for i, delta in enumerate(vcache_berkeley_embedding_thresholds):
            plt.annotate(
                text="",
                xy=(vcache_berkeley_embedding_recall_values[i], vcache_berkeley_embedding_precision_values[i]),
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
    
    yticks = plt.yticks()[0]
    if yticks[0] == 0.0:
        plt.yticks(yticks[1:])

    plt.gca().spines["top"].set_linewidth(1)
    plt.gca().spines["right"].set_linewidth(1)
    plt.gca().spines["bottom"].set_linewidth(1)
    plt.gca().spines["left"].set_linewidth(1)

    filename = results_dir + f"/precision_vs_recall_{timestamp}.pdf"
    plt.savefig(filename, format="pdf", bbox_inches="tight")
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

    ############################################################
    ### Baseline: GPTCache
    gptcache_thresholds = sorted(gptcache_data_frames.keys())
    gptcache_error_rates = []
    gptcache_latencies = []

    avg_latency_no_cache = 0.0

    for threshold in gptcache_thresholds:
        df = gptcache_data_frames[threshold]

        error_rate = compute_error_rate_score(fp=df["fp_list"]) * 100
        avg_latency = compute_avg_latency_score(latency_list=df["latency_vectorq_list"])
        gptcache_error_rates.append(error_rate)
        gptcache_latencies.append(avg_latency)

        avg_latency_no_cache = compute_avg_latency_score(
            latency_list=df["latency_direct_list"]
        )

    if gptcache_thresholds:
        plt.plot(
            gptcache_latencies,
            gptcache_error_rates,
            "o-",
            color="blue",
            linewidth=2,
            label="GPTCache",
            markersize=8,
        )

        for i, threshold in enumerate(gptcache_thresholds):
            if i == 0 or i == len(gptcache_thresholds) - 1:
                label = f"{threshold:.2f}"
                plt.annotate(
                    label,
                    (gptcache_error_rates[i], gptcache_latencies[i]),
                    textcoords="offset points",
                    xytext=(0, 10),
                    ha="center",
                    fontsize=font_size - 4,
                )

        plt.axvline(
            x=avg_latency_no_cache,
            color="grey",
            linestyle="--",
            linewidth=2,
            label="No Cache",
        )

    ############################################################
    ### Baseline: vCache Local
    vcache_local_deltas = sorted(vcache_local_data_frames.keys())
    vcache_local_error_rates = []
    vcache_local_latencies = []

    for delta in vcache_local_deltas:
        if delta == 0.01:
            continue

        df = vcache_local_data_frames[delta]

        error_rate = compute_error_rate_score(fp=df["fp_list"]) * 100
        avg_latency = compute_avg_latency_score(latency_list=df["latency_vectorq_list"])
        vcache_local_error_rates.append(error_rate)
        vcache_local_latencies.append(avg_latency)

    if vcache_local_deltas:
        plt.plot(
            vcache_local_latencies,
            vcache_local_error_rates,
            "o-",
            color="green",
            linewidth=2,
            label="vCache",
            markersize=8,
        )

        for i, _ in enumerate(vcache_local_latencies):
            if i == 0:
                continue

            if i == 0 or i == len(vcache_local_deltas) - 1:
                label = f"{vcache_local_deltas[i]:.2f}"
                plt.annotate(
                    label,
                    (vcache_local_error_rates[i], vcache_local_latencies[i]),
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

        vcache_global_error_rates.append(error_rate)
        vcache_global_latencies.append(avg_latency)

    if vcache_global_deltas:
        plt.plot(
            vcache_global_latencies,
            vcache_global_error_rates,
            "o-",
            color="red",
            linewidth=2,
            label="vCache (Ablation)",
            markersize=8,
        )

        for i, delta in enumerate(vcache_global_deltas):
            if i == 0 or i == len(vcache_global_deltas) - 1:
                label = f"{delta:.2f}"
                plt.annotate(
                    label,
                    (vcache_global_error_rates[i], vcache_global_latencies[i]),
                    textcoords="offset points",
                    xytext=(0, 10),
                    ha="center",
                    fontsize=font_size - 4,
                )

    ############################################################
    ### Baseline: Berkeley Embedding
    berkeley_embedding_thresholds = sorted(berkeley_embedding_data_frames.keys())
    berkeley_embedding_error_rates = []
    berkeley_embedding_latencies = []
    
    for threshold in berkeley_embedding_thresholds:
        df = berkeley_embedding_data_frames[threshold]
        
        error_rate = compute_error_rate_score(fp=df["fp_list"]) * 100
        avg_latency = compute_avg_latency_score(latency_list=df["latency_vectorq_list"])
        berkeley_embedding_error_rates.append(error_rate)
        berkeley_embedding_latencies.append(avg_latency)
        
    if berkeley_embedding_thresholds:
        plt.plot(
            berkeley_embedding_latencies,
            berkeley_embedding_error_rates,
            "o-",
            color="purple",
            linewidth=2,
            label="Berkeley Embedding",
            markersize=8,
        )
        
        for i, threshold in enumerate(berkeley_embedding_thresholds):
            plt.annotate(
                label,
                (berkeley_embedding_error_rates[i], berkeley_embedding_latencies[i]),
                textcoords="offset points",
                xytext=(0, 10),
                ha="center",
                fontsize=font_size - 4,
            )

    ############################################################
    ### vCache + Berkeley Embedding
    vcache_berkeley_embedding_thresholds = sorted(vcache_berkeley_embedding_data_frames.keys())
    vcache_berkeley_embedding_error_rates = []
    vcache_berkeley_embedding_latencies = []
    
    for delta in vcache_berkeley_embedding_thresholds:
        df = vcache_berkeley_embedding_data_frames[delta]   
        
        error_rate = compute_error_rate_score(fp=df["fp_list"]) * 100
        avg_latency = compute_avg_latency_score(latency_list=df["latency_vectorq_list"])
        vcache_berkeley_embedding_error_rates.append(error_rate)
        vcache_berkeley_embedding_latencies.append(avg_latency)
        
    if vcache_berkeley_embedding_thresholds:
        plt.plot(
            vcache_berkeley_embedding_latencies,
            vcache_berkeley_embedding_error_rates,
            "o-",
            color="orange",
            linewidth=2,
            label="vCache + Berkeley Embedding",
            markersize=8,
        )
        
        for i, delta in enumerate(vcache_berkeley_embedding_thresholds):
            plt.annotate(
                label,
                (vcache_berkeley_embedding_error_rates[i], vcache_berkeley_embedding_latencies[i]), 
                textcoords="offset points",
                xytext=(0, 10),
                ha="center",
                fontsize=font_size - 4,
            )

    plt.xlabel("Average Latency (sec)", fontsize=font_size)
    plt.ylabel("Error Rate (%)", fontsize=font_size)
    plt.xlim(left=0.0)
    plt.ylim(bottom=0.0)
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.legend(loc="best", fontsize=font_size - 2)
    plt.tick_params(axis="both", labelsize=font_size - 2)
    
    yticks = plt.yticks()[0]
    if yticks[0] == 0.0:
        plt.yticks(yticks[1:])

    plt.gca().spines["top"].set_linewidth(1)
    plt.gca().spines["right"].set_linewidth(1)
    plt.gca().spines["bottom"].set_linewidth(1)
    plt.gca().spines["left"].set_linewidth(1)

    filename = results_dir + f"/avg_latency_vs_error_rate_{timestamp}.pdf"
    plt.savefig(filename, format="pdf", bbox_inches="tight")
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

    ############################################################
    ### Baseline: GPTCache
    gptcache_thresholds = sorted(gptcache_data_frames.keys())
    gptcache_cache_hit_rates = []
    gptcache_error_rates = []

    for threshold in gptcache_thresholds:
        df = gptcache_data_frames[threshold]

        cache_hit_rate = compute_cache_hit_rate_score(
            cache_hit_list=df["cache_hit_list"]
        ) * 100
        error_rate = compute_error_rate_score(fp=df["fp_list"]) * 100

        gptcache_cache_hit_rates.append(cache_hit_rate)
        gptcache_error_rates.append(error_rate)

    if gptcache_thresholds:
        plt.plot(
            gptcache_error_rates,
            gptcache_cache_hit_rates,
            "o-",
            color="blue",
            linewidth=3,
            label="GPTCache",
            markersize=10,
        )

        for i, threshold in enumerate(gptcache_thresholds):
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
        if delta == 0.01:
            continue

        df = vcache_local_data_frames[delta]

        cache_hit_rate = compute_cache_hit_rate_score(
            cache_hit_list=df["cache_hit_list"]
        ) * 100
        error_rate = compute_error_rate_score(fp=df["fp_list"]) * 100

        vcache_local_cache_hit_rates.append(cache_hit_rate)
        vcache_local_error_rates.append(error_rate)

    if vcache_local_deltas:
        plt.plot(
            vcache_local_error_rates,
            vcache_local_cache_hit_rates,
            "o-",
            color="green",
            linewidth=3,
            label="vCache",
            markersize=10,
        )

        for i, _ in enumerate(vcache_local_error_rates):
            if i == 0:
                continue
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

        cache_hit_rate = compute_cache_hit_rate_score(
            cache_hit_list=df["cache_hit_list"]
        ) * 100
        error_rate = compute_error_rate_score(fp=df["fp_list"]) * 100

        vcache_global_cache_hit_rates.append(cache_hit_rate)
        vcache_global_error_rates.append(error_rate)

    if vcache_global_deltas:
        plt.plot(
            vcache_global_error_rates,
            vcache_global_cache_hit_rates,
            "o-",
            color="red",
            linewidth=3,
            label="vCache (Ablation)",
            markersize=10,
        )

        for i, delta in enumerate(vcache_global_deltas):
            plt.annotate(
                text="",
                xy=(vcache_global_error_rates[i], vcache_global_cache_hit_rates[i]),
                textcoords="offset points",
                xytext=(0, 10),
                ha="center",
                fontsize=font_size - 4,
            )

    ############################################################
    ### Baseline: Berkeley Embedding
    berkeley_embedding_thresholds = sorted(berkeley_embedding_data_frames.keys())
    berkeley_embedding_cache_hit_rates = []
    berkeley_embedding_error_rates = []
    
    for threshold in berkeley_embedding_thresholds:
        df = berkeley_embedding_data_frames[threshold]  
        
        cache_hit_rate = compute_cache_hit_rate_score(
            cache_hit_list=df["cache_hit_list"]
        ) * 100
        error_rate = compute_error_rate_score(fp=df["fp_list"]) * 100
        
        berkeley_embedding_cache_hit_rates.append(cache_hit_rate)
        berkeley_embedding_error_rates.append(error_rate)
        
    if berkeley_embedding_thresholds:
        plt.plot(
            berkeley_embedding_error_rates,
            berkeley_embedding_cache_hit_rates,
            "o-",
            color="purple",
            linewidth=2,
            label="Berkeley Embedding",
            markersize=8,
        )
        
        for i, threshold in enumerate(berkeley_embedding_thresholds):
            plt.annotate(
                text="",
                xy=(berkeley_embedding_error_rates[i], berkeley_embedding_cache_hit_rates[i]),
                textcoords="offset points",
                xytext=(0, 10),
                ha="center",
                fontsize=font_size - 4,
            )

    ############################################################
    ### vCache + Berkeley Embedding
    vcache_berkeley_embedding_thresholds = sorted(vcache_berkeley_embedding_data_frames.keys())
    vcache_berkeley_embedding_cache_hit_rates = []
    vcache_berkeley_embedding_error_rates = []
    
    for delta in vcache_berkeley_embedding_thresholds:
        df = vcache_berkeley_embedding_data_frames[delta]
        
        cache_hit_rate = compute_cache_hit_rate_score(
            cache_hit_list=df["cache_hit_list"]
        ) * 100
        error_rate = compute_error_rate_score(fp=df["fp_list"]) * 100
        
        vcache_berkeley_embedding_cache_hit_rates.append(cache_hit_rate)
        vcache_berkeley_embedding_error_rates.append(error_rate)
        
    if vcache_berkeley_embedding_thresholds:
        plt.plot(
            vcache_berkeley_embedding_error_rates,
            vcache_berkeley_embedding_cache_hit_rates,
            "o-",
            color="orange",
            linewidth=2,
            label="vCache + Berkeley Embedding",
            markersize=8,
        )
        
        for i, delta in enumerate(vcache_berkeley_embedding_thresholds):
            plt.annotate(
                text="",
                xy=(vcache_berkeley_embedding_error_rates[i], vcache_berkeley_embedding_cache_hit_rates[i]),
                textcoords="offset points",
                xytext=(0, 10),
                ha="center",
                fontsize=font_size - 4,
            )

    plt.xlabel("Error Rate (%)", fontsize=font_size)
    plt.ylabel("Cache Hit Rate (%)", fontsize=font_size)
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.legend(loc="best", fontsize=font_size - 2)
    plt.tick_params(axis="both", labelsize=font_size - 2)

    plt.ylim(0, 100)
    yticks = plt.yticks()[0]
    if yticks[0] == 0.0:
        plt.yticks(yticks[1:])

    plt.gca().spines["top"].set_linewidth(1)
    plt.gca().spines["right"].set_linewidth(1)
    plt.gca().spines["bottom"].set_linewidth(1)
    plt.gca().spines["left"].set_linewidth(1)

    filename = results_dir + f"/cache_hit_vs_error_rate_{timestamp}.pdf"
    plt.savefig(filename, format="pdf", bbox_inches="tight")
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
    target_deltas = [0.02, 0.03]
    # Filter out deltas that don't exist in both dictionaries
    available_deltas = []
    for delta in target_deltas:
        if delta in vcache_local_data_frames and delta in vcache_global_data_frames:
            available_deltas.append(delta)
    
    if not available_deltas:
        print(f"No matching delta values found for generating cache hit vs error rate vs sample size plots")
        return

    # Baseline 1) VectorQ (Local)
    vcache_local_error_rates = []

    for delta in available_deltas:
        df = vcache_local_data_frames[delta]
        error_rate = compute_error_rate_score(fp=df["fp_list"])
        vcache_local_error_rates.append(error_rate)

    # Baseline 3) Static thresholds
    # Find the static thresholds that match the VectorQ (Local) error rates
    gptcache_thresholds = sorted(gptcache_data_frames.keys())
    gptcache_error_rates = []

    for threshold in gptcache_thresholds:
        df = gptcache_data_frames[threshold]
        error_rate = compute_error_rate_score(fp=df["fp_list"])
        gptcache_error_rates.append(error_rate)

    matched_gptcache_thresholds = []

    for _, target_error_rate in enumerate(vcache_local_error_rates):
        closest_idx = min(
            range(len(gptcache_error_rates)),
            key=lambda j: abs(gptcache_error_rates[j] - target_error_rate),
        )

        matched_gptcache_thresholds.append(gptcache_thresholds[closest_idx])

    # Plot the results
    for i, delta in enumerate(available_deltas):
        df_vcache_local = vcache_local_data_frames[delta]
        df_vcache_global = vcache_global_data_frames[delta]
        gptcache_threshold = matched_gptcache_thresholds[i]
        df_gptcache = gptcache_data_frames[gptcache_threshold]

        vcache_local_error_rates = compute_error_rate_cumulative_list(
            fp=df_vcache_local["fp_list"]
        )
        vcache_global_error_rates = compute_error_rate_cumulative_list(
            fp=df_vcache_global["fp_list"]
        )
        gptcache_error_rates = compute_error_rate_cumulative_list(
            fp=df_gptcache["fp_list"]
        )

        vcache_local_cache_hit_rates = compute_cache_hit_rate_cumulative_list(
            cache_hit_list=df_vcache_local["cache_hit_list"]
        )
        vcache_global_cache_hit_rates = compute_cache_hit_rate_cumulative_list(
            cache_hit_list=df_vcache_global["cache_hit_list"]
        )
        gptcache_cache_hit_rates = compute_cache_hit_rate_cumulative_list(
            cache_hit_list=df_gptcache["cache_hit_list"]
        )

        sample_sizes = np.arange(1, len(vcache_local_error_rates) + 1)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))

        # Plot 1: Error Rate vs Sample Size
        ax1.plot(
            sample_sizes,
            vcache_local_error_rates,
            "-",
            color="green",
            linewidth=2,
            label=f"VectorQ (δ={delta:.2f})",
        )

        ax1.plot(
            sample_sizes,
            vcache_global_error_rates,
            "-",
            color="red",
            linewidth=2,
            label=f"VectorQ Ablation (δ={delta:.2f})",
        )

        ax1.plot(
            sample_sizes,
            gptcache_error_rates,
            "-",
            color="blue",
            linewidth=2,
            label=f"GPTCache (t={gptcache_threshold:.2f})",
        )

        ax1.set_xlabel("Sample Size", fontsize=font_size)
        ax1.set_ylabel("Cumulative Error Rate (%)", fontsize=font_size)
        ax1.grid(True, linestyle="--", alpha=0.7)
        ax1.legend(fontsize=font_size - 2)
        ax1.tick_params(axis="both", labelsize=font_size - 2)

        # Plot 2: Cache Hit Rate vs Sample Size
        ax2.plot(
            sample_sizes,
            vcache_local_cache_hit_rates,
            "-",
            color="green",
            linewidth=2,
            label=f"VectorQ (δ={delta:.2f})",
        )

        ax2.plot(
            sample_sizes,
            vcache_global_cache_hit_rates,
            "-",
            color="red",
            linewidth=2,
            label=f"VectorQ Ablation (δ={delta:.2f})",
        )

        ax2.plot(
            sample_sizes,
            gptcache_cache_hit_rates,
            "-",
            color="blue",
            linewidth=2,
            label=f"GPTCache (t={gptcache_threshold:.2f})",
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
    vcache_local_data_frames: Dict[float, pd.DataFrame],
    vcache_global_data_frames: Dict[float, pd.DataFrame],
    vcache_berkeley_embedding_data_frames: Dict[float, pd.DataFrame],
    results_dir: str,
    timestamp: str,
    font_size: int,
):
    plt.figure(figsize=(16, 10))

    vcache_local_deltas = sorted(vcache_local_data_frames.keys())

    if vcache_local_deltas:
        error_rates = []
        delta_labels = []

        for delta in vcache_local_deltas:
            df = vcache_local_data_frames[delta]

            error_rate = compute_error_rate_score(fp=df["fp_list"])

            error_rates.append(error_rate)
            delta_labels.append(f"{delta:.3f}")

        x_pos = np.arange(len(vcache_local_deltas))
        bar_width = 0.8

        plt.bar(
            x_pos, error_rates, bar_width, color="skyblue", label="Achieved Error Rate"
        )

        for i, delta in enumerate(vcache_local_deltas):
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
        
        yticks = plt.yticks()[0]
        if yticks[0] == 0.0:
            plt.yticks(yticks[1:])

        plt.gca().spines["top"].set_linewidth(1)
        plt.gca().spines["right"].set_linewidth(1)
        plt.gca().spines["bottom"].set_linewidth(1)
        plt.gca().spines["left"].set_linewidth(1)

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
                vcache_local_deltas[i] + 0.002,
                "",
                ha="center",
                va="bottom",
                fontsize=font_size - 2,
                color="red",
            )

        all_values = error_rates + vcache_local_deltas
        if all_values:
            y_min = 0
            y_max = max(all_values) * 1.15
            plt.ylim(y_min, y_max)

    filename = results_dir + f"/delta_accuracy_{timestamp}.pdf"
    plt.savefig(filename, format="pdf", bbox_inches="tight")
    plt.close()
