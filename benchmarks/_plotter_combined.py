import json
import os
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from benchmarks._plotter_helper import (
    compute_avg_latency_score,
    compute_cache_hit_rate_score,
    compute_error_rate_score,
    compute_false_positive_rate_score,
    compute_precision_score,
    compute_recall_score,
    convert_to_dataframe_from_json_file,
)


def __get_result_files(
    dataset: str, embedding_model_name: str, llm_model_name: str, results_dir: str
):
    base_dir: str = f"{results_dir}{dataset}/{embedding_model_name}/{llm_model_name}/"

    if not os.path.exists(base_dir):
        print(f"No results found in {base_dir}")
        return [], []

    static_files: List[str] = []
    dynamic_files: List[str] = []

    for d in os.listdir(base_dir):
        # Process static threshold directories
        if d.startswith("static_") and os.path.isdir(os.path.join(base_dir, d)):
            dir_path: str = os.path.join(base_dir, d)
            for file in os.listdir(dir_path):
                if file.startswith("results_") and file.endswith(".json"):
                    static_files.append(os.path.join(dir_path, file))

        # Process vectorq (dynamic threshold) directories
        elif d.startswith("vectorq_") and os.path.isdir(os.path.join(base_dir, d)):
            dir_path: str = os.path.join(base_dir, d)
            for file in os.listdir(dir_path):
                if file.startswith("results_") and file.endswith(".json"):
                    dynamic_files.append(os.path.join(dir_path, file))

    return static_files, dynamic_files
    
def generate_combined_plots(dataset: str, embedding_model_name: str, llm_model_name: str, results_dir: str, timestamp: str, font_size: int):
    static_files, dynamic_files = __get_result_files(
        dataset=dataset,
        embedding_model_name=embedding_model_name,
        llm_model_name=llm_model_name,
        results_dir=results_dir
    )
    
    if not static_files and not dynamic_files:
        print(
            f"No folders found for {dataset}, {embedding_model_name}, {llm_model_name}\n"
            f"in {results_dir}"
        )
        return
    
    static_data_frames: Dict[float, pd.DataFrame] = {}
    for static_file_path in static_files:
        with open(static_file_path, "r") as f:
            data: Any = json.load(f)
            dataframe: pd.DataFrame = convert_to_dataframe_from_json_file(data)
            threshold: float = data["config"]["threshold"]
            static_data_frames[threshold] = dataframe

    dynamic_data_frames: Dict[float, pd.DataFrame] = {}
    for dynamic_file_path in dynamic_files:
        with open(dynamic_file_path, "r") as f:
            data: Any = json.load(f)     
            dataframe: pd.DataFrame = convert_to_dataframe_from_json_file(data)   
            delta: float = data["config"]["delta"]
            dynamic_data_frames[delta] = dataframe

    __plot_roc(static_data_frames=static_data_frames, dynamic_data_frames=dynamic_data_frames, results_dir=results_dir, timestamp=timestamp, font_size=font_size)
    __plot_precision_vs_recall(static_data_frames=static_data_frames, dynamic_data_frames=dynamic_data_frames, results_dir=results_dir, timestamp=timestamp, font_size=font_size)
    __plot_avg_latency_vs_error_rate(static_data_frames=static_data_frames, dynamic_data_frames=dynamic_data_frames, results_dir=results_dir, timestamp=timestamp, font_size=font_size)
    __plot_cache_hit_vs_error_rate(static_data_frames=static_data_frames, dynamic_data_frames=dynamic_data_frames, results_dir=results_dir, timestamp=timestamp, font_size=font_size)
    __plot_delta_accuracy(dynamic_data_frames=dynamic_data_frames, results_dir=results_dir, timestamp=timestamp, font_size=font_size)
    
def __plot_roc(static_data_frames: Dict[float, pd.DataFrame], dynamic_data_frames: Dict[float, pd.DataFrame], results_dir: str, timestamp: str, font_size: int):
    plt.figure(figsize=(12, 10))
    
    static_thresholds = sorted(static_data_frames.keys())
    static_tpr_values = []
    static_fpr_values = []
    
    for threshold in static_thresholds:
        df = static_data_frames[threshold]
        
        tpr = compute_recall_score(
            tp=df['true_positive_acc_list'], 
            fn=df['false_negative_acc_list']
        )
    
        fpr = compute_false_positive_rate_score(
            fp=df['false_positive_acc_list'],
            tn=df['true_negative_acc_list']
        )
        
        static_tpr_values.append(tpr)
        static_fpr_values.append(fpr)
    
    if static_thresholds:
        plt.plot(static_fpr_values, static_tpr_values, 'o-', color='blue', linewidth=2, 
                 label='Static thresholds', markersize=8)
        
        for i, threshold in enumerate(static_thresholds):
            if i == 0 or i == len(static_thresholds) - 1:
                label = f"{threshold:.2f}"
            else:
                label = None
            plt.annotate(label, 
                       (static_fpr_values[i], static_tpr_values[i]),
                       textcoords="offset points", 
                       xytext=(0,10), 
                       ha='center',
                       fontsize=font_size-4)
    
    dynamic_deltas = sorted(dynamic_data_frames.keys())
    dynamic_tpr_values = []
    dynamic_fpr_values = []
    
    for delta in dynamic_deltas:
        df = dynamic_data_frames[delta]
        
        tpr = compute_recall_score(
            tp=df['true_positive_acc_list'], 
            fn=df['false_negative_acc_list']
        )
    
        fpr = compute_false_positive_rate_score(
            fp=df['false_positive_acc_list'],
            tn=df['true_negative_acc_list']
        )
        
        dynamic_tpr_values.append(tpr)
        dynamic_fpr_values.append(fpr)
    
    if dynamic_deltas:
        plt.plot(dynamic_fpr_values, dynamic_tpr_values, 'o-', color='green', linewidth=2, 
                 label='Deltas', markersize=8)
        
        for i, delta in enumerate(dynamic_deltas):
            if i == 0 or i == len(dynamic_deltas) - 1:
                label = f"{delta:.2f}"
            else:
                label = None
            plt.annotate(label, 
                       (dynamic_fpr_values[i], dynamic_tpr_values[i]),
                       textcoords="offset points", 
                       xytext=(0,10), 
                       ha='center',
                       fontsize=font_size-4)
    
    plt.plot([0, 1], [0, 1], 'k--', label='Random classifier')
    
    plt.xlabel('False Positive Rate', fontsize=font_size)
    plt.ylabel('True Positive Rate', fontsize=font_size)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(loc='lower right', fontsize=font_size-2)
    plt.tick_params(axis='both', labelsize=font_size-2)
    
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    
    filename = results_dir + f"/roc_{timestamp}.pdf"
    plt.savefig(filename, format="pdf", bbox_inches='tight')
    plt.close()

def __plot_precision_vs_recall(static_data_frames: Dict[float, pd.DataFrame], dynamic_data_frames: Dict[float, pd.DataFrame], results_dir: str, timestamp: str, font_size: int):
    plt.figure(figsize=(12, 10))
    
    static_thresholds = sorted(static_data_frames.keys())
    static_precision_values = []
    static_recall_values = []
    
    for threshold in static_thresholds:
        df = static_data_frames[threshold]
        
        tp = df['true_positive_acc_list'].iloc[-1]
        fp = df['false_positive_acc_list'].iloc[-1]
        fn = df['false_negative_acc_list'].iloc[-1]
        precision = compute_precision_score(tp=pd.Series([tp]), fp=pd.Series([fp]))
        recall = compute_recall_score(tp=pd.Series([tp]), fn=pd.Series([fn]))
    
        static_precision_values.append(precision)
        static_recall_values.append(recall)
    
    if static_thresholds:
        plt.plot(static_recall_values, static_precision_values, 'o-', color='blue', linewidth=2, 
                 label='Static thresholds', markersize=8)
        
        for i, threshold in enumerate(static_thresholds):
            if i == 0 or i == len(static_thresholds) - 1:
                label = f"{threshold:.2f}"
            else:
                label = None
            plt.annotate(label, 
                         (static_recall_values[i], static_precision_values[i]),
                         textcoords="offset points", 
                         xytext=(0,10), 
                         ha='center',
                         fontsize=font_size-4)
    

    dynamic_deltas = sorted(dynamic_data_frames.keys())
    dynamic_precision_values = []
    dynamic_recall_values = []
    
    for delta in dynamic_deltas:
        df = dynamic_data_frames[delta]
        
        tp = df['true_positive_acc_list'].iloc[-1]
        fp = df['false_positive_acc_list'].iloc[-1]
        fn = df['false_negative_acc_list'].iloc[-1]
        precision = compute_precision_score(tp=pd.Series([tp]), fp=pd.Series([fp]))
        recall = compute_recall_score(tp=pd.Series([tp]), fn=pd.Series([fn]))
        
        dynamic_precision_values.append(precision)
        dynamic_recall_values.append(recall)
    
    if dynamic_deltas:
        plt.plot(dynamic_recall_values, dynamic_precision_values, 'o-', color='green', linewidth=2, 
                 label='Deltas', markersize=8)
        
        for i, delta in enumerate(dynamic_deltas):
            if i == 0 or i == len(dynamic_deltas) - 1:
                label = f"{delta:.2f}"
            else:
                label = None
            plt.annotate(label, 
                         (dynamic_recall_values[i], dynamic_precision_values[i]),
                         textcoords="offset points", 
                         xytext=(0,10), 
                         ha='center',
                         fontsize=font_size-4)
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall', fontsize=font_size)
    plt.ylabel('Precision', fontsize=font_size)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(loc='best', fontsize=font_size-2)
    plt.tick_params(axis='both', labelsize=font_size-2)
    
    filename = results_dir + f"/precision_vs_recall_{timestamp}.pdf"
    plt.savefig(filename, format="pdf", bbox_inches='tight')
    plt.close()

def __plot_avg_latency_vs_error_rate(static_data_frames: Dict[float, pd.DataFrame], dynamic_data_frames: Dict[float, pd.DataFrame], results_dir: str, timestamp: str, font_size: int):
    plt.figure(figsize=(12, 10))
    
    static_thresholds = sorted(static_data_frames.keys())
    static_error_rates = []
    static_latencies = []
    
    for threshold in static_thresholds:
        df = static_data_frames[threshold]
        
        error_rate = compute_error_rate_score(
            tp=df['true_positive_acc_list'],
            fp=df['false_positive_acc_list'],
            tn=df['true_negative_acc_list'],
            fn=df['false_negative_acc_list']
        )
        
        avg_latency = compute_avg_latency_score(latency_list=df['latency_vectorq_list'])
        
        static_error_rates.append(error_rate)
        static_latencies.append(avg_latency)
    
    dynamic_deltas = sorted(dynamic_data_frames.keys())
    dynamic_error_rates = []
    dynamic_latencies = []
    
    for delta in dynamic_deltas:
        df = dynamic_data_frames[delta]
        
        error_rate = compute_error_rate_score(
            tp=df['true_positive_acc_list'],
            fp=df['false_positive_acc_list'],
            tn=df['true_negative_acc_list'],
            fn=df['false_negative_acc_list']
        )
        
        avg_latency = compute_avg_latency_score(latency_list=df['latency_vectorq_list'])
        
        dynamic_error_rates.append(error_rate)
        dynamic_latencies.append(avg_latency)
    
    if static_thresholds:
        plt.plot(static_error_rates, static_latencies, 'o-', color='blue', 
                 linewidth=2, label='Static thresholds', markersize=8)
        
        for i, threshold in enumerate(static_thresholds):
            if i == 0 or i == len(static_thresholds) - 1:
                label = f"{threshold:.2f}"
                plt.annotate(label, 
                           (static_error_rates[i], static_latencies[i]),
                           textcoords="offset points", 
                           xytext=(0,10), 
                           ha='center',
                           fontsize=font_size-4)
    
    if dynamic_deltas:
        plt.plot(dynamic_error_rates, dynamic_latencies, 'o-', color='green', 
                 linewidth=2, label='Deltas', markersize=8)
        
        for i, delta in enumerate(dynamic_deltas):
            if i == 0 or i == len(dynamic_deltas) - 1:
                label = f"{delta:.2f}"
                plt.annotate(label, 
                           (dynamic_error_rates[i], dynamic_latencies[i]),
                           textcoords="offset points", 
                           xytext=(0,10), 
                           ha='center',
                           fontsize=font_size-4)
    
    plt.xlabel('Error Rate', fontsize=font_size)
    plt.ylabel('Average Latency (s)', fontsize=font_size)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(loc='best', fontsize=font_size-2)
    plt.tick_params(axis='both', labelsize=font_size-2)
    
    # Adjust axis limits with padding if data exists
    if static_error_rates or dynamic_error_rates:
        all_error_rates = static_error_rates + dynamic_error_rates
        all_latencies = static_latencies + dynamic_latencies
        
        x_min = min(all_error_rates) if all_error_rates else 0
        x_max = max(all_error_rates) if all_error_rates else 1
        y_min = min(all_latencies) if all_latencies else 0
        y_max = max(all_latencies) if all_latencies else 1
        
        x_padding = (x_max - x_min) * 0.05 if x_max > x_min else 0.01
        y_padding = (y_max - y_min) * 0.05 if y_max > y_min else 0.01
        
        plt.xlim(max(0, x_min - x_padding), min(1, x_max + x_padding))
        plt.ylim(max(0, y_min - y_padding), y_max + y_padding)
    
    filename = results_dir + f"/avg_latency_vs_error_rate_{timestamp}.pdf"
    plt.savefig(filename, format="pdf", bbox_inches='tight')
    plt.close()

def __plot_cache_hit_vs_error_rate(static_data_frames: Dict[float, pd.DataFrame], dynamic_data_frames: Dict[float, pd.DataFrame], results_dir: str, timestamp: str, font_size: int):
    plt.figure(figsize=(12, 10))

    static_thresholds = sorted(static_data_frames.keys())
    static_cache_hit_rates = []
    static_error_rates = []
    
    for threshold in static_thresholds:
        df = static_data_frames[threshold]
        
        cache_hit_rate = compute_cache_hit_rate_score(
            cache_hit_list_acc=df['cache_hit_acc_list'], 
            cache_miss_list_acc=df['cache_miss_acc_list']
        )
        
        error_rate = compute_error_rate_score(
            tp=df['true_positive_acc_list'],
            fp=df['false_positive_acc_list'],
            tn=df['true_negative_acc_list'],
            fn=df['false_negative_acc_list']
        )
        
        static_cache_hit_rates.append(cache_hit_rate)
        static_error_rates.append(error_rate)

    if static_thresholds:
        plt.plot(static_error_rates, static_cache_hit_rates, 'o-', color='blue', linewidth=2, 
                 label='Static thresholds', markersize=8)
        
        for i, threshold in enumerate(static_thresholds):
            if i == 0 or i == len(static_thresholds) - 1:
                label = f"{threshold:.2f}"
            else:
                label = None
            plt.annotate(label, 
                       (static_error_rates[i], static_cache_hit_rates[i]),
                       textcoords="offset points", 
                       xytext=(0,10), 
                       ha='center',
                       fontsize=font_size-4)
    

    dynamic_deltas = sorted(dynamic_data_frames.keys())
    dynamic_cache_hit_rates = []
    dynamic_error_rates = []
    
    for delta in dynamic_deltas:
        df = dynamic_data_frames[delta]
 
        cache_hit_rate = compute_cache_hit_rate_score(
            cache_hit_list_acc=df['cache_hit_acc_list'], 
            cache_miss_list_acc=df['cache_miss_acc_list']
        )
        
        error_rate = compute_error_rate_score(
            tp=df['true_positive_acc_list'],
            fp=df['false_positive_acc_list'],
            tn=df['true_negative_acc_list'],
            fn=df['false_negative_acc_list']
        )
        
        dynamic_cache_hit_rates.append(cache_hit_rate)
        dynamic_error_rates.append(error_rate)

    if dynamic_deltas:
        plt.plot(dynamic_error_rates, dynamic_cache_hit_rates, 'o-', color='green', linewidth=2, 
                 label='Deltas', markersize=8)
        
        for i, delta in enumerate(dynamic_deltas):
            if i == 0 or i == len(dynamic_deltas) - 1:
                label = f"{delta:.2f}"
            else:
                label = None
            plt.annotate(label, 
                       (dynamic_error_rates[i], dynamic_cache_hit_rates[i]),
                       textcoords="offset points", 
                       xytext=(0,10), 
                       ha='center',
                       fontsize=font_size-4)
    
    plt.xlabel('Error Rate', fontsize=font_size)
    plt.ylabel('Cache Hit Rate', fontsize=font_size)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(loc='best', fontsize=font_size-2)
    plt.tick_params(axis='both', labelsize=font_size-2)
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    
    filename = results_dir + f"/cache_hit_vs_error_rate_{timestamp}.pdf"
    plt.savefig(filename, format="pdf", bbox_inches='tight')
    plt.close()

def __plot_delta_accuracy(dynamic_data_frames: Dict[float, pd.DataFrame], results_dir: str, timestamp: str, font_size: int):    
    plt.figure(figsize=(12, 10))
    
    deltas = sorted(dynamic_data_frames.keys())
    
    if deltas:
        error_rates = []
        delta_labels = []
        
        for delta in deltas:
            df = dynamic_data_frames[delta]
            
            error_rate = compute_error_rate_score(
                tp=df['true_positive_acc_list'],
                fp=df['false_positive_acc_list'],
                tn=df['true_negative_acc_list'],
                fn=df['false_negative_acc_list']
            )
            
            error_rates.append(error_rate)
            delta_labels.append(f"{delta:.2f}")
        
        x_pos = np.arange(len(deltas))
        bar_width = 0.6
        
        plt.bar(x_pos, error_rates, bar_width, color='skyblue', label='Achieved Error Rate')
        
        for i, delta in enumerate(deltas):
            plt.hlines(delta, i - bar_width/2, i + bar_width/2, colors='red', linestyles='dashed')
        
        plt.scatter(x_pos, deltas, color='red', s=50, zorder=5, label='Delta (Upper Bound)')
        plt.plot(x_pos, deltas, 'r--', alpha=0.7)
        plt.xlabel('Delta Values', fontsize=font_size)
        plt.ylabel('Error Rate', fontsize=font_size)
        plt.xticks(x_pos, delta_labels, fontsize=font_size-2)
        plt.yticks(fontsize=font_size-2)
        plt.legend(fontsize=font_size-2)
        
        for i, err in enumerate(error_rates):
            plt.text(x_pos[i], err + 0.01, f'{err:.3f}', 
                     ha='center', va='bottom', fontsize=font_size-4)
            plt.text(x_pos[i], deltas[i] + 0.01, f'{deltas[i]:.3f}', 
                     ha='center', va='bottom', fontsize=font_size-4, color='red')
        
        all_values = error_rates + deltas
        if all_values:
            y_min = min(0, min(all_values) * 0.9)
            y_max = max(all_values) * 1.1
            plt.ylim(y_min, y_max)
    
    filename = results_dir + f"/delta_accuracy_{timestamp}.pdf"
    plt.savefig(filename, format="pdf", bbox_inches='tight')
    plt.close()
