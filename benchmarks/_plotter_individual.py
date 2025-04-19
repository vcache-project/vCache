import matplotlib.pyplot as plt
import os
import numpy as np
from matplotlib.patches import Patch
import seaborn as sns
from typing import List, Tuple
from benchmark import Benchmark

def plot_error_rate_relative(benchmark: Benchmark, FONT_SIZE=20):
    plt.rcParams.update({'font.size': FONT_SIZE})
    plt.figure(figsize=(16, 7))
    plt.plot(benchmark.sample_sizes, benchmark.error_rates_relative_to_reused_answers, color='blue')

    for i, _ in enumerate(benchmark.sample_sizes):
        plt.annotate("", (benchmark.sample_sizes[i], benchmark.error_rates_relative_to_reused_answers[i]),
                        textcoords="offset points", xytext=(0,10), ha='center')

    plt.xlabel('Number of Samples Processed')
    plt.ylabel('Relative Error Rate (%)')
    
    plt.title(f'Relative Error Rate vs. Number of Samples (Relative To Reused Answers)')
    add_description(benchmark, plt)

    filename = benchmark.output_folder_path + f'/error_rate_relative_{benchmark.timestamp}.pdf'
    plt.savefig(filename, format='pdf')
    plt.close()

def plot_error_rate_absolute(benchmark: Benchmark, FONT_SIZE=20):
    plt.rcParams.update({'font.size': FONT_SIZE})
    plt.figure(figsize=(16, 7))
    plt.plot(benchmark.sample_sizes, benchmark.error_rates_absolute, color='blue')

    for i, _ in enumerate(benchmark.sample_sizes):
        plt.annotate("", (benchmark.sample_sizes[i], benchmark.error_rates_absolute[i]),
                        textcoords="offset points", xytext=(0,10), ha='center')

    plt.xlabel('Number of Samples Processed')
    plt.ylabel('Absolute Error Rate (%)')
    
    plt.title(f'Absolute Error Rate vs. Number of Samples')
    add_description(benchmark, plt)

    filename = filename = benchmark.output_folder_path + f'/error_rate_absolute_{benchmark.timestamp}.pdf'
    plt.savefig(filename, format='pdf')
    plt.close()

def plot_relative_error_rate_step_size_(benchmark: Benchmark, FONT_SIZE=20):
    plt.rcParams.update({'font.size': FONT_SIZE})
    plt.figure(figsize=(16, 7))
    plt.plot(benchmark.sample_sizes, benchmark.error_rates_relative_to_step_size, color='blue')

    for i, rate in enumerate(benchmark.error_rates_relative_to_step_size):
        plt.annotate("", (benchmark.sample_sizes[i], benchmark.error_rates_relative_to_step_size[i]),
                        textcoords="offset points", xytext=(0,10), ha='center')

    plt.xlabel('Number of Samples Processed')
    plt.ylabel(f'Relative Error Rate (step size: {benchmark.step_size})')
    
    plt.title(f'Relative Error Rate per Step Size vs. Number of Samples')
    add_description(benchmark, plt)

    filename = filename = benchmark.output_folder_path + f'/relative_error_rate_step_size_{benchmark.timestamp}.pdf'
    plt.savefig(filename, format='pdf')
    plt.close()

def plot_reuse_rate(benchmark: Benchmark, FONT_SIZE=20):
    plt.rcParams.update({'font.size': FONT_SIZE})
    reuse_rates = [
        (reused / size * 100) if size > 0 else 0 for reused, size in zip(benchmark.total_reused_list, benchmark.sample_sizes)
    ]

    plt.figure(figsize=(16, 7))
    plt.plot(benchmark.sample_sizes, reuse_rates, color='blue')

    for i, rate in enumerate(reuse_rates):
        plt.annotate("", (benchmark.sample_sizes[i], reuse_rates[i]),
                        textcoords="offset points", xytext=(0,10), ha='center')

    plt.xlabel('Number of Samples Processed')
    plt.ylabel('Cache Hit Rate (%)')
    
    plt.title(f'Cache Hit Rate vs. Number of Samples')
    add_description(benchmark, plt)

    filename = filename = benchmark.output_folder_path + f'/reuse_rate_{benchmark.timestamp}.pdf'
    plt.savefig(filename, format='pdf')
    plt.close()

def plot_relative_reuse_rate(benchmark: Benchmark, FONT_SIZE=20):
    plt.rcParams.update({'font.size': FONT_SIZE})
    plt.figure(figsize=(16, 7))
    plt.plot(benchmark.sample_sizes, benchmark.relative_reuse_rates, color='blue')

    for i, rate in enumerate(benchmark.relative_reuse_rates):
        plt.annotate("", (benchmark.sample_sizes[i], benchmark.relative_reuse_rates[i]),
                        textcoords="offset points", xytext=(0,10), ha='center')

    plt.xlabel('Number of Samples Processed')
    plt.ylabel(f'Relative Cache Hit Rate (to step_size: {benchmark.step_size})')

    plt.title(f'Relative Cache Hit Rate vs. Number of Samples')
    add_description(benchmark, plt)

    filename = filename = benchmark.output_folder_path + f'/relative_reuse_rate_step_size_{benchmark.timestamp}.pdf'
    plt.savefig(filename, format='pdf')
    plt.close()
    
def plot_duration_step_size(benchmark: Benchmark, FONT_SIZE=20):
    plt.rcParams.update({'font.size': FONT_SIZE})
    plt.figure(figsize=(16, 7))
    plt.plot(benchmark.sample_sizes, benchmark.inference_time_direct_step_size, label='Direct Inference Duration', color='blue')
    plt.plot(benchmark.sample_sizes, benchmark.inference_time_vectorq_step_size, label='VectorQ Inference Duration', color='orange')
    plt.xlabel('Number of Samples Processed')
    plt.ylabel('Inference Time (seconds)')
    plt.title('Inference Time per Step Size vs. Number of Samples')
    plt.legend()
    plt.grid(True)
    add_description(benchmark, plt)
    filename = filename = benchmark.output_folder_path + f'/relative_duration_step_size_{benchmark.timestamp}.pdf'
    plt.savefig(filename, format='pdf')
    plt.close()

def plot_duration_trend(benchmark: Benchmark, FONT_SIZE=20):
    plt.rcParams.update({'font.size': FONT_SIZE})
    plt.figure(figsize=(16, 7))
    # Convert seconds to minutes
    direct_duration_minutes = [d / 60 for d in benchmark.total_duration_direct_list]
    vectorq_duration_minutes = [d / 60 for d in benchmark.total_duration_vectorq_list]
    
    plt.plot(benchmark.sample_sizes, direct_duration_minutes, label='Total Duration Direct', color='blue')
    plt.plot(benchmark.sample_sizes, vectorq_duration_minutes, label='Total Duration VectorQ', color='orange')
    plt.xlabel('Number of Samples Processed')
    plt.ylabel('Total Duration (minutes)')
    plt.title('Total Inference Time vs. Number of Samples')
    plt.legend()
    add_description(benchmark, plt)
    filename = filename = benchmark.output_folder_path + f'/duration_trend_{benchmark.timestamp}.pdf'
    plt.savefig(filename, bbox_inches='tight')
    plt.close()

def plot_precision(benchmark: Benchmark, FONT_SIZE=20):
    plt.rcParams.update({'font.size': FONT_SIZE})
    plt.figure(figsize=(16, 7))
    plt.plot(benchmark.sample_sizes, benchmark.precision_list, color='blue')
    plt.xlabel('Number of Samples Processed')
    plt.ylabel('Precision')
    plt.title('Precision vs. Number of Samples')
    plt.ylim(0, 1)  # Set fixed y-axis range from 0 to 1
    add_description(benchmark, plt)
    filename = filename = benchmark.output_folder_path + f'/precision_{benchmark.timestamp}.pdf'
    plt.savefig(filename, bbox_inches='tight')
    plt.close()
    
def plot_recall(benchmark: Benchmark, FONT_SIZE=20):
    plt.rcParams.update({'font.size': FONT_SIZE})
    plt.figure(figsize=(16, 7))
    plt.plot(benchmark.sample_sizes, benchmark.recall_list, color='blue')
    plt.xlabel('Number of Samples Processed')
    plt.ylabel('Recall')
    plt.title('Recall vs. Number of Samples')
    plt.ylim(0, 1)  # Set fixed y-axis range from 0 to 1
    add_description(benchmark, plt)
    filename = filename = benchmark.output_folder_path + f'/recall_{benchmark.timestamp}.pdf'
    plt.savefig(filename, bbox_inches='tight')
    plt.close()
    
def plot_accuracy(benchmark: Benchmark, FONT_SIZE=20):
    plt.rcParams.update({'font.size': FONT_SIZE})
    plt.figure(figsize=(16, 7))
    plt.plot(benchmark.sample_sizes, benchmark.accuracy_list, color='blue')
    plt.xlabel('Number of Samples Processed')
    plt.ylabel('Accuracy')
    plt.title('Accuracy vs. Number of Samples')
    plt.ylim(0, 1)  # Set fixed y-axis range from 0 to 1
    add_description(benchmark, plt)   
    filename = filename = benchmark.output_folder_path + f'/accuracy_{benchmark.timestamp}.pdf'
    plt.savefig(filename, bbox_inches='tight')
    plt.close()

def plot_cache_size(benchmark: Benchmark, FONT_SIZE=20):
    """ Plot cache size growth as samples are processed. """
    plt.rcParams.update({'font.size': FONT_SIZE})
    plt.figure(figsize=(16, 7))
    plt.plot(benchmark.sample_sizes, benchmark.cache_size_list, color='blue')
    plt.xlabel('Number of Samples Processed')
    plt.ylabel('Cache Size (MB)')
    plt.title('Cache Size Growth vs. Number of Samples')
    add_description(benchmark, plt)
    filename = benchmark.output_folder_path + f'/cache_size_{benchmark.timestamp}.pdf'
    plt.savefig(filename, bbox_inches='tight')
    plt.close()

def add_description(benchmark: Benchmark, plt):
    if (benchmark.is_dynamic_threshold):
        description = (
            f'VectorQ, rnd_num_ub: {benchmark.rnd_num_ub}, Data Source: {os.path.basename(benchmark.filepath)}\n'
            f'Embedding Model: {benchmark.embedding_model}, LLM Model: {benchmark.llm_model}'
        )
    else:
        embedding_model_name = benchmark.embedding_model[1] if isinstance(benchmark.embedding_model, tuple) else benchmark.embedding_model
        llm_model_name = benchmark.llm_model[1] if isinstance(benchmark.llm_model, tuple) else benchmark.llm_model
        
        description = (
            f'State of the Art, Threshold: {benchmark.threshold}, Data Source: {os.path.basename(benchmark.filepath)}\n'
            f'Embedding Model: {embedding_model_name}, LLM: {llm_model_name}'
        )
    plt.subplots_adjust(bottom=0.2)
    plt.figtext(0.5, 0.02, description, wrap=True, horizontalalignment='center', fontsize=16)
    
def plot_cache_hit_latency_vs_size(benchmark: Benchmark, FONT_SIZE=20):
    cache_sizes = []
    hit_latencies = []
    cache_hits_count = []  # Track the cumulative number of cache hits
    hits_count = 0
    
    for i, reused in enumerate(benchmark.answers_reused):
        if reused and i > 0:  # It's a cache hit and not the first element
            cache_sizes.append(benchmark.cache_size_list[i])
            hit_latencies.append(benchmark.inference_time_vectorq_step_size[i])
            hits_count += 1
            cache_hits_count.append(hits_count)
    
    if len(cache_sizes) < 2:
        print("Not enough cache hits to plot cache hit latency vs cache size")
        return
    
    # Remove outliers using IQR method
    latencies_array = np.array(hit_latencies)
    q1 = np.percentile(latencies_array, 25)
    q3 = np.percentile(latencies_array, 75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    
    # Filter out outliers
    outlier_mask = (latencies_array >= lower_bound) & (latencies_array <= upper_bound)
    filtered_cache_sizes = np.array(cache_sizes)[outlier_mask]
    filtered_hit_latencies = latencies_array[outlier_mask]
    filtered_cache_hits_count = np.array(cache_hits_count)[outlier_mask]
    
    # Print how many outliers were removed
    num_outliers = len(hit_latencies) - len(filtered_hit_latencies)
    if num_outliers > 0:
        print(f"Removed {num_outliers} outliers from hit latencies (out of {len(hit_latencies)})")
        
    plt.rcParams.update({'font.size': FONT_SIZE})
    plt.figure(figsize=(16, 7))
    
    # First y-axis (latency)
    ax1 = plt.gca()
    scatter = ax1.scatter(filtered_cache_sizes, filtered_hit_latencies, alpha=0.4, color='green', label='Cache Hit Latency (sec)')
    
    if len(filtered_cache_sizes) > 5:
        try:
            # Fit a linear regression
            z = np.polyfit(filtered_cache_sizes, filtered_hit_latencies, 1)
            p = np.poly1d(z)
            
            # Plot the trend line
            x_range = np.linspace(min(filtered_cache_sizes), max(filtered_cache_sizes), 100)
            ax1.plot(x_range, p(x_range), "r--", alpha=1.0, label=f"Trend: {z[0]:.6f}x + {z[1]:.4f}", linewidth=4)
        except:
            # If fitting fails, just continue without the trend line
            pass
    
    ax1.set_xlabel("Cache Size (MB)")
    ax1.set_ylabel("Cache Hit Latency (sec)")
    ax1.tick_params(axis='y')
    
    # Second y-axis (number of cache items)
    ax2 = ax1.twinx()
    ax2.plot(filtered_cache_sizes, filtered_cache_hits_count, 'b-', alpha=1.0, label='Number of Cache Hits', linewidth=4)
    ax2.set_ylabel('Number of Cache Hits')
    ax2.tick_params(axis='y')
    
    plt.grid(True, linestyle='--', alpha=0.6)
    
    # Combine legends from both axes
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    
    # Adjust figure to make room for the legend at the top
    # Increased top margin to avoid overlap with title
    plt.subplots_adjust(top=0.82)
    
    # Add legend with custom font size - positioned above the plot
    ax1.legend(lines1 + lines2, labels1 + labels2, 
              loc='upper center', 
              bbox_to_anchor=(0.5, 1.16), # Lowered position to avoid overlap with title
              ncol=3,             # Force all items to be in one row
              frameon=True, 
              fancybox=True, 
              fontsize=FONT_SIZE)
    
    #add_description(benchmark, plt)
    
    filename = benchmark.output_folder_path + f'/cache_hit_latency_vs_size_{benchmark.timestamp}.pdf'
    plt.savefig(filename, format='pdf', bbox_inches='tight')
    plt.close()
    
# TODO: LGS
def plot_combined_thresholds_and_posteriors(benchmark: Benchmark):
    
    for idx, correct_similarities, incorrect_similarities, posteriors in zip(benchmark.correct_x.keys(), benchmark.correct_x.values(), benchmark.incorrect_x.values(), benchmark.posteriors.values()):
        
        if 0.0 in incorrect_similarities:
            incorrect_similarities.remove(0.0)
        if 1.0 in correct_similarities:
            correct_similarities.remove(1.0)
        
        if (len(correct_similarities) < 2) or (len(incorrect_similarities) < 2):
            continue

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12), sharex=True, gridspec_kw={'height_ratios': [2, 1]})
        
        if correct_similarities:
            sns.kdeplot(correct_similarities, color='green', fill=True, alpha=0.2, bw_adjust=0.25, ax=ax1)
        if incorrect_similarities:
            sns.kdeplot(incorrect_similarities, color='red', fill=True, alpha=0.2, bw_adjust=0.25, ax=ax1)
        
        ax1.set_title(f"Embedding {idx}, {len(correct_similarities)} correct, {len(incorrect_similarities)} incorrect")
        ax1.set_ylabel("Density")
        ax1.grid(True, linestyle='--', alpha=0.6)
        ax1.legend(handles=[
            Patch(color='green', label='Correctness KDF'),
            Patch(color='red', label='Incorrectness KDF'),
        ])
        
        #x_values = embedding_posterior.get("x_values", [])
        #posterior = embedding_posterior.get("posterior", [])
        
        # if x_values and posterior:
        #     max_value = max(posterior)
        #     if max_value > 0:
        #         posterior = [p / max_value for p in posterior]
            
        #     ax2.plot(x_values, posterior, color='blue', linewidth=2)
        #     ax2.fill_between(x_values, 0, posterior, color='blue', alpha=0.2)
        #     ax2.set_xlabel("Similarity Value")
        #     ax2.set_ylabel("Incorrect Cache Hit Likelihood")
        #     ax2.grid(True, linestyle='--', alpha=0.6)
        
        for ax in [ax1, ax2]:
            ax.set_xlim(0, 1)
        
        plt.tight_layout()
        
        output_folder_path = benchmark.output_folder_path + f'/thresholds_and_posteriors/'
        filename = benchmark.output_folder_path + f'/thresholds_and_posteriors/combined_embedding_{idx}_{benchmark.timestamp}.pdf'
        if output_folder_path and not os.path.exists(output_folder_path):
            os.makedirs(output_folder_path)
        plt.savefig(filename, format='pdf', bbox_inches='tight')
        plt.close()
        