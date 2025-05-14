#!/usr/bin/env python

import ijson
import matplotlib.pyplot as plt
from collections import defaultdict
import numpy as np

def analyze_benchmark_arena():
    json_file = "/data/acuadron/VectorQ/benchmarks/data/large_scale/benchmark_arena.json"
    
    # Dictionary to store all entries with the same ID_Set
    id_set_entries = defaultdict(list)
    
    # Lists to store response lengths for each model
    nano_lengths = []
    mini_lengths = []
    
    # Counter for entries processed
    entry_count = 0
    max_entries = 60000
    
    print(f"Processing first {max_entries} entries from benchmark arena dataset...")
    
    # Process the JSON file
    with open(json_file, 'rb') as f:
        # Use ijson to stream the JSON data
        for item in ijson.items(f, 'item'):
            # Extract the ID_Set
            id_set = item.get('ID_Set')
            if id_set:
                id_set_entries[id_set].append(item['id'])
            
            # Calculate response lengths for both models
            if 'response_gpt-4.1-nano' in item:
                response = item['response_gpt-4.1-nano']
                if isinstance(response, str):
                    nano_lengths.append(len(response.split()))
            
            if 'response_gpt-4o-mini' in item:
                response = item['response_gpt-4o-mini']
                if isinstance(response, str):
                    mini_lengths.append(len(response.split()))
            
            entry_count += 1
            if entry_count % 10000 == 0:
                print(f"Processed {entry_count} entries...")
            
            if entry_count >= max_entries:
                break
    
    # Count unique ID_Sets
    unique_id_sets = len(id_set_entries)
    print(f"\nNumber of unique set IDs: {unique_id_sets}")
    
    # Calculate cardinality (number of entries per ID_Set)
    cardinality_counts = defaultdict(int)
    for id_set, entries in id_set_entries.items():
        cardinality = len(entries)
        cardinality_counts[cardinality] += 1
    
    # Print histogram data
    print("\nCardinality histogram (set_id count per number of entries):")
    for cardinality, count in sorted(cardinality_counts.items()):
        print(f"{count} set IDs with {cardinality} entries")
    
    # Create a histogram for set ID cardinalities
    plt.figure(figsize=(12, 6))
    cardinalities = list(cardinality_counts.keys())
    counts = list(cardinality_counts.values())
    plt.bar(cardinalities, counts)
    plt.xlabel('Number of Entries per Set ID')
    plt.ylabel('Count of Set IDs')
    plt.title('Histogram of Set ID Cardinalities')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.savefig('set_id_cardinality_histogram.png')
    print("\nHistogram saved as 'set_id_cardinality_histogram.png'")
    
    # Response length statistics
    print("\nResponse length statistics (word count):")
    
    print("\nGPT-4.1-nano responses:")
    if nano_lengths:
        print(f"  Count: {len(nano_lengths)}")
        print(f"  Mean: {np.mean(nano_lengths):.2f}")
        print(f"  Median: {np.median(nano_lengths):.2f}")
        print(f"  Min: {min(nano_lengths)}")
        print(f"  Max: {max(nano_lengths)}")
        print(f"  Standard deviation: {np.std(nano_lengths):.2f}")
    else:
        print("  No responses found")
        
    print("\nGPT-4o-mini responses:")
    if mini_lengths:
        print(f"  Count: {len(mini_lengths)}")
        print(f"  Mean: {np.mean(mini_lengths):.2f}")
        print(f"  Median: {np.median(mini_lengths):.2f}")
        print(f"  Min: {min(mini_lengths)}")
        print(f"  Max: {max(mini_lengths)}")
        print(f"  Standard deviation: {np.std(mini_lengths):.2f}")
    else:
        print("  No responses found")
    
    # Create histograms for response lengths
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # GPT-4.1-nano histogram
    if nano_lengths:
        bins = np.linspace(0, max(nano_lengths), 50)
        ax1.hist(nano_lengths, bins=bins, alpha=0.7, color='blue')
        ax1.set_xlabel('Response Length (words)')
        ax1.set_ylabel('Count')
        ax1.set_title('GPT-4.1-nano Response Length Distribution')
        ax1.grid(linestyle='--', alpha=0.7)
    else:
        ax1.text(0.5, 0.5, 'No data available', ha='center', va='center')
    
    # GPT-4o-mini histogram
    if mini_lengths:
        bins = np.linspace(0, max(mini_lengths), 50)
        ax2.hist(mini_lengths, bins=bins, alpha=0.7, color='green')
        ax2.set_xlabel('Response Length (words)')
        ax2.set_ylabel('Count')
        ax2.set_title('GPT-4o-mini Response Length Distribution')
        ax2.grid(linestyle='--', alpha=0.7)
    else:
        ax2.text(0.5, 0.5, 'No data available', ha='center', va='center')
    
    plt.tight_layout()
    plt.savefig('response_length_distributions.png')
    print("\nResponse length distributions saved as 'response_length_distributions.png'")

if __name__ == "__main__":
    analyze_benchmark_arena()
