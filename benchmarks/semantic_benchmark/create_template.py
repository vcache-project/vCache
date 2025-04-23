#!/usr/bin/env python3
"""
Script to create a template dataset for the LLM and Embedding Comparison Benchmark.
This generates a JSON file with the basic structure but with null values for fields
that will be filled by the benchmark.
"""

import json
import argparse
import os
from typing import List, Dict, Any

def create_template_dataset(prompts: List[str], output_file: str,
                           embedding_model_a: str = "text-embedding-3-large", 
                           embedding_model_b: str = "text-embedding-3-small",
                           llm_a: str = "4o-mini",
                           llm_b: str = "4o"):
    """
    Create a template dataset for the LLM and Embedding Comparison Benchmark.
    
    Args:
        prompts: List of prompt strings
        output_file: Path to save the template dataset
        embedding_model_a: Name of the first embedding model
        embedding_model_b: Name of the second embedding model
        llm_a: Name of the first LLM
        llm_b: Name of the second LLM
    """
    dataset = []
    
    for i, prompt in enumerate(prompts):
        entry = {
            "ID": i + 1,
            "ID_Set": 1,  # Default set ID
            "Prompt": prompt,
            "Prompt_Embedding_A": None,
            "Prompt_Embedding_B": None,
            "Prompt_Embedding_A_Cost": None,
            "Prompt_Embedding_B_Cost": None,
            "Embedding_Model_A": embedding_model_a,
            "Embedding_Model_B": embedding_model_b,
            "Latency_Embedding_Model_A": None,
            "Latency_Embedding_Model_B": None,
            "Answer_LLM_A": None,
            "Answer_LLM_B": None,
            "Answer_LLM_A_Cost": None,
            "Answer_LLM_B_Cost": None,
            "LLM_A": llm_a,
            "LLM_B": llm_b,
            "Latency_LLM_A": None,
            "Latency_LLM_B": None
        }
        dataset.append(entry)
    
    # Create directories if they don't exist
    os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
    
    # Write dataset to file
    with open(output_file, 'w') as f:
        json.dump(dataset, f, indent=2)
    
    print(f"Template dataset with {len(prompts)} samples created at {output_file}")

def load_prompts_from_file(file_path: str) -> List[str]:
    """Load prompts from a text file, one prompt per line."""
    with open(file_path, 'r') as f:
        prompts = [line.strip() for line in f if line.strip()]
    return prompts

def parse_args():
    parser = argparse.ArgumentParser(description="Create a template dataset for the LLM and Embedding Comparison Benchmark")
    
    # Input arguments
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument('--prompts', type=str, nargs='+', help='List of prompts')
    input_group.add_argument('--prompts-file', type=str, help='Path to a file containing prompts (one per line)')
    
    # Output argument
    parser.add_argument('--output', type=str, required=True, help='Path to save the template dataset')
    
    # Model arguments
    parser.add_argument('--embedding-model-a', type=str, default="text-embedding-3-large", 
                        help='Name of the first embedding model')
    parser.add_argument('--embedding-model-b', type=str, default="text-embedding-3-small", 
                        help='Name of the second embedding model')
    parser.add_argument('--llm-a', type=str, default="4o-mini", 
                        help='Name of the first LLM')
    parser.add_argument('--llm-b', type=str, default="4o", 
                        help='Name of the second LLM')
    
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Get prompts
    if args.prompts:
        prompts = args.prompts
    else:
        prompts = load_prompts_from_file(args.prompts_file)
    
    # Create template dataset
    create_template_dataset(
        prompts=prompts,
        output_file=args.output,
        embedding_model_a=args.embedding_model_a,
        embedding_model_b=args.embedding_model_b,
        llm_a=args.llm_a,
        llm_b=args.llm_b
    )

if __name__ == '__main__':
    main() 