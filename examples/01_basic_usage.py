#!/usr/bin/env python3
"""
Basic vCache Usage Example

This example demonstrates the simplest way to use vCache with default configuration.
vCache will automatically cache LLM responses and return cached results for 
semantically similar prompts.

Requirements:
- Set OPENAI_API_KEY environment variable
- Install vCache: pip install -e .

Usage:
    python examples/01_basic_usage.py
"""

import os
from vcache.main import VCache


def main():
    # Check if OpenAI API key is set
    if not os.getenv("OPENAI_API_KEY"):
        print("Please set OPENAI_API_KEY environment variable")
        return

    # Create vCache instance with default configuration
    # This uses:
    # - OpenAI for inference and embeddings
    # - HNSWLib vector database
    # - Dynamic local threshold policy (2% max error rate)
    # - In-memory storage
    vcache = VCache()

    print("=== Basic vCache Usage Example ===\n")

    # First query - will be a cache miss (no cached responses yet)
    print("1. First query (cache miss expected):")
    prompt1 = "What is the capital of France?"
    cache_hit, response, _ = vcache.infer_with_cache_info(prompt1)
    print(f"   Prompt: {prompt1}")
    print(f"   Cache Hit: {cache_hit}")
    print(f"   Response: {response}\n")

    # Second query - semantically similar, should be a cache hit
    print("2. Semantically similar query (cache hit expected):")
    prompt2 = "What city is the capital of France?"
    cache_hit, response, _ = vcache.infer_with_cache_info(prompt2)
    print(f"   Prompt: {prompt2}")
    print(f"   Cache Hit: {cache_hit}")
    print(f"   Response: {response}\n")

    # Third query - different topic, will be a cache miss
    print("3. Different topic query (cache miss expected):")
    prompt3 = "What is the largest planet in our solar system?"
    cache_hit, response, _ = vcache.infer_with_cache_info(prompt3)
    print(f"   Prompt: {prompt3}")
    print(f"   Cache Hit: {cache_hit}")
    print(f"   Response: {response}\n")

    # Fourth query - similar to third, should be a cache hit
    print("4. Similar to previous query (cache hit expected):")
    prompt4 = "Which planet is the biggest in the solar system?"
    cache_hit, response, _ = vcache.infer_with_cache_info(prompt4)
    print(f"   Prompt: {prompt4}")
    print(f"   Cache Hit: {cache_hit}")
    print(f"   Response: {response}\n")

    print("=== Example Complete ===")
    print("Notice how semantically similar prompts return cached responses,")
    print("reducing latency and API costs!")


if __name__ == "__main__":
    main()