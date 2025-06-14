#!/usr/bin/env python3
"""
Custom Caching Policy Example

This example demonstrates how to use different caching policies in vCache.
We compare the default dynamic local threshold policy with a static global threshold policy.

Requirements:
- Set OPENAI_API_KEY environment variable
- Install vCache: pip install -e .

Usage:
    python examples/02_custom_policy.py
"""

import os

from vcache.main import VCache
from vcache.vcache_policy.strategies.dynamic_local_threshold import (
    DynamicLocalThresholdPolicy,
)
from vcache.vcache_policy.strategies.static_global_threshold import (
    StaticGlobalThresholdPolicy,
)


def test_policy(vcache, policy_name, prompts):
    """Test a caching policy with a set of prompts."""
    print(f"\n=== Testing {policy_name} ===")

    for i, prompt in enumerate(prompts, 1):
        cache_hit, response, _ = vcache.infer_with_cache_info(prompt)
        print(f"{i}. Prompt: {prompt}")
        print(f"   Cache Hit: {cache_hit}")
        print(f"   Response: {response[:100]}{'...' if len(response) > 100 else ''}\n")


def main():
    # Check if OpenAI API key is set
    if not os.getenv("OPENAI_API_KEY"):
        print("Please set OPENAI_API_KEY environment variable")
        return

    print("=== Custom Caching Policy Example ===")

    # Test prompts - some similar, some different
    test_prompts = [
        "What is machine learning?",
        "Can you explain machine learning?",  # Similar to first
        "What is artificial intelligence?",  # Related but different
        "Explain AI to me",  # Similar to third
        "What is the weather like today?",  # Completely different
    ]

    # 1. Dynamic Local Threshold Policy (default)
    # This policy learns individual thresholds for each cached prompt
    # and adapts based on observed similarity-correctness patterns
    print("\n1. Dynamic Local Threshold Policy (Default)")
    print("   - Learns individual thresholds for each cached prompt")
    print("   - Adapts based on similarity-correctness patterns")
    print("   - Maximum error rate: 2%")

    dynamic_policy = DynamicLocalThresholdPolicy(delta=0.02)  # 2% max error rate
    vcache_dynamic = VCache(policy=dynamic_policy)
    test_policy(vcache_dynamic, "Dynamic Local Threshold", test_prompts)

    # 2. Static Global Threshold Policy
    # This policy uses a fixed threshold for all prompts
    # Higher threshold = more conservative (fewer cache hits, lower error rate)
    # Lower threshold = more aggressive (more cache hits, higher error rate)
    print("\n2. Static Global Threshold Policy")
    print("   - Uses fixed threshold (0.85) for all prompts")
    print("   - Higher threshold = more conservative caching")
    print("   - Lower threshold = more aggressive caching")

    static_policy = StaticGlobalThresholdPolicy(
        threshold=0.85
    )  # 85% similarity required
    vcache_static = VCache(policy=static_policy)
    test_policy(vcache_static, "Static Global Threshold (0.85)", test_prompts)

    # 3. More aggressive static threshold
    print("\n3. More Aggressive Static Global Threshold Policy")
    print("   - Uses lower threshold (0.70) for more cache hits")
    print("   - May result in more false positives")

    aggressive_policy = StaticGlobalThresholdPolicy(
        threshold=0.70
    )  # 70% similarity required
    vcache_aggressive = VCache(policy=aggressive_policy)
    test_policy(vcache_aggressive, "Static Global Threshold (0.70)", test_prompts)

    print("=== Policy Comparison Complete ===")
    print("Key Differences:")
    print("- Dynamic Local: Learns and adapts per-prompt thresholds")
    print("- Static Global: Uses same threshold for all prompts")
    print("- Lower thresholds: More cache hits, potentially more errors")
    print("- Higher thresholds: Fewer cache hits, fewer errors")


if __name__ == "__main__":
    main()
