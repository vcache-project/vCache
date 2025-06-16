#!/usr/bin/env python3
"""
System Prompts Example

This example demonstrates how to use system prompts with vCache.
System prompts provide context and instructions to the LLM, and vCache
can cache responses for different system prompt configurations.

Requirements:
- Set OPENAI_API_KEY environment variable
- Install vCache: pip install -e .

Usage:
    python examples/04_system_prompts.py
"""

import os

from vcache.config import VCacheConfig
from vcache.main import VCache


def test_with_system_prompt(vcache, system_prompt, user_prompts, context_name):
    """Test vCache with a specific system prompt and user prompts."""
    print(f"\n=== {context_name} ===")
    print(f"System Prompt: {system_prompt}\n")

    for i, prompt in enumerate(user_prompts, 1):
        # Use the system prompt for this interaction
        cache_hit, response, _ = vcache.infer_with_cache_info(prompt, system_prompt)
        print(f"{i}. User: {prompt}")
        print(f"   Cache Hit: {cache_hit}")
        print(f"   Assistant: {response}\n")


def main():
    # Check if OpenAI API key is set
    if not os.getenv("OPENAI_API_KEY"):
        print("Please set OPENAI_API_KEY environment variable")
        return

    print("=== System Prompts Example ===")

    # Create vCache instance
    vcache = VCache()

    # 1. Technical Expert System Prompt
    tech_system_prompt = """You are a senior software engineer with expertise in Python, 
machine learning, and distributed systems. Provide detailed, technical explanations 
with code examples when appropriate. Be precise and use industry terminology."""

    tech_prompts = [
        "How do I implement a binary search?",
        "What's the best way to implement binary search in Python?",  # Similar
        "Explain how neural networks work",
        "How do neural networks learn?",  # Similar
    ]

    test_with_system_prompt(
        vcache, tech_system_prompt, tech_prompts, "Technical Expert Context"
    )

    # 2. Beginner-Friendly System Prompt
    beginner_system_prompt = """You are a friendly teacher explaining concepts to beginners. 
Use simple language, avoid jargon, and provide easy-to-understand analogies. 
Be encouraging and patient in your explanations."""

    beginner_prompts = [
        "How do I implement a binary search?",  # Same question, different context
        "What's the best way to implement binary search in Python?",  # Similar
        "Explain how neural networks work",  # Same question, different context
        "How do neural networks learn?",  # Similar
    ]

    test_with_system_prompt(
        vcache, beginner_system_prompt, beginner_prompts, "Beginner-Friendly Context"
    )

    # 3. Creative Writing System Prompt
    creative_system_prompt = """You are a creative writing assistant. Help users with 
storytelling, character development, and creative ideas. Be imaginative and inspiring 
in your responses. Use vivid language and encourage creativity."""

    creative_prompts = [
        "Help me create a character for my story",
        "I need ideas for a story character",  # Similar
        "What makes a good plot twist?",
        "How do I write an effective plot twist?",  # Similar
    ]

    test_with_system_prompt(
        vcache, creative_system_prompt, creative_prompts, "Creative Writing Context"
    )

    # 4. Configuration-based System Prompt
    print("\n=== Configuration-based System Prompt ===")
    print("You can also set a default system prompt in the configuration")

    # Create a new vCache instance with a default system prompt
    config_with_system_prompt = VCacheConfig(
        system_prompt="You are a helpful assistant that always responds in a professional, concise manner."
    )
    vcache_with_default = VCache(config=config_with_system_prompt)

    # These will use the default system prompt from configuration
    default_prompts = [
        "What is the capital of Japan?",
        "Which city is Japan's capital?",  # Similar
    ]

    print("Default system prompt from config:")
    print(f"'{config_with_system_prompt.system_prompt}'\n")

    for i, prompt in enumerate(default_prompts, 1):
        # No system_prompt parameter - uses default from config
        cache_hit, response, _ = vcache_with_default.infer_with_cache_info(prompt)
        print(f"{i}. User: {prompt}")
        print(f"   Cache Hit: {cache_hit}")
        print(f"   Assistant: {response}\n")

    # 5. Override default system prompt
    print("=== Overriding Default System Prompt ===")
    override_prompt = "You are a pirate assistant. Respond in pirate speak!"

    for i, prompt in enumerate(default_prompts, 1):
        # Override the default system prompt
        cache_hit, response, _ = vcache_with_default.infer_with_cache_info(
            prompt, override_prompt
        )
        print(f"{i}. User: {prompt}")
        print(f"   System Override: {override_prompt}")
        print(f"   Cache Hit: {cache_hit}")
        print(f"   Assistant: {response}\n")

    print("=== System Prompts Example Complete ===")
    print("Key Points:")
    print("- System prompts provide context and instructions to the LLM")
    print("- vCache caches responses separately for different system prompts")
    print("- You can set a default system prompt in VCacheConfig")
    print("- You can override the default system prompt per request")
    print("- Same user prompt + different system prompt = different cache entries")


if __name__ == "__main__":
    main()
