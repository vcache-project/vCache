from dotenv import load_dotenv

from vcache import (
    HNSWLibVectorDB,
    InMemoryEmbeddingMetadataStorage,
    LangChainEmbeddingEngine,
    LLMComparisonSimilarityEvaluator,
    OpenAIInferenceEngine,
    VCache,
    VCacheConfig,
    VerifiedDecisionPolicy,
)

# Load environment variables from .env file (for OPENAI_API_KEY)
load_dotenv()


def main():
    """
    Example demonstrating how to set up and use vCache.
    """
    print("Initializing vCache configuration...")

    # 1. Configure the components for vCache
    config = VCacheConfig(
        # The inference engine generates new responses when no suitable
        # cached response is found.
        inference_engine=OpenAIInferenceEngine(model_name="gpt-3.5-turbo"),
        # The embedding engine creates vector representations of prompts.
        embedding_engine=LangChainEmbeddingEngine(
            model_name="sentence-transformers/all-mpnet-base-v2"
        ),
        # The vector database stores and retrieves prompt embeddings.
        vector_db=HNSWLibVectorDB(),
        # The metadata storage holds the actual responses and other data.
        embedding_metadata_storage=InMemoryEmbeddingMetadataStorage(),
        # The similarity evaluator determines if a cached response is
        # semantically similar enough to a new prompt.
        similarity_evaluator=LLMComparisonSimilarityEvaluator(),
    )

    # 2. Choose a caching policy
    # The VerifiedDecisionPolicy uses a statistical model to decide
    # whether to serve a cached response (exploit) or get a new one (explore).
    policy = VerifiedDecisionPolicy(delta=0.1)

    # 3. Initialize vCache with the configuration and policy
    vcache = VCache(config, policy)
    print("vCache initialized successfully.\n")

    # --- Demonstrate vCache Usage ---

    # First request: This will be a cache miss, and a new response is generated.
    prompt1 = "What is the capital of France?"
    print(f"User Prompt 1: {prompt1}")
    response1 = vcache.infer(prompt=prompt1)
    print(f"vCache Response 1: {response1}\n")

    # Second request: This prompt is semantically similar to the first one.
    # The policy will likely decide to serve the cached response (cache hit).
    prompt2 = "Which city is the capital of France?"
    print(f"User Prompt 2: {prompt2}")
    response2 = vcache.infer(prompt=prompt2)
    print(f"vCache Response 2: {response2}\n")

    # Third request: This is a different question, resulting in another cache miss.
    prompt3 = "What is the tallest mountain in the world?"
    print(f"User Prompt 3: {prompt3}")
    response3 = vcache.infer(prompt=prompt3)
    print(f"vCache Response 3: {response3}\n")

    # You can also get detailed information about the cache interaction
    is_hit, response, _, nn_metadata = vcache.infer_with_cache_info(prompt=prompt2)
    print(f"Cache hit for prompt 2: {is_hit}")
    if is_hit:
        print(f"Served from cache. Nearest neighbor prompt was: '{nn_metadata.prompt}'")


if __name__ == "__main__":
    main()
