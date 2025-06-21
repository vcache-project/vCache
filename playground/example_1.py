from pathlib import Path

import pandas as pd
from tqdm import tqdm

from vcache import (
    HNSWLibVectorDB,
    InMemoryEmbeddingMetadataStorage,
    LLMComparisonSimilarityEvaluator,
    OpenAIEmbeddingEngine,
    OpenAIInferenceEngine,
    VCache,
    VCacheConfig,
    VCachePolicy,
    VerifiedDecisionPolicy,
)

"""
    Run 
       export OPENAI_API_KEY="<your-api-key>"
    before running the script with
       poetry run python example_1.py
"""


def __get_vcache() -> VCache:
    print("Initializing vCache configuration...")

    # 1. Configure the components for vCache
    config: VCacheConfig = VCacheConfig(
        inference_engine=OpenAIInferenceEngine(model_name="gpt-4.1-2025-04-14"),
        embedding_engine=OpenAIEmbeddingEngine(model_name="text-embedding-3-small"),
        vector_db=HNSWLibVectorDB(),
        embedding_metadata_storage=InMemoryEmbeddingMetadataStorage(),
        similarity_evaluator=LLMComparisonSimilarityEvaluator(
            inference_engine=OpenAIInferenceEngine(model_name="gpt-4.1-nano-2025-04-14")
        ),
    )

    # 2. Choose a caching policy
    policy: VCachePolicy = VerifiedDecisionPolicy(delta=0.03)

    # 3. Initialize vCache with the configuration and policy
    vcache: VCache = VCache(config, policy)
    print("vCache initialized successfully.\n")

    return vcache


def main():
    vcache: VCache = __get_vcache()

    print("Loading sample data from parquet file...")
    script_dir: Path = Path(__file__).parent
    df: pd.DataFrame = pd.read_parquet(script_dir / "sample_data.parquet")
    print(f"Loaded {len(df)} rows of data\n")
    print("Processing data with vCache...")

    for i, row in tqdm(df.iterrows(), total=len(df), desc="Processing rows"):
        prompt: str = row["text"]
        system_prompt: str = row["task"]

        response: str = vcache.infer(prompt, system_prompt)
        print(f"Response for request {i}: {response}")

    print("Data processing completed.")


if __name__ == "__main__":
    main()
