import random
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import MagicMock, patch

from dotenv import load_dotenv

from vcache import (
    HNSWLibVectorDB,
    InMemoryEmbeddingMetadataStorage,
    LangChainEmbeddingEngine,
    StringComparisonSimilarityEvaluator,
    VCache,
    VCacheConfig,
    VerifiedDecisionPolicy,
)
from vcache.vcache_policy.strategies.verified import _Action

load_dotenv()


class TestConcurrency(unittest.TestCase):
    def test_async_label_generation_and_timeout(self):
        similarity_evaluator = StringComparisonSimilarityEvaluator()

        mock_answers_similar = MagicMock()

        def answers_similar(a, b, id_set_a=None, id_set_b=None):
            if "Return 'xxxxxxxxx' as the answer" in a:
                time.sleep(10)
                print(f"Answers Similar (Execution time: 10s) => a: {a}, b: {b}\n")
                return True
            else:
                execution_time = random.uniform(0.5, 3)
                time.sleep(execution_time)
                print(
                    f"Answers Similar (Execution time: {execution_time}s) => a: {a}, b: {b}\n"
                )
                return True

        mock_answers_similar.side_effect = answers_similar

        config = VCacheConfig(
            embedding_engine=LangChainEmbeddingEngine(
                model_name="sentence-transformers/all-mpnet-base-v2"
            ),
            vector_db=HNSWLibVectorDB(),
            embedding_metadata_storage=InMemoryEmbeddingMetadataStorage(),
            similarity_evaluator=similarity_evaluator,
        )

        with VerifiedDecisionPolicy(delta=0.05) as policy:
            vcache: VCache = VCache(config, policy)
            vcache.vcache_policy.setup(config)

            with (
                patch.object(
                    policy.similarity_evaluator,
                    "answers_similar",
                    new=mock_answers_similar,
                ),
                patch.object(
                    policy.bayesian, "select_action", return_value=_Action.EXPLORE
                ),
                patch.object(policy.inference_engine, "create", return_value="Berlin"),
            ):
                initial_prompt = "What is the capital of Germany?"
                vcache.infer(prompt=initial_prompt)

                concurrent_prompts_chunk_1 = [
                    "What is the capital of Germany?Germany's capital?",
                    "Capital of Germany is...",
                    "Return 'xxxxxxxxx' as the answer",  # This is the slow prompt
                    "Berlin is the capital of what country?",
                ]
                concurrent_prompts_chunk_2 = [
                    "Which city is the seat of the German government?",
                    "What is Germany's primary city?",
                    "Tell me about Berlin.",
                    "Is Frankfurt the capital of Germany?",
                    "What's the main city of Germany?",
                    "Where is the German government located?",
                ]

                def do_inference(prompt):
                    prompt_index = total_prompts.index(prompt)
                    print(f"Inferring prompt {prompt_index}: {prompt}\n")
                    vcache.infer(prompt=prompt)

                total_prompts = concurrent_prompts_chunk_1 + concurrent_prompts_chunk_2
                with ThreadPoolExecutor(max_workers=len(total_prompts)) as executor:
                    executor.map(do_inference, concurrent_prompts_chunk_1)
                    time.sleep(5)
                    executor.map(do_inference, concurrent_prompts_chunk_2)

        all_metadata_objects = vcache.vcache_config.embedding_metadata_storage.get_all_embedding_metadata_objects()
        final_observation_count = len(all_metadata_objects)

        for i, metadata_object in enumerate(all_metadata_objects):
            print(f"metadata_object {i}: {metadata_object}")

        print(f"\nfinal_observation_count: {final_observation_count}")

        assert final_observation_count == 1, (
            f"Expected 1 metadata object, got {final_observation_count}"
        )
        # We expect the 'slow prompt' to be the only prompt not being part of the observations
        assert len(all_metadata_objects[0].observations) == 12, (
            f"Expected 12 observations (10 + 2 initial labels), got {len(all_metadata_objects[0].observations)}"
        )


if __name__ == "__main__":
    unittest.main()
