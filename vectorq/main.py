from typing import List, Optional, Tuple

from vectorq.config import VectorQConfig
from vectorq.inference_engine.inference_engine import InferenceEngine
from vectorq.vectorq_core.core import VectorQCore


class VectorQBenchmark:
    def __init__(
        self,
        candidate_embedding: List[float],
        candidate_response: str,
    ):
        self.candidate_embedding = candidate_embedding
        self.candidate_response = candidate_response


class VectorQ:
    """
    VectorQ is a main class that contains the VectorQ semantic prompt caching system.
    """

    def __init__(self, vectorq_config=VectorQConfig()):
        self.vectorq_config = vectorq_config

        try:
            self.inference_engine = self.vectorq_config.inference_engine
            self.core = VectorQCore(vectorq_config=vectorq_config)
        except Exception as e:
            print(f"Error initializing VectorQ: {e}")
            raise Exception(f"Error initializing VectorQ: {e}")

    def create(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        benchmark: Optional[VectorQBenchmark] = None,
    ) -> Tuple[bool, str, str]:
        """
        prompt: str - The prompt to create a response for.
        system_prompt: str - The optional system prompt to use for the response. It will override the system prompt in the VectorQConfig if provided.
        benchmark: VectorQBenchmark - The optional benchmark object containing the pre-computed embedding and response.
        Returns: Tuple[bool, str, str] - [is_cache_hit, actual_response, nn_response] (the actual response is the one supposed to be used by the user, the nn_response is for benchmarking purposes)
        """
        if system_prompt is None:
            system_prompt = self.vectorq_config.system_prompt
        if self.vectorq_config.enable_cache:
            is_cache_hit, actual_response, nn_response = self.core.process_request(
                prompt, benchmark, system_prompt
            )
            return is_cache_hit, actual_response, nn_response
        else:
            response = self.inference_engine.create(prompt, system_prompt)
            return False, response, response

    def import_data(self, data: List[str]) -> bool:
        # TODO
        return True

    def flush(self) -> bool:
        # TODO
        return True

    def get_statistics(self) -> str:
        # TODO
        return "No statistics available"

    def get_inference_engine(self) -> InferenceEngine:
        # TODO
        return self.inference_engine
