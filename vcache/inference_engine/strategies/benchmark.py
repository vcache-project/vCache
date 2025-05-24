from typing import override

from vcache.inference_engine.inference_engine import InferenceEngine


class BenchmarkInferenceEngine(InferenceEngine):
    """
    An inference engine implementation that returns pre-computed responses for given prompts.
    It is used for benchmarking purposes.
    """

    def set_next_response(self, response: str):
        self.next_response = response

    @override
    def create(self, prompt: str, system_prompt: str = None) -> str:
        if self.next_response is None:
            raise ValueError("No next response set")
        return self.next_response
