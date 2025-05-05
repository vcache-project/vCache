from typing import Any, override

from vectorq.inference_engine.inference_engine import InferenceEngine


class BenchmarkInferenceEngine(InferenceEngine):
    """
    An inference engine implementation that returns pre-computed responses.
    It is used for benchmarking purposes.
    """

    def set_response(self, response: str):
        self.response = response

    @override
    def infer(self, prompt: str, system_prompt: str = None, **kwargs: Any) -> str:
        if self.response is None:
            raise ValueError("No response set")
        return self.response
