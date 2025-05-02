from typing import override

from vectorq.inference_engine.inference_engine import InferenceEngine


class BenchmarkInferenceEngine(InferenceEngine):
    """
    An inference engine implementation that returns pre-computed responses for given prompts.
    It is used for benchmarking purposes.
    """

    def __init__(self, prompt_response_map: dict[str, str]):
        """
        Initialize the benchmark engine with predefined responses.

        Args:
            prompt_response_map: A dictionary mapping prompts to their pre-computed responses.
                                 Keys are prompt strings and values are response strings.
        """
        super().__init__()
        self.prompt_response_map = prompt_response_map

    @override
    def create(self, prompt: str, system_prompt: str = None) -> str:
        if prompt not in self.prompt_response_map:
            raise ValueError(f"Prompt {prompt} not found in prompt_response_map")
        return self.prompt_response_map[prompt]
