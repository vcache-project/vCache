from typing_extensions import override

from vcache.inference_engine.inference_engine import InferenceEngine


class BenchmarkInferenceEngine(InferenceEngine):
    """
    An inference engine implementation that returns pre-computed responses for given prompts.
    It is used for benchmarking purposes.
    """

    def set_next_response(self, response: str):
        """
        Set the next response to be returned by create.

        Args:
            response: The response to return on next call.
        """
        self.next_response = response

    @override
    def create(self, prompt: str, system_prompt: str = None) -> str:
        """
        Create a response using the pre-set response.

        Args:
            prompt: The prompt to process (ignored in benchmark mode).
            system_prompt: The system prompt (ignored in benchmark mode).

        Returns:
            The pre-set response.

        Raises:
            ValueError: If no response has been set.
        """
        if self.next_response is None:
            raise ValueError("No next response set")
        return self.next_response
