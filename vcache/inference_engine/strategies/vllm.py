import logging
import os
from typing import Any, Dict, List, Optional

import torch
from vllm import LLM, RequestOutput, SamplingParams

from vcache.inference_engine.inference_engine import InferenceEngine

logger = logging.getLogger(__name__)


class VLLMInferenceEngine(InferenceEngine):
    """
    vLLM-based inference engine for generating responses using vLLM's optimized inference.

    This engine provides efficient GPU-accelerated inference with support for:
    - Multi-GPU tensor parallelism
    - Memory optimization
    - Quantization
    - Custom model configurations
    """

    def __init__(
        self,
        model_name: str,
        temperature: float = 0.8,
        gpu_ids: Optional[List[int]] = None,
        gpu_memory_utilization: float = 0.9,
        max_model_len: Optional[int] = None,
        dtype: str = "auto",
        trust_remote_code: bool = False,
        quantization: Optional[str] = None,
        max_tokens: int = 512,
        top_p: float = 0.9,
        top_k: int = -1,
        enforce_eager: bool = False,
    ):
        """
        Initialize vLLM inference engine.

        Args:
            model_name: Hugging Face model ID or local path to model
            temperature: Sampling temperature (0.0 to 2.0). Higher = more random
            gpu_ids: List of GPU IDs to use (e.g., [0, 1]). None = use all available
            gpu_memory_utilization: Fraction of GPU memory to use (0.0 to 1.0)
            max_model_len: Maximum context length. None = use model default
            dtype: Model data type ('auto', 'float16', 'bfloat16', 'float32')
            trust_remote_code: Allow execution of custom model code
            quantization: Quantization method ('awq', 'fp8', 'gptq', etc.)
            max_tokens: Maximum number of tokens to generate per request
            top_p: Nucleus sampling parameter (0.0 to 1.0)
            top_k: Top-k sampling parameter (-1 = disabled)
            enforce_eager: Disable CUDA graphs for debugging (uses more memory)
        """
        super().__init__()

        # Store configuration with proper typing
        self.model_name: str = model_name
        self.temperature: float = temperature
        self.gpu_memory_utilization: float = gpu_memory_utilization
        self.max_model_len: Optional[int] = max_model_len
        self.dtype: str = dtype
        self.trust_remote_code: bool = trust_remote_code
        self.quantization: Optional[str] = quantization
        self.max_tokens: int = max_tokens
        self.top_p: float = top_p
        self.top_k: int = top_k
        self.enforce_eager: bool = enforce_eager
        self.tensor_parallel_size: int = self._configure_gpus(gpu_ids)

        self.llm: LLM = None
        self.sampling_params: SamplingParams = None
        self._is_warmed_up: bool = False

        self._initialize_engine()

    def _configure_gpus(self, gpu_ids: Optional[List[int]]) -> int:
        """
        Configure GPU settings and return tensor parallel size.

        Args:
            gpu_ids: List of GPU IDs to use

        Returns:
            Number of GPUs to use for tensor parallelism
        """
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available. vLLM requires GPU support.")

        available_gpus: int = torch.cuda.device_count()

        if gpu_ids is not None:
            for gpu_id in gpu_ids:
                if gpu_id >= available_gpus:
                    raise ValueError(
                        f"GPU ID {gpu_id} not available. Only {available_gpus} GPUs detected."
                    )

            os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, gpu_ids))
            tensor_parallel_size: int = len(gpu_ids)
        else:
            # Use all available GPUs
            tensor_parallel_size = available_gpus

        logger.info(f"Configured vLLM to use {tensor_parallel_size} GPU(s)")
        return tensor_parallel_size

    def _initialize_engine(self) -> None:
        """Initialize the vLLM LLM engine with configured parameters."""
        logger.info(f"Initializing vLLM engine with model: {self.model_name}")

        llm_kwargs: Dict[str, Any] = {
            "model": self.model_name,
            "tensor_parallel_size": self.tensor_parallel_size,
            "gpu_memory_utilization": self.gpu_memory_utilization,
            "dtype": self.dtype,
            "trust_remote_code": self.trust_remote_code,
            "enforce_eager": self.enforce_eager,
        }

        if self.max_model_len is not None:
            llm_kwargs["max_model_len"] = self.max_model_len

        if self.quantization is not None:
            llm_kwargs["quantization"] = self.quantization

        try:
            self.llm = LLM(**llm_kwargs)

            sampling_kwargs: Dict[str, Any] = {
                "temperature": self.temperature,
                "top_p": self.top_p,
                "max_tokens": self.max_tokens,
            }

            if self.top_k > 0:
                sampling_kwargs["top_k"] = self.top_k

            self.sampling_params = SamplingParams(**sampling_kwargs)

            logger.info("vLLM engine initialized successfully!")

        except Exception as e:
            logger.error(f"Failed to initialize vLLM engine: {str(e)}")
            raise RuntimeError(f"Failed to initialize vLLM engine: {str(e)}")

    def warm_up(self, warm_up_prompt: str = "Hello, how are you?") -> str:
        """
        Warm up the model by making an initial request to prepare it for inference.
        This helps reduce latency for subsequent requests by initializing CUDA contexts
        and loading model weights into GPU memory.

        Args:
            warm_up_prompt: The prompt to use for warming up the model

        Returns:
            The response from the warm-up request
        """
        if self.llm is None:
            raise RuntimeError("vLLM engine not initialized")

        if self._is_warmed_up:
            logger.debug("Model is already warmed up")
            return "Model already warmed up"

        logger.info("Warming up vLLM model...")

        try:
            response: str = self.create(warm_up_prompt)
            self._is_warmed_up = True

            logger.info("Model warm-up completed successfully!")
            return response

        except Exception as e:
            logger.error(f"Error during model warm-up: {str(e)}")
            raise RuntimeError(f"Error during model warm-up: {str(e)}")

    def create(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """
        Generate a response for the given prompt using vLLM.

        Args:
            prompt: The user prompt to generate a response for
            system_prompt: Optional system prompt to guide model behavior

        Returns:
            Generated response text
        """
        if self.llm is None:
            raise RuntimeError("vLLM engine not initialized")

        try:
            full_prompt: str = self._format_prompt(prompt, system_prompt)

            outputs = self.llm.generate([full_prompt], self.sampling_params)

            # Extract the generated text
            if outputs and len(outputs) > 0 and len(outputs[0].outputs) > 0:
                generated_text: str = outputs[0].outputs[0].text.strip()
                return generated_text
            else:
                logger.error("No output generated by vLLM")
                raise RuntimeError("No output generated by vLLM")

        except Exception as e:
            logger.error(f"Error during text generation: {str(e)}")
            raise RuntimeError(f"Error during text generation: {str(e)}")

    def _format_prompt(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """
        Format the prompt with optional system prompt.

        Args:
            prompt: User prompt
            system_prompt: Optional system prompt

        Returns:
            Formatted prompt string
        """
        if system_prompt:
            formatted_prompt: str = (
                f"System: {system_prompt}\n\nUser: {prompt}\n\nAssistant:"
            )
        else:
            formatted_prompt = prompt

        return formatted_prompt

    def generate_batch(
        self, prompts: List[str], system_prompt: Optional[str] = None
    ) -> List[str]:
        """
        Generate responses for multiple prompts in a batch.

        Args:
            prompts: List of prompts to generate responses for
            system_prompt: Optional system prompt applied to all prompts

        Returns:
            List of generated responses corresponding to input prompts
        """
        if self.llm is None:
            raise RuntimeError("vLLM engine not initialized")

        try:
            formatted_prompts: List[str] = [
                self._format_prompt(prompt, system_prompt) for prompt in prompts
            ]
            outputs: List[RequestOutput] = self.llm.generate(
                formatted_prompts, self.sampling_params
            )

            responses: List[str] = []
            for output in outputs:
                if output.outputs and len(output.outputs) > 0:
                    responses.append(output.outputs[0].text.strip())
                else:
                    responses.append("")

            logger.debug(f"Generated batch responses for {len(prompts)} prompts")
            return responses

        except Exception as e:
            logger.error(f"Error during batch text generation: {str(e)}")
            raise RuntimeError(f"Error during batch text generation: {str(e)}")

    def update_sampling_params(self, **kwargs) -> None:
        """
        Update sampling parameters for future generations.

        Args:
            **kwargs: Sampling parameters to update (temperature, top_p, max_tokens, etc.)
        """

        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)

        sampling_kwargs: Dict[str, Any] = {
            "temperature": self.temperature,
            "top_p": self.top_p,
            "max_tokens": self.max_tokens,
        }

        if self.top_k > 0:
            sampling_kwargs["top_k"] = self.top_k

        sampling_kwargs.update(kwargs)

        self.sampling_params = SamplingParams(**sampling_kwargs)

        logger.debug(f"Updated sampling parameters: {kwargs}")

    def is_warmed_up(self) -> bool:
        """
        Check if the model has been warmed up.

        Returns:
            True if the model has been warmed up, False otherwise
        """
        return self._is_warmed_up

    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the loaded model and configuration.

        Returns:
            Dictionary containing model and configuration information
        """
        return {
            "model_name": self.model_name,
            "tensor_parallel_size": self.tensor_parallel_size,
            "gpu_memory_utilization": self.gpu_memory_utilization,
            "max_model_len": self.max_model_len,
            "dtype": self.dtype,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "max_tokens": self.max_tokens,
            "quantization": self.quantization,
            "trust_remote_code": self.trust_remote_code,
            "enforce_eager": self.enforce_eager,
            "cuda_available": torch.cuda.is_available(),
            "cuda_device_count": torch.cuda.device_count()
            if torch.cuda.is_available()
            else 0,
            "is_warmed_up": self._is_warmed_up,
        }
