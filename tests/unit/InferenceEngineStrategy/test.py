import os
from typing import Any, Dict, Type

import pytest
from dotenv import load_dotenv

from vectorq.inference_engine import (
    InferenceEngine,
    LangChainInferenceEngine,
    OpenAIInferenceEngine,
)

load_dotenv()

OPENAI_API_KEY_AVAILABLE = bool(os.environ.get("OPENAI_API_KEY"))
ANTHROPIC_API_KEY_AVAILABLE = bool(os.environ.get("ANTHROPIC_API_KEY"))

# Build parameter list dynamically so we only execute engines we can call
INFERENCE_ENGINE_PARAMS = []
if OPENAI_API_KEY_AVAILABLE:
    INFERENCE_ENGINE_PARAMS.extend(
        [
            pytest.param(
                OpenAIInferenceEngine,
                {"model_name": "gpt-4.1-nano-2025-04-14", "temperature": 0},
            ),
            pytest.param(
                LangChainInferenceEngine,
                {
                    "provider": "openai",
                    "model_name": "gpt-4.1-nano-2025-04-14",
                    "temperature": 0,
                },
            ),
        ]
    )

if ANTHROPIC_API_KEY_AVAILABLE:
    INFERENCE_ENGINE_PARAMS.append(
        pytest.param(
            LangChainInferenceEngine,
            {
                "provider": "anthropic",
                "model_name": "claude-3-haiku-20240307",
                "temperature": 0,
            },
        )
    )


@pytest.mark.skipif(
    not INFERENCE_ENGINE_PARAMS, reason="No compatible API keys found for tests."
)
class TestInferenceEngineStrategy:
    # """Comprehensive tests for the inference engine strategies."""

    @pytest.mark.parametrize(
        "inference_engine_class, engine_params", INFERENCE_ENGINE_PARAMS
    )
    def test_infer(
        self,
        inference_engine_class: Type[InferenceEngine],
        engine_params: Dict[str, Any],
    ) -> None:
        engine: InferenceEngine = inference_engine_class(**engine_params)

        prompt = "What is the capital of France?"
        response = engine.infer(prompt)

        assert response is not None
        assert "paris" in response.lower()

    @pytest.mark.parametrize(
        "inference_engine_class, engine_params", INFERENCE_ENGINE_PARAMS
    )
    def test_infer_with_system_prompt_override(
        self,
        inference_engine_class: Type[InferenceEngine],
        engine_params: Dict[str, Any],
    ) -> None:
        engine: InferenceEngine = inference_engine_class(**engine_params)

        system_prompt = "ALWAYS ANSWER IN UPPERCASE."
        prompt = "Say 'hello world'"
        response = engine.infer(prompt, system_prompt=system_prompt)

        letters = [c for c in response if c.isalpha()]
        if not letters:
            assert False, "No alphabetic characters found in response"
        is_all_upper = sum(1 for c in letters if c.isupper()) / len(letters) > 0.9
        assert is_all_upper

    @pytest.mark.parametrize(
        "inference_engine_class, engine_params", INFERENCE_ENGINE_PARAMS
    )
    @pytest.mark.flaky(reruns=3)
    def test_consistency_with_zero_temperature(
        self,
        inference_engine_class: Type[InferenceEngine],
        engine_params: Dict[str, Any],
    ) -> None:
        # Set temperature to 0 when creating the engine
        params = {**engine_params, "temperature": 0}
        engine: InferenceEngine = inference_engine_class(**params)

        prompt = "Use a short and brief sentence to describe Paris."
        response1 = engine.infer(prompt)
        response2 = engine.infer(prompt)

        assert response1 == response2

    @pytest.mark.parametrize(
        "inference_engine_class, engine_params", INFERENCE_ENGINE_PARAMS
    )
    @pytest.mark.flaky(reruns=3)
    def test_infer_with_temperature_kwarg_override(
        self,
        inference_engine_class: Type[InferenceEngine],
        engine_params: Dict[str, Any],
    ) -> None:
        # Set temperature to 0.9 when creating the engine
        params = {**engine_params, "temperature": 0.9}
        engine: InferenceEngine = inference_engine_class(**params)

        # With temperature 0.9, the response should be different
        prompt = "Use a short and brief sentence to describe Paris."
        response1 = engine.infer(prompt)
        response2 = engine.infer(prompt)
        assert response1 != response2

        # With overriding temperature 0, the response should be almost the same
        params = {**engine_params, "temperature": 0}
        engine: InferenceEngine = inference_engine_class(**params)
        response3 = engine.infer(prompt)
        response4 = engine.infer(prompt)
        assert response3 == response4

    def test_langchain_unsupported_provider(self) -> None:
        engine = LangChainInferenceEngine(provider="unknown", model_name="foo")
        with pytest.raises(Exception) as exc:
            _ = engine.infer("Test")
        assert "Unsupported provider" in str(exc.value)


if __name__ == "__main__":
    pytest.main()
