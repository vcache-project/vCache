import os
import unittest

import pytest

from vectorq.inference_engine import LangChainInferenceEngine, OpenAIInferenceEngine

INFERENCE_ENGINE_PARAMS = [
    pytest.param(
        OpenAIInferenceEngine,
        {"model_name": "gpt-4o-mini", "temperature": 0},
        marks=pytest.mark.skipif(
            not os.environ.get("OPENAI_API_KEY"),
            reason="OPENAI_API_KEY environment variable not set",
        ),
    ),
    pytest.param(
        LangChainInferenceEngine,
        {"provider": "openai", "model_name": "gpt-4o-mini", "temperature": 0},
        marks=pytest.mark.skipif(
            not os.environ.get("OPENAI_API_KEY"),
            reason="OPENAI_API_KEY environment variable not set",
        ),
    ),
    pytest.param(
        LangChainInferenceEngine,
        {"provider": "anthropic", "model_name": "claude-3-5-sonnet", "temperature": 0},
        marks=pytest.mark.skipif(
            not os.environ.get("ANTHROPIC_API_KEY"),
            reason="ANTHROPIC_API_KEY environment variable not set",
        ),
    ),
    pytest.param(
        LangChainInferenceEngine,
        {"provider": "google", "model_name": "gemini-1.5-flash", "temperature": 0},
        marks=pytest.mark.skipif(
            not os.environ.get("GOOGLE_API_KEY"),
            reason="GOOGLE_API_KEY environment variable not set",
        ),
    ),
]


class TestInferenceEngineStrategy:
    """Test all inference engine strategies using parameterization."""

    @pytest.mark.parametrize(
        "inference_engine_class, engine_params", INFERENCE_ENGINE_PARAMS
    )
    def test_create(self, inference_engine_class, engine_params):
        """Test creating responses from different inference engines."""
        engine = inference_engine_class(**engine_params)

        prompt = "What is the capital of France?"
        response = engine.create(prompt)

        # Verify the response has the expected content
        assert "Paris" in response

    @pytest.mark.parametrize(
        "inference_engine_class, engine_params", INFERENCE_ENGINE_PARAMS
    )
    def test_create_with_output_format(self, inference_engine_class, engine_params):
        """Test creating responses with specified output format."""
        engine = inference_engine_class(**engine_params)

        prompt = "List three European capitals."
        output_format = "Provide the answer as a comma-separated list."
        response = engine.create(prompt, output_format)

        # Verify response contains expected cities and follows the format
        assert any(
            city in response for city in ["Paris", "London", "Berlin", "Madrid", "Rome"]
        )

        # Should be in comma-separated format as requested
        assert "," in response

    @pytest.mark.parametrize(
        "inference_engine_class, engine_params", INFERENCE_ENGINE_PARAMS
    )
    def test_consistent_responses(self, inference_engine_class, engine_params):
        """Test that responses are consistent with temperature=0."""
        engine = inference_engine_class(**engine_params)

        prompt = "What is 2+2?"
        response1 = engine.create(prompt)
        response2 = engine.create(prompt)

        # With temperature=0, responses to simple factual questions should be consistent
        assert "4" in response1
        assert "4" in response2


if __name__ == "__main__":
    unittest.main()
