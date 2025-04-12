import unittest
import pytest
import os

from vectorq.config import VectorQConfig
from vectorq.inference_engine import InferenceEngineType
from vectorq.inference_engine.inference_engine import InferenceEngine

INFERENCE_ENGINE_PARAMS = [
    pytest.param(
        InferenceEngineType.OPENAI,
        "gpt-4o-mini",
        marks=pytest.mark.skipif(
            not os.environ.get("OPENAI_API_KEY"),
            reason="OPENAI_API_KEY environment variable not set",
        ),
    ),
    pytest.param(
        InferenceEngineType.LANGCHAIN,
        "openai/gpt-4o-mini",
        marks=pytest.mark.skipif(
            not os.environ.get("OPENAI_API_KEY"),
            reason="OPENAI_API_KEY environment variable not set",
        ),
    ),
    pytest.param(
        InferenceEngineType.LANGCHAIN,
        "anthropic/claude-3-5-sonnet",
        marks=pytest.mark.skipif(
            not os.environ.get("ANTHROPIC_API_KEY"),
            reason="ANTHROPIC_API_KEY environment variable not set",
        ),
    ),
    pytest.param(
        InferenceEngineType.LANGCHAIN,
        "google/gemini-1.5-flash",
        marks=pytest.mark.skipif(
            not os.environ.get("GOOGLE_API_KEY"),
            reason="GOOGLE_API_KEY environment variable not set",
        ),
    ),
]


class TestInferenceEngineStrategy:
    """Test all inference engine strategies using parameterization."""

    @pytest.mark.parametrize(
        "inference_engine_type, inference_engine_model_name", INFERENCE_ENGINE_PARAMS
    )
    def test_create(self, inference_engine_type, inference_engine_model_name):
        """Test creating responses from different inference engines."""
        config = VectorQConfig(
            inference_engine_type=inference_engine_type,
            inference_engine_model_name=inference_engine_model_name,
            inference_engine_temperature=0,
        )

        engine = InferenceEngine(vectorq_config=config)
        prompt = "What is the capital of France?"
        response = engine.create(prompt)

        # Verify the response has the expected content
        assert "Paris" in response

    @pytest.mark.parametrize(
        "inference_engine_type, inference_engine_model_name", INFERENCE_ENGINE_PARAMS
    )
    def test_create_with_output_format(
        self, inference_engine_type, inference_engine_model_name
    ):
        """Test creating responses with specified output format."""
        config = VectorQConfig(
            inference_engine_type=inference_engine_type,
            inference_engine_model_name=inference_engine_model_name,
            inference_engine_temperature=0,
        )

        engine = InferenceEngine(vectorq_config=config)
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
        "inference_engine_type, inference_engine_model_name", INFERENCE_ENGINE_PARAMS
    )
    def test_consistent_responses(
        self, inference_engine_type, inference_engine_model_name
    ):
        """Test that responses are consistent with temperature=0."""
        config = VectorQConfig(
            inference_engine_type=inference_engine_type,
            inference_engine_model_name=inference_engine_model_name,
            inference_engine_temperature=0,
        )

        engine = InferenceEngine(vectorq_config=config)
        prompt = "What is 2+2?"
        response1 = engine.create(prompt)
        response2 = engine.create(prompt)

        # With temperature=0, responses to simple factual questions should be consistent
        assert "4" in response1
        assert "4" in response2


if __name__ == "__main__":
    unittest.main()
