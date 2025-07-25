import os
import unittest
from unittest.mock import Mock, patch

import pytest

from vcache.inference_engine import LangChainInferenceEngine, OpenAIInferenceEngine

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


class TestInferenceEngineErrorHandling:
    """Test error handling behavior for inference engines."""

    def test_openai_policy_violation_fallback(self):
        """Test that OpenAI policy violations return fallback response instead of crashing."""
        engine = OpenAIInferenceEngine(model_name="gpt-4o-mini", temperature=0)
        
        # Mock the client to raise a policy violation error
        with patch.object(engine, 'client') as mock_client:
            mock_completion = Mock()
            mock_client.chat.completions.create.side_effect = Exception(
                "Error code: 400 - {'error': {'message': 'Invalid prompt: your prompt was flagged as potentially violating our usage policy. Please try again with a different prompt', 'type': 'invalid_request_error', 'param': None, 'code': 'invalid_prompt'}}"
            )
            
            response = engine.create("Test prompt")
            
            # Should return fallback response instead of raising exception
            assert response == "I apologize, but I cannot provide a response to this prompt due to content policy restrictions."

    def test_langchain_policy_violation_fallback(self):
        """Test that LangChain policy violations return fallback response instead of crashing."""
        engine = LangChainInferenceEngine(provider="openai", model_name="gpt-4o-mini", temperature=0)
        
        # Mock the chat model to raise a policy violation error
        with patch.object(engine, 'chat_model') as mock_chat_model:
            mock_chat_model.side_effect = Exception(
                "Content flagged: your prompt was flagged as potentially violating our usage policy"
            )
            
            response = engine.create("Test prompt")
            
            # Should return fallback response instead of raising exception
            assert response == "I apologize, but I cannot provide a response to this prompt due to content policy restrictions."

    def test_openai_other_error_still_raises(self):
        """Test that non-policy errors still raise exceptions."""
        engine = OpenAIInferenceEngine(model_name="gpt-4o-mini", temperature=0)
        
        # Mock the client to raise a non-policy error
        with patch.object(engine, 'client') as mock_client:
            mock_client.chat.completions.create.side_effect = Exception("Network error")
            
            with pytest.raises(Exception) as exc_info:
                engine.create("Test prompt")
            
            assert "Error creating completion from OpenAI" in str(exc_info.value)
            assert "Network error" in str(exc_info.value)

    def test_langchain_other_error_still_raises(self):
        """Test that non-policy errors still raise exceptions."""
        engine = LangChainInferenceEngine(provider="openai", model_name="gpt-4o-mini", temperature=0)
        
        # Mock the chat model to raise a non-policy error
        with patch.object(engine, 'chat_model') as mock_chat_model:
            mock_chat_model.side_effect = Exception("API timeout")
            
            with pytest.raises(Exception) as exc_info:
                engine.create("Test prompt")
            
            assert "Error creating completion from LangChain" in str(exc_info.value)
            assert "API timeout" in str(exc_info.value)


if __name__ == "__main__":
    unittest.main()
