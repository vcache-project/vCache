import unittest
from unittest.mock import Mock

from vcache.inference_engine.strategies.open_ai import OpenAIInferenceEngine
from vcache.vcache_core.similarity_evaluator.strategies.llm_comparison import (
    LLMComparisonSimilarityEvaluator,
)
from vcache.vcache_core.similarity_evaluator.strategies.string_comparison import (
    StringComparisonSimilarityEvaluator,
)


class TestStringComparisonSimilarityEvaluator(unittest.TestCase):
    """Test the existing string comparison similarity evaluator."""

    def setUp(self):
        self.evaluator = StringComparisonSimilarityEvaluator()

    def test_identical_strings(self):
        """Test that identical strings are considered similar."""
        result = self.evaluator.answers_similar("Paris", "Paris")
        self.assertTrue(result)

    def test_different_strings(self):
        """Test that different strings are not considered similar."""
        result = self.evaluator.answers_similar("Paris", "London")
        self.assertFalse(result)

    def test_case_insensitive(self):
        """Test that comparison is case insensitive."""
        result = self.evaluator.answers_similar("Paris", "paris")
        self.assertTrue(result)

    def test_whitespace_handling(self):
        """Test that whitespace is handled correctly."""
        result = self.evaluator.answers_similar("  Paris  ", "Paris")
        self.assertTrue(result)

    def test_punctuation_handling(self):
        """Test that punctuation is handled correctly."""
        result = self.evaluator.answers_similar("Paris.", "Paris")
        self.assertTrue(result)


class TestLLMComparisonSimilarityEvaluator(unittest.TestCase):
    """Test the LLM-based comparison similarity evaluator."""

    def setUp(self):
        self.evaluator = LLMComparisonSimilarityEvaluator(
            inference_engine=OpenAIInferenceEngine(model_name="gpt-4.1-nano-2025-04-14")
        )

    def test_no_inference_engine(self):
        """Test behavior when no inference engine is set."""
        # inference_engine should be None by default
        evaluator = LLMComparisonSimilarityEvaluator(inference_engine=None)
        result = evaluator.answers_similar("Paris", "The capital of France")
        self.assertFalse(result)

    def test_inference_engine_returns_yes(self):
        """Test behavior when inference engine returns YES."""
        mock_engine = Mock()
        mock_engine.create.return_value = "YES"
        self.evaluator.inference_engine = mock_engine

        result = self.evaluator.answers_similar("Paris", "The capital of France")
        self.assertTrue(result)

        # Verify the inference engine was called with correct parameters
        mock_engine.create.assert_called_once()
        args, kwargs = mock_engine.create.call_args
        user_prompt, system_prompt = args

        self.assertIn("Paris", user_prompt)
        self.assertIn("The capital of France", user_prompt)
        self.assertIn("semantically equivalent", system_prompt.lower())

    def test_inference_engine_returns_no(self):
        """Test behavior when inference engine returns NO."""
        mock_engine = Mock()
        mock_engine.create.return_value = "NO"
        self.evaluator.inference_engine = mock_engine

        result = self.evaluator.answers_similar("Paris", "London")
        self.assertFalse(result)

    def test_inference_engine_returns_yes_with_extra_text(self):
        """Test behavior when inference engine returns YES with additional text."""
        mock_engine = Mock()
        mock_engine.create.return_value = "YES, these answers are equivalent."
        self.evaluator.inference_engine = mock_engine

        result = self.evaluator.answers_similar("Paris", "The capital of France")
        self.assertTrue(result)

    def test_inference_engine_returns_no_with_extra_text(self):
        """Test behavior when inference engine returns NO with additional text."""
        mock_engine = Mock()
        mock_engine.create.return_value = "NO, these are different."
        self.evaluator.inference_engine = mock_engine

        result = self.evaluator.answers_similar("Paris", "London")
        self.assertFalse(result)

    def test_inference_engine_returns_unexpected_response(self):
        """Test behavior when inference engine returns unexpected response."""
        mock_engine = Mock()
        mock_engine.create.return_value = "MAYBE"
        self.evaluator.inference_engine = mock_engine

        result = self.evaluator.answers_similar("Paris", "The capital of France")
        self.assertFalse(result)

    def test_inference_engine_returns_empty_response(self):
        """Test behavior when inference engine returns empty response."""
        mock_engine = Mock()
        mock_engine.create.return_value = ""
        self.evaluator.inference_engine = mock_engine

        result = self.evaluator.answers_similar("Paris", "The capital of France")
        self.assertFalse(result)

    def test_inference_engine_returns_whitespace_response(self):
        """Test behavior when inference engine returns only whitespace."""
        mock_engine = Mock()
        mock_engine.create.return_value = "   \n\t   "
        self.evaluator.inference_engine = mock_engine

        result = self.evaluator.answers_similar("Paris", "The capital of France")
        self.assertFalse(result)

    def test_inference_engine_case_insensitive_yes(self):
        """Test that YES response is case insensitive."""
        mock_engine = Mock()
        mock_engine.create.return_value = "yes"
        self.evaluator.inference_engine = mock_engine

        result = self.evaluator.answers_similar("Paris", "The capital of France")
        self.assertTrue(result)

    def test_inference_engine_throws_exception(self):
        """Test behavior when inference engine throws an exception."""
        mock_engine = Mock()
        mock_engine.create.side_effect = Exception("API Error")
        self.evaluator.inference_engine = mock_engine

        result = self.evaluator.answers_similar("Paris", "The capital of France")
        self.assertFalse(result)

    def test_prompt_construction(self):
        """Test that prompts are constructed correctly."""
        mock_engine = Mock()
        mock_engine.create.return_value = "YES"
        self.evaluator.inference_engine = mock_engine

        answer_a = "Paris is the capital"
        answer_b = "The capital city is Paris"

        self.evaluator.answers_similar(answer_a, answer_b)

        # Verify the call was made
        mock_engine.create.assert_called_once()
        args, kwargs = mock_engine.create.call_args
        user_prompt, system_prompt = args

        # Check user prompt contains both answers
        self.assertIn(answer_a, user_prompt)
        self.assertIn(answer_b, user_prompt)
        self.assertIn("Answer 1:", user_prompt)
        self.assertIn("Answer 2:", user_prompt)
        self.assertIn("semantically equivalent", user_prompt)

        # Check system prompt contains judge instructions
        self.assertIn("judge", system_prompt.lower())
        self.assertIn("semantically equivalent", system_prompt.lower())
        self.assertIn("YES", system_prompt)
        self.assertIn("NO", system_prompt)

    def test_empty_answers(self):
        """Test behavior with empty answers."""
        mock_engine = Mock()
        mock_engine.create.return_value = "YES"
        self.evaluator.inference_engine = mock_engine

        result = self.evaluator.answers_similar("", "")
        self.assertTrue(result)

        # Verify the call was still made
        mock_engine.create.assert_called_once()

    def test_none_answers(self):
        """Test behavior with None answers."""
        mock_engine = Mock()
        mock_engine.create.return_value = "YES"
        self.evaluator.inference_engine = mock_engine

        # This should work since we convert to string in the f-string
        result = self.evaluator.answers_similar(None, None)
        self.assertTrue(result)

    def test_inheritance(self):
        """Test that LLMComparisonSimilarityEvaluator inherits from SimilarityEvaluator."""
        from vcache.vcache_core.similarity_evaluator.similarity_evaluator import (
            SimilarityEvaluator,
        )

        self.assertIsInstance(self.evaluator, SimilarityEvaluator)

    def test_method_signature(self):
        """Test that answers_similar method has correct signature."""
        import inspect

        sig = inspect.signature(self.evaluator.answers_similar)
        params = list(sig.parameters.keys())

        self.assertEqual(params, ["a", "b", "id_set_a", "id_set_b"])
        self.assertEqual(sig.return_annotation, bool)


if __name__ == "__main__":
    unittest.main()
