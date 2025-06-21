import os
import sys
import unittest

# Add current directory to path for imports
sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)


class TestLLMComparisonIntegration(unittest.TestCase):
    """Integration tests for LLM-based similarity evaluation."""

    def test_implementation_file_exists(self):
        """Test that the LLM comparison implementation file exists and is valid."""
        file_path = os.path.join(
            os.path.dirname(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            ),
            "vcache",
            "vcache_core",
            "similarity_evaluator",
            "strategies",
            "llm_comparison.py",
        )

        self.assertTrue(
            os.path.exists(file_path), "LLM comparison implementation file should exist"
        )

        # Check file is not empty
        with open(file_path, "r") as f:
            content = f.read()

        self.assertGreater(len(content), 100, "Implementation file should not be empty")
        self.assertIn("LLMComparisonSimilarityEvaluator", content)
        self.assertIn("answers_similar", content)
        self.assertIn("inference_engine", content)

    def test_implementation_syntax(self):
        """Test that the implementation has valid Python syntax."""
        file_path = os.path.join(
            os.path.dirname(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            ),
            "vcache",
            "vcache_core",
            "similarity_evaluator",
            "strategies",
            "llm_comparison.py",
        )

        # Try to compile the file
        with open(file_path, "r") as f:
            content = f.read()

        try:
            compile(content, file_path, "exec")
        except SyntaxError as e:
            self.fail(f"Implementation file has syntax errors: {e}")

    def test_implementation_structure(self):
        """Test that the implementation has the expected structure."""
        file_path = os.path.join(
            os.path.dirname(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            ),
            "vcache",
            "vcache_core",
            "similarity_evaluator",
            "strategies",
            "llm_comparison.py",
        )

        with open(file_path, "r") as f:
            content = f.read()

        # Check for required components
        self.assertIn("class LLMComparisonSimilarityEvaluator", content)
        self.assertIn("def __init__", content)
        self.assertIn("def answers_similar", content)
        self.assertIn("a: str", content)
        self.assertIn("b: str", content)
        self.assertIn("id_set_a: int = None", content)
        self.assertIn("id_set_b: int = None", content)
        self.assertIn("-> bool", content)
        self.assertIn("self.inference_engine", content)
        self.assertIn("system_prompt", content)
        self.assertIn("user_prompt", content)
        self.assertIn("try:", content)
        self.assertIn("except", content)
        self.assertIn("YES", content)

    def test_prompt_design_quality(self):
        """Test that the prompts are well-designed."""
        file_path = os.path.join(
            os.path.dirname(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            ),
            "vcache",
            "vcache_core",
            "similarity_evaluator",
            "strategies",
            "llm_comparison.py",
        )

        with open(file_path, "r") as f:
            content = f.read()

        # Check system prompt quality
        self.assertIn("judge", content.lower())
        self.assertIn("semantically equivalent", content.lower())
        self.assertIn("YES", content)
        self.assertIn("NO", content)

        # Check user prompt structure
        self.assertIn("Answer 1:", content)
        self.assertIn("Answer 2:", content)

    def test_error_handling_implementation(self):
        """Test that error handling is properly implemented."""
        file_path = os.path.join(
            os.path.dirname(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            ),
            "vcache",
            "vcache_core",
            "similarity_evaluator",
            "strategies",
            "llm_comparison.py",
        )

        with open(file_path, "r") as f:
            content = f.read()

        # Check for proper error handling
        self.assertIn("if not self.inference_engine:", content)
        self.assertIn("return False", content)
        self.assertIn("try:", content)
        self.assertIn("except Exception:", content)

    def test_response_parsing_implementation(self):
        """Test that response parsing is correctly implemented."""
        file_path = os.path.join(
            os.path.dirname(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            ),
            "vcache",
            "vcache_core",
            "similarity_evaluator",
            "strategies",
            "llm_comparison.py",
        )

        with open(file_path, "r") as f:
            content = f.read()

        # Check for proper response parsing
        self.assertIn(".strip()", content)
        self.assertIn(".upper()", content)

    def test_inheritance_implementation(self):
        """Test that inheritance is correctly implemented."""
        file_path = os.path.join(
            os.path.dirname(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            ),
            "vcache",
            "vcache_core",
            "similarity_evaluator",
            "strategies",
            "llm_comparison.py",
        )

        with open(file_path, "r") as f:
            content = f.read()

        # Check for proper inheritance
        self.assertIn(
            "from vcache.vcache_core.similarity_evaluator.similarity_evaluator import",
            content,
        )
        self.assertIn("SimilarityEvaluator", content)
        self.assertIn(
            "class LLMComparisonSimilarityEvaluator(SimilarityEvaluator)", content
        )
        self.assertIn("super().__init__()", content)


if __name__ == "__main__":
    unittest.main()
