import unittest

from benchmarks.common import resource_metrics
from benchmarks.common.resource_metrics import (
    ResourceSampler,
    count_tokens,
    gpu_utilization_percent,
)


class TestResourceSampler(unittest.TestCase):
    def setUp(self):
        self.sampler = ResourceSampler()

    def test_cpu_percent_returns_non_negative_float(self):
        cpu_percent = self.sampler.cpu_percent()
        self.assertIsInstance(cpu_percent, float)
        self.assertGreaterEqual(cpu_percent, 0.0)

    def test_memory_mb_returns_positive_float(self):
        memory_mb = self.sampler.memory_mb()
        self.assertIsInstance(memory_mb, float)
        # The current process always occupies some resident memory.
        self.assertGreater(memory_mb, 0.0)


class TestGpuUtilizationPercent(unittest.TestCase):
    def test_never_raises_and_returns_none_or_valid_percent(self):
        result = gpu_utilization_percent()
        if result is not None:
            self.assertIsInstance(result, float)
            self.assertGreaterEqual(result, 0.0)
            self.assertLessEqual(result, 100.0)


class TestCountTokens(unittest.TestCase):
    def test_empty_text_returns_zero(self):
        self.assertEqual(count_tokens(""), 0)
        self.assertEqual(count_tokens(None), 0)

    def test_counts_tokens_for_simple_text(self):
        # Regardless of whether tiktoken is available, a short sentence
        # should resolve to a small, positive token count.
        count = count_tokens("Is the sky blue?")
        self.assertGreater(count, 0)
        self.assertLessEqual(count, 10)

    def test_whitespace_fallback_matches_word_count(self):
        # Force the whitespace fallback path regardless of whether tiktoken
        # is installed in this environment.
        original_encoding = resource_metrics._token_encoding
        original_unavailable = resource_metrics._tiktoken_unavailable
        resource_metrics._token_encoding = None
        resource_metrics._tiktoken_unavailable = True
        try:
            self.assertEqual(count_tokens("one two three four"), 4)
        finally:
            resource_metrics._token_encoding = original_encoding
            resource_metrics._tiktoken_unavailable = original_unavailable


if __name__ == "__main__":
    unittest.main()
