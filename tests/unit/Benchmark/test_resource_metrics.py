import unittest
from unittest.mock import MagicMock, patch

import psutil

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

    def test_memory_mb_matches_process_rss(self):
        # `memory_mb` is a pure unit conversion of `Process.memory_info().rss`.
        # Sample rss immediately before/after the call and assert the
        # reported value falls within that bracket, converted to MB.
        process = psutil.Process()
        rss_before_mb = process.memory_info().rss / (1024 * 1024)
        memory_mb = self.sampler.memory_mb()
        rss_after_mb = process.memory_info().rss / (1024 * 1024)

        self.assertIsInstance(memory_mb, float)
        self.assertGreaterEqual(memory_mb, rss_before_mb)
        self.assertLessEqual(memory_mb, rss_after_mb)


class TestGpuUtilizationPercent(unittest.TestCase):
    def setUp(self):
        self._original_unavailable = resource_metrics._gpu_unavailable
        self._original_handle = resource_metrics._gpu_handle
        resource_metrics._gpu_unavailable = False
        resource_metrics._gpu_handle = None

    def tearDown(self):
        resource_metrics._gpu_unavailable = self._original_unavailable
        resource_metrics._gpu_handle = self._original_handle

    def test_never_raises_and_returns_none_or_valid_percent(self):
        result = gpu_utilization_percent()
        if result is not None:
            self.assertIsInstance(result, float)
            self.assertGreaterEqual(result, 0.0)
            self.assertLessEqual(result, 100.0)

    def test_returns_exact_utilization_and_initializes_nvml_once(self):
        fake_pynvml = MagicMock()
        fake_pynvml.nvmlDeviceGetHandleByIndex.return_value = "handle-0"
        fake_pynvml.nvmlDeviceGetUtilizationRates.return_value = MagicMock(gpu=42)

        with patch.dict("sys.modules", {"pynvml": fake_pynvml}):
            first = gpu_utilization_percent()
            second = gpu_utilization_percent()

            self.assertEqual(first, 42.0)
            self.assertEqual(second, 42.0)
            # nvmlInit/handle lookup must happen once, not on every call.
            fake_pynvml.nvmlInit.assert_called_once()
            fake_pynvml.nvmlDeviceGetHandleByIndex.assert_called_once_with(0)
            self.assertEqual(fake_pynvml.nvmlDeviceGetUtilizationRates.call_count, 2)

            resource_metrics._shutdown_nvml()
            fake_pynvml.nvmlShutdown.assert_called_once()
            self.assertIsNone(resource_metrics._gpu_handle)


class TestCountTokens(unittest.TestCase):
    def test_empty_text_returns_zero(self):
        self.assertEqual(count_tokens(""), 0)
        self.assertEqual(count_tokens(None), 0)

    def test_uses_tiktoken_encoding_when_available(self):
        # Verify the exact transformation (`len(encoding.encode(text))`)
        # against a fake encoder, independent of the real tiktoken vocab.
        original_encoding = resource_metrics._token_encoding
        original_unavailable = resource_metrics._tiktoken_unavailable
        fake_encoding = MagicMock()
        fake_encoding.encode.return_value = [1, 2, 3, 4, 5]
        resource_metrics._token_encoding = fake_encoding
        resource_metrics._tiktoken_unavailable = False
        try:
            self.assertEqual(count_tokens("Is the sky blue?"), 5)
            fake_encoding.encode.assert_called_once_with("Is the sky blue?")
        finally:
            resource_metrics._token_encoding = original_encoding
            resource_metrics._tiktoken_unavailable = original_unavailable

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
