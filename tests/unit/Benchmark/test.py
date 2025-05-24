import unittest

import pandas as pd

from benchmarks._plotter_helper import (
    compute_accuracy_cumulative_list,
    compute_accuracy_score,
    compute_avg_latency_score,
    compute_cache_hit_rate_cumulative_list,
    compute_cache_hit_rate_score,
    compute_duration_cumulative_list,
    compute_duration_score,
    compute_error_rate_cumulative_list,
    compute_error_rate_score,
    compute_f1_score_cumulative_list,
    compute_f1_score_score,
    compute_false_positive_rate_cumulative_list,
    compute_false_positive_rate_score,
    compute_precision_cumulative_list,
    compute_precision_score,
    compute_recall_cumulative_list,
    compute_recall_score,
)


class TestVCacheBenchmark(unittest.TestCase):
    def setUp(self):
        self.tp = pd.Series([1, 0, 1, 0, 1])  # True Positives
        self.fp = pd.Series([0, 1, 0, 1, 0])  # False Positives
        self.tn = pd.Series([0, 1, 1, 0, 1])  # True Negatives
        self.fn = pd.Series([1, 0, 0, 1, 0])  # False Negatives
        self.cache_hits = pd.Series([0, 1, 1, 0, 1])  # Cache hits
        self.latencies = pd.Series([0.5, 1.0, 0.2, 1.5, 0.8])  # Latencies

    def test_compute_accuracy_cumulative_list(self):
        result = compute_accuracy_cumulative_list(self.tp, self.fp, self.tn, self.fn)
        # Expected cumulative values:
        # tp cumsum: [1, 1, 2, 2, 3]
        # tn cumsum: [0, 1, 2, 2, 3]
        # fp cumsum: [0, 1, 1, 2, 2]
        # fn cumsum: [1, 1, 1, 2, 2]
        # accuracy = (tp + tn) / (tp + tn + fp + fn)
        expected = pd.Series([0.5, 0.5, 0.667, 0.5, 0.6])
        pd.testing.assert_series_equal(result.round(3), expected.round(3))

    def test_compute_accuracy_score(self):
        # Test compute_accuracy_score function
        result = compute_accuracy_score(self.tp, self.fp, self.tn, self.fn)
        # Final value from cumulative accuracy
        expected = 0.6
        self.assertAlmostEqual(result.item(), expected, places=1)

    def test_compute_precision_cumulative_list(self):
        # Test compute_precision_cumulative_list function
        result = compute_precision_cumulative_list(self.tp, self.fp)
        # Expected cumulative values:
        # tp cumsum: [1, 1, 2, 2, 3]
        # fp cumsum: [0, 1, 1, 2, 2]
        # precision = tp / (tp + fp)
        expected = pd.Series([1.0, 0.5, 0.667, 0.5, 0.6])
        pd.testing.assert_series_equal(result.round(3), expected.round(3))

    def test_compute_precision_score(self):
        # Test compute_precision_score function
        result = compute_precision_score(self.tp, self.fp)
        # Final value from cumulative precision
        expected = 0.6
        self.assertAlmostEqual(result.item(), expected, places=1)

    def test_compute_recall_cumulative_list(self):
        # Test compute_recall_cumulative_list function
        result = compute_recall_cumulative_list(self.tp, self.fn)
        # Expected cumulative values:
        # tp cumsum: [1, 1, 2, 2, 3]
        # fn cumsum: [1, 1, 1, 2, 2]
        # recall = tp / (tp + fn)
        expected = pd.Series([0.5, 0.5, 0.667, 0.5, 0.6])
        pd.testing.assert_series_equal(result.round(3), expected.round(3))

    def test_compute_recall_score(self):
        # Test compute_recall_score function
        result = compute_recall_score(self.tp, self.fn)
        # Final value from cumulative recall
        expected = 0.6
        self.assertAlmostEqual(result.item(), expected, places=1)

    def test_compute_false_positive_rate_cumulative_list(self):
        # Test compute_false_positive_rate_cumulative_list function
        result = compute_false_positive_rate_cumulative_list(self.fp, self.tn)
        # Expected cumulative values:
        # fp cumsum: [0, 1, 1, 2, 2]
        # tn cumsum: [0, 1, 2, 2, 3]
        # false_positive_rate = fp / (fp + tn)
        # First value will be NaN because division by zero (0/0)
        expected = pd.Series([0.0, 0.5, 0.333, 0.5, 0.4])
        # Replace NaN with 0.0 in the result for comparison
        result = result.fillna(0.0)
        pd.testing.assert_series_equal(result.round(3), expected.round(3))

    def test_compute_false_positive_rate_score(self):
        # Test compute_false_positive_rate_score function
        result = compute_false_positive_rate_score(self.fp, self.tn)
        # Final value from cumulative false positive rate
        expected = 0.4
        self.assertAlmostEqual(result.item(), expected, places=1)

    def test_compute_f1_score_cumulative_list(self):
        # Test compute_f1_score_cumulative_list function
        result = compute_f1_score_cumulative_list(self.tp, self.fp, self.fn)
        # Expected using precision and recall calculated above
        # f1 = 2 * precision * recall / (precision + recall)
        # Using the precision and recall values computed in earlier tests
        precision = pd.Series([1.0, 0.5, 0.667, 0.5, 0.6])
        recall = pd.Series([0.5, 0.5, 0.667, 0.5, 0.6])
        expected = 2 * precision * recall / (precision + recall)
        pd.testing.assert_series_equal(result.round(3), expected.round(3))

    def test_compute_f1_score_score(self):
        # Test compute_f1_score_score function
        result = compute_f1_score_score(self.tp, self.fp, self.fn)
        # Final F1 score using precision=0.6 and recall=0.6
        expected = 2 * 0.6 * 0.6 / (0.6 + 0.6)
        self.assertAlmostEqual(result.item(), expected, places=1)

    def test_compute_error_rate_cumulative_list(self):
        # Test compute_error_rate_cumulative_list function
        result = compute_error_rate_cumulative_list(self.fp)
        # Error rate is cumulative average of false positives
        # fp: [0, 1, 0, 1, 0]
        # cumulative avg: [0/1, 1/2, 1/3, 2/4, 2/5]
        expected = pd.Series([0.0, 0.5, 0.333, 0.5, 0.4])
        pd.testing.assert_series_equal(result.round(3), expected.round(3))

    def test_compute_error_rate_score(self):
        # Test compute_error_rate_score function
        result = compute_error_rate_score(self.fp)
        # Final error rate
        expected = 0.4
        self.assertAlmostEqual(result.item(), expected, places=1)

    def test_compute_cache_hit_rate_cumulative_list(self):
        # Test compute_cache_hit_rate_cumulative_list function
        result = compute_cache_hit_rate_cumulative_list(self.cache_hits)
        # Cumulative average of cache hits
        # cache_hits: [0, 1, 1, 0, 1]
        # cumulative avg: [0/1, 1/2, 2/3, 2/4, 3/5]
        expected = pd.Series([0.0, 0.5, 0.667, 0.5, 0.6])
        pd.testing.assert_series_equal(result.round(3), expected.round(3))

    def test_compute_cache_hit_rate_score(self):
        # Test compute_cache_hit_rate_score function
        result = compute_cache_hit_rate_score(self.cache_hits)
        # Final cache hit rate
        expected = 0.6
        self.assertAlmostEqual(result.item(), expected, places=1)

    def test_compute_duration_cumulative_list(self):
        # Test compute_duration_cumulative_list function
        result = compute_duration_cumulative_list(self.latencies)
        # Cumulative sum of latencies
        # latencies: [0.5, 1.0, 0.2, 1.5, 0.8]
        # cumsum: [0.5, 1.5, 1.7, 3.2, 4.0]
        expected = pd.Series([0.5, 1.5, 1.7, 3.2, 4.0])
        pd.testing.assert_series_equal(result.round(3), expected.round(3))

    def test_compute_duration_score(self):
        # Test compute_duration_score function
        result = compute_duration_score(self.latencies)
        # Sum of all latencies
        expected = 4.0
        self.assertAlmostEqual(result.item(), expected, places=1)

    def test_compute_avg_latency_score(self):
        # Test compute_avg_latency_score function
        result = compute_avg_latency_score(self.latencies)
        # Average of latencies
        expected = (0.5 + 1.0 + 0.2 + 1.5 + 0.8) / 5
        self.assertAlmostEqual(result.item(), expected, places=1)


if __name__ == "__main__":
    unittest.main()
