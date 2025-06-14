from typing import TYPE_CHECKING, Any, Tuple

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from benchmarks.benchmark import Benchmark


###################################################################################
### Conversion Functions ##########################################################
###################################################################################
def convert_to_dataframe_from_benchmark(benchmark: "Benchmark") -> tuple:
    data = {
        "cache_hit_list": benchmark.cache_hit_list,
        "cache_miss_list": benchmark.cache_miss_list,
        "tp_list": benchmark.tp_list,
        "fp_list": benchmark.fp_list,
        "tn_list": benchmark.tn_list,
        "fn_list": benchmark.fn_list,
        "latency_direct_list": benchmark.latency_direct_list,
        "latency_vectorq_list": benchmark.latency_vcache_list,
    }
    df = pd.DataFrame(data)

    metadata = {
        "observations_dict": benchmark.observations_dict,
        "gammas_dict": benchmark.gammas_dict,
    }

    return df, metadata


def convert_to_dataframe_from_json_file(
    json_data: Any, keep_split: int = 100
) -> Tuple[pd.DataFrame, dict, int]:
    """
    Convert the json data to a dataframe.
    Args:
        json_data: Any - The json data to convert.
        keep_split: int - The percentage of the data to keep.
        For example, if keep_split is 20, the benchmark will keep the last 20% of the data.
        keep_split ∈ (0, 100]
    Returns:
        df: pd.DataFrame - The dataframe.
        metadata: dict - The metadata.
        chopped_index: int - The index of the data that was chopped.
    """

    cache_hit_list = json_data["cache_hit_list"]
    cache_miss_list = json_data["cache_miss_list"]
    tp_list = json_data["tp_list"]
    fp_list = json_data["fp_list"]
    tn_list = json_data["tn_list"]
    fn_list = json_data["fn_list"]
    latency_direct_list = json_data["latency_direct_list"]
    latency_vcache_list = json_data["latency_vectorq_list"]

    chopped_index = 0
    if keep_split > 0 and keep_split < 100:
        chopped_index = int(len(cache_hit_list) * (100 - keep_split) / 100)
        cache_hit_list = cache_hit_list[chopped_index:]
        cache_miss_list = cache_miss_list[chopped_index:]
        tp_list = tp_list[chopped_index:]
        fp_list = fp_list[chopped_index:]
        tn_list = tn_list[chopped_index:]
        fn_list = fn_list[chopped_index:]
        latency_direct_list = latency_direct_list[chopped_index:]
        latency_vcache_list = latency_vcache_list[chopped_index:]

    data = {
        "cache_hit_list": cache_hit_list,
        "cache_miss_list": cache_miss_list,
        "tp_list": tp_list,
        "fp_list": fp_list,
        "tn_list": tn_list,
        "fn_list": fn_list,
        "latency_direct_list": latency_direct_list,
        "latency_vectorq_list": latency_vcache_list,
    }
    df = pd.DataFrame(data)

    metadata = {
        "observations_dict": json_data["observations_dict"],
        "gammas_dict": json_data["gammas_dict"],
    }

    return df, metadata, chopped_index


###################################################################################
### Stat Functions ################################################################
###################################################################################
def __cumulative_average_stats(data: pd.DataFrame) -> pd.DataFrame:
    """
    Compute the cumulative average stats of <data>.
    Args:
        data: pd.DataFrame - Data [0, 1, 2, 3, 4, 5, ...]
    Returns:
        cumulative_data: pd.DataFrame - Cumulative Data [0/1, 1/2, 3/3, 6/4, 10/5, 15/6, ...]
    Example:
        data = [0.5, 1.0, 1.0, 0.0, ...] # Accuracy
        cumulative_data = [0.5/1, 1.5/2, 2.5/3, 2.5/4, 2.5/5, 3.5/6, ...]
    """
    return data.cumsum() / np.arange(1, len(data) + 1)


def compute_accuracy_cumulative_list(
    tp: pd.DataFrame, fp: pd.DataFrame, tn: pd.DataFrame, fn: pd.DataFrame
) -> pd.DataFrame:
    """
    Compute the entry-wise accuracy. The function accumulates the values of the true positives,
    true negatives, false positives, and false negatives. Afterwards, it computes the accuracy.
    Args:
        tp: pd.DataFrame - True Positives  [0, 1, 0, 0, ...]
        fp: pd.DataFrame - False Positives [1, 0, 0, 0, ...]
        tn: pd.DataFrame - True Negatives  [1, 0, 1, 0, ...]
        fn: pd.DataFrame - False Negatives [0, 0, 0, 0, ...]
    Returns:
        accuracy: pd.DataFrame - Accuracy [0.xx, 0.xx, 0.xx, 0.xx, ...]
    """
    tp = tp.cumsum()
    tn = tn.cumsum()
    fp = fp.cumsum()
    fn = fn.cumsum()
    numerator = tp + tn
    denominator = tp + tn + fp + fn
    accuracy = numerator / denominator
    return accuracy


def compute_accuracy_score(
    tp: pd.DataFrame, fp: pd.DataFrame, tn: pd.DataFrame, fn: pd.DataFrame
) -> float:
    """
    Compute the final accuracy score. The function accumulates the values of the true positives,
    true negatives, false positives, and false negatives. Afterwards, it computes the accuracy and
    returns the last value of the accuracy.
    Args:
        tp: pd.DataFrame - True Positives  [0, 1, 0, 0, ...]
        fp: pd.DataFrame - False Positives [1, 0, 0, 0, ...]
        tn: pd.DataFrame - True Negatives  [1, 0, 1, 0, ...]
        fn: pd.DataFrame - False Negatives [0, 0, 0, 0, ...]
    Returns:
        accuracy: float - Accuracy 0.xx
    """
    accuracy = compute_accuracy_cumulative_list(tp=tp, fp=fp, tn=tn, fn=fn)
    return accuracy.iloc[-1]


def compute_precision_cumulative_list(
    tp: pd.DataFrame, fp: pd.DataFrame
) -> pd.DataFrame:
    """
    Compute the entry-wise precision. The function accumulates the values of the true positives and
    false positives. Afterwards, it computes the precision.
    Args:
        tp: pd.DataFrame - True Positives  [0, 1, 1, 0, ...]
        fp: pd.DataFrame - False Positives [1, 0, 1, 0, ...]
    Returns:
        precision: pd.DataFrame - Precision [0.xx, 0.xx, 0.xx, 0.xx, ...]
    """
    tp = tp.cumsum()
    fp = fp.cumsum()
    denominator = tp + fp
    precision = tp / denominator
    return precision


def compute_precision_score(tp: pd.DataFrame, fp: pd.DataFrame) -> float:
    """
    Compute the final precision score. The function accumulates the values of the true positives and
    false positives. Afterwards, it computes the precision and returns the last value of the precision.
    Args:
        tp: pd.DataFrame - True Positives  [0, 1, 0, 0, ...]
        fp: pd.DataFrame - False Positives [1, 0, 0, 0, ...]
    Returns:
        precision: float - Precision 0.xx
    """
    precision = compute_precision_cumulative_list(tp=tp, fp=fp)
    return precision.iloc[-1]


def compute_recall_cumulative_list(tp: pd.DataFrame, fn: pd.DataFrame) -> pd.DataFrame:
    """
    Compute the entry-wise recall. The function accumulates the values of the true positives and
    false negatives. Afterwards, it computes the recall.
    Args:
        tp: pd.DataFrame - True Positives  [0, 1, 1, 0, ...]
        fn: pd.DataFrame - False Negatives [1, 0, 1, 0, ...]
    Returns:
        recall: pd.DataFrame - Recall [0.xx, 0.xx, 0.xx, 0.xx, ...]
    """
    tp = tp.cumsum()
    fn = fn.cumsum()
    denominator = tp + fn
    recall = tp / denominator
    return recall


def compute_recall_score(tp: pd.DataFrame, fn: pd.DataFrame) -> float:
    """
    Compute the final recall score. The function accumulates the values of the true positives and
    false negatives. Afterwards, it computes the recall and returns the last value of the recall.
    Args:
        tp: pd.DataFrame - True Positives  [0, 1, 0, 0, ...]
        fn: pd.DataFrame - False Negatives [1, 0, 1, 0, ...]
    Returns:
        recall: float - Recall 0.xx
    """
    recall = compute_recall_cumulative_list(tp=tp, fn=fn)
    return recall.iloc[-1]


def compute_false_positive_rate_cumulative_list(
    fp: pd.DataFrame, tn: pd.DataFrame
) -> pd.DataFrame:
    """
    Compute the entry-wise false positive rate. The function accumulates the values of the false positives and
    true negatives. Afterwards, it computes the false positive rate.
    Args:
        fp: pd.DataFrame - False Positives [0, 1, 1, 0, ...]
        tn: pd.DataFrame - True Negatives  [1, 0, 1, 0, ...]
    Returns:
        false_positive_rate: pd.DataFrame - False Positive Rate [0.xx, 0.xx, 0.xx, 0.xx, ...]
    """
    fp = fp.cumsum()
    tn = tn.cumsum()
    denominator = fp + tn
    false_positive_rate = fp / denominator
    return false_positive_rate


def compute_false_positive_rate_score(fp: pd.DataFrame, tn: pd.DataFrame) -> float:
    """
    Compute the final false positive rate score. The function accumulates the values of the false positives and
    true negatives. Afterwards, it computes the false positive rate and returns the last value of the false positive rate.
    Args:
        fp: pd.DataFrame - False Positives [0, 1, 1, 0, ...]
        tn: pd.DataFrame - True Negatives  [1, 0, 1, 0, ...]
    Returns:
        false_positive_rate: float - False Positive Rate 0.xx
    """
    false_positive_rate = compute_false_positive_rate_cumulative_list(fp=fp, tn=tn)
    return false_positive_rate.iloc[-1]


def compute_f1_score_cumulative_list(
    tp: pd.DataFrame, fp: pd.DataFrame, fn: pd.DataFrame
) -> pd.DataFrame:
    """
    Compute the entry-wise F1 score. The function accumulates the values of the true positives,
    false positives, and false negatives. Afterwards, it computes the F1 score.
    Args:
        tp: pd.DataFrame - True Positives  [0, 1, 1, 0, ...]
        fp: pd.DataFrame - False Positives [0, 1, 1, 0, ...]
        fn: pd.DataFrame - False Negatives [1, 0, 1, 0, ...]
    Returns:
        f1_score: pd.DataFrame - F1 Score [0.xx, 0.xx, 0.xx, 0.xx, ...]
    """
    precision = compute_precision_cumulative_list(tp=tp, fp=fp)
    recall = compute_recall_cumulative_list(tp=tp, fn=fn)

    numerator = 2 * precision * recall
    denominator = precision + recall
    f1_score = numerator / denominator
    return f1_score


def compute_f1_score_score(
    tp: pd.DataFrame, fp: pd.DataFrame, fn: pd.DataFrame
) -> float:
    """
    Compute the final F1 score. The function accumulates the values of the true positives,
    false positives, and false negatives. Afterwards, it computes the F1 score and returns the last value of the F1 score.
    Args:
        tp: pd.DataFrame - True Positives  [0, 1, 1, 0, ...]
        fp: pd.DataFrame - False Positives [0, 1, 1, 0, ...]
        fn: pd.DataFrame - False Negatives [1, 0, 1, 0, ...]
    Returns:
        f1_score: float - F1 Score 0.xx
    """
    f1_score = compute_f1_score_cumulative_list(tp=tp, fp=fp, fn=fn)
    return f1_score.iloc[-1]


def compute_error_rate_cumulative_list(fp: pd.DataFrame) -> pd.DataFrame:
    """
    Compute the cumulative error rate.
    Args:
        fp: pd.DataFrame - False Positives  [0, 1, 0, 0, 0, 1, ...]
    Returns:
        error_rate: pd.DataFrame - Error Rate [0/1, 1/2, 1/3, 1/4, 1/5, 2/6, ...]
    """
    error_rate = __cumulative_average_stats(data=fp)
    return error_rate


def compute_error_rate_score(fp: pd.DataFrame) -> float:
    """
    Compute the final error rate score.
    Args:
        fp: pd.DataFrame - False Positives [0, 1, 0, 0, 0, 1, ...]
    Returns:
        error_rate: float - Error Rate 0.xx
    """
    error_rate = compute_error_rate_cumulative_list(fp=fp)
    return error_rate.iloc[-1]


def compute_cache_hit_rate_cumulative_list(
    cache_hit_list: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compute the cumulative cache hit rate.
    Args:
        cache_hit_list: pd.DataFrame - Cache Hits [0, 1, 0, 0, 0, 1, ...]
    Returns:
        cache_hit_rate: pd.DataFrame - Cache Hit Rate [0/1, 1/2, 1/3, 1/4, 1/5, 2/6, ...]
    """
    cache_hit_rate = __cumulative_average_stats(data=cache_hit_list)
    return cache_hit_rate


def compute_cache_hit_rate_score(cache_hit_list: pd.DataFrame) -> float:
    """
    Compute the final cache hit rate score.
    Args:
        cache_hit_list: pd.DataFrame - Cache Hits [0, 1, 0, 0, 0, 1, ...]
    Returns:
        cache_hit_rate: float - Cache Hit Rate 0.xx
    """
    cache_hit_rate = compute_cache_hit_rate_cumulative_list(
        cache_hit_list=cache_hit_list
    )
    return cache_hit_rate.iloc[-1]


def compute_duration_cumulative_list(latency_list: pd.DataFrame) -> pd.DataFrame:
    """
    Compute the cumulative duration.
    Args:
        latency_list: pd.DataFrame - Latency [0, 1, 2, 3, 4, 5, ...]
    Returns:
        duration: pd.DataFrame - Duration [0, 1, 3, 6, 10, 15, ...]
    """
    return latency_list.cumsum()


def compute_duration_score(latency_list: pd.DataFrame) -> float:
    """
    Compute the final duration score.
    Args:
        latency_list: pd.DataFrame - Latency [0, 1, 2, 3, 4, 5, ...]
    Returns:
        duration: float - Duration 0.xx
    """
    return latency_list.sum()


def compute_avg_latency_score(latency_list: pd.DataFrame) -> float:
    """
    Compute the final average latency score.
    Args:
        latency_list: pd.DataFrame - Latency [0, 1, 0.5, 2, 1.5, 0.3, ...]
    Returns:
        avg_latency: float - Average Latency 0.xx
    """
    return latency_list.mean()
