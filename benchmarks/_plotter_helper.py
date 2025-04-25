from typing import TYPE_CHECKING, Any

import pandas as pd

if TYPE_CHECKING:
    from benchmarks.benchmark import Benchmark


def convert_to_dataframe_from_benchmark(benchmark: "Benchmark") -> tuple:
    data = {
        "cache_hit_acc_list": benchmark.cache_hit_acc_list,
        "cache_miss_acc_list": benchmark.cache_miss_acc_list,
        "true_positive_acc_list": benchmark.true_positive_acc_list,
        "false_positive_acc_list": benchmark.false_positive_acc_list,
        "true_negative_acc_list": benchmark.true_negative_acc_list,
        "false_negative_acc_list": benchmark.false_negative_acc_list,
        "latency_direct_list": benchmark.latency_direct_list,
        "latency_vectorq_list": benchmark.latency_vectorq_list,
    }
    df = pd.DataFrame(data)

    metadata = {
        "observations_dict": benchmark.observations_dict,
        "gammas_dict": benchmark.gammas_dict,
    }

    return df, metadata


def convert_to_dataframe_from_json_file(json_data: Any) -> tuple:
    data = {
        "cache_hit_acc_list": json_data["cache_hit_acc_list"],
        "cache_miss_acc_list": json_data["cache_miss_acc_list"],
        "true_positive_acc_list": json_data["true_positive_acc_list"],
        "false_positive_acc_list": json_data["false_positive_acc_list"],
        "true_negative_acc_list": json_data["true_negative_acc_list"],
        "false_negative_acc_list": json_data["false_negative_acc_list"],
        "latency_direct_list": json_data["latency_direct_list"],
        "latency_vectorq_list": json_data["latency_vectorq_list"],
    }
    df = pd.DataFrame(data)

    metadata = {
        "observations_dict": json_data["observations_dict"],
        "gammas_dict": json_data["gammas_dict"],
    }

    return df, metadata


def compute_accuracy_acc_list(
    tp: pd.DataFrame, fp: pd.DataFrame, tn: pd.DataFrame, fn: pd.DataFrame
) -> pd.DataFrame:
    numerator = tp + tn
    denominator = tp + tn + fp + fn
    accuracy = numerator / denominator
    return accuracy


def compute_precision_acc_list(tp: pd.DataFrame, fp: pd.DataFrame) -> pd.DataFrame:
    denominator = tp + fp
    precision = tp / denominator
    return precision


def compute_recall_acc_list(tp: pd.DataFrame, fn: pd.DataFrame) -> pd.DataFrame:
    denominator = tp + fn
    recall = tp / denominator
    return recall


def compute_false_positive_rate_acc_list(
    fp: pd.DataFrame, tn: pd.DataFrame
) -> pd.DataFrame:
    denominator = fp + tn
    false_positive_rate = fp / denominator
    return false_positive_rate


def compute_f1_score_acc_list(
    tp: pd.DataFrame, fp: pd.DataFrame, fn: pd.DataFrame
) -> pd.DataFrame:
    precision = compute_precision_acc_list(tp, fp)
    recall = compute_recall_acc_list(tp, fn)

    numerator = 2 * precision * recall
    denominator = precision + recall
    f1_score = numerator / denominator
    return f1_score


def compute_error_rate_acc_list(
    tp: pd.DataFrame, fp: pd.DataFrame, tn: pd.DataFrame, fn: pd.DataFrame
) -> pd.DataFrame:
    denominator = tp + tn + fp + fn
    error_rate = fp / denominator
    return error_rate


def compute_cache_hit_rate_acc_list(
    cache_hit_list: pd.DataFrame, cache_miss_list: pd.DataFrame
) -> pd.DataFrame:
    denominator = cache_hit_list + cache_miss_list
    cache_hit_rate = cache_hit_list / denominator
    return cache_hit_rate


def compute_duration_acc_list(latency_list: pd.DataFrame) -> pd.DataFrame:
    return latency_list.cumsum()


def compute_accuracy_score(
    tp: pd.DataFrame, fp: pd.DataFrame, tn: pd.DataFrame, fn: pd.DataFrame
) -> float:
    accuracy = compute_accuracy_acc_list(tp, fp, tn, fn)
    return accuracy.iloc[-1]


def compute_precision_score(tp: pd.DataFrame, fp: pd.DataFrame) -> float:
    precision = compute_precision_acc_list(tp, fp)
    return precision.iloc[-1]


def compute_recall_score(tp: pd.DataFrame, fn: pd.DataFrame) -> float:
    recall = compute_recall_acc_list(tp, fn)
    return recall.iloc[-1]


def compute_false_positive_rate_score(fp: pd.DataFrame, tn: pd.DataFrame) -> float:
    false_positive_rate = compute_false_positive_rate_acc_list(fp, tn)
    return false_positive_rate.iloc[-1]


def compute_f1_score_score(
    tp: pd.DataFrame, fp: pd.DataFrame, fn: pd.DataFrame
) -> float:
    f1_score = compute_f1_score_acc_list(tp, fp, fn)
    return f1_score.iloc[-1]


def compute_avg_latency_score(latency_list: pd.DataFrame) -> float:
    return latency_list.mean()


def compute_error_rate_score(
    tp: pd.DataFrame, fp: pd.DataFrame, tn: pd.DataFrame, fn: pd.DataFrame
) -> float:
    error_rate = compute_error_rate_acc_list(tp, fp, tn, fn)
    return error_rate.iloc[-1]


def compute_cache_hit_rate_score(
    cache_hit_list_acc: pd.DataFrame, cache_miss_list_acc: pd.DataFrame
) -> float:
    hit_rate = cache_hit_list_acc.iloc[-1]
    miss_rate = cache_miss_list_acc.iloc[-1]
    return hit_rate / (hit_rate + miss_rate)


def compute_cache_miss_rate_score(
    cache_miss_list_acc: pd.DataFrame, cache_hit_list_acc: pd.DataFrame
) -> float:
    hit_rate = cache_hit_list_acc.iloc[-1]
    miss_rate = cache_miss_list_acc.iloc[-1]
    return miss_rate / (hit_rate + miss_rate)

def compute_duration_score(latency_list: pd.DataFrame) -> float:
    return latency_list.sum()
