import logging

from vcache.inference_engine.strategies.open_ai import OpenAIInferenceEngine


def answers_have_same_meaning_static(answer_a, answer_b):
    """
    Compare two answers to determine if they have the same meaning using static string comparison.

    Args:
        answer_a: The first answer to compare.
        answer_b: The second answer to compare.

    Returns:
        True if the answers have the same meaning, False otherwise.
    """
    answer_a = (
        str(answer_a)
        .strip()
        .rstrip(".")
        .lower()
        .replace(".", "")
        .replace(",", "")
        .replace('"', "")
        .replace("'", "")
        .replace("[", "")
        .replace("]", "")
    )
    answer_b = (
        str(answer_b)
        .strip()
        .rstrip(".")
        .lower()
        .replace(".", "")
        .replace(",", "")
        .replace('"', "")
        .replace("'", "")
        .replace("[", "")
        .replace("]", "")
    )

    answers_have_same_len = len(answer_a.split()) == len(answer_b.split())
    if answers_have_same_len:
        return answer_a == answer_b
    else:
        return False


def answers_have_same_meaning_llm(answer_a, answer_b):
    """
    Compare two answers to determine if they have the same meaning using LLM-based comparison.

    Args:
        answer_a: The first answer to compare.
        answer_b: The second answer to compare.

    Returns:
        True if the answers have the same meaning, False otherwise.
    """
    inference_engine = OpenAIInferenceEngine(
        model_name="gpt-4.1-nano-2025-04-14", temperature=0.0
    )
    system_prompt: str = "You are a judge evaluating whether two answers are semantically equivalent. Respond with only 'YES' if they convey the same meaning, or 'NO' if they differ significantly."
    user_prompt: str = f"Answer 1: {answer_a}\n\nAnswer 2: {answer_b}\n\nAre these answers semantically equivalent?"

    try:
        response: str = (
            inference_engine.create(user_prompt, system_prompt).strip().upper()
        )
        return "YES" in response
    except Exception as e:
        logging.warning(f"Error in LLM comparison: {e}. Returning False.")
        return False
