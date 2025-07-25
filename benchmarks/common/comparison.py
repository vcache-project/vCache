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

    system_prompt: str = """
    You are an expert judge evaluating whether two answers are semantically equivalent for caching purposes. 

    Your task is to determine if two answers convey essentially the same meaning, even if they use different words, phrasing, or structure.

    GUIDELINES:
    - Focus on semantic meaning rather than exact wording
    - Consider answers equivalent if they provide the same information
    - Minor differences in phrasing, word choice, or formatting should be ignored
    - Different examples that illustrate the same concept should be considered equivalent
    - Answers with the same conclusion but different reasoning paths may be equivalent

    EXAMPLE 1:
    Answer 1: "The capital of France is Paris, which is located in the northern part of the country."
    Answer 2: "Paris is the capital city of France."
    Evaluation: YES

    EXAMPLE 2:
    Answer 1: "To solve this equation, multiply both sides by 2 to get x = 10."
    Answer 2: "The solution is x = 5 after dividing both sides by 2."
    Evaluation: NO
    Respond with only "YES" if the answers are semantically equivalent, or "NO" if they differ significantly in meaning."""

    user_prompt: str = f"""
    Answer 1: {answer_a}
    Answer 2: {answer_b}

    Are these answers semantically equivalent?"""

    try:
        response: str = (
            inference_engine.create(user_prompt, system_prompt).strip().upper()
        )
        return "YES" in response
    except Exception as e:
        logging.warning(f"Error in LLM comparison: {e}. Returning False.")
        return False
