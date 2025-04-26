def answers_have_same_meaning_static(answer_a, answer_b):
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
