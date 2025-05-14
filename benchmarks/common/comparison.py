import os
import openai

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


def answers_have_same_meaning_llm(answer_a, answer_b):
    client = openai.OpenAI() # Assumes API key is set in environment variables
    system_message_content = "You are an expert in determining if two answers are semantically similar."
    user_prompt_content = f"""
Please compare the following two answers:
<answer_a>
{answer_a}
</answer_a>
<answer_b>
{answer_b}
</answer_b>

Example:
<example>
<answer_a>
The answer is 
</answer_a>
<answer_b>
The answer is 1.
</answer_b>
Are these two answers semantically similar?
IMPORTANT: Respond ONLY with "SIMILAR" or "DIFFERENT".
"""
    try:
        response = client.chat.completions.create(
            model="gpt-4.1-nano-2025-04-14", # Or your specific model for "4.1 nano"
            messages=[
                {"role": "system", "content": system_message_content},
                {"role": "user", "content": user_prompt_content}
            ],
            temperature=0, # For deterministic output
        )
        llm_output = response.choices[0].message.content
        # Extract the last line for the final decision
        response_lines = llm_output.strip().split('\n')
        final_decision = response_lines[-1].strip().upper()

        if "SIMILAR" in final_decision:
            return True
        elif "DIFFERENT" in final_decision:
            return False
        else:
            print(f"LLM similarity check returned unexpected final decision: '{final_decision}' in response: '{llm_output}'. Defaulting to False.")
            return False
    except Exception as e:
        print(f"Error during OpenAI API call: {e}. Defaulting to False.")
        return False
