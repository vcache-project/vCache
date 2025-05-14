from textwrap import dedent
from vectorq.vectorq_core.similarity_evaluator.similarity_evaluator import (
    SimilarityEvaluator,
)
import openai


class LLMComparisonSimilarityEvaluator(SimilarityEvaluator):
    def __init__(self):
        super().__init__()

    def answers_similar(self, a: str, b: str) -> bool:
        return a == b
        client = openai.OpenAI() # Assumes API key is set in environment variables
        system_message_content = "You are an expert in determining if two answers are semantically similar."
        user_prompt_content = f"""\
            Please compare the following two answers:
            Here are some examples of how to determine semantic similarity:

            Example 1:
            <answer_a>
            The quick brown fox jumps over the lazy dog. This sentence is a classic pangram, meaning it contains every letter of the English alphabet. It's often used for testing typewriters and keyboards.
            </answer_a>
            <answer_b>
            A pangram is a sentence using every letter of a given alphabet at least once. The most famous example in English is "The quick brown fox jumps over the lazy dog," which is commonly utilized for font and keyboard tests.
            </answer_b>
            Are these two answers semantically similar?
            SIMILAR

            Example 2:
            <answer_a>
            The capital of France is Paris, a major European city and a global center for art, fashion, gastronomy and culture. Its 19th-century cityscape is crisscrossed by wide boulevards and the River Seine.
            </answer_a>
            <answer_b>
            Berlin, Germany's capital, dates to the 13th century. Reminders of the city's turbulent 20th-century history include its Holocaust memorial and the Berlin Wall's graffitied remains.
            </answer_b>
            Are these two answers semantically similar?
            RESPONSE:
            DIFFERENT

            <answer_a>
            {a}
            </answer_a>
            <answer_b>
            {b}
            </answer_b>

            Now, based on the examples, are the original two answers semantically similar?
            IMPORTANT: Respond ONLY with "SIMILAR" or "DIFFERENT".
            """
        user_prompt_content = dedent(user_prompt_content)
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
