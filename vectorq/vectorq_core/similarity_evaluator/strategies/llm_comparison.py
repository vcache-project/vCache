from vectorq.vectorq_core.similarity_evaluator.similarity_evaluator import (
    SimilarityEvaluator,
)
from vectorq.inference_engine.inference_engine import InferenceEngine
import logging

logger = logging.getLogger(__name__)


class LLMComparisonSimilarityEvaluator(SimilarityEvaluator):
    def __init__(self, inference_engine: InferenceEngine):
        super().__init__()
        if inference_engine is None:
            raise ValueError("InferenceEngine cannot be None for LLMComparisonSimilarityEvaluator")
        self.inference_engine = inference_engine

    def answers_similar(self, a: str, b: str) -> bool:
        system_message = "You are an expert in determining if two answers are semantically similar."
        user_prompt = f"""
Please compare the following two answers:

<answer_a>
{a}
</answer_a>

<answer_b>
{b}
</answer_b>

Are these two answers semantically similar? 
IMPORTANT: Respond ONLY with "SIMILAR" or "DIFFERENT".
"""
        try:
            response = self.inference_engine.create(
                output_format=system_message,
                prompt=user_prompt
            )
            # Extract the last line for the final decision
            response_lines = response.strip().split('\n')
            final_decision = response_lines[-1].strip().upper()


            if "SIMILAR" in final_decision:
                return True
            elif "DIFFERENT" in final_decision:
                return False
            else:
                logger.warning(f"LLM similarity check returned unexpected final decision: '{final_decision}' in response: '{response}'. Defaulting to False.")
                return False
        except Exception as e:
            logger.error(f"Error during LLM similarity check: {e}. Defaulting to False.")
            return False
