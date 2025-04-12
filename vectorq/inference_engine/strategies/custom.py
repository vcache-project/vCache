from typing import TYPE_CHECKING, Dict, Any

if TYPE_CHECKING:
    from vectorq.config import VectorQConfig
    
from vectorq.inference_engine.strategy import InferenceEngineStrategy

class Custom(InferenceEngineStrategy):
    
    def __init__(self, vectorq_config: "VectorQConfig"):
        super().__init__(vectorq_config=vectorq_config)
    
    def create(self, prompt: str, output_format: str = None) -> str:
        # TODO
        return ""
