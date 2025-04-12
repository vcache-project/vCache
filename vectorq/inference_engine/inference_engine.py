from enum import Enum
from typing import TYPE_CHECKING, List

if TYPE_CHECKING:
    from vectorq.config import VectorQConfig

from vectorq.inference_engine.strategy import InferenceEngineStrategy
from vectorq.inference_engine.strategies.lang_chain import LangChain
from vectorq.inference_engine.strategies.open_ai import OpenAI

class InferenceEngineType(Enum):
    LANGCHAIN = "langchain"
    OPENAI = "openai"
    DUMMY = "dummy" 

class InferenceEngine():
    
    def __init__(self, vectorq_config: "VectorQConfig"):
        self.temperature: int = vectorq_config._inference_engine_temperature
        self._inference_engine_type: InferenceEngineType = vectorq_config._inference_engine_type
        self.strategy: InferenceEngineStrategy = None
        
        try:
            match self._inference_engine_type:
                case InferenceEngineType.LANGCHAIN:
                    self.strategy = LangChain(vectorq_config=vectorq_config)
                case InferenceEngineType.OPENAI:
                    self.strategy = OpenAI(vectorq_config=vectorq_config)
                case InferenceEngineType.DUMMY:
                    self.strategy = DummyStrategy(vectorq_config=vectorq_config)
                case _:
                    raise ValueError(f"Invalid inference engine type")
        except Exception as e:
            raise Exception(f"Error initializing inference engine with type: {self._inference_engine_type}: {e}")
            
    def create(self, prompt: str, output_format: str = None) -> str:
        if self.strategy is None:
            return f"[ERROR] No inference strategy available for: {prompt}"
        return self.strategy.create(prompt, output_format)

class DummyStrategy(InferenceEngineStrategy):
    """A simple fallback strategy that doesn't require any dependencies"""
    
    def __init__(self, vectorq_config: "VectorQConfig"):
        super().__init__(vectorq_config=vectorq_config)
    
    def create(self, prompt: str, output_format: str = None) -> str:
        return f"[DUMMY RESPONSE] This is a placeholder response for: '{prompt}'"
