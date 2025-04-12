from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Dict, Any

if TYPE_CHECKING:
    from vectorq.config import VectorQConfig
    
class InferenceEngineStrategy(ABC):
    """
    Abstract base class for inference engines
    """
    
    def __init__(self, vectorq_config: "VectorQConfig"):
        self.vectorq_config: "VectorQConfig" = vectorq_config
    
    @abstractmethod
    def create(self, prompt: str, output_format: str = None) -> str:
        '''
        prompt: str - The prompt to create an answer for
        output_format: str - The optional output format to use for the response
        returns: str - The answer to the prompt
        '''
        pass
