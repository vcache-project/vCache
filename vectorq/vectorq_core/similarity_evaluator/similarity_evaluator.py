from abc import ABC, abstractmethod

class SimilarityEvaluator(ABC):
    
    @abstractmethod
    def answers_similar(self, a: str, b: str) -> bool:
        '''
        a: str - The first answer
        b: str - The second answer
        returns: bool - True if the answers are similar, False otherwise
        '''
        pass
