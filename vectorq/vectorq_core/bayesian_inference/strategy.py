from abc import ABC, abstractmethod

import numpy as np


class LikelihoodFunctionStrategy(ABC):
    @abstractmethod
    def get_likelihood(self, values: np.ndarray, x_sample: float) -> float:
        pass
