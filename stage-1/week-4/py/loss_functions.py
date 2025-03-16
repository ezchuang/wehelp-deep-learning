from abc import ABC, abstractmethod
from enum import Enum

import numpy as np


class LossFunction(ABC):
    @abstractmethod
    def __call__(self, expect, output):
        pass

class MSE(LossFunction):
    def __call__(self, output: np.array, expect: np.array) -> float:
        return np.mean(np.square(expect - output))

class BinaryCrossEntropy(LossFunction):
    def __call__(self, output: np.array, expect: np.array) -> float:
        return -np.sum(expect * np.log(output) + (1 - expect) * np.log(1 - output))

class CategoricalCrossEntropy(LossFunction):
    def __call__(self, output: np.array, expect: np.array) -> float:
        return -np.sum(expect * np.log(output))

class LossFunctions(Enum):
    MSE = MSE
    BINARYCROSSENTROPY = BinaryCrossEntropy
    CATEGORICALCROSSENTROPY = CategoricalCrossEntropy

    def _get_instance(self) -> LossFunction:
        return self.value()
    
    def __call__(self, *args, **kwargs) -> float:
        instance = self._get_instance()
        return instance(*args, **kwargs)