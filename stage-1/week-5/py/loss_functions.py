from abc import ABC, abstractmethod
from enum import Enum
import numpy as np


class LossFunction(ABC):
    @abstractmethod
    def forward(self, expect, output):
        pass
    
    @abstractmethod
    def derivative(self, output: np.array, expect: np.array) -> np.array:
        pass

class MSE(LossFunction):
    def forward(self, output: np.array, expect: np.array) -> float:
        return np.mean(np.square(expect - output))
    
    def derivative(self, output: np.array, expect: np.array) -> np.array:
        return 2 * (output - expect) / output.size

class BinaryCrossEntropy(LossFunction):
    def forward(self, output: np.array, expect: np.array) -> float:
        return -np.sum(expect * np.log(output) + (1 - expect) * np.log(1 - output))
    
    def derivative(self, output: np.array, expect: np.array) -> np.array:
        return (1 - expect) / (1 - output) - expect / output

class CategoricalCrossEntropy(LossFunction):
    def forward(self, output: np.array, expect: np.array) -> float:
        return -np.sum(expect * np.log(output))
    
    def derivative(self, output: np.array, expect: np.array) -> np.array:
        pass

class LossFunctions(Enum):
    MSE = MSE
    BINARYCROSSENTROPY = BinaryCrossEntropy
    CATEGORICALCROSSENTROPY = CategoricalCrossEntropy

    def _get_instance(self) -> LossFunction:
        return self.value()
    
    def forward(self, *args, **kwargs) -> float:
        instance = self._get_instance()
        return instance.forward(*args, **kwargs)
    
    def derivative(self, *args, **kwargs) -> np.array:
        instance = self._get_instance()
        return instance.derivative(*args, **kwargs)