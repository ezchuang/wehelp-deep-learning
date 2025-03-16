import numpy as np
from abc import ABC, abstractmethod
from enum import Enum


class ActivationFunction(ABC):
    @abstractmethod
    def __call__(self, x):
        pass

class Linear(ActivationFunction):
    def __call__(self, x: np.array) -> np.array:
        return x

class ReLU(ActivationFunction):
    def __call__(self, x: np.array) -> np.array:
        return np.maximum(0, x)

class Sigmoid(ActivationFunction):
    def __call__(self, x: np.array) -> np.array:
        return 1 / (1 + np.exp(-x))

class Softmax(ActivationFunction):
    def __call__(self, x: np.array) -> np.array:
        exp_x = np.exp(x - np.max(x))
        return exp_x / np.sum(exp_x)

class ActivationFunctions(Enum):
    LINEAR = Linear
    RELU = ReLU
    SIGMOID = Sigmoid
    SOFTMAX = Softmax

    def _get_instance(self) -> ActivationFunction:
        return self.value()
    
    def __call__(self, *args, **kwargs) -> np.array:
        instance = self._get_instance()
        return instance(*args, **kwargs)