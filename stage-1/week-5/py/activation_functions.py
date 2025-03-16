import numpy as np
from abc import ABC, abstractmethod
from enum import Enum


class ActivationFunction(ABC):
    @abstractmethod
    def forward(self, x: np.array) -> np.array:
        pass

    @abstractmethod
    def derivative(self, x: np.array) -> np.array:
        pass

class Linear(ActivationFunction):
    def forward(self, x: np.array) -> np.array:
        return x

    def derivative(self, x: np.array) -> np.array:
        return np.ones_like(x)

class ReLU(ActivationFunction):
    def forward(self, x: np.array) -> np.array:
        return np.maximum(0, x)
    
    def derivative(self, x: np.array) -> np.array:
        return (x > 0).astype(x.dtype)

class Sigmoid(ActivationFunction):
    def forward(self, x: np.array) -> np.array:
        return 1 / (1 + np.exp(-x))
    
    def derivative(self, x: np.array) -> np.array:
        s = self.forward(x)
        return s * (1 - s)

class Softmax(ActivationFunction):
    def forward(self, x: np.array) -> np.array:
        exp_x = np.exp(x - np.max(x))
        return exp_x / np.sum(exp_x)
    
    def derivative(self, x: np.array) -> np.array:
        pass

class ActivationFunctions(Enum):
    LINEAR = Linear
    RELU = ReLU
    SIGMOID = Sigmoid
    SOFTMAX = Softmax

    def _get_instance(self) -> ActivationFunction:
        return self.value()
    
    def forward(self, *args, **kwargs) -> np.array:
        instance = self._get_instance()
        return instance.forward(*args, **kwargs)
    
    def derivative(self, *args, **kwargs) -> np.array:
        instance = self._get_instance()
        return instance.derivative(*args, **kwargs)