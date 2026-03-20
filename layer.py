from abc import abstractmethod, ABC
import numpy as np


class Layer(ABC):
    @property
    def weights(self) -> np.ndarray:
        return self.weights
    
    @property
    def biases(self) -> np.ndarray | None:
        return self.biases
    
    @property
    def bias(self) -> float | None:
        return self.bias
    
    @property
    def nonlinearity(self) -> function | None:
        return self.nonlinearity
    
    @abstractmethod
    def __init__(self, size: int) -> None: ...

    @abstractmethod
    def forward(self, input) -> np.ndarray: ...

    @abstractmethod
    def back(self, input) -> np.ndarray: ...

    @abstractmethod
    def init_random(self) -> None: ...

    @abstractmethod
    def init_matrix(
        self,
        weights: np.ndarray,
        biases: np.ndarray,
        bias: float,
        nonlinearity: function
    ): ...


class FullyConnected(Layer):
    ...
