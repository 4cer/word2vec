from abc import abstractmethod, ABC
import numpy as np


from com.layer import Layer


class Layer(ABC):
    @property
    def weights(self) -> np.ndarray:
        return self._weights
    
    @property
    def biases(self) -> np.ndarray | None:
        return self._biases
    
    @property
    def bias(self) -> float | None:
        return self._bias
    
    @property
    def nonlinearity(self) -> Layer | None:
        return self._nonlinearity
    
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
        nonlinearity: Layer
    ): ...


class FullyConnected(Layer):
    ...
