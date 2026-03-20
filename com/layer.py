from abc import abstractmethod, ABC
import numpy as np


from nonlinearity import NonLinearity


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
    def nonlinearity(self) -> NonLinearity | None:
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
        nonlinearity: NonLinearity
    ): ...


class FullyConnected(Layer):
    ...
