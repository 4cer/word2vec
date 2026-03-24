from __future__ import annotations
from typing import TYPE_CHECKING
from abc import abstractmethod, ABC
import numpy as np
from enum import Enum
from typing import Any


if TYPE_CHECKING:
    from com.model import IModel


class ILayer(ABC):
    class LayerType(Enum):
        LINEAR = 0
        RELU = 1
        SOFTMAX = 2
        SIGMOID = 3
        AVERAGINGLINEAR = 4

    def __init__(self, model: IModel) -> None:
        self.on_call = self.forward
        self.model = model
        self.model.layers.append(self._identify())

    def enable_caching(self) -> None:
        self.on_call = self.forward_caching

    def disable_caching(self) -> None:
        self.on_call = self.forward

    def __call__(self, input: np.ndarray) -> np.ndarray:
        return self.on_call(input)

    @property
    def cache(self) -> np.ndarray | None:
        return self._cache
    @cache.setter
    def cache(self, value: np.ndarray):
        self._cache = value

    @abstractmethod
    def forward(self, input: np.ndarray) -> np.ndarray: ...

    @abstractmethod
    def forward_caching(self, input: np.ndarray) -> np.ndarray: ...

    @abstractmethod
    def back(self, input: np.ndarray) -> np.ndarray: ...

    @abstractmethod
    def graph_register(self) -> None: ...

    @abstractmethod
    def _identify(self) -> tuple[Any, Any, Any]: ...


class Linear(ILayer):
    def __init__(
            self,
            model: IModel,
            out_size: int,
            in_size: int
    ) -> None:
        self.weights: np.ndarray = np.ndarray(
            (in_size, out_size),
            dtype=np.float32
        )
        super().__init__(model)

    def init_random(self, min, max):
        self.weights[:] = np.random.uniform(min, max, self.weights.shape).astype(np.float32)

    def init_zeros(self) -> None:
        self.weights[:] = 0.0
    
    def forward(self, input: np.ndarray) -> np.ndarray:
        output = np.matmul(self.weights, input)
        return output
    
    def forward_caching(self, input: np.ndarray) -> np.ndarray:
        output = np.matmul(self.weights, input)
        self.cache = np.copy(input)
        return output
    
    def back(self, input: np.ndarray) -> np.ndarray:
        return (self.weights.T @ input.T).T
    
    def graph_register(self) -> None:
        self.model.handle_graph(self.LayerType.LINEAR)

    def _identify(self) -> tuple[Any, Any, Any]:
        return (self.LayerType.LINEAR, self.weights.shape, self)


class AveragingLinear(Linear):
    def forward(self, input: np.ndarray) -> np.ndarray:
        averaged = input.mean(axis=-3)
        return np.matmul(self.weights, averaged)
 
    def forward_caching(self, input: np.ndarray) -> np.ndarray:
        averaged = input.mean(axis=-3)
        self.cache = np.copy(averaged)
        return np.matmul(self.weights, averaged)
 
    def graph_register(self) -> None:
        self.model.handle_graph(self.LayerType.AVERAGINGLINEAR)
 
    def _identify(self) -> tuple[Any, Any, Any]:
        return (self.LayerType.AVERAGINGLINEAR, self.weights.shape, self)


class ReLU(ILayer):
    def forward(self, input: np.ndarray) -> np.ndarray:
        np.maximum(0, input, out=input)
        return input
    
    def forward_caching(self, input: np.ndarray) -> np.ndarray:
        np.maximum(0, input, out=input)
        self.cache = np.copy(input)
        return input

    def back(self, input: np.ndarray) -> np.ndarray:
        if self.cache is None:
            raise ValueError("ReLU: Failed to find cached forward pass!")
        grad_x = input * (self.cache > 0).astype(input.dtype)
        return grad_x
    
    def graph_register(self) -> None:
        self.model.handle_graph(self.LayerType.RELU)

    def _identify(self) -> tuple[Any, Any, Any]:
        return (self.LayerType.RELU, None, self)


class SoftMax(ILayer):
    def forward(self, input: np.ndarray) -> np.ndarray:
        exp = np.exp(input)
        return exp / np.sum(exp)
    
    def forward_caching(self, input: np.ndarray) -> np.ndarray:
        self.cache = np.copy(input)
        exp = np.exp(input)
        return exp / np.sum(exp)

    def back(self, input: np.ndarray) -> np.ndarray:
        if self.cache is None:
            raise ValueError("SoftMax: Failed to find cached forward pass!")
        raise NotImplementedError("This case basically does not occur.")
    
    def graph_register(self) -> None:
        self.model.handle_graph(self.LayerType.SOFTMAX)

    def _identify(self) -> tuple[Any, Any, Any]:
        return (self.LayerType.SOFTMAX, None, self)


class Sigmoid(ILayer):
    def forward(self, input: np.ndarray) -> np.ndarray:
        return 1 / (1 + np.exp(-input))
    
    def forward_caching(self, input: np.ndarray) -> np.ndarray:
        x = 1 / (1 + np.exp(-input))
        self.cache = np.copy(x)
        return x

    def back(self, input: np.ndarray) -> np.ndarray:
        if not self.cache:
            raise ValueError("Sigmoid: Failed to find cached forward pass!")
        return input * self.cache * (1 - self.cache)
    
    def graph_register(self) -> None:
        self.model.handle_graph(self.LayerType.SIGMOID)

    def _identify(self) -> tuple[Any, Any, Any]:
        return (self.LayerType.SIGMOID, None, self)
