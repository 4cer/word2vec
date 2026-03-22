from abc import abstractmethod, ABC
import numpy as np


class ILayer(ABC):
    def __init__(self) -> None:
        self.on_call = self.forward

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


class Linear(ILayer):
    @property
    def weights(self) -> np.ndarray:
        return self._weights
    
    @property
    def biases(self) -> np.ndarray | None:
        return self._biases
    
    @property
    def bias(self) -> float | None:
        return self._bias
    
    def __init__(
            self,
            in_size: int,
            out_size: int,
            bias: bool = True
    ) -> None:
        super().__init__()
        self._weights: np.ndarray = np.ndarray(
            (in_size, out_size),
            dtype=np.float32
        )

        if not bias:
            self._biases = None
            self._bias = None
        # TODO implement biases exist case -- not necessary for word2vec
    
    def forward(self, input: np.ndarray) -> np.ndarray:
        np.matmul(self.weights, input, out=input)
        return input
    
    def forward_caching(self, input: np.ndarray) -> np.ndarray:
        np.matmul(self.weights, input, out=input)
        self.cache = np.copy(input)
        return input
    
    def back(self, input: np.ndarray) -> np.ndarray:
        return np.transpose(self.weights) * input


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


class SoftMax(ILayer):
    def forward(self, input: np.ndarray) -> np.ndarray:
        exp = np.exp(input)
        return exp / np.sum(exp)
    
    def forward_caching(self, input: np.ndarray) -> np.ndarray:
        x = 1 / (1 + np.exp(-input))
        self.cache = np.copy(x)
        return x

    def back(self, input: np.ndarray) -> np.ndarray:
        if self.cache is None:
            raise ValueError("SoftMax: Failed to find cached forward pass!")
        raise NotImplementedError("This case basically does not occur.")


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
