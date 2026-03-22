from abc import abstractmethod, ABC
import numpy as np


class NonLinearity(ABC):
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


class ReLU(NonLinearity):
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


class SoftMax(NonLinearity):
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
        pass


class Sigmoid(NonLinearity):
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
