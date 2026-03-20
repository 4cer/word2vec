from abc import abstractmethod, ABC
import numpy as np


class NonLinearity(ABC):
    @abstractmethod
    def forward(self, input) -> np.ndarray: ...

    @abstractmethod
    def back(self, input) -> np.ndarray: ...


class ReLU(NonLinearity):
    def forward(self, input) -> np.ndarray:
        raise NotImplementedError()

    def back(self, input) -> np.ndarray:
        raise NotImplementedError()
    
class SoftMax(NonLinearity):
    def forward(self, input) -> np.ndarray:
        raise NotImplementedError()

    def back(self, input) -> np.ndarray:
        raise NotImplementedError()


class Sigmoid(NonLinearity):
    def forward(self, input) -> np.ndarray:
        raise NotImplementedError()

    def back(self, input) -> np.ndarray:
        raise NotImplementedError()
