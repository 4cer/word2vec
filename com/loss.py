from abc import ABC, abstractmethod
import numpy as np


class ILossFunction(ABC):
    @abstractmethod
    def calculate(self, input) -> np.ndarray | float: ...


class CrossEntropy(ILossFunction):
    def calculate(self, input) -> np.ndarray | float:
        raise NotImplementedError("CrossEntropy not implemented!")


class CategoricalCrossEntropy(ILossFunction):
    def calculate(self, input) -> np.ndarray | float:
        raise NotImplementedError("CategoricalCrossEntropy not implemented!")
