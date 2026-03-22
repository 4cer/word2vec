from abc import ABC, abstractmethod
import numpy as np


class LossFunction(ABC):
    @abstractmethod
    def calculate(self, input) -> np.ndarray | float: ...


class CrossEntropy(LossFunction):
    def calculate(self, input) -> np.ndarray | float:
        raise NotImplementedError("CrossEntropy not implemented!")


class CathegoricalCrossEntropy(LossFunction):
    def calculate(self, input) -> np.ndarray | float:
        raise NotImplementedError("CategoricalCrossEntropy not implemented!")
