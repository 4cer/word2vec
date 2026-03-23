from abc import ABC, abstractmethod
import numpy as np
from typing import Any


class ILossFunction(ABC):
    class LossFunctionType:
        CROSSENTROPY = 0
        CATEGORICALCROSSENTROPY = 1
    @abstractmethod
    def forward(
        self,
        input: np.ndarray,
        labels: np.ndarray
    ) -> np.ndarray | float: ...

    @abstractmethod
    def back(
        self,
        input: np.ndarray,
        labels: np.ndarray
    ) -> np.ndarray | float: ...

    @abstractmethod
    def identify(self) -> Any: ...


class CrossEntropy(ILossFunction):
    def forward(
        self,
        input: np.ndarray,
        labels: np.ndarray
    ) -> np.ndarray | float:
        raise NotImplementedError("CrossEntropy not implemented!")
    
    def back(
        self,
        input: np.ndarray,
        labels: np.ndarray
    ) -> np.ndarray | float:
        raise NotImplementedError("CrossEntropy not implemented!")
    
    def identify(self):
        return ILossFunction.LossFunctionType.CROSSENTROPY    


class CategoricalCrossEntropy(ILossFunction):
    def forward(
        self,
        input: np.ndarray,
        labels: np.ndarray
    ) -> np.ndarray | float:
        input_clipped = np.clip(input, 1e-15, 1 - 1e-15)
        return -np.sum(labels * np.log(input_clipped))
    
    def back(
        self,
        input: np.ndarray,
        labels: np.ndarray
    ) -> np.ndarray | float:
        input_clipped = np.clip(input, 1e-15, 1 - 1e-15)
        return -labels / input_clipped
    
    def identify(self):
        return ILossFunction.LossFunctionType.CATEGORICALCROSSENTROPY    
