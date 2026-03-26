from abc import ABC, abstractmethod
import numpy as np
from typing import Any


from com.enums import LossFunctionType


class ILossFunction(ABC):
    """Abstract interface for loss functions.

    Represents any and all loss function calculation methodology performed on
    model output tensors in forward operation, together with the gradient-derived
    backward behavior for backpropagation. Exposes a loss self-identification
    framework to facilitate programmatic backpropagation graph building and the
    collapsing of common combinations with loss functions.
    
    Common loss functions include:
    - Categorical Cross-Entropy (CCE).
    - Binary Cross-Entropy (CE).
    - Mean Square Error (MSE).
    - Root Mean Square Error (RMSE).

    Methods
    ---
        __call__(input: np.ndarray) -> np.ndarray:
            Dispatches to the forward implementation.

        forward() -> np.ndarray:
            Compute loss value(s). Always packed into np.ndarray. Must be
            implemented by subclasses.

        back(input: np.ndarray) -> np.ndarray:
            Backpropagate gradient. Must be implemented by subclasses.

        identify() -> tuple[Any, Any, Any]:
            Return identifying metadata used when registering the layer with the
            model.
    """
    def __call__(
        self,
        input: np.ndarray,
        labels: np.ndarray
    ) -> Any:
        return self.forward(
            input=input,
            labels=labels
        )
    
    @abstractmethod
    def forward(
        self,
        input: np.ndarray,
        labels: np.ndarray
    ) -> np.ndarray: ...

    @abstractmethod
    def back(
        self,
        input: np.ndarray,
        labels: np.ndarray
    ) -> np.ndarray: ...

    @abstractmethod
    def identify(self) -> LossFunctionType: ...


class CrossEntropy(ILossFunction):
    def __init__(self) -> None:
        raise NotImplementedError("CrossEntropy not implemented!")
    
    def forward(
        self,
        input: np.ndarray,
        labels: np.ndarray
    ) -> np.ndarray:
        raise NotImplementedError("CrossEntropy not implemented!")
    
    def back(
        self,
        input: np.ndarray,
        labels: np.ndarray
    ) -> np.ndarray:
        raise NotImplementedError("CrossEntropy not implemented!")
    
    def identify(self) -> LossFunctionType:
        return LossFunctionType.CROSSENTROPY    


class CategoricalCrossEntropy(ILossFunction):
    def forward(
        self,
        input: np.ndarray,
        labels: np.ndarray
    ) -> np.ndarray:
        input_clipped = np.clip(input, 1e-15, 1 - 1e-15)
        return -np.sum(labels * np.log(input_clipped).squeeze(axis=-1))
    
    def back(
        self,
        input: np.ndarray,
        labels: np.ndarray
    ) -> np.ndarray:
        input_clipped = np.clip(input, 1e-15, 1 - 1e-15)
        return -labels / input_clipped
    
    def identify(self) -> LossFunctionType:
        return LossFunctionType.CATEGORICALCROSSENTROPY
