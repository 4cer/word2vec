from __future__ import annotations
from typing import TYPE_CHECKING
from abc import abstractmethod, ABC
import numpy as np


if TYPE_CHECKING:
    from com.loss import ILossFunction
    from com.model import IModel
    import com.layer as layer


def collapsed(output: np.ndarray, label: np.ndarray):
    return output - label

COLLAPSE_TABLE = {
    (ILossFunction.LossFunctionType.CATEGORICALCROSSENTROPY, layer.ILayer.LayerType.SOFTMAX): collapsed
}


class IOptimizer(ABC):
    def __init__(
        self,
        model: IModel, 
        loss: ILossFunction,
        max_iterations: int = -1,
        learning_rate: float = 0.1
    ) -> None:
        self.model: IModel = model
        self.loss: ILossFunction = loss
        self.max_iterations: int = max_iterations
        self.learning_rate: float = learning_rate
        self.built_graph_once: bool = False

    @abstractmethod
    def set_learning_rate(self, learning_rate: float): ...

    @abstractmethod
    def propagate(
            self,
            x: np.ndarray,
            labels: np.ndarray
    ) -> float: ...


class SGD(IOptimizer):
    def __init__(
        self,
        model: IModel, 
        loss: ILossFunction,
        max_iterations: int = -1,
        learning_rate: float = 0.1
    ) -> None:
        super().__init__(model, loss, max_iterations, learning_rate)

        self._graph: list[tuple[layer.ILayer.LayerType, layer.ILayer]] = []
        self._collapse_fn = None

    def build_graph_once(self):
        self._graph = [(lt, ref) for lt, _shape, ref in reversed(self.model.layers)]

        for _lt, ref in self._graph:
            ref.enable_caching()
        
        loss_type = self.loss.identify()
        last_layer_type = self._graph[0][0]
        key = (loss_type, last_layer_type)
        self._collapse_fn = COLLAPSE_TABLE.get(key, None)
        
        self.built_graph_once = True


    def set_learning_rate(self, learning_rate: float):
        self.learning_rate = learning_rate

    def propagate(
            self,
            x: np.ndarray,
            labels: np.ndarray
    ) -> float:
        """Run one full SGD iteration

          1. Forward pass  (populates each layer's cache).
          2. Compute loss.
          3. Backward pass (uses graph built by build_graph_once).
          4. Update Linear weights in-place.
        

        Args:
            x (np.ndarray): model output vector.
            labels (np.ndarray): label 1-hot vector.

        Returns:
            float: scalar loss for this sample.
        """
        if not self.built_graph_once:
            self.build_graph_once()
            
        output: np.ndarray = self.model(x)

        loss_value: float = float(self.loss.forward(output, labels))

        if self._collapse_fn is not None:
            grad: np.ndarray = self._collapse_fn(output, labels)
            start_idx = 1
        else:
            grad = np.atleast_1d(
                np.array(self.loss.back(output, labels), dtype=np.float32)
            )
            start_idx = 0
 
        for lt, layer_ref in self._graph[start_idx:]:
            grad = layer_ref.back(grad)

        if self._collapse_fn is not None:
            grad = self._collapse_fn(output, labels)
            walk_start = 1
        else:
            grad = np.atleast_1d(
                np.array(self.loss.back(output, labels), dtype=np.float32)
            )
            walk_start = 0
 
        for lt, layer_ref in self._graph[walk_start:]:
            from com.layer import Linear
            if lt == layer_ref.LayerType.LINEAR:
                assert isinstance(layer_ref, Linear)

        for lt, layer_ref in self._graph[walk_start:]:
            from com.layer import Linear
            if lt == layer_ref.LayerType.LINEAR:
                assert isinstance(layer_ref, Linear)

                weight_grad = np.outer(grad, layer_ref.cache)
                layer_ref.weights -= self.learning_rate * weight_grad
 
            grad = layer_ref.back(grad)
        
        return loss_value
