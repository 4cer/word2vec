from __future__ import annotations
from typing import TYPE_CHECKING
from abc import abstractmethod, ABC
import numpy as np


from com.loss import ILossFunction
from com.model import IModel
import com.layer as layer


def collapsed(output: np.ndarray, label: np.ndarray):
    return output.squeeze() - label

COLLAPSE_TABLE = {
    (ILossFunction.LossFunctionType.CATEGORICALCROSSENTROPY, layer.ILayer.LayerType.SOFTMAX): collapsed
}


class IOptimizer(ABC):
    def __init__(
        self,
        model: IModel, 
        loss: ILossFunction,
        max_epochs: int = -1,
        learning_rate: float = 0.1
    ) -> None:
        self.model: IModel = model
        self.loss: ILossFunction = loss
        self.max_epochs: int = max_epochs
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
        max_epochs: int = -1,
        learning_rate: float = 0.1
    ) -> None:
        super().__init__(model, loss, max_epochs, learning_rate)

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
 
        # 1 and 2: Forward and loss
        output = self.model(x)
        loss_val = float(self.loss.forward(output, labels))
 
        # Initial gradient + collapsed-layer skip
        if self._collapse_fn is not None:
            dL = self._collapse_fn(output, labels)
            graph_iter = iter(self._graph)
            next(graph_iter)  # skip the collapsed activation
        else:
            dL = self.loss.back(output, labels)
            graph_iter = iter(self._graph)
 
        # 3. Back propagation
        _linear_types = {layer.ILayer.LayerType.LINEAR, layer.ILayer.LayerType.AVERAGINGLINEAR}
        for lt, ref in graph_iter:
            if lt in _linear_types:
                # 4a. Average across batches
                avg2 = np.einsum('bi,bj->ij', dL, ref.cache.squeeze()) / dL.shape[0]
                
                # 4b. Update weights
                ref.weights -= self.learning_rate * avg2
                dL = ref.back(dL)
            else:
                dL = ref.back(dL)
 
        return loss_val
