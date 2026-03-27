from __future__ import annotations
from typing import TYPE_CHECKING
from abc import abstractmethod, ABC
import numpy as np
from typing import Any


if TYPE_CHECKING:
    from com.model import IModel
from com.enums import LayerPurpose
from com.enums import LayerType
from com.enums import LayerPrecision


class ILayer(ABC):
    """Abstract interface model layers.

    Represents any and all operations performed on tensors in forward operation
    of a model, together with the gradient-derived backward behavior for
    backpropagation. Exposes a layer self-identification framework to facilitate
    programmatic backpropagation graph building and the collapsing of common
    combinations with loss functions.
    
    Common layers include:
    - averaging fully connected.
    - linear or fully connected.
    - SoftMax activation function.
    - Rectified Linear Unit (ReLU).

    The primary idea of layer objects is wrapping any and all operations of
    tensor propagation.

    Attributes
    ---
        on_call (Callable[[np.ndarray], np.ndarray])
            Current call handler (forward or forward_caching).
        model (IModel)
            Reference to the owner model.
        _cache (Optional[np.ndarray])
            Internal cache of forward propagation, used in backpropagation.

    Methods
    ---
        enable_caching() -> None:
            Switch call handler to caching-enabled forward.

        disable_caching() -> None:
            Switch call to non-caching forward call handler.

        __call__(input: np.ndarray) -> np.ndarray:
            Dispatch to the active forward implementation.

        forward(input: np.ndarray) -> np.ndarray:
            Compute layer output (no caching). Must be implemented by
            subclasses.

        forward_cached(input: np.ndarray) -> np.ndarray:
            Compute layer output and update cache. Must be implemented by
            subclasses.

        back(input: np.ndarray) -> np.ndarray:
            Backpropagate gradient. Must be implemented by subclasses.

        graph_register() -> None:
            Register nodes/ops with a computation graph. Must be implemented by
            subclasses.

        update_weights(self, dL: np.ndarray, learning_rate: float) -> None:
            Virual method handling weight adjustment in a manner appropriate for
            its type. Must be implemented by subclasses.

        _identify() -> tuple[Any, Any, Any]:
            Return identifying metadata used when registering the layer with the
            model.
    """
    def __init__(
            self,
            model: IModel,
            layer_purposes: set[LayerPurpose] = {LayerPurpose.INFERENCE}
    ) -> None:
        self.on_call = self.forward
        self.model: IModel = model
        self.model.layers.append(self._identify())
        self.layer_purposes: set[LayerPurpose] = layer_purposes

    def enable_caching(self) -> None:
        self.on_call = self.forward_cached

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
    def forward_cached(self, input: np.ndarray) -> np.ndarray: ...

    @abstractmethod
    def back(self, input: np.ndarray) -> np.ndarray: ...

    @abstractmethod
    def graph_register(self) -> None: ...

    @abstractmethod
    def update_weights(self, dL: np.ndarray, learning_rate: float) -> None: ...

    @abstractmethod
    def _identify(self) -> tuple[Any, Any, Any]: ...


class Linear(ILayer):
    def __init__(
            self,
            model: IModel,
            out_size: int,
            in_size: int,
            layer_purposes: set[LayerPurpose] = {LayerPurpose.INFERENCE},
            layer_precision: LayerPrecision = LayerPrecision.FP32
    ) -> None:
        """Linear layer constructor.
        
        When creating linear layers, it is important to remember that the
        default shape for forward propagation is effectively transposed with
        respect to the intuitive orientation. Hence the shape argument order is
        reversed.

        Args:
            model (IModel): Model the layer is being added to.
            out_size (int): The amount of output neurons.
            in_size (int): The amount of input neurons.
            layer_purposes (set[LayerPurpose]): Tags regarding layer purpose.
                This is used to filter layers, when choosing a subset to
                serialize and deserialize.
        """
        self.weights: np.ndarray = np.ndarray(
            (in_size, out_size),
            dtype=layer_precision.value
        )
        super().__init__(model, layer_purposes)

    def init_random(self, min, max):
        self.weights[:] = np.random.uniform(min, max, self.weights.shape).astype(np.float32)

    def init_zeros(self) -> None:
        self.weights[:] = 0.0
    
    def forward(self, input: np.ndarray) -> np.ndarray:
        output = np.matmul(self.weights, input)
        return output
    
    def forward_cached(self, input: np.ndarray) -> np.ndarray:
        output = np.matmul(self.weights, input)
        self.cache = np.copy(input)
        return output
    
    def back(self, input: np.ndarray) -> np.ndarray:
        if self.cache is None:
            raise ValueError("Linear: Failed to find cached forward pass!")
        return (self.weights.T @ input.T).T
    
    def graph_register(self) -> None:
        self.model.handle_graph(LayerType.LINEAR)

    def update_weights(self, dL: np.ndarray, learning_rate: float) -> None:
        if self.cache is None:
            raise ValueError("Linear: Failed to find cached forward pass!")
        # 4a. Average across batches
        avg2 = np.einsum('bi,bj->ij', dL, self.cache.squeeze(axis=-1)) / dL.shape[0]
        # 4b. Update weights
        self.weights -= learning_rate * avg2

    def _identify(self) -> tuple[Any, Any, Any]:
        return (LayerType.LINEAR, self.weights.shape, self)


class ReLU(ILayer):
    def forward(self, input: np.ndarray) -> np.ndarray:
        output = np.maximum(0, input)
        return output
    
    def forward_cached(self, input: np.ndarray) -> np.ndarray:
        self.cache = np.copy(input)
        output = np.maximum(0, input, out=input)
        return output

    def back(self, input: np.ndarray) -> np.ndarray:
        if self.cache is None:
            raise ValueError("ReLU: Failed to find cached forward pass!")
        grad_x = input * (self.cache > 0).astype(input.dtype)
        return grad_x
    
    def graph_register(self) -> None:
        self.model.handle_graph(LayerType.RELU)

    def update_weights(self, dL: np.ndarray, learning_rate: float) -> None:
        pass

    def _identify(self) -> tuple[Any, Any, Any]:
        return (LayerType.RELU, None, self)


class SoftMax(ILayer):
    def forward(self, input: np.ndarray) -> np.ndarray:
        shifted = input - np.max(input, axis=-2, keepdims=True)
        exp = np.exp(shifted)
        return exp / np.sum(exp, axis=-2, keepdims=True)
    
    def forward_cached(self, input: np.ndarray) -> np.ndarray:
        self.cache = np.copy(input)
        shifted = input - np.max(input, axis=-2, keepdims=True)
        exp = np.exp(shifted)
        return exp / np.sum(exp, axis=-2, keepdims=True)

    def back(self, input: np.ndarray) -> np.ndarray:
        if self.cache is None:
            raise ValueError("SoftMax: Failed to find cached forward pass!")
        raise NotImplementedError("This case basically does not occur.")
    
    def graph_register(self) -> None:
        self.model.handle_graph(LayerType.SOFTMAX)

    def update_weights(self, dL: np.ndarray, learning_rate: float) -> None:
        pass

    def _identify(self) -> tuple[Any, Any, Any]:
        return (LayerType.SOFTMAX, None, self)


class Sigmoid(ILayer):
    def forward(self, input: np.ndarray) -> np.ndarray:
        return 1 / (1 + np.exp(-input))
    
    def forward_cached(self, input: np.ndarray) -> np.ndarray:
        x = 1 / (1 + np.exp(-input))
        self.cache = np.copy(x)
        return x

    def back(self, input: np.ndarray) -> np.ndarray:
        if self.cache is None:
            raise ValueError("Sigmoid: Failed to find cached forward pass!")
        return input * self.cache * (1 - self.cache)
    
    def graph_register(self) -> None:
        self.model.handle_graph(LayerType.SIGMOID)

    def update_weights(self, dL: np.ndarray, learning_rate: float) -> None:
        pass

    def _identify(self) -> tuple[Any, Any, Any]:
        return (LayerType.SIGMOID, None, self)


class AveragingLinear(Linear):
    def forward(self, input: np.ndarray) -> np.ndarray:
        averaged = input.mean(axis=-3)
        return np.matmul(self.weights, averaged)

    def forward_cached(self, input: np.ndarray) -> np.ndarray:
        averaged = input.mean(axis=-3)
        self.cache = np.copy(averaged)
        return np.matmul(self.weights, averaged)

    def graph_register(self) -> None:
        self.model.handle_graph(LayerType.AVERAGINGLINEAR)

    def update_weights(self, dL: np.ndarray, learning_rate: float) -> None:
        pass

    def _identify(self) -> tuple[Any, Any, Any]:
        return (LayerType.AVERAGINGLINEAR, self.weights.shape, self)
