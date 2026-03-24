from __future__ import annotations
from typing import TYPE_CHECKING
from abc import abstractmethod, ABC
from typing import Any
import struct
import numpy as np


if TYPE_CHECKING:
    from com.layer import ILayer


# Magic bytes: ASCII 'WGHT' — identifies our checkpoint format
_MAGIC   = b'WGHT'
_VERSION = 1


class IModel(ABC):
    """Abstract interface class for ML models.

    Abstract interface for a model container that manages layers, forward
    execution, optional caching, and checkpoint (de)serialization.

    This class defines the minimal runtime contract for model objects composed
    of ILayer instances. It centralizes:
    - A registry of layers and their identifying metadata (used for weight
    loading and saving and building propagation graphs).
    - The forward entry point used by callers (via call).
    - Optional graph tracing and per-layer forward caching hooks used during
    execution to build computation graphs or retain intermediate activations.
    - Checkpoint file format-aware weight loading and saving in 32-bit float
    format.

    Attributes
    ---
    layers : list[tuple[ILayer.LayerType, Any, ILayer]]
        Ordered list describing the model structure. Each entry is a tuple of
        (layer_type, shape_or_None, layer_ref) where:
        - layer_type (ILayer.LayerType) identifies the layer implementation.
        - shape_or_None (tuple[int, ...] | None) is the expected weight shape
        for this layer or None if the layer has no persisted weights.
        - layer_ref (ILayer) is the live layer object whose weights will be
        read/written during checkpoint operations.

    caching : Callable[[ILayer, numpy.ndarray], None]
        Hook invoked by layers to record forward activations for later
        backpropagation. Defaults to a no-op (no caching). Calling
        enable_caching() on layers or the model should switch this to
        a caching implementation.

    handle_graph : Callable[[ILayer.LayerType], None]
        Hook invoked by layers to register their type with an externally
        provided computation graph. Defaults to a no-op. Enabled by calling
        enable_graph_tracing().
        
    graph : list[ILayer.LayerType] | None
        Optional list used as the target computation graph when graph tracing
        is enabled. When None, graph tracing is disabled.
        
    Methods
    ---
    __call__(x) -> Any:
        Forward entry point; dispatches to the concrete model's _forward
        implementation. Keeps caller-facing API consistent across models.

    _forward(x) -> numpy.ndarray:
        Abstract method that must be implemented by subclasses to define the
        model-specific forward computation. It should iterate over self.layers
        and invoke layer forward methods, using self.caching and self.handle_graph
        as appropriate.

    enable_graph_tracing(graph: list[ILayer.LayerType], persistent_graph: bool = True) -> None:
        Enable graph-tracing mode. Subsequent layer registrations will be
        appended to the provided `graph` list via self.handle_graph. If
        persistent_graph is False the expectation is that the consumer will
        manage clearing the graph between runs; when True the graph is treated
        as persistent across runs.

    disable_graph_tracing() -> None:
        Disable graph tracing and reset the graph handling hook to a no-op.

    cache(layer: ILayer, y: numpy.ndarray) -> None:
        Record a (layer, activation) pair into an internal forward_graph list.
        This is used when collecting forward activations for backpropagation.
        The forward_graph list is created on first use.

    register_in_graph(tag: ILayer.LayerType) -> None:
        Append a layer type tag to the currently enabled graph. Raises a
        RuntimeError if graph tracing was not enabled.

    nope_caching(_layer: ILayer, _y: numpy.ndarray) -> None:
        Default no-op caching implementation.

    nope_graph(tag: ILayer.LayerType) -> None:
        Default no-op graph registration implementation.

    load_weights_fp32(checkpoint_path: str) -> None:
        Read a binary checkpoint file and copy contained 32-bit float weights
        into the model's registered layers.

        See project_root/checkpoint/structure.txt for the binary file format.

    save_weights_fp32(checkpoint_path: str) -> None:
        Serialize the current model weights into a binary checkpoint file as
        contiguous float32 arrays.
        
        See project_root/checkpoint/structure.txt for the binary file format.

    """
    def __init__(self) -> None:
        self.layers: list[tuple[ILayer.LayerType, Any, ILayer]] = []
        self.caching = self.nope_caching
        self.handle_graph = self.nope_graph
        self.graph: list[ILayer.LayerType] | None = None

    def __call__(self, x) -> Any:
        return self._forward(x)

    @abstractmethod
    def _forward(self, x) -> np.ndarray: ...

    def enable_graph_tracing(
            self,
            graph: list[ILayer.LayerType],
            persistent_graph = True
    ) -> None:
        self.graph = graph
        self.persistent_graph = persistent_graph
        self.handle_graph = self.register_in_graph
        pass

    def disable_graph_tracing(self) -> None:
        self.graph = None
        self.handle_graph = self.nope_graph


    def cache(self, layer: ILayer, y: np.ndarray) -> None:
        if not hasattr(self, "forward_graph"):
            self.forward_graph: list[tuple[ILayer, np.ndarray]] = []
        self.forward_graph.append((layer, y))

    def register_in_graph(self, tag: ILayer.LayerType) -> None:
        if self.graph is None:
            raise RuntimeError("Graph not registered at registration time!")
        self.graph.append(tag)

    def nope_caching(self, _layer: ILayer, _y: np.ndarray) -> None:
        pass

    def nope_graph(self, tag: ILayer.LayerType) -> None:
        pass

    def load_weights_fp32(self, checkpoint_path: str) -> None:
        """Load weights from a checkpoint file into initialised model.
 
        The model structure must be fully initialised before this method is
        called. A RuntimeError is raised on any structural mismatch so a silent
        mis-load is impossible.

        File format specification can be found at:
        
        \t`project_root/checkpoint/structure.txt`

        Args:
            checkpoint_path (str): File path pointing to a saved checkpoint.

        Raises:
            RuntimeError: Incorrect magic numbers at the start of the file.
            RuntimeError: Checkpoint file version magic number mismatch with
                `_VERSION` value.
            RuntimeError: Structural error: mismatch of layer count.
            RuntimeError: Structural error: Not enough layers in file.
            RuntimeError: Structural error: Incorrect layer type, in respect to
                the layout set by the _forward override for this model.
            RuntimeError: Structural error: Expected weights for this layer,
                found None.
            RuntimeError: Structural error: Layer shape mismatch checkpoint vs
                model.
        """
        with open(checkpoint_path, 'rb') as f:
 
            magic = f.read(4)
            if magic != _MAGIC:
                raise RuntimeError(
                    f"Not a valid checkpoint file (bad magic: {magic!r})"
                )
            version, ckpt_layer_count = struct.unpack('<II', f.read(8))
            if version != _VERSION:
                raise RuntimeError(
                    f"Unsupported checkpoint version {version} (expected {_VERSION})"
                )
 
            if ckpt_layer_count != len(self.layers):
                raise RuntimeError(
                    f"Layer count mismatch: checkpoint has {ckpt_layer_count} "
                    f"layer(s), model has {len(self.layers)}"
                )
 
            for _ in range(ckpt_layer_count):
                ckpt_idx, ckpt_type_val, ndim = struct.unpack('<III', f.read(12))
 
                if ckpt_idx >= len(self.layers):
                    raise RuntimeError(
                        f"Checkpoint references layer index {ckpt_idx} but "
                        f"model only has {len(self.layers)} layer(s)"
                    )
 
                model_type, model_shape, layer_ref = self.layers[ckpt_idx]
 
                if ckpt_type_val != model_type.value:
                    raise RuntimeError(
                        f"Layer {ckpt_idx} type mismatch: checkpoint has "
                        f"type {ckpt_type_val}, model has {model_type.value} "
                        f"({model_type.name})"
                    )
 
                if ndim == 0:
                    if model_shape is not None:
                        raise RuntimeError(
                            f"Layer {ckpt_idx} ({model_type.name}): checkpoint "
                            f"has no weights but model expects shape {model_shape}"
                        )
                    continue
 
                ckpt_shape = struct.unpack(f'<{ndim}I', f.read(ndim * 4))
                if ckpt_shape != tuple(model_shape):
                    raise RuntimeError(
                        f"Layer {ckpt_idx} ({model_type.name}) shape mismatch: "
                        f"checkpoint {ckpt_shape} vs model {tuple(model_shape)}"
                    )
 
                n_elements = 1
                for d in ckpt_shape:
                    n_elements *= d
                raw = f.read(n_elements * 4)
                loaded = np.frombuffer(raw, dtype=np.float32).reshape(ckpt_shape)
                np.copyto(layer_ref.weights, loaded)

    def save_weights_fp32(self, checkpoint_path: str) -> None:
        """Save weights of the currect state as a checkpoint file.

        Serialize the current model weights into a binary checkpoint file as
        contiguous float32 arrays.

        File format specification can be found at:

        \t`project_root/checkpoint/structure.txt`

        Args:
            checkpoint_path (str): File path pointing to save the state to.
        """ 
        with open(checkpoint_path, 'wb') as f:
            
            f.write(_MAGIC)
            f.write(struct.pack('<II', _VERSION, len(self.layers)))
 
            for idx, (layer_type, shape, layer_ref) in enumerate(self.layers):
                if shape is None:
                    f.write(struct.pack('<III', idx, layer_type.value, 0))
                else:
                    ndim = len(shape)
                    f.write(struct.pack('<III', idx, layer_type.value, ndim))
                    f.write(struct.pack(f'<{ndim}I', *shape))
                    weights: np.ndarray = layer_ref.weights
                    f.write(np.ascontiguousarray(weights, dtype=np.float32).tobytes())
