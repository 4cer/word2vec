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
    def __init__(self) -> None:
        self.layers: list[tuple[ILayer.LayerType, Any, ILayer]] = []
        self.caching = self.nope_caching
        self.handle_graph = self.nope_graph
        self.graph: list[ILayer.LayerType] | None = None

    @abstractmethod
    def forward(self, x) -> np.ndarray: ...

    def enable_training(
            self,
            graph: list[ILayer.LayerType],
            persistent_graph = True
    ) -> None:
        self.graph = graph
        self.persistent_graph = persistent_graph
        self.handle_graph = self.register_in_graph
        pass

    def disable_training(self) -> None:
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
        """Serialize all layer weights to a binary checkpoint file."""
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
