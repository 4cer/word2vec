from abc import abstractmethod, ABC
import numpy as np


from com.layer import ILayer


class IModel(ABC):
    def __init__(self) -> None:
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

    def load_weights(checkpoint_path: str) -> None:
        raise NotImplementedError("Weight loading not implemented yet!")

    def save_weights_fp32(checkpoint_path: str) -> None:
        raise NotImplementedError("Weight saving not implemented yet!")
