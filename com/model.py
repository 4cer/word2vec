from abc import abstractmethod, ABC
import numpy as np


from layer import Layer


class Model(ABC):
    def __init__(self) -> None:
        self.caching = self.nope

    @abstractmethod
    def forward(self, x): ...

    def cache(self, layer: Layer, y: np.ndarray) -> None:
        if not hasattr(self, "forward_graph"):
            self.forward_graph: list[tuple[Layer, np.ndarray]] = []
        self.forward_graph.append((layer, y))

    def nope(self, _layer: Layer, _y: np.ndarray) -> None:
        pass
