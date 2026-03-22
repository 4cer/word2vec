from abc import abstractmethod, ABC
import numpy as np


from com.layer import ILayer


class IModel(ABC):
    def __init__(self) -> None:
        self.caching = self.nope

    @abstractmethod
    def forward(self, x) -> np.ndarray: ...

    def cache(self, layer: ILayer, y: np.ndarray) -> None:
        if not hasattr(self, "forward_graph"):
            self.forward_graph: list[tuple[ILayer, np.ndarray]] = []
        self.forward_graph.append((layer, y))

    def nope(self, _layer: ILayer, _y: np.ndarray) -> None:
        pass
