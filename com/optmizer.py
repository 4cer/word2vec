from abc import abstractmethod, ABC
from loss import LossFunction


class Optimizer(ABC):
    def __init__(
        self,
        loss: LossFunction,
        max_iterations: int = -1,
        learningRate: float = 0.1
    ) -> None:
        pass
