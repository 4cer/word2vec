from abc import abstractmethod, ABC


from com.loss import ILossFunction


class IOptimizer(ABC):
    def __init__(
        self,
        loss: ILossFunction,
        max_iterations: int = -1,
        learning_rate: float = 0.1
    ) -> None:
        self.loss: ILossFunction = loss
        self.max_iterations: int = max_iterations
        self.learning_rate: float = learning_rate

    @abstractmethod
    def set_learning_rate(self, learning_rate: float): ...


class SGD(IOptimizer):
    def __init__(
            self,
            loss: ILossFunction,
            max_iterations: int = -1,
            learning_rate: float = 0.1
    ) -> None:
        super().__init__(loss, max_iterations, learning_rate)

    def set_learning_rate(self, learning_rate: float):
        self.learning_rate = learning_rate

    
