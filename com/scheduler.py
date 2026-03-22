from abc import abstractmethod, ABC
from optmizer import Optimizer


class Scheduler(ABC):
    def __init__(
            self,
            optimizer: Optimizer
    ) -> None:
        self.optimizer: Optimizer = optimizer

    @abstractmethod
    def step(self, **kwargs) -> None:...


class LinearScheduler(Scheduler):
    def __init__(
            self,
            optimizer: Optimizer,
            lr_start: float,
            lr_stop: float,
            until_epoch: int
    ) -> None:
        self.optimizer: Optimizer = optimizer
        self.lr_start: float = lr_start
        self.lr_stop: float = lr_stop
        self.until_epoch: int = until_epoch

        self.step_number = 0

    def step(self, **kwargs) -> None:
        progress: float = (self.step_number / self.until_epoch)
        new_lr: float = (1 - progress) * self.lr_start + progress * self.lr_stop
        self.optimizer.set_learning_rate(new_lr)


    
