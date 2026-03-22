from abc import abstractmethod, ABC


from optmizer import IOptimizer


class IScheduler(ABC):
    def __init__(
            self,
            optimizer: IOptimizer
    ) -> None:
        self.optimizer: IOptimizer = optimizer

    @abstractmethod
    def step(self, **kwargs) -> None:...


class LinearScheduler(IScheduler):
    def __init__(
            self,
            optimizer: IOptimizer,
            lr_start: float,
            lr_stop: float,
            until_epoch: int
    ) -> None:
        self.optimizer: IOptimizer = optimizer
        self.lr_start: float = lr_start
        self.lr_stop: float = lr_stop
        self.until_epoch: int = until_epoch

        self.step_number = 0

    def step(self, **kwargs) -> None:
        if self.step_number > self.until_epoch:
            return
        progress: float = (self.step_number / self.until_epoch)
        new_lr: float = (1 - progress) * self.lr_start + progress * self.lr_stop
        self.optimizer.set_learning_rate(new_lr)
        self.step_number += 1


class PlateauScheduler(IScheduler):
    def __init__(
            self,
            optimizer: IOptimizer,
            factor: float,
            patience: int,
            threshold: float,
            min_lr: float
    ) -> None:
        self.optimizer: IOptimizer = optimizer
        self.factor: float = factor
        self.patience: int = patience
        self.threshold: float = threshold
        self.min_lr: float = min_lr

    def step(self, **kwargs) -> None:
        raise NotImplementedError("PlateauScheduler not implemetned!")


    
