from __future__ import annotations
from typing import TYPE_CHECKING
from abc import abstractmethod, ABC


if TYPE_CHECKING:
    from com.optimizer import IOptimizer


class IScheduler(ABC):
    """Abstract learning-rate scheduler interface.

    Schedulers encapsulate policies for modifying an optimizer's learning rate
    over time (per-epoch, per-step, or based on metrics). They are thin
    coordinators that mutate the attached IOptimizer.learning_rate (and any
    optimizer-internal state that depends on it) according to a scheduling rule.

    Attributes
    ---
        optimizer : IOptimizer
            The optimizer instance whose learning rate the scheduler will adjust.

    Methods
    ---
        step(**kwargs) -> None:
            Advance the scheduler by one step. The meaning of a "step" depends on
            the concrete scheduler implementation (e.g., epoch-based, batch-based,
            metric-triggered). Optional keyword arguments may include:
            - epoch (int): current epoch index
            - metric (float): validation metric to base adjustments on
            - step (int): global step count
            Implementations should document which kwargs they accept and how they
            affect scheduling behavior.
    """
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

    def step(self, **_) -> None:
        self.step_number += 1
        if self.step_number > self.until_epoch:
            return
        progress: float = (self.step_number / self.until_epoch)
        new_lr: float = (1 - progress) * self.lr_start + progress * self.lr_stop
        self.optimizer.set_learning_rate(new_lr)


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
