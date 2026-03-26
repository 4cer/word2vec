from __future__ import annotations
from typing import TYPE_CHECKING, Any
from abc import abstractmethod, ABC


if TYPE_CHECKING:
    from com.optimizer import IOptimizer
from com.enums import PerformanceMetric


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
        
        verbosity : int
            The logging level.
            - 0: Log nothing.
            - 1: Log nothing.
            - 2: Log when change is applied.

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

        def _log_adjustment(self, old: float, new: float) -> None:
            Log learning rate adjustment to stdout prefixed by scheduler type.

        def _noop_adjustment(self, old: float, new: float) -> None:
            Default no-op logging implementation for silent verbosity levels.

        _identify() -> tuple[Any, Any, Any]:
            Return scheduler name when logging learning rate adjustment.
    """
    def __init__(
            self,
            optimizer: IOptimizer,
            verbosity: int = 0
    ) -> None:
        self.optimizer: IOptimizer = optimizer
        self.verbosity = verbosity

        self.logger = [
            self._noop_adjustment,
            self._noop_adjustment,
            self._log_adjustment
        ]

    @abstractmethod
    def step(self, **kwargs) -> None:...

    def _log_adjustment(self, old: float, new: float) -> None:
        print(f"{self._identify} Adjusting learning_rate {old:.4f} -> {new:.4f}")

    def _noop_adjustment(self, old: float, new: float) -> None:
        pass

    @abstractmethod
    def _identify(self) -> str: ...


class LinearScheduler(IScheduler):
    def __init__(
            self,
            optimizer: IOptimizer,
            lr_start: float,
            lr_stop: float,
            until_epoch: int,
            verbosity: int = 0
    ) -> None:
        super().__init__(optimizer=optimizer, verbosity=verbosity)
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

    def _identify(self) -> str:
        return "Linear scheduler:"


class PlateauScheduler(IScheduler):
    def __init__(
            self,
            optimizer: IOptimizer,
            factor: float,
            threshold: float,
            min_lr: float,
            patience: int = 10,
            verbosity: int = 0,
            metric: PerformanceMetric = PerformanceMetric.ACCURACY
    ) -> None:
        super().__init__(
            optimizer=optimizer,
            verbosity=verbosity
        )
        self.factor: float = factor
        self.patience: int = patience
        self.threshold: float = threshold
        self.min_lr: float = min_lr
        
        self.comparator = self._accuracy if metric == PerformanceMetric.ACCURACY else self._loss
        self.test_for: str = "accuracy" if metric == PerformanceMetric.ACCURACY else "loss"
        self.reference: float = 0 if metric == PerformanceMetric.ACCURACY else float("inf")
        self.last_fail_n: int = 0


    def step(self, **kwargs) -> None:
        if self.test_for not in kwargs.keys():
            raise RuntimeError("A metric entry must be passed!")
    
        value: float = kwargs.get(self.test_for, 0.0)
        
        if self.comparator(value, self.threshold, self.reference):
            self.last_fail_n = 0
            self.reference = value
        else:
            self.last_fail_n += 1

        if self.last_fail_n >= self.patience:
            new_lr = self.optimizer.learning_rate * self.factor
            self.logger[self.verbosity](self.optimizer.learning_rate, new_lr)
            self.optimizer.set_learning_rate(max(new_lr, self.min_lr))
            self.last_fail_n = 0

    def _identify(self) -> str:
        return "Plateau scheduler:"
    
    @staticmethod
    def _accuracy(value: float, epsilon: float, reference: float) -> bool:
        return True if value + epsilon > reference else False

    @staticmethod
    def _loss(value: float, epsilon: float, reference: float) -> bool:
        return True if value - epsilon < reference else False
