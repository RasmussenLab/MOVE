__all__ = [
    "DataLoaderConfig",
    "TrainingDataLoaderConfig",
    "TestDataLoaderConfig",
    "TrainingLoopConfig",
]

from dataclasses import dataclass, field
from typing import Optional

from move.conf.optim import LrSchedulerConfig, OptimizerConfig
from move.conf.schema import get_fully_qualname
from move.data.dataloader import MoveDataLoader
from move.training.loop import (
    AnnealingFunction,
    AnnealingSchedule,
    TrainingLoop,
)


@dataclass
class DataLoaderConfig:
    """Configure a data loader."""

    _target_: str = field(
        default=get_fully_qualname(MoveDataLoader), init=False, repr=False
    )
    batch_size: int
    shuffle: bool
    drop_last: bool


@dataclass
class TrainingDataLoaderConfig(DataLoaderConfig):
    """Configure a training data loader, which shuffles data and drops the last
    batch."""

    shuffle: bool = True
    drop_last: bool = True


@dataclass
class TestDataLoaderConfig(DataLoaderConfig):
    """Configure a test data loader, which does not shuffle data and does not
    drop the last batch."""

    shuffle: bool = False
    drop_last: bool = False


@dataclass
class TrainingLoopConfig:
    """Configure a training loop."""

    _target_: str = field(
        default=get_fully_qualname(TrainingLoop), init=False, repr=False
    )

    max_epochs: int

    optimizer_config: OptimizerConfig
    lr_scheduler_config: Optional[LrSchedulerConfig] = None

    max_grad_norm: Optional[float] = None

    annealing_epochs: int = 0
    annealing_function: AnnealingFunction = "linear"
    annealing_schedule: AnnealingSchedule = "monotonic"

    prog_every_n_epoch: Optional[int] = 10

    log_grad: bool = False
