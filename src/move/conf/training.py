__all__ = [
    "DataLoaderConfig",
    "TrainingDataLoaderConfig",
    "TestDataLoaderConfig",
    "TrainingLoopConfig",
]

from dataclasses import dataclass, field
from typing import Optional

from omegaconf import MISSING

from move.conf.config_store import config_store
from move.conf.optim import LrSchedulerConfig, OptimizerConfig
from move.core.qualname import get_fully_qualname
from move.data.dataloader import MoveDataLoader
from move.training.loop import TrainingLoop


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

    max_epochs: int = MISSING

    optimizer_config: OptimizerConfig = MISSING
    lr_scheduler_config: Optional[LrSchedulerConfig] = None

    max_grad_norm: Optional[float] = None

    annealing_epochs: int = 0
    annealing_function: str = "linear"
    annealing_schedule: str = "monotonic"

    prog_every_n_epoch: Optional[int] = 10

    log_grad: bool = False


config_store.store(
    group="task/training_loop_config",
    name="schema_training_loop",
    node=TrainingLoopConfig,
)
config_store.store(
    group="task/training_dataloader_config",
    name="schema_dataloader",
    node=DataLoaderConfig,
)
