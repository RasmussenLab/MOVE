__all__ = ["TrainingLoopConfig"]

from dataclasses import dataclass, field
from typing import Optional

from move.conf.optim import LrSchedulerConfig, OptimizerConfig
from move.conf.schema import get_fully_qualname
from move.training.loop import (
    AnnealingFunction,
    AnnealingSchedule,
    TrainingLoop,
)


@dataclass
class TrainingLoopConfig:
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
