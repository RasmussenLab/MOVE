__all__ = [
    "AdamConfig",
    "AdamWConfig",
    "SgdConfig",
    "ExponentialLrConfig",
    "CosineAnnealingLrConfig",
    "ReduceLrOnPlateauConfig",
]

from dataclasses import dataclass, field

from torch.optim import SGD, Adam, AdamW
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    ExponentialLR,
    ReduceLROnPlateau,
)

from move.conf.config_store import config_store
from move.core.qualname import get_fully_qualname


@dataclass
class OptimizerConfig:
    """Configure an optimizer algorithm."""

    _target_: str


@dataclass
class AdamConfig(OptimizerConfig):
    """Configure Adam algorithm."""

    _target_: str = field(default=get_fully_qualname(Adam), init=False)
    lr: float
    weight_decay: float = 0.0


@dataclass
class AdamWConfig(AdamConfig):
    """Configure AdamW algorithm."""

    _target_: str = field(default=get_fully_qualname(AdamW), init=False)


@dataclass
class SgdConfig(OptimizerConfig):
    """Configure stochastic gradient descent algorithm."""

    _target_: str = field(default=get_fully_qualname(SGD), init=False)
    lr: float
    momentum: float = 0.0
    weight_decay: float = 0.0


@dataclass
class LrSchedulerConfig:
    """Configure a learning rate scheduler."""

    _target_: str


@dataclass
class CosineAnnealingLrConfig(LrSchedulerConfig):
    """Configure a cosine annealing learning rate scheduler."""

    _target_: str = field(default=get_fully_qualname(CosineAnnealingLR), init=False)

    T_max: int
    eta_min: float = 0.0


@dataclass
class ExponentialLrConfig(LrSchedulerConfig):
    """Configure exponential decay learning rate scheduler."""

    _target_: str = field(default=get_fully_qualname(ExponentialLR), init=False)
    gamma: float


@dataclass
class ReduceLrOnPlateauConfig(LrSchedulerConfig):
    """Configure learning rate scheduler set to decay when a metric stops
    improving."""

    _target_: str = field(default=get_fully_qualname(ReduceLROnPlateau), init=False)


config_store.store(
    group="task/training_loop_config/optimizer_config",
    name="optim_adam",
    node=AdamConfig,
)
config_store.store(
    group="task/training_loop_config/optimizer_config",
    name="optim_adamw",
    node=AdamWConfig,
)
config_store.store(
    group="task/training_loop_config/optimizer_config",
    name="optim_sgd",
    node=SgdConfig,
)
config_store.store(
    group="task/training_loop_config/lr_scheduler_config",
    name="optim_lr_scheduler",
    node=LrSchedulerConfig,
)
config_store.store(
    group="task/training_loop_config/lr_scheduler_config",
    name="optim_cosine_annealing_lr",
    node=CosineAnnealingLrConfig,
)
config_store.store(
    group="task/training_loop_config/lr_scheduler_config",
    name="optim_exponential_lr",
    node=ExponentialLrConfig,
)
config_store.store(
    group="task/training_loop_config/lr_scheduler_config",
    name="optim_reduce_lr_on_plateau",
    node=ReduceLrOnPlateauConfig,
)
