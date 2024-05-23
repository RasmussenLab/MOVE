__all__ = [
    "MOVEConfig",
    "EncodeDataConfig",
]

from dataclasses import dataclass, field
from typing import Any, Optional

from hydra.core.config_store import ConfigStore
from omegaconf import MISSING

from move.core.qualname import get_fully_qualname
from move.conf.tasks import InputConfig

from move.tasks.encode_data import EncodeData


@dataclass
class DataConfig:
    raw_data_path: str = MISSING
    interim_data_path: str = MISSING
    results_path: str = MISSING
    sample_names: str = MISSING
    categorical_inputs: list[InputConfig] = MISSING
    continuous_inputs: list[InputConfig] = MISSING
    categorical_names: list[str] = MISSING
    continuous_names: list[str] = MISSING
    categorical_weights: list[int] = MISSING
    continuous_weights: list[int] = MISSING


@dataclass
class TaskConfig:
    """Configure a MOVE task."""


@dataclass
class EncodeDataConfig(TaskConfig):
    """Configure data encoding."""

    _target_: str = field(
        default=get_fully_qualname(EncodeData), init=False, repr=False
    )
    raw_data_path: str
    interim_data_path: str
    sample_names_filename: str
    discrete_inputs: list[dict[str, Any]]
    continuous_inputs: list[dict[str, Any]]


@dataclass
class MOVEConfig:
    defaults: list[Any] = field(default_factory=lambda: [dict(data="base_data")])
    data: DataConfig = MISSING
    task: TaskConfig = MISSING
    seed: Optional[int] = None


# Store config schema
cs = ConfigStore.instance()
cs.store(name="config_schema", node=MOVEConfig)
cs.store(
    group="task",
    name="encode_data_schema",
    node=EncodeDataConfig,
)
