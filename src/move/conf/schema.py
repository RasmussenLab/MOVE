__all__ = [
    "MOVEConfig",
    "EncodeDataConfig",
]

from dataclasses import dataclass, field
from typing import Any, Optional

from hydra.core.config_store import ConfigStore
from omegaconf import MISSING


from move.core.qualname import get_fully_qualname

from move.conf.config_store import config_store
from move.conf.models import ModelConfig
from move.conf.resolvers import register_resolvers
from move.conf.tasks import InputConfig, ReducerConfig
from move.conf.training import (
    DataLoaderConfig,
    TrainingLoopConfig,
    TrainingDataLoaderConfig,
)

from move.tasks.encode_data import EncodeData
from move.tasks.latent_space_analysis import LatentSpaceAnalysis


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
    """Configure a task."""


@dataclass
class EncodeDataConfig(TaskConfig):
    """Configure data encoding."""

    _target_: str = field(
        default=get_fully_qualname(EncodeData), init=False, repr=False
    )
    raw_data_path: str = "${data.raw_data_path}"
    interim_data_path: str = "${data.interim_data_path}"
    sample_names_filename: str = "${data.sample_names}"
    discrete_inputs: list[dict[str, Any]] = "${data.categorical_inputs}"  # type: ignore
    continuous_inputs: list[dict[str, Any]] = "${data.continuous_inputs}"  # type: ignore


@dataclass
class MoveTaskConfig(TaskConfig):
    """Configure generic MOVE task."""

    discrete_dataset_names: list[str] = "${data.categorical_names}"  # type: ignore
    continuous_dataset_names: list[str] = "${data.continuous_names}"  # type: ignore
    model_config: Optional[ModelConfig] = MISSING  # "${model}"  # type: ignore
    training_dataloader_config: DataLoaderConfig = TrainingDataLoaderConfig(16)
    training_loop_config: Optional[TrainingLoopConfig] = MISSING


@dataclass
class LatentSpaceAnalysisConfig(MoveTaskConfig):
    """Configure latent space analysis."""

    defaults: list[Any] = field(
        default_factory=lambda: [
            dict(reducer_config="tsne"),
            dict(training_loop_config="schema_training_loop"),
        ]
    )

    _target_: str = field(
        default=get_fully_qualname(LatentSpaceAnalysis), init=False, repr=False
    )
    interim_data_path: str = "${data.interim_data_path}"
    results_path: str = "${data.results_path}"
    compute_accuracy_metrics: bool = MISSING
    compute_feature_importance: bool = MISSING
    reducer_config: Optional[ReducerConfig] = MISSING
    features_to_plot: Optional[list[str]] = MISSING


@dataclass
class MOVEConfig:
    """Configure MOVE command line."""

    defaults: list[Any] = field(default_factory=lambda: [dict(data="base_data")])
    data: DataConfig = MISSING
    task: TaskConfig = MISSING
    seed: Optional[int] = None


# Store config schema
config_store.store(name="config_schema", node=MOVEConfig)
config_store.store(
    group="task",
    name="encode_data",
    node=EncodeDataConfig,
)
config_store.store(
    group="task",
    name="task_latent_space",
    node=LatentSpaceAnalysisConfig,
)
register_resolvers()
