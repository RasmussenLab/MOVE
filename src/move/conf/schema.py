__all__ = [
    "MOVEConfig",
    "EncodeDataConfig",
]

from dataclasses import dataclass, field
from typing import Any, Optional, Type

from omegaconf import MISSING

from move.conf.config_store import config_store
from move.conf.models import ModelConfig
from move.conf.resolvers import register_resolvers
from move.conf.tasks import InputConfig, ReducerConfig, PerturbationConfig
from move.conf.training import (
    DataLoaderConfig,
    TrainingDataLoaderConfig,
    TrainingLoopConfig,
)
from move.core.qualname import get_fully_qualname
from move.tasks.associations import Associations
from move.tasks.encode_data import EncodeData
from move.tasks.latent_space_analysis import LatentSpaceAnalysis
from move.tasks.tuning import TuneModel


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
    train_frac: float = MISSING
    test_frac: float = MISSING
    valid_frac: float = MISSING


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
    train_frac: float = "${data.train_frac}"  # type: ignore
    test_frac: float = "${data.test_frac}"  # type: ignore
    valid_frac: float = "${data.valid_frac}"  # type: ignore


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
class AssociationsConfig(MoveTaskConfig):
    """Configure associations."""

    defaults: list[Any] = field(
        default_factory=lambda: [
            dict(perturbation_config="perturbation"),
            dict(training_loop_config="schema_training_loop"),
        ]
    )

    _target_: str = field(
        default=get_fully_qualname(Associations), init=False, repr=False
    )
    interim_data_path: str = "${data.interim_data_path}"
    results_path: str = "${data.results_path}"
    perturbation_config: PerturbationConfig = MISSING
    num_refits: int = MISSING
    sig_threshold: float = 0.05
    write_only_sig: bool = True


@dataclass
class TuningConfig(MoveTaskConfig):
    """Configure tuning."""

    defaults: list[Any] = field(
        default_factory=lambda: [
            dict(training_loop_config="schema_training_loop"),
        ]
    )

    _target_: str = field(default=get_fully_qualname(TuneModel), init=False, repr=False)
    interim_data_path: str = "${data.interim_data_path}"
    results_path: str = "${data.results_path}"


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
config_store.store(
    group="task",
    name="task_associations",
    node=AssociationsConfig,
)
config_store.store(
    group="task",
    name="task_tuning",
    node=TuningConfig,
)

register_resolvers()

SUPPORTED_TASKS: tuple[Type, ...] = (
    AssociationsConfig,
    EncodeDataConfig,
    LatentSpaceAnalysisConfig,
    TuningConfig,
)
"""List of tasks that can be ran from the command line."""
