__all__ = [
    "MOVEConfig",
    "EncodeDataConfig",
    "AnalyzeLatentConfig",
    "TuneModelReconstructionConfig",
    "TuneModelStabilityConfig",
    "IdentifyAssociationsConfig",
    "IdentifyAssociationsBayesConfig",
    "IdentifyAssociationsTTestConfig",
]

from dataclasses import dataclass, field
from typing import Any, Optional

from hydra.core.config_store import ConfigStore
from omegaconf import MISSING, OmegaConf

from move.models.vae import VAE
from move.training.training_loop import training_loop


def get_fully_qualname(sth: Any) -> str:
    return ".".join((sth.__module__, sth.__qualname__))


@dataclass
class InputConfig:
    name: str
    weight: int = 1

@dataclass
class ContinuousInputConfig(InputConfig):
    scale: bool = True


@dataclass
class DataConfig:
    raw_data_path: str = MISSING
    interim_data_path: str = MISSING
    results_path: str = MISSING
    sample_names: str = MISSING
    categorical_inputs: list[InputConfig] = MISSING
    continuous_inputs: list[ContinuousInputConfig] = MISSING
    categorical_names: list[str] = MISSING
    continuous_names: list[str] = MISSING
    categorical_weights: list[int] = MISSING
    continuous_weights: list[int] = MISSING


@dataclass
class ModelConfig:
    _target_: str = MISSING
    cuda: bool = MISSING


@dataclass
class VAEConfig(ModelConfig):
    """Configuration for the VAE module."""

    _target_: str = get_fully_qualname(VAE)
    categorical_weights: list[int] = MISSING
    continuous_weights: list[int] = MISSING
    num_hidden: list[int] = MISSING
    num_latent: int = MISSING
    beta: float = MISSING
    dropout: float = MISSING
    cuda: bool = False


@dataclass
class TrainingLoopConfig:
    _target_: str = get_fully_qualname(training_loop)
    num_epochs: int = MISSING
    lr: float = MISSING
    kld_warmup_steps: list[int] = MISSING
    batch_dilation_steps: list[int] = MISSING
    early_stopping: bool = MISSING
    patience: int = MISSING


@dataclass
class TaskConfig:
    """Configuration for a MOVE task.

    Attributes:
        batch_size: Number of samples in a training batch.
        model: Configuration for a model.
        training_loop: Configuration for a training loop.
    """

    batch_size: Optional[int]
    model: Optional[VAEConfig]
    training_loop: Optional[TrainingLoopConfig]


@dataclass
class EncodeDataConfig(TaskConfig):
    """Configuration for a data-encoding task."""

    batch_size = None
    model = None
    training_loop = None


@dataclass
class TuneModelConfig(TaskConfig):
    """Configure the "tune model" task."""

    ...


@dataclass
class TuneModelStabilityConfig(TuneModelConfig):
    """Configure the "tune model" task."""

    num_refits: int = MISSING


@dataclass
class TuneModelReconstructionConfig(TuneModelConfig):
    """Configure the "tune model" task."""

    ...


@dataclass
class AnalyzeLatentConfig(TaskConfig):
    """Configure the "analyze latents" task.

    Attributes:
        feature_names:
            Names of features to visualize."""

    feature_names: list[str] = field(default_factory=list)
    reducer: dict[str, Any] = MISSING


@dataclass
class IdentifyAssociationsConfig(TaskConfig):
    """Configure the "identify associations" task.

    Attributes:
        target_dataset:
            Name of categorical dataset to perturb.
        target_value:
            The value to change to. It should be a category name.
        num_refits:
            Number of times to refit the model.
        sig_threshold:
            Threshold used to determine whether an association is significant.
            In the t-test approach, this is called significance level (alpha).
            In the probabilistc approach, significant associations are selected
            if their FDR is below this threshold.

            This value should be within the range [0, 1].
        save_models:
            Whether to save the weights of each refit. If weights are saved,
            rerunning the task will load them instead of training.
    """

    target_dataset: str = MISSING
    target_value: str = MISSING
    num_refits: int = MISSING
    sig_threshold: float = 0.05
    save_refits: bool = False


@dataclass
class IdentifyAssociationsBayesConfig(IdentifyAssociationsConfig):
    """Configure the probabilistic approach to identify associations."""

    ...


@dataclass
class IdentifyAssociationsTTestConfig(IdentifyAssociationsConfig):
    """Configure the t-test approach to identify associations.

    Args:
        num_latent:
            List of latent space dimensions to train. It should contain four
            elements.
    """

    num_latent: list[int] = MISSING


@dataclass
class MOVEConfig:
    defaults: list[Any] = field(default_factory=lambda: [dict(data="base_data")])
    data: DataConfig = MISSING
    task: TaskConfig = MISSING
    seed: Optional[int] = None


def extract_weights(configs: list[InputConfig]) -> list[int]:
    """Extracts the weights from a list of input configs."""
    return [1 if not hasattr(item, "weight") else item.weight for item in configs]


def extract_names(configs: list[InputConfig]) -> list[str]:
    """Extracts the weights from a list of input configs."""
    return [item.name for item in configs]


# Store config schema
cs = ConfigStore.instance()
cs.store(name="config_schema", node=MOVEConfig)
cs.store(
    group="task",
    name="encode_data",
    node=EncodeDataConfig,
)
cs.store(
    group="task",
    name="tune_model_reconstruction_schema",
    node=TuneModelReconstructionConfig,
)

cs.store(
    group="task",
    name="tune_model_stability_schema",
    node=TuneModelStabilityConfig,
)
cs.store(
    group="task",
    name="analyze_latent_schema",
    node=AnalyzeLatentConfig,
)
cs.store(
    group="task",
    name="identify_associations_bayes_schema",
    node=IdentifyAssociationsBayesConfig,
)
cs.store(
    group="task",
    name="identify_associations_ttest_schema",
    node=IdentifyAssociationsTTestConfig,
)

# Register custom resolvers
OmegaConf.register_new_resolver("weights", extract_weights)
OmegaConf.register_new_resolver("names", extract_names)
