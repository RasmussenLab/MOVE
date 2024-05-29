__all__ = []

from dataclasses import dataclass, field
from typing import Any, Optional

from omegaconf import MISSING

from move.core.qualname import get_fully_qualname
from move.models.vae_legacy import VAE
from move.training.training_loop import training_loop


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
    """Configure a MOVE task."""


@dataclass
class ModelTaskConfig(TaskConfig):
    """Configure a MOVE task involving a training loop.

    Attributes:
        batch_size: Number of samples in a training batch.
        model: Configuration for a model.
        training_loop: Configuration for a training loop.
    """

    batch_size: Optional[int]
    model: Optional[VAEConfig]
    training_loop: Optional[TrainingLoopConfig]


@dataclass
class TuneModelConfig(ModelTaskConfig):
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
class AnalyzeLatentConfig(ModelTaskConfig):
    """Configure the "analyze latents" task.

    Attributes:
        feature_names:
            Names of features to visualize.
    """

    feature_names: list[str] = field(default_factory=list)
    reducer: dict[str, Any] = MISSING


@dataclass
class IdentifyAssociationsConfig(ModelTaskConfig):
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
