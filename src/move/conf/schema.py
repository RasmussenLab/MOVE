__all__ = ["MOVEConfig"]

from dataclasses import dataclass

from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf


@dataclass
class InputConfig:
    name: str
    weight: int


@dataclass
class DataConfig:
    na_value: str
    raw_data_path: str
    interim_data_path: str
    processed_data_path: str
    categorical_inputs: list[InputConfig]
    continuous_inputs: list[InputConfig]


@dataclass
class ModelConfig:
    _target_: str
    cuda: bool


@dataclass
class VAEConfig(ModelConfig):
    categorical_weights: list[int]
    continuous_weights: list[int]
    num_hidden: list[int]
    num_latent: int
    beta: float
    dropout: float


@dataclass
class TrainingConfig:
    cuda: bool
    lr: float
    num_epochs: int
    kld_steps: list[int]
    batch_steps: list[int]


@dataclass
class MOVEConfig:
    data: DataConfig
    model: VAEConfig
    training: TrainingConfig
    name: str
    seed: int


def extract_weights(configs: list[InputConfig]) -> list[int]:
    """Extracts the weights from a list of input configs."""
    return [item.weight for item in configs]


# Store config schema
cs = ConfigStore.instance()
cs.store(name="main", node=MOVEConfig)

# Register custom resolvers
OmegaConf.register_new_resolver("weights", extract_weights)
