__all__ = ["MOVEConfig"]

from dataclasses import dataclass

from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf
from typing import List

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
    categorical_inputs: List[InputConfig]
    continuous_inputs: List[InputConfig]


@dataclass
class ModelConfig:
    _target_: str
    cuda: bool


@dataclass
class VAEConfig(ModelConfig):
    categorical_weights: List[int]
    continuous_weights: List[int]
    num_hidden: int
    num_layers: int
    num_latent: List[int]
    beta: float
    dropout: float


@dataclass
class TrainingConfig:
    cuda: bool
    lr: float
    num_epochs: int
    kld_steps: List[int]
    batch_steps: List[int]
    version: str

@dataclass
class TuningReconstructionConfig:
    num_hidden: List[int]
    num_latent: List[int]
    num_layers: List[int]
    beta: List[float]
    dropout: List[float]
    batch_sizes: List[int]

@dataclass
class TuningStabilityConfig:
    num_hidden: List[int]
    num_latent: List[int]
    num_layers: List[int]
    beta: List[float]
    dropout: List[float]
    batch_sizes: List[int]        
    
@dataclass
class MOVEConfig:
    data: DataConfig
    model: VAEConfig
    training: TrainingConfig
    tuning_reconstruction: TuningReconstructionConfig
    tuning_stability: TuningStabilityConfig
    name: str
    seed: int


def extract_weights(configs: List[InputConfig]) -> List[int]:
    """Extracts the weights from a list of input configs."""
    return [item.weight for item in configs]

def extract_names(configs: List[InputConfig]) -> List[int]:
    """Extracts the weights from a list of input configs."""
    return [item.name for item in configs]

# Store config schema
cs = ConfigStore.instance()
cs.store(name="config", node=MOVEConfig)

# Register custom resolvers
OmegaConf.register_new_resolver("weights", extract_weights)
OmegaConf.register_new_resolver("names", extract_names)
