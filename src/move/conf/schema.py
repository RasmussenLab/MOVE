__all__ = ["MOVEConfig"]

from dataclasses import dataclass

from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf
from typing import List

@dataclass
class InputConfig:
    name: str
    weight: float

@dataclass
class DataConfig:
    user_conf: str
    na_value: str
    raw_data_path: str
    interim_data_path: str
    processed_data_path: str
    version: str
    ids_file_name: str
    ids_has_header: bool
    ids_colname: str
    categorical_inputs: List[InputConfig]
    continuous_inputs: List[InputConfig]
    data_of_interest: str
    categorical_names: List[str]
    continuous_names: List[str]
    categorical_weights: List[float]
    continuous_weights: List[float]
    data_features_to_visualize_notebook4: List[str]
    write_omics_results_notebook5: List[str]
                
@dataclass
class ModelConfig:
    _target_: str
    cuda: bool

@dataclass
class VAEConfig(ModelConfig):
    user_conf: str
    seed: int
    cuda: bool
    lrate: float
    num_epochs: int
    patience: int
    kld_steps: List[int]
    batch_steps: List[int]
        
@dataclass
class TuningReconstructionConfig:
    user_config: str
    num_hidden: List[int]
    num_latent: List[int]
    num_layers: List[int]
    beta: List[float]
    dropout: List[float]
    batch_sizes: List[int]
    repeats: int
    max_param_combos_to_save: int

@dataclass
class TuningStabilityConfig:
    user_config: str
    num_hidden: List[int]
    num_latent: List[int]
    num_layers: List[int]
    beta: List[float]
    dropout: List[float]
    batch_sizes: List[int]
    repeat: int  
    tuned_num_epochs: int
        
@dataclass
class TrainingLatentConfig:
    user_config: str
    num_hidden: int
    num_latent: int
    num_layers: int
    dropout: float
    beta: float
    batch_sizes: int 
    tuned_num_epochs: int
        
@dataclass
class TrainingAssociationConfig:
    user_config: str
    num_hidden: int
    num_latent: List[int]
    num_layers: int
    dropout: float
    beta: float
    batch_sizes: int 
    repeats: int 
    tuned_num_epochs: int

@dataclass
class MOVEConfig:
    data: DataConfig
    model: VAEConfig
    tune_reconstruction: TuningReconstructionConfig
    tune_stability: TuningStabilityConfig
    train_latent: TrainingLatentConfig
    train_association: TrainingAssociationConfig
    name: str


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
