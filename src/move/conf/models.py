__all__ = ["VaeConfig", "VaeTConfig"]

from dataclasses import dataclass, field

from move.conf.config_store import config_store
from move.core.qualname import get_fully_qualname
from move.models.vae import Vae
from move.models.vae_distribution import VaeDistribution
from move.models.vae_t import VaeT


@dataclass
class ModelConfig:
    """Configure a model."""

    _target_: str


@dataclass
class VaeConfig(ModelConfig):
    """Configure a variational encoder."""

    _target_: str = field(default=get_fully_qualname(Vae), init=False)

    num_hidden: list[int]
    num_latent: int
    kl_weight: float
    dropout_rate: float
    use_cuda: bool = False


@dataclass
class VaeNormalConfig(VaeConfig):
    """Configure a t-distribution variational autoencoder."""

    _target_: str = field(default=get_fully_qualname(VaeDistribution), init=False)


@dataclass
class VaeTConfig(VaeConfig):
    """Configure a t-distribution variational autoencoder."""

    _target_: str = field(default=get_fully_qualname(VaeT), init=False)


config_store.store(
    group="task/model_config",
    name="vae",
    node=VaeConfig,
)

config_store.store(
    group="task/model_config",
    name="vae_normal",
    node=VaeNormalConfig,
)
