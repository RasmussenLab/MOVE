from __future__ import annotations

__license__ = "MIT"
__version__ = (1, 2, 1)
__all__ = ["conf", "data", "models", "training", "VAE"]

HYDRA_VERSION_BASE = "1.2"

from move import conf, models, training, data
from move.models.vae import VAE
