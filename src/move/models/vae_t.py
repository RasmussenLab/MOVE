__all__ = ["VaeT"]

from typing import Type

from torch.distributions import Distribution, StudentT

from move.models.vae_distribution import VaeDistribution


class VaeT(VaeDistribution):
    """Variational autoencoder with a Student-t distribution on its decoder."""

    @property
    def decoder_distribution(self) -> Type[Distribution]:
        return StudentT
