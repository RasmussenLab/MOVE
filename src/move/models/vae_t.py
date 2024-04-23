__all__ = ["VaeT"]

from move.models.vae_distribution import VaeDistribution


class VaeT(VaeDistribution):
    """Variational autoencoder with a Student-t distribution on its decoder."""

    ...
