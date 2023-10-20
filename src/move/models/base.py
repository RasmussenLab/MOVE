__all__ = ["BaseVae"]

from abc import ABC, abstractmethod
from typing import Any, TypedDict

import torch
from torch import nn


class VaeOutput(TypedDict):
    z_loc: torch.Tensor
    z_scale: torch.Tensor
    x_recon: torch.Tensor


class LossDict(TypedDict):
    elbo: torch.Tensor
    discrete_loss: torch.Tensor
    continuous_loss: torch.Tensor
    kl_div: torch.Tensor
    kl_weight: float


class BaseVae(nn.Module, ABC):
    embedding_args: int = 2
    output_args: int = 1
    encoder: nn.Module
    decoder: nn.Module

    def __call__(self, *args: Any, **kwds: Any) -> VaeOutput:
        return super().__call__(*args, **kwds)

    @abstractmethod
    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, ...]:
        ...

    @abstractmethod
    def reparameterize(self, loc: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        ...

    @abstractmethod
    def decode(self, z: torch.Tensor) -> tuple[torch.Tensor, ...]:
        ...

    @abstractmethod
    def compute_loss(self, batch: torch.Tensor, annealing_factor: float) -> LossDict:
        ...

    @torch.no_grad()
    @abstractmethod
    def project(self, batch: torch.Tensor) -> torch.Tensor:
        """Create latent representation."""
        ...

    @torch.no_grad()
    @abstractmethod
    def reconstruct(self, batch: torch.Tensor) -> torch.Tensor:
        """Create reconstruction."""
        ...
