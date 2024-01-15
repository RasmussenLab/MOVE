__all__ = ["BaseVae"]

import inspect
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Type, TypedDict, TypeVar, cast, OrderedDict

import torch
from torch import nn

from move.models.layers.chunk import SplitInput, SplitOutput

T = TypeVar("T", bound="BaseVae")


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


class SerializedModel(TypedDict):
    config: dict[str, Any]
    state_dict: OrderedDict[str, torch.Tensor]


class BaseVae(nn.Module, ABC):
    embedding_args: int = 2
    output_args: int = 1
    encoder: nn.Module
    decoder: nn.Module
    split_input: SplitInput
    split_output: SplitOutput

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

    @classmethod
    def reload(cls: Type[T], model_path: Path) -> T:
        """Reload a model from its serialized config and state dict."""
        model_dict = cast(SerializedModel, torch.load(model_path))
        model = cls(**model_dict["config"])
        model.load_state_dict(model_dict["state_dict"])
        return model

    def save(self, model_path: Path) -> None:
        """Save the serialized config and state dict of the model to disk."""
        argnames = inspect.signature(self.__init__).parameters.keys()
        model = SerializedModel(
            config={argname: getattr(self, argname) for argname in argnames},
            state_dict=self.state_dict(),
        )
        torch.save(model, model_path)
