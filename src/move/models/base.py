__all__ = ["BaseVae"]

import inspect
from abc import ABC, abstractmethod
from importlib import import_module
from pathlib import Path
from typing import (
    Any,
    OrderedDict,
    Type,
    TypedDict,
    TypeVar,
    Union,
    cast,
    overload,
)

import torch
from torch import nn

from move.core.qualname import get_fully_qualname
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
    num_latent: int

    def __call__(self, *args: Any, **kwds: Any) -> VaeOutput:
        return super().__call__(*args, **kwds)

    @abstractmethod
    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, ...]: ...

    @abstractmethod
    def reparameterize(
        self, loc: torch.Tensor, scale: torch.Tensor
    ) -> torch.Tensor: ...

    @abstractmethod
    def decode(self, z: torch.Tensor) -> tuple[torch.Tensor, ...]: ...

    @abstractmethod
    def compute_loss(
        self, batch: torch.Tensor, annealing_factor: float
    ) -> LossDict: ...

    @torch.no_grad()
    @abstractmethod
    def project(self, batch: torch.Tensor) -> torch.Tensor:
        """Create latent representation."""
        ...

    @overload
    @abstractmethod
    def reconstruct(
        self, batch: torch.Tensor, as_one: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor]: ...

    @overload
    @abstractmethod
    def reconstruct(self, batch: torch.Tensor, as_one: bool = True) -> torch.Tensor: ...

    @torch.no_grad()
    @abstractmethod
    def reconstruct(
        self, batch: torch.Tensor, as_one: bool = False
    ) -> Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """Create reconstruction."""
        ...

    @classmethod
    def reload(cls: Type[T], model_path: Path) -> T:
        """Reload a model from its serialized config and state dict."""
        model_dict = cast(SerializedModel, torch.load(model_path))
        target = model_dict["config"].pop("_target_")
        module_name, class_name = target.rsplit(".", 1)
        module = import_module(module_name)
        cls_: Type = getattr(module, class_name)
        model = cls_(**model_dict["config"])
        model.load_state_dict(model_dict["state_dict"])
        return model

    def freeze(self) -> None:
        """Freeze all parameters."""
        for param in self.parameters():
            param.requires_grad = False
        self.eval()

    def save(self, model_path: Path) -> None:
        """Save the serialized config and state dict of the model to disk."""
        argnames = inspect.signature(self.__init__).parameters.keys()
        config = {argname: getattr(self, argname) for argname in argnames}
        config["_target_"] = get_fully_qualname(self)
        model = SerializedModel(
            config=config,
            state_dict=self.state_dict(),
        )
        torch.save(model, model_path)


def reload_vae(model_path: Path) -> BaseVae:
    """Alias of `BaseVae.reload`."""
    return BaseVae.reload(model_path)
