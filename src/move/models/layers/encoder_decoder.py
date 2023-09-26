__all__ = ["Encoder", "Decoder"]

from typing import Any, Sequence, Union, overload, cast

import torch
from torch import nn

from move.models.layers.chunk import Chunk


def build_network(
    input_dim: int,
    compress_dims: Sequence[int],
    output_dim: int,
    dropout_rate: float,
    activation_fun_name: str,
) -> list[nn.Module]:
    """Build a network that takes # input dimensions, (de)compresses them, and
    returns # output dimensions using a sequence of linear, non-linear, dropout,
    and batch normalization layers."""

    activation_fun = getattr(nn, activation_fun_name)
    assert issubclass(activation_fun, nn.Module)

    layers = []
    layer_dims = [input_dim, *compress_dims, output_dim]

    out_features = None
    for in_features, out_features in zip(layer_dims, layer_dims[1:]):
        layers.append(nn.Linear(in_features, out_features))
        layers.append(activation_fun())
        if dropout_rate > 0:
            layers.append(nn.Dropout(dropout_rate))
        layers.append(nn.BatchNorm1d(out_features))

    assert out_features is not None
    layers.append(nn.Linear(out_features, output_dim))

    return layers


class Encoder(nn.Sequential):
    def __init__(
        self,
        input_dim: int,
        compress_dims: Sequence[int],
        embedding_dim: int,
        embedding_args: int = 2,
        dropout_rate: float = 0.0,
        activation_fun_name: str = "LeakyReLU",
    ) -> None:
        self.num_args = embedding_args
        layers = build_network(input_dim, compress_dims, embedding_dim * embedding_args, dropout_rate, activation_fun_name)
        layers.append(Chunk(embedding_args))
        super().__init__(*layers)

    def __call__(self, *args: Any, **kwds: Any) -> tuple[torch.Tensor, ...]:
        return super().__call__(*args, **kwds)

class Decoder(Encoder):
    def __init__(
        self,
        embedding_dim: int,
        compress_dims: Sequence[int],
        output_dim: int,
        output_args: int = 1,
        dropout_rate: float = 0.0,
        activation_fun_name: str = "LeakyReLU",
    ) -> None:
        super().__init__(
            embedding_dim,
            compress_dims,
            output_dim,
            output_args,
            dropout_rate,
            activation_fun_name
        )
