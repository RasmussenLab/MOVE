__all__ = ["Chunk"]

import torch
from torch import nn


class Chunk(nn.Module):
    def __init__(self, chunks: int, dim: int = -1) -> None:
        super().__init__()
        self.chunks = chunks
        self.dim = dim

    def forward(self, input: torch.Tensor) -> tuple[torch.Tensor, ...]:
        if self.chunks == 1:
            return input,
        return tuple(torch.chunk(input, self.chunks, self.dim))
