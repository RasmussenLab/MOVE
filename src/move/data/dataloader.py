__all__ = ["MoveDataLoader"]

from typing import Iterator

import torch
from torch.utils.data import DataLoader

from move.data.dataset import NamedDataset, MoveDataset


class MoveDataLoader(DataLoader):
    dataset: MoveDataset

    @property
    def datasets(self) -> list[NamedDataset]:
        return list(self.dataset.datasets.values())

    def __iter__(self) -> Iterator[tuple[torch.Tensor, ...]]:
        return super().__iter__()
