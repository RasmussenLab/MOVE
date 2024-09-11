__all__ = ["MoveDataLoader"]

from typing import Iterator

import torch
from torch.utils.data import DataLoader

from move.data.dataset import MoveDataset, NamedDataset


class MoveDataLoader(DataLoader):
    dataset: MoveDataset

    @property
    def datasets(self) -> list[NamedDataset]:
        return [dataset for dataset in self.dataset._list if not dataset.is_metadata]

    def __iter__(self) -> Iterator[tuple[torch.Tensor, ...]]:
        return super().__iter__()
