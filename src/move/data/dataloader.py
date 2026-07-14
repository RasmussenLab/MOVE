__all__ = ["MoveDataLoader"]

from typing import Iterator

import torch
from torch.utils.data import DataLoader

from move.data.dataset import MoveDataset, NamedDataset


class MoveDataLoader(DataLoader):
    """A ``torch.utils.data.DataLoader`` specialized for a :class:`MoveDataset`,
    yielding batches as tuples of tensors (one per constituent dataset, or two
    if the underlying dataset has a perturbation)."""

    dataset: MoveDataset

    @property
    def datasets(self) -> list[NamedDataset]:
        """Constituent datasets of the underlying :class:`MoveDataset`."""
        return list(self.dataset.datasets.values())

    def __iter__(self) -> Iterator[tuple[torch.Tensor, ...]]:
        return super().__iter__()
