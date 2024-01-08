__all__ = ["MoveDataLoader"]

from torch.utils.data import DataLoader

from move.data.dataset import NamedDataset, MoveDataset


class MoveDataLoader(DataLoader):
    dataset: MoveDataset

    @property
    def datasets(self) -> list[NamedDataset]:
        return self.dataset.datasets
