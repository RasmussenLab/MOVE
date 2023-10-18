__all__ = ["MoveDataLoader"]

from torch.utils.data import DataLoader, Dataset

from move.data.dataset import MoveDataset


class MoveDataLoader(DataLoader):
    dataset: MoveDataset
