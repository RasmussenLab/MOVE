__all__ = ["MoveDataLoader"]

from torch.utils.data import Dataset, DataLoader

from move.data.dataset import MoveDataset


class MoveDataLoader(DataLoader):
    dataset: MoveDataset
