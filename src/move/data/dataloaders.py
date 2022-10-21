__all__ = ["MOVEDataset", "make_dataloader"]

from functools import reduce
from typing import Any, Optional

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

from move.core.typing import FloatArray, IntArray


class MOVEDataset(TensorDataset):
    "Characterizes a dataset for PyTorch"

    def __init__(
        self,
        cat_all: Optional[torch.Tensor] = None,
        con_all: Optional[torch.Tensor] = None,
        cat_shapes: Optional[list[tuple[int, ...]]] = None,
        con_shapes: Optional[list[int]] = None,
    ) -> None:
        # Check
        num_samples = None if cat_all is None else cat_all.shape[0]
        if con_all is not None:
            if num_samples is None:
                num_samples = con_all.shape[0]
            elif num_samples != con_all.shape[0]:
                raise ValueError(
                    "Number of samples between categorical and continuous "
                    "datasets must match."
                )
        elif num_samples is None:
            raise ValueError("Categorical and continuous data cannot be both empty.")
        self.num_samples = num_samples
        self.cat_all = cat_all
        self.cat_shapes = cat_shapes
        self.con_all = con_all
        self.con_shapes = con_shapes

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(
        self, idx: int
    ) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        cat_slice = None if self.cat_all is None else self.cat_all[idx]
        con_slice = None if self.con_all is None else self.con_all[idx]
        return cat_slice, con_slice


def concat_cat_list(
    cat_list: list[FloatArray],
) -> tuple[list[tuple[int, ...]], FloatArray]:
    cat_shapes = []
    cat_flat = []
    for cat in cat_list:
        cat_shapes.append(cat.shape)
        cat_flat.append(cat.reshape(cat.shape[0], -1))
    cat_all = np.concatenate(cat_flat, axis=1)
    mask = cat_all.sum(axis=1) > 5  # True if row sum is greater than 5
    return cat_shapes, cat_all


def concat_con_list(
    con_list: list[FloatArray],
) -> tuple[list[int], FloatArray]:
    con_shapes = [con.shape[1] for con in con_list]
    con_all: FloatArray = np.concatenate(con_list, axis=1)
    mask = con_all.sum(axis=1) != 0  # True if row sum is not zero
    return con_shapes, con_all


def make_dataset(
    cat_list: Optional[list[FloatArray]] = None,
    con_list: Optional[list[FloatArray]] = None,
) -> MOVEDataset:
    """Creates a dataset that combines categorical and continuous datasets.

    Args:
        cat_list:
            List of categorical datasets (`num_samples` x `num_features`
            x `num_categories`). Defaults to None.
        con_list:
            List of continuous datasets (`num_samples` x `num_features`).
            Defaults to None.

    Raises:
        ValueError: If both inputs are None

    Returns:
        MOVEDataset
    """
    if cat_list is None and con_list is None:
        raise ValueError("At least one type of data must be in the input")

    cat_shapes, cat_all = None, None
    if cat_list:
        cat_shapes, cat_all = concat_cat_list(cat_list)

    con_shapes, con_all = None, None
    if con_list:
        con_shapes, con_all = concat_con_list(con_list)

    if cat_all is not None:
        cat_all = torch.from_numpy(cat_all)

    if con_all is not None:
        con_all = torch.from_numpy(con_all)

    return MOVEDataset(cat_all, con_all, cat_shapes, con_shapes)


def make_dataloader(
    cat_list: Optional[list[FloatArray]] = None,
    con_list: Optional[list[FloatArray]] = None,
    **kwargs
) -> DataLoader:
    """Creates a DataLoader that combines categorical and continuous datasets.

    Args:
        cat_list:
            List of categorical datasets (`num_samples` x `num_features`
            x `num_categories`). Defaults to None.
        con_list:
            List of continuous datasets (`num_samples` x `num_features`).
            Defaults to None.
        **kwargs:
            Arguments to pass to the DataLoader (e.g., batch size)

    Raises:
        ValueError: If both inputs are None

    Returns:
        DataLoader
    """
    dataset = make_dataset(cat_list, con_list)
    return DataLoader(dataset, **kwargs)
