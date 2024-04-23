__all__ = ["MOVEDataset", "make_dataset", "make_dataloader", "split_samples"]

from typing import Optional

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from move.core.typing import BoolArray, FloatArray


class MOVEDataset(TensorDataset):
    """
    Characterizes a dataset for PyTorch

    Args:
        cat_all:
            categorical input matrix (N_samples, N_variables x N_max-classes.
        con_all:
            normalized continuous input matrix (N_samples, N_variables).
        cat_shapes:
            list of tuples corresponding to number of features (N_variables,
            N_max-classes) of each categorical class.
        con_shapes:
            list of tuples corresponding to number of features
            (N_variables) of each continuous class.

    Raises:
        ValueError:
            Number of samples between categorical and continuous datasets must
            match.
        ValueError:
            Categorical and continuous data cannot be both empty.
    """

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

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        cat_slice = torch.empty(0) if self.cat_all is None else self.cat_all[idx]
        con_slice = torch.empty(0) if self.con_all is None else self.con_all[idx]
        return cat_slice, con_slice

def concat_cat_list(
    cat_list: list[FloatArray],
) -> tuple[list[tuple[int, ...]], FloatArray]:
    """
    Concatenate a list of categorical data
    Args:
        cat_list: list with each categorical class data
    Returns:
        (tuple): a tuple containing:
            cat_shapes:
                list of categorical data classes shapes (N_variables,
                 N_max-classes)
            cat_all (FloatArray):
                2D array of concatenated patients categorical data
    """

    cat_shapes = []
    cat_flat = []
    for cat in cat_list:
        cat_shape = (cat.shape[1], cat.shape[2])
        cat_shapes.append(cat_shape)
        cat_flat.append(cat.reshape(cat.shape[0], -1))
    cat_all = np.concatenate(cat_flat, axis=1)
    return cat_shapes, cat_all


def concat_con_list(
    con_list: list[FloatArray],
) -> tuple[list[int], FloatArray]:
    """
    Concatenate a list of continuous data
    Args:
        con_list: list with each continuous class data
    Returns:
        (tuple): a tuple containing:
            n_con_shapes:
                list of continuous data classes shapes (in 1D) (N_variables)
            con_all:
                2D array of concatenated patients continuous data
    """
    con_shapes = [con.shape[1] for con in con_list]
    con_all: FloatArray = np.concatenate(con_list, axis=1)
    return con_shapes, con_all


def make_dataset(
    cat_list: Optional[list[FloatArray]] = None,
    con_list: Optional[list[FloatArray]] = None,
    mask: Optional[BoolArray] = None,
) -> MOVEDataset:
    """Creates a dataset that combines categorical and continuous datasets.

    Args:
        cat_list:
            List of categorical datasets (`num_samples` x `num_features`
            x `num_categories`). Defaults to None.
        con_list:
            List of continuous datasets (`num_samples` x `num_features`).
            Defaults to None.
        mask:
            Boolean array to mask samples. Defaults to None.

    Raises:
        ValueError: If both inputs are None

    Returns:
        MOVEDataset
    """
    if not cat_list and not con_list:
        raise ValueError("At least one type of data must be in the input")

    cat_shapes, cat_all = [], None
    if cat_list:
        cat_shapes, cat_all = concat_cat_list(cat_list)

    con_shapes, con_all = [], None
    if con_list:
        con_shapes, con_all = concat_con_list(con_list)

    if cat_all is not None:
        cat_all = torch.from_numpy(cat_all)
        if mask is not None:
            cat_all = cat_all[mask]

    if con_all is not None:
        con_all = torch.from_numpy(con_all)
        if mask is not None:
            con_all = con_all[mask]

    return MOVEDataset(cat_all, con_all, cat_shapes, con_shapes)


def make_dataloader(
    cat_list: Optional[list[FloatArray]] = None,
    con_list: Optional[list[FloatArray]] = None,
    mask: Optional[BoolArray] = None,
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
        mask:
            Boolean array to mask samples. Defaults to None.
        **kwargs:
            Arguments to pass to the DataLoader (e.g., batch size)

    Raises:
        ValueError: If both inputs are None

    Returns:
        DataLoader
    """
    dataset = make_dataset(cat_list, con_list, mask)
    return DataLoader(dataset, **kwargs)


def split_samples(
    num_samples: int,
    train_frac: float,
) -> BoolArray:
    """Generate mask to randomly split samples into training and test sets.

    Args:
        num_samples: Number of samples to split.
        train_frac: Fraction of samples corresponding to training set.

    Returns:
        Boolean array to mask test samples.
    """
    sample_ids = np.arange(num_samples)
    train_size = int(train_frac * num_samples)

    rng = np.random.default_rng()
    train_ids = rng.permutation(sample_ids)[:train_size]

    return np.isin(sample_ids, train_ids)
