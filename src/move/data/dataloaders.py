__all__ = ["MOVEDataset", "make_dataloader"]

from functools import reduce
from typing import Optional

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from move.core.typing import BoolArray, FloatArray


class MOVEDataset(TensorDataset):    
    """
    Characterizes a dataset for PyTorch

    Attributes:
        TensorDataset (class): class to represent data as list of tensors
        
    Args:
        cat_all (torch.Tensor, optional): categorical input matrix (N_patients, N_variables x N_max-classes. Defaults to None.
        con_all (torch.Tensor, optional): normalized continuous input matrix (N_patients, N_variables). Defaults to None.
        cat_shapes (list, optional): list of tuples correspoding to number of features (N_variables, N_max-classes) of each categorical class. Defaults to None.
        con_shapes (list, optional): list of tuples correspoding to number of features (N_variables) of each continuous class. Defaults to None.

    Raises:
        ValueError: Number of samples between categorical and continuous datasets must match.
        ValueError: Categorical and continuous data cannot be both empty.
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
        """
        Return number of samples samples

        Returns:
            int: number of samples
        """  
        return self.num_samples

    def __getitem__(
        self, idx: int
    ) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Get item based on selected index.

        Args:
            idx (int): index of item

        Returns:
            (tuple): a tuple containing:
                cat_slice (torch.Tensor): categorical features data of selected item
                con_slice (torch.Tensor): continuous features data of selected item
        """   
        cat_slice = None if self.cat_all is None else self.cat_all[idx]
        con_slice = None if self.con_all is None else self.con_all[idx]
        return cat_slice, con_slice


def concat_cat_list(
    cat_list: list[FloatArray],
) -> tuple[list[tuple[int, ...]], BoolArray, FloatArray]:
    """
    Concatenate a list of categorical data

    Args:
        cat_list (list[FloatArray]): list with each categorical class data

    Returns:
        (tuple): a tuple containing:
            cat_shapes (list[tuple[int, ...]]): list of categorical data classes shapes (N_patients, N_variables, N_max-classes)
            mask (BoolArray):  mask for patients with no measurments
            cat_all (FloatArray): 2D array of concatenated patients categorical data
    """  
    cat_shapes = []
    cat_flat = []
    for cat in cat_list:
        cat_shapes.append(cat.shape)
        cat_flat.append(cat.reshape(cat.shape[0], -1))
    cat_all = np.concatenate(cat_flat, axis=1)
    mask = cat_all.sum(axis=1) > 5  # True if row sum is greater than 5
    return cat_shapes, mask, cat_all


def concat_con_list(
    con_list: list[FloatArray],
) -> tuple[list[int], BoolArray, FloatArray]:
    """
    Concatenate a list of continuous data

    Args:
        con_list (list[FloatArray]): list with each continuous class data

    Returns:
        (tuple): a tuple containing:
            n_con_shapes (list[int]): list of continuous data classes shapes (in 1D) (N_variables)
            mask (BoolArray):  mask for patients with no measurments
            con_all (FloatArray): 2D array of concatenated patients continuous data
    """    
    con_shapes = [con.shape[1] for con in con_list]
    con_all: FloatArray = np.concatenate(con_list, axis=1)
    mask = con_all.sum(axis=1) != 0  # True if row sum is not zero
    return con_shapes, mask, con_all


def make_dataloader(
    cat_list: Optional[list[FloatArray]] = None,
    con_list: Optional[list[FloatArray]] = None,
    **kwargs
) -> tuple[BoolArray, DataLoader]:
    """
    Creates a DataLoader that combines categorical and continuous datasets.

    Args:
        cat_list: list of categorical datasets (# samples x # features
        x # classes). Defaults to None.
        con_list: list of continuous datasets (# samples x # features).
        Defaults to None.
        **kwargs: Arguments to pass to the DataLoader (e.g., batch size)

    Raises:
        ValueError: If both inputs are None

    Returns:
        (tuple): a tuple containing:
            mask (BoolArray): mask to remove rows (samples) with all zeros
            dataloader (Dataloader): Dataloader
    """
    if cat_list is None and con_list is None:
        raise ValueError("At least one type of data must be in the input")

    cat_shapes, cat_mask, cat_all = [None] * 3
    if cat_list:
        cat_shapes, cat_mask, cat_all = concat_cat_list(cat_list)

    con_shapes, con_mask, con_all = [None] * 3
    if con_list:
        con_shapes, con_mask, con_all = concat_con_list(con_list)

    mask: BoolArray = reduce(
        np.logical_and, [mask for mask in [cat_mask, con_mask] if mask is not None]
    )
    if cat_all is not None:
        cat_all = torch.from_numpy(cat_all[mask])

    if con_all is not None:
        con_all = torch.from_numpy(con_all[mask])

    dataset = MOVEDataset(cat_all, con_all, cat_shapes, con_shapes)
    dataloader = DataLoader(dataset, **kwargs)
    return mask, dataloader
