__all__ = ["MOVEDataset", "make_dataloader"]

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from typing import Tuple

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
        cat_all: torch.Tensor = None,
        con_all: torch.Tensor = None,
        cat_shapes: list = None,
        con_shapes: list = None,
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

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor]:
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


def concat_cat_list(cat_list):
    """
    Concatenate a list of categorical data

    Args:
        cat_list (list): list with each categorical class data

    Returns:
        (tuple): a tuple containing:
            cat_shapes (list): list of categorical data classes shapes (N_patients, N_variables, N_max-classes)
            mask (np.array):  mask for patients with no measurments
            cat_all (np.array): 2D array of concatenated patients categorical data
    """    
    cat_shapes = list()
    first = 0

    for cat_d in cat_list:
        cat_shapes.append(cat_d.shape)
        cat_input = cat_d.reshape(cat_d.shape[0], -1)

        if first == 0:
            cat_all = cat_input
            del cat_input
            first = 1
        else:
            cat_all = np.concatenate((cat_all, cat_input), axis=1)

    # Make mask for patients with no measurments
    catsum = cat_all.sum(axis=1)
    mask = catsum > 5
    del catsum
    return cat_shapes, mask, cat_all


def concat_con_list(con_list):
    """
    Concatenate a list of continuous data

    Args:
        con_list (list): list with each continuous class data

    Returns:
        (tuple): a tuple containing:
            n_con_shapes (list): list of continuous data classes shapes (in 1D) (N_variables)
            mask:  mask for patients with no measurments
            con_all: 2D array of concatenated patients continuous data
    """    
    n_con_shapes = []

    first = 0
    for con_d in con_list:

        n_con_shapes.append(con_d.shape[1])

        if first == 0:
            con_all = con_d
            first = 1
        else:
            con_all = np.concatenate((con_all, con_d), axis=1)

    consum = con_all.sum(axis=1)
    mask &= consum != 0
    del consum
    return n_con_shapes, mask, con_all


def make_dataloader(cat_list=None, con_list=None, batchsize=10, cuda=False):
    """
    Create a DataLoader for input of each data type - categorical,
    continuous and potentially each omcis set (currently proteomics, target
    metabolomicas, untarget metabolomics and transcriptomics).

    Args:
        cat_list (list, optional): list of categorical input matrix (N_patients x N_variables x N_max-classes). Defaults to None.
        con_list (list, optional): list of normalized continuous input matrix (N_patients x N_variables). Defaults to None.
        batchsize (int, optional): Starting size of minibatches for dataloader. Defaults to 10.
        cuda (bool, optional): Pagelock memory of dataloader (use when using GPU acceleration). Defaults to False.

    Raises:
        ValueError: At least one type of data must be in the input

    Returns:
        (Dataloader): An object feeding data to the VAE
    """    

    if cat_list is None and con_list is None:
        raise ValueError("At least one type of data must be in the input")

    # Handle categorical data sets
    if not (cat_list is None):
        cat_shapes, mask, cat_all = concat_cat_list(cat_list)

    # else:
    mask = [True] * len(con_list[0])

    # Concetenate con datasetsand make final mask
    if not (con_list is None):
        n_con_shapes, mask, con_all = concat_con_list(con_list)

    # Create dataset
    if not (cat_list is None or con_list is None):
        cat_all = cat_all[mask]
        con_all = con_all[mask]

        cat_all = torch.from_numpy(cat_all)
        con_all = torch.from_numpy(con_all)

        dataset = MOVEDataset(
            con_all=con_all,
            con_shapes=n_con_shapes,
            cat_all=cat_all,
            cat_shapes=cat_shapes,
        )
    elif not (con_list is None):
        con_all = con_all[mask]
        con_all = torch.from_numpy(con_all)
        dataset = MOVEDataset(con_all=con_all, con_shapes=n_con_shapes)
    elif not (cat_list is None):
        cat_all = cat_all[mask]
        cat_all = torch.from_numpy(cat_all)
        dataset = MOVEDataset(cat_all=cat_all, cat_shapes=cat_shapes)
    # Create dataloader
    dataloader = DataLoader(
        dataset=dataset, batch_size=batchsize, drop_last=True, shuffle=True
    ) 
    return mask, dataloader
