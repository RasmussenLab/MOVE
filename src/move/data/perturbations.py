__all__ = ["perturb_categorical_data", "perturb_continuous_data"]

from typing import cast

import numpy as np
import torch
from torch.utils.data import DataLoader

from move.data.dataloaders import MOVEDataset
from move.data.preprocessing import feature_min_max


def perturb_categorical_data(
    baseline_dataloader: DataLoader,
    cat_dataset_names: list[str],
    target_dataset_name: str,
    target_value: np.ndarray,
) -> list[DataLoader]:
    """Add perturbations to categorical data. For each feature in the target
    dataset, change its value to target.

    Args:
        baseline_dataloader: Baseline dataloader
        cat_dataset_names: List of categorical dataset names
        target_dataset_name: Target categorical dataset to perturb
        target_value: Target value

    Returns:
        List of dataloaders containing all perturbed datasets
    """

    baseline_dataset = cast(MOVEDataset, baseline_dataloader.dataset)
    assert baseline_dataset.cat_shapes is not None
    assert baseline_dataset.cat_all is not None

    target_idx = cat_dataset_names.index(target_dataset_name)
    splits = np.cumsum(
        [0] + [int.__mul__(*shape) for shape in baseline_dataset.cat_shapes]
    )
    slice_ = slice(*splits[target_idx : target_idx + 2])

    target_shape = baseline_dataset.cat_shapes[target_idx]
    num_features = target_shape[0]  # CHANGE

    dataloaders = []
    for i in range(num_features):
        perturbed_cat = baseline_dataset.cat_all.clone()
        target_dataset = perturbed_cat[:, slice_].view(
            baseline_dataset.num_samples, *target_shape
        )
        target_dataset[:, i, :] = torch.FloatTensor(target_value)
        perturbed_dataset = MOVEDataset(
            perturbed_cat,
            baseline_dataset.con_all,
            baseline_dataset.cat_shapes,
            baseline_dataset.con_shapes,
        )
        perturbed_dataloader = DataLoader(
            perturbed_dataset,
            shuffle=False,
            batch_size=baseline_dataloader.batch_size,
        )
        dataloaders.append(perturbed_dataloader)
    return dataloaders


def perturb_continuous_data(
    baseline_dataloader: DataLoader,
    con_dataset_names: list[str],
    target_dataset_name: str,
    target_value: float,
) -> list[DataLoader]:
    """Add perturbations to continuous data. For each feature in the target
    dataset, change its value to target.

    Args:
        baseline_dataloader: Baseline dataloader
        con_dataset_names: List of continuous dataset names
        target_dataset_name: Target continuous dataset to perturb
        target_value: Target value

    Returns:
        List of dataloaders containing all perturbed datasets
    """

    baseline_dataset = cast(MOVEDataset, baseline_dataloader.dataset)
    assert baseline_dataset.con_shapes is not None
    assert baseline_dataset.con_all is not None

    target_idx = con_dataset_names.index(target_dataset_name)
    splits = np.cumsum([0] + baseline_dataset.con_shapes)
    slice_ = slice(*splits[target_idx : target_idx + 2])

    num_features = baseline_dataset.con_shapes[target_idx]

    dataloaders = []
    for i in range(num_features):
        perturbed_con = baseline_dataset.con_all.clone()
        target_dataset = perturbed_con[:, slice_]
        target_dataset[:, i] = torch.FloatTensor([target_value])
        perturbed_dataset = MOVEDataset(
            baseline_dataset.cat_all,
            perturbed_con,
            baseline_dataset.cat_shapes,
            baseline_dataset.con_shapes,
        )
        perturbed_dataloader = DataLoader(
            perturbed_dataset,
            shuffle=False,
            batch_size=baseline_dataloader.batch_size,
        )
        dataloaders.append(perturbed_dataloader)

    return dataloaders

def perturb_continuous_data_extended(
    baseline_dataloader: DataLoader,
    con_dataset_names: list[str],
    target_dataset_name: str,
    perturbation_type: str,
) -> list[DataLoader]:
    """Add perturbations to continuous data. For each feature in the target
    dataset, change the value to its minimum or maximum for all samples.

    Args:
        baseline_dataloader: Baseline dataloader
        con_dataset_names: List of continuous dataset names
        target_dataset_name: Target continuous dataset to perturb
        perturbation_type: 'minimum' or 'maximum'.

    Returns:
        List of dataloaders containing all perturbed datasets
    """

    baseline_dataset = cast(MOVEDataset, baseline_dataloader.dataset)
    assert baseline_dataset.con_shapes is not None
    assert baseline_dataset.con_all is not None

    target_idx = con_dataset_names.index(target_dataset_name) # dataset index 
    splits = np.cumsum([0] + baseline_dataset.con_shapes)
    slice_ = slice(*splits[target_idx : target_idx + 2])

    num_features = baseline_dataset.con_shapes[target_idx]
    dataloaders = []
    for i in range(num_features):
        perturbed_con = baseline_dataset.con_all.clone()
        perturbed_con = perturbed_con[:, slice_]
        # Change the desired feature by its standardized minimum value or maximum value
        min_feat_val_list, max_feat_val_list= feature_min_max(perturbed_con)
        if perturbation_type == 'minimum':
            perturbed_con[:, i] = torch.FloatTensor([min_feat_val_list[i]])
        elif perturbation_type == 'maximum':
            perturbed_con[:, i] = torch.FloatTensor([max_feat_val_list[i]])

        perturbed_dataset = MOVEDataset(
            baseline_dataset.cat_all,
            perturbed_con,
            baseline_dataset.cat_shapes,
            baseline_dataset.con_shapes,
        )

        perturbed_dataloader = DataLoader(
            perturbed_dataset,
            shuffle=False,
            batch_size=baseline_dataloader.batch_size,
        )
        dataloaders.append(perturbed_dataloader)
    return dataloaders