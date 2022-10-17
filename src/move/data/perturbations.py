__all__ = ["perturb_data"]

from typing import cast

import numpy as np
import torch
from torch.utils.data import DataLoader

from move.data.dataloaders import MOVEDataset


def perturb_data(
    baseline_dataloader: DataLoader,
    cat_dataset_names: list[str],
    target_dataset_name: str,
    target_value: np.ndarray,
) -> list[DataLoader]:
    """Add perturbations to data. For each feature in the target
    categorical dataset change its value to target

    Args:
        cat_list: List of categorical datasets
        baseline_dataloader: Baseline dataloader
        cat_dataset_names: List of categorical dataset names
        target_dataset_name: Target categorical dataset to perturb
        target_value: Target value

    Returns:
        List of dataloaders, containing all perturbed datasets
    """

    baseline_dataset = cast(MOVEDataset, baseline_dataloader.dataset)
    assert baseline_dataset.cat_shapes is not None and baseline_dataset.cat_all is not None

    target_idx = cat_dataset_names.index(target_dataset_name)
    splits = np.cumsum([0] + [int.__mul__(*shape[1:]) for shape in baseline_dataset.cat_shapes])
    slice_ = slice(*splits[target_idx:target_idx + 2])

    target_shape = baseline_dataset.cat_shapes[target_idx]
    num_features = target_shape[1]  # CHANGE

    dataloaders = []
    for i in range(num_features):
        perturbed_cat = baseline_dataset.cat_all.clone()
        target_dataset = perturbed_cat[:, slice_].view(*target_shape)
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


