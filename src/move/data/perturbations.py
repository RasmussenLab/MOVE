__all__ = ["perturb_data"]

from typing import Any
import numpy as np
import torch
from torch.utils.data import DataLoader

from move.data.dataloaders import MOVEDataset, make_dataloader
from move.utils.model_utils import get_start_end_positions


def perturb_data(
    cat_list: list[np.ndarray],
    cat_names: list[str],
    con_list: list[np.ndarray],
    target_dataset_name: str,
    target_value: Any,
) -> list[DataLoader]:
    target_idx = cat_names.index(target_dataset_name)
    target_shape = cat_list[target_idx].shape
    num_samples = cat_list[target_idx].shape[0]
    _, baseline_dataloader = make_dataloader(
        cat_list, con_list, shuffle=False, batch_size=num_samples
    )
    start, end = get_start_end_positions(cat_list, cat_names, target_dataset_name)
    cat_all: torch.Tensor = baseline_dataloader.dataset.cat_all
    num_features = target_shape[1]

    dataloaders = []
    for i in range(num_features):
        perturbed_cat = cat_all.clone()
        target_dataset = perturbed_cat[:, start:end].view(*target_shape)
        target_dataset[:, i, :] = target_value
        perturbed_dataset = MOVEDataset(
            perturbed_cat,
            baseline_dataloader.dataset.con_all,
            baseline_dataloader.dataset.con_shapes,
            baseline_dataloader.dataset.cat_shapes,
        )
        perturbed_dataloader = DataLoader(
            perturbed_dataset, shuffle=False, batch_size=num_samples
        )
        dataloaders.append(perturbed_dataloader)
    dataloaders.append(baseline_dataloader)
    return dataloaders
