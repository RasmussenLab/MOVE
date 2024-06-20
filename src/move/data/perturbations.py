__all__ = [
    "perturb_categorical_data",
    "perturb_continuous_data",
    "perturb_continuous_data_extended_one",
    "perturb_continuous_data_extended",
]

from pathlib import Path
from typing import Literal, Optional, cast

import numpy as np
import torch
from torch.utils.data import DataLoader

from move.core.logging import get_logger
from move.data.dataloaders import MOVEDataset
from move.data.preprocessing import feature_stats
from move.visualization.dataset_distributions import plot_value_distributions

ContinuousPerturbationType = Literal["minimum", "maximum", "plus_std", "minus_std"]


# Also in analyze_latent.py
def perturb_continuous_data_one(
    baseline_dataloader: DataLoader,
    con_dataset_names: list[str],
    target_dataset_name: str,
    target_value: float,
    index_pert_feat: int,  # Index of the datasetto perturb
) -> DataLoader:  # change list(DataLoader) to just one DataLoader
    """Add perturbations to continuous data. For each feature in the target
    dataset, change its value to target.

    Args:
        baseline_dataloader: Baseline dataloader
        con_dataset_names: List of continuous dataset names
        target_dataset_name: Target continuous dataset to perturb
        target_value: Target value. In analyze_latent, it will be 0

    Returns:
        One dataloader, with the ith dataset perturbed
    """

    baseline_dataset = cast(MOVEDataset, baseline_dataloader.dataset)
    assert baseline_dataset.con_shapes is not None
    assert baseline_dataset.con_all is not None

    target_idx = con_dataset_names.index(target_dataset_name)
    splits = np.cumsum([0] + baseline_dataset.con_shapes)
    slice_ = slice(*splits[target_idx : target_idx + 2])

    num_features = baseline_dataset.con_shapes[target_idx]
    # dataloaders = []
    i = index_pert_feat
    # Instead of the loop, we do it only for one
    # for i in range(num_features):
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

    return perturbed_dataloader


def perturb_categorical_data_one(
    baseline_dataloader: DataLoader,
    cat_dataset_names: list[str],
    target_dataset_name: str,
    target_value: np.ndarray,
    index_pert_feat: int,
) -> DataLoader:
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
    num_features = target_shape[0]

    i = index_pert_feat
    # dataloaders = []
    # for i in range(num_features):
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

    return perturbed_dataloader


def perturb_continuous_data_extended_one(  # We will keep the input almost the same, to make everything easier
    # However, I have to introduce a variable that allows me to index the specific dataloader I want to create (index_pert_feat)
    # And I eliminate the output directory, an image is not generated now (too many features)
    baseline_dataloader: DataLoader,
    con_dataset_names: list[str],
    target_dataset_name: str,
    perturbation_type: ContinuousPerturbationType,
    index_pert_feat: int,
) -> (
    DataLoader
):  # But we change the output from list[DataLoader] to just one DataLoader
    logger = get_logger(__name__)
    """Add perturbations to continuous data. For each feature in the target
    dataset, change the feature's value in all samples (in rows):
    1,2) substituting this feature in all samples by the feature's minimum/maximum value.
    3,4) Adding/Substracting one standard deviation to the sample's feature value.

    Args:
        baseline_dataloader: Baseline dataloader
        con_dataset_names: List of continuous dataset names
        target_dataset_name: Target continuous dataset to perturb
        perturbation_type: 'minimum', 'maximum', 'plus_std' or 'minus_std'.
        #output_subpath: path where the figure showing the perturbation will be saved
        index_pert_feat: Index we want to perturb

    Returns:
        - Dataloader with the ith feature (index_pert_feat) perturbed.

    Note:
        This function was created so that it could generalize to non-normalized
        datasets. Scaling is done per dataset, not per feature -> slightly different stds
        feature to feature.
    """
    logger.debug(
        f"Inside perturb_continuous_data_extended_one for feature {index_pert_feat}"
    )

    baseline_dataset = cast(MOVEDataset, baseline_dataloader.dataset)
    assert baseline_dataset.con_shapes is not None
    assert baseline_dataset.con_all is not None

    target_idx = con_dataset_names.index(target_dataset_name)  # dataset index
    splits = np.cumsum([0] + baseline_dataset.con_shapes)
    slice_ = slice(*splits[target_idx : target_idx + 2])

    # Use it only if we want to perturb all features in the target dataset
    num_features = baseline_dataset.con_shapes[target_idx]

    # Now, instead of the for loop that iterates over all the features we want to perturb, we do it only for one feature, the one
    # indicated in index_pert_feat
    logger.debug(f"Setting up perturbed_con for feature {index_pert_feat}")

    perturbed_con = baseline_dataset.con_all.clone()
    target_dataset = perturbed_con[:, slice_]

    logger.debug(
        f"Changing to desired perturbation value for feature {index_pert_feat}"
    )
    # Change the desired feature value by:
    min_feat_val_list, max_feat_val_list, std_feat_val_list = feature_stats(
        target_dataset
    )
    if perturbation_type == "minimum":
        target_dataset[:, index_pert_feat] = torch.FloatTensor(
            [min_feat_val_list[index_pert_feat]]
        )
    elif perturbation_type == "maximum":
        target_dataset[:, index_pert_feat] = torch.FloatTensor(
            [max_feat_val_list[index_pert_feat]]
        )
    elif perturbation_type == "plus_std":
        target_dataset[:, index_pert_feat] += torch.FloatTensor(
            [std_feat_val_list[index_pert_feat]]
        )
    elif perturbation_type == "minus_std":
        target_dataset[:, index_pert_feat] -= torch.FloatTensor(
            [std_feat_val_list[index_pert_feat]]
        )
    logger.debug(f"Perturbation succesful for feature {index_pert_feat}")
    # We used this for a plot, so no need to create it now
    # perturbations_list.append(target_dataset[:, i].numpy())

    logger.debug(
        f"Creating perturbed dataset and dataloader for feature {index_pert_feat}"
    )

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
    # dataloaders.append(perturbed_dataloader)

    logger.debug(
        f"Finished perturb_continuous_data_extended_one for feature {index_pert_feat}"
    )

    return perturbed_dataloader


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
    num_features = target_shape[0]

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
    perturbation_type: ContinuousPerturbationType,
    output_subpath: Optional[Path] = None,
) -> list[DataLoader]:
    """Add perturbations to continuous data. For each feature in the target
    dataset, change the feature's value in all samples (in rows):
    1,2) substituting this feature in all samples by the feature's minimum/maximum value
    3,4) Adding/Substracting one standard deviation to the sample's feature value

    Args:
        baseline_dataloader: Baseline dataloader
        con_dataset_names: List of continuous dataset names
        target_dataset_name: Target continuous dataset to perturb
        perturbation_type: 'minimum', 'maximum', 'plus_std' or 'minus_std'.
        output_subpath: path where the figure showing the perturbation will be saved

    Returns:
        - List of dataloaders containing all perturbed datasets
        - Plot of the feature value distribution after the perturbation. Note that
          all perturbations are collapsed into one single plot.

    Note:
        This function was created so that it could generalize to non-normalized
        datasets. Scaling is done per dataset, not per feature -> slightly different stds
        feature to feature.
    """
    logger = get_logger(__name__)
    logger.debug("Inside perturb_extended, creating baseline dataset")
    baseline_dataset = cast(MOVEDataset, baseline_dataloader.dataset)
    assert baseline_dataset.con_shapes is not None
    assert baseline_dataset.con_all is not None

    logger.debug("Creating target_ics, splits, and slice")
    target_idx = con_dataset_names.index(target_dataset_name)  # dataset index
    splits = np.cumsum([0] + baseline_dataset.con_shapes)
    slice_ = slice(*splits[target_idx : target_idx + 2])

    num_features = baseline_dataset.con_shapes[target_idx]
    dataloaders = []
    perturbations_list = []

    for i in range(num_features):
        logger.debug(f"Getting perturbed dataset for feature {i}")
        perturbed_con = baseline_dataset.con_all.clone()
        target_dataset = perturbed_con[:, slice_]
        # Change the desired feature value by:
        min_feat_val_list, max_feat_val_list, std_feat_val_list = feature_stats(
            target_dataset
        )
        if perturbation_type == "minimum":
            target_dataset[:, i] = torch.FloatTensor([min_feat_val_list[i]])
        elif perturbation_type == "maximum":
            target_dataset[:, i] = torch.FloatTensor([max_feat_val_list[i]])
        elif perturbation_type == "plus_std":
            target_dataset[:, i] += torch.FloatTensor([std_feat_val_list[i]])
        elif perturbation_type == "minus_std":
            target_dataset[:, i] -= torch.FloatTensor([std_feat_val_list[i]])

        # perturbations_list.append(target_dataset[:, i].numpy())

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
    logger.debug("Finished perturb_continuous_data_extended function")

    # Plot the perturbations for all features, collapsed in one plot:
    # if output_subpath is not None:
    #   fig = plot_value_distributions(np.array(perturbations_list).transpose())
    #  fig_path = str(
    #     output_subpath / f"perturbation_distribution_{target_dataset_name}.png"
    # )
    # fig.savefig(fig_path)

    # Plot the perturbations for all features, collapsed in one plot:
    if output_subpath is not None:
        fig = plot_value_distributions(np.array(perturbations_list).transpose())
        fig_path = str(
            output_subpath / f"perturbation_distribution_{target_dataset_name}.png"
        )
        fig.savefig(fig_path)

    return dataloaders
