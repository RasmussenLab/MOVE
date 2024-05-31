__all__ = ["identify_associations"]

from functools import reduce
from os.path import exists
from pathlib import Path
from typing import Literal, Sized, Union, cast

import hydra
import numpy as np
import pandas as pd
import torch
from omegaconf import OmegaConf
from scipy.stats import ks_2samp, pearsonr  # type: ignore
from torch.utils.data import DataLoader

from move.analysis.metrics import get_2nd_order_polynomial
from move.conf.schema import (
    IdentifyAssociationsBayesConfig,
    IdentifyAssociationsConfig,
    IdentifyAssociationsKSConfig,
    IdentifyAssociationsTTestConfig,
    MOVEConfig,
)
from move.core.logging import get_logger
from move.core.typing import BoolArray, FloatArray, IntArray
from move.data import io
from move.data.dataloaders import MOVEDataset, make_dataloader
from move.data.perturbations import (
    ContinuousPerturbationType,
    perturb_categorical_data,
    perturb_continuous_data_extended,
)
from move.data.preprocessing import one_hot_encode_single
from move.models.vae import VAE
from move.visualization.dataset_distributions import (
    plot_correlations,
    plot_cumulative_distributions,
    plot_feature_association_graph,
    plot_reconstruction_movement,
)

TaskType = Literal["bayes", "ttest", "ks"]
CONTINUOUS_TARGET_VALUE = ["minimum", "maximum", "plus_std", "minus_std"]


def _get_task_type(
    task_config: IdentifyAssociationsConfig,
) -> TaskType:
    task_type = OmegaConf.get_type(task_config)
    if task_type is IdentifyAssociationsBayesConfig:
        return "bayes"
    if task_type is IdentifyAssociationsTTestConfig:
        return "ttest"
    if task_type is IdentifyAssociationsKSConfig:
        return "ks"
    raise ValueError("Unsupported type of task!")


def _validate_task_config(
    task_config: IdentifyAssociationsConfig, task_type: TaskType
) -> None:
    if not (0.0 <= task_config.sig_threshold <= 1.0):
        raise ValueError("Significance threshold must be within [0, 1].")
    if task_type == "ttest":
        task_config = cast(IdentifyAssociationsTTestConfig, task_config)
        if len(task_config.num_latent) != 4:
            raise ValueError("4 latent space dimensions required.")


def prepare_for_categorical_perturbation(
    config: MOVEConfig,
    interim_path: Path,
    baseline_dataloader: DataLoader,
    cat_list: list[FloatArray],
) -> tuple[
    list[DataLoader],
    BoolArray,
    BoolArray,
]:
    """
    This function creates the required dataloaders and masks
    for further categorical association analysis.

    Args:
        config: main configuration file
        interim_path: path where the intermediate outputs are saved
        baseline_dataloader: reference dataloader that will be perturbed
        cat_list: list of arrays with categorical data

    Returns:
        dataloaders: all dataloaders, including baseline appended last.
        nan_mask: mask for Nans
        feature_mask: masks the column for the perturbed feature.
    """

    # Read original data and create perturbed datasets
    task_config = cast(IdentifyAssociationsConfig, config.task)
    logger = get_logger(__name__)

    # Loading mappings:
    mappings = io.load_mappings(interim_path / "mappings.json")
    target_mapping = mappings[task_config.target_dataset]
    target_value = one_hot_encode_single(target_mapping, task_config.target_value)
    logger.debug(
        f"Target value: {task_config.target_value} => {target_value.astype(int)[0]}"
    )

    dataloaders = perturb_categorical_data(
        baseline_dataloader,
        config.data.categorical_names,
        task_config.target_dataset,
        target_value,
    )
    dataloaders.append(baseline_dataloader)

    baseline_dataset = cast(MOVEDataset, baseline_dataloader.dataset)

    assert baseline_dataset.con_all is not None
    orig_con = baseline_dataset.con_all
    nan_mask = (orig_con == 0).numpy()  # NaN values encoded as 0s
    logger.debug(f"# NaN values: {np.sum(nan_mask)}/{orig_con.numel()}")

    target_dataset_idx = config.data.categorical_names.index(task_config.target_dataset)
    target_dataset = cat_list[target_dataset_idx]
    feature_mask = np.all(target_dataset == target_value, axis=2)  # 2D: N x P
    feature_mask |= np.sum(target_dataset, axis=2) == 0

    return (
        dataloaders,
        nan_mask,
        feature_mask,
    )


def prepare_for_continuous_perturbation(
    config: MOVEConfig,
    output_subpath: Path,
    baseline_dataloader: DataLoader,
) -> tuple[
    list[DataLoader],
    BoolArray,
    BoolArray,
]:
    """
    This function creates the required dataloaders and masks
    for further continuous association analysis.

    Args:
        config:
            main configuration file.
        output_subpath:
            path where the output plots for continuous analysis are saved.
        baseline_dataloader:
            reference dataloader that will be perturbed.

    Returns:
        dataloaders:
            list with all dataloaders, including baseline appended last.
        nan_mask:
            mask for NaNs
        feature_mask:
            same as `nan_mask`, in this case.
    """

    # Read original data and create perturbed datasets
    logger = get_logger(__name__)
    task_config = cast(IdentifyAssociationsConfig, config.task)

    dataloaders = perturb_continuous_data_extended(
        baseline_dataloader,
        config.data.continuous_names,
        task_config.target_dataset,
        cast(ContinuousPerturbationType, task_config.target_value),
        output_subpath,
    )
    dataloaders.append(baseline_dataloader)

    baseline_dataset = cast(MOVEDataset, baseline_dataloader.dataset)

    assert baseline_dataset.con_all is not None
    orig_con = baseline_dataset.con_all
    nan_mask = (orig_con == 0).numpy()  # NaN values encoded as 0s
    logger.debug(f"# NaN values: {np.sum(nan_mask)}/{orig_con.numel()}")
    feature_mask = nan_mask

    return (dataloaders, nan_mask, feature_mask)


def _bayes_approach(
    config: MOVEConfig,
    task_config: IdentifyAssociationsBayesConfig,
    train_dataloader: DataLoader,
    baseline_dataloader: DataLoader,
    dataloaders: list[DataLoader],
    models_path: Path,
    num_perturbed: int,
    num_samples: int,
    num_continuous: int,
    nan_mask: BoolArray,
    feature_mask: BoolArray,
) -> tuple[Union[IntArray, FloatArray], ...]:

    assert task_config.model is not None
    device = torch.device("cuda" if task_config.model.cuda else "cpu")

    # Train models
    logger = get_logger(__name__)
    logger.info("Training models")
    mean_diff = np.zeros((num_perturbed, num_samples, num_continuous))
    normalizer = 1 / task_config.num_refits

    # Last appended dataloader is the baseline
    baseline_dataset = cast(MOVEDataset, baseline_dataloader.dataset)

    for j in range(task_config.num_refits):
        # Initialize model
        model: VAE = hydra.utils.instantiate(
            task_config.model,
            continuous_shapes=baseline_dataset.con_shapes,
            categorical_shapes=baseline_dataset.cat_shapes,
        )
        if j == 0:
            logger.debug(f"Model: {model}")

        # Train/reload model
        model_path = models_path / f"model_{task_config.model.num_latent}_{j}.pt"
        if model_path.exists():
            logger.debug(f"Re-loading refit {j + 1}/{task_config.num_refits}")
            model.load_state_dict(torch.load(model_path))
            model.to(device)
        else:
            logger.debug(f"Training refit {j + 1}/{task_config.num_refits}")
            model.to(device)
            hydra.utils.call(
                task_config.training_loop,
                model=model,
                train_dataloader=train_dataloader,
            )
            if task_config.save_refits:
                torch.save(model.state_dict(), model_path)
        model.eval()

        # Calculate baseline reconstruction
        _, baseline_recon = model.reconstruct(baseline_dataloader)
        min_feat, max_feat = np.zeros((num_perturbed, num_continuous)), np.zeros(
            (num_perturbed, num_continuous)
        )
        min_baseline, max_baseline = np.min(baseline_recon, axis=0), np.max(
            baseline_recon, axis=0
        )

        # Calculate perturb reconstruction => keep track of mean difference
        for i in range(num_perturbed):
            _, perturb_recon = model.reconstruct(dataloaders[i])
            diff = perturb_recon - baseline_recon  # 2D: N x C
            mean_diff[i, :, :] += diff * normalizer

            min_perturb, max_perturb = np.min(perturb_recon, axis=0), np.max(
                perturb_recon, axis=0
            )
            min_feat[i, :], max_feat[i, :] = np.min(
                [min_baseline, min_perturb], axis=0
            ), np.max([max_baseline, max_perturb], axis=0)

    # Calculate Bayes factors
    logger.info("Identifying significant features")
    bayes_k = np.empty((num_perturbed, num_continuous))
    bayes_mask = np.zeros(np.shape(bayes_k))
    for i in range(num_perturbed):
        mask = feature_mask[:, [i]] | nan_mask  # 2D: N x C
        diff = np.ma.masked_array(mean_diff[i, :, :], mask=mask)  # 2D: N x C
        prob = np.ma.compressed(np.mean(diff > 1e-8, axis=0))  # 1D: C
        bayes_k[i, :] = np.log(prob + 1e-8) - np.log(1 - prob + 1e-8)
        if task_config.target_value in CONTINUOUS_TARGET_VALUE:
            bayes_mask[i, :] = (
                baseline_dataloader.dataset.con_all[0, :]
                - dataloaders[i].dataset.con_all[0, :]
            )

    bayes_mask[bayes_mask != 0] = 1
    bayes_mask = np.array(bayes_mask, dtype=bool)

    # Calculate Bayes probabilities
    bayes_abs = np.abs(bayes_k)
    bayes_p = np.exp(bayes_abs) / (1 + np.exp(bayes_abs))  # 2D: N x C
    bayes_abs[bayes_mask] = np.min(
        bayes_abs
    )  # Bring feature_i feature_i associations to minimum
    sort_ids = np.argsort(bayes_abs, axis=None)[::-1]  # 1D: N x C
    prob = np.take(bayes_p, sort_ids)  # 1D: N x C
    logger.debug(f"Bayes proba range: [{prob[-1]:.3f} {prob[0]:.3f}]")

    # Sort Bayes
    bayes_k = np.take(bayes_k, sort_ids)  # 1D: N x C

    # Calculate FDR
    fdr = np.cumsum(1 - prob) / np.arange(1, prob.size + 1)  # 1D
    idx = np.argmin(np.abs(fdr - task_config.sig_threshold))
    logger.debug(f"FDR range: [{fdr[0]:.3f} {fdr[-1]:.3f}]")

    return sort_ids[:idx], prob[:idx], fdr[:idx], bayes_k[:idx]


def _ttest_approach(
    task_config: IdentifyAssociationsTTestConfig,
    train_dataloader: DataLoader,
    baseline_dataloader: DataLoader,
    dataloaders: list[DataLoader],
    models_path: Path,
    interim_path: Path,
    num_perturbed: int,
    num_samples: int,
    num_continuous: int,
    nan_mask: BoolArray,
    feature_mask: BoolArray,
) -> tuple[Union[IntArray, FloatArray], ...]:

    from scipy.stats import ttest_rel

    assert task_config.model is not None
    device = torch.device("cuda" if task_config.model.cuda else "cpu")

    # Train models
    logger = get_logger(__name__)
    logger.info("Training models")
    pvalues = np.empty(
        (
            len(task_config.num_latent),
            task_config.num_refits,
            num_perturbed,
            num_continuous,
        )
    )

    # Last appended dataloader is the baseline
    baseline_dataset = cast(MOVEDataset, baseline_dataloader.dataset)

    for k, num_latent in enumerate(task_config.num_latent):
        for j in range(task_config.num_refits):

            # Initialize model
            model: VAE = hydra.utils.instantiate(
                task_config.model,
                continuous_shapes=baseline_dataset.con_shapes,
                categorical_shapes=baseline_dataset.cat_shapes,
                num_latent=num_latent,
            )
            if j == 0:
                logger.debug(f"Model: {model}")

            # Train model
            model_path = models_path / f"model_{num_latent}_{j}.pt"
            if model_path.exists():
                logger.debug(f"Re-loading refit {j + 1}/{task_config.num_refits}")
                model.load_state_dict(torch.load(model_path))
                model.to(device)
            else:
                logger.debug(f"Training refit {j + 1}/{task_config.num_refits}")
                model.to(device)
                hydra.utils.call(
                    task_config.training_loop,
                    model=model,
                    train_dataloader=train_dataloader,
                )
                if task_config.save_refits:
                    torch.save(model.state_dict(), model_path)
            model.eval()

            # Get baseline reconstruction and baseline difference
            _, baseline_recon = model.reconstruct(baseline_dataloader)
            baseline_diff = np.empty((10, num_samples, num_continuous))
            for i in range(10):
                _, recon = model.reconstruct(baseline_dataloader)
                baseline_diff[i, :, :] = recon - baseline_recon
            baseline_diff = np.mean(baseline_diff, axis=0)  # 2D: N x C
            baseline_diff = np.where(nan_mask, np.nan, baseline_diff)

            # T-test between baseline and perturb difference
            for i in range(num_perturbed):
                _, perturb_recon = model.reconstruct(dataloaders[i])
                perturb_diff = perturb_recon - baseline_recon
                mask = feature_mask[:, [i]] | nan_mask  # 2D: N x C
                _, pvalues[k, j, i, :] = ttest_rel(
                    a=np.where(mask, np.nan, perturb_diff),
                    b=np.where(mask, np.nan, baseline_diff),
                    axis=0,
                    nan_policy="omit",
                )

    # Correct p-values (Bonferroni)
    pvalues = np.minimum(pvalues * num_continuous, 1.0)
    np.save(interim_path / "pvals.npy", pvalues)

    # Find significant hits
    overlap_thres = task_config.num_refits // 2
    reject = pvalues <= task_config.sig_threshold  # 4D: L x R x P x C
    overlap = reject.sum(axis=1) >= overlap_thres  # 3D: L x P x C
    sig_ids = overlap.sum(axis=0) >= 3  # 2D: P x C
    sig_ids = np.flatnonzero(sig_ids)  # 1D

    # Report median p-value
    masked_pvalues = np.ma.masked_array(pvalues, mask=~reject)  # 4D
    masked_pvalues = np.ma.median(masked_pvalues, axis=1)  # 3D
    masked_pvalues = np.ma.median(masked_pvalues, axis=0)  # 2D
    sig_pvalues = np.ma.compressed(np.take(masked_pvalues, sig_ids))  # 1D

    return sig_ids, sig_pvalues


def _ks_approach(
    config: MOVEConfig,
    task_config: IdentifyAssociationsKSConfig,
    train_dataloader: DataLoader,
    baseline_dataloader: DataLoader,
    dataloaders: list[DataLoader],
    models_path: Path,
    num_perturbed: int,
    num_samples: int,
    num_continuous: int,
    con_names: list[list[str]],
    output_path: Path,
) -> tuple[Union[IntArray, FloatArray], ...]:
    """
    Find associations between continuous features using Kolmogorov-Smirnov distances.
    When perturbing feature A, this function measures the shift of the reconstructed
    distribution for feature B (over samples) from 1) the baseline reconstruction to 2)
    the reconstruction when perturbing A.

    If A and B are related the perturbation of A in the input will lead to a change in
    feature B's reconstruction, that will be measured by KS distance.

    Associations are then ranked according to KS distance (absolute value).


    Args:
        config: MOVE main configuration.
        task_config: IdentifyAssociationsKSConfig configuration.
        train_dataloader: training DataLoader.
        baseline_dataloader: unperturbed DataLoader.
        dataloaders: list of DataLoaders where DataLoader[i] is obtained by perturbing feature i
                     in the target dataset.
        models_path: path to the models.
        num_perturbed: number of perturbed features.
        num_samples: total number of samples
        num_continuous: number of continuous features (all continuous datasets concatenated).
        con_names: list of lists where eah inner list contains the feature names of a specific continuous dataset
        output_path: path where QC summary metrics will be saved.

    Returns:
        sort_ids: list with flattened IDs of the associations above the significance threshold.
        ks_distance: Ordered list with signed KS scores. KS scores quantify the direction and
                     magnitude of the shift in feature B's reconstruction when perturbing feature A.


    !!! Note !!!:

    The sign of the KS score can be misleading: negative sign means positive shift.
    since the cumulative distribution starts growing later and is found below the reference
    (baseline). Hence:
    a) with plus_std, negative sign means a positive correlation.
    b) with minus_std, negative sign means a negative correlation.
    """

    assert task_config.model is not None
    device = torch.device("cuda" if task_config.model.cuda else "cpu")
    figure_path = output_path / "figures"
    figure_path.mkdir(exist_ok=True, parents=True)

    # Data containers
    stats = np.empty((task_config.num_refits, num_perturbed, num_continuous))
    stat_signs = np.empty_like(stats)
    rec_corr, slope = np.empty((task_config.num_refits, num_continuous)), np.empty(
        (task_config.num_refits, num_continuous)
    )
    ks_mask = np.zeros((num_perturbed, num_continuous))
    latent_matrix = np.empty(
        (num_samples, task_config.model.num_latent, len(dataloaders))
    )

    # Last appended dataloader is the baseline
    baseline_dataset = cast(MOVEDataset, baseline_dataloader.dataset)

    # Train models
    logger = get_logger(__name__)
    logger.info("Training models")

    target_dataset_idx = config.data.continuous_names.index(task_config.target_dataset)
    perturbed_names = con_names[target_dataset_idx]

    for j in range(task_config.num_refits):  # Train num_refits models

        # Initialize model
        model: VAE = hydra.utils.instantiate(
            task_config.model,
            continuous_shapes=baseline_dataset.con_shapes,
            categorical_shapes=baseline_dataset.cat_shapes,
        )
        if j == 0:
            logger.debug(f"Model: {model}")

        # Train/reload model
        model_path = models_path / f"model_{task_config.model.num_latent}_{j}.pt"
        if model_path.exists():
            logger.debug(f"Re-loading refit {j + 1}/{task_config.num_refits}")
            model.load_state_dict(torch.load(model_path))
            model.to(device)
        else:
            logger.debug(f"Training refit {j + 1}/{task_config.num_refits}")
            model.to(device)
            hydra.utils.call(
                task_config.training_loop,
                model=model,
                train_dataloader=train_dataloader,
            )
            if task_config.save_refits:
                torch.save(model.state_dict(), model_path)
        model.eval()

        # Calculate baseline reconstruction
        _, baseline_recon = model.reconstruct(baseline_dataloader)
        min_feat = np.zeros((num_perturbed, num_continuous))
        max_feat = np.zeros((num_perturbed, num_continuous))
        min_baseline = np.min(baseline_recon, axis=0)
        max_baseline = np.max(baseline_recon, axis=0)

        # QC of feature's reconstruction ##############################
        logger.debug("Calculating quality control of the feature reconstructions")
        # Correlation and slope for each feature's reconstruction
        feature_names = reduce(list.__add__, con_names)

        for k in range(num_continuous):
            x = baseline_dataloader.dataset.con_all.numpy()[:, k]  # baseline_recon[:,i]
            y = baseline_recon[:, k]
            x_pol, y_pol, (a2, a1, a) = get_2nd_order_polynomial(x, y)
            slope[j, k] = a1
            rec_corr[j, k] = pearsonr(x, y).statistic

            if (
                feature_names[k] in task_config.perturbed_feature_names
                or feature_names[k] in task_config.target_feature_names
            ):

                # Plot correlations
                fig = plot_correlations(x, y, x_pol, y_pol, a2, a1, a, k)
                fig.savefig(
                    figure_path
                    / f"Input_vs_reconstruction_correlation_feature_{k}_refit_{j}.png",
                    dpi=50,
                )

        # Calculate perturbed reconstruction and shifts #############################
        logger.debug("Computing KS scores")

        # Save original latent space for first refit:
        if j == 0:
            latent = model.project(baseline_dataloader)
            latent_matrix[:, :, -1] = latent

        for i, pert_feat in enumerate(perturbed_names):
            _, perturb_recon = model.reconstruct(dataloaders[i])
            min_perturb = np.min(perturb_recon, axis=0)
            max_perturb = np.max(perturb_recon, axis=0)
            min_feat[i, :] = np.min([min_baseline, min_perturb], axis=0)
            max_feat[i, :] = np.max([max_baseline, max_perturb], axis=0)

            # Save latent representation for perturbed samples
            if j == 0:
                latent_pert = model.project(dataloaders[i])
                latent_matrix[:, :, i] = latent_pert

            for k, targ_feat in enumerate(feature_names):
                # Calculate ks factors: measure distance between baseline and perturbed
                # reconstruction distributions per feature (k)
                res = ks_2samp(perturb_recon[:, k], baseline_recon[:, k])
                stats[j, i, k] = res.statistic
                stat_signs[j, i, k] = res.statistic_sign

                if (
                    pert_feat in task_config.perturbed_feature_names
                    and targ_feat in task_config.target_feature_names
                ):

                    # Plotting preliminary results:
                    n_bins = 50
                    hist_base, edges = np.histogram(
                        baseline_recon[:, k],
                        bins=np.linspace(min_feat[i, k], max_feat[i, k], n_bins),
                        density=True,
                    )
                    hist_pert, edges = np.histogram(
                        perturb_recon[:, k],
                        bins=np.linspace(min_feat[i, k], max_feat[i, k], n_bins),
                        density=True,
                    )

                    # Cumulative distribution:
                    fig = plot_cumulative_distributions(
                        edges,
                        hist_base,
                        hist_pert,
                        f"Cumulative_perturbed_{i}_measuring_{k}_stats_{stats[j, i, k]}",
                    )
                    fig.savefig(
                        figure_path
                        / f"Cumulative_refit_{j}_perturbed_{i}_measuring_{k}_stats_{stats[j, i, k]}.png"
                    )

                    # Feature changes:
                    fig = plot_reconstruction_movement(baseline_recon, perturb_recon, k)
                    fig.savefig(
                        figure_path / f"Changes_pert_{i}_on_feat_{k}_refit_{j}.png"
                    )

    # Save latent space matrix:
    np.save(output_path / "latent_location.npy", latent_matrix)
    np.save(output_path / "perturbed_features_list.npy", np.array(perturbed_names))

    # Creating a mask for self associations
    logger.debug("Creating self-association mask")
    for i in range(num_perturbed):
        if task_config.target_value in CONTINUOUS_TARGET_VALUE:
            ks_mask[i, :] = (
                baseline_dataloader.dataset.con_all[0, :]
                - dataloaders[i].dataset.con_all[0, :]
            )
    ks_mask[ks_mask != 0] = 1
    ks_mask = np.array(ks_mask, dtype=bool)

    # Take the median of KS values (with sign) over refits.
    final_stats = np.nanmedian(stats * stat_signs, axis=0)
    final_stats[ks_mask] = (
        0.0  # Zero all masked values, placing them at end of the ranking
    )

    # KS-threshold:
    ks_thr = np.sqrt(-np.log(task_config.sig_threshold / 2) * 1 / (num_samples))
    logger.info(f"Suggested absolute KS threshold is: {ks_thr}")

    # Sort associations by absolute KS value
    sort_ids = np.argsort(abs(final_stats), axis=None)[::-1]  # 1D: N x C
    ks_distance = np.take(final_stats, sort_ids)  # 1D: N x C

    # Writing Quality control csv file. Mean slope and correlation over refits as qc metrics.
    logger.info("Writing QC file")
    qc_df = pd.DataFrame({"Feature names": feature_names})
    qc_df["slope"] = np.nanmean(slope, axis=0)
    qc_df["reconstruction_correlation"] = np.nanmean(rec_corr, axis=0)
    qc_df.to_csv(output_path / "QC_summary_KS.tsv", sep="\t", index=False)

    # Return first idx associations: redefined for reasonable threshold

    return sort_ids[abs(ks_distance) >= ks_thr], ks_distance[abs(ks_distance) >= ks_thr]


def save_results(
    config: MOVEConfig,
    con_shapes: list[int],
    cat_names: list[list[str]],
    con_names: list[list[str]],
    output_path: Path,
    sig_ids,
    extra_cols,
    extra_colnames,
) -> None:
    """
    This function saves the obtained associations in a TSV file containing
    the following columns:
        feature_a_id
        feature_b_id
        feature_a_name
        feature_b_name
        feature_b_dataset
        proba/p_value: number quantifying the significance of the association

    Args:
        config: main config
        con_shapes: tuple with the number of features per continuous dataset
        cat_names: list of lists of names for the categorical features.
                   Each inner list corresponds to a separate dataset.
        con_names: list of lists of names for the continuous features.
                   Each inner list corresponds to a separate dataset.
        output_path: path where the results will be saved
        sig_ids: ids for the significat features
        extra_cols: extra data when calling the approach function
        extra_colnames: names for the extra data columns
    """
    logger = get_logger(__name__)
    logger.info(f"Significant hits found: {sig_ids.size}")
    task_config = cast(IdentifyAssociationsConfig, config.task)
    task_type = _get_task_type(task_config)

    num_continuous = sum(con_shapes)  # C

    if sig_ids.size > 0:
        sig_ids = np.vstack((sig_ids // num_continuous, sig_ids % num_continuous)).T
        logger.info("Writing results")
        results = pd.DataFrame(sig_ids, columns=["feature_a_id", "feature_b_id"])

        # Check if the task is for continuous or categorical data
        if task_config.target_value in CONTINUOUS_TARGET_VALUE:
            target_dataset_idx = config.data.continuous_names.index(
                task_config.target_dataset
            )
            a_df = pd.DataFrame(dict(feature_a_name=con_names[target_dataset_idx]))
        else:
            target_dataset_idx = config.data.categorical_names.index(
                task_config.target_dataset
            )
            a_df = pd.DataFrame(dict(feature_a_name=cat_names[target_dataset_idx]))
        a_df.index.name = "feature_a_id"
        a_df.reset_index(inplace=True)
        feature_names = reduce(list.__add__, con_names)
        b_df = pd.DataFrame(dict(feature_b_name=feature_names))
        b_df.index.name = "feature_b_id"
        b_df.reset_index(inplace=True)
        results = results.merge(a_df, on="feature_a_id", how="left").merge(
            b_df, on="feature_b_id", how="left"
        )
        results["feature_b_dataset"] = pd.cut(
            results["feature_b_id"],
            bins=cast(list[int], np.cumsum([0] + con_shapes)),
            right=False,
            labels=config.data.continuous_names,
        )
        for col, colname in zip(extra_cols, extra_colnames):
            results[colname] = col
        results.to_csv(
            output_path / f"results_sig_assoc_{task_type}.tsv", sep="\t", index=False
        )


def identify_associations(config: MOVEConfig) -> None:
    """
    Leads to the execution of the appropriate association
    identification tasks. The function is organized in three
    blocks:
        1) Prepare the data and create the dataloaders with their masks.
        2) Evaluate associations using bayes or ttest approach.
        3) Save results.
    """
    # DATA PREPARATION ######################
    # Read original data and create perturbed datasets####

    logger = get_logger(__name__)
    task_config = cast(IdentifyAssociationsConfig, config.task)
    task_type = _get_task_type(task_config)
    _validate_task_config(task_config, task_type)

    interim_path = Path(config.data.interim_data_path)

    models_path = interim_path / "models"
    if task_config.save_refits:
        models_path.mkdir(exist_ok=True)

    output_path = Path(config.data.results_path) / "identify_associations"
    output_path.mkdir(exist_ok=True, parents=True)

    # Load datasets:
    cat_list, cat_names, con_list, con_names = io.load_preprocessed_data(
        interim_path,
        config.data.categorical_names,
        config.data.continuous_names,
    )

    train_dataloader = make_dataloader(
        cat_list,
        con_list,
        shuffle=True,
        batch_size=task_config.batch_size,
        drop_last=True,
    )

    con_shapes = [con.shape[1] for con in con_list]

    num_samples = len(cast(Sized, train_dataloader.sampler))  # N
    num_continuous = sum(con_shapes)  # C
    logger.debug(f"# continuous features: {num_continuous}")

    # Creating the baseline dataloader:
    baseline_dataloader = make_dataloader(
        cat_list, con_list, shuffle=False, batch_size=task_config.batch_size
    )

    # Indentify associations between continuous features:
    logger.info(f"Perturbing dataset: '{task_config.target_dataset}'")
    if task_config.target_value in CONTINUOUS_TARGET_VALUE:
        logger.info(f"Beginning task: identify associations continuous ({task_type})")
        logger.info(f"Perturbation type: {task_config.target_value}")
        output_subpath = Path(output_path) / "perturbation_visualization"
        output_subpath.mkdir(exist_ok=True, parents=True)
        (
            dataloaders,
            nan_mask,
            feature_mask,
        ) = prepare_for_continuous_perturbation(
            config, output_subpath, baseline_dataloader
        )

    # Identify associations between categorical and continuous features:
    else:
        logger.info("Beginning task: identify associations categorical")
        (
            dataloaders,
            nan_mask,
            feature_mask,
        ) = prepare_for_categorical_perturbation(
            config, interim_path, baseline_dataloader, cat_list
        )

    num_perturbed = len(dataloaders) - 1  # P
    logger.debug(f"# perturbed features: {num_perturbed}")

    # APPROACH EVALUATION ##########################

    if task_type == "bayes":
        task_config = cast(IdentifyAssociationsBayesConfig, task_config)
        sig_ids, *extra_cols = _bayes_approach(
            config,
            task_config,
            train_dataloader,
            baseline_dataloader,
            dataloaders,
            models_path,
            num_perturbed,
            num_samples,
            num_continuous,
            nan_mask,
            feature_mask,
        )

        extra_colnames = ["proba", "fdr", "bayes_k"]

    elif task_type == "ttest":
        task_config = cast(IdentifyAssociationsTTestConfig, task_config)
        sig_ids, *extra_cols = _ttest_approach(
            task_config,
            train_dataloader,
            baseline_dataloader,
            dataloaders,
            models_path,
            interim_path,
            num_perturbed,
            num_samples,
            num_continuous,
            nan_mask,
            feature_mask,
        )

        extra_colnames = ["p_value"]

    elif task_type == "ks":
        task_config = cast(IdentifyAssociationsKSConfig, task_config)
        sig_ids, *extra_cols = _ks_approach(
            config,
            task_config,
            train_dataloader,
            baseline_dataloader,
            dataloaders,
            models_path,
            num_perturbed,
            num_samples,
            num_continuous,
            con_names,
            output_path,
        )

        extra_colnames = ["ks_distance"]

    else:
        raise ValueError()

    # RESULTS ################################
    save_results(
        config,
        con_shapes,
        cat_names,
        con_names,
        output_path,
        sig_ids,
        extra_cols,
        extra_colnames,
    )

    if exists(output_path / f"results_sig_assoc_{task_type}.tsv"):
        association_df = pd.read_csv(
            output_path / f"results_sig_assoc_{task_type}.tsv", sep="\t"
        )
        _ = plot_feature_association_graph(association_df, output_path)
        _ = plot_feature_association_graph(association_df, output_path, layout="spring")
