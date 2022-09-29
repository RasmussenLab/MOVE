__all__ = ["identify_associations"]

from functools import reduce
from pathlib import Path

import hydra
import numpy as np
import pandas as pd
from omegaconf import OmegaConf

from move.conf.schema import (
    IdentifyAssociationsConfig,
    IdentifyAssociationsBayesConfig,
    IdentifyAssociationsTTestConfig,
    MOVEConfig,
)
from move.core.logging import get_logger
from move.data import io
from move.data.dataloaders import make_dataloader
from move.data.perturbations import perturb_data
from move.data.preprocessing import one_hot_encode_single
from move.models.vae import VAE
from move.training.training_loop import TrainingLoopOutput


def identify_associations(config: MOVEConfig):
    """Trains multiple models to identify associations between the dataset
    of interest and the continuous datasets."""

    logger = get_logger(__name__)
    task_config: IdentifyAssociationsConfig = config.task

    interim_path = Path(config.data.interim_data_path)
    output_path = Path(config.data.processed_data_path) / "identify_associations"
    output_path.mkdir(exist_ok=True, parents=True)

    # Read original data and create perturbed datasets
    logger.info(f"Perturbing dataset: '{task_config.target_dataset}'")
    cat_list, cat_names, con_list, con_names = io.read_data(config)

    mappings = io.load_mappings(interim_path / "mappings.json")
    target_mapping = mappings[task_config.target_dataset]
    target_value = one_hot_encode_single(target_mapping, task_config.target_value)
    logger.debug(
        f"Target value: {task_config.target_value} => {target_value.astype(int)[0]}"
    )

    train_mask, train_dataloader = make_dataloader(
        cat_list,
        con_list,
        shuffle=True,
        batch_size=task_config.batch_size,
        drop_last=True,
    )
    logger.debug(f"Masked training samples: {np.sum(~train_mask)}/{train_mask.size}")

    con_shapes = [con.shape[1] for con in con_list]
    dataloaders = perturb_data(
        cat_list,
        con_list,
        config.data.categorical_names,
        task_config.target_dataset,
        target_value,
    )

    baseline_dataloader = dataloaders[-1]
    num_features = len(dataloaders) - 1  # F
    num_samples = len(baseline_dataloader.sampler)  # N
    num_continuous = sum(con_shapes)  # C
    logger.debug(f"# perturbed features: {num_features}")
    logger.debug(f"# continuous features: {num_continuous}")

    orig_con = baseline_dataloader.dataset.con_all
    nan_mask = (orig_con == 0).numpy()  # NaN values encoded as 0s
    logger.debug(f"# NaN values: {np.sum(nan_mask)}/{orig_con.numel()}")

    target_dataset_idx = config.data.categorical_names.index(task_config.target_dataset)
    target_dataset = cat_list[target_dataset_idx]
    feature_mask = np.all(target_dataset == target_value, axis=2)
    feature_mask |= np.sum(target_dataset, axis=2) == 0

    def _bayes_approach(task_config: IdentifyAssociationsBayesConfig) -> np.ndarray:
        # Train models
        logger.info("Training models")
        mean_diff = np.zeros((num_features, num_samples, num_continuous))
        normalizer = 1 / task_config.num_refits
        for j in range(task_config.num_refits):

            # Initialize model
            model: VAE = hydra.utils.instantiate(
                task_config.model,
                continuous_shapes=baseline_dataloader.dataset.con_shapes,
                categorical_shapes=baseline_dataloader.dataset.cat_shapes,
            )
            if j == 0:
                logger.debug(f"Model: {model}")

            # Train model
            logger.debug(f"Training refit {j + 1}/{task_config.num_refits}")
            _: TrainingLoopOutput = hydra.utils.call(
                task_config.training_loop,
                model=model,
                train_dataloader=train_dataloader,
            )
            model.eval()

            # Calculate baseline reconstruction
            _, baseline_recon = model.reconstruct(baseline_dataloader)
            # Calculate perturb reconstruction => keep track of mean difference
            for i in range(num_features):
                _, perturb_recon = model.reconstruct(dataloaders[i])
                diff = perturb_recon - baseline_recon  # shape: N x C
                mean_diff[i, :, :] += diff * normalizer

        # Calculate Bayes factors
        logger.info("Identifying significant features")
        bayes_k = np.empty((num_features, num_continuous))
        for i in range(num_features):
            mask = feature_mask[:, [i]] | nan_mask  # N x C
            diff = np.ma.masked_array(mean_diff[i, :, :], mask=mask)  # shape: N x C
            prob = np.ma.compressed(np.mean(diff > 1e-8, axis=0))  # shape: C
            bayes_k[i, :] = np.abs(np.log(prob + 1e-8) - np.log(1 - prob + 1e-8))

        # Calculate Bayes probabilities
        bayes_p = np.exp(bayes_k) / (1 + np.exp(bayes_k))
        sort_ids = np.argsort(bayes_k, axis=None)[::-1]
        prob = np.take(bayes_p, sort_ids)  # shape: NC
        logger.debug(f"Bayes proba range: [{prob[-1]:.3f} {prob[0]:.3f}]")

        # Calculate FDR
        fdr = np.cumsum(1 - prob) / np.arange(1, prob.size + 1)
        idx = np.argmin(np.abs(fdr - task_config.sig_threshold))
        logger.debug(f"FDR range: [{fdr[0]:.3f} {fdr[-1]:.3f}]")

        return sort_ids[:idx]

    def _ttest_approach(task_config: IdentifyAssociationsTTestConfig) -> np.ndarray:
        from scipy.stats import ttest_rel

        # Train models
        logger.info("Training models")
        pvalues = np.empty(
            (
                len(task_config.num_latent),
                task_config.num_refits,
                num_features,
                num_continuous,
            )
        )

        for k, num_latent in enumerate(task_config.num_latent):
            for j in range(task_config.num_refits):

                # Initialize model
                model: VAE = hydra.utils.instantiate(
                    task_config.model,
                    continuous_shapes=baseline_dataloader.dataset.con_shapes,
                    categorical_shapes=baseline_dataloader.dataset.cat_shapes,
                    num_latent=num_latent,
                )
                if j == 0:
                    logger.debug(f"Model: {model}")

                # Train model
                logger.debug(f"Training refit {j + 1}/{task_config.num_refits}")
                _ = hydra.utils.call(
                    task_config.training_loop,
                    model=model,
                    train_dataloader=train_dataloader,
                )
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
                for i in range(num_features):
                    _, perturb_recon = model.reconstruct(dataloaders[i])
                    perturb_diff = perturb_recon - baseline_recon
                    mask = feature_mask[:, [i]] | nan_mask
                    _, pvalues[k, j, i, :] = ttest_rel(
                        a=np.where(mask, np.nan, perturb_diff),
                        b=np.where(mask, np.nan, baseline_diff),
                        axis=0,
                        nan_policy="omit",
                    )

        # Correct p-values (Bonferroni)
        pvalues = np.minimum(pvalues * num_continuous, 1.0)

        # Find significant hits
        overlap_thres = task_config.num_refits // 2
        reject = pvalues <= task_config.sig_threshold  # 4D: L x R x F x C
        overlap = reject.sum(axis=1) >= overlap_thres  # 3D: L x F x C
        sig_ids = overlap.sum(axis=0) >= 3  # 2D: F x C

        return np.flatnonzero(sig_ids)  # 1D

    task_type = OmegaConf.get_type(task_config)
    if task_type is IdentifyAssociationsBayesConfig:
        sig_ids = _bayes_approach(task_config)
    elif task_type is IdentifyAssociationsTTestConfig:
        sig_ids = _ttest_approach(task_config)
    else:
        raise ValueError("Unsupported type of task")

    # Prepare results
    logger.debug(f"Significant hits found: {sig_ids.size}")

    if sig_ids.size > 0:
        sig_ids = np.vstack((sig_ids // num_continuous, sig_ids % num_continuous)).T

        logger.info("Writing results")
        results = pd.DataFrame(sig_ids, columns=["feature_a_id", "feature_b_id"])
        results.sort_values("feature_a_id", inplace=True)
        a_df = pd.DataFrame(dict(x=cat_names[target_dataset_idx])).reset_index()
        a_df.columns = ["feature_a_id", "feature_a_name"]
        con_names = reduce(list.__add__, con_names)
        b_df = pd.DataFrame(dict(x=con_names)).reset_index()
        b_df.columns = ["feature_b_id", "feature_b_name"]
        results = results.merge(a_df, on="feature_a_id").merge(b_df, on="feature_b_id")
        results["feature_b_dataset"] = pd.cut(
            results.feature_a_id,
            bins=np.cumsum([0] + con_shapes),
            right=False,
            labels=config.data.continuous_names,
        )
        results.to_csv(output_path / "results_sig_assoc.tsv", sep="\t")
