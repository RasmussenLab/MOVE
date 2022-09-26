__all__ = ["identify_associations"]

from functools import reduce
from pathlib import Path

import hydra
import numpy as np
import pandas as pd

from move.conf.schema import IdentifyAssociationsBayesConfig, MOVEConfig
from move.data.io import read_data
from move.data.perturbations import perturb_data
from move.models.vae import VAE
from move.training.training_loop import TrainingLoopOutput


def identify_associations(config: MOVEConfig):
    """Trains multiple models to identify associations between the dataset
    of interest and the continuous datasets."""
    # TODO: logging
    task_config: IdentifyAssociationsBayesConfig = config.task
    # Create output folders
    output_path = Path(config.data.processed_data_path) / "05_identify_associations"
    output_path.mkdir(exist_ok=True, parents=True)
    # Read original data and create perturbed datasets
    cat_list, cat_names, con_list, con_names = read_data(config)
    con_shapes = [con.shape[1] for con in con_list]
    dataloaders = perturb_data(
        cat_list,
        con_list,
        config.data.categorical_names,
        task_config.target_dataset,
        task_config.target_value,
    )
    baseline_dataloader = dataloaders[-1]
    num_features = len(dataloaders) - 1  # F
    num_samples = len(baseline_dataloader.sampler)  # N
    num_continuous = sum(con_shapes)  # C

    # Train models
    mean_diff = np.zeros((num_features, num_samples, num_continuous))
    normalizer = 1 / task_config.num_refits
    for _ in range(task_config.num_refits):
        # Initialize model
        model: VAE = hydra.utils.instantiate(
            task_config.model,
            continuous_shapes=baseline_dataloader.dataset.con_shapes,
            categorical_shapes=baseline_dataloader.dataset.cat_shapes,
        )
        # Train model
        _: TrainingLoopOutput = hydra.utils.call(
            task_config.training_loop,
            model=model,
            train_dataloader=baseline_dataloader,
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
    bayes_k = np.empty((num_features, num_continuous))
    for i in range(num_features):
        diff = mean_diff[i, :, :]  # shape: N x C
        prob = np.mean(diff > 1e-8, axis=0)  # shape: C
        bayes_k[i, :] = np.abs(np.log(prob + 1e-8) - np.log(1 - prob + 1e-8))
    # Calculate Bayes probabilities
    bayes_p = np.exp(bayes_k) / (1 + np.exp(bayes_k))
    sort_ids = np.argsort(bayes_k, axis=None)[::-1]
    prob = np.take(bayes_p, sort_ids)  # shape: NC
    # Calculate FDR
    fdr = np.cumsum(1 - prob) / np.arange(1, prob.size + 1)
    idx = np.argmin(np.abs(fdr - task_config.fdr_threshold))
    # Find significant IDs
    sig_ids = sort_ids[:idx]
    sig_ids = np.vstack((sig_ids // num_continuous, sig_ids % num_continuous)).T

    # Prepare results
    results = pd.DataFrame(sig_ids, columns=["feature_a_id", "feature_b_id"])
    results.sort_values("feature_a_id", inplace=True)
    dataset_idx = config.data.categorical_names.index(task_config.target_dataset)
    a_df = pd.DataFrame(dict(x=cat_names[dataset_idx])).reset_index()
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
