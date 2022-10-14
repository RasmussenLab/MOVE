__all__ = ["analyze_latent"]

from pathlib import Path
from typing import cast

import hydra
import numpy as np
import pandas as pd
import torch
from sklearn.base import TransformerMixin
from torch.utils.data import SequentialSampler

import move.visualization as viz
from move.analysis.metrics import calculate_accuracy, calculate_cosine_similarity
from move.core.logging import get_logger
from move.core.typing import FloatArray
from move.conf.schema import AnalyzeLatentConfig, MOVEConfig
from move.data import io
from move.data.dataloaders import MOVEDataset, make_dataloader
from move.models.vae import VAE
from move.training.training_loop import TrainingLoopOutput


def find_feature_values(
    feature_name: str,
    feature_names_lists: list[list[str]],
    feature_values: list[FloatArray],
) -> tuple[int, FloatArray]:
    """Looks for the feature in the list of datasets and returns its values.

    Args:
        feature_name: Lookup key
        feature_names_lists: List of lists with feature names for each dataset
        feature_values: List of data arrays, each representing a dataset

    Raises:
        KeyError: If feature does not exist in any dataset

    Returns:
        Tuple containing (1) index of dataset containing feature and (2)
        values corresponding to the feature
    """
    dataset_index, feature_index = [None] * 2
    for dataset_index, feature_names in enumerate(feature_names_lists):
        try:
            feature_index = feature_names.index(feature_name)
        except ValueError:
            continue
        break
    if dataset_index is not None and feature_index is not None:
        return (
            dataset_index,
            np.take(feature_values[dataset_index], feature_index, axis=1)
        )
    raise KeyError(f"Feature '{feature_name}' not in any dataset.")


def _validate_task_config(task_config: AnalyzeLatentConfig) -> None:
    if "_target_" not in task_config.reducer:
        raise ValueError("Reducer class not specified properly.")


def analyze_latent(config: MOVEConfig) -> None:
    """Trains one model to inspect its latent space."""

    logger = get_logger(__name__)
    logger.info("Beginning task: analyze latent space")
    task_config = cast(AnalyzeLatentConfig, config.task)
    _validate_task_config(task_config)

    raw_data_path = Path(config.data.raw_data_path)
    interim_path = Path(config.data.interim_data_path)
    output_path = Path(config.data.processed_data_path) / "latent_space"
    output_path.mkdir(exist_ok=True, parents=True)

    logger.debug("Reading data")
    sample_names = io.read_names(raw_data_path / f"{config.data.sample_names}.txt")
    cat_list, cat_names, con_list, con_names = io.load_preprocessed_data(
        interim_path,
        config.data.categorical_names,
        config.data.continuous_names,
    )
    train_mask, train_dataloader = make_dataloader(
        cat_list,
        con_list,
        shuffle=True,
        batch_size=task_config.batch_size,
        drop_last=True,
    )
    train_dataset = cast(MOVEDataset, train_dataloader.dataset)
    logger.debug(f"Masked training samples: {np.sum(~train_mask)}/{train_mask.size}")

    assert task_config.model is not None
    model: VAE = hydra.utils.instantiate(
        task_config.model,
        continuous_shapes=train_dataset.con_shapes,
        categorical_shapes=train_dataset.cat_shapes,
    )
    logger.debug(f"Model: {model}")

    model_path = output_path / "model.pt"
    if model_path.exists():
        logger.debug("Re-loading model")
        model.load_state_dict(torch.load(model_path))
    else:
        logger.debug("Training model")
        output: TrainingLoopOutput = hydra.utils.call(
            task_config.training_loop,
            model=model,
            train_dataloader=train_dataloader,
        )
        losses = output[:-1]
        torch.save(model.state_dict(), model_path)
        logger.info("Generating visualizations")
        logger.debug("Generating plot: loss curves")
        fig = viz.plot_loss_curves(losses)
        fig_path = str(output_path / "loss_curve.png")
        fig.savefig(fig_path, bbox_inches="tight")
        fig_df = pd.DataFrame(dict(zip(viz.LOSS_LABELS, losses)))
        fig_df.index.name = "epoch"
        fig_df.to_csv("loss_curve.tsv", sep="\t")
    model.eval()

    test_mask, test_dataloader = make_dataloader(
        cat_list,
        con_list,
        shuffle=False,
        batch_size=len(cast(SequentialSampler, train_dataloader.sampler)),
    )
    logger.info("Projecting into latent space")
    latent_space = model.project(test_dataloader)
    reducer: TransformerMixin = hydra.utils.instantiate(task_config.reducer)
    embedding = reducer.fit_transform(latent_space)

    mappings = io.load_mappings(interim_path / "mappings.json")
    for feature_name in task_config.feature_names:
        logger.debug(f"Generating plot: latent space + '{feature_name}'")
        is_categorical = False
        try:
            dataset_index, feature_values = find_feature_values(feature_name, cat_names, cat_list)
            is_categorical = True
        except KeyError:
            dataset_index, feature_values = find_feature_values(feature_name, con_names, con_list)

        if is_categorical:
            # Convert one-hot encoding to category codes
            is_nan = feature_values.sum(axis=1) == 0
            feature_values = np.argmax(feature_values, axis=1)[test_mask]

            dataset_name = config.data.categorical_names[dataset_index]
            fig = viz.plot_latent_space_with_cat(
                embedding,
                feature_name,
                feature_values,
                mappings[dataset_name],
                is_nan,
            )
        else:
            feature_values = feature_values[test_mask]
            fig = viz.plot_latent_space_with_con(embedding, feature_name, feature_values)

        fig_path = str(output_path / f"latent_space_{feature_name}.png")
        fig.savefig(fig_path, bbox_inches="tight")

    logger.info("Reconstructing")
    cat_recons, con_recons = model.reconstruct(test_dataloader)
    con_recons = np.split(con_recons, model.continuous_shapes[:-1], axis=1)
    scores = []
    labels = config.data.categorical_names + config.data.continuous_names
    for cat, cat_recon in zip(cat_list, cat_recons):
        accuracy = calculate_accuracy(cat, cat_recon)
        scores.append(accuracy)
    for con, con_recon in zip(con_list, con_recons):
        cosine_sim = calculate_cosine_similarity(con, con_recon)
        scores.append(cosine_sim)

    logger.debug("Generating plot: reconstruction metrics")
    fig = viz.plot_metrics_boxplot(scores, labels)
    fig_path = str(output_path / "reconstruction_metrics.png")
    fig.savefig(fig_path, bbox_inches="tight")