__all__ = ["analyze_latent"]

from pathlib import Path
from typing import cast

import hydra
import numpy as np
import pandas as pd
import torch

import move.visualization as viz
from move.conf.schema import AnalyzeLatentConfig, MOVEConfig
from move.core.logging import get_logger
from move.data import io
from move.data.dataloaders import MOVEDataset, make_dataloader
from move.models.vae import VAE
from move.training.training_loop import TrainingLoopOutput


def analyze_latent(config: MOVEConfig):
    """Trains one model to inspect its latent space."""

    logger = get_logger(__name__)
    logger.info("Beginning task: analyze latent space")
    task_config = cast(AnalyzeLatentConfig, config.task)

    output_path = Path(config.data.processed_data_path) / "latent_space"
    output_path.mkdir(exist_ok=True, parents=True)

    cat_list, _, con_list, _ = io.load_preprocessed_data(
        Path(config.data.interim_data_path),
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
        logger.debug("Generating plot: loss curves")
        loss_fig = viz.plot_loss_curves(losses)
        loss_fig.savefig(str(output_path / "loss_curve.png"), bbox_inches="tight")
        loss_df = pd.DataFrame(dict(zip(viz.LOSS_LABELS, losses)))
        loss_df.index.name = "epoch"
        loss_df.to_csv("loss_curve.tsv", sep="\t")
    model.eval()

    