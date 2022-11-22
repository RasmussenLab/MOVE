__all__ = ["tune_model"]

from pathlib import Path
from random import shuffle
from typing import Any, cast

import hydra
import numpy as np
import pandas as pd
from hydra.core.hydra_config import HydraConfig
from hydra.types import RunMode
from matplotlib.cbook import boxplot_stats

from move.analysis.metrics import calculate_accuracy, calculate_cosine_similarity
from move.conf.schema import MOVEConfig, TuneModelConfig
from move.core.logging import get_logger
from move.core.typing import BoolArray, FloatArray
from move.data import io
from move.data.dataloaders import MOVEDataset, make_dataloader, split_samples
from move.models.vae import VAE


def _get_record(values: FloatArray, **kwargs) -> dict[str, Any]:
    record = kwargs
    bxp_stats, *_ = boxplot_stats(values)
    bxp_stats.pop("fliers")
    record.update(bxp_stats)
    return record


def tune_model(config: MOVEConfig) -> float:
    """Train multiple models to tune the model hyperparameters."""
    hydra_config = HydraConfig.get()
    if hydra_config.mode != RunMode.MULTIRUN:
        raise ValueError("This task must run in multirun mode.")

    # Delete sweep run config
    sweep_config_path = Path(hydra_config.sweep.dir).joinpath("multirun.yaml")
    if sweep_config_path.exists():
        sweep_config_path.unlink()

    job_num = hydra_config.job.num + 1

    logger = get_logger(__name__)
    logger.info(f"Beginning task: tune model {job_num}")
    logger.info(f"Job name: {hydra_config.job.override_dirname}")
    task_config = cast(TuneModelConfig, config.task)

    interim_path = Path(config.data.interim_data_path)
    output_path = Path(config.data.processed_data_path) / "tune_model"
    output_path.mkdir(exist_ok=True, parents=True)

    logger.debug("Reading data")

    cat_list, _, con_list, _ = io.load_preprocessed_data(
        interim_path,
        config.data.categorical_names,
        config.data.continuous_names,
    )

    split_path = interim_path / "split_mask.npy"
    if split_path.exists():
        split_mask: BoolArray = np.load(split_path)
    else:
        num_samples = cat_list[0].shape[0] if cat_list else con_list[0].shape[0]
        split_mask = split_samples(num_samples, 0.9)
        np.save(split_path, split_mask)

    train_dataloader = make_dataloader(
        cat_list,
        con_list,
        split_mask,
        shuffle=True,
        batch_size=task_config.batch_size,
        drop_last=True,
    )
    train_dataset = cast(MOVEDataset, train_dataloader.dataset)

    assert task_config.model is not None
    model: VAE = hydra.utils.instantiate(
        task_config.model,
        continuous_shapes=train_dataset.con_shapes,
        categorical_shapes=train_dataset.cat_shapes,
    )
    logger.debug(f"Model: {model}")

    logger.debug("Training model")
    hydra.utils.call(
        task_config.training_loop,
        model=model,
        train_dataloader=train_dataloader,
    )
    model.eval()

    logger.info("Reconstructing")
    logger.info("Computing reconstruction metrics")
    label = [hp.split("=") for hp in hydra_config.job.override_dirname.split(",")]
    records = []
    splits = zip(["train", "test"], [split_mask, ~split_mask])
    for split_name, mask in splits:
        dataloader = make_dataloader(
            cat_list,
            con_list,
            mask,
            shuffle=False,
            batch_size=np.count_nonzero(mask),
        )
        cat_recons, con_recons = model.reconstruct(dataloader)
        con_recons = np.split(con_recons, model.continuous_shapes[:-1], axis=1)
        for cat, cat_recon, dataset_name in zip(
            cat_list, cat_recons, config.data.categorical_names
        ):
            accuracy = calculate_accuracy(cat[mask], cat_recon)
            record = _get_record(
                accuracy,
                job_num=job_num,
                **dict(label),
                metric="accuracy",
                dataset=dataset_name,
                split=split_name,
            )
            records.append(record)
        for con, con_recon, dataset_name in zip(
            con_list, con_recons, config.data.continuous_names
        ):
            cosine_sim = calculate_cosine_similarity(con[mask], con_recon)
            record = _get_record(
                cosine_sim,
                job_num=job_num,
                **dict(label),
                metric="cosine_similarity",
                dataset=dataset_name,
                split=split_name,
            )
            records.append(record)

    logger.info("Writing results")
    df_path = output_path / "reconstruction_stats.tsv"
    header = not df_path.exists()
    df = pd.DataFrame.from_records(records)
    df.to_csv(df_path, sep="\t", mode="a", header=header, index=False)

    return 0.0
