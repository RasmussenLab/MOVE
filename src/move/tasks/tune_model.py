__all__ = ["tune_model"]

from pathlib import Path
from typing import Union

import hydra
from hydra.core.hydra_config import HydraConfig
from hydra.types import RunMode

from move.data import io
from move.conf.schema import MOVEConfig
from move.core.logging import get_logger
from move.models.vae import VAE


def tune_model(config: MOVEConfig) -> float:
    """Train multiple models to tune the model hyperparameters."""
    hydra_config = HydraConfig.get()
    if hydra_config.mode != RunMode.MULTIRUN:
        raise ValueError("This task must run in multirun mode.")

    job_num = hydra_config.job.num + 1

    logger = get_logger(__name__)
    logger.info(f"Beginning task: tune model {job_num}")
    logger.info(f"Job name: {hydra_config.job.override_dirname}")

    raw_data_path = Path(config.data.raw_data_path)
    interim_path = Path(config.data.interim_data_path)
    output_path = Path(config.data.processed_data_path) / "tune_model"
    output_path.mkdir(exist_ok=True, parents=True)

    logger.debug("Reading data")

    sample_names = io.read_names(raw_data_path / f"{config.data.sample_names}.txt")
    cat_list, cat_names, con_list, con_names = io.load_preprocessed_data(
        interim_path,
        config.data.categorical_names,
        config.data.continuous_names,
    )

    return 0.0
