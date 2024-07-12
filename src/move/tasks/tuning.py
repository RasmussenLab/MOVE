__all__ = ["TuneModel"]

from pathlib import Path
from random import shuffle
from typing import Any, Literal, cast

import hydra
import numpy as np
import pandas as pd
import torch
from hydra.core.hydra_config import HydraConfig
from hydra.types import RunMode
from matplotlib.cbook import boxplot_stats
from numpy.typing import ArrayLike
from omegaconf import OmegaConf
from sklearn.metrics.pairwise import cosine_similarity

from move.analysis.metrics import (
    calculate_accuracy,
    calculate_cosine_similarity,
)
from move.core.logging import get_logger
from move.core.typing import BoolArray, FloatArray, PathLike
from move.data import io
from move.data.dataloaders import MOVEDataset, make_dataloader, split_samples
from move.tasks.base import CsvWriterMixin
from move.tasks.move import MoveTask

TaskType = Literal["reconstruction", "stability"]


def _get_record(values: ArrayLike, **kwargs) -> dict[str, Any]:
    record = kwargs
    bxp_stats, *_ = boxplot_stats(values)
    bxp_stats.pop("fliers")
    record.update(bxp_stats)
    return record


class TuneModel(CsvWriterMixin, MoveTask):
    """Run a model with a set of hyperparameters and report metrics, such as,
    reconstruction accuracy, loss, or stability.

    Args:
        interim_data_path:
            Directory where encoded data is stored
        results_path:
            Directory where results will be saved
        discrete_dataset_names:
            Names of discrete datasets
        continuous_dataset_names:
            Names of continuous datasets
        model_config:
            Config of the VAE
        training_dataloader_config:
            Config of the training data loader
        training_loop_config:
            Config of the training loop
    """

    loop_filename: str = "loop.yaml"
    model_filename_fmt: str = "model_{}.pt"
    results_subdir: str = "tuning"
    results_filename: str = "metrics.csv"

    def __init__(
        self,
        interim_data_path: PathLike,
        results_path: PathLike,
        **kwargs,
    ) -> None:
        # Check that Hydra is being used (e.g., not called from a script/notebook)
        try:
            hydra_config = HydraConfig.get()
        except ValueError:
            raise ValueError("Use the command line to run this task.")

        if hydra_config.mode != RunMode.MULTIRUN:
            raise ValueError("This task must run in multirun mode.")

        super().__init__(
            input_dir=interim_data_path,
            output_dir=Path(results_path) / self.results_subdir,
            **kwargs,
        )

        # Delete sweep run config
        sweep_config_path = Path(hydra_config.sweep.dir).joinpath("multirun.yaml")
        if sweep_config_path.exists():
            sweep_config_path.unlink()

        kv_sep = hydra_config.job.config.override_dirname.kv_sep
        item_sep = hydra_config.job.config.override_dirname.item_sep

        self.job_num = hydra_config.job.num + 1
        self.job_name = hydra_config.job.override_dirname
        self.override_dict: dict[str, str] = {}
        for item in hydra_config.job.override_dirname.split(item_sep):
            key, value = item.split(kv_sep)
            self.override_dict[key] = value

    def run(self):
        model_path = self.output_dir / self.model_filename_fmt.format(self.job_num)

        self.init_csv_writer(
            self.output_dir / self.results_filename,
            mode="a",
            fieldnames=["job_num", *self.override_dict.keys()],
            extrasaction="ignore",
        )
        self.add_row_to_buffer(dict(job_num=self.job_num, **self.override_dict))
        self.close_csv_writer()
