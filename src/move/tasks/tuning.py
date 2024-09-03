__all__ = ["TuneModel"]

from collections import defaultdict
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np
import torch
from hydra.core.hydra_config import HydraConfig
from hydra.types import RunMode
from matplotlib.cbook import boxplot_stats
from numpy.typing import ArrayLike
from sklearn.metrics.pairwise import cosine_similarity

from move.analysis.metrics import (
    calculate_accuracy,
    calculate_cosine_similarity,
)
from move.core.exceptions import FILE_EXISTS_WARNING
from move.core.typing import FloatArray, PathLike
from move.data.dataloader import MoveDataLoader
from move.models.base import reload_vae, BaseVae, LossDict
from move.tasks.base import CsvWriterMixin
from move.tasks.move import MoveTask

TaskType = Literal["reconstruction", "stability"]

BOXPLOT_STATS = ["mean", "med", "q1", "q3", "iqr", "cilo", "cihi", "whislo", "whishi"]


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
        batch_size:
            Number of samples in one batch (used during training and testing)
        model_config:
            Config of the VAE
        training_loop_config:
            Config of the training loop
    """

    loss_filename: str = "loss.csv"
    metrics_filename: str = "metrics.csv"
    model_filename_fmt: str = "model_{}.pt"
    results_subdir: str = "tuning"

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

    def record_metrics(
        self, model: BaseVae, dataloader_dict: dict[str, MoveDataLoader]
    ):
        """Record accuracy or cosine similarity metric for each dataloader.

        Args:
            model: VAE model
            dataloader_dict: Dict of dataloaders corresponding to different data subsets
        """

        colnames = [
            "job_num",
            *self.override_dict.keys(),
            "metric",
            "dataset_name",
            "split",
        ]
        colnames.extend(BOXPLOT_STATS)

        self.init_csv_writer(
            self.output_dir / self.metrics_filename,
            mode="a",
            fieldnames=colnames,
            extrasaction="ignore",
        )

        for split, dataloader in dataloader_dict.items():
            datasets = dataloader.datasets
            scores_per_dataset: dict[str, list[FloatArray]] = defaultdict(list)

            for (batch,) in dataloader:
                batch_disc, batch_cont = model.split_input(batch)
                recon = model.reconstruct(batch, as_one=True)
                recon_disc, recon_cont = model.split_input(recon)

                for i, dataset in enumerate(datasets[: len(batch_disc)]):
                    target = batch_disc[i].numpy()
                    preds = torch.argmax(
                        (torch.log_softmax(recon_disc[i], dim=-1)), dim=-1
                    ).numpy()
                    scores = calculate_accuracy(target, preds)
                    scores_per_dataset[dataset.name].append(scores)

                for i, dataset in enumerate(datasets[len(batch_disc) :]):
                    target = batch_cont[i].numpy()
                    preds = recon_cont[i].numpy()
                    scores = calculate_cosine_similarity(target, preds)
                    scores_per_dataset[dataset.name].append(scores)

            for dataset in datasets:
                metric = (
                    "accuracy"
                    if dataset.data_type == "discrete"
                    else "cosine_similarity"
                )
                csv_row: dict[str, Any] = dict(
                    job_num=self.job_num,
                    **self.override_dict,
                    metric=metric,
                    dataset_name=dataset.name,
                    split=split,
                )
                scores = np.concatenate(scores_per_dataset[dataset.name], axis=0)
                bxp_stas, *_ = boxplot_stats(scores)
                bxp_stas.pop("fliers")
                csv_row.update(bxp_stas)

                self.add_row_to_buffer(csv_row)

        # Append to file
        self.close_csv_writer(True)

    def record_loss(self, model: BaseVae, dataloader: MoveDataLoader):
        """Record final loss in a CSV row."""

        colnames = ["job_num", *self.override_dict.keys()]
        colnames.extend(LossDict.__annotations__.keys())

        self.init_csv_writer(
            self.output_dir / self.loss_filename,
            mode="a",
            fieldnames=colnames,
            extrasaction="ignore",
        )

        loss_epoch = None

        for (batch,) in dataloader:
            loss_batch = model.compute_loss(batch, 1.0)
            if loss_epoch is None:
                loss_epoch = loss_batch
            else:
                for key in loss_batch.keys():
                    loss_epoch[key] += loss_batch[key]

        csv_row: dict[str, Any] = dict(job_num=self.job_num, **self.override_dict)

        assert loss_epoch is not None
        for key, value in loss_epoch.items():
            if isinstance(value, torch.Tensor):
                csv_row[key] = value.item() / len(dataloader)
            else:
                csv_row[key] = cast(float, value) / len(dataloader)

        # Append to file
        self.add_row_to_buffer(csv_row)
        self.close_csv_writer(True)

    def run(self):
        model_path = self.output_dir / self.model_filename_fmt.format(self.job_num)
        loss_filepath = self.output_dir / self.loss_filename
        metrics_filepath = self.output_dir / self.metrics_filename

        for filepath in (loss_filepath, metrics_filepath):
            if filepath.exists() and self.job_num == 1:
                filepath.unlink()
                self.logger.warning(FILE_EXISTS_WARNING.format(filepath))

        dataloaders = {
            "train": self.make_dataloader("train"),
            "test": self.make_dataloader("test"),
        }

        if model_path.exists():
            self.logger.warning(
                f"A model file was found: '{model_path}' and will be reloaded. "
                "Erase the file if you wish to train a new model."
            )
            self.logger.debug(f"Re-loading model {self.job_num}")
            model = reload_vae(model_path)
        else:
            self.logger.debug(f"Training a new model {self.job_num}")
            model = self.init_model(dataloaders["train"])
            training_loop = self.init_training_loop(False)
            training_loop.train(model, dataloaders["train"])
            model.save(model_path)

        model.freeze()

        self.record_loss(model, dataloaders["test"])
        self.record_metrics(model, dataloaders)


class TuneStability(TuneModel):
    """Train a number of models and compute the stability of their latent space.

    Args:
        num_refits: Number of models to train
    """

    stabilility_filename: str = "stability.csv"

    def __init__(self, num_refits: int, **kwargs) -> None:
        super().__init__(**kwargs)
        self.num_refits = num_refits
        self.baseline_cosine_sim = None

    def calculate_stability(self, latent_repr: FloatArray) -> float:
        """Compute stability (mean difference between the cosine similarities of two
        latent representations).

        Args:
            latent_repr: Latent representation"""

        if self.baseline_cosine_sim is None:
            raise ValueError("Cannot calculate stability without a baseline.")
        cosine_sim = cosine_similarity(latent_repr)
        abs_diff = np.absolute(cosine_sim - self.baseline_cosine_sim)
        # Remove diagonal (cosine similarity with itself)
        diff = abs_diff[~np.eye(abs_diff.shape[0], dtype=bool)].reshape(
            abs_diff.shape[0], -1
        )
        return np.mean(diff).item()

    def run(self) -> None:
        results_filepath = self.output_dir / self.stabilility_filename

        if results_filepath.exists() and self.job_num == 1:
            results_filepath.unlink()
            self.logger.warning(FILE_EXISTS_WARNING.format(results_filepath))

        train_dataloader = self.make_dataloader("train")
        test_dataloader = self.make_dataloader("test")

        stabs: list[float] = []

        for i in range(self.num_refits):
            self.logger.debug(f"Refit: {i+1}/{self.num_refits}")
            model = self.init_model(train_dataloader)
            training_loop = self.init_training_loop(False)
            training_loop.train(model, train_dataloader)
            model.freeze()

            # Create latent representation for all samples
            latent_reprs = []
            for (batch,) in test_dataloader:
                latent_reprs.append(model.project(batch))
            latent_repr = torch.concat(latent_reprs).numpy()

            if self.baseline_cosine_sim is None:
                # Store first cosine similarity as baseline
                self.baseline_cosine_sim = cosine_similarity(latent_repr)
            else:
                # Calculate stability from a pair of cosine similarity arrays
                stability = self.calculate_stability(latent_repr)
                stabs.append(stability)

        # Append row to CSV file
        colnames = ["job_num", *self.override_dict.keys(), "metric"]
        colnames.extend(BOXPLOT_STATS)

        self.init_csv_writer(
            results_filepath,
            mode="a",
            fieldnames=colnames,
            extrasaction="ignore",
        )
        csv_row: dict[str, Any] = dict(
            job_num=self.job_num,
            **self.override_dict,
            metric="stability",
        )
        bxp_stas, *_ = boxplot_stats(stabs)
        bxp_stas.pop("fliers")
        csv_row.update(bxp_stas)

        self.add_row_to_buffer(csv_row)
        self.close_csv_writer()
