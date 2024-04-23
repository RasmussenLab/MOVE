__all__ = ["calculate_accuracy", "calculate_cosine_similarity"]

from abc import ABC, abstractmethod
from typing import cast

import numpy as np
import pandas as pd
import torch

import move.visualization as viz
from move.core.typing import FloatArray
from move.data.dataloader import MoveDataLoader
from move.data.dataset import ContinuousDataset, DiscreteDataset
from move.models.base import BaseVae
from move.tasks.base import CsvWriterMixin, ParentTask, SubTask


def calculate_accuracy(
    original_input: FloatArray, reconstruction: FloatArray
) -> FloatArray:
    """Compute accuracy per sample.

    Args:
        original_input: Original labels (one-hot encoded as a 3D array).
        reconstruction: Reconstructed labels (2D array).

    Returns:
        Array of accuracy scores.
    """
    if original_input.ndim != 3:
        raise ValueError("Expected original input to have three dimensions.")
    if reconstruction.ndim != 2:
        raise ValueError("Expected reconstruction to have two dimensions.")
    if original_input[:, :, 0].shape != reconstruction.shape:
        raise ValueError(
            f"Original input {original_input.shape} and reconstruction "
            f"{reconstruction.shape} shapes do not match."
        )

    is_nan = original_input.sum(axis=2) == 0
    original_input = np.argmax(original_input, axis=2)  # 3D => 2D
    y_true = np.ma.masked_array(original_input, mask=is_nan)
    y_pred = np.ma.masked_array(reconstruction, mask=is_nan)

    num_features = np.ma.count(y_true, axis=1)
    scores = np.ma.filled(np.sum(y_true == y_pred, axis=1) / num_features, 0)

    return scores


def calculate_cosine_similarity(
    original_input: FloatArray, reconstruction: FloatArray
) -> FloatArray:
    """Compute cosine similarity per sample.

    Args:
        original_input: Original values (2D array).
        reconstruction: Reconstructed values (2D array).

    Returns:
        Array of similarities.
    """
    if any((original_input.ndim != 2, reconstruction.ndim != 2)):
        raise ValueError("Expected both inputs to have two dimensions.")
    if original_input.shape != reconstruction.shape:
        raise ValueError(
            f"Original input {original_input.shape} and reconstruction "
            f"{reconstruction.shape} shapes do not match."
        )

    is_nan = original_input == 0
    x = np.ma.masked_array(original_input, mask=is_nan)
    y = np.ma.masked_array(reconstruction, mask=is_nan)

    # Equivalent to `np.diag(sklearn.metrics.pairwise.cosine_similarity(x, y))`
    # But can handle masked arrays
    scores = np.ma.compressed(np.sum(x * y, axis=1)) / (norm(x) * norm(y))

    return scores


def norm(x: np.ma.MaskedArray, axis: int = 1) -> FloatArray:
    """Return Euclidean norm. This function is equivalent to `np.linalg.norm`,
    but it can handle masked arrays.

    Args:
        x: 2D masked array
        axis: Axis along which to the operation is performed. Defaults to 1.

    Returns:
        1D array with the specified axis removed.
    """
    return np.ma.compressed(np.sqrt(np.sum(x**2, axis=axis)))


class ComputeAccuracyMetrics(CsvWriterMixin, SubTask):
    """Compute accuracy metrics between original input and reconstruction (use
    cosine similarity for continuous dataset reconstructions)."""

    filename = "reconstruction_metrics.csv"

    def __init__(
        self, parent: ParentTask, model: BaseVae, dataloader: MoveDataLoader
    ) -> None:
        self.parent = parent
        self.model = model
        self.dataloader = dataloader

    def plot(self) -> None:
        if self.parent and self.csv_filepath:
            fig_df = pd.read_csv(self.csv_filepath, index_col=None)
            scores = [fig_df[col].to_list() for col in fig_df.columns]
            # TODO: viz.plot_metrics_boxplot(scores, fig_df.columns)

    @torch.no_grad()
    def run(self) -> None:
        if self.parent:
            csv_filepath = self.parent.output_dir / self.filename
            colnames = self.dataloader.dataset.names
            self.init_csv_writer(
                csv_filepath, fieldnames=colnames, extrasaction="ignore"
            )
        else:
            self.log("No parent task, metrics will not be saved.", "WARNING")

        self.log("Computing accuracy metrics")

        datasets = self.dataloader.datasets
        for batch in self.dataloader:
            batch_disc, batch_cont = self.model.split_input(batch[0])
            recon_disc, recon_cont = self.model.reconstruct(batch[0])

            scores_per_dataset = {}
            for i, dataset in enumerate(datasets[: len(batch_disc)]):
                target = batch_disc[i].numpy()
                preds = torch.argmax(
                    (torch.log_softmax(recon_disc[i], dim=-1)), dim=-1
                ).numpy()
                scores = calculate_accuracy(target, preds)
                scores_per_dataset[dataset.name] = scores

            for i, dataset in enumerate(datasets[len(batch_disc) :]):
                target = batch_cont[i].numpy()
                preds = recon_cont[i].numpy()
                scores = calculate_cosine_similarity(target, preds)
                scores_per_dataset[dataset.name] = scores

            self.write_cols(scores_per_dataset)

        self.close_csv_writer()
