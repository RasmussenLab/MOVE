__all__ = ["ComputeAccuracyMetrics"]

from typing import cast

import pandas as pd
import torch

import move.visualization as viz
from move.analysis.metrics import calculate_accuracy, calculate_cosine_similarity
from move.data.dataloader import MoveDataLoader
from move.data.dataset import DiscreteDataset, ContinuousDataset
from move.models.base import BaseVae
from move.tasks.base import SubTaskWritesCsv


class ComputeAccuracyMetrics(SubTaskWritesCsv):
    """Compute accuracy metrics between original input and reconstruction (use
    cosine similarity for continuous dataset reconstructions)."""

    filename = "reconstruction_metrics.csv"

    def __init__(self, model: BaseVae, dataloader: MoveDataLoader) -> None:
        self.model = model
        self.dataloader = dataloader

    def plot(self) -> None:
        if self.parent and self.csv_filepath:
            fig_df = pd.read_csv(self.csv_filepath, index_col=None)
            scores = [fig_df[col].to_list() for col in fig_df.columns]
            # TODO: viz.plot_metrics_boxplot(scores, fig_df.columns)

    def run(self) -> None:
        if self.parent:
            csv_filepath = self.parent.output_path / self.filename
            colnames = ["sample_name"] + self.dataloader.dataset.names
            self.init_csv_writer(
                csv_filepath, fieldnames=colnames, extrasaction="ignore"
            )
        else:
            self.log("No parent task, metrics will not be saved.", "WARNING")
        self.log("Computing accuracy metrics")
        scores_per_dataset = {}
        for batch in self.dataloader:
            batch_disc, batch_cont = self.model.split_output(batch)
            recon = self.model.reconstruct(batch)
            recon_disc, recon_cont = self.model.split_output(recon)
            for i, dataset in enumerate(self.dataloader.datasets):
                if isinstance(dataset, DiscreteDataset):
                    target = batch_disc[i].numpy()
                    preds = torch.argmax(
                        (torch.log_softmax(recon_disc[i], dim=-1)), dim=-1
                    ).numpy()
                    scores = calculate_accuracy(target, preds)
                elif isinstance(dataset, ContinuousDataset):
                    target = cast(torch.Tensor, batch_cont[i]).numpy()
                    preds = cast(torch.Tensor, recon_cont[i]).numpy()
                    scores = calculate_cosine_similarity(target, preds)
                else:
                    raise ValueError()
                scores_per_dataset[dataset.name] = scores
            self.write_cols(scores_per_dataset)
        self.close_csv_writer()
