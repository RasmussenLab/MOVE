__all__ = ["FeatureImportance"]

from typing import TYPE_CHECKING

import pandas as pd
import torch

import move.visualization as viz
from move.core.exceptions import UnsetProperty
from move.data.io import sanitize_filename
from move.tasks.base import CsvWriterMixin, ParentTask, SubTask

if TYPE_CHECKING:
    from move.data.dataloader import MoveDataLoader
    from move.models.base import BaseVae


class FeatureImportance(CsvWriterMixin, SubTask):
    """Compute feature importance in latent space.

    Feature importance is computed per feature per dataset. For each dataset,
    a file will be created.

    Feature importance is computed as the sum of differences in latent
    variables generated when a feature is present/removed."""

    data_filename_fmt: str = "feature_importance_{}.csv"
    plot_filename_fmt: str = "feature_importance_{}.png"

    def __init__(
        self, parent: ParentTask, model: "BaseVae", dataloader: "MoveDataLoader"
    ) -> None:
        self.parent = parent
        self.model = model
        self.dataloader = dataloader

    def plot(self) -> None:
        if self.parent is None:
            return
        for dataset in self.dataloader.datasets:
            csv_filename = self.data_filename_fmt.format(dataset.name)
            csv_filepath = self.parent.output_dir / sanitize_filename(csv_filename)
            fig_filename = self.plot_filename_fmt.format(dataset.name)
            fig_filepath = self.parent.output_dir / sanitize_filename(fig_filename)

            diffs = pd.read_csv(csv_filepath)

            if dataset.data_type == "continuous":
                fig = viz.plot_continuous_feature_importance(
                    diffs.values, dataset.tensor.numpy(), dataset.feature_names
                )
            else:
                # Categorical dataset is re-shaped to 3D shape
                dataset_shape = getattr(dataset, "original_shape")
                fig = viz.plot_categorical_feature_importance(
                    diffs.values,
                    dataset.tensor.reshape(-1, *dataset_shape).numpy(),
                    dataset.feature_names,
                    getattr(dataset, "mapping"),
                )

            fig.savefig(fig_filepath, bbox_inches="tight")

    @torch.no_grad()
    def run(self) -> None:
        for dataset in self.dataloader.datasets:
            self.log(f"Computing feature importance: '{dataset}'")
            # Create a file for each dataset
            # File is transposed; each column is a sample, each row a feature
            if self.parent:
                csv_filename = sanitize_filename(self.data_filename_fmt.format(dataset))
                csv_filepath = self.parent.output_dir / csv_filename
                colnames = ["feature_name"] + [""] * len(self.dataloader.dataset)
                self.init_csv_writer(
                    csv_filepath, fieldnames=colnames, extrasaction="ignore"
                )
            else:
                raise UnsetProperty("Parent task")

            # Make a perturbation for each feature
            for feature_name in dataset.feature_names:
                value = None if dataset.data_type == "discrete" else 0.0
                self.dataloader.dataset.perturb(dataset.name, feature_name, value)
                row = [feature_name]
                for tup in self.dataloader:
                    batch, pert_batch, _ = tup
                    z = self.model.project(batch)
                    z_pert = self.model.project(pert_batch)
                    diff = torch.sum(z_pert - z, dim=-1)
                    row.extend(diff.tolist())
                self.write_row(row)

            self.close_csv_writer(clear=True)

            # Transpose CSV file, so each row is a sample, each column a feature
            pd.read_csv(csv_filepath).T.to_csv(csv_filepath, index=False, header=False)

        # Clear perturbation
        self.dataloader.dataset.perturbation = None
