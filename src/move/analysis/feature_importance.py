__all__ = []

from typing import cast

import pandas as pd
import torch

import move.visualization as viz
from move.core.exceptions import UnsetProperty
from move.data.dataloader import MoveDataLoader
from move.data.io import sanitize_filename
from move.models.base import BaseVae
from move.tasks.base import CsvWriterMixin, ParentTask, SubTask


class FeatureImportance(CsvWriterMixin, SubTask):
    """Compute feature importance in latent space.

    Feature importance is computed per feature per dataset. For each dataset,
    a file will be created.

    Feature importance is computed as the sum of differences in latent
    variables generated when a feature is present/removed."""

    filename_fmt = "feature_importance_{}.csv"

    def __init__(
        self, parent: ParentTask, model: BaseVae, dataloader: MoveDataLoader
    ) -> None:
        self.parent = parent
        self.model = model
        self.dataloader = dataloader

    def plot(self) -> None:
        raise NotImplementedError()

    @torch.no_grad()
    def run(self) -> None:
        for dataset in self.dataloader.datasets:
            self.log(f"Computing feature importance: '{dataset}'")
            # Create a file for each dataset
            # File is transposed; each column is a sample, each row a feature
            if self.parent:
                csv_filename = sanitize_filename(self.filename_fmt.format(dataset))
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
