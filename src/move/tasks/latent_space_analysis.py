__all__ = []

from pathlib import Path
from typing import Any, Optional

import hydra
import numpy as np
import pandas as pd
import torch
from numpy.typing import NDArray
from sklearn.base import TransformerMixin

import move.visualization as viz
from move.analysis.feature_importance import FeatureImportance
from move.analysis.metrics import ComputeAccuracyMetrics
from move.conf.tasks import ReducerConfig
from move.core.exceptions import UnsetProperty
from move.core.typing import PathLike
from move.data.dataloader import MoveDataLoader
from move.data.dataset import DiscreteDataset
from move.data.io import sanitize_filename
from move.models.base import BaseVae
from move.tasks.base import CsvWriterMixin, SubTask
from move.tasks.move import MoveTask


class LatentSpaceAnalysis(MoveTask):
    """Analyze latent space.

        1. Train a model (or reload it if it already exists).
        2. Obtain latent representation of input data.
        3. If configured, reduce latent representation to 2 dimensions.
        4. If configured, compute how accurately each dataset can be reconstructed.
        5. If configured, compute the feature importance in the latent space.

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
        compute_accuracy_metrics:
            Whether accuracy metrics for each dataset will be computed.
        compute_feature_importance:
            Whether feature importance for each feature in the latent space will
            be computed. May take a while depending on number of features.
        reducer_config:
            Config of the reducer used to further reduce the dimensions of the
            latent space to two dimensions. Expected to behave like a
            transformer from scikit-learn.
        features_to_plot:
            List of feature names to generate color-coded latent space plots.
            If not given, no latent space will be generated, but a CSV file
            containing all latent representations will still be created.
    """

    loop_filename: str = "loop.yaml"
    model_filename: str = "model.pt"
    results_subdir: str = "latent_space"

    def __init__(
        self,
        interim_data_path: PathLike,
        results_path: PathLike,
        compute_accuracy_metrics: bool,
        compute_feature_importance: bool,
        reducer_config: Optional[ReducerConfig] = None,
        features_to_plot: Optional[list[str]] = None,
        **kwargs,
    ) -> None:
        super().__init__(
            input_dir=interim_data_path,
            output_dir=Path(results_path) / self.results_subdir,
            **kwargs,
        )
        self.reducer_config = reducer_config
        self.features_to_plot = features_to_plot
        self.compute_accuracy_metrics = compute_accuracy_metrics
        self.compute_feature_importance = compute_feature_importance

    def run(self) -> Any:
        model_path = self.output_dir / self.model_filename

        if model_path.exists():
            self.logger.debug("Re-loading model")
            model = BaseVae.reload(model_path)
        else:
            self.logger.debug("Training a new model")
            train_dataloader = self.make_dataloader()
            model = self.init_model(train_dataloader)
            training_loop = self.init_training_loop()
            training_loop.train(model, train_dataloader)
            training_loop.plot()
            training_loop.to_yaml(self.output_dir / self.loop_filename)
            model.save(model_path)

        model.eval()

        test_dataloader = self.make_dataloader(split="test")

        subtask = Project(model, test_dataloader, self.reducer_config)
        subtask.parent = self
        subtask.run()
        if self.features_to_plot:
            subtask.plot(self.features_to_plot)

        if self.compute_accuracy_metrics:
            subtask = ComputeAccuracyMetrics(self, model, test_dataloader)
            subtask.run()

        if self.compute_feature_importance:
            subtask = FeatureImportance(self, model, test_dataloader)
            subtask.run()


class Project(CsvWriterMixin, SubTask):
    """Use a variational autoencoder to compress input data from a dataloader
    into a latent space. Additionally, use a reducer to further compress the
    latent space into an even lower-dimensional space that can be easily
    visualized (e.g., in 2D or 3D).

    Args:
        model: Variational autoencoder model
        dataloader: Data loader
    """

    filename: str = "latent_space.csv"
    plot_filename_fmt: str = "latent_space_{}.png"
    reducer: Optional[TransformerMixin]

    def __init__(
        self,
        model: BaseVae,
        dataloader: MoveDataLoader,
        reducer_config: Optional[ReducerConfig],
        output_dir: Optional[PathLike] = None,
    ):
        self.model = model
        self.dataloader = dataloader
        if reducer_config is None:
            self.reducer = None
        else:
            self.reducer = hydra.utils.instantiate(reducer_config)
        if output_dir is not None:
            self.output_dir = output_dir

    @property
    def num_features(self) -> int:
        return self.model.num_latent

    @property
    def num_reduced_features(self) -> int:
        if self.reducer is None:
            return 0
        return getattr(self.reducer, "n_components")

    def plot(self, feature_names: list[str]) -> None:
        # NOTE: assumes 2D
        if self.csv_filepath is None:
            raise ValueError("No CSV data found")
        data = pd.read_csv(self.csv_filepath)
        latent_space = np.take(data.values, (0, 1), axis=1)
        for name in feature_names:
            try:
                dataset = self.dataloader.dataset.find(name)
            except KeyError as e:
                self.log(str(e), "WARNING")
                continue
            # Obtain target values for color coding
            target_values = dataset.select(name).numpy()
            if isinstance(dataset, DiscreteDataset):
                # Convert one-hot encoded values to category codes
                is_nan = target_values.sum(axis=1) == 0
                target_values = np.argmax(target_values, axis=1)
                assert dataset.mapping is not None
                code2cat_map = {
                    str(code): category for category, code in dataset.mapping.items()
                }
                fig = viz.plot_latent_space_with_cat(
                    latent_space, name, target_values, code2cat_map, is_nan
                )
            else:
                fig = viz.plot_latent_space_with_con(latent_space, name, target_values)
            fig_filename = sanitize_filename(self.plot_filename_fmt.format(name))
            fig_path = str(self.output_dir / fig_filename)
            fig.savefig(fig_path, bbox_inches="tight")

    @torch.no_grad()
    def run(self) -> None:
        if self.parent is None:
            raise UnsetProperty("Output directory")

        colnames = [f"reduced_dim{i}" for i in range(self.num_reduced_features)]
        colnames.extend([f"dim{i}" for i in range(self.num_features)])

        csv_filepath = self.parent.output_dir / self.filename
        self.init_csv_writer(csv_filepath, fieldnames=colnames, extrasaction="ignore")

        self.log("Compressing input data")

        if self.dataloader.dataset.perturbation is not None:
            self.log("Dataset's perturbation will be removed", "WARNING")
            self.dataloader.dataset.remove_perturbation()

        tensors = []
        for (batch,) in self.dataloader:
            tensors.append(self.model.project(batch))

        latent_space: NDArray = torch.cat(tensors, dim=0).numpy()
        if self.reducer is None:
            array = latent_space
        else:
            self.log("Reducing data to two dimensions")
            reduced_latent_space = self.reducer.fit_transform(latent_space)
            array = np.hstack((reduced_latent_space, latent_space))

        self.write_cols({colname: array[:, i] for i, colname in enumerate(colnames)})

        self.close_csv_writer()
