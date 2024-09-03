__all__ = []

from pathlib import Path
from typing import Any, Iterator

import numpy as np
import pandas as pd
import torch

import move.visualization as viz
from move.conf.tasks import PerturbationConfig
from move.core.typing import PathLike
from move.models.base import BaseVae
from move.tasks.base import CsvWriterMixin
from move.tasks.move import MoveTask


class Associations(CsvWriterMixin, MoveTask):
    """Find associations by applying a perturbation to dataset and checking
    which features significantly change after perturbation.

        1. Train and refit # models (or reload them if already existing).
        2. Perturb a feature in the dataset.
        3. Obtain reconstructions before/after perturbation.
        4. Compute difference between reconstructions.
        5. Average difference over # re-fits
        6. Compute indicator variable as probability difference > 0. Significant
           features are either: majorly lower than 0 or majorly greater than 0.
           Thus, when ranking features by this probability those with 50% will
           be considered not significant.
        7. Rank by indicator variable, and calculate FDR as cumulative
           probability of being significant.
        8. Select as significant the features below a determined threshold.

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
        perturbation_config:
            Config of the perturbation
        num_refits:
            Number of times to refit the model
        sig_threshold:
            Threshold used to determine whether an association is significant.
            Significant associations are selected if their FDR is below this
            threshold. Value should range between (0, 1)
        write_only_sig:
            Whether all or only significant hits are written in output file.
    """

    loop_filename: str = "loop.yaml"
    model_filename_fmt: str = "model_{}.pt"
    results_subdir: str = "associations"
    results_filename: str = "associations.csv"

    def __init__(
        self,
        interim_data_path: PathLike,
        results_path: PathLike,
        perturbation_config: PerturbationConfig,
        num_refits: int,
        sig_threshold: float = 0.05,
        write_only_sig: bool = True,
        **kwargs,
    ) -> None:
        if not (0 < sig_threshold < 1.0):
            raise ValueError("Significant threshold should be in range (0, 1)")

        super().__init__(
            input_dir=interim_data_path,
            output_dir=Path(results_path) / self.results_subdir,
            **kwargs,
        )
        self.perturbation_config = perturbation_config
        self.num_refits = num_refits
        self.sig_threshold = sig_threshold
        self.write_only_sig = write_only_sig

    def get_trained_model(self, train_dataloader) -> Iterator[BaseVae]:
        """Yield a trained model. Model will be trained from scratch or
        re-loaded if it already exists."""
        for i in range(self.num_refits):
            model_path = self.output_dir / self.model_filename_fmt.format(i)

            if model_path.exists():
                if i == 0:
                    self.logger.debug(f"Re-loading models")
                model = BaseVae.reload(model_path)
            else:
                if i == 0:
                    self.logger.debug(f"Training models from scratch")
                model = self.init_model(train_dataloader)
                training_loop = self.init_training_loop(False)  # prevent write
                training_loop.train(model, train_dataloader)
                if i == 1:
                    training_loop.to_yaml(self.output_dir / self.loop_filename)
                model.save(model_path)
            model.freeze()
            yield model

    def run(self) -> Any:

        colnames = ["perturbed_feature", "target_feature", "prob", "bayes_k"]
        self.init_csv_writer(
            self.output_dir / self.results_filename,
            fieldnames=colnames,
            extrasaction="ignore",
        )

        # Prep dataloaders
        train_dataloader = self.make_dataloader()
        test_dataloader = self.make_dataloader(split="test")
        num_discrete_indices = [test_dataloader.dataset.num_discrete_features]
        dataset = test_dataloader.dataset

        # Gather perturbations
        perturbation_names: list[str] = []
        if self.perturbation_config.target_feature_name is None:
            perturbation_names.extend(
                dataset.feature_names_of(self.perturbation_config.target_dataset_name)
            )
        else:
            perturbation_names.append(self.perturbation_config.target_feature_name)
        num_perturbations = len(perturbation_names)

        # Compute bayes factor per feature per perturbation
        for i, perturbed_feature_name in enumerate(perturbation_names, 1):
            mean_diff = None
            for model in self.get_trained_model(train_dataloader):
                # Perturb feature
                dataset.perturb(
                    self.perturbation_config.target_dataset_name,
                    perturbed_feature_name,
                    self.perturbation_config.target_value,
                )

                # Compute reconstruction differences (only continuous)
                diff_list = []
                for orig_batch, pert_batch, pert_mask in test_dataloader:
                    _, orig_recon = model.reconstruct(orig_batch)
                    _, pert_recon = model.reconstruct(pert_batch)
                    _, orig_input = torch.tensor_split(
                        orig_batch, num_discrete_indices, dim=-1
                    )
                    diff = pert_recon - orig_recon
                    diff[orig_input == 0] = 0.0  # mark NaN as 0
                    diff = diff[pert_mask, :]  # ignore unperturbed features
                    diff_list.append(diff)

                dataset.remove_perturbation()

                # Concatenate and normalize reconstruction differences
                cat_diff = torch.cat(diff_list) / self.num_refits
                if mean_diff is None:
                    mean_diff = cat_diff
                else:
                    mean_diff += cat_diff

            self.logger.info(f"Perturbing ({i}/{num_perturbations})")
            assert mean_diff is not None

            prob = torch.sum(mean_diff > 1e-8, dim=0) / mean_diff.count_nonzero(dim=0)
            bayes_k = torch.log(prob + 1e-8) - torch.log(1 - prob + 1e-8)
            abs_prob = torch.special.expit(torch.abs(bayes_k))

            self.write_cols(
                {
                    "perturbed_feature": [perturbed_feature_name]
                    * dataset.num_continuous_features,
                    "target_feature": dataset.continuous_feature_names,
                    "prob": abs_prob.numpy(),
                    "bayes_k": bayes_k.numpy(),
                }
            )

        self.logger.info("Complete! Writing out results")
        self.close_csv_writer()

        # Sort results, compute FDR
        assert self.csv_filepath is not None

        results = pd.read_csv(self.csv_filepath)
        results.sort_values("prob", ascending=False, inplace=True, ignore_index=True)
        results["fdr"] = np.cumsum(1 - results["prob"]) / np.arange(1, len(results) + 1)
        results["pred_significant"] = results["fdr"] < self.sig_threshold

        sig = results[results.pred_significant]
        self.logger.info(f"Significant associations found: {len(sig)}")

        if self.write_only_sig:
            results = sig
            results.drop(columns=["pred_significant"], inplace=True)

        results.to_csv(self.csv_filepath, index=False)
