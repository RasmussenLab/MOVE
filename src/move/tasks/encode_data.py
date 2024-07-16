__all__ = ["EncodeData"]

import numpy as np
import torch

from move.conf.tasks import InputConfig
from move.core.typing import PathLike
from move.data import io, preprocessing
from move.data.splitting import split_samples
from move.tasks.base import ParentTask


class EncodeData(ParentTask):
    """Encode discrete and continuous datasets. By default, discrete data is
    one-hot encoded, whereas continuous data is z-score normalized.

    Args:
        raw_data_path:
            Directory where "raw data" is stored
        interim_data_path:
            Directory where pre-processed data will be saved
        sample_names_filename:
            Filename of file containing names given to each sample
        discrete_inputs:
            List of configs for each discrete dataset. Each config is a dict
            containing keys 'name' and 'preprocessing'
        continuous_inputs:
            Same as `discrete_inputs`, but for continuous datasets
        train_frac:
            Fraction of samples corresponding to training set.
        test_frac:
            Fraction of samples corresponding to test set.
        valid_frac:
            Fraction of samples corresponding to validation set.
    """

    indices_filename = "indices.pt"
    sample_names: list[str]
    train_indices: torch.Tensor

    def __init__(
        self,
        raw_data_path: PathLike,
        interim_data_path: PathLike,
        sample_names_filename: str,
        discrete_inputs: list[InputConfig],
        continuous_inputs: list[InputConfig],
        train_frac: float = 0.9,
        test_frac: float = 0.1,
        valid_frac: float = 0.0,
    ) -> None:
        super().__init__(raw_data_path, interim_data_path)
        self.sample_names_filepath = self.input_dir / f"{sample_names_filename}.txt"
        self.discrete_inputs = discrete_inputs
        self.continuous_inputs = continuous_inputs
        self.split_fracs = (train_frac, test_frac, valid_frac)

    def encode_datasets(
        self,
        input_configs: list[InputConfig],
        default_op_name: preprocessing.PreprocessingOpName,
    ) -> None:
        """Read TSV files containing datasets and run pre-processing operations.

        Args:
            input_configs:
                List of configs, each with a dataset file name and operation
                name. Valid operation names are 'none', 'one_hot_encode', 'standardize',
                'log_and_standardize'
            default_op_name:
                Default operation if no operation in config
        """
        for config in input_configs:
            op_name: preprocessing.PreprocessingOpName = getattr(
                config, "preprocessing", default_op_name
            )
            action_name = "Reading" if op_name == "none" else "Encoding"
            dataset_name = getattr(config, "name")
            self.logger.info(f"{action_name} '{dataset_name}'")
            dataset_path = self.input_dir / f"{dataset_name}.tsv"
            if not dataset_path.exists():
                dataset_path = self.input_dir / f"{dataset_name}.csv"
            enc_data_path = self.output_dir / f"{dataset_name}.pt"
            if enc_data_path.exists():
                self.logger.warning(
                    f"File '{enc_data_path.name}' already exists. It will be "
                    "overwritten."
                )
            # Read and encode data
            feature_names, values = io.read_tsv(dataset_path, self.sample_names)
            mapping = None
            if op_name in ("standardize", "log_and_standardize"):
                values = preprocessing.standardize(values, self.train_indices)
            elif op_name == "one_hot_encode":
                values, mapping = preprocessing.one_hot_encode(values)
            else:
                values = preprocessing.fill(values)
            tensor = torch.from_numpy(values).float()
            # Save data
            data = {
                "dataset_name": dataset_name,
                "tensor": tensor,
                "feature_names": feature_names.tolist(),
            }
            if mapping is not None:
                data["mapping"] = mapping
            torch.save(data, enc_data_path, pickle_protocol=4)

    def split_samples(self) -> None:
        """Create indices to split data into training, test, and validation subsets."""
        indices = split_samples(len(self.sample_names), *self.split_fracs)
        ind_dict = dict(
            zip(("train_indices", "test_indices", "valid_indices"), indices)
        )
        torch.save(ind_dict, self.output_dir / self.indices_filename, pickle_protocol=4)
        self.train_indices = ind_dict["train_indices"]

    def run(self) -> None:
        """Encode data."""
        self.logger.info("Beginning task: encode data")
        self.sample_names = io.read_names(self.sample_names_filepath)
        self.split_samples()
        self.encode_datasets(self.discrete_inputs, "one_hot_encode")
        self.encode_datasets(self.continuous_inputs, "standardize")
