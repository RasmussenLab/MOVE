__all__ = ["EncodeData"]

import numpy as np
import torch

from move.conf.tasks import InputConfig
from move.core.typing import PathLike
from move.data import io, preprocessing
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
    """

    mappings: dict[str, dict[str, int]]

    def __init__(
        self,
        raw_data_path: PathLike,
        interim_data_path: PathLike,
        sample_names_filename: str,
        discrete_inputs: list[InputConfig],
        continuous_inputs: list[InputConfig],
    ) -> None:
        super().__init__(raw_data_path, interim_data_path)
        self.sample_names_filepath = self.input_dir / f"{sample_names_filename}.txt"
        self.discrete_inputs = discrete_inputs
        self.continuous_inputs = continuous_inputs
        self.mappings = {}

    def encode_datasets(
        self,
        input_configs: list[InputConfig],
        default_op_name: preprocessing.PreprocessingOpName,
    ) -> None:
        """Read TSV files containing datasets and run pre-processing operations.

        Args:
            input_configs:
                List of configs, each with a dataset file name and operation
                name. Valid operation names are 'none', 'one_hot_encoding', and
                'standardization'
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
            enc_data_path = self.output_dir / f"{dataset_name}.pt"
            if enc_data_path.exists():
                self.logger.warning(
                    f"File '{enc_data_path.name}' already exists. It will be "
                    "overwritten."
                )
            # Read and encode data
            feature_names, values = io.read_tsv(dataset_path, self.sample_names)
            mapping = None
            if op_name == "standardization":
                values, mask_1d = preprocessing.scale(values)
                feature_names = feature_names[mask_1d]
                self.logger.debug(f"Columns with zero variance: {np.sum(~mask_1d)}")
            elif op_name == "one_hot_encoding":
                values, mapping = preprocessing.one_hot_encode(values)
                self.mappings[dataset_name] = mapping
            else:
                values = preprocessing.fill(values)
            tensor = torch.from_numpy(values).float()
            # Save data
            data = {
                "dataset_name": dataset_name,
                "tensor": tensor,
                "feature_names": feature_names.tolist(),
            }
            if mapping:
                data["mapping"] = mapping
            torch.save(data, enc_data_path)

    def run(self) -> None:
        """Encode data."""
        self.logger.info("Beginning task: encode data")
        self.sample_names = io.read_names(self.sample_names_filepath)
        self.encode_datasets(self.discrete_inputs, "one_hot_encoding")
        self.encode_datasets(self.continuous_inputs, "standardization")
