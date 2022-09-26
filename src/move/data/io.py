__all__ = [
    "dump_feature_names",
    "read_config",
    "read_sample_names",
    "read_tsv",
]

from pathlib import Path
from typing import Union, TYPE_CHECKING

import hydra
import numpy as np
import pandas as pd
from numpy.typing import ArrayLike
from omegaconf import DictConfig, OmegaConf

from move import conf
from move.core.typing import PathLike

if TYPE_CHECKING:
    from move.conf.schema import MOVEConfig


def read_config(filepath: Union[str, Path] = None) -> DictConfig:
    """Composes configuration for the MOVE framework.

    Args:
        filepath: Path to YAML configuration file

    Returns:
        Merged configuration
    """
    with hydra.initialize_config_module(conf.__name__):
        base_config = hydra.compose("main")

    if filepath is not None:
        user_config = OmegaConf.load(filepath)
        return OmegaConf.merge(base_config, user_config)
    else:
        return base_config


def read_cat(filepath: PathLike) -> np.ndarray:
    """Reads categorical data in a NumPy file.

    Args:
        filepath: Path to NumPy file containing a categorical dataset

    Returns:
        The dataset
    """
    data = np.load(filepath)
    data = data.astype(np.float32)
    return data


def read_con(filepath: PathLike) -> tuple[np.ndarray]:
    """Reads continuous data in a NumPy file and filters out columns (features)
    whose sum is zero.

    Args:
        filepath: Path to NumPy file containing a continuous dataset

    Returns:
        The dataset and a mask of excluded features
    """
    data = np.load(filepath)
    data = data.astype(np.float32)
    data[np.isnan(data)] = 0
    mask_col = data.sum(axis=0) != 0
    data = data[:, mask_col]
    return data, mask_col


def read_header(filepath: PathLike, mask: ArrayLike = None) -> list[str]:
    """Reads features names from the headers

    Args:
        filepath: Path to file
        mask: Mask to exclude names from header

    Returns:
        list of strings of elements in the header
    """

    header = pd.read_csv(filepath, sep="\t", header=None)
    header = header.iloc[:, 0].astype("str")
    if mask is not None:
        header = header[mask]
    return header.to_list()


def read_data(
    config: "MOVEConfig",
) -> tuple[list[np.ndarray], list[list[str]], list[np.ndarray], list[list[str]]]:
    """Reads the pre-processed categorical and continuous data.

    Args:
        config: Hydra configuration

    Returns:
        Returns two pairs of list containing the pre-processed data and its
        headers
    """
    interim_path = Path(config.data.interim_data_path)

    categorical_data, categorical_headers = [], []
    for input_config in config.data.categorical_inputs:
        name = input_config.name
        data = read_cat(interim_path / f"{name}.npy")
        categorical_data.append(data)
        header = read_header(interim_path / f"{name}.txt")
        categorical_headers.append(header)

    continuous_data, continuous_headers = [], []
    for input_config in config.data.continuous_inputs:
        name = input_config.name
        data, mask = read_con(interim_path / f"{name}.npy")
        continuous_data.append(data)
        header = read_header(interim_path / f"{name}.txt", mask)
        continuous_headers.append(header)

    return categorical_data, categorical_headers, continuous_data, continuous_headers


def read_sample_names(path: PathLike) -> list[str]:
    """Reads sample names from a text file. The text file should have one line
    per sample name.

    Args:
        path: Path to the text file

    Returns:
        A list of sample names
    """
    with open(path, "r", encoding="utf-8") as file:
        return [i.strip() for i in file.readlines()]


def read_tsv(path: PathLike) -> tuple[np.ndarray, np.ndarray]:
    """Read a dataset from a TSV file. The TSV is expected to have an index
    column (0th index).

    Args:
        path: Path to TSV

    Returns:
        Tuple containing (1) feature names and (2) 2D matrix (samples x
        features)
    """
    data = pd.read_csv(path, index_col=0, sep="\t").sort_index()
    return data.columns.values, data.values


def dump_feature_names(path: PathLike, names: np.ndarray) -> None:
    with open(path, "w", encoding="utf-8") as file:
        file.writelines([f"{name}\n" for name in names])
