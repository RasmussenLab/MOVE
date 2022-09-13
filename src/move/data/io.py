__all__ = ["read_config"]

from pathlib import Path
from typing import Union

import hydra
import numpy as np
import pandas as pd
from numpy.typing import ArrayLike
from omegaconf import OmegaConf

from move import conf
from move.conf.schema import MOVEConfig
from move.core.typing import PathLike


def read_config(filepath: Union[str, Path] = None) -> MOVEConfig:
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


def read_header(filepath: PathLike, mask: ArrayLike=None) -> list[str]:
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
    config: MOVEConfig,
) -> tuple[list[np.ndarray], list[list[str]], list[np.ndarray], list[list[str]]]:
    """Reads the pre-processed categorical and continuous data.

    Args:
        config: Hydra configuration

    Returns:
        Returns two pairs of list containing the pre-processed data and its
        headers
    """
    interim_path = Path(config.data.interim_data_path)
    headers_path = Path(config.data.headers_path)

    categorical_data, categorical_headers = [], []
    for input_config in config.data.categorical_inputs:
        name = input_config.name
        data = read_cat(interim_path / f"{name}.npy")
        categorical_data.append(data)
        header = read_header(headers_path / f"{name}.txt")
        categorical_headers.append(header)

    continuous_data, continuous_headers = [], []
    for input_config in config.data.continuous_inputs:
        name = input_config.name
        data, mask = read_con(interim_path / f"{name}.npy")
        continuous_data.append(data)
        header = read_header(headers_path / f"{name}.txt", mask)
        continuous_headers.append(header)

    return categorical_data, categorical_headers, continuous_data, continuous_headers
