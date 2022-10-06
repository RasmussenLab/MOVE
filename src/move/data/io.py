__all__ = [
    "dump_names",
    "dump_mappings",
    "load_mappings",
    "load_preprocessed_data",
    "read_config",
    "read_names",
    "read_tsv",
]

import json
from pathlib import Path
from typing import TYPE_CHECKING, Optional, Union, cast

import hydra
import numpy as np
import pandas as pd
from omegaconf import DictConfig, OmegaConf

from move import conf
from move.core.typing import BoolArray, FloatArray, ObjectArray, PathLike


def read_config(filepath: Optional[Union[str, Path]] = None) -> DictConfig:
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
        return cast(DictConfig, OmegaConf.merge(base_config, user_config))
    else:
        return base_config


def load_categorical_dataset(filepath: PathLike) -> FloatArray:
    """Loads categorical data in a NumPy file.

    Args:
        filepath: Path to NumPy file containing a categorical dataset

    Returns:
        NumPy array containing categorical data
    """
    return np.load(filepath).astype(np.float32)


def load_continuous_dataset(filepath: PathLike) -> tuple[FloatArray, BoolArray]:
    """Loads continuous data from a NumPy file and filters out columns
    (features) whose sum is zero. Additionally, encodes NaN values as zeros.

    Args:
        filepath: Path to NumPy file containing a continuous dataset

    Returns:
        Tuple containing (1) the NumPy dataset and (2) a mask marking columns
        (i.e., features) that were not filtered out
    """
    data = np.load(filepath).astype(np.float32)
    data[np.isnan(data)] = 0
    mask_col = np.abs(data).sum(axis=0) != 0
    data = data[:, mask_col]
    return data, mask_col


def load_preprocessed_data(
    path: Path,
    categorical_dataset_names: list[str],
    continuous_dataset_names: list[str],
) -> tuple[list[FloatArray], list[list[str]], list[FloatArray], list[list[str]]]:
    """Loads the pre-processed categorical and continuous data.

    Args:
        path: Where the data is saved
        categorical_dataset_names: List of names of the categorical datasets
        continuous_dataset_names: List of names of the continuous datasets

    Returns:
        Returns two pairs of list containing (1, 3) the pre-processed data and
        (2, 4) the lists of names of each feature
    """

    categorical_data, categorical_var_names = [], []
    for dataset_name in categorical_dataset_names:
        data = load_categorical_dataset(path / f"{dataset_name}.npy")
        categorical_data.append(data)
        var_names = read_names(path / f"{dataset_name}.txt")
        categorical_var_names.append(var_names)

    continuous_data, continuous_var_names = [], []
    for dataset_name in continuous_dataset_names:
        data, keep = load_continuous_dataset(path / f"{dataset_name}.npy")
        continuous_data.append(data)
        var_names = read_names(path / f"{dataset_name}.txt")
        var_names = [name for i, name in enumerate(var_names) if keep[i]]
        continuous_var_names.append(var_names)

    return (
        categorical_data,
        categorical_var_names,
        continuous_data,
        continuous_var_names,
    )


def read_names(path: PathLike) -> list[str]:
    """Reads sample names from a text file. The text file should have one line
    per sample name.

    Args:
        path: Path to the text file

    Returns:
        A list of sample names
    """
    with open(path, "r", encoding="utf-8") as file:
        return [i.strip() for i in file.readlines()]


def read_tsv(
    path: PathLike, sample_names: Optional[list[str]] = None
) -> tuple[ObjectArray, np.ndarray]:
    """Read a dataset from a TSV file. The TSV is expected to have an index
    column (0th index).

    Args:
        path: Path to TSV
        index: List of sample names used to sort/filter samples

    Returns:
        Tuple containing (1) feature names and (2) 2D matrix (samples x
        features)
    """
    data = pd.read_csv(path, index_col=0, sep="\t")
    if sample_names is not None:
        data.index = data.index.astype(str, False)
        data = data.loc[sample_names]
    return data.columns.values, data.values


def load_mappings(path: PathLike) -> dict[str, dict[str, int]]:
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def dump_mappings(path: PathLike, mappings: dict[str, dict[str, int]]) -> None:
    with open(path, "w", encoding="utf-8") as file:
        json.dump(mappings, file, indent=4, ensure_ascii=False)


def dump_names(path: PathLike, names: np.ndarray) -> None:
    with open(path, "w", encoding="utf-8") as file:
        file.writelines([f"{name}\n" for name in names])
