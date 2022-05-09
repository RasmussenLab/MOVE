__all__ = ["read_config"]

from pathlib import Path
from typing import Union

import hydra
import numpy as np
from omegaconf import OmegaConf

from move import conf
from move.conf.schema import MOVEConfig


def read_config(filepath: Union[str, Path]) -> MOVEConfig:
    """Composes configuration for the MOVE framework.

    Parameters
    ----------
    filepath : Union[str, Path]
        Path to YAML configuration file

    Returns
    -------
    move.conf.MOVEConfig
    """
    with hydra.initialize_config_module(conf.__name__):
        base_config = hydra.compose("main")

    user_config = OmegaConf.load(filepath)
    return OmegaConf.merge(base_config, user_config)


# NOTE: Adapted from Notebook #4

# Functions for reading data
def read_cat(file):
    data = np.load(file)
    data = data.astype(np.float32)
    data_input = data.reshape(data.shape[0], -1)
    return data, data_input  # ???: Is second output necessary?


def read_con(file):
    data = np.load(file)
    data = data.astype(np.float32)
    data[np.isnan(data)] = 0
    consum = data.sum(axis=0)
    mask_col = consum != 0
    data = data[:, mask_col]
    return data, mask_col


def read_header(file, mask=None, start=1):
    with open(file, "r") as f:
        h = f.readline().rstrip().split("\t")[start:]
    if not mask is None:
        h = np.array(h)
        h = h[mask]
    return h


def read_data(config: MOVEConfig):
    categorical_vars = []
    continuous_vars = []

    categorical_headers = []
    continuous_headers = []

    output_path = Path(config.data.interim_data_path)

    for var_list, header_list, input_configs in [
        (categorical_vars, categorical_headers, config.data.categorical_inputs),
        (continuous_vars, continuous_headers, config.data.continuous_inputs),
    ]:
        for input_config in input_configs:
            var, _ = read_cat(output_path / f"{input_config.name}.npy")
            header = read_header(output_path / f"{input_config.name}.tsv")
            var_list.append(var)
            header_list += header

    return categorical_vars, categorical_headers, continuous_vars, continuous_headers
