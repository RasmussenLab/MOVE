from collections import namedtuple
from warnings import warn
from hashlib import md5
from pathlib import Path

import requests
import numpy as np

FileInfo = namedtuple("FileInfo", ["url", "expected_checksum", "expected_size"])
FILES = {
    "maize_microbiome": FileInfo(
        "https://github.com/jorgemf/DeepLatentMicrobiome/raw/91e384b7115978bb3c"
        "d0f61c7dd3d8ffc866efc3/Datasets/otu_table_all_80.csv",
        "3e741373d53e98947b484a528e0238ae",
        7897316,
    ),
    "maize_metadata": FileInfo(
        "https://github.com/jorgemf/DeepLatentMicrobiome/raw/91e384b7115978bb3c"
        "d0f61c7dd3d8ffc866efc3/Datasets/metadata_table_all_80.csv",
        "ab10b0ffeaef5ca23deb0f2bc4fad65a",
        8146942,
    ),
}


def fetch(dataset_name: str, destination: Path):
    """Downloads a maize dataset.

    Parameters
    ----------
    dataset_name: {'maize_microbiome', 'maize_metadata'}
        Name of the dataset to download
    destination: Path
        Where to save the data as
    """
    if dataset_name not in FILES:
        raise KeyError(
            "Unexpected dataset name. Available names are:"
            f" {', '.join(FILES.keys())}."
        )

    # Check if file exists
    destination = Path(destination)
    if destination.exists():
        warn("A file exists at destination. Skipping download.")
        return destination

    # Download file
    file_info = FILES[dataset_name]
    hash_ = md5()
    with requests.get(file_info.url, stream=True) as request:
        request.raise_for_status()
        with open(destination, "wb") as file:
            for chunk in request.iter_content(chunk_size=8192):
                file.write(chunk)
                hash_.update(chunk)

    # Test file integrity
    size = destination.stat().st_size
    checksum = hash_.hexdigest()
    if checksum != file_info.expected_checksum or size != file_info.expected_size:
        raise IOError(f"The downloaded file is corrupted. Try redownloading.")

    return destination

def scale(x: np.array):
    """Performs standard scaling.
    
    Parameters
    ----------
    x : array-like
    """
    return (x - np.mean(x, axis=0, keepdims=True)) / np.std(x, axis=0, keepdims=True)

def rclr(x: np.array, axis: int = 0):
    """Performs robust centered log-ratio (rclr) transformation.
    
    Parameters
    ----------
    x : array-like
        Array containing compositional data
    axis : int
        Axis along which the geometric mean is computed. Default is 0 (per
        sample/column).
    """
    logx = np.ma.log(x).filled(np.nan)
    gmean = np.nanmean(logx, axis=axis, keepdims=True)
    return logx - gmean
