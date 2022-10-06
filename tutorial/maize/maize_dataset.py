import argparse
from collections import namedtuple
from hashlib import md5
from pathlib import Path
from warnings import warn

import pandas as pd
import requests

FileInfo = namedtuple("FileInfo", ["url", "expected_checksum", "expected_size"])
FILES = {
    "maize_microbiome": FileInfo(
        "https://github.com/jorgemf/DeepLatentMicrobiome/raw/91e384b7115978bb3c"
        "d0f61c7dd3d8ffc866efc3/Datasets/otu_table_all_80.csv",
        "3e741373d53e98947b484a528e0238ae",
        7897316,
    ),
    "maize_microbiome_names": FileInfo(
        "https://github.com/jorgemf/DeepLatentMicrobiome/raw/91e384b7115978bb3c"
        "d0f61c7dd3d8ffc866efc3/Datasets/tax_table_all_80_cleanNames.csv",
        "b4c57c6f34b1afa3e4a81ef742e931d6",
        67346,
    ),
    "maize_metadata": FileInfo(
        "https://github.com/jorgemf/DeepLatentMicrobiome/raw/91e384b7115978bb3c"
        "d0f61c7dd3d8ffc866efc3/Datasets/metadata_table_all_80.csv",
        "ab10b0ffeaef5ca23deb0f2bc4fad65a",
        8146942,
    ),
}


def fetch(dataset_name: str, destination: Path) -> Path:
    """Downloads a maize dataset.

    Args:
        dataset_name:
            Name of the dataset to download: `maize_microbiome`,
            `maize_metadata`, or `maize_microbiome_names`
        destination:
            Where to save the data as

    Returns:
        Path to downloaded data.
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


def prepare_data(savedir: Path) -> None:
    """Downloads the maize datasets and formats them for MOVE.
    
    Args:
        savedir: directory to save the data in
    """

    # Process OTUs
    values_path = fetch("maize_microbiome", savedir / "maize_microbiome.tsv")
    values = pd.read_csv(values_path, sep="\t", index_col=0).T.sort_index()
    values.to_csv(values_path, sep="\t")

    # Process metadata (split into separate files)
    values_path = fetch("maize_metadata", savedir / "maize_metadata.tsv")
    values = pd.read_csv(values_path, sep="\t", index_col=0).sort_index()
    values.index.name = None

    col_names = ["Field", "INBREDS", "Maize_Line"]
    var_names = ["field", "variety", "line"]

    for col_name, var_name in zip(col_names, var_names):
        category = values[col_name]
        category.name = var_name
        category.to_frame().to_csv(savedir / f"maize_{var_name}.tsv", sep="\t")

    data = values[["age", "Temperature", "Precipitation3Days"]]
    data.columns = [col_name.lower() for col_name in data.columns]
    data.to_csv(values_path, sep="\t")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("savedir", type=str)
    args = parser.parse_args()

    savedir = Path(getattr(args, "savedir"))

    prepare_data(savedir)
