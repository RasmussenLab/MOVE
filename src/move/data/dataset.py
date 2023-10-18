__all__ = ["DiscreteDataset", "ContinuousDataset", "MoveDataset"]

import operator
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Literal, NoReturn, Optional, Union, cast

import torch
from torch.utils.data import Dataset

from move.core.typing import EncodedData

DataType = Literal["continuous", "discrete"]


class NamedDataset(Dataset, ABC):
    """A dataset with a name and names for its features.

    Args:
        tensor: Data
        name: Name of the dataset
        feature_names: Name of each feature contained in dataset"""

    def __init__(
        self,
        tensor: torch.Tensor,
        dataset_name: str,
        feature_names: Optional[list[str]] = None,
    ) -> None:
        self.tensor = tensor
        self.name = dataset_name
        if feature_names is not None:
            self._validate_names(feature_names)
        self.feature_names = feature_names

    def __add__(self, other: "NamedDataset") -> "MoveDataset":
        return MoveDataset(self, other)

    def __getitem__(self, index: int) -> torch.Tensor:
        return self.tensor[index]

    def __len__(self):
        return self.tensor.size(0)

    def __repr__(self) -> str:
        return str(self)

    def __str__(self) -> str:
        return f'{self.__class__.__name__}("{self.name}")'

    def _validate_names(self, feature_names: list[str]) -> Union[None, NoReturn]:
        if len(feature_names) != self.num_features:
            raise ValueError(
                f"Number of features ({self.num_features}) must match "
                f"number of feature names {len(feature_names)}."
            )

    @property
    @abstractmethod
    def data_type(self) -> DataType:
        raise NotImplementedError()

    @property
    def num_features(self) -> int:
        return self.tensor.size(1)

    @classmethod
    def load(cls, path: Path):
        """Load dataset.

        Args:
            path: Path to encoded data
        """
        enc_data = cast(EncodedData, torch.load(path))
        return cls(**enc_data)


class DiscreteDataset(NamedDataset):
    """A dataset for discrete values. Discrete data is expected to be a one-hot
    encoded tensor of three dimensions corresponding to number of samples,
    features, and classes."""

    def __init__(
        self,
        tensor: torch.Tensor,
        dataset_name: str,
        feature_names: Optional[list[str]] = None,
        mapping: Optional[dict[str, int]] = None,
    ):
        if tensor.dim() != 3:
            raise ValueError("Discrete datasets must have three dimensions.")
        *_, dim0, dim1 = tensor.shape
        self.original_shape = (dim0, dim1)
        self.mapping = mapping
        flattened_tensor = torch.flatten(tensor, start_dim=1)
        super().__init__(flattened_tensor, dataset_name, feature_names)

    def _validate_names(self, feature_names: list[str]) -> None:
        if len(feature_names) != self.original_shape[0]:
            raise ValueError(
                f"Number of features ({self.original_shape[0]}) must match "
                f"number of feature names {len(feature_names)}."
            )

    @property
    def data_type(self) -> DataType:
        return "discrete"

    @property
    def num_classes(self) -> int:
        return self.original_shape[1]

    @property
    def num_features(self) -> int:
        return operator.mul(*self.original_shape)


class ContinuousDataset(NamedDataset):
    """A dataset for continuous values."""

    @property
    def data_type(self) -> DataType:
        return "continuous"


class MoveDataset(Dataset):
    """Multi-omics dataset composed of one or more ."""

    def __init__(self, *args: NamedDataset) -> None:
        self.datasets = sorted(args, key=operator.attrgetter("data_type"), reverse=True)
        if not all(
            len(self.datasets[0]) == len(dataset) for dataset in self.datasets[1:]
        ):
            raise ValueError("Size mismatch between datasets")

    def __getitem__(self, index: int) -> torch.Tensor:
        return torch.cat([dataset[index] for dataset in self.datasets], dim=-1)

    def __len__(self) -> int:
        return len(self.datasets[0])

    def __repr__(self) -> str:
        return str(self)

    def __str__(self) -> str:
        dataset_count = len(self.datasets)
        s = "s" if dataset_count != 1 else ""
        return f"{self.__class__.__name__}({dataset_count} dataset{s})"

    def _repr_html_(self) -> str:
        rows = ""
        for dataset in self.datasets:
            num_classes = (
                str(dataset.num_classes)
                if isinstance(dataset, DiscreteDataset)
                else "N/A"
            )
            rows += (
                "<tr><td style='text-align:left'>"
                + "</td><td style='text-align:center'>".join(
                    (
                        dataset.name,
                        dataset.data_type,
                        f"{dataset.num_features:,}",
                        num_classes,
                    )
                )
                + "</td></tr>"
            )
        return f"""<table>
            <thead>
            <tr>
                <th colspan="4" style="text-align:center">
                    <abbr title="Multi-omics variational auto-encoders"
                    style="font-variant:small-caps; text-transform: lowercase">
                    MOVE</abbr> dataset ({len(self):,} samples)
                </th>
            </tr>
            <tr>
                <th style="text-align:center">data</th>
                <th style="text-align:center">type</th>
                <th style="text-align:center"># features</th>
                <th style="text-align:center"># classes</th>
            </tr>
            </thead>
            <tbody>{rows}</tbody>
        </table>"""

    @property
    def num_features(self) -> int:
        return sum(dataset.num_features for dataset in self.datasets)

    @property
    def num_discrete_features(self) -> int:
        return sum(
            dataset.num_features
            for dataset in self.datasets
            if isinstance(dataset, DiscreteDataset)
        )

    @property
    def num_continuous_features(self) -> int:
        return sum(
            dataset.num_features
            for dataset in self.datasets
            if isinstance(dataset, ContinuousDataset)
        )

    @property
    def discrete_shapes(self) -> list[tuple[int, int]]:
        return [
            dataset.original_shape
            for dataset in self.datasets
            if isinstance(dataset, DiscreteDataset)
        ]

    @property
    def continuous_shapes(self) -> list[int]:
        return [
            dataset.num_features
            for dataset in self.datasets
            if isinstance(dataset, ContinuousDataset)
        ]

    @classmethod
    def load(
        cls,
        path: Path,
        discrete_dataset_names: list[str],
        continuous_dataset_names: list[str],
    ) -> "MoveDataset":
        """Load dataset.

        Args:
            path: Path to encoded data
            discrete_dataset_names: Names of discrete datasets
            continuous_dataset_names: Names of continuous datasets
        """
        datasets = []
        for dataset_name in discrete_dataset_names:
            datasets.append(DiscreteDataset.load(path / f"{dataset_name}.pt"))
        for dataset_name in continuous_dataset_names:
            datasets.append(ContinuousDataset.load(path / f"{dataset_name}.pt"))
        return cls(*datasets)
