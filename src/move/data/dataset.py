__all__ = ["DiscreteDataset", "ContinuousDataset", "MoveDataset"]

import operator
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Literal, NoReturn, Optional, Type, TypeVar, Union, cast

import pandas as pd
import torch
from torch.utils.data import Dataset

from move.core.exceptions import UnsetProperty
from move.core.typing import EncodedData

DataType = Literal["continuous", "discrete"]
Index = Union[int, tuple[str, int], tuple[int, int]]
T = TypeVar("T", bound="NamedDataset")


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
        else:
            feature_names = [f"{self.name}_{i}" for i in range(self.num_features)]
        self.feature_names = feature_names

    def __add__(self, other: "NamedDataset") -> "MoveDataset":
        return MoveDataset(self, other)

    def __getitem__(self, index: int) -> torch.Tensor:
        return self.tensor[index]

    def __len__(self):
        return self.tensor.size(0)

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}("{self.name}")'

    def __str__(self) -> str:
        return self.name

    def _validate_names(self, feature_names: list[str]) -> Union[None, NoReturn]:
        num_feature_names = len(feature_names)
        if num_feature_names != self.num_feature_names:
            raise ValueError(
                f"Number of features ({self.num_features}) must match "
                f"number of feature names {len(feature_names)}."
            )
        if num_feature_names != len(set(feature_names)):
            raise ValueError("Duplicate feature names")

    @property
    @abstractmethod
    def data_type(self) -> DataType:
        raise NotImplementedError()

    @property
    def num_features(self) -> int:
        return self.tensor.size(1)

    @property
    def num_feature_names(self) -> int:
        return self.num_features

    @classmethod
    def load(cls: Type[T], path: Path) -> T:
        """Load dataset.

        Args:
            path: Path to encoded data
        """
        enc_data = cast(EncodedData, torch.load(path))
        return cls(**enc_data)

    def select(self, feature_name: str) -> torch.Tensor:
        """Slice and return values corresponding to a single feature."""
        slice_ = self.feature_slice(feature_name)
        return self.tensor[:, slice_]

    def feature_slice(self, feature_name: str) -> slice:
        """Return a slice object containing start and end position of a feature
        in the dataset."""
        if feature_name not in self.feature_names:
            raise KeyError(f"{feature_name} not found")
        idx = self.feature_names.index(feature_name)
        num_classes = getattr(self, "num_classes", 1)
        start = idx * num_classes
        stop = (idx + 1) * num_classes
        return slice(start, stop)


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

    @property
    def data_type(self) -> DataType:
        return "discrete"

    @property
    def num_classes(self) -> int:
        return self.original_shape[1]

    @property
    def num_features(self) -> int:
        return operator.mul(*self.original_shape)

    @property
    def num_feature_names(self) -> int:
        return self.original_shape[0]

    def one_hot_encode(self, value: Union[str, float, None]) -> torch.Tensor:
        """One-hot encode a single value.

        Args:
            value: category"""
        if self.mapping is None:
            raise ValueError("Unknown encoding")
        encoded_value = torch.zeros(len(self.mapping))
        if not pd.isna(value):
            code = self.mapping[str(value)]
            encoded_value[code] = 1
        return encoded_value


class ContinuousDataset(NamedDataset):
    """A dataset for continuous values."""

    @property
    def data_type(self) -> DataType:
        return "continuous"


class MoveDataset(Dataset):
    """Multi-omics dataset composed of one or more datasets (both categorical
    and continuous).

    When indexed, returns a flat concatenation of the indexed elements of all
    constituent datasets.

    A MOVE dataset can have a perturbation in one of its features. This
    changes all the values of that feature. A perturbed dataset will return a
    tuple when indexed. In the first position, it will contain the original
    output as if there was no perturbation. The second element of the tuple
    will correspond to the output affected by the perturbation. Lastly, the
    third element is a boolean indicating whether the perturbation changed or
    not the original value."""

    def __init__(self, *args: NamedDataset) -> None:
        if len(args) > 1 and not all(
            len(args[0]) == len(dataset) for dataset in args[1:]
        ):
            raise ValueError("Size mismatch between datasets")
        self._list = sorted(args, key=operator.attrgetter("data_type"), reverse=True)
        self.datasets = {dataset.name: dataset for dataset in self._list}
        if len(self.datasets) != len(args):
            raise ValueError("One or more datasets have the same name")
        self._perturbation = None

    def __getitem__(self, index: int) -> tuple[torch.Tensor, ...]:
        indices = None
        items = [[dataset[index] for dataset in self.datasets.values()]]
        if self.perturbation is not None:
            values = []
            for dataset in self._list:
                if dataset.name == self.perturbation.dataset_name:
                    left, _, right = torch.tensor_split(
                        dataset[index], self.perturbation.feature_indices
                    )
                    values.extend((left, self.perturbation.mapped_value, right))
                    indices = torch.all(
                        dataset[index][self.perturbation.feature_slice]
                        != self.perturbation.mapped_value
                    )
                else:
                    values.append(dataset[index])
            items.append(values)
        out = tuple(torch.cat(item, dim=-1) for item in items)
        if indices is not None:
            out += (indices,)
        return out

    def __len__(self) -> int:
        return len(self._list[0])

    def __repr__(self) -> str:
        dataset_count = len(self._list)
        s = "s" if dataset_count != 1 else ""
        return f"{self.__class__.__name__}({dataset_count} dataset{s})"

    def _repr_html_(self) -> str:
        rows = ""
        for dataset in self._list:
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
        return sum(dataset.num_features for dataset in self._list)

    @property
    def num_discrete_features(self) -> int:
        return sum(
            dataset.num_features
            for dataset in self._list
            if isinstance(dataset, DiscreteDataset)
        )

    @property
    def num_continuous_features(self) -> int:
        return sum(
            dataset.num_features
            for dataset in self._list
            if isinstance(dataset, ContinuousDataset)
        )

    @property
    def discrete_shapes(self) -> list[tuple[int, int]]:
        return [
            dataset.original_shape
            for dataset in self._list
            if isinstance(dataset, DiscreteDataset)
        ]

    @property
    def continuous_shapes(self) -> list[int]:
        return [
            dataset.num_features
            for dataset in self._list
            if isinstance(dataset, ContinuousDataset)
        ]

    @property
    def names(self) -> list[str]:
        return list(self.datasets.keys())

    @property
    def feature_names(self) -> list[str]:
        feature_names = []
        for dataset in self._list:
            if dataset.feature_names:
                feature_names.extend(dataset.feature_names)
            else:
                raise ValueError("Missing feature names in one or more datasets")
        return feature_names

    @property
    def continuous_feature_names(self) -> list[str]:
        feature_names = []
        for dataset in self._list:
            if dataset.data_type == "continuous":
                feature_names.extend(dataset.feature_names)
        return feature_names

    @property
    def discrete_feature_names(self) -> list[str]:
        feature_names = []
        for dataset in self._list:
            if dataset.data_type == "discrete":
                feature_names.extend(dataset.feature_names)
        return feature_names

    @property
    def perturbation(self) -> Optional["Perturbation"]:
        return self._perturbation

    @perturbation.setter
    def perturbation(self, value: Optional["Perturbation"]) -> None:
        if value is not None:
            if value.dataset_name not in self.datasets:
                raise KeyError(
                    f"Target dataset '{value.dataset_name}' not found in "
                    "MOVE dataset"
                )
            dataset = self.datasets[value.dataset_name]
            if value.feature_name not in dataset.feature_names:
                raise KeyError(
                    f"Target feature {value.feature_name} not found in "
                    f"'{dataset}' dataset"
                )
            if isinstance(dataset, DiscreteDataset):
                value.mapped_value = dataset.one_hot_encode(value.target_value)
            else:
                value.mapped_value = torch.FloatTensor([value.target_value])
            value.feature_slice = dataset.feature_slice(value.feature_name)
        self._perturbation = value

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

    def feature_names_of(self, dataset_name: str) -> list[str]:
        """Return feature names of a constituent dataset."""
        return self.datasets[dataset_name].feature_names

    def find(self, feature_name) -> NamedDataset:
        """Return constituent dataset which contains feature name."""
        for dataset in self._list:
            if feature_name in dataset.feature_names:
                return dataset
        raise KeyError(f"{feature_name} not found in any dataset")

    def perturb(
        self, dataset_name: str, feature_name: str, value: Union[str, float, None]
    ) -> None:
        """Add a perturbation to a feature in a constituent dataset.

        Args:
            dataset_name: Name of dataset to perturb
            feature_name: Name of feature in dataset to perturb
            value: Value of perturbation
        """
        self.perturbation = Perturbation(dataset_name, feature_name, value)

    def remove_perturbation(self) -> None:
        """Remove perturbation from dataset."""
        self.perturbation = None

    def select(self, feature_name: str) -> torch.Tensor:
        """Slice and return values corresponding to a single feature. If the
        same feature name exists in more than one dataset, the first matching
        feature will be returned."""
        for dataset in self._list:
            if feature_name in dataset.feature_names:
                return dataset.select(feature_name)
        raise KeyError(f"{feature_name} not found in any dataset")


class Perturbation:
    """Perturbation in a MOVE dataset. A perturbation will target a feature in
    one of the MOVE datasets. All the values of that feature will be replaced
    by the defined target value. For example, target 'metformin' feature in
    'drugs' dataset and change value from 0 to 1."""

    def __init__(
        self,
        target_dataset_name: str,
        target_feature_name: str,
        target_value: Union[str, float, None],
    ) -> None:
        self.dataset_name = target_dataset_name
        self.feature_name = target_feature_name
        self.target_value = target_value

    def __repr__(self) -> str:
        return str(self)

    def __str__(self) -> str:
        return f'{self.__class__.__name__}("{self.dataset_name}/{self.feature_name}")'

    @property
    def feature_indices(self) -> tuple[int, int]:
        slice_ = self.feature_slice
        return slice_.start, slice_.stop

    @property
    def feature_slice(self) -> slice:
        if (slice_ := getattr(self, "_feature_slice", None)) is None:
            raise UnsetProperty("Target feature indices")
        return slice_

    @feature_slice.setter
    def feature_slice(self, value: slice) -> None:
        self._feature_slice = value

    @property
    def mapped_value(self) -> torch.Tensor:
        if (value := getattr(self, "_mapped_value", None)) is None:
            raise UnsetProperty("Encoded target value")
        return value

    @mapped_value.setter
    def mapped_value(self, mapped_value: torch.Tensor) -> None:
        self._mapped_value = mapped_value
