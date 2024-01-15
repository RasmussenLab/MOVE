__all__ = ["Chunk", "SplitInput", "SplitOutput"]

import itertools
import operator
from typing import Any, Optional, Union, Type, TypeVar, cast

import torch
from torch import nn

from move.data.dataset import ContinuousDataset, DiscreteDataset, MoveDataset

DiscreteData = list[torch.Tensor]
ContinuousData = list[torch.Tensor]
ContinuousDistribution = list[tuple[torch.Tensor, ...]]
SplitData = Union[
    tuple[DiscreteData, ContinuousData], tuple[DiscreteData, ContinuousDistribution]
]

T = TypeVar("T", bound="SplitOutput")


class Chunk(nn.Module):
    """Split output into a given number of chunks.

    Args:
        chunks: Number of chunks
        dim: Dimension to apply operation in
    """

    chunks: int
    dim: int

    def __init__(self, chunks: int, dim: int = -1) -> None:
        super().__init__()
        self.chunks = chunks
        self.dim = dim

    def __call__(self, *args: Any, **kwds: Any) -> tuple[torch.Tensor, ...]:
        return super().__call__(*args, **kwds)

    def forward(self, input: torch.Tensor) -> tuple[torch.Tensor, ...]:
        if self.chunks == 1:
            return (input,)
        return tuple(torch.chunk(input, self.chunks, self.dim))


class SplitOutput(nn.Module):
    """Split and re-shape concatenated datasets into their original shapes.

    Args:
        discrete_dataset_shapes:
            Number of features and classes of each discrete dataset.
        continuous_dataset_shapes:
            Number of features of each continuous dataset.
        distribution_name:
            If given, continuous variables will be treated as distribution
            arguments. For instance, for a normal distribution, the continuous
            subset of the output will be split into mean and standard deviation.
    """

    num_discrete_features: int
    num_continuous_features: int
    num_features: int
    num_expected_features: int
    num_distribution_args: int
    discrete_dataset_shapes: list[tuple[int, ...]]
    discrete_dataset_shapes_1d: list[int]
    continuous_dataset_shapes: list[int]
    discrete_split_indices: list[int]
    continuous_split_indices: list[int]

    def __init__(
        self,
        discrete_dataset_shapes: list[tuple[int, ...]],
        continuous_dataset_shapes: list[int],
        distribution_name: Optional[str] = None,
    ) -> None:
        super().__init__()

        self.num_distribution_args = 1
        if distribution_name is not None:
            if distribution_name in ("Normal", "LogNormal"):
                self.num_distribution_args = 2
            elif distribution_name == "StudentT":
                self.num_distribution_args = 3
            elif distribution_name not in ("Bernoulli", "Categorical", "Exponential"):
                raise ValueError("Unsupported distribution")

        self.discrete_dataset_shapes = discrete_dataset_shapes
        self.continuous_dataset_shapes = continuous_dataset_shapes

        # Flatten discrete dataset shapes (normally, 2D)
        (*self.discrete_dataset_shapes_1d,) = itertools.starmap(
            operator.mul, self.discrete_dataset_shapes
        )

        # Count num. features
        self.num_discrete_features = sum(self.discrete_dataset_shapes_1d)
        self.num_continuous_features = sum(self.continuous_dataset_shapes)
        self.num_features = self.num_discrete_features + self.num_continuous_features
        self.num_expected_features = self.num_discrete_features + (
            self.num_continuous_features * self.num_distribution_args
        )

        # Compute split indices
        *self.discrete_split_indices, _ = itertools.accumulate(
            self.discrete_dataset_shapes_1d
        )
        *self.continuous_split_indices, _ = itertools.accumulate(
            [
                shape * self.num_distribution_args
                for shape in self.continuous_dataset_shapes
            ]
        )

    def __call__(
        self, *args: Any, **kwds: Any
    ) -> tuple[DiscreteData, ContinuousDistribution]:
        return super().__call__(*args, **kwds)

    @classmethod
    def from_move_dataset(cls: Type[T], move_dataset: MoveDataset) -> T:
        discrete_dataset_shapes = []
        continuous_dataset_shapes = []
        for dataset in move_dataset:
            if isinstance(dataset, DiscreteDataset):
                discrete_dataset_shapes.append(dataset.original_shape)
            elif isinstance(dataset, ContinuousDataset):
                continuous_dataset_shapes.append(dataset.num_features)
            else:
                raise ValueError("Unsupported dataset type detected.")
        return cls(discrete_dataset_shapes, continuous_dataset_shapes)

    def forward(self, x: torch.Tensor) -> SplitData:
        if x.dim() != 2:
            raise ValueError("Input expected to be 2D.")

        if x.size(1) != self.num_expected_features:
            raise ValueError(
                f"Size mismatch: input ({x.size(1)}) is not equal to expected "
                f"number of features ({self.num_expected_features})."
            )

        # Split into discrete/continuous sets
        discrete_x, continuous_x = torch.tensor_split(
            x, [self.num_discrete_features], dim=-1
        )

        # Split and re-shape discrete set into 3D subsets
        discrete_subsets_flat = torch.tensor_split(
            discrete_x, self.discrete_split_indices, dim=-1
        )
        discrete_subsets = [
            torch.reshape(subset, (-1, *shape))
            for subset, shape in zip(
                discrete_subsets_flat, self.discrete_dataset_shapes
            )
        ]

        # Split continuous set into subsets
        # If outputs are distributions, split into correct # of arguments
        continous_subsets_multi = torch.tensor_split(
            continuous_x, self.continuous_split_indices, dim=-1
        )
        if self.num_distribution_args > 1:
            continous_subsets = [
                tuple(torch.chunk(subset, self.num_distribution_args, dim=-1))
                for subset in continous_subsets_multi
            ]
            return discrete_subsets, continous_subsets

        if isinstance(self, SplitInput):
            return discrete_subsets, list(continous_subsets_multi)

        return discrete_subsets, [(sub,) for sub in continous_subsets_multi]


class SplitInput(SplitOutput):
    """Alias of `SplitOutput`."""

    def __init__(
        self,
        discrete_dataset_shapes: list[tuple[int, ...]],
        continuous_dataset_shapes: list[int],
    ) -> None:
        super().__init__(discrete_dataset_shapes, continuous_dataset_shapes, None)

    def __call__(self, *args: Any, **kwds: Any) -> tuple[DiscreteData, ContinuousData]:
        return cast(
            tuple[DiscreteData, ContinuousData], super().__call__(*args, **kwds)
        )
