__all__ = ["Chunk", "SplitInput", "SplitOutput"]

import itertools
import operator
from typing import TYPE_CHECKING, Any, Optional, Type, TypeVar, Union, cast

import torch
from torch import nn
from torch.distributions import Distribution, constraints

if TYPE_CHECKING:
    from move.data.dataset import MoveDataset

DiscreteData = list[torch.Tensor]
ContinuousData = list[torch.Tensor]
ContinuousDistribution = list[dict[str, torch.Tensor]]
SplitData = Union[
    tuple[DiscreteData, ContinuousData], tuple[DiscreteData, ContinuousDistribution]
]

SUPPORTED_DISTRIBUTIONS = [
    "Normal",
    "StudentT",
    # other distributions possible but untested
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
        distribution_name_or_cls:
            If given, continuous variables will be treated as distribution
            arguments. For instance, for a normal distribution, the continuous
            subset of the output will be split into mean and standard deviation.
            This can be either the name of a class from the `torch.distributions`
            module or a class that can be instantiated.
    """

    num_discrete_features: int
    num_continuous_features: int
    num_features: int
    num_expected_features: int
    num_distribution_args: int
    discrete_dataset_shapes: list[tuple[int, int]]
    discrete_dataset_shapes_1d: list[int]
    continuous_dataset_shapes: list[int]
    discrete_split_indices: list[int]
    continuous_split_indices: list[int]
    discrete_activation: Optional[nn.Module]
    continuous_activation: Optional[nn.Module]

    def __init__(
        self,
        discrete_dataset_shapes: list[tuple[int, int]],
        continuous_dataset_shapes: list[int],
        distribution_name_or_cls: Union[str, Type[Distribution], None] = None,
        discrete_activation_name: Optional[str] = None,
        continuous_activation_name: Optional[str] = None,
    ) -> None:
        super().__init__()

        self.distribution: Optional[Type[Distribution]] = None
        self.num_distribution_args = 1
        if distribution_name_or_cls is not None:
            if isinstance(distribution_name_or_cls, str):
                if distribution_name_or_cls not in SUPPORTED_DISTRIBUTIONS:
                    raise ValueError("Unsupported distribution")
                self.distribution = getattr(
                    torch.distributions, distribution_name_or_cls, None
                )
            else:
                if not issubclass(distribution_name_or_cls, Distribution):
                    raise ValueError("Not a distribution")
                self.distribution = distribution_name_or_cls
        if self.distribution is not None:
            self.num_distribution_args = len(self.distribution.arg_constraints)  # type: ignore

        activation_funs = []
        for name in [discrete_activation_name, continuous_activation_name]:
            if name is not None:
                activation_fun = getattr(nn, name)
                assert issubclass(activation_fun, nn.Module)
                activation_funs.append(activation_fun())
            else:
                activation_funs.append(None)
        self.discrete_activation, self.continuous_activation = activation_funs

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

    def __call__(self, *args: Any, **kwds: Any) -> SplitData:
        return super().__call__(*args, **kwds)

    @classmethod
    def from_move_dataset(cls: Type[T], move_dataset: "MoveDataset") -> T:
        """Create layer from shapes of discrete and continuous datasets contained in a
        MOVE dataset."""
        discrete_dataset_shapes = []
        continuous_dataset_shapes = []
        for dataset in move_dataset.discrete_datasets:
            discrete_dataset_shapes.append(dataset.original_shape)
        for dataset in move_dataset.continuous_datasets:
            continuous_dataset_shapes.append(dataset.num_features)
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
        if self.discrete_activation is not None:
            discrete_x = self.discrete_activation(discrete_x)
        if self.continuous_activation is not None:
            continuous_x = self.continuous_activation(continuous_x)

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
        continous_subsets = list(
            torch.tensor_split(continuous_x, self.continuous_split_indices, dim=-1)
        )
        if self.num_distribution_args > 1:
            if self.distribution is not None:
                continous_distributions = []
                # For each distribution, split into correct # arguments
                # Example: if modeling a Normal distribution, split into loc and scale
                # Chunks are saved in dictionary
                # If distribution arg constrained positive (e.g. scale), apply transform
                for subset in continous_subsets:
                    chunks = torch.chunk(subset, self.num_distribution_args, dim=-1)
                    args = {}
                    for arg, (arg_name, arg_constraint) in zip(
                        chunks, self.distribution.arg_constraints.items()  # type: ignore
                    ):
                        if arg_constraint is constraints.positive:
                            arg = torch.exp(arg * 0.5)
                        args[arg_name] = arg
                    continous_distributions.append(args)
            return discrete_subsets, continous_distributions

        if isinstance(self, SplitInput):
            return discrete_subsets, list(continous_subsets)

        return discrete_subsets, continous_subsets


class SplitInput(SplitOutput):
    """Alias of `SplitOutput`."""

    def __init__(
        self,
        discrete_dataset_shapes: list[tuple[int, int]],
        continuous_dataset_shapes: list[int],
    ) -> None:
        super().__init__(discrete_dataset_shapes, continuous_dataset_shapes, None)

    def __call__(self, *args: Any, **kwds: Any) -> tuple[DiscreteData, ContinuousData]:
        return cast(
            tuple[DiscreteData, ContinuousData], super().__call__(*args, **kwds)
        )
