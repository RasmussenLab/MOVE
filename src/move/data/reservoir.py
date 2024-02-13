__all__ = ["Reservoir", "PairedReservoir"]

import math
import random

import torch

from move.core.exceptions import UnsetProperty


class ReservoirTest:
    """Generate a random sample of k items from a stream of n items, where
    n is either unknown or very large.

    This implementation uses the so-called Algorithm R."""

    # based on: https://richardstartin.github.io/posts/reservoir-sampling

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.reservoir = torch.empty(capacity)
        self.idx = 0

    def add(self, value: torch.Tensor) -> None:
        if self.idx < self.capacity:
            self.reservoir[self.idx] = value
        else:
            repl_idx = math.floor(random.random() * self.idx)
            if repl_idx < self.capacity:
                self.reservoir[repl_idx] = value
        self.idx += 1


class Reservoir:
    """Generate a random sample (reservoir) from a stream of `n` items, where
    `n` is either unknown or very large.

    This implementation uses the so-called Algorithm R.

    Args:
        capacity: Number of items in the reservoir
    """

    def __init__(self, capacity: int):
        self.capacity = capacity
        self._reservoir = None
        self.idx = 0
        self.total_samples = 0

    def __call__(self) -> torch.Tensor:
        return self.reservoir

    @property
    def reservoir(self) -> torch.Tensor:
        if self._reservoir is None:
            raise UnsetProperty("Reservoir")
        return self._reservoir

    def add(self, stream: torch.Tensor):
        """Select a random sample from stream and add it to the reservoir.

        Args:
            stream: tensor with samples in its first dimension
        """

        num_samples = stream.size(0)
        self.total_samples += num_samples

        # Init reservoir
        if self._reservoir is None:
            self._reservoir = torch.empty((self.capacity, *stream.shape[1:]))
        elif self._reservoir.shape[1:] != stream.shape[1:]:
            raise ValueError(f"Shape mismatch between reservoir and stream")

        # Fill empty reservoir
        if self.idx < self.capacity:
            stop = min(self.capacity - self.idx, num_samples)
            self._reservoir[self.idx : self.idx + stop] = stream[:stop]
            self.idx += stop
            if stop == num_samples:
                return

            stream = stream[stop:]
            num_samples -= stop

        # Sample and fill reservoir
        i = torch.arange(self.idx, self.total_samples)
        j = torch.floor(torch.rand(i.shape) * i).long()
        replace = j < self.capacity

        for a, b in zip(j[replace], i[replace] - self.idx):
            self._reservoir[a] = stream[b]

        self.idx += num_samples


class PairedReservoir(Reservoir):
    """Genereate a paired set of random samples (reservoirs) from a paired set
    of streams, whose size is either unknown or very large."""

    def __init__(self, capacity: int):
        super().__init__(capacity)

    def __call__(self) -> tuple[torch.Tensor, ...]:
        return self.reservoir

    @property
    def reservoir(self) -> tuple[torch.Tensor, ...]:
        if self._reservoir is None:
            raise UnsetProperty("Reservoir")
        return self._reservoir

    def add(self, *streams: torch.Tensor):
        """Select a random sample from a paired set of streams and add it to
        the corresponding reservoir.

        Args:
            streams: Tensors with samples in its first dimension
        """
        streams_ = list(streams)
        stream1 = streams_[0]
        num_samples = stream1.size(0)

        if not all(stream.size(0) == num_samples for stream in streams_[1:]):
            raise ValueError("Streams must have the same number of samples")

        self.total_samples += num_samples

        # Init reservoir
        if self._reservoir is None:
            self._reservoir = tuple(
                [
                    torch.empty((self.capacity, *stream1.shape[1:]))
                    for _ in range(len(streams_))
                ]
            )
        elif len(streams_) != len(self._reservoir):
            raise ValueError("Size mismatch between number of streams and reservoirs")

        # Fill empty reservoir
        if self.idx < self.capacity:
            stop = min(self.capacity - self.idx, num_samples)

            for i, reservoir in enumerate(self._reservoir):
                stream = streams_[i]
                reservoir[self.idx : self.idx + stop] = stream[:stop]
                streams_[i] = stream[stop:]

            self.idx += stop

            if stop == num_samples:
                return

            num_samples -= stop

        # Sample and fill reservoir
        i = torch.arange(self.idx, self.total_samples)
        j = torch.floor(torch.rand(i.shape) * i).long()
        replace = j < self.capacity

        for m, n in zip(j[replace], i[replace] - self.idx):
            for reservoir, stream in zip(self._reservoir, streams_):
                reservoir[m] = stream[n]

        self.idx += num_samples
