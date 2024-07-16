__all__ = ["split_samples"]

import torch


def split_samples(
    num_samples: int,
    train_frac: float = 0.9,
    test_frac: float = 0.1,
    valid_frac: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Randomly split samples into training, test, and validation sets.

    Args:
        num_samples: Number of samples to split.
        train_frac: Fraction of samples corresponding to training set.
        test_frac: Fraction of samples corresponding to test set.
        valid_frac: Fraction of samples corresponding to validation set.

    Returns:
        Tuple containing indices corresponding to each subset.
    """
    if (train_frac + test_frac + valid_frac) != 1.0:
        raise ValueError("The sum of the subset fractions must be equal to one.")

    train_size = int(train_frac * num_samples)
    test_size = int(test_frac * num_samples)

    perm = torch.randperm(num_samples)

    tup = tuple(torch.tensor_split(perm, (train_size, train_size + test_size)))
    assert len(tup) == 3
    return tup
