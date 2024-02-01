import math

import torch


def hdi_bounds(x: torch.Tensor, hdi_prob: float = 0.95) -> torch.Tensor:
    """Return high density interval bounds (HDI) of a samples-features matrix.
    The HDI represents the range within which most of the samples are located.

    Args:
        x: Matrix (`num_samples` x `num_features`)
        hdi_prob: Percentage of samples inside the HDI

    Returns:
        Two-column tensor (`num_features` x 2)
    """
    # adapated from arviz

    if x.dim() != 2:
        raise ValueError("Can only calculate for matrices with two dimensions")

    n = x.size(0)
    x, _ = torch.sort(x, dim=0)

    interval_idx_inc = math.floor(hdi_prob * n)
    num_intervals = n - interval_idx_inc

    interval_width = x[interval_idx_inc:] - x[:num_intervals]
    min_idx = torch.argmin(interval_width, dim=0)

    hdi_min = torch.diag(x[min_idx])
    hdi_max = torch.diag(x[min_idx + interval_idx_inc])

    return torch.stack((hdi_min, hdi_max)).T
