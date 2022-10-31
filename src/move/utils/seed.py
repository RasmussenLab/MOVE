__all__ = ["set_global_seed"]

import random
from typing import Optional

import numpy as np
import torch
import os


def set_global_seed(seed: Optional[int]) -> None:
    """
    Sets global seed

    Args:
        seed (Optional[int]): integer for seed
    """    
    if seed is None:
        return
    random.seed(seed) #
    np.random.seed(seed) #
    os.environ['PYTHONASSEED'] = str(seed)
    torch.manual_seed(seed) #
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)