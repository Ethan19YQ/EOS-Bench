# -*- coding: utf-8 -*-
"""
algorithms/random_utils.py

Main functionality:
This module provides random seed utilities for reproducible experiments.
It supports seed resolution, independent random generator creation, child-seed
derivation, and optional global seed initialization for Python, NumPy, and PyTorch.
"""

from __future__ import annotations

import os
import time
import random
from typing import Optional, Tuple

def resolve_seed(seed: Optional[int]) -> int:
    """If seed is None, automatically generate a 31-bit positive integer seed.
    If seed is an integer, return it directly after masking."""
    if seed is None:
        return (time.time_ns() ^ random.SystemRandom().randrange(1 << 30)) & 0x7FFFFFFF
    return int(seed) & 0x7FFFFFFF

def make_rng(seed: Optional[int]) -> random.Random:
    """Create an independent random number generator.
    This is recommended for internal algorithm use."""
    return random.Random(resolve_seed(seed))

def spawn_seeds(base_seed: Optional[int], n: int) -> Tuple[int, ...]:
    """Derive n child seeds from a base seed.
    base_seed can also be None."""
    rng = make_rng(base_seed)
    return tuple(rng.randrange(1 << 31) for _ in range(max(0, int(n))))

def set_global_seeds(seed: Optional[int]) -> int:
    """Set global random seeds for Python, NumPy, and PyTorch if available.
    Returns the final seed that is actually used."""
    s = resolve_seed(seed)
    random.seed(s)
    os.environ["PYTHONHASHSEED"] = str(s)

    try:
        import numpy as np  # type: ignore
        np.random.seed(s)
    except Exception:
        pass

    try:
        import torch  # type: ignore
        torch.manual_seed(s)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(s)
    except Exception:
        pass

    return s