# -*- coding: utf-8 -*-
"""
algorithms/random_utils.py
统一随机数工具

目标
----
- 默认“每次运行不同”：seed=None 时使用 time_ns + SystemRandom 混合生成种子；
- 允许可复现：显式传入 seed=int；
- 提供一个可选的全局种子设置（numpy/torch），便于 RL 训练可复现。
"""

from __future__ import annotations

import os
import time
import random
from typing import Optional, Tuple

def resolve_seed(seed: Optional[int]) -> int:
    """None => 自动生成一个 31bit 正整数种子；int => 原样返回。"""
    if seed is None:
        return (time.time_ns() ^ random.SystemRandom().randrange(1 << 30)) & 0x7FFFFFFF
    return int(seed) & 0x7FFFFFFF

def make_rng(seed: Optional[int]) -> random.Random:
    """创建一个独立 RNG（推荐用于算法内部）。"""
    return random.Random(resolve_seed(seed))

def spawn_seeds(base_seed: Optional[int], n: int) -> Tuple[int, ...]:
    """从一个 base_seed 派生 n 个子种子（base_seed=None 也可）。"""
    rng = make_rng(base_seed)
    return tuple(rng.randrange(1 << 31) for _ in range(max(0, int(n))))

def set_global_seeds(seed: Optional[int]) -> int:
    """设置 python/numpy/torch 的全局随机种子（若相关库可用）。返回最终使用的 seed。"""
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
