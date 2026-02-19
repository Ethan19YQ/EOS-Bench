# -*- coding: utf-8 -*-
"""
algorithms/factory.py
统一算法创建入口（接口统一、模块化）

设计目标
--------
- 上层只关心 algo_name + config dict；
- 所有算法都遵循 BaseSchedulerAlgorithm.search(...) 接口（由 schedulers.constraint_model.BaseSchedulerAlgorithm 约定）；
- 元启发式默认 seed=None：每次运行自动随机，避免“每次都一样”；
- 支持 ObjectiveWeights 目标驱动。
"""

from __future__ import annotations

from dataclasses import asdict
from typing import Any, Dict, Optional, Tuple

from algorithms.objectives import ObjectiveWeights

# MIP
from algorithms.mip import MIPScheduler, MIPConfig

# heuristics
from algorithms.heuristics import (
    HeuristicCompletionFirstScheduler,
    HeuristicProfitFirstScheduler,
    HeuristicBalanceFirstScheduler,
    HeuristicTimelinessFirstScheduler,
    HeuristicConfig,
)

# meta
from algorithms.meta_sa import SimulatedAnnealingScheduler, SAConfig
from algorithms.meta_ga import GeneticAlgorithmScheduler, GAConfig
from algorithms.meta_aco import AntColonyScheduler, ACOConfig

# PPO


def create_algorithm(
    algo_name: str,
    objective_weights: Optional[ObjectiveWeights] = None,
    cfg_overrides: Optional[Dict[str, Any]] = None,
):
    """创建调度算法实例。"""
    name = (algo_name or "").lower().strip()
    ow = objective_weights or ObjectiveWeights(1.0, 0.0, 0.0, 0.0)
    o = cfg_overrides or {}

    # ---------------- MIP ----------------
    if name in ("mip", "milp"):
        cfg = MIPConfig(weights=ow)
        cfg = _apply_overrides(cfg, o)
        return MIPScheduler(cfg=cfg)

    # ------------- Heuristics -------------
    if name in ("completion_first", "completion", "tcr"):
        cfg = _apply_overrides(HeuristicConfig(), o)
        return HeuristicCompletionFirstScheduler(cfg=cfg)

    if name in ("profit_first", "profit", "tp"):
        cfg = _apply_overrides(HeuristicConfig(), o)
        return HeuristicProfitFirstScheduler(cfg=cfg)

    if name in ("timeliness_first", "timeliness", "tm"):
        cfg = _apply_overrides(HeuristicConfig(), o)
        return HeuristicTimelinessFirstScheduler(cfg=cfg)

    if name in ("balance_first", "balance", "bd"):
        cfg = _apply_overrides(HeuristicConfig(), o)
        return HeuristicBalanceFirstScheduler(cfg=cfg)

    # ------------ Meta-heuristics ---------
    if name in ("sa", "simulated_annealing"):
        # 注意：seed 默认 None => 每次运行不同
        cfg = SAConfig(weights=ow)
        cfg = _apply_overrides(cfg, o)
        return SimulatedAnnealingScheduler(
            max_iterations=cfg.max_iterations,
            initial_temperature=cfg.initial_temperature,
            cooling_rate=cfg.cooling_rate,
            neighbor_attempts=cfg.neighbor_attempts,
            restarts=cfg.restarts,
            seed=cfg.seed,
            weights=cfg.weights,
        )

    if name in ("ga", "genetic", "genetic_algorithm"):
        cfg = GAConfig(weights=ow)
        cfg = _apply_overrides(cfg, o)
        return GeneticAlgorithmScheduler(cfg=cfg)

    if name in ("aco", "ant", "ant_colony"):
        cfg = ACOConfig(weights=ow)
        cfg = _apply_overrides(cfg, o)
        return AntColonyScheduler(cfg=cfg)

    # ---------------- PPO ----------------
    if name in ("ppo",):
        try:
            from algorithms.ppo.learning import PPOLearningScheduler, PPORunConfig
        except Exception as e:
            raise ImportError(
                "PPO 模块或依赖缺失：无法导入 algorithms.ppo.* 。\n"
                "请确认工程中存在 algorithms/ppo 目录且已安装 torch 等依赖。"
            ) from e
        run_cfg = _apply_overrides(PPORunConfig(), o)
        model_path = o.get("model_path") or o.get("rl_model_path") or None
        return PPOLearningScheduler(model_path=model_path, run_cfg=run_cfg)

    raise ValueError(
        f"未知算法: {algo_name}. 支持: mip, completion_first, profit_first, balance_first, sa, ga, aco, ppo"
    )


def _apply_overrides(cfg_obj, overrides: Dict[str, Any]):
    """把 overrides 中同名字段写入 dataclass（安全忽略未知键）。"""
    if not overrides:
        return cfg_obj
    for k, v in overrides.items():
        if hasattr(cfg_obj, k):
            setattr(cfg_obj, k, v)
    return cfg_obj


def config_to_dict(cfg_obj) -> Dict[str, Any]:
    try:
        return asdict(cfg_obj)
    except Exception:
        return dict(cfg_obj.__dict__)