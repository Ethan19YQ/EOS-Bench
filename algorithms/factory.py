# -*- coding: utf-8 -*-
"""
algorithms/factory.py

Main functionality:
This module provides a unified factory function for creating scheduling
algorithm instances, including MIP, heuristic methods, meta-heuristic methods,
and PPO-based reinforcement learning schedulers. It also supports safe
configuration overriding and configuration export.
"""

from __future__ import annotations

from dataclasses import asdict
from typing import Any, Dict, Optional, Tuple

from algorithms.objectives import ObjectiveWeights

# MIP (Mixed Integer Programming)
from algorithms.mip import MIPScheduler, MIPConfig

# Heuristics
from algorithms.heuristics import (
    HeuristicCompletionFirstScheduler,
    HeuristicProfitFirstScheduler,
    HeuristicBalanceFirstScheduler,
    HeuristicTimelinessFirstScheduler,
    HeuristicConfig,
)

# Meta-heuristic algorithms
from algorithms.meta_sa import SimulatedAnnealingScheduler, SAConfig
from algorithms.meta_ga import GeneticAlgorithmScheduler, GAConfig
from algorithms.meta_aco import AntColonyScheduler, ACOConfig

# PPO (Reinforcement Learning)


def create_algorithm(
    algo_name: str,
    objective_weights: Optional[ObjectiveWeights] = None,
    cfg_overrides: Optional[Dict[str, Any]] = None,
):
    """
    Create a scheduling algorithm instance.
    """
    name = (algo_name or "").lower().strip()
    ow = objective_weights or ObjectiveWeights(1.0, 0.0, 0.0, 0.0)
    o = cfg_overrides or {}

    # ---------------- MIP ----------------
    if name in ("mip", "milp"):
        cfg = MIPConfig(weights=ow)
        cfg = _apply_overrides(cfg, o)
        return MIPScheduler(cfg=cfg)

    # ------------- Heuristics -------------
    # Completion rate first
    if name in ("completion_first", "completion", "tcr"):
        cfg = _apply_overrides(HeuristicConfig(), o)
        return HeuristicCompletionFirstScheduler(cfg=cfg)

    # Profit first
    if name in ("profit_first", "profit", "tp"):
        cfg = _apply_overrides(HeuristicConfig(), o)
        return HeuristicProfitFirstScheduler(cfg=cfg)

    # Timeliness first
    if name in ("timeliness_first", "timeliness", "tm"):
        cfg = _apply_overrides(HeuristicConfig(), o)
        return HeuristicTimelinessFirstScheduler(cfg=cfg)

    # Balance first
    if name in ("balance_first", "balance", "bd"):
        cfg = _apply_overrides(HeuristicConfig(), o)
        return HeuristicBalanceFirstScheduler(cfg=cfg)

    # ------------ Meta-heuristics ---------
    # Simulated Annealing
    if name in ("sa", "simulated_annealing"):
        # Note: seed defaults to None, which means different results for each run
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

    # Genetic Algorithm
    if name in ("ga", "genetic", "genetic_algorithm"):
        cfg = GAConfig(weights=ow)
        cfg = _apply_overrides(cfg, o)
        return GeneticAlgorithmScheduler(cfg=cfg)

    # Ant Colony Optimization
    if name in ("aco", "ant", "ant_colony"):
        cfg = ACOConfig(weights=ow)
        cfg = _apply_overrides(cfg, o)
        return AntColonyScheduler(cfg=cfg)

    # ---------------- PPO ----------------
    if name in ("ppo",):
        try:
            # Lazy import to reduce unnecessary dependency loading
            from algorithms.ppo.learning import PPOLearningScheduler, PPORunConfig
        except Exception as e:
            raise ImportError(
                "PPO module or dependencies are missing: cannot import algorithms.ppo.*.\n"
                "Please ensure the algorithms/ppo directory exists and dependencies such as torch are installed."
            ) from e
        run_cfg = _apply_overrides(PPORunConfig(), o)
        model_path = o.get("model_path") or o.get("rl_model_path") or None
        return PPOLearningScheduler(model_path=model_path, run_cfg=run_cfg)

    raise ValueError(
        f"Unknown algorithm: {algo_name}. "
        f"Supported: mip, completion_first, profit_first, balance_first, sa, ga, aco, ppo"
    )


def _apply_overrides(cfg_obj, overrides: Dict[str, Any]):
    """
    Write fields with matching names from overrides into the dataclass,
    while safely ignoring unknown keys.
    """
    if not overrides:
        return cfg_obj
    for k, v in overrides.items():
        if hasattr(cfg_obj, k):
            setattr(cfg_obj, k, v)
    return cfg_obj


def config_to_dict(cfg_obj) -> Dict[str, Any]:
    """
    Convert a configuration object to a dictionary.
    """
    try:
        return asdict(cfg_obj)
    except Exception:
        return dict(cfg_obj.__dict__)