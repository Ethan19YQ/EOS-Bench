# -*- coding: utf-8 -*-
"""
rl_utils.py

Main functionality:
This module provides utility functions for reinforcement learning workflows,
including JSON path normalization, scenario JSON scanning for training, and
default model-path generation.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional
import re


def normalize_json_path(input_name: str, default_dir: Path) -> Path:
    """
    Normalize the input JSON name to a Path object.

    - Supports user input as a stem without .json
    - Supports user input as a relative or absolute path
    """
    p = Path(input_name)
    if p.suffix.lower() != ".json":
        p = Path(str(p) + ".json")

    if not p.is_absolute():
        p = default_dir / p

    return p.resolve()


def scan_scenario_jsons(output_dir: Path) -> List[Path]:
    """
    Scan all scenario JSON files under output_dir for training,
    excluding the schedules subdirectory.

    Recommended scenario JSON naming is generally:
      Scenario_*.json

    However, this function does not strictly depend on naming.
    It mainly excludes the schedules directory and obvious
    *_schedule.json / scheduler_*.json files.
    """
    output_dir = output_dir.resolve()
    schedules_dir = (output_dir / "schedules").resolve()

    results: List[Path] = []
    for p in output_dir.rglob("*.json"):
        try:
            rp = p.resolve()
        except Exception:
            continue

        # Exclude the schedules directory
        if schedules_dir in rp.parents:
            continue

        # Exclude output scheduling result files to avoid mixing them into training
        name = rp.name.lower()
        if "schedule" in name or name.startswith("scheduler_"):
            continue

        results.append(rp)

    results.sort()
    return results


def default_model_path(base_dir: Path, algo_name: str = "ppo") -> Path:
    """
    Unified model save path: output/models/ppo_model.pt
    """
    model_dir = base_dir / "output" / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    return (model_dir / f"{algo_name}_model.pt").resolve()