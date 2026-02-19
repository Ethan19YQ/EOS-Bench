# -*- coding: utf-8 -*-
"""
rl_utils.py
强化学习调度（PPO）辅助工具 / Reinforcement Learning Scheduling (PPO) Auxiliary Tools

本模块功能 / Module Functionality
----------
1) 自动扫描 output/ 下的场景 JSON（训练用）；
   / Automatically scan scenario JSONs under output/ (for training);
2) 统一处理文件名（允许只传 stem 不带 .json）；
   / Uniformly process filenames (allow passing stem without .json);
3) 提供模型保存/加载路径规范。
   / Provide standard paths for model saving/loading.

说明 / Description
----
- 训练时：从 output/ 递归找到场景 JSON，但排除 output/schedules/ 目录（避免把调度结果当训练数据）
  / Training: Recursively find scenario JSONs from output/, but exclude the output/schedules/ directory (to avoid treating scheduling results as training data)
- 测试时：需要用户指定一个场景 JSON 文件名（stem 或路径）
  / Testing: Requires the user to specify a scenario JSON filename (stem or path)
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional
import re


def normalize_json_path(input_name: str, default_dir: Path) -> Path:
    """
    将输入的 json 名称标准化为 Path。
    / Normalize the input JSON name to a Path object.

    - 支持用户输入：不带 .json 的 stem
      / Support user input: stem without .json
    - 支持用户输入：相对路径/绝对路径
      / Support user input: relative/absolute paths
    """
    p = Path(input_name)
    if p.suffix.lower() != ".json":
        p = Path(str(p) + ".json")

    if not p.is_absolute():
        p = default_dir / p

    return p.resolve()


def scan_scenario_jsons(output_dir: Path) -> List[Path]:
    """
    扫描 output_dir 下所有“场景 JSON”（训练用），排除 schedules 子目录。
    / Scan all "scenario JSONs" under output_dir (for training), excluding the schedules subdirectory.

    推荐场景 JSON 命名一般为：
    / Recommended scenario JSON naming is generally:
      Scenario_*.json
    但这里不强依赖命名，主要排除 schedules 目录和明显的 *_schedule.json / scheduler_*.json。
    / But here we do not strictly rely on naming, mainly excluding the schedules directory and obvious *_schedule.json / scheduler_*.json files.
    """
    output_dir = output_dir.resolve()
    schedules_dir = (output_dir / "schedules").resolve()

    results: List[Path] = []
    for p in output_dir.rglob("*.json"):
        try:
            rp = p.resolve()
        except Exception:
            continue

        # 排除 schedules 目录 / Exclude the schedules directory
        if schedules_dir in rp.parents:
            continue

        # 排除已输出的调度结果（尽量避免混入训练）
        # / Exclude output scheduling results (to avoid mixing them into training)
        name = rp.name.lower()
        if "schedule" in name or name.startswith("scheduler_"):
            continue

        results.append(rp)

    results.sort()
    return results


def default_model_path(base_dir: Path, algo_name: str = "ppo") -> Path:
    """
    统一模型保存路径：output/models/ppo_model.pt
    / Unified model save path: output/models/ppo_model.pt
    """
    model_dir = base_dir / "output" / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    return (model_dir / f"{algo_name}_model.pt").resolve()