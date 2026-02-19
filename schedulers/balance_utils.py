# -*- coding: utf-8 -*-
"""schedulers/balance_utils.py
说明 / Description
----
计算多星任务负载均衡度的工具函数。
Utility functions for calculating the workload balance degree across multiple satellites.
"""

from __future__ import annotations

from typing import Dict

from .scenario_loader import SchedulingProblem
from .constraint_model import Schedule


def total_required_work(problem: SchedulingProblem) -> float:
    """全任务总需求时长（秒），常数。
       / Total required duration for all tasks (in seconds), constant."""
    return float(sum(float(t.required_duration) for t in problem.tasks.values()))


def compute_sat_workloads(problem: SchedulingProblem, schedule: Schedule) -> Dict[str, float]:
    """统计每颗卫星的已安排工作时长（秒）。
       / Calculate the scheduled work duration (in seconds) for each satellite."""
    workloads: Dict[str, float] = {sid: 0.0 for sid in problem.satellites.keys()}
    for a in schedule.assignments:
        dt = (a.sat_end_time - a.sat_start_time).total_seconds()
        workloads[a.satellite_id] = workloads.get(a.satellite_id, 0.0) + float(dt)
    return workloads


def balance_degree_from_workloads(problem: SchedulingProblem, workloads: Dict[str, float]) -> float:
    """从 workloads 直接计算 BD（负载均衡度）。
       / Calculate Balance Degree (BD) directly from workloads."""
    sat_ids = list(problem.satellites.keys())
    n = len(sat_ids)
    if n <= 0:
        return 0.0

    denom = 2.0 * total_required_work(problem)
    if denom <= 0:
        return 0.0

    total_load = sum(float(workloads.get(sid, 0.0)) for sid in sat_ids)
    mu = total_load / float(n)

    sum_abs = 0.0
    for sid in sat_ids:
        sum_abs += abs(float(workloads.get(sid, 0.0)) - mu)

    bd = 1.0 - (sum_abs / denom)
    if bd < 0.0:
        return 0.0
    if bd > 1.0:
        return 1.0
    return bd


def balance_degree(problem: SchedulingProblem, schedule: Schedule) -> float:
    """从 schedule 计算 BD（按工作时长）。
       / Calculate Balance Degree (BD) from the schedule (based on work duration)."""
    workloads = compute_sat_workloads(problem, schedule)
    return balance_degree_from_workloads(problem, workloads)