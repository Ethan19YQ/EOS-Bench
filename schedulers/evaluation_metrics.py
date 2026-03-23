# -*- coding: utf-8 -*-
"""
evaluation_metrics.py

Main functionality:
This module computes the main evaluation metrics for scheduling results,
including task profit, task completion rate, balance degree, timeliness,
runtime, optional robustness variance, and optional MIP solver gap.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from .scenario_loader import SchedulingProblem
from .constraint_model import Schedule


@dataclass
class EvaluationMetrics:
    """
    Data structure for scheduling metric results.
    """
    task_profit: float                 # TP (Task Profit)
    task_completion_rate: float        # TCR (Task Completion Rate)
    balance_degree: float              # BD (Balance Degree)
    timeliness_metric: float           # TM (Timeliness Metric)
    runtime_efficiency: float          # RT (seconds)
    robustness_variance: Optional[float] = None  # RV, optional

    # Solver gap as a relative gap, if available, mainly for MIP/CBC
    mip_gap: Optional[float] = None

    @property
    def runtime_sec(self) -> float:
        """Compatible with the old field name, equivalent to runtime_efficiency in seconds."""
        return self.runtime_efficiency


def _compute_task_profit(problem: SchedulingProblem, schedule: Schedule) -> float:
    """
    TP = sum(p_i * x_i)

    p_i: profit of task i, here using task priority
    x_i: 1 if task i is scheduled, otherwise 0
    """
    assigned_task_ids = set(schedule.assigned_task_ids)
    tp = 0.0
    for tid, task in problem.tasks.items():
        if tid in assigned_task_ids:
            tp += float(task.priority)
    return tp


def _compute_task_completion_rate(problem: SchedulingProblem, schedule: Schedule) -> float:
    """
    TCR = number of completed tasks / total number of tasks
    """
    total_tasks = len(problem.tasks)
    if total_tasks == 0:
        return 0.0
    assigned_task_ids = set(schedule.assigned_task_ids)
    return len(assigned_task_ids) / float(total_tasks)


def _compute_balance_degree(problem: SchedulingProblem, schedule: Schedule) -> float:
    """BD (Balance Degree), fully consistent with the BalanceScore in the objective function, based on workload duration."""
    from .balance_utils import balance_degree
    return balance_degree(problem, schedule)


def _compute_timeliness_metric(problem: SchedulingProblem, schedule: Schedule) -> float:
    """Unified definition: call timeliness_utils.timeliness_metric."""
    from schedulers.timeliness_utils import timeliness_metric
    return timeliness_metric(problem, schedule)


def _compute_robustness_variance(tp_samples: List[float]) -> Optional[float]:
    """
    RV = Var(TP_1, ..., TP_k) = 1/(k-1) * sum((TP_i - TP_mean)^2)

    Returns a value only when the number of samples k >= 2,
    otherwise returns None.
    """
    k = len(tp_samples)
    if k < 2:
        return None

    mean_tp = sum(tp_samples) / float(k)
    var = sum((x - mean_tp) ** 2 for x in tp_samples) / float(k - 1)
    return var


def compute_evaluation_metrics(
    problem: SchedulingProblem,
    schedule: Schedule,
    runtime_seconds: float,
    robustness_tp_samples: Optional[List[float]] = None,
    mip_gap: Optional[float] = None,
) -> EvaluationMetrics:
    """
    Compute all six metrics together.

    RV is calculated only when multiple TP samples are provided.
    """
    tp = _compute_task_profit(problem, schedule)
    tcr = _compute_task_completion_rate(problem, schedule)
    bd = _compute_balance_degree(problem, schedule)
    tm = _compute_timeliness_metric(problem, schedule)
    rv = _compute_robustness_variance(robustness_tp_samples) if robustness_tp_samples else None

    # If gap is not explicitly passed in, try reading it from schedule.metadata
    # MIP writes it there when available
    if mip_gap is None:
        try:
            mip_gap = schedule.metadata.get("mip_gap")
        except Exception:
            mip_gap = None

    return EvaluationMetrics(
        task_profit=tp,
        task_completion_rate=tcr,
        balance_degree=bd,
        timeliness_metric=tm,
        runtime_efficiency=float(runtime_seconds),
        robustness_variance=rv,
        mip_gap=mip_gap,
    )