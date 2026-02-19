# -*- coding: utf-8 -*-
"""
evaluation_metrics.py
调度方案评价指标计算模块 / Scheduling scheme evaluation metrics calculation module

本文件在整个项目中的角色 / Role of this file in the entire project
--------------------------------
1. 根据调度结果 Schedule 与原始场景 SchedulingProblem，
   计算多种评价指标，用于不同调度算法的公平对比：
   / Based on the scheduling result Schedule and the original scenario SchedulingProblem,
   calculate various evaluation metrics for fair comparison among different scheduling algorithms:

   - TP  : Task Profit（任务收益 / Task Profit）
   - TCR : Task Completion Rate（任务完成率 / Task Completion Rate）
   - BD  : Balance Degree（负载均衡度 / Balance Degree）
   - TM  : Timeliness Metric（时效性指标 / Timeliness Metric）
   - RT  : Runtime Efficiency（运行时间 / Runtime Efficiency）
   - RV  : Robustness Variance（鲁棒性方差，可选，多次运行时计算 / Robustness Variance, optional, calculated over multiple runs）

2. 该模块不依赖具体算法，仅依赖：
   / This module does not depend on specific algorithms, it only depends on:
   - schedulers.scenario_loader.SchedulingProblem
   - schedulers.constraint_model.Schedule
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from .scenario_loader import SchedulingProblem
from .constraint_model import Schedule


@dataclass
class EvaluationMetrics:
    """
    调度指标结果数据结构
    / Data structure for scheduling metric results
    """
    task_profit: float                 # TP (任务收益 / Task Profit)
    task_completion_rate: float        # TCR (任务完成率 / Task Completion Rate)
    balance_degree: float              # BD (负载均衡度 / Balance Degree)
    timeliness_metric: float           # TM (时效性指标 / Timeliness Metric)
    runtime_efficiency: float          # RT (秒 / seconds)
    robustness_variance: Optional[float] = None  # RV，可选 / RV, optional

    # 求解器 gap（相对 gap，若可获得；主要用于 MIP/CBC）。
    # / Solver gap (relative gap, if available; mainly used for MIP/CBC).
    mip_gap: Optional[float] = None

    @property
    def runtime_sec(self) -> float:
        """兼容旧字段名：等同于 runtime_efficiency（秒）。
           / Compatible with old field name: equivalent to runtime_efficiency (seconds)."""
        return self.runtime_efficiency



def _compute_task_profit(problem: SchedulingProblem, schedule: Schedule) -> float:
    """
    TP = sum(p_i * x_i)
    p_i : 任务 i 的 profit（这里用任务 priority） / profit of task i (here we use task priority)
    x_i : 若任务被调度则为 1，否则为 0 / 1 if task is scheduled, 0 otherwise
    """
    assigned_task_ids = set(schedule.assigned_task_ids)
    tp = 0.0
    for tid, task in problem.tasks.items():
        if tid in assigned_task_ids:
            tp += float(task.priority)
    return tp


def _compute_task_completion_rate(problem: SchedulingProblem, schedule: Schedule) -> float:
    """
    TCR = (已完成任务数) / (总任务数)
    / TCR = (Number of completed tasks) / (Total number of tasks)
    """
    total_tasks = len(problem.tasks)
    if total_tasks == 0:
        return 0.0
    assigned_task_ids = set(schedule.assigned_task_ids)
    return len(assigned_task_ids) / float(total_tasks)


def _compute_balance_degree(problem: SchedulingProblem, schedule: Schedule) -> float:
    """BD（负载均衡度）：与目标函数的 BalanceScore 完全同口径（按工作时长）。
       / BD (Balance Degree): Completely consistent with the BalanceScore of the objective function (based on working hours)."""
    from .balance_utils import balance_degree
    return balance_degree(problem, schedule)


def _compute_timeliness_metric(problem: SchedulingProblem, schedule: Schedule) -> float:
    """统一口径：调用 timeliness_utils.timeliness_metric。
       / Unified standard: call timeliness_utils.timeliness_metric."""
    from schedulers.timeliness_utils import timeliness_metric
    return timeliness_metric(problem, schedule)



def _compute_robustness_variance(tp_samples: List[float]) -> Optional[float]:
    """
    RV = Var(TP_1, ..., TP_k) = 1/(k-1) * sum((TP_i - TP_mean)^2)
    仅当样本数 k >= 2 时返回值，否则返回 None。
    / Returns a value only when the number of samples k >= 2, otherwise returns None.
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
    综合计算六个指标（RV 在传入多次 TP 样本时才计算）。
    / Comprehensively calculate the six metrics (RV is calculated only when multiple TP samples are passed in).
    """
    tp = _compute_task_profit(problem, schedule)
    tcr = _compute_task_completion_rate(problem, schedule)
    bd = _compute_balance_degree(problem, schedule)
    tm = _compute_timeliness_metric(problem, schedule)
    rv = _compute_robustness_variance(robustness_tp_samples) if robustness_tp_samples else None

    # 若未显式传入 gap，尝试从 schedule.metadata 读取（MIP 会写入）。
    # / If gap is not explicitly passed in, try to read from schedule.metadata (MIP will write to it).
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