# algorithms/objectives.py
# -*- coding: utf-8 -*-
"""
统一目标（0~1 归一化）：

- ProfitScore     : sum(priority_assigned) / sum(priority_all)
- CompletionScore : assigned_tasks / total_tasks
- TimelinessScore: 1 - TM（与 evaluation_metrics.TM 完全一致）
- BalanceScore(BD): **按工作时长**的负载均衡度（与 evaluation_metrics.BD 完全一致）
                   BD = 1 - sum_s |workload_s - mu| / (2 * total_required)

其中：
- workload_s      : 卫星 s 已安排的观测段总时长（秒）
- mu             : sum(workload_s) / n_sats
- total_required : sum(task.required_duration)（全任务需求时长，常数）
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

from schedulers.scenario_loader import SchedulingProblem
from schedulers.constraint_model import Schedule
from schedulers.balance_utils import (
    total_required_work,
    compute_sat_workloads,
    balance_degree_from_workloads,
    balance_degree,
)


@dataclass(frozen=True)
class ObjectiveWeights:
    w_profit: float = 1.0
    w_completion: float = 0.0
    w_timeliness: float = 0.0
    w_balance: float = 0.0

    def normalized(self) -> "ObjectiveWeights":
        s = float(self.w_profit + self.w_completion + self.w_timeliness + self.w_balance)
        if s <= 0:
            return ObjectiveWeights(1.0, 0.0, 0.0, 0.0)
        return ObjectiveWeights(
            self.w_profit / s,
            self.w_completion / s,
            self.w_timeliness / s,
            self.w_balance / s,
        )
class ObjectiveModel:
    def __init__(self, problem: SchedulingProblem, weights: ObjectiveWeights) -> None:
        self.problem = problem
        self.weights = weights.normalized()

        self.total_tasks = max(1, len(problem.tasks))
        self.total_priority = float(sum(float(t.priority) for t in problem.tasks.values()))
        if self.total_priority <= 0:
            self.total_priority = 1.0

        self.sat_ids: List[str] = list(problem.satellites.keys())
        self.n_sats = max(1, len(self.sat_ids))

        # 常数：全任务需求总时长（用于 BD 归一化）
        self.total_required = max(1e-9, total_required_work(problem))

    def profit_score(self, schedule: Schedule) -> float:
        assigned = set(schedule.assigned_task_ids)
        if not assigned:
            return 0.0
        s = 0.0
        for tid in assigned:
            t = self.problem.tasks.get(tid)
            if t is not None:
                s += float(t.priority)
        return max(0.0, min(1.0, s / self.total_priority))

    def completion_score(self, schedule: Schedule) -> float:
        return float(len(set(schedule.assigned_task_ids))) / float(self.total_tasks)

    # ---- Balance (workload-based BD) ----

    
    def timeliness_metric(self, schedule: Schedule) -> float:
        """TM（越小越好）。口径与 evaluation_metrics.TM 完全一致。"""
        from schedulers.timeliness_utils import timeliness_metric
        return timeliness_metric(self.problem, schedule)

    def timeliness_score(self, schedule: Schedule) -> float:
        """TimelinessScore = 1 - TM（越大越好），范围 [0,1]。"""
        from schedulers.timeliness_utils import timeliness_score
        return timeliness_score(self.problem, schedule)
    def balance_score(self, schedule: Schedule) -> float:
        # 与 evaluation_metrics.BD 同口径
        return balance_degree(self.problem, schedule)

    def balance_score_from_workloads(self, workloads: Dict[str, float]) -> float:
        # workloads: {sat_id: workload_seconds}
        return balance_degree_from_workloads(self.problem, workloads)

    # 兼容旧接口（不建议再用）：把 counts 当作 workload=counts（单位任务）
    def balance_score_from_counts(self, counts: Dict[str, int], T: int) -> float:
        if not self.sat_ids:
            return 0.0
        workloads = {sid: float(counts.get(sid, 0)) for sid in self.sat_ids}
        return balance_degree_from_workloads(self.problem, workloads)

    def score(self, schedule: Schedule) -> float:
        w = self.weights
        return (
            w.w_profit * self.profit_score(schedule)
            + w.w_completion * self.completion_score(schedule)
            + w.w_timeliness * self.timeliness_score(schedule)
            + w.w_balance * self.balance_score(schedule)
        )
