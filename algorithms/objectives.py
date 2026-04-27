# algorithms/objectives.py
# -*- coding: utf-8 -*-
"""
Main functionality:
This module defines the objective-weight configuration and the objective
evaluation model for scheduling. It provides normalized multi-objective
weights and computes profit, completion, timeliness, balance, and overall
weighted scores for a schedule.
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
    """Objective weights configuration."""
    w_profit: float = 1.0
    w_completion: float = 0.0
    w_timeliness: float = 0.0
    w_balance: float = 0.0

    def normalized(self) -> "ObjectiveWeights":
        """
        Normalize the weights so that their sum equals 1.
        """
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
    """
    Objective evaluation model used to assess a schedule on different metrics
    and compute the weighted total score.
    """

    def __init__(self, problem: SchedulingProblem, weights: ObjectiveWeights) -> None:
        self.problem = problem
        self.weights = weights.normalized()

        self.total_tasks = max(1, len(problem.tasks))
        self.total_priority = float(sum(float(t.priority) for t in problem.tasks.values()))
        if self.total_priority <= 0:
            self.total_priority = 1.0

        self.sat_ids: List[str] = list(problem.satellites.keys())
        self.n_sats = max(1, len(self.sat_ids))

        # Constant: total required duration of all tasks, used for BD normalization
        self.total_required = max(1e-9, total_required_work(problem))

    def profit_score(self, schedule: Schedule) -> float:
        """Calculate the profit score."""
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
        """Calculate the completion score."""
        return float(len(set(schedule.assigned_task_ids))) / float(self.total_tasks)

    # ---- Balance based on workload BD ----

    def timeliness_metric(self, schedule: Schedule) -> float:
        """TM, where smaller is better. The definition is strictly consistent with evaluation_metrics.TM."""
        from schedulers.timeliness_utils import timeliness_metric
        return timeliness_metric(self.problem, schedule)

    def timeliness_score(self, schedule: Schedule) -> float:
        """TimelinessScore = 1 - TM, where larger is better, with range [0, 1]."""
        from schedulers.timeliness_utils import timeliness_score
        return timeliness_score(self.problem, schedule)

    def balance_score(self, schedule: Schedule) -> float:
        """
        Calculate the balance score.

        This follows the same definition as evaluation_metrics.BD.
        """
        return balance_degree(self.problem, schedule)

    def balance_score_from_workloads(self, workloads: Dict[str, float]) -> float:
        """Calculate the balance score based on workload distribution.

        workloads: {sat_id: workload_seconds}
        """
        return balance_degree_from_workloads(self.problem, workloads)

    # Compatible with the old interface, but not recommended:
    # treat counts as workloads measured in unit tasks
    def balance_score_from_counts(self, counts: Dict[str, int], T: int) -> float:
        if not self.sat_ids:
            return 0.0
        workloads = {sid: float(counts.get(sid, 0)) for sid in self.sat_ids}
        return balance_degree_from_workloads(self.problem, workloads)

    def score(self, schedule: Schedule) -> float:
        """
        Calculate the final weighted total score.
        """
        w = self.weights
        return (
                w.w_profit * self.profit_score(schedule)
                + w.w_completion * self.completion_score(schedule)
                + w.w_timeliness * self.timeliness_score(schedule)
                + w.w_balance * self.balance_score(schedule)
        )