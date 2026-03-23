# -*- coding: utf-8 -*-
"""schedulers/timeliness_utils.py

Main functionality:
This module provides timeliness evaluation utilities for scheduling results.
It computes the timeliness metric based on task start times relative to the
scenario horizon, and converts it into a timeliness score in the range [0, 1].
"""

from __future__ import annotations

from typing import Dict
from schedulers.scenario_loader import SchedulingProblem
from schedulers.constraint_model import Schedule

def timeliness_metric(problem: SchedulingProblem, schedule: Schedule) -> float:
    total_tasks = len(problem.tasks)
    if total_tasks <= 0:
        return 0.0

    ts = problem.start_time
    T = float((problem.end_time - problem.start_time).total_seconds())
    if T <= 0:
        return 0.0

    assigned_index: Dict[str, object] = {a.task_id: a for a in schedule.assignments}

    numerator = 0.0
    for tid in problem.tasks.keys():
        a = assigned_index.get(tid)
        if a is not None:
            dt = float((a.sat_start_time - ts).total_seconds())
            # Clamp the value so that, in extreme cases where the time is slightly
            # earlier than start_time, dt < 0 does not affect the metric
            if dt < 0:
                dt = 0.0
            numerator += dt
        else:
            numerator += T

    return numerator / (T * float(total_tasks))

def timeliness_score(problem: SchedulingProblem, schedule: Schedule) -> float:
    tm = timeliness_metric(problem, schedule)
    score = 1.0 - tm
    if score < 0.0:
        return 0.0
    if score > 1.0:
        return 1.0
    return score