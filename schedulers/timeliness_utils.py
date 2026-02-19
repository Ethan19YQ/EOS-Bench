# -*- coding: utf-8 -*-
"""schedulers/timeliness_utils.py

统一的时效性指标（Timeliness Metric, TM）计算工具。
/ Unified Timeliness Metric (TM) calculation tool.

目的 / Purpose：
- 让 evaluation_metrics.TM 与 algorithms.objectives 中的 TimelinessScore 使用同一套口径。
  / Ensure evaluation_metrics.TM and TimelinessScore in algorithms.objectives use the same calculation standard.
- TM 越小越好；TimelinessScore = 1 - TM（越大越好）。
  / The smaller the TM, the better; TimelinessScore = 1 - TM (the larger, the better).

TM 定义（与原 evaluation_metrics 中一致） / TM Definition (consistent with original evaluation_metrics)：
TM = [sum((t_i - t_s) * x_i) + sum(T * (1 - x_i))] / (T * N)

其中 / Where：
- t_i : 任务 i 的执行时间，这里采用“卫星观测开始时间 sat_start_time”
        / Execution time of task i, here using "satellite observation start time sat_start_time"
- t_s : 场景开始时间 problem.start_time / Scenario start time problem.start_time
- T   : 场景总时长 (problem.end_time - problem.start_time).seconds / Total scenario duration in seconds
- x_i : 若任务 i 被调度则为 1，否则为 0 / 1 if task i is scheduled, 0 otherwise
- N   : 总任务数 / Total number of tasks

说明 / Note：
- 未完成任务按照 T 计入惩罚（等价于 delay_norm=1）。
  / Unfinished tasks are penalized by T (equivalent to delay_norm=1).
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
            # clamp：极端情况下若时间略早于 start_time，也不要让 dt<0 影响指标
            # / clamp: in extreme cases where time is slightly earlier than start_time, do not let dt<0 affect the metric
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