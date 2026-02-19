# -*- coding: utf-8 -*-
"""
schedulers/engine.py
调度引擎与算法接口 / Scheduling Engine and Algorithm Interface

说明 / Description
----
- BaseSchedulerAlgorithm：算法统一接口（Protocol） / Unified algorithm interface (Protocol);
- SchedulingEngine：统一的“构建初始解 -> 调用算法 -> 返回结果”流程 / Unified "build initial solution -> call algorithm -> return result" workflow.
"""

from __future__ import annotations

from typing import Protocol

from .scenario_loader import SchedulingProblem
from .constraint_model import ConstraintModel, Schedule


class BaseSchedulerAlgorithm(Protocol):
    """
    调度算法的基本接口协议。
    Basic interface protocol for scheduling algorithms.
    """
    def search(
        self,
        problem: SchedulingProblem,
        constraint_model: ConstraintModel,
        initial_schedule: Schedule,
    ) -> Schedule:
        """
        执行搜索/优化过程并返回最终调度方案。
        Execute the search/optimization process and return the final schedule.
        """
        ...


class SchedulingEngine:
    """
    调度引擎，负责协调问题、约束模型和具体算法。
    Scheduling engine responsible for coordinating the problem, constraint model, and specific algorithm.
    """
    def __init__(
        self,
        problem: SchedulingProblem,
        constraint_model: ConstraintModel,
        algorithm: BaseSchedulerAlgorithm,
    ) -> None:
        self.problem = problem
        self.constraint_model = constraint_model
        self.algorithm = algorithm

    def run(self) -> Schedule:
        """
        运行调度流程：先构建初始解，然后交由算法进行搜索优化。
        Run the scheduling workflow: first build the initial solution, then pass it to the algorithm for search optimization.
        """
        initial_schedule = self.constraint_model.build_initial_schedule()
        return self.algorithm.search(
            problem=self.problem,
            constraint_model=self.constraint_model,
            initial_schedule=initial_schedule,
        )