# -*- coding: utf-8 -*-
"""
schedulers/engine.py

Main functionality:
This module defines the scheduling engine and the base algorithm protocol.
The engine coordinates the scheduling problem, constraint model, and selected
algorithm, and runs the workflow by first generating an initial schedule and
then invoking the algorithm for optimization.
"""

from __future__ import annotations

from typing import Protocol

from .scenario_loader import SchedulingProblem
from .constraint_model import ConstraintModel, Schedule


class BaseSchedulerAlgorithm(Protocol):
    """
    Basic interface protocol for scheduling algorithms.
    """
    def search(
        self,
        problem: SchedulingProblem,
        constraint_model: ConstraintModel,
        initial_schedule: Schedule,
    ) -> Schedule:
        """
        Execute the search or optimization process and return the final schedule.
        """
        ...


class SchedulingEngine:
    """
    Scheduling engine responsible for coordinating the problem,
    constraint model, and specific algorithm.
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
        Run the scheduling workflow by first building the initial solution,
        then passing it to the algorithm for search and optimization.
        """
        initial_schedule = self.constraint_model.build_initial_schedule()
        return self.algorithm.search(
            problem=self.problem,
            constraint_model=self.constraint_model,
            initial_schedule=initial_schedule,
        )