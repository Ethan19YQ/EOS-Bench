# algorithms/meta_sa.py
# -*- coding: utf-8 -*-

"""
Main functionality:
This module implements a Simulated Annealing (SA) scheduler for satellite task
scheduling. It supports neighbor generation, Metropolis acceptance, temperature
cooling, and multi-restart search under configurable objective weights.
"""

from __future__ import annotations

import math
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Optional

from schedulers.scenario_loader import SchedulingProblem
from schedulers.engine import BaseSchedulerAlgorithm
from schedulers.constraint_model import ConstraintModel, Schedule

from algorithms.objectives import ObjectiveWeights, ObjectiveModel
from algorithms.random_utils import make_rng


@dataclass
class SAConfig:
    """Simulated Annealing configuration."""
    max_iterations: int = 2000  # Maximum number of iterations
    initial_temperature: float = 100.0  # Initial temperature
    cooling_rate: float = 0.99  # Cooling rate
    neighbor_attempts: int = 50  # Maximum attempts to find a neighbor solution
    restarts: int = 1  # Number of restarts
    seed: Optional[int] = None  # Random seed
    weights: ObjectiveWeights = field(
        default_factory=lambda: ObjectiveWeights(1.0, 0.0, 0.0, 0.0))  # Objective weights


class SimulatedAnnealingScheduler(BaseSchedulerAlgorithm):
    """Simulated Annealing scheduler."""

    def __init__(
            self,
            max_iterations: int = 2000,
            initial_temperature: float = 100.0,
            cooling_rate: float = 0.99,
            neighbor_attempts: float = 50,
            restarts: int = 1,
            seed: Optional[int] = None,
            weights: Optional[ObjectiveWeights] = None,
    ) -> None:
        self.cfg = SAConfig(
            max_iterations=max_iterations,
            initial_temperature=initial_temperature,
            cooling_rate=cooling_rate,
            neighbor_attempts=int(neighbor_attempts),
            restarts=max(1, int(restarts)),
            seed=seed,
            weights=weights or ObjectiveWeights(1.0, 0.0, 0.0, 0.0),
        )

    def _generate_neighbor(self, cm: ConstraintModel, current: Schedule, task_ids: list[str], rng) -> Optional[
        Schedule]:
        """
        Generate a neighbor solution by randomly selecting one task, removing its
        existing assignment if present, and trying to find a new random feasible
        position for it.
        """
        for _ in range(self.cfg.neighbor_attempts):
            new_s = deepcopy(current)

            # Randomly pick one task
            tid = rng.choice(task_ids)
            # Remove the old assignment of this task
            for a in new_s.get_assignments_for_task(tid):
                new_s.assignments.remove(a)

            # Try to find a new random feasible assignment
            task = cm.problem.tasks[tid]
            new_a = cm.build_feasible_assignment_for_task(task=task, schedule=new_s, randomized=True, rng=rng)
            if new_a is not None:
                new_s.assignments.append(new_a)
                return new_s
        return None

    def _run_once(self, problem, cm, initial, seed) -> Schedule:
        """Run one complete simulated annealing process."""
        rng = make_rng(seed)
        obj = ObjectiveModel(problem, self.cfg.weights)

        # Initialize the current solution and the best solution
        cur = deepcopy(initial)
        cur_score = obj.score(cur)
        best = deepcopy(cur)
        best_score = cur_score

        T = float(self.cfg.initial_temperature)
        task_ids = list(problem.tasks.keys())

        # Main annealing loop
        for _ in range(self.cfg.max_iterations):
            # Cooling
            T *= self.cfg.cooling_rate
            if T <= 1e-9:
                break

            # Generate a neighbor
            nb = self._generate_neighbor(cm, cur, task_ids, rng)
            if nb is None:
                continue

            # Calculate the score difference
            nb_score = obj.score(nb)
            delta = nb_score - cur_score

            # Metropolis criterion:
            # accept if better; otherwise accept a worse solution with a probability
            # that decreases as the temperature decreases
            if delta >= 0 or rng.random() < math.exp(delta / max(1e-9, T)):
                cur = nb
                cur_score = nb_score
                # Update the global best solution
                if cur_score > best_score:
                    best_score = cur_score
                    best = deepcopy(cur)

        return best

    def search(self, problem: SchedulingProblem, constraint_model: ConstraintModel,
               initial_schedule: Schedule) -> Schedule:
        """Main search interface with multiple random restarts to avoid local optima."""
        # Run multiple random restarts and keep the best result
        best = deepcopy(initial_schedule)
        best_score = ObjectiveModel(problem, self.cfg.weights).score(best)

        base_seed = self.cfg.seed
        for r in range(self.cfg.restarts):
            s = (None if base_seed is None else base_seed + r * 9973)
            cand = self._run_once(problem, constraint_model, initial_schedule, s)
            sc = ObjectiveModel(problem, self.cfg.weights).score(cand)
            if sc > best_score:
                best, best_score = cand, sc

        return best