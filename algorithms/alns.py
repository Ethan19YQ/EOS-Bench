"""Adaptive large neighbourhood search using EOS-Bench's common constraints.

This is a configurable baseline, not a reproduction of a published ALNS solver.
Repair considers a bounded task shortlist and samples feasible placements from
the shared constraint model. Regret-2 is recomputed after every insertion.
"""
from __future__ import annotations

import math
import time
from copy import deepcopy
from dataclasses import asdict, dataclass, field
from typing import Optional

from algorithms.objectives import ObjectiveModel, ObjectiveWeights
from algorithms.random_utils import make_rng, resolve_seed
from schedulers.constraint_model import Schedule
from schedulers.engine import BaseSchedulerAlgorithm


@dataclass
class ALNSConfig:
    max_iterations: int = 200
    removal_fraction_min: float = 0.05
    removal_fraction_max: float = 0.20
    repair_task_limit: int = 32
    worst_candidate_limit: int = 128
    placement_attempts: int = 4
    segment_length: int = 20
    reaction_factor: float = 0.2
    initial_temperature: float = 0.02  # ObjectiveModel uses normalised scores.
    cooling_rate: float = 0.99
    time_limit_s: Optional[float] = None  # Soft search cap; reported runtime also includes initialisation.
    seed: Optional[int] = None
    weights: ObjectiveWeights = field(default_factory=ObjectiveWeights)


class ALNSScheduler(BaseSchedulerAlgorithm):
    destroy_names = ("random", "worst", "related")
    repair_names = ("greedy", "regret2", "randomised_greedy")

    def __init__(self, cfg: Optional[ALNSConfig] = None):
        self.cfg = cfg or ALNSConfig()
        c = self.cfg
        for name in ("max_iterations", "repair_task_limit", "worst_candidate_limit", "placement_attempts", "segment_length"):
            value = getattr(c, name)
            if not isinstance(value, int) or value < (0 if name == "max_iterations" else 1):
                raise ValueError(f"{name} must be a valid non-negative/positive integer")
        if not 0 < c.removal_fraction_min <= c.removal_fraction_max <= 1:
            raise ValueError("Removal fractions must satisfy 0 < min <= max <= 1")
        if not 0 < c.reaction_factor <= 1 or not 0 < c.cooling_rate <= 1:
            raise ValueError("Reaction factor and cooling rate must be in (0, 1]")
        if not math.isfinite(c.initial_temperature) or c.initial_temperature <= 0:
            raise ValueError("initial_temperature must be positive and finite")
        if c.time_limit_s is not None and (not math.isfinite(c.time_limit_s) or c.time_limit_s <= 0):
            raise ValueError("time_limit_s must be positive and finite, or None")

    def _destroy(self, current, name, problem, obj, rng, deadline=math.inf):
        items = list(current.assignments)
        if not items:
            return Schedule(), []
        count = max(1, math.ceil(len(items) * rng.uniform(
            self.cfg.removal_fraction_min, self.cfg.removal_fraction_max)))
        # Bound destruction as well as repair so a large instance is not almost
        # emptied by a neighbourhood that can repair only a small task shortlist.
        count = min(count, len(items), self.cfg.repair_task_limit)
        if name == "random":
            removed = rng.sample(items, count)
        elif name == "worst":
            score = obj.score(current)
            # Rank exact marginal losses within a sampled pool, avoiding a full
            # O(n^2) objective sweep on large schedules. Always cover count items.
            indices = list(range(len(items)))
            pool_size = min(len(items), max(count, self.cfg.worst_candidate_limit))
            if pool_size < len(items):
                indices = rng.sample(indices, pool_size)
            ranked = []
            for i in indices:
                if time.perf_counter() >= deadline:
                    break
                loss = score - obj.score(Schedule(items[:i] + items[i + 1:]))
                ranked.append((loss, rng.random(), items[i]))
            ranked.sort(key=lambda x: (x[0], x[1]))
            removed = [row[2] for row in ranked[:count]]
        else:
            anchor = rng.choice(items)
            horizon = max(1., (problem.end_time - problem.start_time).total_seconds())
            ranked = [(float(a.satellite_id != anchor.satellite_id)
                       + abs((a.sat_start_time - anchor.sat_start_time).total_seconds()) / horizon,
                       rng.random(), a) for a in items]
            ranked.sort(key=lambda x: (x[0], x[1]))
            removed = [row[2] for row in ranked[:count]]
        ids = {a.task_id for a in removed}
        return Schedule([a for a in items if a.task_id not in ids]), [a.task_id for a in removed]

    def _repair(self, partial, removed, name, problem, cm, obj, rng, deadline):
        used = set(partial.assigned_task_ids)
        others = [tid for tid in sorted(problem.tasks) if tid not in used and tid not in removed]
        rng.shuffle(others)
        # Keep half the shortlist for previously unassigned tasks when possible.
        # Otherwise a full initial schedule could repeatedly restore only itself.
        limit = self.cfg.repair_task_limit
        quota = min(len(others), max(1, limit // 2)) if others else 0
        rng.shuffle(removed)
        pending = removed[:limit - quota] + others[:quota]
        remaining = [tid for tid in removed + others if tid not in pending]
        pending += remaining[:max(0, limit - len(pending))]
        while pending and time.perf_counter() < deadline:
            base = obj.score(partial)
            choices = []
            for tid in pending:
                if time.perf_counter() >= deadline:
                    break
                candidates, seen = [], set()
                for attempt in range(self.cfg.placement_attempts):
                    if time.perf_counter() >= deadline:
                        break
                    a = cm.build_feasible_assignment_for_task(
                        problem.tasks[tid], partial, randomized=(attempt > 0), rng=rng)
                    if a is None:
                        continue
                    key = (a.satellite_id, a.sat_start_time, a.sat_end_time,
                           a.ground_station_id, a.gs_start_time, a.gs_end_time)
                    if key in seen:
                        continue
                    seen.add(key)
                    score = obj.score(Schedule(partial.assignments + [a]))
                    candidates.append((score - base, a))
                if not candidates:
                    continue
                candidates.sort(key=lambda row: row[0], reverse=True)
                gain, assignment = candidates[0]
                if gain < -1e-12:
                    continue
                # Not inserting is an explicit zero-gain alternative.
                second = max(0., candidates[1][0]) if len(candidates) > 1 else 0.
                regret = gain - second
                priority = regret if name == "regret2" else gain
                if name == "randomised_greedy":
                    priority *= rng.uniform(0.5, 1.5)
                choices.append((priority, gain, rng.random(), assignment))
            if not choices:
                break
            assignment = max(choices, key=lambda row: row[:3])[3]
            partial.assignments.append(assignment)
            pending.remove(assignment.task_id)
        return partial

    def search(self, problem, constraint_model, initial_schedule):
        c = self.cfg
        seed = resolve_seed(c.seed)
        rng = make_rng(seed)
        obj = ObjectiveModel(problem, c.weights)
        started = time.perf_counter()
        deadline = started + c.time_limit_s if c.time_limit_s is not None else math.inf
        current = deepcopy(initial_schedule)
        # Fail loudly instead of silently benchmarking an invalid initial plan.
        check = Schedule()
        for a in current.assignments:
            if not constraint_model.is_feasible_assignment(a, check):
                raise ValueError("ALNS received an infeasible initial schedule")
            check.assignments.append(a)
        current_score = obj.score(current)
        best, best_score = deepcopy(current), current_score
        initial_score = best_score
        weights = [[1.] * len(self.destroy_names), [1.] * len(self.repair_names)]
        scores = [[0.] * len(row) for row in weights]
        counts = [[0] * len(row) for row in weights]
        uses = [[0] * len(row) for row in weights]
        temperature = c.initial_temperature
        accepted = improvements = iterations = 0

        def update_weights():
            for group in range(2):
                for i in range(len(weights[group])):
                    if counts[group][i]:
                        weights[group][i] = max(0.05, (1 - c.reaction_factor) * weights[group][i]
                            + c.reaction_factor * scores[group][i] / counts[group][i])
                    scores[group][i] = 0.
                    counts[group][i] = 0

        for iteration in range(c.max_iterations):
            if not problem.tasks or time.perf_counter() >= deadline:
                break
            di = rng.choices(range(len(self.destroy_names)), weights[0])[0]
            ri = rng.choices(range(len(self.repair_names)), weights[1])[0]
            partial, removed = self._destroy(current, self.destroy_names[di], problem, obj, rng, deadline)
            candidate = self._repair(partial, removed, self.repair_names[ri], problem,
                                     constraint_model, obj, rng, deadline)
            candidate_score = obj.score(candidate)
            delta = candidate_score - current_score
            reward = 0.
            if delta >= 0 or rng.random() < math.exp(delta / max(temperature, 1e-12)):
                accepted += 1
                reward = 2. if delta > 1e-12 else 1.
                current, current_score = candidate, candidate_score
                if current_score > best_score + 1e-12:
                    best, best_score = deepcopy(current), current_score
                    improvements += 1
                    reward = 5.
            for group, index in enumerate((di, ri)):
                scores[group][index] += reward
                counts[group][index] += 1
                uses[group][index] += 1
            iterations = iteration + 1
            if iterations % c.segment_length == 0:
                update_weights()
            temperature *= c.cooling_rate
        if iterations % c.segment_length:
            update_weights()
        best.metadata["alns"] = {
            "config": asdict(c), "resolved_seed": seed, "iterations": iterations,
            "initial_score": initial_score, "best_score": best_score,
            "accepted": accepted, "improvements": improvements,
            "destroy_weights": dict(zip(self.destroy_names, weights[0])),
            "repair_weights": dict(zip(self.repair_names, weights[1])),
            "destroy_uses": dict(zip(self.destroy_names, uses[0])),
            "repair_uses": dict(zip(self.repair_names, uses[1])),
            "stop_reason": "time_limit" if time.perf_counter() >= deadline else
                           ("empty_problem" if not problem.tasks else "iterations"),
        }
        return best
