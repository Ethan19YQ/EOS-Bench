# algorithms/heuristics.py
# -*- coding: utf-8 -*-

"""
Main functionality:
This module implements several heuristic scheduling algorithms, including
completion-first, profit-first, timeliness-first, and balance-first strategies.
It also defines shared heuristic configuration and helper utilities for
candidate selection and randomized search behavior.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Dict, List, Tuple
import time
import random

from schedulers.scenario_loader import SchedulingProblem
from schedulers.engine import BaseSchedulerAlgorithm
from schedulers.constraint_model import ConstraintModel, Schedule, Assignment

from algorithms.candidate_pool import enumerate_task_candidates


# =========================
# Config
# =========================

@dataclass
class HeuristicConfig:

    max_candidates_per_task: int = 256
    max_checks_per_task: int = 256
    completion_scan_tasks_per_iter: int = 256
    randomized: bool = True
    seed: Optional[int] = None
    early_accept: bool = True


# =========================
# Small helpers
# =========================

def _finish_ts(a: Assignment) -> float:
    """Get the timestamp when the task is finished, including downlink."""
    t = a.gs_end_time or a.sat_end_time
    return float(t.timestamp())


def _busy(a: Assignment) -> float:
    """Calculate the total occupied duration of this assignment in seconds."""
    s = (a.sat_end_time - a.sat_start_time).total_seconds()
    if a.gs_start_time and a.gs_end_time:
        s += (a.gs_end_time - a.gs_start_time).total_seconds()
    return float(s)


def _make_rng(seed: Optional[int]) -> random.Random:
    """Create a random generator."""
    if seed is None:
        # Default to a different value for each run at nanosecond resolution
        seed = int(time.time_ns() & 0x7FFFFFFF)
    return random.Random(seed)


def _shuffle_if(rng: random.Random, items: List, do: bool) -> None:
    """Conditionally shuffle the list."""
    if do and len(items) > 1:
        rng.shuffle(items)


def _pick_best_feasible_for_task(
    *,
    tid: str,
    task_priority: float,
    candidates: List[Assignment],
    schedule: Schedule,
    cm: ConstraintModel,
    rng: random.Random,
    cfg: HeuristicConfig,
    key_fn,
) -> Optional[Assignment]:
    """Pick one feasible assignment from candidates, with a check limit and optional early accept."""
    if not candidates:
        return None

    # Candidate order is important:
    # at large scale, do not keep a fixed order, otherwise the search will
    # repeatedly follow the same path
    cand_list = candidates
    if cfg.randomized:
        cand_list = list(candidates)  # Avoid modifying the shared list
        rng.shuffle(cand_list)

    best_a: Optional[Assignment] = None
    best_key = None

    checks = 0
    for a in cand_list:
        # Limit the number of checks so each task does not scan all candidates
        checks += 1
        if checks > cfg.max_checks_per_task:
            break

        if not cm.is_feasible_assignment(a, schedule):
            continue

        if cfg.early_accept:
            return a

        k = key_fn(a, task_priority)
        if best_a is None or k < best_key:
            best_a = a
            best_key = k

    return best_a


# =========================
# Heuristics
# =========================


class HeuristicCompletionFirstScheduler(BaseSchedulerAlgorithm):


    def __init__(self, cfg: Optional[HeuristicConfig] = None) -> None:
        self.cfg = cfg or HeuristicConfig()
        self.rng = _make_rng(self.cfg.seed)

    def search(
        self,
        problem: SchedulingProblem,
        constraint_model: ConstraintModel,
        initial_schedule: Schedule,
    ) -> Schedule:
        schedule = Schedule()

        # Generate the candidate pool once.
        # Truncation and optional random sampling are handled by enumerate.
        cand_map: Dict[str, List[Assignment]] = {}
        task_ids = list(problem.tasks.keys())
        for tid in task_ids:
            task = problem.tasks[tid]
            cand_map[tid] = enumerate_task_candidates(
                problem=problem,
                task=task,
                placement_mode=constraint_model.placement_mode,
                downlink_duration_ratio=constraint_model.downlink_duration_ratio,
                max_candidates=self.cfg.max_candidates_per_task,
                random_samples_per_window=0,
                seed=None,  # Candidates themselves do not strongly depend on seed; we shuffle later
            )

        remaining = set(task_ids)

        # Task sample list used for per-iteration sampling
        remaining_list = task_ids[:]
        _shuffle_if(self.rng, remaining_list, self.cfg.randomized)

        while remaining:
            # Sample a subset of tasks each round to estimate MRV
            candidates_to_check = list(remaining)
            if len(candidates_to_check) > self.cfg.completion_scan_tasks_per_iter:
                # Sample a subset for checking
                candidates_to_check = self.rng.sample(
                    candidates_to_check, self.cfg.completion_scan_tasks_per_iter
                )

            best_tid = None
            best_feasible_cnt = None
            best_pick: Optional[Assignment] = None

            for tid in candidates_to_check:
                task = problem.tasks[tid]
                cands = cand_map.get(tid, [])
                if not cands:
                    continue

                # Roughly count feasible candidates, only within the check limit
                cnt = 0
                pick: Optional[Assignment] = None
                checks = 0

                local = cands
                if self.cfg.randomized:
                    local = list(cands)
                    self.rng.shuffle(local)

                for a in local:
                    checks += 1
                    if checks > self.cfg.max_checks_per_task:
                        break
                    if constraint_model.is_feasible_assignment(a, schedule):
                        cnt += 1
                        if pick is None:
                            pick = a
                        # With early_accept, it is enough to know that a feasible candidate exists
                        if self.cfg.early_accept:
                            break

                if cnt <= 0 or pick is None:
                    continue

                if best_tid is None or cnt < best_feasible_cnt:
                    best_tid = tid
                    best_feasible_cnt = cnt
                    best_pick = pick

            if best_tid is None or best_pick is None:
                # If sampling finds no feasible candidate, try one full scan
                # with the same limit to avoid missing a good choice
                best_tid = None
                best_key = None
                best_pick = None
                for tid in list(remaining):
                    task = problem.tasks[tid]
                    a = _pick_best_feasible_for_task(
                        tid=tid,
                        task_priority=float(task.priority),
                        candidates=cand_map.get(tid, []),
                        schedule=schedule,
                        cm=constraint_model,
                        rng=self.rng,
                        cfg=self.cfg,
                        # Completion-first fallback:
                        # try to finish early and use less busy time
                        key_fn=lambda aa, pr: (_finish_ts(aa), _busy(aa), -pr),
                    )
                    if a is None:
                        continue
                    k = (_finish_ts(a), _busy(a), -float(task.priority), tid)
                    if best_tid is None or k < best_key:
                        best_tid = tid
                        best_key = k
                        best_pick = a

                if best_tid is None or best_pick is None:
                    break

            schedule.assignments.append(best_pick)
            remaining.remove(best_tid)

        return schedule


class HeuristicProfitFirstScheduler(BaseSchedulerAlgorithm):


    def __init__(self, cfg: Optional[HeuristicConfig] = None) -> None:
        self.cfg = cfg or HeuristicConfig()
        self.rng = _make_rng(self.cfg.seed)

    def search(
        self,
        problem: SchedulingProblem,
        constraint_model: ConstraintModel,
        initial_schedule: Schedule,
    ) -> Schedule:
        schedule = Schedule()

        # Candidate pool
        cand_map: Dict[str, List[Assignment]] = {}
        task_ids = list(problem.tasks.keys())
        for tid in task_ids:
            task = problem.tasks[tid]
            cand_map[tid] = enumerate_task_candidates(
                problem=problem,
                task=task,
                placement_mode=constraint_model.placement_mode,
                downlink_duration_ratio=constraint_model.downlink_duration_ratio,
                max_candidates=self.cfg.max_candidates_per_task,
                random_samples_per_window=0,
                seed=None,
            )

        # Sort by descending priority; tasks with the same priority can be shuffled
        task_ids.sort(key=lambda tid: float(problem.tasks[tid].priority), reverse=True)
        if self.cfg.randomized:
            # Shuffle within same-priority blocks
            i = 0
            while i < len(task_ids):
                j = i + 1
                pi = float(problem.tasks[task_ids[i]].priority)
                while j < len(task_ids) and float(problem.tasks[task_ids[j]].priority) == pi:
                    j += 1
                block = task_ids[i:j]
                self.rng.shuffle(block)
                task_ids[i:j] = block
                i = j

        for tid in task_ids:
            task = problem.tasks[tid]
            a = _pick_best_feasible_for_task(
                tid=tid,
                task_priority=float(task.priority),
                candidates=cand_map.get(tid, []),
                schedule=schedule,
                cm=constraint_model,
                rng=self.rng,
                cfg=self.cfg,
                # Profit-first:
                # use earlier finish and lower busy time as tie-breakers
                key_fn=lambda aa, pr: (-pr, _finish_ts(aa), _busy(aa)),
            )
            if a is None:
                continue
            schedule.assignments.append(a)

        return schedule


class HeuristicTimelinessFirstScheduler(BaseSchedulerAlgorithm):


    def __init__(self, cfg: Optional[HeuristicConfig] = None) -> None:
        self.cfg = cfg or HeuristicConfig()
        self.rng = _make_rng(self.cfg.seed)

    def search(
        self,
        problem: SchedulingProblem,
        constraint_model: ConstraintModel,
        initial_schedule: Schedule,
    ) -> Schedule:
        schedule = Schedule()

        cand_map: Dict[str, List[Assignment]] = {}
        task_ids = list(problem.tasks.keys())
        earliest_start: Dict[str, float] = {}

        for tid in task_ids:
            task = problem.tasks[tid]
            cands = enumerate_task_candidates(
                problem=problem,
                task=task,
                placement_mode=constraint_model.placement_mode,
                downlink_duration_ratio=constraint_model.downlink_duration_ratio,
                max_candidates=self.cfg.max_candidates_per_task,
                random_samples_per_window=0,
                seed=None,
            )
            cand_map[tid] = cands
            if cands:
                # Approximation:
                # the earliest sat_start_time among candidates,
                # without feasibility checking, used only for sorting
                st = min(a.sat_start_time for a in cands)
                earliest_start[tid] = float(st.timestamp())
            else:
                earliest_start[tid] = float("inf")

        # Sort tasks by earliest start time ascending;
        # tasks in the same bucket can be shuffled
        task_ids.sort(key=lambda tid: (earliest_start.get(tid, float("inf")), -float(problem.tasks[tid].priority)))
        if self.cfg.randomized:
            # Shuffle task blocks with approximately the same start time
            # to avoid a fixed ordering
            i = 0
            while i < len(task_ids):
                j = i + 1
                si = earliest_start.get(task_ids[i], float("inf"))
                while j < len(task_ids) and abs(earliest_start.get(task_ids[j], float("inf")) - si) < 1e-6:
                    j += 1
                block = task_ids[i:j]
                self.rng.shuffle(block)
                task_ids[i:j] = block
                i = j

        for tid in task_ids:
            task = problem.tasks[tid]
            a = _pick_best_feasible_for_task(
                tid=tid,
                task_priority=float(task.priority),
                candidates=cand_map.get(tid, []),
                schedule=schedule,
                cm=constraint_model,
                rng=self.rng,
                cfg=self.cfg,
                # Timeliness:
                # earlier sat_start is better; priority and busy time are secondary
                key_fn=lambda aa, pr: (aa.sat_start_time, -pr, _busy(aa), _finish_ts(aa)),
            )
            if a is None:
                continue
            schedule.assignments.append(a)

        return schedule


class HeuristicBalanceFirstScheduler(BaseSchedulerAlgorithm):

    def __init__(self, cfg: Optional[HeuristicConfig] = None) -> None:
        self.cfg = cfg or HeuristicConfig()
        self.rng = _make_rng(self.cfg.seed)

    def search(
        self,
        problem: SchedulingProblem,
        constraint_model: ConstraintModel,
        initial_schedule: Schedule,
    ) -> Schedule:
        schedule = Schedule()

        # Workload accumulation
        workloads: Dict[str, float] = {sid: 0.0 for sid in problem.satellites.keys()}

        # Candidate pool
        cand_map: Dict[str, List[Assignment]] = {}
        task_ids = list(problem.tasks.keys())
        for tid in task_ids:
            task = problem.tasks[tid]
            cand_map[tid] = enumerate_task_candidates(
                problem=problem,
                task=task,
                placement_mode=constraint_model.placement_mode,
                downlink_duration_ratio=constraint_model.downlink_duration_ratio,
                max_candidates=self.cfg.max_candidates_per_task,
                random_samples_per_window=0,
                seed=None,
            )

        # First sort by descending priority to avoid failing to schedule tasks
        # just for the sake of balance
        task_ids.sort(key=lambda tid: float(problem.tasks[tid].priority), reverse=True)
        _shuffle_if(self.rng, task_ids, self.cfg.randomized)

        for tid in task_ids:
            task = problem.tasks[tid]
            cands = cand_map.get(tid, [])
            if not cands:
                continue

            # Select a candidate for this task that improves balance
            def key_fn(a: Assignment, pr: float) -> Tuple[float, float, float, float]:
                # Smaller is better:
                # assign the task to the satellite with smaller current workload
                wl = workloads.get(a.satellite_id, 0.0)
                # Workload after adding the current task increment
                dur = float((a.sat_end_time - a.sat_start_time).total_seconds())
                wl_after = wl + dur
                return (wl_after, _finish_ts(a), _busy(a), -pr)

            a = _pick_best_feasible_for_task(
                tid=tid,
                task_priority=float(task.priority),
                candidates=cands,
                schedule=schedule,
                cm=constraint_model,
                rng=self.rng,
                cfg=self.cfg,
                key_fn=key_fn,
            )
            if a is None:
                continue

            schedule.assignments.append(a)
            dur = float((a.sat_end_time - a.sat_start_time).total_seconds())
            workloads[a.satellite_id] = workloads.get(a.satellite_id, 0.0) + dur

        return schedule