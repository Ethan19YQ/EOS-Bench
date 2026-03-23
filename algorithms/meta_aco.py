# algorithms/meta_aco.py
# -*- coding: utf-8 -*-

"""
Main functionality:
This module implements an Ant Colony Optimization (ACO) scheduler for satellite
task scheduling. It supports diversified candidate generation, pheromone-based
solution construction, lightweight balance guidance, early stopping, restart
mechanisms, and reproducible random-seed handling.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from copy import deepcopy
from bisect import bisect_left

from schedulers.scenario_loader import SchedulingProblem
from schedulers.engine import BaseSchedulerAlgorithm
from schedulers.constraint_model import ConstraintModel, Schedule, Assignment
from schedulers.transition_utils import compute_transition_time_agile, delta_g_between

from algorithms.candidate_pool import enumerate_task_candidates
from algorithms.objectives import ObjectiveWeights, ObjectiveModel
from algorithms.random_utils import make_rng

import hashlib


def _stable_hash_int(s: str) -> int:
    """Stable hash unaffected by PYTHONHASHSEED, used for reproducible seed derivation
    in parallel or repeated runs."""
    h = hashlib.md5(s.encode("utf-8"), usedforsecurity=False).hexdigest()
    return int(h[:8], 16)


@dataclass
class ACOConfig:
    # For speed, the default scale is reduced.
    # It can still be increased in main or cfg_overrides.
    ants: int = 50
    iterations: int = 200
    alpha: float = 1.0
    beta: float = 2.0
    evaporation: float = 0.25
    q: float = 1.0
    max_candidates_per_task: int = 256
    random_samples_per_window: int = 10
    restarts: int = 1
    seed: Optional[int] = None
    weights: ObjectiveWeights = field(default_factory=lambda: ObjectiveWeights(1.0, 0.0, 0.0, 0.0))

    # Early stopping: stop if the global optimum is not improved for
    # a certain number of consecutive iterations
    early_stop_patience: int = 15
    early_stop_min_delta: float = 1e-9

    # Pheromone update: only update the top K solutions
    # By default, this updates only the best solution of the current iteration
    top_k_update: int = 50

    # Maximum number of candidate attempts per task during solution construction
    # This avoids scanning all feasible candidates
    attempt_limit_per_task: int = 50

    # Optional bias toward skipping the task
    # Larger values make skipping easier
    skip_bias: float = 0.02

    # Epsilon-greedy in the construction phase:
    # select a candidate randomly with a small probability to increase exploration
    epsilon_greedy: float = 0.03

    # Transition-time model parameters, passed from main or main_scheduler
    # The defaults are consistent with ConstraintModel
    agility_profile: str = "Standard-Agility"
    non_agile_transition_s: float = 10.0


def _weighted_choice(weights: List[float], rng) -> int:
    """Draw an index from weights according to probability.
    The weights do not need to be normalized."""
    total = 0.0
    for w in weights:
        total += w
    if total <= 0:
        return len(weights) - 1
    r = rng.random() * total
    s = 0.0
    for i, w in enumerate(weights):
        s += w
        if s >= r:
            return i
    return len(weights) - 1


class _AntState:

    def __init__(self, problem: SchedulingProblem, cfg: ACOConfig):
        self.problem = problem
        self.cfg = cfg
        self.sat_intervals: Dict[str, List[Tuple]] = {sid: [] for sid in problem.satellites.keys()}
        self.used_storage: Dict[Tuple[str, int], float] = {}
        self.used_power: Dict[Tuple[str, int], float] = {}

    def _transition_s(self, sat_id: str, prev_angles, next_angles) -> float:
        sat = self.problem.satellites[sat_id]
        if (sat.maneuverability_type or "agile").lower() != "agile":
            return float(self.cfg.non_agile_transition_s)
        dg = delta_g_between(prev_angles, next_angles)
        if dg is None:
            # Conservative handling when angle data is missing:
            # use a larger transition time to reduce the probability of infeasibility
            return float(self.cfg.non_agile_transition_s)
        return float(compute_transition_time_agile(dg, self.cfg.agility_profile))

    def quick_feasible(self, a: Assignment) -> bool:
        """Quick feasibility filtering: time overlap, transition time, and per-orbit resources.
        It does not check more complex constraints such as downlink, which are
        finally determined by ConstraintModel.
        """
        sid = a.satellite_id
        sat = self.problem.satellites.get(sid)
        if sat is None:
            return False

        # Per-orbit resource
        orb = int(getattr(a, "orbit_number", 0) or 0)
        key = (sid, orb)
        used_s = self.used_storage.get(key, 0.0)
        used_p = self.used_power.get(key, 0.0)
        if used_s + float(getattr(a, "data_volume_GB", 0.0) or 0.0) > float(sat.max_data_storage_GB or 0.0) + 1e-9:
            return False
        if used_p + float(getattr(a, "power_cost_W", 0.0) or 0.0) > float(sat.max_power_W or 0.0) + 1e-9:
            return False

        # Time overlap and transition time only need to check neighboring intervals
        # in the sorted list
        lst = self.sat_intervals[sid]
        st = a.sat_start_time
        pos = bisect_left([x[0] for x in lst], st)
        prev = lst[pos - 1] if pos > 0 else None
        nxt = lst[pos] if pos < len(lst) else None

        if prev is not None:
            prev_st, prev_et, prev_angles = prev[0], prev[1], prev[2]
            if prev_et > st:
                return False
            gap = self._transition_s(sid, prev_angles, a.sat_angles)
            if prev_et.timestamp() + gap > st.timestamp() + 1e-9:
                return False

        et = a.sat_end_time
        if nxt is not None:
            nxt_st, nxt_et, nxt_angles = nxt[0], nxt[1], nxt[2]
            if et > nxt_st:
                return False
            gap = self._transition_s(sid, a.sat_angles, nxt_angles)
            if et.timestamp() + gap > nxt_st.timestamp() + 1e-9:
                return False

        return True

    def add(self, a: Assignment) -> None:
        sid = a.satellite_id
        lst = self.sat_intervals[sid]
        st = a.sat_start_time
        pos = bisect_left([x[0] for x in lst], st)
        lst.insert(pos, (a.sat_start_time, a.sat_end_time, a.sat_angles, a))

        orb = int(getattr(a, "orbit_number", 0) or 0)
        key = (sid, orb)
        self.used_storage[key] = self.used_storage.get(key, 0.0) + float(getattr(a, "data_volume_GB", 0.0) or 0.0)
        self.used_power[key] = self.used_power.get(key, 0.0) + float(getattr(a, "power_cost_W", 0.0) or 0.0)


def _precompute_eta(problem: SchedulingProblem, obj: ObjectiveModel, task_ids: List[str], cand_map: Dict[str, List[Assignment]]) -> Dict[str, List[float]]:

    w = obj.weights  # Normalized
    wp = float(w.w_profit)
    wc = float(w.w_completion)
    wt = float(w.w_timeliness)

    total_priority = float(obj.total_priority) if float(obj.total_priority) > 0 else 1.0
    horizon_sec = float((problem.end_time - problem.start_time).total_seconds())
    if horizon_sec <= 0:
        horizon_sec = 1.0

    eta_map: Dict[str, List[float]] = {}
    for tid in task_ids:
        task = problem.tasks[tid]
        profit_part = float(task.priority) / total_priority  # [0,1]
        completion_part = 1.0

        cands = cand_map.get(tid, [])
        etas: List[float] = []
        for a in cands:
            delay = float((a.sat_start_time - problem.start_time).total_seconds())
            if delay < 0:
                delay = 0.0
            time_part = 1.0 - min(1.0, delay / horizon_sec)  # Larger means earlier

            # Static eta weighted by objective weights
            eta = wp * profit_part + wc * completion_part + wt * time_part
            etas.append(max(1e-6, float(eta)))
        eta_map[tid] = etas
    return eta_map


class AntColonyScheduler(BaseSchedulerAlgorithm):
    def __init__(self, cfg: Optional[ACOConfig] = None) -> None:
        self.cfg = cfg or ACOConfig()

    def _build_candidates(self, problem: SchedulingProblem, cm: ConstraintModel, seed: Optional[int]):
        task_ids = list(problem.tasks.keys())
        cand_map: Dict[str, List[Assignment]] = {}
        for tid in task_ids:
            cand_map[tid] = enumerate_task_candidates(
                problem=problem,
                task=problem.tasks[tid],
                placement_mode=cm.placement_mode,
                downlink_duration_ratio=cm.downlink_duration_ratio,
                max_candidates=self.cfg.max_candidates_per_task,
                random_samples_per_window=self.cfg.random_samples_per_window,
                seed=(None if seed is None else int(seed) + (_stable_hash_int(tid) % 100000)),
            )
            # Add candidate-order randomness for meta-heuristics.
            # This does not change the candidate set, only shuffles the order.
            if seed is None:
                task_rng = make_rng(None)
            else:
                task_rng = make_rng(int(seed) + (_stable_hash_int(tid) % 100000))
            task_rng.shuffle(cand_map[tid])
        return task_ids, cand_map

    def _run_once(self, problem, cm, seed) -> Schedule:
        rng = make_rng(seed)
        obj = ObjectiveModel(problem, self.cfg.weights)
        task_ids, cand_map = self._build_candidates(problem, cm, seed)

        eta_map = _precompute_eta(problem, obj, task_ids, cand_map)

        pheromone: Dict[str, List[float]] = {}
        for tid in task_ids:
            m = max(1, len(cand_map.get(tid, [])))
            pheromone[tid] = [1.0] * m

        best = Schedule()
        best_score = obj.score(best)
        no_improve = 0

        wp_balance = float(obj.weights.w_balance)

        for it in range(self.cfg.iterations):
            ant_results: List[Tuple[float, Dict[str, int], Schedule]] = []

            for _a in range(self.cfg.ants):
                sch = Schedule()
                state = _AntState(problem, self.cfg)
                workloads = {sid: 0.0 for sid in obj.sat_ids}
                chosen_idx: Dict[str, int] = {}

                # Shuffle task order to increase randomness
                order = task_ids[:]
                rng.shuffle(order)

                for tid in order:
                    cands = cand_map.get(tid, [])
                    if not cands:
                        continue

                    etas = eta_map.get(tid, [])
                    m = len(cands)
                    # weights_all[0] represents skip, and the remaining entries
                    # correspond to candidate k=0..m-1
                    # Higher completion weight discourages skipping
                    eff_skip = float(self.cfg.skip_bias) * (1.0 - float(obj.weights.w_completion))
                    weights_all: List[float] = [max(1e-9, eff_skip)]
                    for k in range(m):
                        tau = pheromone[tid][k]
                        eta = etas[k] if k < len(etas) else 1e-6
                        bal_factor = 1.0
                        if wp_balance > 0:
                            # Lightweight balance guidance:
                            # prioritize satellites with smaller current workloads
                            sid = cands[k].satellite_id
                            # Larger balance weight provides stronger guidance
                            bal_gamma = 1.0 + 4.0 * float(wp_balance)
                            bal_factor = (1.0 / (1.0 + float(workloads.get(sid, 0.0)))) ** bal_gamma
                        w = (float(tau) ** self.cfg.alpha) * (float(eta) ** self.cfg.beta) * float(bal_factor)
                        weights_all.append(max(1e-12, float(w)))

                    tried: set[int] = set()
                    accepted = False
                    for _try in range(max(1, int(self.cfg.attempt_limit_per_task))):
                        # Epsilon-greedy:
                        # with a small probability, perform random exploration to avoid
                        # premature collapse caused by pheromone and heuristic guidance
                        if float(self.cfg.epsilon_greedy) > 0 and rng.random() < float(self.cfg.epsilon_greedy):
                            pick = rng.randrange(0, len(weights_all))
                        else:
                            pick = _weighted_choice(weights_all, rng)
                        if pick == 0:
                            # Skip
                            accepted = True
                            break
                        k = pick - 1
                        if k < 0 or k >= m or k in tried:
                            continue
                        tried.add(k)
                        a = cands[k]

                        # Quick filtering for resources, time, and transition
                        if not state.quick_feasible(a):
                            continue
                        # Final feasibility check, including downlink and similar constraints
                        if not cm.is_feasible_assignment(a, sch):
                            continue

                        sch.assignments.append(a)
                        state.add(a)
                        chosen_idx[tid] = k
                        dur = float((a.sat_end_time - a.sat_start_time).total_seconds())
                        workloads[a.satellite_id] = workloads.get(a.satellite_id, 0.0) + dur
                        accepted = True
                        break

                    # If no candidate is accepted, it is also treated as a skip
                    if not accepted:
                        continue

                sc = obj.score(sch)
                ant_results.append((sc, chosen_idx, sch))

            # Best solution of the current iteration
            ant_results.sort(key=lambda x: x[0], reverse=True)
            iter_best_score, iter_best_choice, iter_best_schedule = ant_results[0]

            if iter_best_score > best_score + float(self.cfg.early_stop_min_delta):
                best_score = iter_best_score
                best = deepcopy(iter_best_schedule)
                no_improve = 0
            else:
                no_improve += 1

            # Evaporation
            for tid in task_ids:
                ph = pheromone[tid]
                for k in range(len(ph)):
                    ph[k] = max(1e-9, float(ph[k]) * (1.0 - float(self.cfg.evaporation)))

            # Deposit: only update the top K solutions
            # By default, only the best of the current iteration is updated
            K = max(1, int(self.cfg.top_k_update))
            for rank in range(min(K, len(ant_results))):
                sc, choice, _sch = ant_results[rank]
                deposit = float(self.cfg.q) * max(1e-6, float(sc))
                for tid, k in choice.items():
                    if 0 <= k < len(pheromone[tid]):
                        pheromone[tid][k] += deposit

            # Early stopping
            if no_improve >= max(1, int(self.cfg.early_stop_patience)):
                break

        return best

    def search(self, problem: SchedulingProblem, constraint_model: ConstraintModel, initial_schedule: Schedule) -> Schedule:
        obj = ObjectiveModel(problem, self.cfg.weights)
        best = deepcopy(initial_schedule)
        best_score = obj.score(best)

        base = self.cfg.seed
        for r in range(max(1, int(self.cfg.restarts))):
            s = (None if base is None else base + r * 99991)
            cand = self._run_once(problem, constraint_model, s)
            sc = obj.score(cand)
            if sc > best_score:
                best, best_score = cand, sc

        return best