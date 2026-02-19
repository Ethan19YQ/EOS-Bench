# algorithms/meta_aco.py
# -*- coding: utf-8 -*-

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
    """稳定 hash（不受 PYTHONHASHSEED 影响），用于并行/多次运行的可复现种子派生。"""
    h = hashlib.md5(s.encode("utf-8"), usedforsecurity=False).hexdigest()
    return int(h[:8], 16)


@dataclass
class ACOConfig:
    # 为了速度，默认降低规模；仍可在 main/cfg_overrides 中改大
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

    # 早停：连续多少轮未提升全局最优则停止
    early_stop_patience: int = 15
    early_stop_min_delta: float = 1e-9

    # 信息素更新：只更新前 K 个解（默认只更新本轮最好）
    top_k_update: int = 50

    # 构造解时每个任务最多尝试多少个候选（避免全量可行性扫描）
    attempt_limit_per_task: int = 50

    # 可选：偏向“跳过该任务”的权重（越大越容易 skip）
    skip_bias: float = 0.02

    # 构造阶段 ε-greedy：以小概率随机选候选（增加探索，避免塌缩到同一路径）
    epsilon_greedy: float = 0.03

    # 转换时间模型参数（从 main/main_scheduler 传入；默认与 ConstraintModel 一致）
    agility_profile: str = "Standard-Agility"
    non_agile_transition_s: float = 10.0


def _weighted_choice(weights: List[float], rng) -> int:
    """从 weights 中按权重抽一个 index。weights 可以不归一化。"""
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
    """用于 ACO 构造阶段的轻量增量状态：
    - 按卫星维护已安排区间（按开始时间排序），只与前后邻居做冲突/转换检查
    - 按 (sat, orbit) 维护资源累积（O(1)）
    说明：该状态只做“快速过滤”，最终仍以 ConstraintModel.is_feasible_assignment 为准。
    """

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
            # 角度缺失时保守处理：给一个较大的转换时间，降低不可行概率
            return float(self.cfg.non_agile_transition_s)
        return float(compute_transition_time_agile(dg, self.cfg.agility_profile))

    def quick_feasible(self, a: Assignment) -> bool:
        """快速可行性过滤：时间重叠/转换时间/每圈资源。
        不检查下传等复杂约束（由 ConstraintModel 最终判定）。
        """
        sid = a.satellite_id
        sat = self.problem.satellites.get(sid)
        if sat is None:
            return False

        # per-orbit resource
        orb = int(getattr(a, "orbit_number", 0) or 0)
        key = (sid, orb)
        used_s = self.used_storage.get(key, 0.0)
        used_p = self.used_power.get(key, 0.0)
        if used_s + float(getattr(a, "data_volume_GB", 0.0) or 0.0) > float(sat.max_data_storage_GB or 0.0) + 1e-9:
            return False
        if used_p + float(getattr(a, "power_cost_W", 0.0) or 0.0) > float(sat.max_power_W or 0.0) + 1e-9:
            return False

        # time overlap + transition only needs check neighbors in sorted list
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
    """预计算启发式信息 eta（静态部分）。

    目标：让构造概率能真实反映 *当前* 优化目标（profit/completion/timeliness），从而不同权重会产生不同的构造偏好。

    设计：
    - profit_part：任务优先级归一化（与候选无关，但与任务相关）
    - completion_part：常数 1（鼓励完成更多任务）
    - time_part：候选开始越早越好，取 (1 - delay/T) ∈ [0,1]

    注意：balance 依赖当前部分解（工作量分布），不适合静态缓存；在构造阶段通过动态 bal_factor 处理。
    """
    w = obj.weights  # 已归一化
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
            time_part = 1.0 - min(1.0, delay / horizon_sec)  # 越早越大

            # 静态 eta：按权重加权
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
            # 为元启发式增加候选顺序随机性（不改变候选集合，只打乱顺序）
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

                # task 顺序打乱（增加随机性）
                order = task_ids[:]
                rng.shuffle(order)

                for tid in order:
                    cands = cand_map.get(tid, [])
                    if not cands:
                        continue

                    etas = eta_map.get(tid, [])
                    m = len(cands)
                    # weights_all[0] 表示 skip，其余对应候选 k=0..m-1
                    # completion 权重越高，越不鼓励 skip
                    eff_skip = float(self.cfg.skip_bias) * (1.0 - float(obj.weights.w_completion))
                    weights_all: List[float] = [max(1e-9, eff_skip)]
                    for k in range(m):
                        tau = pheromone[tid][k]
                        eta = etas[k] if k < len(etas) else 1e-6
                        bal_factor = 1.0
                        if wp_balance > 0:
                            # 轻量均衡引导：优先选择当前工作量较小的卫星
                            sid = cands[k].satellite_id
                            # balance 权重越大，引导越强
                            bal_gamma = 1.0 + 4.0 * float(wp_balance)
                            bal_factor = (1.0 / (1.0 + float(workloads.get(sid, 0.0)))) ** bal_gamma
                        w = (float(tau) ** self.cfg.alpha) * (float(eta) ** self.cfg.beta) * float(bal_factor)
                        weights_all.append(max(1e-12, float(w)))

                    tried: set[int] = set()
                    accepted = False
                    for _try in range(max(1, int(self.cfg.attempt_limit_per_task))):
                        # ε-greedy：少量概率随机探索，避免信息素+启发式过早塌缩
                        if float(self.cfg.epsilon_greedy) > 0 and rng.random() < float(self.cfg.epsilon_greedy):
                            pick = rng.randrange(0, len(weights_all))
                        else:
                            pick = _weighted_choice(weights_all, rng)
                        if pick == 0:
                            # skip
                            accepted = True
                            break
                        k = pick - 1
                        if k < 0 or k >= m or k in tried:
                            continue
                        tried.add(k)
                        a = cands[k]

                        # 快速过滤（资源/时间/转换）
                        if not state.quick_feasible(a):
                            continue
                        # 最终可行性（含下传等）
                        if not cm.is_feasible_assignment(a, sch):
                            continue

                        sch.assignments.append(a)
                        state.add(a)
                        chosen_idx[tid] = k
                        dur = float((a.sat_end_time - a.sat_start_time).total_seconds())
                        workloads[a.satellite_id] = workloads.get(a.satellite_id, 0.0) + dur
                        accepted = True
                        break

                    # 若未接受任何候选，也视为 skip
                    if not accepted:
                        continue

                sc = obj.score(sch)
                ant_results.append((sc, chosen_idx, sch))

            # 本轮最优
            ant_results.sort(key=lambda x: x[0], reverse=True)
            iter_best_score, iter_best_choice, iter_best_schedule = ant_results[0]

            if iter_best_score > best_score + float(self.cfg.early_stop_min_delta):
                best_score = iter_best_score
                best = deepcopy(iter_best_schedule)
                no_improve = 0
            else:
                no_improve += 1

            # 蒸发
            for tid in task_ids:
                ph = pheromone[tid]
                for k in range(len(ph)):
                    ph[k] = max(1e-9, float(ph[k]) * (1.0 - float(self.cfg.evaporation)))

            # 沉积：只更新前 K 个解（默认仅本轮最好）
            K = max(1, int(self.cfg.top_k_update))
            for rank in range(min(K, len(ant_results))):
                sc, choice, _sch = ant_results[rank]
                deposit = float(self.cfg.q) * max(1e-6, float(sc))
                for tid, k in choice.items():
                    if 0 <= k < len(pheromone[tid]):
                        pheromone[tid][k] += deposit

            # 早停
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
