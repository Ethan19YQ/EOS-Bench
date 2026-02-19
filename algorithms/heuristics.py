# algorithms/heuristics.py
# -*- coding: utf-8 -*-

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
    """启发式配置（强调大规模下的可控耗时）

    说明
    ----
    - max_candidates_per_task: 候选池上限（候选生成阶段就截断）
    - max_checks_per_task: 选择某个任务的落点时，最多检查多少个候选（可行性判定次数上限）
    - completion_scan_tasks_per_iter: completion-first 每轮从 remaining 中最多抽查多少任务来找 MRV
      （避免每轮对所有 remaining 任务做全量可行性扫描）
    - randomized: 是否打乱候选/任务顺序（提升随机性，避免总是同一个解）
    - seed: 随机种子；None 时会用 time_ns 生成（默认每次运行不同）
    - early_accept: 若为 True，找到第一个可行候选就接受（更快，但可能略降质量）
      若为 False，会在检查上限内选择“最优”的可行候选（更稳，但更慢）
    """
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
    t = a.gs_end_time or a.sat_end_time
    return float(t.timestamp())


def _busy(a: Assignment) -> float:
    s = (a.sat_end_time - a.sat_start_time).total_seconds()
    if a.gs_start_time and a.gs_end_time:
        s += (a.gs_end_time - a.gs_start_time).total_seconds()
    return float(s)


def _make_rng(seed: Optional[int]) -> random.Random:
    if seed is None:
        # 默认每次运行不同（纳秒级）
        seed = int(time.time_ns() & 0x7FFFFFFF)
    return random.Random(seed)


def _shuffle_if(rng: random.Random, items: List, do: bool) -> None:
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
    """在候选中选一个可行落点（带检查上限 + 可选 early accept）"""
    if not candidates:
        return None

    # 候选顺序（重要：大规模时不要固定顺序；否则会稳定地走同一条路径）
    cand_list = candidates
    if cfg.randomized:
        cand_list = list(candidates)  # 避免污染共享列表
        rng.shuffle(cand_list)

    best_a: Optional[Assignment] = None
    best_key = None

    checks = 0
    for a in cand_list:
        # 上限：避免每个任务把候选全扫完
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
    """完成率优先（MRV）：优先选择“当前最容易失败/候选最少”的任务。

    优化点
    ----
    原实现每一轮对 remaining 的每个任务都全量过滤 feasible candidates（非常慢）。
    新实现做了三件事：
    1) 每轮只抽查部分任务（completion_scan_tasks_per_iter），大幅降耗；
    2) 对每个任务只检查 max_checks_per_task 个候选；
    3) 找到可行候选后可 early_accept。
    """

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

        # 候选池一次性生成（截断 + 可选随机采样由 enumerate 决定）
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
                seed=None,  # 候选本身不强依赖 seed（我们后面会 shuffle）
            )

        remaining = set(task_ids)

        # 任务抽查列表（用于每轮抽样）
        remaining_list = task_ids[:]
        _shuffle_if(self.rng, remaining_list, self.cfg.randomized)

        while remaining:
            # 每轮抽查部分任务以估计 MRV
            candidates_to_check = list(remaining)
            if len(candidates_to_check) > self.cfg.completion_scan_tasks_per_iter:
                # 抽样检查
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

                # 粗略统计可行数量（只检查上限内的候选）
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
                        # early_accept 情况下：只要知道“有可行”就够了
                        if self.cfg.early_accept:
                            break

                if cnt <= 0 or pick is None:
                    continue

                if best_tid is None or cnt < best_feasible_cnt:
                    best_tid = tid
                    best_feasible_cnt = cnt
                    best_pick = pick

            if best_tid is None or best_pick is None:
                # 抽样没找到可行：再尝试一次全量扫描（仍带上限），避免漏掉
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
                        # completion-first fallback：尽量早结束且忙碌少
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
    """收益优先：按 task.priority 从高到低遍历任务，给每个任务找一个可行落点。

    优化点
    ----
    原实现是“while + 对 remaining 全量扫描 + 对每个任务全量扫描候选”，复杂度接近 O(T^2*C)。
    新实现改成“一次遍历任务”，每个任务最多检查 max_checks_per_task 个候选，复杂度 O(T*K)。
    """

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

        # 候选池
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

        # 按 priority 降序；同优先级可随机打散
        task_ids.sort(key=lambda tid: float(problem.tasks[tid].priority), reverse=True)
        if self.cfg.randomized:
            # 同优先级块内打散
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
                # profit-first：尽量早结束/忙碌少作为 tie-break
                key_fn=lambda aa, pr: (-pr, _finish_ts(aa), _busy(aa)),
            )
            if a is None:
                continue
            schedule.assignments.append(a)

        return schedule


class HeuristicTimelinessFirstScheduler(BaseSchedulerAlgorithm):
    """时效性优先：尽量让任务尽早执行（最小化 TM 的近似：更早的 sat_start 更好）。

    优化点
    ----
    原实现每轮对 remaining 全量扫描并为每个任务找最早可行候选，成本高。
    新实现：按“最早候选开始时间”对任务做一次预排序，然后单遍扫描插入，
    每个任务最多检查 max_checks_per_task 个候选。
    """

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
                # 近似：候选里最早的 sat_start_time（不检查可行性，仅用于排序）
                st = min(a.sat_start_time for a in cands)
                earliest_start[tid] = float(st.timestamp())
            else:
                earliest_start[tid] = float("inf")

        # 任务按最早开始时间升序；同一 bucket 可随机打散
        task_ids.sort(key=lambda tid: (earliest_start.get(tid, float("inf")), -float(problem.tasks[tid].priority)))
        if self.cfg.randomized:
            # 对近似同 start 的任务块打散（防止固定顺序）
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
                # timeliness：sat_start 越早越好；priority/忙碌作为次级
                key_fn=lambda aa, pr: (aa.sat_start_time, -pr, _busy(aa), _finish_ts(aa)),
            )
            if a is None:
                continue
            schedule.assignments.append(a)

        return schedule


class HeuristicBalanceFirstScheduler(BaseSchedulerAlgorithm):
    """均衡优先：基于“工作时长”尽量均匀分配（与你的 BD 指标口径一致）。

    优化点
    ----
    原实现每个候选都 copy workloads 并调用 obj.balance_score_from_workloads（非常慢）。
    新实现使用轻量策略：
    - 优先把任务分配给当前 workload 最小的卫星（更符合“工作时长均衡”直觉）；
    - 同时仍做可行性检查；
    - 每任务最多检查 max_checks_per_task 个候选；
    - tie-break：更早完成、更少 busy、更高 priority。
    """

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

        # 工作时长累计
        workloads: Dict[str, float] = {sid: 0.0 for sid in problem.satellites.keys()}

        # 候选池
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

        # 先按 priority 降序，防止“为均衡而不排任务”
        task_ids.sort(key=lambda tid: float(problem.tasks[tid].priority), reverse=True)
        _shuffle_if(self.rng, task_ids, self.cfg.randomized)

        for tid in task_ids:
            task = problem.tasks[tid]
            cands = cand_map.get(tid, [])
            if not cands:
                continue

            # 为该任务选择一个“能提升均衡”的候选
            def key_fn(a: Assignment, pr: float) -> Tuple[float, float, float, float]:
                # 越小越好：把任务放给当前 workload 更小的卫星
                wl = workloads.get(a.satellite_id, 0.0)
                # 加一点增量后的 workload
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
