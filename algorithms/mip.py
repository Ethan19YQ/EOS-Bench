# algorithms/mip.py
# -*- coding: utf-8 -*-
"""
完整 MILP 单入口求解（PuLP）。
支持：收益/完成率/均衡/多目标加权。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from copy import deepcopy
from datetime import timedelta
import os
import re
import tempfile
import uuid

from schedulers.scenario_loader import SchedulingProblem
from schedulers.engine import BaseSchedulerAlgorithm
from schedulers.constraint_model import ConstraintModel, Schedule, Assignment
from schedulers.transition_utils import compute_transition_time_agile, delta_g_between

from algorithms.candidate_pool import enumerate_task_candidates, assignments_conflict
from algorithms.objectives import ObjectiveWeights
from algorithms.random_utils import resolve_seed, make_rng, spawn_seeds


def _parse_cbc_gap_from_log(log_text: str) -> Optional[float]:
    """从 CBC 日志中解析相对 gap（0~1）。

    说明：
    - CBC 的日志格式在不同版本/不同停止条件下差异很大。
    - 有时不会出现显式 "Gap:" 行，而是只给出 "Objective value" 与 "Lower bound/Best possible"。
    因此这里采用：
    1) 优先解析显式 gap（取最后一次出现，避免早期阶段性 gap 误导）；
    2) 若没有显式 gap，则尝试解析 incumbent/bound 并计算相对 gap。
    """

    def _to_float(s: str) -> Optional[float]:
        s = s.strip()
        # 支持 "12.3%" 形式
        if s.endswith("%"):
            try:
                return float(s[:-1].strip()) / 100.0
            except Exception:
                return None
        try:
            return float(s)
        except Exception:
            return None

    # 1) 显式 gap：取最后一次出现
    gap_patterns = [
        r"Relative\s+gap\s*[:=]\s*([0-9.+-eE%]+)",
        r"\bGap\b\s*[:=]\s*([0-9.+-eE%]+)",
        r"\bgap\b\s*\(.*?\)\s*[:=]\s*([0-9.+-eE%]+)",
    ]
    for pat in gap_patterns:
        ms = list(re.finditer(pat, log_text, flags=re.IGNORECASE))
        if ms:
            v = _to_float(ms[-1].group(1))
            if v is not None:
                # 兜底裁剪
                if v < 0:
                    v = 0.0
                if v > 1.0:
                    # 有些 CBC 输出 gap 用百分数但没带 %，做保守处理
                    if v <= 100.0:
                        v = v / 100.0
                    else:
                        v = 1.0
                return float(v)

    # 2) 无显式 gap：用 incumbent 与 bound 计算
    # 常见文本片段：
    #  - "Objective value:                123.456"
    #  - "Lower bound:                   120.000"
    #  - "Best possible:                120.000"
    inc_patterns = [
        r"Objective\s+value\s*[:=]\s*([0-9.+-eE]+)",
        r"Objective\s*[:=]\s*([0-9.+-eE]+)",
    ]
    bound_patterns = [
        r"Lower\s+bound\s*[:=]\s*([0-9.+-eE]+)",
        r"Best\s+possible\s*[:=]\s*([0-9.+-eE]+)",
        r"Best\s+bound\s*[:=]\s*([0-9.+-eE]+)",
    ]
    incumbent = None
    for pat in inc_patterns:
        ms = list(re.finditer(pat, log_text, flags=re.IGNORECASE))
        if ms:
            incumbent = _to_float(ms[-1].group(1))
            if incumbent is not None:
                break
    bound = None
    for pat in bound_patterns:
        ms = list(re.finditer(pat, log_text, flags=re.IGNORECASE))
        if ms:
            bound = _to_float(ms[-1].group(1))
            if bound is not None:
                break
    if incumbent is not None and bound is not None:
        denom = abs(incumbent) if abs(incumbent) > 1e-9 else 1.0
        g = abs(incumbent - bound) / denom
        if g < 0:
            g = 0.0
        if g > 1.0:
            g = 1.0
        return float(g)

    return None



def _extract_angles_first(sat_angles: object) -> Optional[Dict[str, float]]:
    if sat_angles is None:
        return None
    if isinstance(sat_angles, dict):
        if "pitch_angles" in sat_angles:
            try:
                return {"pitch": float(sat_angles["pitch_angles"][0]),
                        "yaw": float(sat_angles["yaw_angles"][0]),
                        "roll": float(sat_angles["roll_angles"][0])}
            except Exception:
                return None
        if "pitch_angle" in sat_angles:
            try:
                return {"pitch": float(sat_angles["pitch_angle"]),
                        "yaw": float(sat_angles["yaw_angle"]),
                        "roll": float(sat_angles["roll_angle"])}
            except Exception:
                return None
    return None

def _extract_angles_last(sat_angles: object) -> Optional[Dict[str, float]]:
    if sat_angles is None:
        return None
    if isinstance(sat_angles, dict):
        if "pitch_angles" in sat_angles:
            try:
                return {"pitch": float(sat_angles["pitch_angles"][-1]),
                        "yaw": float(sat_angles["yaw_angles"][-1]),
                        "roll": float(sat_angles["roll_angles"][-1])}
            except Exception:
                return None
        if "pitch_angle" in sat_angles:
            try:
                return {"pitch": float(sat_angles["pitch_angle"]),
                        "yaw": float(sat_angles["yaw_angle"]),
                        "roll": float(sat_angles["roll_angle"])}
            except Exception:
                return None
    return None

def _transition_time_s(
    problem: SchedulingProblem,
    sat_id: str,
    prev_a,
    next_a,
    agility_profile: str = "Standard-Agility",
    non_agile_transition_s: float = 10.0,
) -> float:
    sat = problem.satellites.get(sat_id)
    if sat is None:
        return 0.0
    if sat.maneuverability_type == "non_agile":
        return float(non_agile_transition_s)

    dg = delta_g_between(prev_a.sat_angles, next_a.sat_angles)
    if dg is None:
        return 11.66
    return float(compute_transition_time_agile(dg, agility_profile))

def _satellite_conflict_with_transition(problem: SchedulingProblem, a, b, agility_profile: str, non_agile_transition_s: float) -> bool:
    # basic overlap of any satellite occupied intervals (sat + maybe gs)
    if a.satellite_id != b.satellite_id:
        return False
    # 기존 overlap check: sat segments overlap
    if (a.sat_start_time < b.sat_end_time) and (b.sat_start_time < a.sat_end_time):
        return True
    # add transition gap constraint between observation segments
    if a.sat_start_time <= b.sat_start_time:
        trans = _transition_time_s(problem, a.satellite_id, a, b, agility_profile=agility_profile, non_agile_transition_s=non_agile_transition_s)
        return a.sat_end_time + timedelta(seconds=trans) > b.sat_start_time
    else:
        trans = _transition_time_s(problem, a.satellite_id, b, a, agility_profile=agility_profile, non_agile_transition_s=non_agile_transition_s)
        return b.sat_end_time + timedelta(seconds=trans) > a.sat_start_time
@dataclass
class MIPConfig:
    max_candidates_per_task: int = 128
    # MILP 求解时间上限（秒）。默认 3600 秒（1 小时）。
    time_limit_sec: int = 3600
    solver: str = "cbc"  # cbc / glpk
    weights: ObjectiveWeights = field(default_factory=lambda: ObjectiveWeights(1.0, 0.0, 0.0, 0.0))

    agility_profile: str = "Standard-Agility"  # 转换时间模型速度档
    non_agile_transition_s: float = 10.0  # 非敏捷固定转换时间（秒）

    # MIP 候选生成的随机种子：
    # - None（默认）：每次运行不同（但单次运行内部一致）；
    # - int：可复现。
    seed: Optional[int] = None

    # 下传窗口放置的随机采样次数（每个 comm window 额外采样多少种 downlink 起止）。
    # 注意：会被 max_candidates_per_task 截断。
    random_samples_per_window: int = 1

    # 打印关键调试信息（候选数、冲突对数、求解状态等）
    verbose: bool = True

    # 关键：避免“为了均衡等目标而选择空解”。
    # 只要存在候选解，就强制至少安排这么多任务（默认 1）。
    enforce_min_scheduled_tasks: int = 1

    # 分层目标时，主目标的放大系数：
    # 让“多安排任务/更高收益/更高完成率”的收益远大于均衡项带来的惩罚，
    # 从而避免模型通过“不安排任务”来提升均衡。
    primary_boost: float = 1000.0


class MIPScheduler(BaseSchedulerAlgorithm):
    def __init__(self, cfg: Optional[MIPConfig] = None) -> None:
        self.cfg = cfg or MIPConfig()

    def _get_solver(self, pulp, log_path: Optional[str] = None):
        """根据配置返回 PuLP 求解器实例。

        对 CBC：尽量启用 logPath 以便解析 gap（若 PuLP 版本支持）。
        """
        if self.cfg.solver.lower() == "glpk":
            return pulp.GLPK_CMD(msg=self.cfg.verbose, timeLimit=self.cfg.time_limit_sec)

        # CBC
        # 重要：当启用 logPath 时，PuLP 会将 CBC 输出重定向到日志文件。
        # 为了确保日志里包含 gap/bound/objective 等关键文本，这里强制 msg=True。
        kwargs = dict(msg=True if log_path else self.cfg.verbose, timeLimit=self.cfg.time_limit_sec)
        if log_path:
            # PuLP 新版本支持 logPath；老版本不支持会抛 TypeError，外层兜底。
            kwargs["logPath"] = log_path
        return pulp.PULP_CBC_CMD(**kwargs)

    def _build_candidate_map(self, problem: SchedulingProblem, cm: ConstraintModel) -> Dict[str, List[Assignment]]:
        """为每个任务生成候选集合。

        需求变更：候选生成不要固定顺序。
        - cfg.seed=None：每次运行自动生成一个 base_seed（运行内一致，运行间不同）；
        - cfg.seed=int：完全可复现。
        """
        base_seed = resolve_seed(self.cfg.seed)
        task_ids = list(problem.tasks.keys())
        seeds = spawn_seeds(base_seed, len(task_ids))

        cand_map: Dict[str, List[Assignment]] = {}
        for idx, tid in enumerate(task_ids):
            task = problem.tasks[tid]
            cand_list = enumerate_task_candidates(
                problem=problem,
                task=task,
                placement_mode=cm.placement_mode,
                downlink_duration_ratio=cm.downlink_duration_ratio,
                max_candidates=self.cfg.max_candidates_per_task,
                random_samples_per_window=int(self.cfg.random_samples_per_window),
                seed=int(seeds[idx]),
            )

            # 再额外打散一次候选顺序（不改变集合，只改变顺序）。
            # 这样即使 enumerate_task_candidates 内部有“earliest/center/latest优先”，
            # 在 max_candidates 截断前也能引入更多随机性。
            rng = make_rng(int(seeds[idx]) ^ 0x9E3779B9)
            rng.shuffle(cand_list)
            cand_map[tid] = cand_list

        if self.cfg.verbose:
            total_c = sum(len(v) for v in cand_map.values())
            zero = sum(1 for v in cand_map.values() if len(v) == 0)
            print(f"[MIP] candidate_map built: tasks={len(task_ids)}, total_candidates={total_c}, zero_candidate_tasks={zero}, base_seed={base_seed}")

        return cand_map

    def _precompute_conflicts(self, problem: SchedulingProblem, cand_map: Dict[str, List[Assignment]]) -> List[Tuple[Tuple[str, int], Tuple[str, int]]]:
        all_ids: List[Tuple[str, int]] = []
        for tid, lst in cand_map.items():
            for k in range(len(lst)):
                all_ids.append((tid, k))

        conflicts: List[Tuple[Tuple[str, int], Tuple[str, int]]] = []
        for i in range(len(all_ids)):
            tid_i, ki = all_ids[i]
            ai = cand_map[tid_i][ki]
            for j in range(i + 1, len(all_ids)):
                tid_j, kj = all_ids[j]
                if tid_i == tid_j:
                    continue
                aj = cand_map[tid_j][kj]
                if assignments_conflict(ai, aj) or _satellite_conflict_with_transition(problem, ai, aj, self.cfg.agility_profile, self.cfg.non_agile_transition_s):
                    conflicts.append(((tid_i, ki), (tid_j, kj)))
        return conflicts

    def search(self, problem: SchedulingProblem, constraint_model: ConstraintModel, initial_schedule: Schedule) -> Schedule:
        try:
            import pulp  # type: ignore
        except Exception as e:
            raise RuntimeError("MIP 需要安装 pulp：pip install pulp") from e

        if self.cfg.verbose:
            print(
                "[MIP] start: "
                f"solver={self.cfg.solver}, time_limit_sec={self.cfg.time_limit_sec}, "
                f"max_candidates_per_task={self.cfg.max_candidates_per_task}, "
                f"random_samples_per_window={self.cfg.random_samples_per_window}, seed={self.cfg.seed}"
            )

        cand_map = self._build_candidate_map(problem, constraint_model)
        if all(len(v) == 0 for v in cand_map.values()):
            return deepcopy(initial_schedule)

        conflicts = self._precompute_conflicts(problem, cand_map)
        if self.cfg.verbose:
            print(f"[MIP] precompute_conflicts: pairs={len(conflicts)}")

        # 统一：权重归一化后再使用。
        # 但为了避免“只优化均衡导致空解”，我们采用**分层/字典序风格**：
        # 先最大化(收益/完成率)的主目标；在主目标相同的情况下，再优化均衡。
        w = self.cfg.weights.normalized()
        wp, wc, wt, wb = w.w_profit, w.w_completion, w.w_timeliness, w.w_balance

        # 如果用户把 profit/completion 都设成 0（例如只想看均衡），
        # 按你的需求：仍然要尽可能安排任务，所以默认以 completion 作为主目标。
        if abs(wp) < 1e-12 and abs(wc) < 1e-12 and abs(wt) < 1e-12:
            wp, wc = 0.0, 1.0

        if self.cfg.verbose:
            print(
                "[MIP] objective_weights (normalized): "
                f"profit={wp:.4f}, completion={wc:.4f}, timeliness={wt:.4f}, balance={wb:.4f}; "
                f"primary_boost={self.cfg.primary_boost}"
            )

        sat_ids = list(problem.satellites.keys())
        n_sats = max(1, len(sat_ids))
        total_tasks = max(1, len(problem.tasks))
        total_priority = float(sum(t.priority for t in problem.tasks.values()))
        if total_priority <= 0:
            total_priority = 1.0
        total_required = float(sum(float(t.required_duration) for t in problem.tasks.values()))
        if total_required <= 0:
            total_required = 1.0

        model = pulp.LpProblem("SchedulingMIP", pulp.LpMaximize)

        x: Dict[Tuple[str, int], "pulp.LpVariable"] = {}
        z: Dict[str, "pulp.LpVariable"] = {}

        for tid, lst in cand_map.items():
            if not lst:
                continue
            z[tid] = pulp.LpVariable(f"z_{tid}", lowBound=0, upBound=1, cat=pulp.LpContinuous)
            for k in range(len(lst)):
                x[(tid, k)] = pulp.LpVariable(f"x_{tid}_{k}", lowBound=0, upBound=1, cat=pulp.LpBinary)

        if self.cfg.verbose:
            print(f"[MIP] variables: x(binary)={len(x)}, z(tasks)={len(z)}")

        for tid, lst in cand_map.items():
            if not lst:
                continue
            model += pulp.lpSum([x[(tid, k)] for k in range(len(lst))]) <= 1
            model += z[tid] == pulp.lpSum([x[(tid, k)] for k in range(len(lst))])

        if self.cfg.verbose:
            print(f"[MIP] variables: z(tasks)={len(z)}, x(candidates)={len(x)}")

        for (a, b) in conflicts:
            (tid_i, ki), (tid_j, kj) = a, b
            if (tid_i, ki) in x and (tid_j, kj) in x:
                model += x[(tid_i, ki)] + x[(tid_j, kj)] <= 1

        # per-orbit resource constraints (storage/power) -- 每圈约束
        orbit_constraint_cnt = 0
        for sid, sat in problem.satellites.items():
            max_storage = float(getattr(sat, "max_data_storage_GB", 0.0) or 0.0)
            max_power = float(getattr(sat, "max_power_W", 0.0) or 0.0)
            if max_storage <= 0 and max_power <= 0:
                continue

            orbit_items: Dict[int, List[Tuple[str, int, Assignment]]] = {}
            for tid, lst in cand_map.items():
                for k, cand in enumerate(lst):
                    if cand.satellite_id != sid:
                        continue
                    orbit = int(getattr(cand, "orbit_number", 0) or 0)
                    orbit_items.setdefault(orbit, []).append((tid, k, cand))

            for orbit, items in orbit_items.items():
                if max_storage > 0:
                    model += pulp.lpSum(
                        [float(getattr(c, "data_volume_GB", 0.0) or 0.0) * x[(tid, k)]
                         for tid, k, c in items if (tid, k) in x]
                    ) <= max_storage
                    orbit_constraint_cnt += 1

                if max_power > 0:
                    model += pulp.lpSum(
                        [float(getattr(c, "power_cost_W", 0.0) or 0.0) * x[(tid, k)]
                         for tid, k, c in items if (tid, k) in x]
                    ) <= max_power
                    orbit_constraint_cnt += 1

        if self.cfg.verbose:
            print(f"[MIP] constraints: conflict_pairs={len(conflicts)}, orbit_resource_constraints={orbit_constraint_cnt}")

        if self.cfg.verbose:
            print(f"[MIP] constraints: orbit_resource={orbit_constraint_cnt}")


        # per-satellite workload (seconds, observation duration)
        y: Dict[str, "pulp.LpVariable"] = {sid: pulp.LpVariable(f"y_{sid}", lowBound=0) for sid in sat_ids}
        for sid in sat_ids:
            terms = []
            for tid, lst in cand_map.items():
                for k, a in enumerate(lst):
                    if a.satellite_id == sid and (tid, k) in x:
                        dur = float((a.sat_end_time - a.sat_start_time).total_seconds())
                        terms.append(dur * x[(tid, k)])
            model += y[sid] == pulp.lpSum(terms)

        # total scheduled workload
        W = pulp.LpVariable("W", lowBound=0)
        model += W == pulp.lpSum([y[sid] for sid in sat_ids])

        T = pulp.LpVariable("T", lowBound=0)
        model += T == pulp.lpSum([z[tid] for tid in z])

        # 只要存在候选，就至少安排 N 个任务（默认 1），防止空解。
        if self.cfg.enforce_min_scheduled_tasks > 0 and len(z) > 0:
            min_n = min(int(self.cfg.enforce_min_scheduled_tasks), len(z))
            if min_n > 0:
                model += T >= float(min_n)

        mu = pulp.LpVariable("mu", lowBound=0)
        model += n_sats * mu == W

        d: Dict[str, "pulp.LpVariable"] = {sid: pulp.LpVariable(f"d_{sid}", lowBound=0) for sid in sat_ids}
        for sid in sat_ids:
            model += d[sid] >= y[sid] - mu
            model += d[sid] >= mu - y[sid]

        profit_norm = (1.0 / total_priority) * pulp.lpSum([float(problem.tasks[tid].priority) * z[tid] for tid in z])
        completion_norm = (1.0 / float(total_tasks)) * pulp.lpSum([z[tid] for tid in z])
        imbalance_norm = (1.0 / (2.0 * float(total_required))) * pulp.lpSum([d[sid] for sid in sat_ids])

        # Timeliness（时效性）：与 evaluation_metrics.TM 同口径
        Tsec = float((problem.end_time - problem.start_time).total_seconds())
        if Tsec <= 0:
            Tsec = 1.0
        ts = problem.start_time

        delay_sec: Dict[Tuple[str, int], float] = {}
        for tid, lst in cand_map.items():
            for k, a in enumerate(lst):
                ds = float((a.sat_start_time - ts).total_seconds())
                if ds < 0:
                    ds = 0.0
                delay_sec[(tid, k)] = ds

        schedulable = list(z.keys())
        unschedulable_count = int(total_tasks - len(schedulable))

        tm_numer = pulp.lpSum([delay_sec[(tid, k)] * x[(tid, k)] for (tid, k) in x]) \
            + Tsec * float(unschedulable_count) \
            + Tsec * (float(len(schedulable)) - pulp.lpSum([z[tid] for tid in schedulable]))

        tm_expr = (1.0 / (Tsec * float(total_tasks))) * tm_numer
        timeliness_score = 1.0 - tm_expr  # 越大越好

        primary = wp * profit_norm + wc * completion_norm + wt * timeliness_score

        if self.cfg.verbose:
            print(
                "[MIP] objective weights(normalized): "
                f"profit={wp:.3f}, completion={wc:.3f}, timeliness={wt:.3f}, balance={wb:.3f}; "
                f"primary_boost={self.cfg.primary_boost:.1f}"
            )

        # 分层目标：主目标优先，均衡作为次目标（除非 wb=0）。
        # 通过 primary_boost 放大主目标，保证不会为了均衡而牺牲“安排任务/收益/完成率”。
        if wb > 0:
            model += self.cfg.primary_boost * primary - wb * imbalance_norm
        else:
            model += primary

        # CBC gap: 通过 logPath 解析（若可用）。
        log_path: Optional[str] = None
        if self.cfg.solver.lower() != "glpk":
            # 仅 CBC/COIN-OR 系列尝试解析 gap
            try:
                # 并行安全：每次求解创建唯一日志文件，避免多个进程写同一 log 导致 CBC 启动失败
                uid = uuid.uuid4().hex
                fd, log_path = tempfile.mkstemp(prefix=f"cbc_{os.getpid()}_{uid}_", suffix=".log")
                os.close(fd)
            except Exception:
                log_path = None

        # 先尝试带 logPath（便于解析 gap）；若因日志文件/并行等原因导致 CBC 启动失败，则降级重试
        try:
            try:
                solver = self._get_solver(pulp, log_path=log_path)
            except TypeError:
                # 兼容老版本 PuLP：不支持 logPath
                solver = self._get_solver(pulp, log_path=None)
                log_path = None

            model.solve(solver)

        except Exception as e:
            # 某些 Windows/并行场景下，CBC 可能因 logPath/权限/资源等原因启动失败
            print(f"[MIP][WARN] CBC solve failed (logPath={log_path}). retry without logPath. err={e}", flush=True)
            solver = self._get_solver(pulp, log_path=None)
            log_path = None
            model.solve(solver)

        # 求解状态（用于 gap 兜底）
        try:
            solve_status = pulp.LpStatus[model.status]
        except Exception:
            solve_status = str(model.status)

        # 尝试解析 gap
        mip_gap: Optional[float] = None
        if log_path and os.path.exists(log_path):
            try:
                text = open(log_path, "r", encoding="utf-8", errors="ignore").read()
                mip_gap = _parse_cbc_gap_from_log(text)
            except Exception:
                mip_gap = None
            finally:
                try:
                    os.remove(log_path)
                except Exception:
                    pass

        # 若已证明最优但日志没有 gap 字段，则 gap=0
        if mip_gap is None and str(solve_status).lower() == "optimal":
            mip_gap = 0.0

        if self.cfg.verbose:
            status = solve_status
            try:
                objv = float(pulp.value(model.objective) or 0.0)
            except Exception:
                objv = 0.0
            if mip_gap is None:
                print(f"[MIP] solved: status={status}, objective={objv:.6f}")
            else:
                print(f"[MIP] solved: status={status}, objective={objv:.6f}, gap={mip_gap:.6g}")

        chosen: List[Assignment] = []
        for (tid, k), var in x.items():
            if float(var.value() or 0.0) > 0.5:
                chosen.append(cand_map[tid][k])

        chosen.sort(key=lambda a: a.sat_start_time)
        schedule = Schedule()
        if mip_gap is not None:
            schedule.metadata["mip_gap"] = float(mip_gap)
        for a in chosen:
            if constraint_model.is_feasible_assignment(a, schedule):
                schedule.assignments.append(a)

        if self.cfg.verbose:
            if len(chosen) != len(schedule.assignments):
                print(f"[MIP] repair: chosen={len(chosen)} -> feasible_kept={len(schedule.assignments)}")
            else:
                print(f"[MIP] result: assignments={len(schedule.assignments)}")

        return schedule
