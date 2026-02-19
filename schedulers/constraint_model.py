# -*- coding: utf-8 -*-
"""
"""

from __future__ import annotations

from dataclasses import dataclass, field

from schedulers.transition_utils import compute_transition_time_agile, delta_g_between
from typing import List, Dict, Optional, Protocol
from datetime import datetime, timedelta

from .scenario_loader import (
    SchedulingProblem,
    SchedulingTask,
    TaskWindow,
    CommWindow,
)


# ==============================
# 1. 基础结构：Assignment / Schedule
#    Basic Structures: Assignment / Schedule
# ==============================

@dataclass
class Assignment:
    """
    单个任务的调度结果：
    Scheduling result for a single task:

    - task_id: 任务 ID； / Task ID;
    - satellite_id: 执行该任务的卫星 ID； / ID of the satellite executing the task;
    - sat_start_time / sat_end_time: 卫星执行任务（观测）的时间段； / Time period for the satellite to execute the task (observation);
    - sat_window_id: 使用的观测窗口 ID； / ID of the observation window used;

    - ground_station_id: 地面站 ID（如果有地面站）； / Ground station ID (if there is one);
    - gs_start_time / gs_end_time: 地面站接收数传的时间段； / Time period for the ground station to receive data transmission;
    - gs_window_id: 使用的通信窗口 ID。 / ID of the communication window used.

    注意：
    Note:
    - 当场景没有地面站时，ground_station_id 及相关字段为 None。
      When the scenario has no ground stations, ground_station_id and related fields are None.
    """
    task_id: str
    satellite_id: str
    sat_start_time: datetime
    sat_end_time: datetime
    sat_window_id: int
    sensor_id: str = ""
    orbit_number: int = 0

    # 资源消耗（用于每圈约束），在候选生成时预计算填充
    # Resource consumption (used for per-orbit constraints), pre-calculated and filled during candidate generation
    data_volume_GB: float = 0.0
    power_cost_W: float = 0.0

    sat_angles: Optional[object] = None  # 任务执行角度数据（敏捷为切片，非敏捷为单组） / Task execution angle data (slice for agile, single set for non-agile)

    ground_station_id: Optional[str] = None
    gs_start_time: Optional[datetime] = None
    gs_end_time: Optional[datetime] = None
    gs_window_id: Optional[int] = None


@dataclass
class Schedule:
    """调度方案：一组 Assignment。 / Scheduling plan: a set of Assignments."""
    assignments: List[Assignment] = field(default_factory=list)

    # 运行过程中的附加信息（不影响调度语义）。
    # Additional information during execution (does not affect scheduling semantics).
    # 例如：MIP 求解器 gap、求解状态、调试统计等。
    # For example: MIP solver gap, solution status, debugging statistics, etc.
    metadata: dict = field(default_factory=dict)

    def get_assignments_for_satellite(self, satellite_id: str) -> List[Assignment]:
        return [a for a in self.assignments if a.satellite_id == satellite_id]

    def get_assignments_for_task(self, task_id: str) -> List[Assignment]:
        return [a for a in self.assignments if a.task_id == task_id]

    def get_assignments_for_ground_station(self, ground_station_id: str) -> List[Assignment]:
        return [a for a in self.assignments if a.ground_station_id == ground_station_id]

    @property
    def assigned_task_ids(self) -> List[str]:
        return list({a.task_id for a in self.assignments})


# ==============================
# 2. 窗口内部时间安排策略
#    Time Placement Strategy within a Window
# ==============================

class TimePlacementStrategy:
    """
    窗口内部时间安排策略（可扩展）：
    Time placement strategy within a window (extensible):
    - 'earliest'：紧前安排； / Place as early as possible;
    - 'center'：居中安排； / Place in the center;
    - 未来可以扩展 'latest' 等策略。 / In the future, strategies like 'latest' can be extended.
    """

    @staticmethod
    def place(
        window_start: datetime,
        window_end: datetime,
        required_duration: float,
        mode: str = "earliest",
    ) -> Optional[tuple[datetime, datetime]]:
        """
        在给定窗口内，根据策略选择任务的开始 / 结束时间。
        Select the start/end time of the task within the given window based on the strategy.
        返回 (start_time, end_time) 或 None（无法放下）。
        Returns (start_time, end_time) or None (cannot fit).
        """
        total_window = (window_end - window_start).total_seconds()
        if required_duration > total_window:
            return None

        if mode == "earliest":
            start = window_start
            end = start + timedelta(seconds=required_duration)
            return start, end

        if mode == "center":
            slack = total_window - required_duration
            offset = slack / 2.0
            start = window_start + timedelta(seconds=offset)
            end = start + timedelta(seconds=required_duration)
            return start, end

        # 默认回退到紧前安排
        # Default fallback to earliest placement
        start = window_start
        end = start + timedelta(seconds=required_duration)
        return start, end


# ==============================
# 3. 约束模型：ConstraintModel
#    Constraint Model: ConstraintModel
# ==============================

class ConstraintModel:
    """
    调度约束与目标函数的统一封装。
    Unified encapsulation of scheduling constraints and objective functions.

    主要方法 / Main methods:
    - is_feasible_assignment: 判断在当前 Schedule 上，添加一个 Assignment 是否满足约束；
      / Check if adding an Assignment to the current Schedule satisfies constraints;
    - build_feasible_assignment_for_task: 为单个任务构建一个可行调度方案（观测 + 数传）；
      / Build a feasible scheduling plan (observation + transmission) for a single task;
    - build_initial_schedule: 基于简单启发式构造初始解；
      / Construct an initial solution based on simple heuristics;
    - evaluate: 计算 Schedule 的目标函数值（越大越好）。
      / Calculate the objective function value of the Schedule (larger is better).
    """

    def __init__(
        self,
        problem: SchedulingProblem,
        placement_mode: str = "earliest",
        unassigned_penalty: float = 1000.0,
        downlink_duration_ratio: float = 1.0,
        agility_profile: str = "Standard-Agility",
        non_agile_transition_s: float = 10.0,
    ) -> None:
        """
        参数说明 / Parameter Description
        --------
        placement_mode:
            窗口内部时间安排策略： / Time placement strategy within a window:
              - "earliest"：窗口内尽早安排； / Place as early as possible within the window;
              - "center"：居中安排。 / Place in the center.

        unassigned_penalty:
            每个未分配任务的惩罚值。 / Penalty value for each unassigned task.

        downlink_duration_ratio:
            数传持续时间 = 观测持续时间 * ratio； / Data transmission duration = observation duration * ratio;
            如果不想区分，可以设为 1.0。 / If you do not want to differentiate, it can be set to 1.0.
        """
        self.problem = problem
        self.placement_mode = placement_mode
        self.unassigned_penalty = unassigned_penalty
        self.downlink_duration_ratio = downlink_duration_ratio

        self.has_ground_stations = len(problem.ground_stations) > 0

    # ---------- 约束检查 / Constraint Checking ----------

    @staticmethod
    def _intervals_overlap(a_start: datetime, a_end: datetime,
                           b_start: datetime, b_end: datetime) -> bool:
        """判断两个时间区间是否重叠 / Check if two time intervals overlap"""
        return not (a_end <= b_start or a_start >= b_end)


    def _estimate_data_volume_GB(self, satellite_id: str, sensor_id: str, sat_start: datetime, sat_end: datetime) -> float:
        """容量消耗：data_rate_Mbps * duration_s -> GB。 / Capacity consumption: data_rate_Mbps * duration_s -> GB."""
        sat = self.problem.satellites.get(satellite_id)
        if sat is None:
            return 0.0
        spec = sat.sensors.get(sensor_id)
        if spec is None:
            return 0.0
        dur = max(0.0, (sat_end - sat_start).total_seconds())
        mbits = float(spec.data_rate_Mbps) * float(dur)
        return mbits / (8.0 * 1024.0)

    def _estimate_power_cost_W(self, satellite_id: str, sensor_id: str) -> float:
        """能量/功率消耗：按任务次数计，每个任务固定消耗 power_consumption_W。
           / Energy/Power consumption: Calculated per task, each task consumes a fixed power_consumption_W."""
        sat = self.problem.satellites.get(satellite_id)
        if sat is None:
            return 0.0
        spec = sat.sensors.get(sensor_id)
        if spec is None:
            return 0.0
        return float(spec.power_consumption_W)

    @staticmethod
    def _extract_angles_first(sat_angles: object) -> Optional[Dict[str, float]]:
        """从 sat_angles 中取“起始角度”。支持 dict(list) 与 dict(scalar)。
           / Extract the "starting angle" from sat_angles. Supports dict(list) and dict(scalar)."""
        if sat_angles is None:
            return None
        if isinstance(sat_angles, dict):
            # agile style: pitch_angles/yaw_angles/roll_angles
            if "pitch_angles" in sat_angles:
                try:
                    return {
                        "pitch": float(sat_angles["pitch_angles"][0]),
                        "yaw": float(sat_angles["yaw_angles"][0]),
                        "roll": float(sat_angles["roll_angles"][0]),
                    }
                except Exception:
                    return None
            # non-agile style: pitch_angle/yaw_angle/roll_angle
            if "pitch_angle" in sat_angles:
                try:
                    return {
                        "pitch": float(sat_angles["pitch_angle"]),
                        "yaw": float(sat_angles["yaw_angle"]),
                        "roll": float(sat_angles["roll_angle"]),
                    }
                except Exception:
                    return None
        return None

    @staticmethod
    def _extract_angles_last(sat_angles: object) -> Optional[Dict[str, float]]:
        """从 sat_angles 中取“结束角度”（最后一个）。
           / Extract the "ending angle" (the last one) from sat_angles."""
        if sat_angles is None:
            return None
        if isinstance(sat_angles, dict):
            if "pitch_angles" in sat_angles:
                try:
                    return {
                        "pitch": float(sat_angles["pitch_angles"][-1]),
                        "yaw": float(sat_angles["yaw_angles"][-1]),
                        "roll": float(sat_angles["roll_angles"][-1]),
                    }
                except Exception:
                    return None
            if "pitch_angle" in sat_angles:
                # non-agile: scalar same for start/end
                try:
                    return {
                        "pitch": float(sat_angles["pitch_angle"]),
                        "yaw": float(sat_angles["yaw_angle"]),
                        "roll": float(sat_angles["roll_angle"]),
                    }
                except Exception:
                    return None
        return None

    def _transition_time_s(self, satellite_id: str, prev_a: Assignment, next_a: Assignment) -> float:
        """相邻任务姿态转换时间（秒）。 / Attitude transition time between adjacent tasks (in seconds).

        - non_agile：固定 non_agile_transition_s（由 main/main_scheduler 配置） / Fixed non_agile_transition_s (configured by main/main_scheduler)
        - agile：按用户给定分段模型： / According to the user-provided piecewise model:
            Δg = |Δroll| + |Δpitch| + |Δyaw|
            Trans(Δg) 见 / see schedulers/transition_utils.py
        """
        sat = self.problem.satellites.get(satellite_id)
        if sat is None:
            return 0.0

        if sat.maneuverability_type == "non_agile":
            return float(self.non_agile_transition_s)

        dg = delta_g_between(prev_a.sat_angles, next_a.sat_angles)
        if dg is None:
            # 若缺角度数据，退化为最小常数（保守） / If angle data is missing, degenerate to minimum constant (conservative)
            return 11.66

        return float(compute_transition_time_agile(dg, self.agility_profile))

    def _check_per_orbit_resource(self, assignment: Assignment, schedule: Schedule) -> bool:
        """检查每圈容量/能量限制。 / Check per-orbit capacity/energy limits."""
        sat = self.problem.satellites.get(assignment.satellite_id)
        if sat is None:
            return True

        max_storage = float(getattr(sat, "max_data_storage_GB", 0.0) or 0.0)
        max_power = float(getattr(sat, "max_power_W", 0.0) or 0.0)
        # 若上限为 0，则视作不限制（兼容旧场景） / If the upper limit is 0, it is considered unrestricted (compatible with old scenarios)
        if max_storage <= 0 and max_power <= 0:
            return True

        orbit = int(getattr(assignment, "orbit_number", 0) or 0)

        cur_storage = 0.0
        cur_power = 0.0
        for a in schedule.get_assignments_for_satellite(assignment.satellite_id):
            if int(getattr(a, "orbit_number", 0) or 0) != orbit:
                continue
            cur_storage += float(getattr(a, "data_volume_GB", 0.0) or 0.0)
            cur_power += float(getattr(a, "power_cost_W", 0.0) or 0.0)

        cur_storage += float(getattr(assignment, "data_volume_GB", 0.0) or 0.0)
        cur_power += float(getattr(assignment, "power_cost_W", 0.0) or 0.0)

        if max_storage > 0 and cur_storage - max_storage > 1e-9:
            return False
        if max_power > 0 and cur_power - max_power > 1e-9:
            return False
        return True

    def is_feasible_assignment(self, assignment: Assignment, schedule: Schedule) -> bool:
        """检查在当前 schedule 上增加 assignment 是否可行。
           / Check if adding an assignment to the current schedule is feasible."""

        task = self.problem.tasks[assignment.task_id]

        # 1) 任务只能被分配一次 / A task can only be assigned once
        if schedule.get_assignments_for_task(assignment.task_id):
            return False

        # 2) 卫星观测段长度必须满足任务需求 / The length of the satellite observation segment must meet the task requirements
        actual_duration = (assignment.sat_end_time - assignment.sat_start_time).total_seconds()
        if actual_duration + 1e-6 < task.required_duration:
            return False

        # 3) 如有地面站：检查数传段基本合法性 / If there is a ground station: check basic validity of the transmission segment
        if self.has_ground_stations:
            if assignment.ground_station_id is None:
                return False
            if assignment.gs_start_time is None or assignment.gs_end_time is None:
                return False
            if assignment.gs_start_time <= assignment.sat_end_time:
                # 数传必须在观测完成之后 / Transmission must be after observation is completed
                return False

        # 4) 同一颗卫星上不能与其它任务的任何工作段（观测/数传）重叠
        # / On the same satellite, it cannot overlap with any working segment (observation/transmission) of other tasks
        sat_assigns = schedule.get_assignments_for_satellite(assignment.satellite_id)
        for a in sat_assigns:
            # 相邻任务姿态转换时间约束（只对卫星观测段）
            # / Attitude transition time constraint between adjacent tasks (only for satellite observation segments)
            if assignment.sat_start_time >= a.sat_start_time:
                trans_s = self._transition_time_s(assignment.satellite_id, a, assignment)
                if a.sat_end_time + timedelta(seconds=trans_s) > assignment.sat_start_time:
                    return False
            else:
                trans_s = self._transition_time_s(assignment.satellite_id, assignment, a)
                if assignment.sat_end_time + timedelta(seconds=trans_s) > a.sat_start_time:
                    return False

            # a 的卫星观测段 / a's satellite observation segment
            if self._intervals_overlap(
                assignment.sat_start_time, assignment.sat_end_time,
                a.sat_start_time, a.sat_end_time,
            ):
                return False
            # a 的数传段 / a's transmission segment
            if a.gs_start_time is not None and a.gs_end_time is not None:
                if self._intervals_overlap(
                    assignment.sat_start_time, assignment.sat_end_time,
                    a.gs_start_time, a.gs_end_time,
                ):
                    return False
                if assignment.gs_start_time is not None and assignment.gs_end_time is not None:
                    if self._intervals_overlap(
                        assignment.gs_start_time, assignment.gs_end_time,
                        a.gs_start_time, a.gs_end_time,
                    ):
                        return False
                # assignment 的数传段与 a 的观测段 / assignment's transmission segment and a's observation segment
                if assignment.gs_start_time is not None and assignment.gs_end_time is not None:
                    if self._intervals_overlap(
                        assignment.gs_start_time, assignment.gs_end_time,
                        a.sat_start_time, a.sat_end_time,
                    ):
                        return False

        # 5) 地面站约束：同一时间只能服务一个任务 / Ground station constraint: can only serve one task at the same time
        if self.has_ground_stations and assignment.ground_station_id is not None:
            gs_assigns = schedule.get_assignments_for_ground_station(assignment.ground_station_id)
            for a in gs_assigns:
                if a.gs_start_time is None or a.gs_end_time is None:
                    continue
                if self._intervals_overlap(
                    assignment.gs_start_time, assignment.gs_end_time,
                    a.gs_start_time, a.gs_end_time,
                ):
                    return False

        # 6) 每圈容量/能量限制 / Per-orbit capacity/energy constraints
        if not self._check_per_orbit_resource(assignment, schedule):
            return False


        return True

    # ---------- 为单任务构建一个可行 assignment / Build a feasible assignment for a single task ----------

    def build_feasible_assignment_for_task(
            self,
            task: SchedulingTask,
            schedule: Schedule,
            randomized: bool = False,
            rng=None,
    ) -> Optional[Assignment]:
        import random
        import math
        from datetime import timedelta

        # 随机源：传入 rng 时使用 rng；否则退回全局 random 模块（兼容旧调用）
        # Random source: Use rng if passed; otherwise fallback to global random module (compatible with old calls)
        _rnd = rng if rng is not None else random


        if task.required_duration <= 0 or not task.windows:
            return None

        def place_in_window(ws, we, dur) -> Optional[tuple[datetime, datetime]]:
            total = (we - ws).total_seconds()
            if dur > total:
                return None
            if not randomized:
                return TimePlacementStrategy.place(ws, we, dur, mode=self.placement_mode)

            # randomized=True：在可行范围内随机采样开始时间 / Sample start time randomly within feasible range
            slack = total - dur
            if slack <= 1e-9:
                start = ws
            else:
                start = ws + timedelta(seconds=_rnd.random() * slack)
            end = start + timedelta(seconds=dur)
            return start, end
        def place_observation_in_task_window(w: TaskWindow) -> Optional[tuple[datetime, datetime, Optional[list]]]:
            """根据 duration_s 与 time_step 在可见窗口内选取真正执行子窗口，并产出角度数据。
               / Select actual execution sub-window within visible window based on duration_s and time_step, and output angle data."""
            sat = self.problem.satellites.get(w.satellite_id)
            sat_type = getattr(sat, "maneuverability_type", "agile") if sat is not None else "agile"

            total = (w.end_time - w.start_time).total_seconds()
            dur = task.required_duration
            step = float(getattr(w, "time_step", 1.0) or 1.0)
            if dur <= 0 or dur > total:
                return None

            max_offset = total - dur
            max_k = int(math.floor((max_offset + 1e-9) / step))
            if max_k < 0:
                return None

            # 非敏捷：只能取窗口中间那一段（并对齐到 time_step）
            # Non-agile: can only take the middle segment of the window (and align to time_step)
            if sat_type == "non_agile":
                center_offset = max_offset / 2.0
                k = int(round(center_offset / step))
                k = max(0, min(max_k, k))
            else:
                # 敏捷：可按步长在窗口内任意选择；若 randomized=True 则随机选，否则按 placement_mode 选
                # Agile: can freely select within window according to step size; randomly if randomized=True, else by placement_mode
                if randomized:
                    k = _rnd.randrange(0, max_k + 1)
                else:
                    if self.placement_mode == "center":
                        k = int(round((max_offset / 2.0) / step))
                    elif self.placement_mode == "latest":
                        k = max_k
                    else:  # earliest / default
                        k = 0
                    k = max(0, min(max_k, k))

            start = w.start_time + timedelta(seconds=k * step)
            end = start + timedelta(seconds=dur)

            # 角度数据：敏捷 -> 取子窗口对应的切片；非敏捷 -> 单组数据
            # Angle data: Agile -> take slice corresponding to sub-window; Non-agile -> single data set
            def _slice_angles_payload(payload, start_idx: int, count: int):
                if payload is None:
                    return None
                if isinstance(payload, list):
                    return payload[start_idx : start_idx + count]
                if isinstance(payload, dict):
                    out = {}
                    for kk, vv in payload.items():
                        if isinstance(vv, (list, dict)):
                            out[kk] = _slice_angles_payload(vv, start_idx, count)
                        else:
                            out[kk] = vv
                    return out
                return payload

            if sat_type == "non_agile":
                angles = getattr(w, "non_agile_data", None)
            else:
                ad = getattr(w, "agile_data", None)
                # 用 ceil 保证覆盖 duration（避免 round 导致短一格）
                # Use ceil to ensure duration is covered (avoiding round causing it to be one step short)
                n = max(1, int(math.ceil((dur / step) - 1e-9)))
                angles = _slice_angles_payload(ad, k, n)

            return start, end, angles


        obs_windows = list(task.windows)
        if randomized:
            _rnd.shuffle(obs_windows)
        else:
            obs_windows.sort(key=lambda w: w.start_time)

        if not self.has_ground_stations:
            for w in obs_windows:
                obs_sel = place_observation_in_task_window(w)
                if obs_sel is None:
                    continue
                sat_start, sat_end, sat_angles = obs_sel
                assignment = Assignment(
                    task_id=task.id,
                    satellite_id=w.satellite_id,
                    sat_start_time=sat_start,
                    sat_end_time=sat_end,
                    sat_window_id=w.window_id,
                    sensor_id=getattr(w, "sensor_id", "") or "",
                    orbit_number=int(getattr(w, "orbit_number", 0) or 0),
                    data_volume_GB=self._estimate_data_volume_GB(w.satellite_id, getattr(w, "sensor_id", "") or "", sat_start, sat_end),
                    power_cost_W=self._estimate_power_cost_W(w.satellite_id, getattr(w, "sensor_id", "") or ""),
                    sat_angles=sat_angles,
                )
                if self.is_feasible_assignment(assignment, schedule):
                    return assignment
            return None

        downlink_duration = task.required_duration * self.downlink_duration_ratio
        comm_windows = self.problem.comm_windows

        for w in obs_windows:
            obs_sel = place_observation_in_task_window(w)
            if obs_sel is None:
                continue
            sat_start, sat_end, sat_angles = obs_sel

            candidate_cw = [cw for cw in comm_windows if cw.satellite_id == w.satellite_id]
            if not candidate_cw:
                continue

            if randomized:
                _rnd.shuffle(candidate_cw)
            else:
                candidate_cw.sort(key=lambda c: c.start_time)

            for cw in candidate_cw:
                earliest_start = max(cw.start_time, sat_end)
                latest_end = cw.end_time
                if (latest_end - earliest_start).total_seconds() < downlink_duration:
                    continue

                placement_dl = place_in_window(earliest_start, cw.end_time, downlink_duration)
                if placement_dl is None:
                    continue
                gs_start, gs_end = placement_dl

                assignment = Assignment(
                    task_id=task.id,
                    satellite_id=w.satellite_id,
                    sat_start_time=sat_start,
                    sat_end_time=sat_end,
                    sat_window_id=w.window_id,
                    sensor_id=getattr(w, "sensor_id", "") or "",
                    orbit_number=int(getattr(w, "orbit_number", 0) or 0),
                    data_volume_GB=self._estimate_data_volume_GB(w.satellite_id, getattr(w, "sensor_id", "") or "", sat_start, sat_end),
                    power_cost_W=self._estimate_power_cost_W(w.satellite_id, getattr(w, "sensor_id", "") or ""),
                    sat_angles=sat_angles,
                    ground_station_id=cw.ground_station_id,
                    gs_start_time=gs_start,
                    gs_end_time=gs_end,
                    gs_window_id=cw.window_id,
                )

                if self.is_feasible_assignment(assignment, schedule):
                    return assignment

        return None

    # ---------- 初始解生成（简单贪心） / Initial solution generation (simple greedy) ----------

    def build_initial_schedule(self) -> Schedule:
        """
        构建一个简单的可行初始解：
        Build a simple feasible initial solution:
        - 任务按优先级从高到低排序； / Tasks sorted by priority from high to low;
        - 对每个任务调用 build_feasible_assignment_for_task； / Call build_feasible_assignment_for_task for each task;
        - 若找到可行安排，则加入 schedule。 / If a feasible arrangement is found, add to schedule.
        """
        tasks_sorted = sorted(
            self.problem.tasks.values(),
            key=lambda t: t.priority,
            reverse=True,
        )

        schedule = Schedule()

        for task in tasks_sorted:
            assignment = self.build_feasible_assignment_for_task(
                task=task,
                schedule=schedule,
                randomized=False,
            )
            if assignment is not None:
                schedule.assignments.append(assignment)
            # 未找到可行安排则该任务保持未分配，由目标函数惩罚
            # If no feasible arrangement is found, the task remains unassigned and is penalized by the objective function

        return schedule

    # ---------- 目标函数 / Objective Function ----------

    def evaluate(self, schedule: Schedule) -> float:
        """
        目标函数（越大越好）：
        Objective function (larger is better):
        - 已分配任务的优先级之和； / Sum of priorities of assigned tasks;
        - 未分配任务数量 * unassigned_penalty 作为惩罚。 / Number of unassigned tasks * unassigned_penalty as penalty.
        """
        assigned_task_ids = set(schedule.assigned_task_ids)
        total_priority = sum(
            self.problem.tasks[tid].priority for tid in assigned_task_ids
        )

        total_tasks = len(self.problem.tasks)
        unassigned = total_tasks - len(assigned_task_ids)
        penalty = unassigned * self.unassigned_penalty

        return total_priority - penalty