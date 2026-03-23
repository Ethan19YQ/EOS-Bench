# -*- coding: utf-8 -*-
"""
schedulers/constraint_model.py

Main functionality:
This module defines the core scheduling data structures and constraint model,
including Assignment and Schedule, time placement strategies within windows,
feasibility checking, per-orbit resource validation, feasible assignment
construction for individual tasks, initial schedule generation, and basic
objective evaluation.
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
# 1. Basic structures: Assignment / Schedule
# ==============================

@dataclass
class Assignment:

    task_id: str
    satellite_id: str
    sat_start_time: datetime
    sat_end_time: datetime
    sat_window_id: int
    sensor_id: str = ""
    orbit_number: int = 0

    # Resource consumption used for per-orbit constraints,
    # pre-calculated and filled during candidate generation
    data_volume_GB: float = 0.0
    power_cost_W: float = 0.0

    sat_angles: Optional[object] = None  # Task execution angle data: slice for agile, single set for non-agile

    ground_station_id: Optional[str] = None
    gs_start_time: Optional[datetime] = None
    gs_end_time: Optional[datetime] = None
    gs_window_id: Optional[int] = None


@dataclass
class Schedule:
    """Scheduling plan: a set of Assignments."""
    assignments: List[Assignment] = field(default_factory=list)

    # Additional information generated during execution that does not affect scheduling semantics.
    # Examples include MIP solver gap, solve status, and debugging statistics.
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
# 2. Time placement strategy within a window
# ==============================

class TimePlacementStrategy:
    """
    Extensible time placement strategy within a window:
    - 'earliest': place as early as possible
    - 'center': place in the center
    - strategies such as 'latest' can be added in the future
    """

    @staticmethod
    def place(
        window_start: datetime,
        window_end: datetime,
        required_duration: float,
        mode: str = "earliest",
    ) -> Optional[tuple[datetime, datetime]]:
        """
        Select the task start and end time within the given window based on the strategy.
        Returns (start_time, end_time) or None if the task cannot fit.
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

        # Default fallback to earliest placement
        start = window_start
        end = start + timedelta(seconds=required_duration)
        return start, end


# ==============================
# 3. Constraint model: ConstraintModel
# ==============================

class ConstraintModel:

    def __init__(
        self,
        problem: SchedulingProblem,
        placement_mode: str = "earliest",
        unassigned_penalty: float = 1000.0,
        downlink_duration_ratio: float = 1.0,
        agility_profile: str = "Standard-Agility",
        non_agile_transition_s: float = 10.0,
    ) -> None:

        self.problem = problem
        self.placement_mode = placement_mode
        self.unassigned_penalty = unassigned_penalty
        self.downlink_duration_ratio = downlink_duration_ratio

        self.has_ground_stations = len(problem.ground_stations) > 0

    # ---------- Constraint checking ----------

    @staticmethod
    def _intervals_overlap(a_start: datetime, a_end: datetime,
                           b_start: datetime, b_end: datetime) -> bool:
        """Check whether two time intervals overlap."""
        return not (a_end <= b_start or a_start >= b_end)

    def _estimate_data_volume_GB(self, satellite_id: str, sensor_id: str, sat_start: datetime, sat_end: datetime) -> float:
        """Capacity consumption: convert data_rate_Mbps * duration_s to GB."""
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
        """Energy or power consumption: calculated per task, with each task consuming a fixed power_consumption_W."""
        sat = self.problem.satellites.get(satellite_id)
        if sat is None:
            return 0.0
        spec = sat.sensors.get(sensor_id)
        if spec is None:
            return 0.0
        return float(spec.power_consumption_W)

    @staticmethod
    def _extract_angles_first(sat_angles: object) -> Optional[Dict[str, float]]:
        """Extract the starting angle from sat_angles. Supports dict(list) and dict(scalar)."""
        if sat_angles is None:
            return None
        if isinstance(sat_angles, dict):
            # Agile style: pitch_angles / yaw_angles / roll_angles
            if "pitch_angles" in sat_angles:
                try:
                    return {
                        "pitch": float(sat_angles["pitch_angles"][0]),
                        "yaw": float(sat_angles["yaw_angles"][0]),
                        "roll": float(sat_angles["roll_angles"][0]),
                    }
                except Exception:
                    return None
            # Non-agile style: pitch_angle / yaw_angle / roll_angle
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
        """Extract the ending angle, that is, the last angle, from sat_angles."""
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
                # Non-agile: scalar values are identical for start and end
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
        """Attitude transition time between adjacent tasks in seconds.

        - non_agile: fixed non_agile_transition_s configured by main or main_scheduler
        - agile: follow the user-defined piecewise model:
            Δg = |Δroll| + |Δpitch| + |Δyaw|
            see schedulers/transition_utils.py
        """
        sat = self.problem.satellites.get(satellite_id)
        if sat is None:
            return 0.0

        if sat.maneuverability_type == "non_agile":
            return float(self.non_agile_transition_s)

        dg = delta_g_between(prev_a.sat_angles, next_a.sat_angles)
        if dg is None:
            # If angle data is missing, fall back to a minimum constant conservatively
            return 11.66

        return float(compute_transition_time_agile(dg, self.agility_profile))

    def _check_per_orbit_resource(self, assignment: Assignment, schedule: Schedule) -> bool:
        """Check per-orbit capacity and energy limits."""
        sat = self.problem.satellites.get(assignment.satellite_id)
        if sat is None:
            return True

        max_storage = float(getattr(sat, "max_data_storage_GB", 0.0) or 0.0)
        max_power = float(getattr(sat, "max_power_W", 0.0) or 0.0)
        # If the upper limit is 0, treat it as unrestricted for compatibility with older scenarios
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
        """Check whether adding an assignment to the current schedule is feasible."""

        task = self.problem.tasks[assignment.task_id]

        # 1) A task can only be assigned once
        if schedule.get_assignments_for_task(assignment.task_id):
            return False

        # 2) The satellite observation segment must satisfy the task duration requirement
        actual_duration = (assignment.sat_end_time - assignment.sat_start_time).total_seconds()
        if actual_duration + 1e-6 < task.required_duration:
            return False

        # 3) If ground stations exist, check the basic validity of the downlink segment
        if self.has_ground_stations:
            if assignment.ground_station_id is None:
                return False
            if assignment.gs_start_time is None or assignment.gs_end_time is None:
                return False
            if assignment.gs_start_time <= assignment.sat_end_time:
                # Downlink must start after observation is completed
                return False

        # 4) On the same satellite, the assignment must not overlap with any working segment
        # of other tasks, including observation and transmission
        sat_assigns = schedule.get_assignments_for_satellite(assignment.satellite_id)
        for a in sat_assigns:
            # Attitude transition time constraint between adjacent tasks,
            # applied only to satellite observation segments
            if assignment.sat_start_time >= a.sat_start_time:
                trans_s = self._transition_time_s(assignment.satellite_id, a, assignment)
                if a.sat_end_time + timedelta(seconds=trans_s) > assignment.sat_start_time:
                    return False
            else:
                trans_s = self._transition_time_s(assignment.satellite_id, assignment, a)
                if assignment.sat_end_time + timedelta(seconds=trans_s) > a.sat_start_time:
                    return False

            # Observation segment of a
            if self._intervals_overlap(
                assignment.sat_start_time, assignment.sat_end_time,
                a.sat_start_time, a.sat_end_time,
            ):
                return False
            # Transmission segment of a
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
                # Transmission segment of assignment against observation segment of a
                if assignment.gs_start_time is not None and assignment.gs_end_time is not None:
                    if self._intervals_overlap(
                        assignment.gs_start_time, assignment.gs_end_time,
                        a.sat_start_time, a.sat_end_time,
                    ):
                        return False

        # 5) Ground station constraint: only one task can be served at the same time
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

        # 6) Per-orbit capacity and energy constraints
        if not self._check_per_orbit_resource(assignment, schedule):
            return False

        return True

    # ---------- Build one feasible assignment for a single task ----------

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

        # Random source: use the passed rng if available;
        # otherwise fall back to the global random module for compatibility
        _rnd = rng if rng is not None else random

        if task.required_duration <= 0 or not task.windows:
            return None

        def place_in_window(ws, we, dur) -> Optional[tuple[datetime, datetime]]:
            total = (we - ws).total_seconds()
            if dur > total:
                return None
            if not randomized:
                return TimePlacementStrategy.place(ws, we, dur, mode=self.placement_mode)

            # If randomized=True, sample the start time randomly within the feasible range
            slack = total - dur
            if slack <= 1e-9:
                start = ws
            else:
                start = ws + timedelta(seconds=_rnd.random() * slack)
            end = start + timedelta(seconds=dur)
            return start, end

        def place_observation_in_task_window(w: TaskWindow) -> Optional[tuple[datetime, datetime, Optional[list]]]:
            """Select the actual execution sub-window within the visible window based on
            duration_s and time_step, and produce the corresponding angle data."""
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

            # Non-agile: only the middle segment of the window is allowed,
            # aligned to time_step
            if sat_type == "non_agile":
                center_offset = max_offset / 2.0
                k = int(round(center_offset / step))
                k = max(0, min(max_k, k))
            else:
                # Agile: any start aligned to time_step is allowed.
                # If randomized=True, pick randomly; otherwise follow placement_mode.
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

            # Angle data:
            # Agile -> slice corresponding to the selected sub-window
            # Non-agile -> a single fixed data set
            def _slice_angles_payload(payload, start_idx: int, count: int):
                if payload is None:
                    return None
                if isinstance(payload, list):
                    return payload[start_idx: start_idx + count]
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
                # Use ceil to ensure the duration is fully covered and avoid
                # being one step short because of round
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

    # ---------- Initial solution generation using a simple greedy method ----------

    def build_initial_schedule(self) -> Schedule:
        """
        Build a simple feasible initial solution:
        - sort tasks by priority from high to low
        - call build_feasible_assignment_for_task for each task
        - if a feasible assignment is found, add it to the schedule
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
            # If no feasible assignment is found, the task remains unassigned
            # and will be penalized by the objective function

        return schedule

    # ---------- Objective function ----------

    def evaluate(self, schedule: Schedule) -> float:
        """
        Objective function where larger is better:
        - sum of the priorities of assigned tasks
        - penalty equals number of unassigned tasks multiplied by unassigned_penalty
        """
        assigned_task_ids = set(schedule.assigned_task_ids)
        total_priority = sum(
            self.problem.tasks[tid].priority for tid in assigned_task_ids
        )

        total_tasks = len(self.problem.tasks)
        unassigned = total_tasks - len(assigned_task_ids)
        penalty = unassigned * self.unassigned_penalty

        return total_priority - penalty