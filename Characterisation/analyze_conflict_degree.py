# -*- coding: utf-8 -*-
"""
Created on Mon Mar 23 2026

@author: QianY 

analyze_conflict_degree.py

Main features:
1. Parse the scheduling problem from a scenario JSON file.
2. Analyse the core task conflict degree metrics required for the final summary.
3. Only the necessary 5 underlying metrics are retained to compute the final Excel output.
4. "Task pair conflict statistics" supports optional parallelisation.
5. Accounts for transition time, with the transition model consistent with the current project.

Compared to the original version:
- Stripped out unused metric calculations (resource tightness, ground station contention, etc.) to optimise execution speed.
- All comments and console outputs use British English.
"""

import bisect
import gc
import json
import math
import time
import multiprocessing as mp
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple


# =============================================================================
# 0) Global Constants
# =============================================================================

AGILITY_PROFILES: Dict[str, Tuple[float, float, float, float]] = {
    "High-Agility": (3.00, 4.00, 5.00, 6.00),
    "Standard-Agility": (1.50, 2.00, 2.50, 3.00),
    "Low-Agility": (0.75, 1.00, 1.25, 1.50),
    "Limited-Agility": (0.50, 0.67, 0.83, 1.00),
}

A1 = 5.0
A2 = 10.0
A3 = 16.0
A4 = 22.0
C_SMALL = 11.66  # Fixed transition time (seconds) when Δg <= 10

# In-process global context used exclusively for parallel task pair statistics
_PAIRWISE_CONTEXT: Dict[str, Any] = {}


# =============================================================================
# 1) Data Structures
# =============================================================================

@dataclass
class SensorSpec:
    sensor_id: str
    data_rate_Mbps: float = 0.0
    power_consumption_W: float = 0.0


@dataclass
class SchedulingSatellite:
    id: str
    maneuverability_type: str = "agile"
    max_data_storage_GB: float = 0.0
    max_power_W: float = 0.0
    slew_rate_deg_per_s: float = 1.0
    stabilization_time_s: float = 0.0
    sensors: Dict[str, SensorSpec] = field(default_factory=dict)


@dataclass
class TaskWindow:
    window_id: int
    satellite_id: str
    mission_id: str
    sensor_id: str
    orbit_number: int
    start_time: datetime
    end_time: datetime
    time_step: float = 1.0
    agile_data: Optional[object] = None
    non_agile_data: Optional[object] = None

    @property
    def duration(self) -> float:
        return (self.end_time - self.start_time).total_seconds()


@dataclass
class CommWindow:
    window_id: int
    satellite_id: str
    ground_station_id: str
    start_time: datetime
    end_time: datetime


@dataclass
class SchedulingTask:
    id: str
    priority: float
    required_duration: float
    windows: List[TaskWindow] = field(default_factory=list)


@dataclass
class Candidate:
    task_id: str
    satellite_id: str
    sat_start_time: datetime
    sat_end_time: datetime
    sat_window_id: int
    sensor_id: str = ""
    orbit_number: int = 0
    data_volume_GB: float = 0.0
    power_cost_W: float = 0.0
    sat_angles: Optional[object] = None
    ground_station_id: Optional[str] = None
    gs_start_time: Optional[datetime] = None
    gs_end_time: Optional[datetime] = None
    gs_window_id: Optional[int] = None


@dataclass
class SchedulingProblem:
    scenario_id: str
    start_time: datetime
    end_time: datetime
    global_time_step: float
    satellites: Dict[str, SchedulingSatellite]
    ground_stations: Dict[str, dict]
    tasks: Dict[str, SchedulingTask]
    comm_windows: List[CommWindow]


@dataclass
class ConflictAnalysisResult:
    """Slimmed down result dataclass containing only the necessary metrics."""
    scenario_id: str
    file_name: str
    rel_path: str
    n_satellites: int
    n_tasks: int

    avg_candidates_per_task: float
    hard_task_ratio_k2: float

    task_conflict_density: float
    mean_pair_conflict_ratio: float

    observation_contention_index: float

    analysis_seconds: float
    note: str = ""


# ======== Compact data structures exclusively for task pair statistics ========

@dataclass
class PairwiseCandidate:
    """Lightweight candidate exclusively for task pair statistics."""
    task_id: str
    satellite_id: str
    ground_station_id: Optional[str]
    sat_start_time: datetime
    sat_end_time: datetime
    gs_start_time: Optional[datetime]
    gs_end_time: Optional[datetime]
    sat_angles: Optional[object]

    sat_start_s: float
    sat_end_s: float
    occ_end_s: float
    gs_start_s: Optional[float]
    gs_end_s: Optional[float]
    has_gs: bool

    first_attitude: Optional[Tuple[float, float, float]]
    last_attitude: Optional[Tuple[float, float, float]]


@dataclass
class SortedIntervalView:
    """Sorted view of an interval group for fast overlap counting."""
    items: List[PairwiseCandidate]
    starts: List[float]
    ends: List[float]


@dataclass
class TaskPairwiseMeta:
    """Preprocessed resource indices for a single task."""
    has_candidates: bool
    sat_set: Set[str]
    gs_set: Set[str]
    by_sat: Dict[str, List[PairwiseCandidate]]
    by_gs: Dict[str, List[PairwiseCandidate]]
    by_gs_sat: Dict[str, Dict[str, List[PairwiseCandidate]]]
    sat_counts: Dict[str, int]
    gs_counts: Dict[str, int]
    gs_sat_counts: Dict[str, Dict[str, int]]
    gs_views: Dict[str, SortedIntervalView]


# =============================================================================
# 2) Logging
# =============================================================================

def log(msg: str) -> None:
    """Output logs with a timestamp."""
    now = time.strftime("%H:%M:%S")
    print(f"[{now}] {msg}", flush=True)


# =============================================================================
# 3) Basic Utilities
# =============================================================================

def parse_iso_time(time_str: str) -> datetime:
    s = (time_str or "").strip()
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    dt = datetime.fromisoformat(s)
    if dt.tzinfo is not None:
        dt = dt.astimezone(timezone.utc).replace(tzinfo=None)
    return dt


def infer_maneuverability_type(sat: dict) -> str:
    def _norm(v: Any) -> Optional[str]:
        if v is None:
            return None
        if isinstance(v, str):
            s = v.strip().lower()
            if "non" in s:
                return "non_agile"
            if "agile" in s:
                return "agile"
        return None

    mcap = sat.get("maneuverability_capability")
    if isinstance(mcap, dict):
        v = _norm(mcap.get("maneuverability_type") or mcap.get("type"))
        if v:
            return v

    v = _norm(sat.get("maneuverability_type"))
    if v:
        return v

    return "agile"


def normalize_profile_name(name: str) -> str:
    if not name:
        return "Standard-Agility"
    n = name.strip()
    low = n.lower().replace("_", "-")
    if low in {"high", "high-agility", "highagility"}:
        return "High-Agility"
    if low in {"standard", "standard-agility", "standardagility"}:
        return "Standard-Agility"
    if low in {"low", "low-agility", "lowagility"}:
        return "Low-Agility"
    if low in {"limited", "limited-agility", "limitedagility"}:
        return "Limited-Agility"
    return n


def get_profile_velocities(profile_name: str) -> Tuple[float, float, float, float]:
    key = normalize_profile_name(profile_name)
    return AGILITY_PROFILES.get(key, AGILITY_PROFILES["Standard-Agility"])


def compute_transition_time_agile(delta_g_deg: float, profile_name: str) -> float:
    dg = float(delta_g_deg)
    if dg <= 10.0:
        return C_SMALL

    v1, v2, v3, v4 = get_profile_velocities(profile_name)
    if dg <= 30.0:
        return A1 + dg / v1
    if dg <= 60.0:
        return A2 + dg / v2
    if dg <= 90.0:
        return A3 + dg / v3
    return A4 + dg / v4


def _get_first_last_from_list(x: Any, want_first: bool) -> Any:
    if not isinstance(x, list) or not x:
        return None
    return x[0] if want_first else x[-1]


def _attitude_from_mapping(m: Dict[str, Any]) -> Optional[Dict[str, float]]:
    if all(k in m for k in ("roll", "pitch", "yaw")):
        try:
            return {"roll": float(m["roll"]), "pitch": float(m["pitch"]), "yaw": float(m["yaw"])}
        except Exception:
            return None

    if all(k in m for k in ("gamma", "pi", "psi")):
        try:
            return {"roll": float(m["gamma"]), "pitch": float(m["pi"]), "yaw": float(m["psi"])}
        except Exception:
            return None

    if all(k in m for k in ("pitch_angle", "yaw_angle", "roll_angle")):
        try:
            return {
                "roll": float(m["roll_angle"]),
                "pitch": float(m["pitch_angle"]),
                "yaw": float(m["yaw_angle"]),
            }
        except Exception:
            return None

    return None


def extract_attitude_from_angles(angles: Any, want_first: bool) -> Optional[Dict[str, float]]:
    if angles is None:
        return None

    if isinstance(angles, list):
        rec = _get_first_last_from_list(angles, want_first)
        if rec is None:
            return None
        if isinstance(rec, dict):
            return _attitude_from_mapping(rec)
        if isinstance(rec, (list, tuple)) and len(rec) >= 3:
            try:
                return {"roll": float(rec[0]), "pitch": float(rec[1]), "yaw": float(rec[2])}
            except Exception:
                return None
        return None

    if isinstance(angles, dict):
        for keyset in (("roll", "pitch", "yaw"), ("gamma", "pi", "psi")):
            if all(k in angles for k in keyset):
                try:
                    r = _get_first_last_from_list(angles[keyset[0]], want_first)
                    p = _get_first_last_from_list(angles[keyset[1]], want_first)
                    y = _get_first_last_from_list(angles[keyset[2]], want_first)
                    if r is None or p is None or y is None:
                        return None
                    return {"roll": float(r), "pitch": float(p), "yaw": float(y)}
                except Exception:
                    return None

        att = _attitude_from_mapping(angles)
        if att is not None:
            return att

        for v in angles.values():
            if isinstance(v, dict):
                att = _attitude_from_mapping(v)
                if att is not None:
                    return att

    return None


def delta_g_between(prev_angles: Any, next_angles: Any) -> Optional[float]:
    a1 = extract_attitude_from_angles(prev_angles, want_first=False)
    a2 = extract_attitude_from_angles(next_angles, want_first=True)
    if a1 is None or a2 is None:
        return None
    return float(
        abs(a2["roll"] - a1["roll"]) +
        abs(a2["pitch"] - a1["pitch"]) +
        abs(a2["yaw"] - a1["yaw"])
    )


def _to_attitude_tuple(att: Optional[Dict[str, float]]) -> Optional[Tuple[float, float, float]]:
    if att is None:
        return None
    return (float(att["roll"]), float(att["pitch"]), float(att["yaw"]))


def intervals_overlap(a_start: datetime, a_end: datetime, b_start: datetime, b_end: datetime) -> bool:
    return not (a_end <= b_start or a_start >= b_end)


def merge_intervals(intervals: List[Tuple[datetime, datetime]]) -> List[Tuple[datetime, datetime]]:
    if not intervals:
        return []
    intervals = sorted(intervals, key=lambda x: x[0])
    merged = [intervals[0]]
    for s, e in intervals[1:]:
        last_s, last_e = merged[-1]
        if s <= last_e:
            if e > last_e:
                merged[-1] = (last_s, e)
        else:
            merged.append((s, e))
    return merged


def estimate_data_volume_GB(
    satellites: Dict[str, SchedulingSatellite],
    satellite_id: str,
    sensor_id: str,
    sat_start: datetime,
    sat_end: datetime,
) -> float:
    sat = satellites.get(satellite_id)
    if sat is None:
        return 0.0
    spec = sat.sensors.get(sensor_id)
    if spec is None:
        return 0.0
    dur = max(0.0, (sat_end - sat_start).total_seconds())
    mbits = float(spec.data_rate_Mbps) * dur
    return mbits / (8.0 * 1024.0)


def estimate_power_cost_W(
    satellites: Dict[str, SchedulingSatellite],
    satellite_id: str,
    sensor_id: str,
) -> float:
    sat = satellites.get(satellite_id)
    if sat is None:
        return 0.0
    spec = sat.sensors.get(sensor_id)
    if spec is None:
        return 0.0
    return float(spec.power_consumption_W)


def slice_angles_payload(payload: Any, start_idx: int, count: int) -> Any:
    if payload is None:
        return None
    if isinstance(payload, list):
        return payload[start_idx: start_idx + count]
    if isinstance(payload, dict):
        out = {}
        for k, v in payload.items():
            if isinstance(v, (list, dict)):
                out[k] = slice_angles_payload(v, start_idx, count)
            else:
                out[k] = v
        return out
    return payload


# =============================================================================
# 4) JSON Loading
# =============================================================================

def load_scheduling_problem_from_json(json_path: Path) -> SchedulingProblem:
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    scenario_id = data.get("scenario_id", json_path.stem)

    meta = data["metadata"]
    start_time = parse_iso_time(meta["creation_time"])
    duration_s = float(meta["duration"])
    end_time = start_time + timedelta(seconds=duration_s)
    global_time_step = float(meta.get("time_step", 1.0) or 1.0)

    satellites: Dict[str, SchedulingSatellite] = {}
    for sat in data.get("satellites", []):
        sat_id = sat["id"]
        specs = sat.get("satellite_specs", {}) or {}
        max_data_storage_GB = float(specs.get("max_data_storage_GB", 0.0) or 0.0)
        max_power_W = float(specs.get("max_power_W", 0.0) or 0.0)

        man = sat.get("maneuverability_capability", {}) or {}
        slew_rate = float(man.get("slew_rate_deg_per_s", 1.0) or man.get("slew_rate", 1.0) or 1.0)
        stab_time = float(man.get("stabilization_time_s", 0.0) or man.get("stabilization_time", 0.0) or 0.0)

        sensors: Dict[str, SensorSpec] = {}
        obs_cap = sat.get("observation_capability", {}) or {}
        for s in obs_cap.get("sensors", []) or []:
            sid = s.get("sensor_id")
            if not sid:
                continue
            sensors[sid] = SensorSpec(
                sensor_id=sid,
                data_rate_Mbps=float(s.get("data_rate_Mbps", 0.0) or s.get("data_rate", 0.0) or 0.0),
                power_consumption_W=float(s.get("power_consumption_W", 0.0) or s.get("power_consumption", 0.0) or 0.0),
            )

        satellites[sat_id] = SchedulingSatellite(
            id=sat_id,
            maneuverability_type=infer_maneuverability_type(sat),
            max_data_storage_GB=max_data_storage_GB,
            max_power_W=max_power_W,
            slew_rate_deg_per_s=slew_rate,
            stabilization_time_s=stab_time,
            sensors=sensors,
        )

    ground_stations: Dict[str, dict] = {}
    for gs in data.get("ground_stations", []):
        gs_id = gs["id"]
        ground_stations[gs_id] = {"id": gs_id}

    tasks: Dict[str, SchedulingTask] = {}
    for mission in data.get("missions", []):
        mid = mission["id"]
        priority = float(mission.get("priority", 1.0))
        required_duration = 0.0
        obs_req = mission.get("observation_requirement")
        if obs_req is not None:
            required_duration = float(obs_req.get("duration_s", 0.0))

        tasks[mid] = SchedulingTask(
            id=mid,
            priority=priority,
            required_duration=required_duration,
            windows=[],
        )

    window_counter = 0
    for ow in data.get("observation_windows", []):
        sat_id = ow["satellite_id"]
        mid = ow["mission_id"]
        if mid not in tasks or sat_id not in satellites:
            continue

        for tw in ow.get("time_windows", []):
            start = parse_iso_time(tw["start_time"])
            end = parse_iso_time(tw["end_time"])
            if end <= start:
                continue

            window_counter += 1
            task_window = TaskWindow(
                window_id=window_counter,
                satellite_id=sat_id,
                mission_id=mid,
                sensor_id=str(ow.get("sensor_id", "")),
                orbit_number=int(tw.get("orbit_number", ow.get("orbit_number", 0)) or 0),
                start_time=start,
                end_time=end,
                time_step=float(tw.get("time_step", ow.get("time_step", global_time_step)) or global_time_step or 1.0),
                agile_data=tw.get("agile_data", ow.get("agile_data")),
                non_agile_data=tw.get("non_agile_data", ow.get("non_agile_data")),
            )
            tasks[mid].windows.append(task_window)

    comm_windows: List[CommWindow] = []
    comm_counter = 0
    for cw in data.get("communication_windows", []):
        sat_id = cw["satellite_id"]
        gs_id = cw["ground_station_id"]
        if sat_id not in satellites or gs_id not in ground_stations:
            continue

        for tw in cw.get("time_windows", []):
            start = parse_iso_time(tw["start_time"])
            end = parse_iso_time(tw["end_time"])
            if end <= start:
                continue

            comm_counter += 1
            comm_windows.append(
                CommWindow(
                    window_id=comm_counter,
                    satellite_id=sat_id,
                    ground_station_id=gs_id,
                    start_time=start,
                    end_time=end,
                )
            )

    return SchedulingProblem(
        scenario_id=scenario_id,
        start_time=start_time,
        end_time=end_time,
        global_time_step=global_time_step,
        satellites=satellites,
        ground_stations=ground_stations,
        tasks=tasks,
        comm_windows=comm_windows,
    )


# =============================================================================
# 5) Candidate Generation (Sequential to save parallel overhead)
# =============================================================================

def build_comm_by_sat(comm_windows: List[CommWindow]) -> Dict[str, List[CommWindow]]:
    d: Dict[str, List[CommWindow]] = defaultdict(list)
    for cw in comm_windows:
        d[cw.satellite_id].append(cw)
    for sid in d:
        d[sid].sort(key=lambda x: x.start_time)
    return d


def enumerate_task_candidates(
    task: SchedulingTask,
    satellites: Dict[str, SchedulingSatellite],
    comm_by_sat: Dict[str, List[CommWindow]],
    downlink_duration_ratio: float = 1.0,
    observation_step_multiplier: int = 1,
    max_candidates_per_task: Optional[int] = None,
) -> List[Candidate]:
    """
    Generate candidates for a single task:
    - Agile satellites: Enumerate observation sub-windows by time_step * multiplier.
    - Non-agile satellites: Extract only the middle section.
    - If ground stations exist: Match each observation candidate with earliest/center/latest data transmission slots.
    """
    if task.required_duration <= 0 or not task.windows:
        return []

    cands: List[Candidate] = []
    has_ground_stations = len(comm_by_sat) > 0
    downlink_duration = task.required_duration * float(downlink_duration_ratio)
    step_mul = max(1, int(observation_step_multiplier))

    def _reach_cap() -> bool:
        return (max_candidates_per_task is not None) and (len(cands) >= int(max_candidates_per_task))

    for w in sorted(task.windows, key=lambda x: x.start_time):
        if _reach_cap():
            break

        total = (w.end_time - w.start_time).total_seconds()
        dur = float(task.required_duration)
        step = float(w.time_step or 1.0) * step_mul

        if dur <= 0 or dur > total:
            continue

        sat = satellites.get(w.satellite_id)
        sat_type = (sat.maneuverability_type if sat is not None else "agile").lower()

        max_offset = total - dur
        max_k = int(math.floor((max_offset + 1e-9) / step))
        if max_k < 0:
            continue

        placements: List[Tuple[datetime, datetime, Any]] = []

        def angles_for(k: int) -> Any:
            if sat_type == "non_agile":
                return getattr(w, "non_agile_data", None)
            ad = getattr(w, "agile_data", None)
            n = max(1, int(math.ceil((dur / float(w.time_step or 1.0)) - 1e-9)))
            start_idx = int(round((k * step) / float(w.time_step or 1.0)))
            return slice_angles_payload(ad, start_idx, n)

        if sat_type == "non_agile":
            center_k = int(round((max_offset / 2.0) / step))
            center_k = max(0, min(max_k, center_k))
            sat_start = w.start_time + timedelta(seconds=center_k * step)
            sat_end = sat_start + timedelta(seconds=dur)
            placements.append((sat_start, sat_end, angles_for(center_k)))
        else:
            must = [0, int(round((max_offset / 2.0) / step)), max_k]
            must = [max(0, min(max_k, x)) for x in must]
            must_uniq: List[int] = []
            for x in must:
                if x not in must_uniq:
                    must_uniq.append(x)

            visited = set()
            for k in must_uniq:
                sat_start = w.start_time + timedelta(seconds=k * step)
                sat_end = sat_start + timedelta(seconds=dur)
                placements.append((sat_start, sat_end, angles_for(k)))
                visited.add(k)

            for k in range(0, max_k + 1):
                if k in visited:
                    continue
                sat_start = w.start_time + timedelta(seconds=k * step)
                sat_end = sat_start + timedelta(seconds=dur)
                placements.append((sat_start, sat_end, angles_for(k)))

        if not has_ground_stations:
            for sat_start, sat_end, sat_angles in placements:
                cands.append(
                    Candidate(
                        task_id=task.id,
                        satellite_id=w.satellite_id,
                        sat_start_time=sat_start,
                        sat_end_time=sat_end,
                        sat_window_id=w.window_id,
                        sensor_id=w.sensor_id,
                        orbit_number=w.orbit_number,
                        data_volume_GB=estimate_data_volume_GB(satellites, w.satellite_id, w.sensor_id, sat_start, sat_end),
                        power_cost_W=estimate_power_cost_W(satellites, w.satellite_id, w.sensor_id),
                        sat_angles=sat_angles,
                    )
                )
                if _reach_cap():
                    break
            continue

        sat_comm_windows = comm_by_sat.get(w.satellite_id, [])
        if not sat_comm_windows:
            continue

        for sat_start, sat_end, sat_angles in placements:
            if _reach_cap():
                break

            for cw in sat_comm_windows:
                earliest = max(cw.start_time, sat_end)
                avail = (cw.end_time - earliest).total_seconds()
                if avail < downlink_duration:
                    continue

                dl_places: List[Tuple[datetime, datetime]] = []
                slack = (cw.end_time - earliest).total_seconds() - downlink_duration
                if slack < -1e-9:
                    continue

                # earliest
                gs_start = earliest
                gs_end = gs_start + timedelta(seconds=downlink_duration)
                if gs_end <= cw.end_time:
                    dl_places.append((gs_start, gs_end))

                # center
                gs_start = earliest + timedelta(seconds=max(0.0, slack / 2.0))
                gs_end = gs_start + timedelta(seconds=downlink_duration)
                if gs_end <= cw.end_time:
                    dl_places.append((gs_start, gs_end))

                # latest
                gs_end = cw.end_time
                gs_start = gs_end - timedelta(seconds=downlink_duration)
                if gs_start >= earliest:
                    dl_places.append((gs_start, gs_end))

                seen = set()
                for gs_start, gs_end in dl_places:
                    key = (gs_start, gs_end)
                    if key in seen:
                        continue
                    seen.add(key)

                    cands.append(
                        Candidate(
                            task_id=task.id,
                            satellite_id=w.satellite_id,
                            sat_start_time=sat_start,
                            sat_end_time=sat_end,
                            sat_window_id=w.window_id,
                            sensor_id=w.sensor_id,
                            orbit_number=w.orbit_number,
                            data_volume_GB=estimate_data_volume_GB(satellites, w.satellite_id, w.sensor_id, sat_start, sat_end),
                            power_cost_W=estimate_power_cost_W(satellites, w.satellite_id, w.sensor_id),
                            sat_angles=sat_angles,
                            ground_station_id=cw.ground_station_id,
                            gs_start_time=gs_start,
                            gs_end_time=gs_end,
                            gs_window_id=cw.window_id,
                        )
                    )
                    if _reach_cap():
                        break

    return cands


def build_candidate_map(
    problem: SchedulingProblem,
    downlink_duration_ratio: float,
    observation_step_multiplier: int,
    max_candidates_per_task: Optional[int],
) -> Dict[str, List[Candidate]]:
    task_ids = list(problem.tasks.keys())
    total = len(task_ids)
    cand_map: Dict[str, List[Candidate]] = {}
    comm_by_sat = build_comm_by_sat(problem.comm_windows)

    log("Starting to construct the candidate set...")
    for idx, tid in enumerate(task_ids, start=1):
        cands = enumerate_task_candidates(
            task=problem.tasks[tid],
            satellites=problem.satellites,
            comm_by_sat=comm_by_sat,
            downlink_duration_ratio=downlink_duration_ratio,
            observation_step_multiplier=observation_step_multiplier,
            max_candidates_per_task=max_candidates_per_task,
        )
        cand_map[tid] = cands
        if idx % 10 == 0 or idx == total:
            log(f"Candidate construction progress: {idx}/{total}")

    return cand_map


# =============================================================================
# 6) Candidate Conflict Judgement
# =============================================================================

def assignment_sat_intervals(a: Candidate) -> List[Tuple[datetime, datetime]]:
    res = [(a.sat_start_time, a.sat_end_time)]
    if a.gs_start_time is not None and a.gs_end_time is not None:
        res.append((a.gs_start_time, a.gs_end_time))
    return res


def transition_time_s(
    satellites: Dict[str, SchedulingSatellite],
    satellite_id: str,
    prev_a: Candidate,
    next_a: Candidate,
    agility_profile: str,
    non_agile_transition_s: float,
) -> float:
    sat = satellites.get(satellite_id)
    if sat is None:
        return 0.0

    if (sat.maneuverability_type or "agile").lower() != "agile":
        return float(non_agile_transition_s)

    dg = delta_g_between(prev_a.sat_angles, next_a.sat_angles)
    if dg is None:
        return 11.66
    return float(compute_transition_time_agile(dg, agility_profile))


def assignments_conflict_with_transition(
    a: Candidate,
    b: Candidate,
    satellites: Dict[str, SchedulingSatellite],
    agility_profile: str,
    non_agile_transition_s: float,
) -> bool:
    """
    Judge whether two candidates are conflicting:
    1. Any usage intervals on the same satellite overlap.
    2. Insufficient transition time between observations on the same satellite.
    3. Data transmission intervals overlap on the same ground station.
    """
    if a.satellite_id == b.satellite_id:
        for s1, e1 in assignment_sat_intervals(a):
            for s2, e2 in assignment_sat_intervals(b):
                if intervals_overlap(s1, e1, s2, e2):
                    return True

        if a.sat_start_time <= b.sat_start_time:
            trans = transition_time_s(
                satellites=satellites,
                satellite_id=a.satellite_id,
                prev_a=a,
                next_a=b,
                agility_profile=agility_profile,
                non_agile_transition_s=non_agile_transition_s,
            )
            if a.sat_end_time + timedelta(seconds=trans) > b.sat_start_time:
                return True
        else:
            trans = transition_time_s(
                satellites=satellites,
                satellite_id=a.satellite_id,
                prev_a=b,
                next_a=a,
                agility_profile=agility_profile,
                non_agile_transition_s=non_agile_transition_s,
            )
            if b.sat_end_time + timedelta(seconds=trans) > a.sat_start_time:
                return True

    if (
        a.ground_station_id is not None and
        b.ground_station_id is not None and
        a.ground_station_id == b.ground_station_id and
        a.gs_start_time is not None and
        a.gs_end_time is not None and
        b.gs_start_time is not None and
        b.gs_end_time is not None
    ):
        if intervals_overlap(a.gs_start_time, a.gs_end_time, b.gs_start_time, b.gs_end_time):
            return True

    return False


# =============================================================================
# 7) Task Pair Conflict Statistics (Optional Parallelisation)
# =============================================================================

def _candidate_to_pairwise(c: Candidate, scenario_start: Optional[datetime] = None) -> PairwiseCandidate:
    if scenario_start is None:
        scenario_start = min(c.sat_start_time, c.gs_start_time or c.sat_start_time)

    sat_start_s = (c.sat_start_time - scenario_start).total_seconds()
    sat_end_s = (c.sat_end_time - scenario_start).total_seconds()
    gs_start_s = (c.gs_start_time - scenario_start).total_seconds() if c.gs_start_time is not None else None
    gs_end_s = (c.gs_end_time - scenario_start).total_seconds() if c.gs_end_time is not None else None
    occ_end_s = max(sat_end_s, gs_end_s if gs_end_s is not None else sat_end_s)

    first_att = _to_attitude_tuple(extract_attitude_from_angles(c.sat_angles, want_first=True))
    last_att = _to_attitude_tuple(extract_attitude_from_angles(c.sat_angles, want_first=False))

    return PairwiseCandidate(
        task_id=c.task_id,
        satellite_id=c.satellite_id,
        ground_station_id=c.ground_station_id,
        sat_start_time=c.sat_start_time,
        sat_end_time=c.sat_end_time,
        gs_start_time=c.gs_start_time,
        gs_end_time=c.gs_end_time,
        sat_angles=c.sat_angles,
        sat_start_s=sat_start_s,
        sat_end_s=sat_end_s,
        occ_end_s=occ_end_s,
        gs_start_s=gs_start_s,
        gs_end_s=gs_end_s,
        has_gs=(gs_start_s is not None and gs_end_s is not None and c.ground_station_id is not None),
        first_attitude=first_att,
        last_attitude=last_att,
    )


def _sorted_interval_view(items: List[PairwiseCandidate], use_gs: bool) -> SortedIntervalView:
    if use_gs:
        kept = [x for x in items if x.has_gs and x.gs_start_s is not None and x.gs_end_s is not None]
        kept.sort(key=lambda x: x.gs_start_s)
        starts = [float(x.gs_start_s) for x in kept]
        ends = sorted(float(x.gs_end_s) for x in kept)
        return SortedIntervalView(items=kept, starts=starts, ends=ends)

    kept = sorted(items, key=lambda x: x.sat_start_s)
    starts = [x.sat_start_s for x in kept]
    ends = sorted(x.sat_end_s for x in kept)
    return SortedIntervalView(items=kept, starts=starts, ends=ends)


def build_task_pairwise_meta(
    task_ids: List[str],
    cand_map: Dict[str, List[Candidate]],
    scenario_start: datetime,
) -> Dict[str, TaskPairwiseMeta]:
    """Construct resource indices for task pair statistics."""
    meta_map: Dict[str, TaskPairwiseMeta] = {}
    total = len(task_ids)
    log("Starting to construct task pair statistical indices...")

    for idx, tid in enumerate(task_ids, start=1):
        raw_cands = cand_map.get(tid, [])
        if not raw_cands:
            meta_map[tid] = TaskPairwiseMeta(
                has_candidates=False, sat_set=set(), gs_set=set(), by_sat={},
                by_gs={}, by_gs_sat={}, sat_counts={}, gs_counts={}, gs_sat_counts={}, gs_views={},
            )
            continue

        pcs = [_candidate_to_pairwise(c, scenario_start=scenario_start) for c in raw_cands]

        by_sat: Dict[str, List[PairwiseCandidate]] = defaultdict(list)
        by_gs: Dict[str, List[PairwiseCandidate]] = defaultdict(list)
        by_gs_sat: Dict[str, Dict[str, List[PairwiseCandidate]]] = defaultdict(lambda: defaultdict(list))

        for pc in pcs:
            by_sat[pc.satellite_id].append(pc)
            if pc.has_gs and pc.ground_station_id is not None:
                by_gs[pc.ground_station_id].append(pc)
                by_gs_sat[pc.ground_station_id][pc.satellite_id].append(pc)

        for sid in list(by_sat.keys()):
            by_sat[sid].sort(key=lambda x: x.sat_start_s)
        for gs in list(by_gs.keys()):
            by_gs[gs].sort(key=lambda x: x.gs_start_s if x.gs_start_s is not None else -1.0)
            for sid in list(by_gs_sat[gs].keys()):
                by_gs_sat[gs][sid].sort(key=lambda x: x.gs_start_s if x.gs_start_s is not None else -1.0)

        sat_counts = {sid: len(lst) for sid, lst in by_sat.items()}
        gs_counts = {gs: len(lst) for gs, lst in by_gs.items()}
        gs_sat_counts = {gs: {sid: len(lst) for sid, lst in mp.items()} for gs, mp in by_gs_sat.items()}
        gs_views = {gs: _sorted_interval_view(lst, use_gs=True) for gs, lst in by_gs.items()}

        meta_map[tid] = TaskPairwiseMeta(
            has_candidates=True,
            sat_set=set(by_sat.keys()),
            gs_set=set(by_gs.keys()),
            by_sat=dict(by_sat),
            by_gs=dict(by_gs),
            by_gs_sat={gs: dict(mp) for gs, mp in by_gs_sat.items()},
            sat_counts=sat_counts,
            gs_counts=gs_counts,
            gs_sat_counts=gs_sat_counts,
            gs_views=gs_views,
        )

        if idx % 20 == 0 or idx == total:
            log(f"Task pair statistical index progress: {idx}/{total}")

    return meta_map


def init_pairwise_worker(context: Dict[str, Any]) -> None:
    global _PAIRWISE_CONTEXT
    _PAIRWISE_CONTEXT = context


def _same_gs_overlap_count(view_a: SortedIntervalView, view_b: SortedIntervalView) -> int:
    """
    Accurately tally overlapping pairs for two sets of half-open intervals [start, end).
    For each interval 'a' in A, the number of overlaps is:
        #{b.start < a.end} - #{b.end <= a.start}
    """
    if not view_a.items or not view_b.items:
        return 0

    b_starts = view_b.starts
    b_ends = view_b.ends
    cnt = 0
    for a in view_a.items:
        a_start = float(a.gs_start_s)
        a_end = float(a.gs_end_s)
        n1 = bisect.bisect_left(b_starts, a_end)       # b.start < a.end
        n2 = bisect.bisect_right(b_ends, a_start)      # b.end <= a.start
        overlap = n1 - n2
        if overlap > 0:
            cnt += overlap
    return cnt


def _pair_transition_time_from_prepared(
    satellite: SchedulingSatellite,
    prev_c: PairwiseCandidate,
    next_c: PairwiseCandidate,
    agility_profile: str,
    non_agile_transition_s: float,
) -> float:
    if (satellite.maneuverability_type or "agile").lower() != "agile":
        return float(non_agile_transition_s)

    if prev_c.last_attitude is None or next_c.first_attitude is None:
        return 11.66

    dg = (
        abs(next_c.first_attitude[0] - prev_c.last_attitude[0]) +
        abs(next_c.first_attitude[1] - prev_c.last_attitude[1]) +
        abs(next_c.first_attitude[2] - prev_c.last_attitude[2])
    )
    return float(compute_transition_time_agile(dg, agility_profile))


def _same_sat_conflict_given_order(
    earlier: PairwiseCandidate,
    later: PairwiseCandidate,
    satellite: SchedulingSatellite,
    agility_profile: str,
    non_agile_transition_s: float,
) -> bool:
    """Exact judgement when it is known that earlier.sat_start <= later.sat_start."""
    # 1) Any usage interval overlap (observation + transmission)
    if later.sat_start_s < earlier.sat_end_s:
        return True
    if earlier.has_gs and earlier.gs_start_s is not None and earlier.gs_end_s is not None:
        if later.sat_start_s < earlier.gs_end_s and later.sat_end_s > earlier.gs_start_s:
            return True
        if later.has_gs and later.gs_start_s is not None and later.gs_end_s is not None:
            if later.gs_start_s < earlier.gs_end_s and later.gs_end_s > earlier.gs_start_s:
                return True

    # 2) Insufficient transition time between observation sections
    trans_s = _pair_transition_time_from_prepared(
        satellite=satellite,
        prev_c=earlier,
        next_c=later,
        agility_profile=agility_profile,
        non_agile_transition_s=non_agile_transition_s,
    )
    return (earlier.sat_end_s + trans_s) > later.sat_start_s


def _count_same_sat_conflicts(
    list_a: List[PairwiseCandidate],
    list_b: List[PairwiseCandidate],
    satellite: SchedulingSatellite,
    agility_profile: str,
    non_agile_transition_s: float,
) -> int:
    """
    Precisely count the number of conflicting pairs among candidates on the same satellite.
    """
    if not list_a or not list_b:
        return 0

    sat_type = (satellite.maneuverability_type or "agile").lower()
    if sat_type != "agile":
        trans_upper = float(non_agile_transition_s)
    else:
        trans_upper = float(compute_transition_time_agile(360.0, agility_profile))

    starts_a = [x.sat_start_s for x in list_a]
    starts_b = [x.sat_start_s for x in list_b]
    count = 0

    # Direction 1: A is earlier, or A and B start simultaneously (same start is handled on A's side).
    for a in list_a:
        bound = max(a.occ_end_s, a.sat_end_s + trans_upper)
        lo = bisect.bisect_left(starts_b, a.sat_start_s)
        hi = bisect.bisect_left(starts_b, bound)
        for idx in range(lo, hi):
            b = list_b[idx]
            if _same_sat_conflict_given_order(
                earlier=a, later=b, satellite=satellite,
                agility_profile=agility_profile, non_agile_transition_s=non_agile_transition_s,
            ):
                count += 1

    # Direction 2: B is strictly earlier than A. 
    # Together with Direction 1, this guarantees every pair is only counted once.
    for b in list_b:
        bound = max(b.occ_end_s, b.sat_end_s + trans_upper)
        lo = bisect.bisect_right(starts_a, b.sat_start_s)
        hi = bisect.bisect_left(starts_a, bound)
        for idx in range(lo, hi):
            a = list_a[idx]
            if _same_sat_conflict_given_order(
                earlier=b, later=a, satellite=satellite,
                agility_profile=agility_profile, non_agile_transition_s=non_agile_transition_s,
            ):
                count += 1

    return count


def _count_same_gs_diff_sat_conflicts(
    gs_sat_map_a: Dict[str, List[PairwiseCandidate]],
    gs_sat_map_b: Dict[str, List[PairwiseCandidate]],
) -> int:
    """Tally transmission overlap conflicts for candidates on the same ground station but different satellites."""
    if not gs_sat_map_a or not gs_sat_map_b:
        return 0

    count = 0
    sat_ids_a = sorted(gs_sat_map_a.keys())
    sat_ids_b = sorted(gs_sat_map_b.keys())

    for sat_a in sat_ids_a:
        list_a = gs_sat_map_a[sat_a]
        view_a = _sorted_interval_view(list_a, use_gs=True)
        if not view_a.items:
            continue
        for sat_b in sat_ids_b:
            if sat_a == sat_b:
                continue
            list_b = gs_sat_map_b[sat_b]
            view_b = _sorted_interval_view(list_b, use_gs=True)
            if not view_b.items:
                continue
            count += _same_gs_overlap_count(view_a, view_b)
    return count


def _comparable_pair_count(meta_i: TaskPairwiseMeta, meta_j: TaskPairwiseMeta) -> int:
    """
    Precisely count "comparable candidate pairs" (same_sat OR same_gs)
    using set logic to avoid double-counting.
    """
    common_sats = meta_i.sat_set & meta_j.sat_set
    common_gs = meta_i.gs_set & meta_j.gs_set

    total = 0

    # same_sat section
    for sid in common_sats:
        total += meta_i.sat_counts.get(sid, 0) * meta_j.sat_counts.get(sid, 0)

    # same_gs section (excluding duplicates already counted in same_sat)
    for gs in common_gs:
        gs_total = meta_i.gs_counts.get(gs, 0) * meta_j.gs_counts.get(gs, 0)
        same_sat_dup = 0
        map_i = meta_i.gs_sat_counts.get(gs, {})
        map_j = meta_j.gs_sat_counts.get(gs, {})
        for sid in (set(map_i.keys()) & set(map_j.keys())):
            same_sat_dup += map_i.get(sid, 0) * map_j.get(sid, 0)
        total += (gs_total - same_sat_dup)

    return int(total)


def _estimate_task_workloads(task_ids: List[str], meta_map: Dict[str, TaskPairwiseMeta]) -> List[float]:
    n = len(task_ids)
    workloads: List[float] = []
    for i, tid in enumerate(task_ids):
        meta = meta_map.get(tid)
        cand_n = 0
        if meta is not None and meta.has_candidates:
            cand_n = sum(meta.sat_counts.values())
        workloads.append(max(1.0, cand_n) * max(1, n - i - 1))
    return workloads


def _build_workload_balanced_chunks(task_ids: List[str], meta_map: Dict[str, TaskPairwiseMeta], chunk_size: int) -> List[Tuple[int, int]]:
    """
    Balance workload blocks while aiming for ceil(n / chunk_size) total chunks.
    """
    n = len(task_ids)
    if n <= 0:
        return []

    target_chunk_num = max(1, int(math.ceil(n / max(1, int(chunk_size)))))
    workloads = _estimate_task_workloads(task_ids, meta_map)
    total_work = sum(workloads)
    target_work = total_work / target_chunk_num if target_chunk_num > 0 else total_work

    chunks: List[Tuple[int, int]] = []
    start = 0
    acc = 0.0
    for i, w in enumerate(workloads):
        acc += w
        if (i + 1 - start) >= chunk_size and acc >= target_work:
            chunks.append((start, i + 1))
            start = i + 1
            acc = 0.0

    if start < n:
        chunks.append((start, n))
    return chunks


def pair_conflict_worker(i_start: int, i_end: int) -> Dict[str, Any]:
    """
    Worker for task pair conflict statistics.
    Processes task_ids[i_start:i_end] as left-hand tasks paired with all tasks j > i.
    """
    ctx = _PAIRWISE_CONTEXT
    task_ids: List[str] = ctx["task_ids"]
    meta_map: Dict[str, TaskPairwiseMeta] = ctx["meta_map"]
    satellites: Dict[str, SchedulingSatellite] = ctx["satellites"]
    agility_profile = ctx["agility_profile"]
    non_agile_transition_s = ctx["non_agile_transition_s"]

    total_task_pairs = 0
    comparable_task_pairs = 0
    pairs_with_conflict = 0
    pair_conflict_ratio_sum = 0.0

    n = len(task_ids)

    for i in range(i_start, min(i_end, n)):
        tid_i = task_ids[i]
        meta_i = meta_map.get(tid_i)

        for j in range(i + 1, n):
            tid_j = task_ids[j]
            meta_j = meta_map.get(tid_j)
            total_task_pairs += 1

            if meta_i is None or meta_j is None or (not meta_i.has_candidates) or (not meta_j.has_candidates):
                continue

            # Task-level resource pre-filtering: No overlap possible if there are no common satellites/ground stations.
            common_sats = meta_i.sat_set & meta_j.sat_set
            common_gs = meta_i.gs_set & meta_j.gs_set
            if not common_sats and not common_gs:
                continue

            comparable_pairs = _comparable_pair_count(meta_i, meta_j)
            if comparable_pairs <= 0:
                continue

            conflicting_pairs = 0

            # 1) same_sat conflicts
            for sid in common_sats:
                sat = satellites.get(sid)
                if sat is None:
                    continue
                conflicting_pairs += _count_same_sat_conflicts(
                    list_a=meta_i.by_sat.get(sid, []), list_b=meta_j.by_sat.get(sid, []),
                    satellite=sat, agility_profile=agility_profile, non_agile_transition_s=non_agile_transition_s,
                )

            # 2) same_gs but different satellite conflicts
            for gs in common_gs:
                conflicting_pairs += _count_same_gs_diff_sat_conflicts(
                    gs_sat_map_a=meta_i.by_gs_sat.get(gs, {}), gs_sat_map_b=meta_j.by_gs_sat.get(gs, {}),
                )

            comparable_task_pairs += 1
            pair_conflict_ratio_sum += (conflicting_pairs / comparable_pairs)
            if conflicting_pairs > 0:
                pairs_with_conflict += 1

    return {
        "total_task_pairs": total_task_pairs,
        "comparable_task_pairs": comparable_task_pairs,
        "pairs_with_conflict": pairs_with_conflict,
        "pair_conflict_ratio_sum": pair_conflict_ratio_sum,
    }


def compute_pair_conflict_stats(
    task_ids: List[str],
    cand_map: Dict[str, List[Candidate]],
    satellites: Dict[str, SchedulingSatellite],
    agility_profile: str,
    non_agile_transition_s: float,
    workers: int,
    chunk_size: int,
    max_parallel_total_candidates: int,
    scenario_start: Optional[datetime] = None,
) -> Dict[str, float]:
    """Calculate task pair conflict statistics. Only this part supports optional parallelisation."""
    n = len(task_ids)
    if n <= 1:
        return {
            "task_conflict_density": 0.0,
            "mean_pair_conflict_ratio": 0.0,
        }

    total_candidates = sum(len(cand_map.get(tid, [])) for tid in task_ids)
    if scenario_start is None:
        starts = []
        for tid in task_ids:
            for c in cand_map.get(tid, []):
                starts.append(c.sat_start_time)
                if c.gs_start_time is not None:
                    starts.append(c.gs_start_time)
        scenario_start = min(starts) if starts else datetime(1970, 1, 1)

    meta_map = build_task_pairwise_meta(task_ids=task_ids, cand_map=cand_map, scenario_start=scenario_start)
    chunks = _build_workload_balanced_chunks(task_ids, meta_map, max(1, int(chunk_size)))
    total_chunk = len(chunks)

    agg = {
        "total_task_pairs": 0,
        "comparable_task_pairs": 0,
        "pairs_with_conflict": 0,
        "pair_conflict_ratio_sum": 0.0,
    }

    use_parallel = (
        workers > 1 and
        n >= 40 and
        total_candidates <= int(max_parallel_total_candidates)
    )

    context = {
        "task_ids": task_ids,
        "meta_map": meta_map,
        "satellites": satellites,
        "agility_profile": agility_profile,
        "non_agile_transition_s": non_agile_transition_s,
    }

    if not use_parallel:
        if workers > 1 and total_candidates > int(max_parallel_total_candidates):
            log(
                f"Total candidates = {total_candidates} exceeds the parallel threshold {max_parallel_total_candidates}. "
                f"Falling back to sequential mode to avoid memory pressure."
            )
        else:
            log("Starting to calculate task pair conflicts sequentially...")

        init_pairwise_worker(context)
        for idx, (i_start, i_end) in enumerate(chunks, start=1):
            part = pair_conflict_worker(i_start, i_end)
            for k in agg:
                agg[k] += part[k]
            if idx % 5 == 0 or idx == total_chunk:
                log(f"Task pair conflict statistics progress: {idx}/{total_chunk}")
    else:
        log(f"Starting to calculate task pair conflicts in parallel, workers={workers} ...")
        ctx = mp.get_context("spawn")
        with ProcessPoolExecutor(
            max_workers=workers,
            mp_context=ctx,
            initializer=init_pairwise_worker,
            initargs=(context,),
        ) as executor:
            futures = [executor.submit(pair_conflict_worker, i_start, i_end) for i_start, i_end in chunks]
            done_cnt = 0
            for fut in as_completed(futures):
                part = fut.result()
                for k in agg:
                    agg[k] += part[k]
                done_cnt += 1
                if done_cnt % 5 == 0 or done_cnt == total_chunk:
                    log(f"Task pair conflict statistics progress: {done_cnt}/{total_chunk}")

    total_task_pairs = agg["total_task_pairs"]
    comparable_task_pairs = agg["comparable_task_pairs"]
    pairs_with_conflict = agg["pairs_with_conflict"]
    pair_conflict_ratio_sum = agg["pair_conflict_ratio_sum"]

    task_conflict_density = pairs_with_conflict / total_task_pairs if total_task_pairs > 0 else 0.0
    mean_pair_conflict_ratio = pair_conflict_ratio_sum / comparable_task_pairs if comparable_task_pairs > 0 else 0.0

    return {
        "task_conflict_density": float(task_conflict_density),
        "mean_pair_conflict_ratio": float(mean_pair_conflict_ratio),
    }


# =============================================================================
# 8) Continuous Time Contention Index
# =============================================================================

def compute_continuous_contention_index(
    intervals_by_resource_task: Dict[Tuple[str, str], List[Tuple[datetime, datetime]]]
) -> float:
    """
    Continuous Time Contention Index:
    For each resource, tally the number of simultaneously active tasks n(t) along the time axis,
    then compute: CI = ∫ C(n(t),2) dt / ∫ max(n(t),1) dt
    """
    by_resource: Dict[str, List[Tuple[datetime, datetime]]] = defaultdict(list)

    for (resource_id, _task_id), intervals in intervals_by_resource_task.items():
        merged = merge_intervals(intervals)
        for s, e in merged:
            if e > s:
                by_resource[resource_id].append((s, e))

    numerator = 0.0
    denominator = 0.0

    for _resource_id, intervals in by_resource.items():
        events: List[Tuple[datetime, int]] = []
        for s, e in intervals:
            events.append((s, +1))
            events.append((e, -1))

        if not events:
            continue

        events.sort(key=lambda x: (x[0], x[1]))
        active = 0
        prev_t = events[0][0]
        idx = 0
        m = len(events)

        while idx < m:
            cur_t = events[idx][0]
            dt = (cur_t - prev_t).total_seconds()
            if dt > 0:
                numerator += (active * (active - 1) / 2.0) * dt
                denominator += max(active, 1) * dt

            while idx < m and events[idx][0] == cur_t and events[idx][1] == -1:
                active += events[idx][1]
                idx += 1
            while idx < m and events[idx][0] == cur_t and events[idx][1] == +1:
                active += events[idx][1]
                idx += 1

            prev_t = cur_t

    if denominator <= 0:
        return 0.0
    return float(numerator / denominator)


# =============================================================================
# 9) Main Analysis Flow for a Single Scenario
# =============================================================================

def analyze_conflict_degree(
    problem: SchedulingProblem,
    file_name: str,
    rel_path: str,
    workers: int,
    agility_profile: str,
    non_agile_transition_s: float,
    downlink_duration_ratio: float,
    pair_chunk_size: int,
    observation_step_multiplier: int,
    max_candidates_per_task: Optional[int],
    max_parallel_total_candidates: int,
) -> ConflictAnalysisResult:
    t0 = time.time()

    log(f"Starting to analyse scenario: {problem.scenario_id}")
    n_tasks = len(problem.tasks)
    n_sats = len(problem.satellites)
    log(f"Satellites={n_sats}, Tasks={n_tasks}, Ground Stations={len(problem.ground_stations)}")

    # A) Candidate Construction (Sequential)
    cand_map = build_candidate_map(
        problem=problem,
        downlink_duration_ratio=downlink_duration_ratio,
        observation_step_multiplier=observation_step_multiplier,
        max_candidates_per_task=max_candidates_per_task,
    )

    candidate_counts = [len(cand_map.get(tid, [])) for tid in problem.tasks.keys()]
    avg_candidates_per_task = (sum(candidate_counts) / n_tasks) if n_tasks > 0 else 0.0
    hard_task_ratio_k2 = sum(1 for x in candidate_counts if x <= 2) / n_tasks if n_tasks > 0 else 0.0

    total_candidates = sum(candidate_counts)
    log(f"Candidate construction completed, total candidates = {total_candidates}")

    # B) Task Pair Conflict Statistics (Optional Parallelisation)
    task_ids = list(problem.tasks.keys())
    pair_stats = compute_pair_conflict_stats(
        task_ids=task_ids,
        cand_map=cand_map,
        satellites=problem.satellites,
        agility_profile=agility_profile,
        non_agile_transition_s=non_agile_transition_s,
        workers=workers,
        chunk_size=pair_chunk_size,
        max_parallel_total_candidates=max_parallel_total_candidates,
        scenario_start=problem.start_time,
    )

    log("Task pair conflict statistics completed.")

    # C) Observation Contention Index
    obs_intervals_by_sat_task: Dict[Tuple[str, str], List[Tuple[datetime, datetime]]] = defaultdict(list)
    for _tid, cands in cand_map.items():
        for c in cands:
            obs_intervals_by_sat_task[(c.satellite_id, c.task_id)].append((c.sat_start_time, c.sat_end_time))

    observation_contention_index = compute_continuous_contention_index(obs_intervals_by_sat_task)
    log(f"Observation contention index calculation completed: CI_obs={observation_contention_index:.6f}")

    t1 = time.time()
    elapsed = t1 - t0
    log(f"Scenario analysis completed. Total time: {elapsed:.2f}s")

    del cand_map
    gc.collect()

    return ConflictAnalysisResult(
        scenario_id=problem.scenario_id,
        file_name=file_name,
        rel_path=rel_path,
        n_satellites=n_sats,
        n_tasks=n_tasks,
        avg_candidates_per_task=float(avg_candidates_per_task),
        hard_task_ratio_k2=float(hard_task_ratio_k2),
        task_conflict_density=float(pair_stats["task_conflict_density"]),
        mean_pair_conflict_ratio=float(pair_stats["mean_pair_conflict_ratio"]),
        observation_contention_index=float(observation_contention_index),
        analysis_seconds=float(elapsed),
        note="",
    )