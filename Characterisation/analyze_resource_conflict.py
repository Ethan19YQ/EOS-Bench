# -*- coding: utf-8 -*-
"""
Created on Mon Mar 23 2026

@author: QianY 

analyze_resource_conflict.py

Main features:
1. Parse the scheduling problem from a scenario JSON file.
2. Analyse the degree of conflict in the scenario from a "satellite resource timeline" perspective.
3. Analyse resource conflicts satellite by satellite, considering observation window overlaps and insufficient transition times.
4. Support optional parallelisation across satellites with progress logging.
5. Provide a slimmed-down data structure explicitly feeding into the combined analysis tool.

Notes:
- The reading method and transition time model are kept consistent with analyze_conflict_degree.py.
- Stripped out unused metrics (GINI, Top 10% share, missing transition seconds, etc.) to optimise execution speed.
- All comments and console outputs use British English.
"""

import gc
import json
import math
import multiprocessing as mp
import re
import time
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple


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
C_SMALL = 11.66


# =============================================================================
# 1) Data Structures
# =============================================================================

@dataclass
class SchedulingSatellite:
    id: str
    maneuverability_type: str = "agile"


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
        return max(0.0, (self.end_time - self.start_time).total_seconds())


@dataclass
class SchedulingProblem:
    scenario_id: str
    start_time: datetime
    end_time: datetime
    global_time_step: float
    satellites: Dict[str, SchedulingSatellite]
    tasks: Dict[str, dict]
    task_windows: List[TaskWindow]


@dataclass
class ConflictSegment:
    start_time: datetime
    end_time: datetime
    task_ids: Tuple[str, ...]
    conflict_task_count: int
    has_overlap_conflict: bool
    has_transition_conflict: bool

    @property
    def duration_s(self) -> float:
        return max(0.0, (self.end_time - self.start_time).total_seconds())

    @property
    def conflict_source_label(self) -> str:
        if self.has_overlap_conflict and self.has_transition_conflict:
            return "Mixed"
        if self.has_transition_conflict:
            return "Transition"
        return "Overlap"


@dataclass
class SatelliteConflictResult:
    """Slimmed down satellite result dataclass containing only required attributes."""
    satellite_id: str
    conflict_step_count: int
    peak_conflict_task_count: int
    mean_conflict_task_count_on_conflict_steps: float
    conflict_duration_s: float = 0.0
    weighted_excess_demand_area: float = 0.0
    conflict_segments: List[ConflictSegment] = field(default_factory=list)


@dataclass
class ScenarioResourceConflictResult:
    """Slimmed down scenario result dataclass containing only necessary metrics."""
    scenario_id: str
    file_name: str
    rel_path: str
    n_satellites: int
    n_tasks: int
    
    satellite_conflict_ratio: float
    mean_conflict_task_count_on_conflict_steps: float
    total_conflict_duration_s: float
    peak_conflict_task_count: int
    conflict_coverage_ratio: float
    weighted_excess_demand_area: float
    avg_elasticity_score: float

    analysis_seconds: float
    note: str = ""


# =============================================================================
# 2) Logging
# =============================================================================

def log(msg: str) -> None:
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
    return AGILITY_PROFILES.get(normalize_profile_name(profile_name), AGILITY_PROFILES["Standard-Agility"])


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


def get_max_transition_time_upper_bound(profile_name: str, max_delta_g_deg: float) -> float:
    return compute_transition_time_agile(max(10.0, float(max_delta_g_deg)), profile_name)


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
    return abs(a2["roll"] - a1["roll"]) + abs(a2["pitch"] - a1["pitch"]) + abs(a2["yaw"] - a1["yaw"])


def sanitize_filename(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", str(name))


def _apply_task_counter_delta(counter_obj: Counter, task_ids: Iterable[str], delta: int) -> None:
    for tid in task_ids:
        counter_obj[tid] += delta
        if counter_obj[tid] <= 0:
            counter_obj.pop(tid, None)


# =============================================================================
# 4) File Scanning
# =============================================================================

def is_probable_scenario_json(json_path: Path, output_dir: Path) -> bool:
    name = json_path.name.lower()
    if not name.endswith(".json"):
        return False
    if name.startswith("conflict_analysis_") or name.startswith("resource_conflict_") or name.startswith("scheduler_"):
        return False
    try:
        rel = json_path.resolve().relative_to(output_dir.resolve())
        if "schedules" in rel.parts:
            return False
    except Exception:
        pass
    return name.startswith("scenario_")


def iter_scenario_jsons(output_dir: Path) -> Iterable[Path]:
    for p in output_dir.rglob("*.json"):
        if p.is_file() and is_probable_scenario_json(p, output_dir):
            yield p.resolve()


def count_scenario_jsons(output_dir: Path) -> int:
    return sum(1 for _ in iter_scenario_jsons(output_dir))


def resolve_single_scenario_path(scenario_file: str, output_dir: Path) -> Path:
    p = Path(scenario_file)
    if p.exists():
        return p.resolve()
    name = scenario_file
    if not name.lower().endswith(".json"):
        name += ".json"
    candidate = output_dir / name
    if candidate.exists():
        return candidate.resolve()
    raise FileNotFoundError(f"Scenario file not found: {scenario_file}, attempted path: {candidate}")


# =============================================================================
# 5) JSON Loading
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
        satellites[sat["id"]] = SchedulingSatellite(
            id=sat["id"],
            maneuverability_type=infer_maneuverability_type(sat),
        )

    tasks: Dict[str, dict] = {}
    for mission in data.get("missions", []):
        tasks[mission["id"]] = {"id": mission["id"]}

    task_windows: List[TaskWindow] = []
    window_counter = 0
    for ow in data.get("observation_windows", []):
        sat_id = ow["satellite_id"]
        mission_id = ow["mission_id"]
        if sat_id not in satellites:
            continue
        for tw in ow.get("time_windows", []):
            start = parse_iso_time(tw["start_time"])
            end = parse_iso_time(tw["end_time"])
            if end <= start:
                continue
            window_counter += 1
            task_windows.append(
                TaskWindow(
                    window_id=window_counter,
                    satellite_id=sat_id,
                    mission_id=mission_id,
                    sensor_id=str(ow.get("sensor_id", "")),
                    orbit_number=int(tw.get("orbit_number", ow.get("orbit_number", 0)) or 0),
                    start_time=start,
                    end_time=end,
                    time_step=float(tw.get("time_step", ow.get("time_step", global_time_step)) or global_time_step or 1.0),
                    agile_data=tw.get("agile_data", ow.get("agile_data")),
                    non_agile_data=tw.get("non_agile_data", ow.get("non_agile_data")),
                )
            )

    return SchedulingProblem(
        scenario_id=scenario_id,
        start_time=start_time,
        end_time=end_time,
        global_time_step=global_time_step,
        satellites=satellites,
        tasks=tasks,
        task_windows=task_windows,
    )


# =============================================================================
# 6) Core Analysis
# =============================================================================

def transition_time_s(
    satellite: SchedulingSatellite,
    prev_w: TaskWindow,
    next_w: TaskWindow,
    agility_profile: str,
    non_agile_transition_s: float,
    default_transition_when_angle_missing_s: float,
) -> float:
    if (satellite.maneuverability_type or "agile").lower() != "agile":
        return float(non_agile_transition_s)
    dg = delta_g_between(prev_w.agile_data, next_w.agile_data)
    if dg is None:
        return float(default_transition_when_angle_missing_s)
    return float(compute_transition_time_agile(dg, agility_profile))


def build_satellite_window_map(task_windows: List[TaskWindow]) -> Dict[str, List[TaskWindow]]:
    d: Dict[str, List[TaskWindow]] = defaultdict(list)
    for w in task_windows:
        d[w.satellite_id].append(w)
    for sid in d:
        d[sid].sort(key=lambda x: (x.start_time, x.end_time, x.mission_id, x.window_id))
    return d


def _segment_to_step_indices(
    seg_start: datetime,
    seg_end: datetime,
    scenario_start: datetime,
    scenario_end: datetime,
    step_s: float,
    total_steps: int,
) -> Tuple[int, int]:
    if seg_end <= seg_start:
        return 1, 0
    clipped_start = max(seg_start, scenario_start)
    clipped_end = min(seg_end, scenario_end)
    if clipped_end <= clipped_start:
        return 1, 0
    start_offset = (clipped_start - scenario_start).total_seconds()
    end_offset = (clipped_end - scenario_start).total_seconds()
    start_idx = max(0, int(math.floor(start_offset / step_s)))
    end_idx = min(total_steps - 1, int(math.ceil(end_offset / step_s) - 1))
    return start_idx, end_idx


def merge_conflict_segments_for_output(segments: List[ConflictSegment]) -> List[ConflictSegment]:
    """
    Merge continuous conflict segments for detailed output:
    As long as the task set is identical and the time is contiguous, they are merged into one longer time window.
    """
    if not segments:
        return []

    ordered = sorted(segments, key=lambda x: (x.start_time, x.end_time, x.task_ids))
    merged: List[ConflictSegment] = []
    eps = 1e-6

    for seg in ordered:
        if not merged:
            merged.append(seg)
            continue

        last = merged[-1]
        is_contiguous = abs((seg.start_time - last.end_time).total_seconds()) <= eps
        same_tasks = seg.task_ids == last.task_ids

        if same_tasks and is_contiguous:
            merged[-1] = ConflictSegment(
                start_time=last.start_time,
                end_time=max(last.end_time, seg.end_time),
                task_ids=last.task_ids,
                conflict_task_count=max(last.conflict_task_count, seg.conflict_task_count),
                has_overlap_conflict=last.has_overlap_conflict or seg.has_overlap_conflict,
                has_transition_conflict=last.has_transition_conflict or seg.has_transition_conflict,
            )
        else:
            merged.append(seg)

    return merged


def _safe_ratio(num: float, den: float) -> float:
    return float(num / den) if den > 0 else 0.0


def analyze_task_elasticity(problem: SchedulingProblem) -> float:
    """Calculates purely the average elasticity score for tasks: |optional_satellites| * |windows|"""
    per_task: Dict[str, Dict[str, Any]] = {}
    for tid in problem.tasks.keys():
        per_task[tid] = {"satellites": set(), "windows": 0}

    for w in problem.task_windows:
        rec = per_task.setdefault(w.mission_id, {"satellites": set(), "windows": 0})
        rec["satellites"].add(w.satellite_id)
        rec["windows"] += 1

    if not per_task:
        return 0.0

    elasticity_scores: List[float] = []
    for rec in per_task.values():
        elasticity_scores.append(float(len(rec["satellites"]) * rec["windows"]))

    return sum(elasticity_scores) / len(per_task)


def analyze_single_satellite_resource_conflict(
    satellite: SchedulingSatellite,
    windows: List[TaskWindow],
    scenario_start: datetime,
    scenario_end: datetime,
    analysis_time_step_s: float,
    agility_profile: str,
    non_agile_transition_s: float,
    default_transition_when_angle_missing_s: float,
    transition_search_horizon_s: float,
) -> SatelliteConflictResult:
    
    total_span_s = max(0.0, (scenario_end - scenario_start).total_seconds())
    total_steps = max(1, int(math.ceil(total_span_s / analysis_time_step_s)))

    if not windows:
        return SatelliteConflictResult(
            satellite_id=satellite.id,
            conflict_step_count=0,
            peak_conflict_task_count=0,
            mean_conflict_task_count_on_conflict_steps=0.0,
        )

    obs_events: List[Tuple[datetime, int, int, str]] = []
    for w in windows:
        obs_events.append((w.start_time, 1, +1, w.mission_id))
        obs_events.append((w.end_time, 0, -1, w.mission_id))

    transition_events: List[Tuple[datetime, int, int, Tuple[str, ...]]] = []

    sorted_windows = sorted(windows, key=lambda x: (x.start_time, x.end_time, x.window_id))
    horizon_s = max(0.0, float(transition_search_horizon_s))
    n = len(sorted_windows)

    for i in range(n):
        wi = sorted_windows[i]
        for j in range(i + 1, n):
            wj = sorted_windows[j]
            time_from_end = (wj.start_time - wi.end_time).total_seconds()
            if time_from_end > horizon_s:
                break
            if wj.start_time < wi.end_time:
                continue
            required_transition_s = transition_time_s(
                satellite=satellite,
                prev_w=wi,
                next_w=wj,
                agility_profile=agility_profile,
                non_agile_transition_s=non_agile_transition_s,
                default_transition_when_angle_missing_s=default_transition_when_angle_missing_s,
            )
            available_gap_s = max(0.0, (wj.start_time - wi.end_time).total_seconds())
            if required_transition_s > available_gap_s + 1e-9 and wi.mission_id != wj.mission_id:
                if wj.start_time > wi.end_time:
                    task_pair = tuple(sorted((wi.mission_id, wj.mission_id)))
                    transition_events.append((wi.end_time, 1, +1, task_pair))
                    transition_events.append((wj.start_time, 0, -1, task_pair))

    all_times: Set[datetime] = set()
    for t, _, _, _ in obs_events:
        all_times.add(t)
    for t, _, _, _ in transition_events:
        all_times.add(t)

    if not all_times:
        return SatelliteConflictResult(
            satellite_id=satellite.id,
            conflict_step_count=0,
            peak_conflict_task_count=0,
            mean_conflict_task_count_on_conflict_steps=0.0,
        )

    obs_events.sort(key=lambda x: (x[0], x[1]))
    transition_events.sort(key=lambda x: (x[0], x[1]))
    event_times = sorted(all_times)

    obs_ptr = 0
    trans_ptr = 0
    obs_counter: Counter = Counter()
    trans_counter: Counter = Counter()
    conflict_segments: List[ConflictSegment] = []
    prev_time = event_times[0]

    for current_time in event_times:
        if current_time > prev_time:
            obs_tasks = set(obs_counter.keys())
            trans_tasks = set(trans_counter.keys())
            union_tasks = obs_tasks | trans_tasks
            obs_conflict_flag = len(obs_tasks) >= 2
            trans_conflict_flag = len(trans_tasks) >= 2
            if len(union_tasks) >= 2:
                conflict_segments.append(
                    ConflictSegment(
                        start_time=prev_time,
                        end_time=current_time,
                        task_ids=tuple(sorted(union_tasks)),
                        conflict_task_count=len(union_tasks),
                        has_overlap_conflict=obs_conflict_flag,
                        has_transition_conflict=trans_conflict_flag,
                    )
                )

        while obs_ptr < len(obs_events) and obs_events[obs_ptr][0] == current_time and obs_events[obs_ptr][1] == 0:
            _, _, delta, task_id = obs_events[obs_ptr]
            _apply_task_counter_delta(obs_counter, [task_id], delta)
            obs_ptr += 1
        while trans_ptr < len(transition_events) and transition_events[trans_ptr][0] == current_time and transition_events[trans_ptr][1] == 0:
            _, _, delta, task_pair = transition_events[trans_ptr]
            _apply_task_counter_delta(trans_counter, task_pair, delta)
            trans_ptr += 1
        while obs_ptr < len(obs_events) and obs_events[obs_ptr][0] == current_time and obs_events[obs_ptr][1] == 1:
            _, _, delta, task_id = obs_events[obs_ptr]
            _apply_task_counter_delta(obs_counter, [task_id], delta)
            obs_ptr += 1
        while trans_ptr < len(transition_events) and transition_events[trans_ptr][0] == current_time and transition_events[trans_ptr][1] == 1:
            _, _, delta, task_pair = transition_events[trans_ptr]
            _apply_task_counter_delta(trans_counter, task_pair, delta)
            trans_ptr += 1

        prev_time = current_time

    step_map: Dict[int, Dict[str, Any]] = {}
    for seg in conflict_segments:
        start_idx, end_idx = _segment_to_step_indices(
            seg.start_time, seg.end_time, scenario_start, scenario_end, analysis_time_step_s, total_steps
        )
        if end_idx < start_idx:
            continue
        for idx in range(start_idx, end_idx + 1):
            rec = step_map.get(idx)
            if rec is None:
                step_map[idx] = {
                    "conflict_task_count": seg.conflict_task_count,
                    "task_ids_len": len(seg.task_ids)
                }
            else:
                rec["conflict_task_count"] = max(rec["conflict_task_count"], seg.conflict_task_count)
                rec["task_ids_len"] = max(rec["task_ids_len"], len(seg.task_ids))

    conflict_task_count_sum = 0
    peak_conflict_task_count = 0

    for idx in step_map.values():
        conflict_task_count = max(int(idx["conflict_task_count"]), idx["task_ids_len"])
        conflict_task_count_sum += conflict_task_count
        peak_conflict_task_count = max(peak_conflict_task_count, conflict_task_count)

    conflict_step_count = len(step_map)
    mean_conflict_task_count = conflict_task_count_sum / conflict_step_count if conflict_step_count > 0 else 0.0
    conflict_duration_s = sum(seg.duration_s for seg in conflict_segments)
    weighted_excess_demand_area = sum(max(0, seg.conflict_task_count - 1) * seg.duration_s for seg in conflict_segments)

    return SatelliteConflictResult(
        satellite_id=satellite.id,
        conflict_step_count=conflict_step_count,
        peak_conflict_task_count=peak_conflict_task_count,
        mean_conflict_task_count_on_conflict_steps=float(mean_conflict_task_count),
        conflict_duration_s=float(conflict_duration_s),
        weighted_excess_demand_area=float(weighted_excess_demand_area),
        conflict_segments=conflict_segments,
    )


# =============================================================================
# 7) Parallel Entry
# =============================================================================

def satellite_worker(payload: Dict[str, Any]) -> SatelliteConflictResult:
    return analyze_single_satellite_resource_conflict(
        satellite=payload["satellite"],
        windows=payload["windows"],
        scenario_start=payload["scenario_start"],
        scenario_end=payload["scenario_end"],
        analysis_time_step_s=payload["analysis_time_step_s"],
        agility_profile=payload["agility_profile"],
        non_agile_transition_s=payload["non_agile_transition_s"],
        default_transition_when_angle_missing_s=payload["default_transition_when_angle_missing_s"],
        transition_search_horizon_s=payload["transition_search_horizon_s"],
    )

def analyze_satellites_with_optional_parallel(
    problem: SchedulingProblem,
    analysis_time_step_s: float,
    workers: int,
    agility_profile: str,
    non_agile_transition_s: float,
    default_transition_when_angle_missing_s: float,
    transition_search_horizon_s: float,
) -> List[SatelliteConflictResult]:
    
    windows_by_sat = build_satellite_window_map(problem.task_windows)
    payloads: List[Dict[str, Any]] = []
    
    for sid in sorted(problem.satellites.keys()):
        payloads.append(
            {
                "satellite": problem.satellites[sid],
                "windows": windows_by_sat.get(sid, []),
                "scenario_start": problem.start_time,
                "scenario_end": problem.end_time,
                "analysis_time_step_s": analysis_time_step_s,
                "agility_profile": agility_profile,
                "non_agile_transition_s": non_agile_transition_s,
                "default_transition_when_angle_missing_s": default_transition_when_angle_missing_s,
                "transition_search_horizon_s": transition_search_horizon_s,
            }
        )

    total = len(payloads)
    if total == 0:
        return []

    real_workers = max(1, int(workers))
    use_parallel = real_workers > 1 and total >= 2
    results: List[SatelliteConflictResult] = []

    if not use_parallel:
        log("Starting sequential analysis of resource conflicts across satellites...")
        for idx, payload in enumerate(payloads, start=1):
            results.append(satellite_worker(payload))
            if idx % 5 == 0 or idx == total:
                log(f"Satellite progress: {idx}/{total}")
        return results

    log(f"Starting parallel analysis of resource conflicts, workers={real_workers} ...")
    ctx = mp.get_context("spawn")
    with ProcessPoolExecutor(max_workers=real_workers, mp_context=ctx) as executor:
        futures = [executor.submit(satellite_worker, payload) for payload in payloads]
        done_cnt = 0
        for fut in as_completed(futures):
            results.append(fut.result())
            done_cnt += 1
            if done_cnt % 5 == 0 or done_cnt == total:
                log(f"Satellite progress: {done_cnt}/{total}")
                
    results.sort(key=lambda x: x.satellite_id)
    return results


# =============================================================================
# 8) Scenario-level Aggregation
# =============================================================================

def analyze_resource_conflict_scenario(
    problem: SchedulingProblem,
    file_name: str,
    rel_path: str,
    workers: int,
    analysis_time_step_s: Optional[float],
    agility_profile: str,
    non_agile_transition_s: float,
    default_transition_when_angle_missing_s: float,
    transition_search_horizon_s: Optional[float],
    long_conflict_thresholds_s: List[float],
    critical_feasible_duration_s: float,
    low_elasticity_threshold: float,
) -> Tuple[ScenarioResourceConflictResult, List[SatelliteConflictResult]]:
    
    t0 = time.time()
    step_s = float(analysis_time_step_s or problem.global_time_step or 1.0)
    if step_s <= 0:
        step_s = 1.0
    if transition_search_horizon_s is None:
        transition_search_horizon_s = max(
            get_max_transition_time_upper_bound(agility_profile, 360.0),
            non_agile_transition_s,
            default_transition_when_angle_missing_s,
        )

    log(f"Starting to analyse scenario: {problem.scenario_id}")
    log(f"Satellites={len(problem.satellites)}, Tasks={len(problem.tasks)}, Windows={len(problem.task_windows)}")

    sat_results = analyze_satellites_with_optional_parallel(
        problem=problem,
        analysis_time_step_s=step_s,
        workers=workers,
        agility_profile=agility_profile,
        non_agile_transition_s=non_agile_transition_s,
        default_transition_when_angle_missing_s=default_transition_when_angle_missing_s,
        transition_search_horizon_s=transition_search_horizon_s,
    )

    satellites_with_conflict = sum(1 for r in sat_results if r.conflict_duration_s > 0)
    total_conflict_steps = sum(r.conflict_step_count for r in sat_results)
    peak_conflict_task_count = max((r.peak_conflict_task_count for r in sat_results), default=0)
    total_conflict_duration_s = sum(r.conflict_duration_s for r in sat_results)
    weighted_excess_demand_area = sum(r.weighted_excess_demand_area for r in sat_results)

    mean_conflict_task_count = 0.0
    if total_conflict_steps > 0:
        weighted_sum = sum(r.mean_conflict_task_count_on_conflict_steps * r.conflict_step_count for r in sat_results)
        mean_conflict_task_count = weighted_sum / total_conflict_steps

    scenario_duration_s = max(0.0, (problem.end_time - problem.start_time).total_seconds())
    denom_sat_time = len(problem.satellites) * scenario_duration_s
    conflict_coverage_ratio = _safe_ratio(total_conflict_duration_s, denom_sat_time)

    avg_elasticity_score = analyze_task_elasticity(problem)

    elapsed = time.time() - t0
    log(f"Scenario analysis completed. Total time: {elapsed:.2f}s")

    scenario_result = ScenarioResourceConflictResult(
        scenario_id=problem.scenario_id,
        file_name=file_name,
        rel_path=rel_path,
        n_satellites=len(problem.satellites),
        n_tasks=len(problem.tasks),
        satellite_conflict_ratio=_safe_ratio(satellites_with_conflict, len(problem.satellites)),
        total_conflict_duration_s=total_conflict_duration_s,
        peak_conflict_task_count=peak_conflict_task_count,
        mean_conflict_task_count_on_conflict_steps=mean_conflict_task_count,
        conflict_coverage_ratio=conflict_coverage_ratio,
        weighted_excess_demand_area=weighted_excess_demand_area,
        avg_elasticity_score=avg_elasticity_score,
        analysis_seconds=elapsed,
        note="",
    )
    return scenario_result, sat_results