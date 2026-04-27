# -*- coding: utf-8 -*-
"""
scenario_loader.py

Main functionality:
This module loads a scheduling scenario from a JSON file and converts it into
in-memory scheduling objects, including satellites, ground stations, tasks,
observation windows, and communication windows.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime, timedelta, timezone


# ==============================
# 1. Basic data structures
# ==============================

@dataclass
class TaskWindow:
    """
    A single observation visibility time window for a specific task
    on a specific satellite.

    Note:
    orbit_number is supported since v21 and is used for per-orbit
    resource constraints.
    """
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
        """Total window length in seconds."""
        return (self.end_time - self.start_time).total_seconds()


@dataclass
class CommWindow:
    """
    Satellite-ground station communication window.

    This is independent of tasks and only describes that a
    satellite-ground station pair can perform data transmission
    during a certain time period.
    """
    window_id: int
    satellite_id: str
    ground_station_id: str
    start_time: datetime
    end_time: datetime

    @property
    def duration(self) -> float:
        return (self.end_time - self.start_time).total_seconds()


@dataclass
class SchedulingTask:
    """
    Scheduling task object corresponding to a mission in JSON.
    """
    id: str
    priority: float
    required_duration: float
    windows: List[TaskWindow] = field(default_factory=list)


@dataclass
class SensorSpec:
    """Sensor parameters used for data volume and energy consumption calculation."""
    sensor_id: str
    data_rate_Mbps: float = 0.0
    power_consumption_W: float = 0.0


@dataclass
class SchedulingSatellite:

    id: str
    maneuverability_type: str = "agile"  # "agile" / "non_agile"

    # Per-orbit limits
    max_data_storage_GB: float = 0.0
    max_power_W: float = 0.0

    # Maneuverability parameters for agile satellites
    slew_rate_deg_per_s: float = 1.0
    stabilization_time_s: float = 0.0

    sensors: Dict[str, SensorSpec] = field(default_factory=dict)


@dataclass
class SchedulingGroundStation:
    """Ground station object from a scheduling perspective."""
    id: str


@dataclass
class SchedulingProblem:

    scenario_id: str
    start_time: datetime
    end_time: datetime
    satellites: Dict[str, SchedulingSatellite]
    ground_stations: Dict[str, SchedulingGroundStation]
    tasks: Dict[str, SchedulingTask]
    comm_windows: List[CommWindow]


# ==============================
# 2. JSON parsing helpers
# ==============================

def _parse_iso_time(time_str: str) -> datetime:

    s = (time_str or "").strip()
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    dt = datetime.fromisoformat(s)
    if dt.tzinfo is not None:
        dt = dt.astimezone(timezone.utc).replace(tzinfo=None)
    return dt


def _infer_maneuverability_type(sat: dict) -> str:

    def _norm(v) -> Optional[str]:
        if v is None:
            return None
        if isinstance(v, str):
            s = v.strip().lower()
            if "non" in s:
                return "non_agile"
            if "agile" in s:
                return "agile"
        return None

    # 0) maneuverability_capability, a common field in this benchmark
    mcap = sat.get("maneuverability_capability")
    if isinstance(mcap, dict):
        v = _norm(mcap.get("maneuverability_type") or mcap.get("type"))
        if v:
            return v

    # 1) Top level
    v = _norm(sat.get("maneuverability_type"))
    if v:
        return v

    # 2) attributes
    attrs = sat.get("attributes")
    if isinstance(attrs, dict):
        v = _norm(attrs.get("maneuverability_type"))
        if v:
            return v

    # 3) payload, single object
    payload = sat.get("payload")
    if isinstance(payload, dict):
        v = _norm(payload.get("maneuverability_type"))
        if v:
            return v
        p_attrs = payload.get("attributes")
        if isinstance(p_attrs, dict):
            v = _norm(p_attrs.get("maneuverability_type"))
            if v:
                return v

    # 4) payloads, list
    payloads = sat.get("payloads")
    if isinstance(payloads, list):
        for p in payloads:
            if not isinstance(p, dict):
                continue
            v = _norm(p.get("maneuverability_type"))
            if v:
                return v
            p_attrs = p.get("attributes")
            if isinstance(p_attrs, dict):
                v = _norm(p_attrs.get("maneuverability_type"))
                if v:
                    return v

    # Default: agile
    return "agile"


def load_scheduling_problem_from_json(json_path: str | Path) -> SchedulingProblem:

    import json

    json_path = Path(json_path)
    if not json_path.exists():
        raise FileNotFoundError(f"Scenario JSON file not found: {json_path}")

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    scenario_id = data.get("scenario_id", json_path.stem)

    # Scenario time information
    meta = data["metadata"]
    start_time = _parse_iso_time(meta["creation_time"])
    duration_s = float(meta["duration"])
    end_time = start_time + timedelta(seconds=duration_s)

    # Global time_step, used as fallback if missing at the window level
    global_time_step = float(meta.get("time_step", 1.0) or 1.0)

    # Satellite list
    satellites: Dict[str, SchedulingSatellite] = {}
    for sat in data.get("satellites", []):
        sat_id = sat["id"]

        # Per-orbit limits
        specs = sat.get("satellite_specs", {}) or {}
        max_data_storage_GB = float(specs.get("max_data_storage_GB", 0.0) or 0.0)
        max_power_W = float(specs.get("max_power_W", 0.0) or 0.0)

        # Maneuverability parameters
        man = sat.get("maneuverability_capability", {}) or {}
        slew_rate = float(man.get("slew_rate_deg_per_s", 1.0) or 1.0)
        stab_time = float(man.get("stabilization_time_s", 0.0) or 0.0)

        # Sensors
        sensors: Dict[str, SensorSpec] = {}
        obs_cap = sat.get("observation_capability", {}) or {}
        for s in obs_cap.get("sensors", []) or []:
            sid = s.get("sensor_id")
            if not sid:
                continue
            sensors[sid] = SensorSpec(
                sensor_id=sid,
                data_rate_Mbps=float(s.get("data_rate_Mbps", 0.0) or 0.0),
                power_consumption_W=float(s.get("power_consumption_W", 0.0) or 0.0),
            )

        satellites[sat_id] = SchedulingSatellite(
            id=sat_id,
            maneuverability_type=_infer_maneuverability_type(sat),
            max_data_storage_GB=max_data_storage_GB,
            max_power_W=max_power_W,
            slew_rate_deg_per_s=slew_rate,
            stabilization_time_s=stab_time,
            sensors=sensors,
        )

    # Ground station list, which may be empty
    ground_stations: Dict[str, SchedulingGroundStation] = {}
    for gs in data.get("ground_stations", []):
        gs_id = gs["id"]
        ground_stations[gs_id] = SchedulingGroundStation(id=gs_id)

    # Task list
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

    # Observation windows -> task-satellite windows
    window_counter = 0
    for ow in data.get("observation_windows", []):
        sat_id = ow["satellite_id"]
        mid = ow["mission_id"]

        if mid not in tasks or sat_id not in satellites:
            continue

        for tw in ow.get("time_windows", []):
            start = _parse_iso_time(tw["start_time"])
            end = _parse_iso_time(tw["end_time"])
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
                time_step=float(tw.get("time_step", ow.get("time_step", global_time_step))),
                agile_data=tw.get("agile_data", ow.get("agile_data")),
                non_agile_data=tw.get("non_agile_data", ow.get("non_agile_data")),
            )
            tasks[mid].windows.append(task_window)

    # Communication windows -> satellite-ground station windows, task-independent
    comm_windows: List[CommWindow] = []
    comm_counter = 0
    for cw in data.get("communication_windows", []):
        sat_id = cw["satellite_id"]
        gs_id = cw["ground_station_id"]
        if sat_id not in satellites or gs_id not in ground_stations:
            continue

        for tw in cw.get("time_windows", []):
            start = _parse_iso_time(tw["start_time"])
            end = _parse_iso_time(tw["end_time"])
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
        satellites=satellites,
        ground_stations=ground_stations,
        tasks=tasks,
        comm_windows=comm_windows,
    )