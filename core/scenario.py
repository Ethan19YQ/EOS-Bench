# -*- coding: utf-8 -*-
"""
core/scenario.py

Main functionality:
This module defines the scenario container and related utilities for mission generation,
visibility window calculation, scenario validation, and JSON export. It supports both
random mission generation and mission loading from target files.
"""

from __future__ import annotations

import json
import random
import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime, timedelta, timezone

from .models import (
    Satellite,
    Mission,
    MissionInfo,
    GroundStation,
    ObservationRequirement,
    ObservationWindow,
    CommunicationWindow,
    TimeWindow,
)

from utils.visibility import calculate_windows_orekit


MU_EARTH = 3.986004418e14  # m^3/s^2 (WGS84)


def _to_utc(dt: datetime) -> datetime:
    """Treat naive datetime as UTC; convert aware datetime to UTC."""
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


# ----------------------------------------------------------------------------
# Target file reading cache (one copy per process)
# ----------------------------------------------------------------------------

_CITY_TARGETS_CACHE: Dict[str, Dict[str, Tuple[float, float]]] = {}


def _parse_lat_lon(value: Any) -> Tuple[float, float]:
    """Parse value into (lat, lon).

    Compatible with two common formats:
    - [lon, lat]
    - [lat, lon]

    Identification rules:
    - If the 1st value is in [-180, 180] and the 2nd is in [-90, 90],
      it looks more like (lon, lat)
    - If the 1st is in [-90, 90] and the 2nd is in [-180, 180],
      it looks more like (lat, lon)
    - Otherwise, default to (lon, lat)
    """
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError(f"Target coordinate format error, expected list/tuple of length 2, got: {value!r}")

    a = float(value[0])
    b = float(value[1])

    a_is_lon = -180.0 <= a <= 180.0
    b_is_lat = -90.0 <= b <= 90.0

    a_is_lat = -90.0 <= a <= 90.0
    b_is_lon = -180.0 <= b <= 180.0

    # Looks more like [lon, lat]
    if a_is_lon and b_is_lat and not (a_is_lat and b_is_lon):
        lon, lat = a, b
        return lat, lon

    # Looks more like [lat, lon]
    if a_is_lat and b_is_lon and not (a_is_lon and b_is_lat):
        lat, lon = a, b
        return lat, lon

    # Both look possible or neither looks valid: default to [lon, lat]
    lon, lat = a, b
    return lat, lon


def load_city_targets(cities_json_path: str) -> Dict[str, Tuple[float, float]]:
    """Read the city target file and return {name: (lat, lon)}.

    - cities_json_path: full path or relative path
    - The return value is cached within the process
    """
    if cities_json_path in _CITY_TARGETS_CACHE:
        return _CITY_TARGETS_CACHE[cities_json_path]

    with open(cities_json_path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    if not isinstance(raw, dict):
        raise ValueError(f"The top level of the city target file must be a dict, actually got: {type(raw)}")

    parsed: Dict[str, Tuple[float, float]] = {}
    for name, coord in raw.items():
        lat, lon = _parse_lat_lon(coord)
        parsed[str(name)] = (round(float(lat), 6), round(float(lon), 6))

    if not parsed:
        raise ValueError(f"City target file is empty: {cities_json_path}")

    _CITY_TARGETS_CACHE[cities_json_path] = parsed
    return parsed


@dataclass
class ScenarioMetadata:
    """Scenario metadata."""

    name: str
    creation_time: datetime
    duration: float
    time_step: float
    description: Optional[str] = None
    extra: Dict[str, Any] = field(default_factory=dict)

    @property
    def end_time(self) -> datetime:
        return self.creation_time + timedelta(seconds=self.duration)


class ScenarioType(Enum):
    BASIC = "basic"
    MULTI_SAT_MULTI_GS = "multi_sat_multi_gs"
    OTHER = "other"


class Scenario:
    """Scenario object that organizes satellites, missions, and ground stations,
    and provides window calculation and JSON export."""
    def __init__(
        self,
        scenario_id: str,
        satellites: List[Satellite],
        missions: List[Mission],
        ground_stations: List[GroundStation],
        mission_info: MissionInfo,
        metadata: ScenarioMetadata,
        scenario_type: Optional[ScenarioType] = None,
    ) -> None:
        self.id = scenario_id
        self.satellites = satellites
        self.missions = missions
        self.ground_stations = ground_stations
        self.mission_info = mission_info
        self.metadata = metadata
        self.scenario_type = scenario_type or self._infer_scenario_type()

        # Pre-calculate the orbital period for each satellite
        # for outputting the orbit_number field
        self._orbit_period_s: Dict[str, float] = {}
        for sat in self.satellites:
            try:
                a_km = float(sat.orbital_params.semi_major_axis)
                a_m = a_km * 1000.0
                if a_m > 0:
                    period_s = 2.0 * math.pi * math.sqrt((a_m ** 3) / MU_EARTH)
                    if period_s > 0:
                        self._orbit_period_s[str(sat.id)] = float(period_s)
            except Exception:
                continue

        self._observation_windows: List[ObservationWindow] = []
        self._communication_windows: List[CommunicationWindow] = []

        self._validate()

    # ------------------------------------------------------------------
    # Basic properties
    # ------------------------------------------------------------------

    @property
    def start_time(self) -> datetime:
        return self.metadata.creation_time

    @property
    def end_time(self) -> datetime:
        return self.metadata.end_time

    @property
    def duration(self) -> float:
        return self.metadata.duration

    @property
    def time_step(self) -> float:
        return self.metadata.time_step

    # ------------------------------------------------------------------
    # Validation and type inference
    # ------------------------------------------------------------------

    def _infer_scenario_type(self) -> ScenarioType:
        ns, nm, ng = len(self.satellites), len(self.missions), len(self.ground_stations)
        if ns <= 3 and nm <= 20 and ng <= 3:
            return ScenarioType.BASIC
        if ns > 10 or nm > 200 or ng > 5:
            return ScenarioType.MULTI_SAT_MULTI_GS
        return ScenarioType.OTHER

    def _validate(self) -> None:
        if self.metadata.duration <= 0:
            raise ValueError("metadata.duration must be positive")
        if self.metadata.time_step <= 0:
            raise ValueError("metadata.time_step must be positive")

        sat_ids = [s.id for s in self.satellites]
        if len(sat_ids) != len(set(sat_ids)):
            raise ValueError("Satellite IDs must be unique")
        mis_ids = [m.id for m in self.missions]
        if len(mis_ids) != len(set(mis_ids)):
            raise ValueError("Mission IDs must be unique")
        gs_ids = [g.id for g in self.ground_stations]
        if len(gs_ids) != len(set(gs_ids)):
            raise ValueError("Ground station IDs must be unique")

    # ------------------------------------------------------------------
    # Visibility window calculation (Orekit)
    # ------------------------------------------------------------------

    def calculate_visibility_windows(
        self,
        ensure_each_mission_has_window: bool = True,
        data_path: Optional[str] = None,
        min_elevation_deg: float = 5.0,
        coarse_step_min: float = 60.0,
        max_retries: int = 50,
    ) -> Dict[str, List]:
        """Calculate observation and communication windows."""

        obs_w, comm_w, updated_missions = calculate_windows_orekit(
            scenario_name=self.metadata.name,
            start_time=self.start_time,
            end_time=self.end_time,
            satellites=self.satellites,
            missions=self.missions,
            ground_stations=self.ground_stations,
            mission_info=self.mission_info,
            step_seconds=self.time_step,
            ensure_each_mission_has_window=ensure_each_mission_has_window,
            min_elevation_deg=min_elevation_deg,
            coarse_step_min=coarse_step_min,
            max_retries=max_retries,
            data_path=data_path,
        )

        self._observation_windows = obs_w
        self._communication_windows = comm_w
        self.missions = updated_missions

        return {
            "observation_windows": self._observation_windows,
            "communication_windows": self._communication_windows,
        }

    # ------------------------------------------------------------------
    # Window querying
    # ------------------------------------------------------------------

    def get_observation_windows(
        self,
        satellite_id: Optional[str] = None,
        mission_id: Optional[str] = None,
        sensor_id: Optional[str] = None,
    ) -> List[ObservationWindow]:
        windows = self._observation_windows
        if satellite_id is not None:
            windows = [w for w in windows if w.satellite_id == satellite_id]
        if mission_id is not None:
            windows = [w for w in windows if w.mission_id == mission_id]
        if sensor_id is not None:
            windows = [w for w in windows if w.sensor_id == sensor_id]
        return windows

    def get_communication_windows(
        self,
        satellite_id: Optional[str] = None,
        station_id: Optional[str] = None,
    ) -> List[CommunicationWindow]:
        windows = self._communication_windows
        if satellite_id is not None:
            windows = [w for w in windows if w.satellite_id == satellite_id]
        if station_id is not None:
            windows = [w for w in windows if w.ground_station_id == station_id]
        return windows

    # ------------------------------------------------------------------
    # JSON export
    # ------------------------------------------------------------------

    def _time_window_to_dict(self, tw: TimeWindow, satellite_id: Optional[str] = None) -> Dict[str, Any]:
        out: Dict[str, Any] = {
            "start_time": tw.start_time.isoformat(),
            "end_time": tw.end_time.isoformat(),
        }
        # Prefer to use tw.orbit_number written during visibility calculation
        if getattr(tw, "orbit_number", None) is not None:
            out["orbit_number"] = int(getattr(tw, "orbit_number"))
        # Compatible with old data:
        # if not provided, estimate it from the satellite orbital period,
        # counted as orbit 1 from the scenario start
        elif satellite_id is not None:
            period_s = self._orbit_period_s.get(str(satellite_id))
            if period_s and period_s > 0:
                dt_s = max(0.0, (_to_utc(tw.start_time) - _to_utc(self.start_time)).total_seconds())
                out["orbit_number"] = int(dt_s // period_s) + 1
            else:
                out["orbit_number"] = None
        else:
            out["orbit_number"] = None
        if tw.agile_data is not None:
            out["agile_data"] = {
                "pitch_angles": tw.agile_data.pitch_angles,
                "yaw_angles": tw.agile_data.yaw_angles,
                "roll_angles": tw.agile_data.roll_angles,
            }
        if tw.non_agile_data is not None:
            out["non_agile_data"] = {
                "pitch_angle": tw.non_agile_data.pitch_angle,
                "yaw_angle": tw.non_agile_data.yaw_angle,
                "roll_angle": tw.non_agile_data.roll_angle,
            }
        return out

    def export_to_json(self, filename: str, include_windows: bool = True) -> None:
        data: Dict[str, Any] = {
            "scenario_id": self.id,
            "scenario_type": self.scenario_type.value,
            "metadata": {
                "name": self.metadata.name,
                "creation_time": self.metadata.creation_time.isoformat(),
                "duration": self.metadata.duration,
                "time_step": self.metadata.time_step,
                "description": self.metadata.description,
                "extra": self.metadata.extra,
            },
            "satellites": [s.to_dict() for s in self.satellites],
            "missions": [m.to_dict() for m in self.missions],
        }

        # Do not export the ground_stations field when there are no ground stations
        if self.ground_stations:
            data["ground_stations"] = [gs.to_dict() for gs in self.ground_stations]
        else:
            # Keep the original logic:
            # do not export satellite communication_capability when there are no ground stations
            # to reduce JSON size
            for sat in data["satellites"]:
                sat.pop("communication_capability", None)

        if include_windows:
            data["observation_windows"] = [
                {
                    "satellite_id": ow.satellite_id,
                    "sensor_id": ow.sensor_id,
                    "mission_id": ow.mission_id,
                    "time_windows": [self._time_window_to_dict(tw, satellite_id=ow.satellite_id) for tw in ow.time_window],
                }
                for ow in self._observation_windows
            ]
            if self.ground_stations and self._communication_windows:
                data["communication_windows"] = [
                    {
                        "satellite_id": cw.satellite_id,
                        "ground_station_id": cw.ground_station_id,
                        "time_windows": [self._time_window_to_dict(tw, satellite_id=cw.satellite_id) for tw in cw.time_window],
                    }
                    for cw in self._communication_windows
                ]

        with open(filename, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)


# -----------------------------------------------------------------------------
# Mission generation (called by main_generate.py)
# -----------------------------------------------------------------------------


def generate_missions_dict(
    num_missions: int,
    mission_info: MissionInfo,
    priority_range: Tuple[int, int] = (1, 10),
    duration_range: Tuple[int, int] = (5, 15),
    mission_id_prefix: str = "M",
    targets_file_path: Optional[str] = None,
) -> Dict[str, Tuple[float, float, float, ObservationRequirement]]:
    """Generate a mission point dictionary: {mission_id: (lat, lon, priority, obs_req)}.

    Two modes:
    - targets_file_path=None: randomly generate according to the MissionInfo range,
      mission_id uses prefix + sequence number
    - targets_file_path=...: read the city target file,
      mission_id directly uses the key in the file

    When reading target files:
    only (name, lat, lon) are used, while priority and duration are still
    generated according to the random rules.
    """

    missions: Dict[str, Tuple[float, float, float, ObservationRequirement]] = {}

    # -------------------------
    # Mode 2: read target points from file
    # -------------------------
    if targets_file_path:
        # Read all targets in the file:
        # - The number of missions is determined by the file
        #   and num_missions / missions_number is ignored
        # - mission_id directly uses the JSON key
        targets = load_city_targets(targets_file_path)

        for base_name, (lat, lon) in targets.items():
            priority = random.randint(priority_range[0], priority_range[1])
            duration = random.randint(duration_range[0], duration_range[1])
            obs_req = ObservationRequirement(duration=float(duration))
            missions[str(base_name)] = (float(lat), float(lon), float(priority), obs_req)

        return missions

    # -------------------------
    # Mode 1: random generation by range
    # -------------------------

    if num_missions <= 0:
        return {}

    for idx in range(1, num_missions + 1):
        if mission_info.distribution_type == 0:
            lat_range = mission_info.latitude_range[0]
            lon_range = mission_info.longitude_range[0]
        else:
            region_idx = random.randint(0, len(mission_info.latitude_range) - 1)
            lat_range = mission_info.latitude_range[region_idx]
            lon_range = mission_info.longitude_range[region_idx]

        latitude = round(random.uniform(lat_range[0], lat_range[1]), 6)
        longitude = round(random.uniform(lon_range[0], lon_range[1]), 6)

        priority = random.randint(priority_range[0], priority_range[1])
        duration = random.randint(duration_range[0], duration_range[1])

        obs_req = ObservationRequirement(duration=float(duration))
        mission_id = f"{mission_id_prefix}{idx:03d}"
        missions[mission_id] = (latitude, longitude, float(priority), obs_req)

    return missions