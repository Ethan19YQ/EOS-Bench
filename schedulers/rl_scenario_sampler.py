# -*- coding: utf-8 -*-
"""
rl_scenario_sampler.py

Main functionality:
This module builds temporary training scenarios for reinforcement learning by
randomly sampling satellites, missions, planning horizons, capacity modes, and
agility profiles from an existing scenario JSON file. It also clips observation
and communication windows to the sampled time range and writes the sampled
scenario to a temporary JSON file.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import json
import random
import re
from datetime import datetime, timedelta, timezone


# -------------------------
# helpers
# -------------------------


def _parse_iso(s: str) -> datetime:
    # Keep consistent with scenario_loader._parse_iso_time
    # while avoiding circular dependency here
    s = (s or "").strip()
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    dt = datetime.fromisoformat(s)
    if dt.tzinfo is not None:
        dt = dt.astimezone(timezone.utc).replace(tzinfo=None)
    return dt


def _format_iso(dt: datetime) -> str:
    # Output a unified +00:00 string for simplicity
    # Note: the loader strips tzinfo consistently, so it does not matter
    # whether tzinfo is present here
    return dt.replace(microsecond=0).isoformat() + "+00:00"


def _slice_angle_data(data: Any, start_idx: int, end_idx: int) -> Any:
    """Robust slicing for agile_data and non_agile_data.

    Supports:
    - list: direct slicing
    - dict: apply the same slice to list values in the dict, while preserving others
    - other types: return as is
    """
    if data is None:
        return None
    if isinstance(data, list):
        return data[start_idx:end_idx]
    if isinstance(data, dict):
        out = {}
        for k, v in data.items():
            if isinstance(v, list):
                out[k] = v[start_idx:end_idx]
            else:
                out[k] = v
        return out
    return data


def parse_scale_from_name(stem: str) -> Tuple[Optional[int], Optional[int], Optional[float]]:
    """Parse (Sats, M, T_days) from the scenario filename stem.
    Return None if parsing fails.
    """
    m1 = re.search(r"Sats(\d+)", stem)
    m2 = re.search(r"_M(\d+)", stem)
    m3 = re.search(r"_T([0-9.]+)d", stem)
    sats = int(m1.group(1)) if m1 else None
    tasks = int(m2.group(1)) if m2 else None
    tdays = float(m3.group(1)) if m3 else None
    return sats, tasks, tdays


CAPACITY_MODES = [
    "Low-Capacity",
    "Standard-Capacity",
    "High-Capacity",
    "Mixed-Capacity A",
    "Mixed-Capacity B",
    "Mixed-Capacity C",
]

AGILITY_PROFILES = [
    "High-Agility",
    "Standard-Agility",
    "Low-Agility",
    "Limited-Agility",
]


def _apply_capacity_mode_to_satellites(sats: List[dict], mode: str, rng: random.Random) -> None:
    """Modify the raw JSON satellites list in place by adjusting per-orbit max capacity.

    Note:
    The real benchmark may define capacity modes more precisely.
    Here, diversity is created by scaling max_data_storage_GB and max_power_W,
    while sensor consumption remains unchanged so constraints become tighter or looser.
    """

    def get_specs(sat: dict) -> dict:
        specs = sat.get("satellite_specs")
        if not isinstance(specs, dict):
            specs = {}
            sat["satellite_specs"] = specs
        return specs

    def scale_one(sat: dict, factor: float) -> None:
        specs = get_specs(sat)
        if "max_data_storage_GB" in specs and isinstance(specs["max_data_storage_GB"], (int, float)):
            specs["max_data_storage_GB"] = float(specs["max_data_storage_GB"]) * factor
        if "max_power_W" in specs and isinstance(specs["max_power_W"], (int, float)):
            specs["max_power_W"] = float(specs["max_power_W"]) * factor

    n = len(sats)
    if n <= 0:
        return

    if mode == "Low-Capacity":
        for s in sats:
            scale_one(s, 0.6)
        return
    if mode == "Standard-Capacity":
        # No change
        return
    if mode == "High-Capacity":
        for s in sats:
            scale_one(s, 1.6)
        return

    # Mixed modes: group satellites and scale proportionally
    idx = list(range(n))
    rng.shuffle(idx)

    if mode == "Mixed-Capacity A":
        # 1/3 low, 1/3 standard, 1/3 high
        a = n // 3
        b = 2 * n // 3
        low = idx[:a]
        std = idx[a:b]
        high = idx[b:]
    elif mode == "Mixed-Capacity B":
        # 25% low, 50% standard, 25% high
        a = n // 4
        b = 3 * n // 4
        low = idx[:a]
        std = idx[a:b]
        high = idx[b:]
    else:  # Mixed-Capacity C
        # 50% low, 25% standard, 25% high
        a = n // 2
        b = 3 * n // 4
        low = idx[:a]
        std = idx[a:b]
        high = idx[b:]

    for i in low:
        scale_one(sats[i], 0.6)
    for i in high:
        scale_one(sats[i], 1.6)
    # Standard group remains unchanged


@dataclass
class SampledScenarioInfo:
    base_json: Path
    tmp_json: Path
    sampled_sats: int
    sampled_tasks: int
    sampled_days: float
    capacity_mode: str
    agility_profile: str


def build_temp_training_scenario(
    base_json: Path,
    *,
    out_dir: Path,
    rng: random.Random,
) -> SampledScenarioInfo:
    """Generate a temporary training scenario file from base_json and return its metadata."""
    raw = json.loads(Path(base_json).read_text(encoding="utf-8"))

    sats_base, tasks_base, days_base = parse_scale_from_name(Path(base_json).stem)

    # Fallback: if parsing fails, use actual counts
    sats_list = raw.get("satellites") or raw.get("satellite") or []
    missions_list = raw.get("missions") or raw.get("tasks") or []
    if sats_base is None:
        sats_base = len(sats_list)
    if tasks_base is None:
        tasks_base = len(missions_list)
    if days_base is None:
        # Compute from time fields
        try:
            st = _parse_iso(raw.get("start_time") or raw.get("scenario_start_time") or raw.get("start"))
            et = _parse_iso(raw.get("end_time") or raw.get("scenario_end_time") or raw.get("end"))
            days_base = max(0.5, (et - st).total_seconds() / 86400.0)
        except Exception:
            days_base = 1.0

    # Randomly generate scale within the range 1..base
    sampled_sats = rng.randint(1, max(1, int(sats_base)))
    sampled_tasks = rng.randint(1, max(1, int(tasks_base)))

    # T': use 0.5 day as the step size
    steps = max(1, int(round(days_base / 0.5)))
    sampled_days = rng.randint(1, steps) * 0.5

    # Randomly select capacity mode and agility profile
    capacity_mode = rng.choice(CAPACITY_MODES)
    agility_profile = rng.choice(AGILITY_PROFILES)

    # Select satellites
    sat_ids = [s.get("id") for s in sats_list if isinstance(s, dict) and s.get("id") is not None]
    sat_ids = [str(x) for x in sat_ids]
    rng.shuffle(sat_ids)
    keep_sat_ids = set(sat_ids[:sampled_sats])
    sats_new = [s for s in sats_list if isinstance(s, dict) and str(s.get("id")) in keep_sat_ids]

    # Select missions
    mission_ids = [m.get("id") for m in missions_list if isinstance(m, dict) and m.get("id") is not None]
    mission_ids = [str(x) for x in mission_ids]
    rng.shuffle(mission_ids)
    keep_mission_ids = set(mission_ids[:sampled_tasks])
    missions_new = [m for m in missions_list if isinstance(m, dict) and str(m.get("id")) in keep_mission_ids]

    # Clip the time range
    # Field names depend on what the loader supports, so this tries to stay compatible
    start_key = "start_time" if "start_time" in raw else ("scenario_start_time" if "scenario_start_time" in raw else None)
    end_key = "end_time" if "end_time" in raw else ("scenario_end_time" if "scenario_end_time" in raw else None)
    if start_key and end_key:
        st = _parse_iso(raw[start_key])
        et = st + timedelta(days=sampled_days)
        raw[start_key] = _format_iso(st)
        raw[end_key] = _format_iso(et)
    else:
        st = None
        et = None

    # Filter observation_windows
    obs_wins = raw.get("observation_windows") or []
    obs_new: List[dict] = []
    for w in obs_wins:
        if not isinstance(w, dict):
            continue
        sid = str(w.get("satellite_id"))
        mid = str(w.get("mission_id"))
        if sid not in keep_sat_ids or mid not in keep_mission_ids:
            continue

        # If st and et exist, clip to [st, et]
        if st is not None and et is not None:
            try:
                ws = _parse_iso(w.get("start_time"))
                we = _parse_iso(w.get("end_time"))
            except Exception:
                obs_new.append(w)
                continue

            # Skip if there is no intersection
            if we <= st or ws >= et:
                continue

            # Clip
            clip_s = max(ws, st)
            clip_e = min(we, et)
            if clip_e <= clip_s:
                continue

            time_step = float(w.get("time_step", 1.0) or 1.0)
            # Slice index by time step
            start_idx = int(max(0.0, round((clip_s - ws).total_seconds() / time_step)))
            end_idx = int(max(0.0, round((clip_e - ws).total_seconds() / time_step)))
            if end_idx <= start_idx:
                continue

            # Update time and angle data
            w = dict(w)
            w["start_time"] = _format_iso(clip_s)
            w["end_time"] = _format_iso(clip_e)
            if "agile_data" in w:
                w["agile_data"] = _slice_angle_data(w.get("agile_data"), start_idx, end_idx)
            if "non_agile_data" in w:
                w["non_agile_data"] = _slice_angle_data(w.get("non_agile_data"), start_idx, end_idx)
            obs_new.append(w)
        else:
            obs_new.append(w)

    # Filter communication windows
    comm_wins = raw.get("communication_windows") or []
    comm_new: List[dict] = []
    for w in comm_wins:
        if not isinstance(w, dict):
            continue
        sid = str(w.get("satellite_id"))
        if sid not in keep_sat_ids:
            continue
        if st is not None and et is not None:
            try:
                ws = _parse_iso(w.get("start_time"))
                we = _parse_iso(w.get("end_time"))
            except Exception:
                comm_new.append(w)
                continue
            if we <= st or ws >= et:
                continue
            w = dict(w)
            w["start_time"] = _format_iso(max(ws, st))
            w["end_time"] = _format_iso(min(we, et))
        comm_new.append(w)

    # Write back
    raw["satellites"] = sats_new
    raw["missions"] = missions_new
    raw["observation_windows"] = obs_new
    if "communication_windows" in raw:
        raw["communication_windows"] = comm_new

    # Apply capacity mode
    _apply_capacity_mode_to_satellites(raw["satellites"], capacity_mode, rng)

    # Record sampling metadata for easier troubleshooting in training logs
    meta = raw.get("metadata")
    if not isinstance(meta, dict):
        meta = {}
        raw["metadata"] = meta
    meta["rl_sampling"] = {
        "base_json": str(Path(base_json).name),
        "sampled_sats": sampled_sats,
        "sampled_tasks": sampled_tasks,
        "sampled_days": sampled_days,
        "capacity_mode": capacity_mode,
        "agility_profile": agility_profile,
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    tmp_path = out_dir / f"tmp_rl_{Path(base_json).stem}_s{sampled_sats}_m{sampled_tasks}_t{sampled_days:g}_{rng.randrange(10**9)}.json"
    tmp_path.write_text(json.dumps(raw, ensure_ascii=False, indent=2), encoding="utf-8")

    return SampledScenarioInfo(
        base_json=Path(base_json).resolve(),
        tmp_json=tmp_path.resolve(),
        sampled_sats=sampled_sats,
        sampled_tasks=sampled_tasks,
        sampled_days=sampled_days,
        capacity_mode=capacity_mode,
        agility_profile=agility_profile,
    )