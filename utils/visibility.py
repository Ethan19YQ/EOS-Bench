# -*- coding: utf-8 -*-
"""
utils/visibility.py

Main functionality:
This module provides Orekit-based visibility window calculation utilities for
scenario generation. It initializes and reuses the Orekit runtime environment,
computes observation and communication visibility windows, evaluates satellite
attitude constraints, assigns orbit numbers, and converts the results into the
data structures used by the scenario model.
"""

from __future__ import annotations

import os
import math
import random
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Dict, Iterable, List, Optional, Tuple

# ====== Domain models (from core/models.py) ======
from core.models import (
    Satellite,
    Mission,
    GroundStation,
    MissionInfo,
    TimeWindow,
    ObservationWindow,
    CommunicationWindow,
    AgileData,
    NonAgileData,
)

# -----------------------------------------------------------------------------
# 0) Orekit initialization (done only once per process)
# -----------------------------------------------------------------------------

_VM_STARTED = False
_DATA_LOADED = False
_CONTEXT: "_OrekitContext | None" = None


def _to_utc(dt: datetime) -> datetime:
    """Treat naive datetime as UTC; convert aware datetime to UTC."""
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)




def _orbit_number_from_time(
    *,
    window_start: datetime,
    scenario_start: datetime,
    orbit_period_s: float | None,
) -> int | None:
    """Calculate which orbit number the window belongs to, starting from 1.

    Convention:
    - The orbit containing the scenario start time is orbit 1
    - window start_time is used as the assignment timestamp
    - orbit_period_s is pre-calculated by Orekit Orbit.getKeplerianPeriod()
    """
    if orbit_period_s is None or orbit_period_s <= 0:
        return None
    dt_s = max(0.0, (_to_utc(window_start) - _to_utc(scenario_start)).total_seconds())
    return int(dt_s // float(orbit_period_s)) + 1

def _find_orekit_data_zip(data_path: Optional[str] = None) -> str:
    """Locate orekit-data.zip."""
    # 1) Explicitly specified
    if data_path and os.path.exists(data_path):
        return data_path

    # 2) Default: utils/orekit-data.zip (same directory as this file)
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    cand = os.path.join(cur_dir, "orekit-data.zip")
    if os.path.exists(cand):
        return cand

    # 3) Current working directory
    cand = os.path.join(os.getcwd(), "orekit-data.zip")
    if os.path.exists(cand):
        return cand

    raise FileNotFoundError(
        "orekit-data.zip not found. Please place it in:\n"
        "  1) utils/orekit-data.zip\n"
        "or 2) the current working directory\n"
        "or specify it explicitly via calculate_windows_orekit(data_path=...)."
    )


def ensure_orekit(data_path: Optional[str] = None) -> None:
    """Ensure the JVM is started and Orekit data is loaded, only once per process."""
    global _VM_STARTED, _DATA_LOADED, _CONTEXT

    # 1) Start JVM (can only be called once)
    if not _VM_STARTED:
        import orekit_jpype as orekit

        # Optional: if orekit-jpype[jdk4py] is installed, automatically set JAVA_HOME
        try:
            import jdk4py  # type: ignore

            os.environ.setdefault("JAVA_HOME", str(jdk4py.JAVA_HOME))
        except Exception:
            pass

        orekit.initVM()
        _VM_STARTED = True

    # 2) Load orekit-data
    if not _DATA_LOADED:
        from orekit_jpype.pyhelpers import setup_orekit_curdir

        zip_path = _find_orekit_data_zip(data_path)
        setup_orekit_curdir(zip_path)
        _DATA_LOADED = True

    # 3) Initialize context
    if _CONTEXT is None:
        _CONTEXT = _OrekitContext.create()


# -----------------------------------------------------------------------------
# 1) Orekit context (reuse whenever possible)
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class _OrekitContext:
    """Common reusable context object for Orekit calculations."""

    # Java objects (type hints are kept flexible to avoid mypy/JVM conflicts)
    inertial: object
    itrf: object
    earth: object
    utc: object
    mu: float

    @staticmethod
    def create() -> "_OrekitContext":
        # Lazy import of Java classes
        from org.orekit.frames import FramesFactory
        from org.orekit.utils import IERSConventions, Constants
        from org.orekit.bodies import OneAxisEllipsoid
        from org.orekit.time import TimeScalesFactory

        inertial = FramesFactory.getEME2000()
        itrf = FramesFactory.getITRF(IERSConventions.IERS_2010, True)

        ae = Constants.WGS84_EARTH_EQUATORIAL_RADIUS
        f = Constants.WGS84_EARTH_FLATTENING
        earth = OneAxisEllipsoid(ae, f, itrf)

        utc = TimeScalesFactory.getUTC()
        mu = float(Constants.WGS84_EARTH_MU)

        return _OrekitContext(inertial=inertial, itrf=itrf, earth=earth, utc=utc, mu=mu)


# KeplerianOrbit cache: the same satellite may be reused for multiple missions
# within a single process
_ORBIT_CACHE: Dict[str, object] = {}


def _orbit_cache_key(
    a_m: float,
    e: float,
    i_deg: float,
    raan_deg: float,
    argp_deg: float,
    m_deg: float,
    epoch_iso: str,
) -> str:
    return (
        f"a={a_m:.3f}|e={e:.8f}|i={i_deg:.6f}|raan={raan_deg:.6f}|"
        f"argp={argp_deg:.6f}|m={m_deg:.6f}|epoch={epoch_iso}"
    )


def _build_keplerian_orbit(
    ctx: _OrekitContext,
    a_m: float,
    e: float,
    i_deg: float,
    raan_deg: float,
    argp_deg: float,
    m_deg: float,
    epoch_dt: datetime,
) -> object:
    """Construct a KeplerianOrbit, or reuse it from cache."""
    from org.orekit.orbits import KeplerianOrbit, PositionAngleType
    from orekit_jpype.pyhelpers import datetime_to_absolutedate

    epoch_dt = _to_utc(epoch_dt)
    epoch_iso = epoch_dt.isoformat()
    key = _orbit_cache_key(a_m, e, i_deg, raan_deg, argp_deg, m_deg, epoch_iso)

    if key in _ORBIT_CACHE:
        return _ORBIT_CACHE[key]

    orbit = KeplerianOrbit(
        float(a_m),
        float(e),
        math.radians(float(i_deg)),
        math.radians(float(argp_deg)),
        math.radians(float(raan_deg)),
        math.radians(float(m_deg)),
        PositionAngleType.MEAN,
        ctx.inertial,
        datetime_to_absolutedate(epoch_dt),
        ctx.mu,
    )
    _ORBIT_CACHE[key] = orbit
    return orbit


# -----------------------------------------------------------------------------
# 2) Single pair (satellite-target) visibility and attitude calculation
# -----------------------------------------------------------------------------


def compute_visibility_and_attitude(
    *,
    a_m: float,
    e: float,
    i_deg: float,
    raan_deg: float,
    argp_deg: float,
    m_deg: float,
    epoch_dt: datetime,
    start_dt: datetime,
    end_dt: datetime,
    step_s: float,
    target_lat_deg: float,
    target_lon_deg: float,
    target_alt_m: float,
    max_yaw_deg: float,
    max_pitch_deg: float,
    max_roll_deg: float,
    min_elevation_deg: float = 5.0,
    coarse_step_min: float = 60.0,
) -> Tuple[List[Tuple[datetime, datetime]], List[dict], List[dict]]:
    """Return (visibility_windows, samples, visible_attitudes).
    """
    # Ensure Orekit context is initialized, once per process in multiprocessing
    if _CONTEXT is None:
        ensure_orekit(None)
    assert _CONTEXT is not None
    ctx = _CONTEXT


    # Java imports
    from org.orekit.frames import TopocentricFrame
    from org.orekit.bodies import GeodeticPoint
    from org.orekit.propagation.analytical import KeplerianPropagator
    from org.orekit.attitudes import TargetPointing
    from org.hipparchus.geometry.euclidean.threed import Vector3D, RotationOrder, RotationConvention
    from orekit_jpype.pyhelpers import datetime_to_absolutedate

    # -------- Standardize time as UTC-aware --------
    epoch_dt = _to_utc(epoch_dt)
    start_dt = _to_utc(start_dt)
    end_dt = _to_utc(end_dt)

    if step_s <= 0:
        raise ValueError("step_s must be a positive number")


    # -------- Target point and attitude model --------
    target_geo = GeodeticPoint(
        math.radians(float(target_lat_deg)),
        math.radians(float(target_lon_deg)),
        float(target_alt_m),
    )
    target_frame = TopocentricFrame(ctx.earth, target_geo, "TARGET")

    attitude_provider = TargetPointing(ctx.inertial, target_geo, ctx.earth)

    # Orbit and propagator (orbit can be cached)
    orbit = _build_keplerian_orbit(ctx, a_m, e, i_deg, raan_deg, argp_deg, m_deg, epoch_dt)
    propagator = KeplerianPropagator(orbit, attitude_provider)

    min_el_rad = math.radians(float(min_elevation_deg))

    step_fine = float(step_s)
    coarse_step = max(step_fine, float(coarse_step_min))

    fine_td = timedelta(seconds=step_fine)
    coarse_td = timedelta(seconds=coarse_step)

    # The target position in ITRF is reusable and will be transformed to the
    # inertial frame at each timestamp
    target_in_itrf = ctx.earth.transform(target_geo)

    def eval_full(dt: datetime) -> dict:
        """Propagate once and compute elevation, off-nadir, yaw, pitch, roll, and visibility."""
        date = datetime_to_absolutedate(dt)
        state = propagator.propagate(date)
        pv = state.getPVCoordinates(ctx.inertial)
        sat_pos = pv.getPosition()

        elevation_rad = target_frame.getElevation(sat_pos, ctx.inertial, date)
        elevation_deg = math.degrees(float(elevation_rad))

        visible = False
        off_nadir_deg = None
        yaw_deg = pitch_deg = roll_deg = None

        if elevation_rad > min_el_rad:
            # target -> inertial
            transform = ctx.itrf.getTransformTo(ctx.inertial, date)
            target_in_inertial = transform.transformPosition(target_in_itrf)

            los_vec = target_in_inertial.subtract(sat_pos)
            los_unit = los_vec.normalize()
            nadir_vec = sat_pos.negate().normalize()

            off_nadir_rad = Vector3D.angle(los_unit, nadir_vec)
            off_nadir_deg = math.degrees(float(off_nadir_rad))

            try:
                rotation = state.getAttitude().getRotation()
                angles = rotation.getAngles(RotationOrder.ZYX, RotationConvention.FRAME_TRANSFORM)
                yaw_deg = math.degrees(float(angles[0]))
                pitch_deg = math.degrees(float(angles[1]))
                roll_deg = math.degrees(float(angles[2]))

                if (
                    abs(yaw_deg) <= float(max_yaw_deg)
                    and abs(pitch_deg) <= float(max_pitch_deg)
                    and abs(roll_deg) <= float(max_roll_deg)
                ):
                    visible = True
            except Exception:
                # If attitude angle solving fails, treat it as unavailable
                visible = False

        return {
            "time": dt,
            "visible": visible,
            "elevation_deg": elevation_deg,
            "off_nadir_deg": off_nadir_deg,
            "yaw_deg": yaw_deg,
            "pitch_deg": pitch_deg,
            "roll_deg": roll_deg,
        }

    # ------------------------------------------------------------------
    # A) step_fine >= coarse_step_min: directly scan the full range
    # ------------------------------------------------------------------
    if step_fine >= coarse_step_min - 1e-9:
        samples: List[dict] = []
        visible_attitudes: List[dict] = []
        windows: List[Tuple[datetime, datetime]] = []

        last_visible = False
        window_start: Optional[datetime] = None

        t = start_dt
        while t <= end_dt:
            s = eval_full(t)
            samples.append(s)

            if s["visible"]:
                visible_attitudes.append(
                    {
                        "time": s["time"],
                        "elevation_deg": s["elevation_deg"],
                        "off_nadir_deg": s["off_nadir_deg"],
                        "yaw_deg": s["yaw_deg"],
                        "pitch_deg": s["pitch_deg"],
                        "roll_deg": s["roll_deg"],
                    }
                )

            if s["visible"] and not last_visible:
                window_start = t
            elif (not s["visible"]) and last_visible and window_start is not None:
                windows.append((window_start, t))
                window_start = None

            last_visible = bool(s["visible"])
            t += fine_td

        if last_visible and window_start is not None:
            windows.append((window_start, end_dt))

        return windows, samples, visible_attitudes

    # ------------------------------------------------------------------
    # B) Two-stage algorithm:
    #    coarse scan for geometrically visible elevation intervals,
    #    then fine scan for attitude constraints inside candidate intervals
    # ------------------------------------------------------------------

    # 1) Coarse scan: only check elevation > min_elevation
    geom_intervals: List[Tuple[datetime, datetime]] = []
    last_geom = False
    geom_start: Optional[datetime] = None

    t = start_dt
    while t <= end_dt:
        date = datetime_to_absolutedate(t)
        state = propagator.propagate(date)
        sat_pos = state.getPVCoordinates(ctx.inertial).getPosition()
        el = target_frame.getElevation(sat_pos, ctx.inertial, date)
        geom_vis = bool(el > min_el_rad)

        if geom_vis and not last_geom:
            geom_start = t
        elif (not geom_vis) and last_geom and geom_start is not None:
            geom_intervals.append((geom_start, t))
            geom_start = None

        last_geom = geom_vis
        t += coarse_td

    if last_geom and geom_start is not None:
        geom_intervals.append((geom_start, end_dt))

    if not geom_intervals:
        return [], [], []

    # 2) Fine scan inside candidate intervals
    samples = []
    visible_attitudes = []
    windows: List[Tuple[datetime, datetime]] = []

    last_visible = False
    window_start = None

    for (g_start, g_end) in geom_intervals:
        local_start = max(start_dt, g_start - fine_td)
        local_end = min(end_dt, g_end + fine_td)

        t = local_start
        while t <= local_end:
            s = eval_full(t)
            samples.append(s)

            if s["visible"]:
                visible_attitudes.append(
                    {
                        "time": s["time"],
                        "elevation_deg": s["elevation_deg"],
                        "off_nadir_deg": s["off_nadir_deg"],
                        "yaw_deg": s["yaw_deg"],
                        "pitch_deg": s["pitch_deg"],
                        "roll_deg": s["roll_deg"],
                    }
                )

            if s["visible"] and not last_visible:
                window_start = t
            elif (not s["visible"]) and last_visible and window_start is not None:
                windows.append((window_start, t))
                window_start = None

            last_visible = bool(s["visible"])
            t += fine_td

    if last_visible and window_start is not None:
        windows.append((window_start, end_dt))

    return windows, samples, visible_attitudes


# -----------------------------------------------------------------------------
# 3) Mission and satellite object to orbit parameter and constraint mapping
# -----------------------------------------------------------------------------


def _extract_orbit_params(sat: Satellite) -> Dict[str, object]:
    """Uniformly extract Satellite.orbital_params into the six elements and epoch required by Orekit."""
    op = sat.orbital_params

    # Existing model convention: semi_major_axis is in km, convert to m here
    a_m = float(op.semi_major_axis) * 1000.0

    return {
        "a_m": a_m,
        "e": float(op.eccentricity),
        "i_deg": float(op.inclination),
        "raan_deg": float(op.right_ascension_of_ascending_node),
        "argp_deg": float(op.argument_of_perigee),
        "m_deg": float(op.mean_anomaly),
        "epoch_dt": _parse_epoch(op.initial_representation_Epoch),
    }


def _parse_epoch(epoch: object) -> datetime:
    """Support multiple epoch formats for compatibility with existing JSON and string formats."""
    if isinstance(epoch, datetime):
        return _to_utc(epoch)

    # Number: treated as UNIX seconds
    if isinstance(epoch, (int, float)):
        return datetime.fromtimestamp(float(epoch), tz=timezone.utc)

    s = str(epoch).strip()

    # ISO 8601
    try:
        dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
        return _to_utc(dt)
    except Exception:
        pass

    # Fallback for common formats
    for fmt in (
        "%Y-%m-%d %H:%M:%S",
        "%Y/%m/%d %H:%M:%S",
        "%Y-%m-%dT%H:%M:%S",
        "%Y/%m/%dT%H:%M:%S",
    ):
        try:
            dt = datetime.strptime(s, fmt)
            return dt.replace(tzinfo=timezone.utc)
        except Exception:
            continue

    # Final fallback: treat as current UTC time to avoid an immediate crash.
    # This may affect visibility results, so the input should be corrected early.
    return datetime.now(tz=timezone.utc)


def _get_sat_att_limits_deg(sat: Satellite, default: Tuple[float, float, float]) -> Tuple[float, float, float]:
    """Extract attitude limits from Satellite.maneuverability_capability."""
    mc = getattr(sat, "maneuverability_capability", None)
    if mc is None:
        return default
    # Convention in models.py: max_pitch, max_yaw, and max_roll are all in degrees
    return (float(mc.max_yaw_angle), float(mc.max_pitch_angle), float(mc.max_roll_angle))


def generate_random_coordinates(mission_info: MissionInfo) -> Tuple[float, float]:
    """Randomly generate (lat, lon) according to the distribution settings in MissionInfo."""
    if mission_info.distribution_type == 0:
        lat_min, lat_max = mission_info.latitude_range[0]
        lon_min, lon_max = mission_info.longitude_range[0]
    elif mission_info.distribution_type == 1:
        idx = random.randint(0, len(mission_info.latitude_range) - 1)
        lat_min, lat_max = mission_info.latitude_range[idx]
        lon_min, lon_max = mission_info.longitude_range[idx]
    else:
        raise ValueError(f"Unsupported distribution_type: {mission_info.distribution_type}")

    lat = round(random.uniform(lat_min, lat_max), 4)
    lon = round(random.uniform(lon_min, lon_max), 4)
    return lat, lon


def _middle_value(values: List[float]) -> float:
    if not values:
        return 0.0
    return values[len(values) // 2]


def _build_time_windows_from_visible_attitudes(
    visibility_windows: List[Tuple[datetime, datetime]],
    visible_attitudes: List[dict],
    *,
    scenario_start_dt: datetime,
    orbit_period_s: float | None,
) -> List[TimeWindow]:
    """Organize the results of compute_visibility_and_attitude into a list of models.TimeWindow."""
    # Build an index to make slicing by window easier
    times = [va["time"] for va in visible_attitudes]

    tws: List[TimeWindow] = []
    for (ws, we) in visibility_windows:
        # Visible sample points inside the window.
        idxs = [i for i, t in enumerate(times) if (t >= ws and t < we + timedelta(microseconds=1))]
        if not idxs:
            continue

        pitch_list = [float(visible_attitudes[i]["pitch_deg"]) for i in idxs]
        yaw_list = [float(visible_attitudes[i]["yaw_deg"]) for i in idxs]
        roll_list = [float(visible_attitudes[i]["roll_deg"]) for i in idxs]

        agile = AgileData(pitch_angles=pitch_list, yaw_angles=yaw_list, roll_angles=roll_list)
        non_agile = NonAgileData(
            pitch_angle=_middle_value(pitch_list),
            yaw_angle=_middle_value(yaw_list),
            roll_angle=_middle_value(roll_list),
        )

        orbit_no = _orbit_number_from_time(window_start=ws, scenario_start=scenario_start_dt, orbit_period_s=orbit_period_s)


        tws.append(TimeWindow(start_time=ws, end_time=we, orbit_number=orbit_no, agile_data=agile, non_agile_data=non_agile))

    return tws


# -----------------------------------------------------------------------------
# 4) Scenario-level window calculation with Orekit (supports ensure toggle)
# -----------------------------------------------------------------------------


def calculate_windows_orekit(
    *,
    scenario_name: str,
    start_time: datetime,
    end_time: datetime,
    satellites: List[Satellite],
    missions: List[Mission],
    ground_stations: List[GroundStation],
    mission_info: MissionInfo,
    step: Optional[float] = None,
    step_seconds: Optional[float] = None,
    ensure_each_mission_has_window: bool = True,
    max_resample_tries: int = 50,
    max_retries: Optional[int] = None,
    min_elevation_deg: float = 5.0,
    coarse_step_min: float = 60.0,
    data_path: Optional[str] = None,
    progress: bool = True,
    progress_every: int = 1,
) -> Tuple[List[ObservationWindow], List[CommunicationWindow], List[Mission]]:

    if step is None:
        if step_seconds is None:
            raise TypeError("calculate_windows_orekit() must provide step or step_seconds (unit: seconds)")
        step = float(step_seconds)
    else:
        step = float(step)

    if max_retries is not None:
        max_resample_tries = int(max_retries)

    # data_path specifies the location of orekit-data.zip.
    # In multiprocessing, it is loaded only once per process.
    ensure_orekit(data_path)

    DEFAULT_ATT_LIMITS = (45.0, 45.0, 45.0)  # yaw, pitch, roll

    # 1) Pre-extract satellite orbit parameters and attitude limits
    orbit_map: Dict[str, Dict[str, object]] = {}
    att_limit_map: Dict[str, Tuple[float, float, float]] = {}

    for sat in satellites:
        sat_id = str(getattr(sat, "id", None) or getattr(sat, "satellite_id", ""))
        if not sat_id:
            continue
        orbit_map[sat_id] = _extract_orbit_params(sat)
        att_limit_map[sat_id] = _get_sat_att_limits_deg(sat, DEFAULT_ATT_LIMITS)

    # 1.1) Pre-calculate the Keplerian period in seconds for each satellite
    # using Orekit Orbit.getKeplerianPeriod()
    period_map: Dict[str, float] = {}
    if _CONTEXT is not None:
        ctx = _CONTEXT
        for _sat_id, _op in orbit_map.items():
            try:
                _orbit = _build_keplerian_orbit(
                    ctx=ctx,
                    a_m=float(_op["a_m"]),
                    e=float(_op["e"]),
                    i_deg=float(_op["i_deg"]),
                    raan_deg=float(_op["raan_deg"]),
                    argp_deg=float(_op["argp_deg"]),
                    m_deg=float(_op["m_deg"]),
                    epoch_dt=_op["epoch_dt"],
                )
                _period_s = float(_orbit.getKeplerianPeriod())
                if _period_s > 0:
                    period_map[_sat_id] = _period_s
            except Exception:
                # For rare anomalous inputs, fall back to leaving orbit_number empty
                continue

    observation_windows: List[ObservationWindow] = []
    communication_windows: List[CommunicationWindow] = []

    # 2) Missions may need coordinate resampling,
    # so create a shallow copy of the list while modifying the objects in place
    updated_missions: List[Mission] = list(missions)
    # 3) Observation windows
    total_missions = len(updated_missions)
    for mission_idx, mission in enumerate(updated_missions, start=1):
        mission_id = str(getattr(mission, "id", None) or getattr(mission, "mission_id", ""))
        # Progress output is independent of ensure_each_mission_has_window
        if progress and (mission_idx == 1 or mission_idx % max(1, int(progress_every)) == 0 or mission_idx == total_missions):
            import os as _os
            print(
                f"[INFO] Scenario {scenario_name} (pid={_os.getpid()}): Computing mission {mission_idx}/{total_missions} (mission_id={mission_id})",
                flush=True,
            )
        tgt = getattr(mission, "target_location", None)
        if not mission_id or tgt is None:
            continue

        tries = 0
        while True:
            lat = getattr(tgt, "latitude", None)
            lon = getattr(tgt, "longitude", None)
            alt_km = float(getattr(tgt, "altitude", 0.0) or 0.0)

            if lat is None or lon is None:
                break

            mission_has_window = False
            tmp_obs_windows_for_this_mission: List[ObservationWindow] = []

            for sat in satellites:
                sat_id = str(getattr(sat, "id", None) or getattr(sat, "satellite_id", ""))
                if not sat_id or sat_id not in orbit_map:
                    continue

                yaw_lim, pitch_lim, roll_lim = att_limit_map.get(sat_id, DEFAULT_ATT_LIMITS)

                vis_windows, _samples, visible_attitudes = compute_visibility_and_attitude(
                    a_m=float(orbit_map[sat_id]["a_m"]),
                    e=float(orbit_map[sat_id]["e"]),
                    i_deg=float(orbit_map[sat_id]["i_deg"]),
                    raan_deg=float(orbit_map[sat_id]["raan_deg"]),
                    argp_deg=float(orbit_map[sat_id]["argp_deg"]),
                    m_deg=float(orbit_map[sat_id]["m_deg"]),
                    epoch_dt=orbit_map[sat_id]["epoch_dt"],  # type: ignore
                    start_dt=start_time,
                    end_dt=end_time,
                    step_s=float(step),
                    target_lat_deg=float(lat),
                    target_lon_deg=float(lon),
                    target_alt_m=float(alt_km) * 1000.0,
                    max_yaw_deg=yaw_lim,
                    max_pitch_deg=pitch_lim,
                    max_roll_deg=roll_lim,
                    min_elevation_deg=float(min_elevation_deg),
                    coarse_step_min=float(coarse_step_min),
                )

                if not vis_windows:
                    continue

                # The satellite has at least one window for this mission
                tws = _build_time_windows_from_visible_attitudes(
                    vis_windows,
                    visible_attitudes,
                    scenario_start_dt=start_time,
                    orbit_period_s=period_map.get(sat_id),
                )
                if not tws:
                    continue

                mission_has_window = True

                obs_win = ObservationWindow(
                    satellite_id=sat_id,
                    sensor_id=f"{sat_id}_sensor",
                    mission_id=mission_id,
                    time_window=tws,
                )
                tmp_obs_windows_for_this_mission.append(obs_win)

            if mission_has_window:
                # If forced-window mode is enabled and resampling occurred,
                # a prompt could be printed here for data-generation debugging
                # if ensure_each_mission_has_window and tries > 0:
                #     print(f"Mission {mission_id} found observation window after {tries + 1} attempts", flush=True)
                observation_windows.extend(tmp_obs_windows_for_this_mission)
                break

            # No windows found at all
            if not ensure_each_mission_has_window:
                break

            tries += 1
            if tries >= max_resample_tries:
                # Reached the retry limit and still found no window,
                # so stop forcing to avoid an infinite loop
                break

            # print(f"Mission {mission_id} did not find an observation window, reallocating position (attempt {tries})", flush=True)
            new_lat, new_lon = generate_random_coordinates(mission_info)
            tgt.latitude = new_lat
            tgt.longitude = new_lon

    # 4) Communication windows: skip if ground_stations is empty
    if ground_stations:
        for sat in satellites:
            sat_id = str(getattr(sat, "id", None) or getattr(sat, "satellite_id", ""))
            if not sat_id or sat_id not in orbit_map:
                continue

            for gs in ground_stations:
                gs_id = str(getattr(gs, "id", None) or getattr(gs, "station_id", ""))
                loc = getattr(gs, "location", None)
                if not gs_id or loc is None:
                    continue

                lat = getattr(loc, "latitude", None)
                lon = getattr(loc, "longitude", None)
                alt_km = float(getattr(loc, "altitude", 0.0) or 0.0)
                if lat is None or lon is None:
                    continue

                vis_windows, _samples, _visible_attitudes = compute_visibility_and_attitude(
                    a_m=float(orbit_map[sat_id]["a_m"]),
                    e=float(orbit_map[sat_id]["e"]),
                    i_deg=float(orbit_map[sat_id]["i_deg"]),
                    raan_deg=float(orbit_map[sat_id]["raan_deg"]),
                    argp_deg=float(orbit_map[sat_id]["argp_deg"]),
                    m_deg=float(orbit_map[sat_id]["m_deg"]),
                    epoch_dt=orbit_map[sat_id]["epoch_dt"],  # type: ignore
                    start_dt=start_time,
                    end_dt=end_time,
                    step_s=float(step),
                    target_lat_deg=float(lat),
                    target_lon_deg=float(lon),
                    target_alt_m=float(alt_km) * 1000.0,
                    # Communication windows are not subject to attitude constraints,
                    # so the limits are relaxed to 180 degrees
                    max_yaw_deg=180.0,
                    max_pitch_deg=180.0,
                    max_roll_deg=180.0,
                    min_elevation_deg=float(min_elevation_deg),
                    coarse_step_min=float(coarse_step_min),
                )

                if not vis_windows:
                    continue

                tws: List[TimeWindow] = []


                for (ws, we) in vis_windows:


                    orbit_no = _orbit_number_from_time(window_start=ws, scenario_start=start_time, orbit_period_s=period_map.get(sat_id))


                    tws.append(TimeWindow(start_time=ws, end_time=we, orbit_number=orbit_no, agile_data=None, non_agile_data=None))
                communication_windows.append(
                    CommunicationWindow(satellite_id=sat_id, ground_station_id=gs_id, time_window=tws)
                )

    return observation_windows, communication_windows, updated_missions