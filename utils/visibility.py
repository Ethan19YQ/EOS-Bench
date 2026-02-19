# -*- coding: utf-8 -*-
"""utils/visibility.py
说明 / Description
----
包含使用 Orekit 进行卫星与目标、地面站之间可见性计算及姿态模拟的函数。
Contains functions for computing visibility and attitude between satellites, targets, and ground stations using Orekit.
"""

from __future__ import annotations

import os
import math
import random
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Dict, Iterable, List, Optional, Tuple

# ====== 领域模型（来自 core/models.py） / Domain models (from core/models.py) ======
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
# 0) Orekit 初始化（每个进程只做一次） / Orekit Initialization (done only once per process)
# -----------------------------------------------------------------------------

_VM_STARTED = False
_DATA_LOADED = False
_CONTEXT: "_OrekitContext | None" = None


def _to_utc(dt: datetime) -> datetime:
    """把 naive datetime 当作 UTC；aware datetime 转到 UTC。
       / Treat naive datetime as UTC; convert aware datetime to UTC."""
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)




def _orbit_number_from_time(
    *,
    window_start: datetime,
    scenario_start: datetime,
    orbit_period_s: float | None,
) -> int | None:
    """计算窗口所属的“第几圈”（从 1 开始）。
       / Calculate which 'orbit number' the window belongs to (starting from 1).

    约定： / Convention:
    - 场景开始时间所在的圈为 1 / The orbit containing the scenario start time is 1
    - 使用窗口 start_time 作为归属时刻 / Use window start_time as the assigning time
    - orbit_period_s 由 Orekit Orbit.getKeplerianPeriod() 预先计算 / orbit_period_s is pre-calculated by Orekit Orbit.getKeplerianPeriod()
    """
    if orbit_period_s is None or orbit_period_s <= 0:
        return None
    dt_s = max(0.0, (_to_utc(window_start) - _to_utc(scenario_start)).total_seconds())
    return int(dt_s // float(orbit_period_s)) + 1

def _find_orekit_data_zip(data_path: Optional[str] = None) -> str:
    """定位 orekit-data.zip。 / Locate orekit-data.zip."""
    # 1) 显式指定 / 1) Explicitly specified
    if data_path and os.path.exists(data_path):
        return data_path

    # 2) 默认：utils/orekit-data.zip（与本文件同目录） / 2) Default: utils/orekit-data.zip (same directory as this file)
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    cand = os.path.join(cur_dir, "orekit-data.zip")
    if os.path.exists(cand):
        return cand

    # 3) 当前工作目录 / 3) Current working directory
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
    """确保 JVM 已启动且 Orekit 数据已加载（每个进程仅一次）。
       / Ensure JVM is started and Orekit data is loaded (only once per process)."""
    global _VM_STARTED, _DATA_LOADED, _CONTEXT

    # 1) 启动 JVM（只可调用一次） / 1) Start JVM (can only be called once)
    if not _VM_STARTED:
        import orekit_jpype as orekit

        # 可选：若安装了 orekit-jpype[jdk4py]，自动设置 JAVA_HOME
        # / Optional: If orekit-jpype[jdk4py] is installed, automatically set JAVA_HOME
        try:
            import jdk4py  # type: ignore

            os.environ.setdefault("JAVA_HOME", str(jdk4py.JAVA_HOME))
        except Exception:
            pass

        orekit.initVM()
        _VM_STARTED = True

    # 2) 加载 orekit-data / 2) Load orekit-data
    if not _DATA_LOADED:
        from orekit_jpype.pyhelpers import setup_orekit_curdir

        zip_path = _find_orekit_data_zip(data_path)
        setup_orekit_curdir(zip_path)
        _DATA_LOADED = True

    # 3) 初始化上下文 / 3) Initialize context
    if _CONTEXT is None:
        _CONTEXT = _OrekitContext.create()


# -----------------------------------------------------------------------------
# 1) Orekit 上下文（尽量复用） / Orekit Context (try to reuse)
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class _OrekitContext:
    """Orekit 计算的常用上下文对象（可复用）。
       / Common context object for Orekit calculations (reusable)."""

    # Java objects（类型提示不强制，避免 mypy/JVM 类型冲突） / Java objects (type hints not forced to avoid mypy/JVM type conflicts)
    inertial: object
    itrf: object
    earth: object
    utc: object
    mu: float

    @staticmethod
    def create() -> "_OrekitContext":
        # 延迟导入 Java 类 / Lazy import Java classes
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


# KeplerianOrbit 缓存：同一颗卫星在一个进程里会被多次用于不同任务
# / KeplerianOrbit cache: The same satellite will be used multiple times for different tasks in a single process
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
    """构造（或从缓存复用）KeplerianOrbit。
       / Construct (or reuse from cache) a KeplerianOrbit."""
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
# 2) 单对（卫星-目标）可见性+姿态计算 / Single pair (Satellite-Target) visibility + attitude calculation
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
    """返回 (visibility_windows, samples, visible_attitudes)。
       / Returns (visibility_windows, samples, visible_attitudes).

    说明：该实现源自原 utils/cs.py 的两阶段算法，但修复了原文件中局部变量判断的一个小 bug。
    / Note: This implementation is derived from the two-stage algorithm in the original utils/cs.py, but fixes a minor bug in local variable evaluation from the original file.
    """
    # 确保 Orekit 上下文已初始化（多进程下每个进程一次） / Ensure Orekit context is initialized (once per process in multi-processing)
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

    # -------- 时间统一为 UTC-aware / Standardize time as UTC-aware --------
    epoch_dt = _to_utc(epoch_dt)
    start_dt = _to_utc(start_dt)
    end_dt = _to_utc(end_dt)

    if step_s <= 0:
        raise ValueError("step_s must be a positive number")


    # -------- 目标点与姿态模型 / Target point and attitude model --------
    target_geo = GeodeticPoint(
        math.radians(float(target_lat_deg)),
        math.radians(float(target_lon_deg)),
        float(target_alt_m),
    )
    target_frame = TopocentricFrame(ctx.earth, target_geo, "TARGET")

    attitude_provider = TargetPointing(ctx.inertial, target_geo, ctx.earth)

    # 轨道与传播器（orbit 可缓存） / Orbit and propagator (orbit is cacheable)
    orbit = _build_keplerian_orbit(ctx, a_m, e, i_deg, raan_deg, argp_deg, m_deg, epoch_dt)
    propagator = KeplerianPropagator(orbit, attitude_provider)

    min_el_rad = math.radians(float(min_elevation_deg))

    step_fine = float(step_s)
    coarse_step = max(step_fine, float(coarse_step_min))

    fine_td = timedelta(seconds=step_fine)
    coarse_td = timedelta(seconds=coarse_step)

    # 目标在 ITRF 的位置可复用（每个时刻再变换到惯性系） / Target position in ITRF is reusable (transformed to inertial frame at each step)
    target_in_itrf = ctx.earth.transform(target_geo)

    def eval_full(dt: datetime) -> dict:
        """传播一次并计算 (elevation/off-nadir/yaw/pitch/roll/visible)。
           / Propagate once and compute (elevation/off-nadir/yaw/pitch/roll/visible)."""
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
                # 姿态角解算失败，按不可用处理 / Attitude angle calculation failed, treat as unavailable
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
    # A) step_fine >= coarse_step_min：直接全程扫描（与原 cs.py 单步长分支一致）
    #    / step_fine >= coarse_step_min: Direct full scan (consistent with single-step branch in original cs.py)
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
    # B) 两阶段算法：粗扫仰角几何可见区间 -> 候选区间内精扫姿态约束
    #    / Two-stage algorithm: Coarse scan for elevation geometric visible intervals -> Fine scan for attitude constraints within candidate intervals
    # ------------------------------------------------------------------

    # 1) 粗扫（只看 elevation > min_elevation） / Coarse scan (only check elevation > min_elevation)
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

    # 2) 在候选区间精扫 / Fine scan in candidate intervals
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
# 3) 任务/卫星对象 -> 轨道参数/约束映射 / Mission/Satellite object -> Orbit parameters/Constraint mapping
# -----------------------------------------------------------------------------


def _extract_orbit_params(sat: Satellite) -> Dict[str, object]:
    """把 Satellite.orbital_params 统一提取为 orekit 需要的六根 + epoch。
       / Uniformly extract Satellite.orbital_params into the 6 elements + epoch required by Orekit."""
    op = sat.orbital_params

    # 现有模型约定：semi_major_axis 为 km，这里转 m / Existing model convention: semi_major_axis is in km, converting to m here
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
    """支持多种 epoch 表达（尽量兼容现有 JSON/字符串格式）。
       / Supports various epoch expressions (trying to be compatible with existing JSON/string formats)."""
    if isinstance(epoch, datetime):
        return _to_utc(epoch)

    # 数字：按 UNIX 秒 / Number: treated as UNIX seconds
    if isinstance(epoch, (int, float)):
        return datetime.fromtimestamp(float(epoch), tz=timezone.utc)

    s = str(epoch).strip()

    # ISO 8601
    try:
        dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
        return _to_utc(dt)
    except Exception:
        pass

    # 常见格式兜底 / Fallback for common formats
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

    # 最后兜底：当作 UTC now（避免直接崩掉，但会影响可见性，建议尽早修正输入）
    # / Final fallback: treat as UTC now (prevents immediate crash but affects visibility; early correction of input is recommended)
    return datetime.now(tz=timezone.utc)


def _get_sat_att_limits_deg(sat: Satellite, default: Tuple[float, float, float]) -> Tuple[float, float, float]:
    """从 Satellite.maneuverability_capability 中提取姿态限制。
       / Extract attitude limits from Satellite.maneuverability_capability."""
    mc = getattr(sat, "maneuverability_capability", None)
    if mc is None:
        return default
    # 按 models.py 约定：max_pitch/max_yaw/max_roll 均为 deg / Convention in models.py: max_pitch/max_yaw/max_roll are all in deg
    return (float(mc.max_yaw_angle), float(mc.max_pitch_angle), float(mc.max_roll_angle))


def generate_random_coordinates(mission_info: MissionInfo) -> Tuple[float, float]:
    """按 MissionInfo 的分布设置随机生成 (lat, lon)。
       / Randomly generate (lat, lon) based on the distribution settings in MissionInfo."""
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
    """把 compute_visibility_and_attitude 的结果组织成 models.TimeWindow 列表。
       / Organize the results of compute_visibility_and_attitude into a list of models.TimeWindow."""
    # 建索引，便于按窗口切片 / Build an index for easy slicing by window
    times = [va["time"] for va in visible_attitudes]

    tws: List[TimeWindow] = []
    for (ws, we) in visibility_windows:
        # 该窗口内的可见采样点（cs.py 中 we 通常是“首次不可见”的时刻，因此用 < we 更稳妥）
        # / Visible sample points within this window (in cs.py, 'we' is usually the time of "first non-visible", so '< we' is safer)
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
# 4) 场景级：Orekit 计算窗口（支持 ensure 开关） / Scenario Level: Orekit calculates windows (supports ensure toggle)
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
    """顶层接口：使用 Orekit 计算整个场景的观测窗口和通信窗口。
       / Top-level interface: Use Orekit to calculate observation and communication windows for the entire scenario.

    Args:
        ensure_each_mission_has_window:
            True  -> 若某任务在场景周期内对所有卫星都没有观测窗口，则重采样任务坐标直到有窗口；
                     / If a mission has no observation windows for any satellite during the scenario period, resample task coordinates until a window is found;
            False -> 不做强制保证，允许某些任务没有任何窗口（更快，且更贴近真实“不可观测”任务）。
                     / Does not enforce guarantees, allowing some missions to have no windows (faster, and closer to real "unobservable" tasks).

    Returns:
        observation_windows, communication_windows, updated_missions

    说明：updated_missions 可能与输入 missions 不同（当 ensure 开启且发生重采样时）。
    / Note: updated_missions may be different from the input missions (when ensure is enabled and resampling occurs).
    """
    # 兼容旧参数名： / Compatible with old parameter names:
    # - step_seconds: 旧版入口使用 step_seconds，本函数内部统一用 step（秒）
    #                 / The old entry used step_seconds, internally this function uniformly uses step (seconds)
    # - max_retries: 旧版入口使用 max_retries，本函数内部统一用 max_resample_tries
    #                / The old entry used max_retries, internally this function uniformly uses max_resample_tries
    if step is None:
        if step_seconds is None:
            raise TypeError("calculate_windows_orekit() must provide step or step_seconds (unit: seconds)")
        step = float(step_seconds)
    else:
        step = float(step)

    if max_retries is not None:
        max_resample_tries = int(max_retries)

    # data_path: 指定 orekit-data.zip 的位置（多进程下每个进程只加载一次）
    # / data_path: Specifies the location of orekit-data.zip (loaded only once per process in multi-processing)
    ensure_orekit(data_path)

    DEFAULT_ATT_LIMITS = (45.0, 45.0, 45.0)  # yaw, pitch, roll

    # 1) 预提取卫星轨道 & 姿态限制 / Pre-extract satellite orbit & attitude limits
    orbit_map: Dict[str, Dict[str, object]] = {}
    att_limit_map: Dict[str, Tuple[float, float, float]] = {}

    for sat in satellites:
        sat_id = str(getattr(sat, "id", None) or getattr(sat, "satellite_id", ""))
        if not sat_id:
            continue
        orbit_map[sat_id] = _extract_orbit_params(sat)
        att_limit_map[sat_id] = _get_sat_att_limits_deg(sat, DEFAULT_ATT_LIMITS)

    # 1.1) 预计算每颗卫星的开普勒周期（秒）——使用 Orekit Orbit.getKeplerianPeriod()
    # / Pre-calculate the Keplerian period (seconds) for each satellite -- using Orekit Orbit.getKeplerianPeriod()
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
                # 有极少数异常输入时兜底为“不填 orbit_number” / Fallback to "do not fill orbit_number" for very rare anomalous inputs
                continue

    observation_windows: List[ObservationWindow] = []
    communication_windows: List[CommunicationWindow] = []

    # 2) missions：可能需要重采样坐标，因此做浅拷贝列表（对象本身会被原地修改）
    # / missions: May need to resample coordinates, so create a shallow copy list (objects themselves will be modified in-place)
    updated_missions: List[Mission] = list(missions)
    # 3) 观测窗口 / Observation windows
    total_missions = len(updated_missions)
    for mission_idx, mission in enumerate(updated_missions, start=1):
        mission_id = str(getattr(mission, "id", None) or getattr(mission, "mission_id", ""))
        # 进度输出：不受 ensure_each_mission_has_window 影响 / Progress output: not affected by ensure_each_mission_has_window
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

                # 该 sat 对该 mission 有窗口 / The sat has a window for the mission
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
                # 如果开启了强制窗口并且经历过重采样，输出提示信息（便于调试数据生成）
                # / If forced windows are enabled and resampling has occurred, output a prompt (useful for debugging data generation)
                # if ensure_each_mission_has_window and tries > 0:
                #     print(f"Mission {mission_id} found observation window after {tries + 1} attempts", flush=True)
                observation_windows.extend(tmp_obs_windows_for_this_mission)
                break

            # 没有任何窗口 / No windows at all
            if not ensure_each_mission_has_window:
                break

            tries += 1
            if tries >= max_resample_tries:
                # 到达上限仍找不到窗口，则放弃强制（避免死循环） / Reached limit and still no window found, so give up forcing (avoid infinite loop)
                break

            # print(f"Mission {mission_id} did not find an observation window, reallocating position (attempt {tries})", flush=True)
            new_lat, new_lon = generate_random_coordinates(mission_info)
            tgt.latitude = new_lat
            tgt.longitude = new_lon

    # 4) 通信窗口（若 ground_stations 为空，跳过） / Communication windows (if ground_stations is empty, skip)
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
                    # 通信窗口不受姿态约束：放宽到 180° / Communication windows are not restricted by attitude: relaxed to 180°
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