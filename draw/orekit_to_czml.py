# -*- coding: utf-8 -*-
"""
Orekit 场景 → CZML → Cesium 展示
/ Orekit Scenario -> CZML -> Cesium Visualization

功能（支持多颗卫星的 Scenario_*_SatsX_*.json）：
/ Features (Supports multi-satellite Scenario_*_SatsX_*.json):
  1. 从场景 json 读取：
     / Read from scenario json:
       - 多颗卫星轨道六根 + 历元（epoch）
         / Multiple satellites' 6 orbital elements + epoch
       - 所有地面目标（missions）的经纬度
         / Lat/Lon of all ground targets (missions)
  2. 使用 Orekit 做 Kepler 传播，在规划时间区间内采样每颗卫星的地面轨迹
     / Use Orekit for Kepler propagation, sample ground tracks for each satellite within the planning time interval
  3. 生成 orbit.czml：
     / Generate orbit.czml:
       - 每颗卫星一个动态实体（带 point + “仅当前一圈”的 path）
         / One dynamic entity per satellite (with point + "current orbit only" path)
       - 多个静态地面目标实体（每个 mission 一点）
         / Multiple static ground target entities (one point per mission)

规划方案 Scenario_*_schedule.json 仍然在 Cesium 的 html 里加载，
用于画“卫星-目标连线”和左右侧的指标 / 列表，不在本脚本中处理。
/ The scheduling plan Scenario_*_schedule.json is still loaded in Cesium's html
to draw "satellite-target lines" and side metrics/lists, which are not handled in this script.
"""

import json
import os
import math
from datetime import datetime, timedelta, timezone

# ========= 1. 启动 JVM + Orekit 初始化 / 1. Start JVM + Orekit Initialization =========
import orekit_jpype as orekit

# 尽量用 jdk4py 自动设置 JAVA_HOME（如果安装了 [jdk4py]）
# / Try to use jdk4py to automatically set JAVA_HOME (if [jdk4py] is installed)
try:
    import jdk4py  # type: ignore

    os.environ.setdefault("JAVA_HOME", str(jdk4py.JAVA_HOME))
except Exception:
    pass

# JVM 只能启动一次 / JVM can only be started once
orekit.initVM()

from orekit_jpype.pyhelpers import (
    setup_orekit_curdir,
    datetime_to_absolutedate,
)


def init_orekit(data_path: str = "orekit-data.zip") -> None:
    """加载 orekit-data 数据（zip 或目录都行）
       / Load orekit-data (either zip or directory works)"""
    if not os.path.exists(data_path):
        raise FileNotFoundError(
            f"{data_path} not found. Please download orekit-data.zip from the Orekit website "
            f"and place it in the current directory, or modify init_orekit(data_path=...) to the actual path."
        )
    setup_orekit_curdir(data_path)


# ========= 2. 用 Orekit 传播轨道，得到地面经纬高 / 2. Propagate orbit with Orekit to get ground lat/lon/alt =========
def propagate_satellite_groundtrack(
        # 轨道六根（单位：a 米, 角度用度） / 6 orbital elements (unit: a in meters, angles in degrees)
        a_m: float,
        e: float,
        i_deg: float,
        raan_deg: float,
        argp_deg: float,
        m_deg: float,
        # 轨道历元（epoch） / Orbit epoch
        epoch_dt: datetime,
        # 仿真时间设置 / Simulation time settings
        start_dt: datetime,
        end_dt: datetime,
        step_s: float,
):
    """
    用 Orekit 做简单的 Kepler 传播，输出地面经纬高。
    / Use Orekit for simple Kepler propagation, output ground lat/lon/alt.



    返回 samples 列表，每个元素是：
    / Returns a list of samples, where each element is:
        {
            "time": datetime(UTC),
            "lon_deg": float,
            "lat_deg": float,
            "alt_m": float,
        }
    """
    from org.orekit.frames import FramesFactory
    from org.orekit.time import TimeScalesFactory
    from org.orekit.bodies import OneAxisEllipsoid
    from org.orekit.orbits import KeplerianOrbit, PositionAngleType
    from org.orekit.propagation.analytical import KeplerianPropagator
    from org.orekit.utils import IERSConventions, Constants

    # 时间统一为 UTC aware / Standardize time as UTC aware
    def to_utc(dt: datetime) -> datetime:
        if dt.tzinfo is None:
            return dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)

    epoch_dt = to_utc(epoch_dt)
    start_dt = to_utc(start_dt)
    end_dt = to_utc(end_dt)

    epoch_date = datetime_to_absolutedate(epoch_dt)
    _ = TimeScalesFactory.getUTC()

    # 坐标系、地球模型 / Coordinate systems, Earth model
    inertial = FramesFactory.getEME2000()
    ae = Constants.WGS84_EARTH_EQUATORIAL_RADIUS
    f = Constants.WGS84_EARTH_FLATTENING
    itrf = FramesFactory.getITRF(IERSConventions.IERS_2010, True)
    earth = OneAxisEllipsoid(ae, f, itrf)

    # 轨道 + 传播器 / Orbit + Propagator
    mu = Constants.WGS84_EARTH_MU
    orbit = KeplerianOrbit(
        float(a_m),
        float(e),
        math.radians(i_deg),
        math.radians(argp_deg),
        math.radians(raan_deg),
        math.radians(m_deg),
        PositionAngleType.MEAN,
        inertial,
        epoch_date,
        mu,
    )
    propagator = KeplerianPropagator(orbit)

    samples: list[dict] = []
    current_dt = start_dt
    step_td = timedelta(seconds=step_s)

    while current_dt <= end_dt:
        date = datetime_to_absolutedate(current_dt)

        state = propagator.propagate(date)
        sat_pos = state.getPVCoordinates(inertial).getPosition()

        # ECI → 地固 + 椭球面地理坐标 / ECI -> ECEF + Ellipsoidal geographic coordinates
        geo = earth.transform(sat_pos, inertial, date)  # GeodeticPoint
        lat_deg = math.degrees(geo.getLatitude())
        lon_deg = math.degrees(geo.getLongitude())
        alt_m = float(geo.getAltitude())

        samples.append(
            {
                "time": current_dt,
                "lon_deg": lon_deg,
                "lat_deg": lat_deg,
                "alt_m": alt_m,
            }
        )

        current_dt += step_td

    return samples


# ========= 3. 把多颗卫星的仿真结果写成一个 CZML 文档 / 3. Write multi-satellite simulation results into a CZML document =========
def build_czml(
        sat_samples: dict[str, list[dict]],
        missions: list[dict],
        sat_orbit_trail_time: dict[str, float],
        output_path: str = "orbit.czml",
) -> None:
    """
    sat_samples: {sat_id: [sample, ...], ...}
    missions:    和之前一样，每个 mission 一个目标点 / Same as before, one target point per mission
    sat_orbit_trail_time: {sat_id: 该轨道周期(s)，用于 path.trailTime} / {sat_id: orbit period(s), used for path.trailTime}
    """
    # 没有任何卫星，直接报错 / No satellites, raise error directly
    non_empty_sats = {sid: s for sid, s in sat_samples.items() if s}
    if not non_empty_sats:
        raise ValueError("sat_samples is empty or all satellite samples are empty, cannot generate CZML")

    # 计算整个场景的时间区间（所有卫星样本的最早 / 最晚）
    # / Calculate the time interval for the entire scenario (earliest/latest of all satellite samples)
    global_start: datetime | None = None
    global_end: datetime | None = None
    for samples in non_empty_sats.values():
        s0 = samples[0]["time"]
        s1 = samples[-1]["time"]
        if global_start is None or s0 < global_start:
            global_start = s0
        if global_end is None or s1 > global_end:
            global_end = s1

    assert global_start is not None and global_end is not None

    def to_iso_z(dt: datetime) -> str:
        dt = dt.astimezone(timezone.utc)
        return dt.strftime("%Y-%m-%dT%H:%M:%SZ")

    epoch_str = to_iso_z(global_start)
    end_str = to_iso_z(global_end)
    availability = f"{epoch_str}/{end_str}"

    # 1) document 包（包含 clock） / 1) document packet (contains clock)
    doc_packet = {
        "id": "document",
        "name": "Orekit multi-satellite demo",
        "version": "1.0",
        "clock": {
            "interval": availability,
            "currentTime": epoch_str,
            "multiplier": 60,
            "range": "LOOP_STOP",
            "step": "SYSTEM_CLOCK_MULTIPLIER",
        },
    }

    packets = [doc_packet]

    # 2) 每颗卫星一个实体 / 2) One entity per satellite
    color_table = [
        [0, 255, 255, 255],  # 青色 / Cyan
        [255, 255, 0, 255],  # 黄色 / Yellow
        [255, 0, 255, 255],  # 品红 / Magenta
        [0, 255, 0, 255],  # 绿色 / Green
        [255, 128, 0, 255],  # 橙色 / Orange
    ]

    for idx, (sat_id, samples) in enumerate(sorted(non_empty_sats.items())):
        rgba = color_table[idx % len(color_table)]

        # cartographicDegrees: [t, lon, lat, h, t, lon, lat, h, ...]
        carto: list[float] = []
        for s in samples:
            t_offset = (s["time"] - global_start).total_seconds()
            carto.extend(
                [
                    float(t_offset),
                    float(s["lon_deg"]),
                    float(s["lat_deg"]),
                    float(s["alt_m"]),
                ]
            )

        orbit_trail_time_s = float(sat_orbit_trail_time.get(sat_id, 0.0))

        sat_packet = {
            "id": sat_id,
            "name": sat_id,
            "availability": availability,
            "position": {
                "epoch": epoch_str,
                "cartographicDegrees": carto,
            },
            "path": {
                "show": True,
                "width": 2,
                "material": {
                    "solidColor": {
                        "color": {"rgba": rgba},
                    }
                },
                "leadTime": 0,
                # 只显示当前时间点往前一圈轨迹 / Only show the trajectory of the previous orbit from the current time point
                "trailTime": orbit_trail_time_s if orbit_trail_time_s > 0 else 0,
            },
            "point": {
                "pixelSize": 8,
                "color": {"rgba": rgba},
                "outlineColor": {"rgba": [0, 0, 0, 255]},
                "outlineWidth": 1,
            },
            "label": {
                "text": sat_id,
                "font": "14px sans-serif",
                "fillColor": {"rgba": [255, 255, 255, 255]},
                "outlineColor": {"rgba": [0, 0, 0, 255]},
                "outlineWidth": 2,
                "style": "FILL_AND_OUTLINE",
                "pixelOffset": {"cartesian2": [0, -20]},
                "verticalOrigin": "BOTTOM",
            },
        }

        packets.append(sat_packet)

    # 3) 地面目标点（一个 mission 一个点） / 3) Ground target points (one point per mission)
    for m in missions:
        target_packet = {
            "id": m["id"],
            "name": f"Target {m['id']}",
            "position": {
                "cartographicDegrees": [
                    float(m["lon_deg"]),
                    float(m["lat_deg"]),
                    float(m["alt_m"]),
                ]
            },
            "point": {
                "pixelSize": 10,
                "color": {"rgba": [255, 0, 0, 255]},
                "outlineColor": {"rgba": [255, 255, 255, 255]},
                "outlineWidth": 2,
            },
            "label": {
                "text": m["id"],
                "font": "13px sans-serif",
                "fillColor": {"rgba": [255, 255, 0, 255]},
                "outlineColor": {"rgba": [0, 0, 0, 255]},
                "outlineWidth": 2,
                "style": "FILL_AND_OUTLINE",
                "pixelOffset": {"cartesian2": [0, -20]},
                "verticalOrigin": "BOTTOM",
            },
        }
        packets.append(target_packet)

    # 4) 写成 orbit.czml / 4) Write as orbit.czml
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(packets, f, ensure_ascii=False, indent=2)

    print(f"[OK] CZML written to: {os.path.abspath(output_path)}")


# ========= 4. 从场景 JSON 读取所有卫星 & 目标，串起来 / 4. Read all satellites & targets from scenario JSON and link them =========
def demo(
        # 默认用你刚上传的多星场景，也可以自己改路径
        # / Default to the multi-satellite scenario you just uploaded, you can also change the path yourself
        scenario_path: str = "output/Scenario_S1_Sats3_M10_T0.5d_dist1.json",
        output_czml: str = "orbit.czml",
):
    # 1) Orekit 数据 / 1) Orekit data
    init_orekit("orekit-data.zip")

    # 2) 读取场景 JSON / 2) Read scenario JSON
    if not os.path.exists(scenario_path):
        raise FileNotFoundError(f"Scenario file not found: {scenario_path}")

    with open(scenario_path, "r", encoding="utf-8") as f:
        scenario = json.load(f)

    satellites = scenario.get("satellites", [])
    if not satellites:
        raise ValueError("scenario['satellites'] is empty")

    # —— 仿真时间区间（用 metadata.creation_time + duration） ——
    # / -- Simulation time interval (using metadata.creation_time + duration) --
    meta = scenario.get("metadata", {})
    start_iso = meta.get("creation_time", "2025-11-18T12:00:00")
    duration_s = float(meta.get("duration", 43200.0))
    step_s = float(meta.get("time_step", 10.0))

    start_dt = datetime.fromisoformat(start_iso).replace(tzinfo=timezone.utc)
    end_dt = start_dt + timedelta(seconds=duration_s)

    print(f"[*] Scenario: {scenario.get('scenario_id', 'Unknown')}")
    print(
        f"    Time interval: {start_dt.isoformat()} -> {end_dt.isoformat()}, "
        f"step={step_s} s, number of satellites={len(satellites)}"
    )

    # —— 每颗卫星传播轨迹 ——
    # / -- Propagation trajectory for each satellite --
    sat_samples: dict[str, list[dict]] = {}
    sat_orbit_trail_time: dict[str, float] = {}

    from org.orekit.utils import Constants as OrekitConstants
    mu = OrekitConstants.WGS84_EARTH_MU

    for sat in satellites:
        sat_id = sat["id"]
        op = sat["orbital_params"]

        a_m = float(op["semi_major_axis_km"]) * 1000.0
        e = float(op["eccentricity"])
        i_deg = float(op["inclination_deg"])
        raan_deg = float(op["right_ascension_of_ascending_node_deg"])
        argp_deg = float(op["argument_of_perigee_deg"])
        m_deg = float(op["mean_anomaly_deg"])

        epoch_str = op["epoch"]
        epoch_dt = datetime.strptime(
            epoch_str, "%d %b %Y %H:%M:%S.%f"
        ).replace(tzinfo=timezone.utc)

        print(
            f"    Satellite {sat_id}: a={a_m:.1f} m, e={e}, i={i_deg} deg, "
            f"RAAN={raan_deg} deg, ω={argp_deg} deg, M={m_deg} deg"
        )

        samples = propagate_satellite_groundtrack(
            a_m=a_m,
            e=e,
            i_deg=i_deg,
            raan_deg=raan_deg,
            argp_deg=argp_deg,
            m_deg=m_deg,
            epoch_dt=epoch_dt,
            start_dt=start_dt,
            end_dt=end_dt,
            step_s=step_s,
        )

        print(f"        Number of samples: {len(samples)}")
        sat_samples[sat_id] = samples

        # 轨道周期用于 “只显示当前一圈轨迹”
        # / Orbital period used to "only show current orbit trajectory"
        orbit_trail_time_s = 2.0 * math.pi * math.sqrt(a_m ** 3 / mu)
        sat_orbit_trail_time[sat_id] = orbit_trail_time_s
        print(f"        Estimated orbital period ~= {orbit_trail_time_s:.1f} s")

    # —— 所有 mission 的目标点 ——
    # / -- Target points for all missions --
    missions_raw = scenario.get("missions", [])
    missions: list[dict] = []
    for m in missions_raw:
        loc = m["target_location"]
        alt_km = loc.get("altitude_km", 0.0)
        missions.append(
            {
                "id": m["id"],
                "lat_deg": float(loc["latitude"]),
                "lon_deg": float(loc["longitude"]),
                "alt_m": float(alt_km) * 1000.0,
                "priority": m.get("priority"),
            }
        )

    # 3) 生成 CZML / 3) Generate CZML
    build_czml(
        sat_samples=sat_samples,
        missions=missions,
        sat_orbit_trail_time=sat_orbit_trail_time,
        output_path=output_czml,
    )


if __name__ == "__main__":
    demo()