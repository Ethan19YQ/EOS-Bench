# -*- coding: utf-8 -*-
"""
scenario_loader.py
调度场景数据加载与解析模块 / Scheduling Scenario Data Loading and Parsing Module

本文件在整个项目中的角色 / Role of this file in the entire project
--------------------------------
1. 从 Scenario.export_to_json 导出的场景 JSON 文件中，解析出调度所需信息：
   / Parse the scheduling information from the scenario JSON file exported by Scenario.export_to_json:
   - 任务（任务 ID、优先级、所需观测持续时间）； / Tasks (Task ID, priority, required observation duration);
   - 卫星（卫星 ID）； / Satellites (Satellite ID);
   - 任务-卫星可见性窗口（observation_windows）； / Task-Satellite visibility windows (observation_windows);
   - 卫星-地面站可见性窗口（communication_windows，用于数传）。 / Satellite-Ground Station visibility windows (communication_windows, used for data transmission).

2. 构建统一的调度问题数据结构 SchedulingProblem：
   / Build a unified scheduling problem data structure, SchedulingProblem:
   - 便于后续调度算法（模拟退火、启发式、MILP 等）统一使用； / Facilitates unified use by subsequent scheduling algorithms (Simulated Annealing, Heuristics, MILP, etc.);
   - 不包含任何调度决策，仅是“输入数据”。 / Contains no scheduling decisions, purely "input data".

3. 支持两种场景：
   / Supports two types of scenarios:
   - 无地面站：只调度“卫星执行任务”； / No ground stations: Only schedule "satellites executing tasks";
   - 有地面站：对每个任务同时调度“卫星观测 + 卫星-地面站数传”， / With ground stations: Simultaneously schedule "satellite observation + satellite-to-ground data transmission" for each task.
     约束由 constraint_model 中统一处理。 / Constraints are uniformly handled in constraint_model.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime, timedelta, timezone


# ==============================
# 1. 基础数据结构 / Basic Data Structures
# ==============================

@dataclass
class TaskWindow:
    """
    单个任务在某颗卫星上的一个观测可见性时间窗。
    / A single observation visibility time window for a specific task on a specific satellite.

    注意：从 v21 起支持 orbit_number（用于每圈资源约束）。
    / Note: orbit_number is supported since v21 (used for per-orbit resource constraints).
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
        """窗口总长度（秒） / Total window length (seconds)"""
        return (self.end_time - self.start_time).total_seconds()



@dataclass
class CommWindow:
    """
    卫星-地面站通信窗口：
    / Satellite-Ground Station communication window:
    - 与任务无关，仅描述 (卫星, 地面站) 在某时间段可进行数传。
      / Independent of tasks, only describes that a (Satellite, Ground Station) pair can perform data transmission during a certain time period.
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
    调度任务对象，对应 JSON 中的 mission。
    / Scheduling task object, corresponding to 'mission' in JSON.
    """
    id: str
    priority: float
    required_duration: float          # 任务观测所需时间（秒） / Required observation time for the task (seconds)
    windows: List[TaskWindow] = field(default_factory=list)


@dataclass
class SensorSpec:
    """传感器参数（用于数据量/能耗计算） / Sensor parameters (used for data volume/energy consumption calculation)"""
    sensor_id: str
    data_rate_Mbps: float = 0.0
    power_consumption_W: float = 0.0


@dataclass
class SchedulingSatellite:
    """调度视角下的卫星对象 / Satellite object from a scheduling perspective

    新增字段： / Newly added fields:
    - max_data_storage_GB / max_power_W：每圈（orbit）容量/能量限制上限；
      / max_data_storage_GB / max_power_W: Per-orbit capacity/energy limits;
    - slew_rate_deg_per_s / stabilization_time_s：敏捷卫星姿态机动参数；
      / slew_rate_deg_per_s / stabilization_time_s: Agile satellite attitude maneuver parameters;
    - sensors：该卫星可用的传感器参数（data_rate_Mbps / power_consumption_W）。
      / sensors: Available sensor parameters for the satellite (data_rate_Mbps / power_consumption_W).
    """
    id: str
    maneuverability_type: str = "agile"  # "agile" / "non_agile"

    # per-orbit limits
    max_data_storage_GB: float = 0.0
    max_power_W: float = 0.0

    # maneuverability params (agile)
    slew_rate_deg_per_s: float = 1.0
    stabilization_time_s: float = 0.0

    sensors: Dict[str, SensorSpec] = field(default_factory=dict)



@dataclass
class SchedulingGroundStation:
    """调度视角下的地面站对象 / Ground station object from a scheduling perspective"""
    id: str


@dataclass
class SchedulingProblem:
    """
    调度问题整体描述：
    / Overall description of the scheduling problem:
    - scenario_id: 原场景 ID； / Original scenario ID;
    - start_time / end_time: 场景时间范围； / Scenario time range;
    - satellites: 所有卫星； / All satellites;
    - ground_stations: 所有地面站（可能为空）； / All ground stations (may be empty);
    - tasks: 所有任务； / All tasks;
    - comm_windows: 所有卫星-地面站通信窗口（可能为空）。 / All satellite-ground station communication windows (may be empty).
    """
    scenario_id: str
    start_time: datetime
    end_time: datetime
    satellites: Dict[str, SchedulingSatellite]
    ground_stations: Dict[str, SchedulingGroundStation]
    tasks: Dict[str, SchedulingTask]
    comm_windows: List[CommWindow]


# ==============================
# 2. JSON 解析辅助 / JSON Parsing Helpers
# ==============================

def _parse_iso_time(time_str: str) -> datetime:
    """
    解析 ISO 时间字符串，并统一转换为 **naive UTC datetime**（不带 tzinfo）。
    / Parse ISO time strings and uniformly convert them to **naive UTC datetime** (without tzinfo).

    说明： / Description:
    - 场景 JSON 里既可能出现不带时区的时间（如 '2025-11-18T12:00:00'），
      / Time without timezone may appear in scenario JSON (e.g., '2025-11-18T12:00:00'),
      也可能出现带 '+00:00' 的时间（如 '2025-11-18T16:25:49+00:00'）或以 'Z' 结尾。
      / as well as time with '+00:00' (e.g., '2025-11-18T16:25:49+00:00') or ending with 'Z'.
    - 项目内部统一使用 naive datetime 做差与比较，避免 aware/naive 混用报错。
      / The project internally uses naive datetime for differences and comparisons to avoid errors from mixing aware/naive datetimes.
    """
    s = (time_str or '').strip()
    if s.endswith('Z'):
        s = s[:-1] + '+00:00'
    dt = datetime.fromisoformat(s)
    if dt.tzinfo is not None:
        dt = dt.astimezone(timezone.utc).replace(tzinfo=None)
    return dt



def _infer_maneuverability_type(sat: dict) -> str:
    """从卫星对象中尽量稳健地推断 maneuverability_type。
       / Robustly infer maneuverability_type from the satellite object.

    场景数据里该字段的位置在不同版本中可能不同：
    / The location of this field in scenario data may vary across different versions:
    - sat["maneuverability_type"]
    - sat["attributes"]["maneuverability_type"]
    - sat["payload"]["maneuverability_type"] 或 sat["payload"]["attributes"][...] / or sat["payload"]["attributes"][...]
    - sat["payloads"][i]["maneuverability_type"] 或 sat["payloads"][i]["attributes"][...] / or sat["payloads"][i]["attributes"][...]
    只要能读到包含 'agile' 或 'non' 的字符串，就归一化为 'agile' / 'non_agile'。
    / As long as a string containing 'agile' or 'non' can be read, it is normalized to 'agile' / 'non_agile'.
    """
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

    # 0) maneuverability_capability（本 benchmark 场景常见字段） / maneuverability_capability (common field in this benchmark scenario)
    mcap = sat.get('maneuverability_capability')
    if isinstance(mcap, dict):
        v = _norm(mcap.get('maneuverability_type') or mcap.get('type'))
        if v:
            return v

    # 1) 顶层 / Top level
    v = _norm(sat.get("maneuverability_type"))
    if v:
        return v

    # 2) attributes / attributes
    attrs = sat.get("attributes")
    if isinstance(attrs, dict):
        v = _norm(attrs.get("maneuverability_type"))
        if v:
            return v

    # 3) payload（单个） / payload (single)
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

    # 4) payloads（列表） / payloads (list)
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

    # 默认：敏捷 / Default: agile
    return "agile"


def load_scheduling_problem_from_json(json_path: str | Path) -> SchedulingProblem:
    """
    从 Scenario.export_to_json 输出的 JSON 文件构建 SchedulingProblem。
    / Build SchedulingProblem from the JSON file output by Scenario.export_to_json.

    预期 JSON 结构（部分）： / Expected JSON structure (partial):
    {
      "scenario_id": "...",
      "metadata": {
        "creation_time": "...",
        "duration": 12345.0,
        ...
      },
      "satellites": [...],
      "missions": [...],
      "ground_stations": [...],           # 可能不存在/为空 / May not exist/be empty
      "observation_windows": [...],
      "communication_windows": [...],     # 只有在有地面站且有数传计算时才存在 / Exists only if there are ground stations and data transmission is calculated
      ...
    }
    """
    import json

    json_path = Path(json_path)
    if not json_path.exists():
        raise FileNotFoundError(f"场景 JSON 文件不存在: {json_path} / Scenario JSON file not found: {json_path}")

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    scenario_id = data.get("scenario_id", json_path.stem)

    # 场景时间信息 / Scenario time information
    meta = data["metadata"]
    start_time = _parse_iso_time(meta["creation_time"])
    duration_s = float(meta["duration"])
    end_time = start_time + timedelta(seconds=duration_s)

    # 全局 time_step（若窗口级别缺失则回退到这里） / Global time_step (fallback to this if missing at the window level)
    global_time_step = float(meta.get("time_step", 1.0) or 1.0)

    # 卫星列表 / Satellite list
    satellites: Dict[str, SchedulingSatellite] = {}
    for sat in data.get("satellites", []):
        sat_id = sat["id"]

        # per-orbit limits
        specs = sat.get("satellite_specs", {}) or {}
        max_data_storage_GB = float(specs.get("max_data_storage_GB", 0.0) or 0.0)
        max_power_W = float(specs.get("max_power_W", 0.0) or 0.0)

        # maneuverability params
        man = sat.get("maneuverability_capability", {}) or {}
        slew_rate = float(man.get("slew_rate_deg_per_s", 1.0) or 1.0)
        stab_time = float(man.get("stabilization_time_s", 0.0) or 0.0)

        # sensors
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

    # 地面站列表（可为空） / Ground station list (can be empty)
    ground_stations: Dict[str, SchedulingGroundStation] = {}
    for gs in data.get("ground_stations", []):
        gs_id = gs["id"]
        ground_stations[gs_id] = SchedulingGroundStation(id=gs_id)

    # 任务列表 / Task list
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

    # 观测窗口 -> 任务-卫星窗口 / Observation windows -> Task-Satellite windows
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
                time_step=float(tw.get("time_step", ow.get("time_step", global_time_step))) ,
                agile_data=tw.get("agile_data", ow.get("agile_data")),
                non_agile_data=tw.get("non_agile_data", ow.get("non_agile_data")),
            )
            tasks[mid].windows.append(task_window)

    # 通信窗口 -> 卫星-地面站窗口（与任务无关） / Communication windows -> Satellite-Ground station windows (independent of tasks)
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