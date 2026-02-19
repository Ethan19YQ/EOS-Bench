# -*- coding: utf-8 -*-
"""main_generate.py
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from datetime import datetime
import os
import json
import time

from core.models import (
    OrbitalType,
    OrbitalParameters,
    TargetLocation,
    Mission,
    GroundStationLocation,
    GroundStation,
    MissionInfo,
    Satellite,
    ObservationSensor,
    ObservationCapability,
    ManeuverabilityCapability,
    ManeuverabilityType,
    CommunicationCapability,
    CommunicationAntenna,
    AntennaType,
    AntennaFrequency,
    SatelliteSpecs,
    SensorResolution,
    SensorMode,
)

from core.scenario import Scenario, ScenarioMetadata, generate_missions_dict


BASE_DIR = Path(__file__).resolve().parent
INPUT_DIR = BASE_DIR / "input"
CITIES_DIR = INPUT_DIR / "cities_data"
OUTPUT_DIR = BASE_DIR / "output"
SUMMARY_TXT = OUTPUT_DIR / f"scenario_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)



# ----------------------------------------------------------------------------
# A) 随机任务分布参数 / Random mission distribution parameters
# ----------------------------------------------------------------------------

RANDOM_MISSION_INFOS: list[MissionInfo] = [
    # 多区域分布 / Multi-region distribution
    MissionInfo(
        distribution_type=1,
        latitude_range=[
            [20.0, 40.0],
            [35.0, 65.0],
            [20.0, 60.0],
            [10.0, -30.0],
            [-30.0, -10.0],
        ],
        longitude_range=[
            [70.0, 130.0],
            [-10.0, 60.0],
            [-125.0, -60.0],
            [-80.0, -30.0],
            [110.0, 155.0],
        ],
    ),
    # 全球均匀 / Global uniform distribution
    MissionInfo(
        distribution_type=0,
        latitude_range=[[-80.0, 80.0]],
        longitude_range=[[-180.0, 180.0]],
    ),
]

# 读取目标文件模式下的占位 MissionInfo（不会用到 distribution_type）
# / Placeholder MissionInfo for target file mode (distribution_type will not be used)
DUMMY_MISSION_INFO_FOR_CITIES = MissionInfo(
    distribution_type=0,
    latitude_range=[[-80.0, 80.0]],
    longitude_range=[[-180.0, 180.0]],
)


# ----------------------------------------------------------------------------
# 0) 进程内缓存（每个 worker 进程各自持有） / In-process cache (held independently by each worker process)
# ----------------------------------------------------------------------------

_SAT_DICT_CACHE: dict[str, dict] = {}
_SAT_OBJECT_CACHE: dict[str, list[Satellite]] = {}


# ----------------------------------------------------------------------------
# 1) 构造静态对象 / Construct static objects
# ----------------------------------------------------------------------------


def _load_satellites_from_json_path(sat_json_path: str) -> list[Satellite]:
    """子进程内读取卫星 JSON，并缓存 Satellite 对象列表（避免重复解析/构造）。
       / Read satellite JSON within the subprocess and cache the list of Satellite objects (to avoid repeated parsing/construction)."""
    if sat_json_path in _SAT_OBJECT_CACHE:
        return _SAT_OBJECT_CACHE[sat_json_path]

    if sat_json_path not in _SAT_DICT_CACHE:
        with open(sat_json_path, "r", encoding="utf-8") as f:
            _SAT_DICT_CACHE[sat_json_path] = json.load(f)

    selected_satellites = _SAT_DICT_CACHE[sat_json_path]
    satellites = build_satellites_from_json(selected_satellites)
    _SAT_OBJECT_CACHE[sat_json_path] = satellites
    return satellites


def build_satellites_from_json(selected_satellites: dict) -> list[Satellite]:
    """根据卫星字典构建 Satellite 列表。
       / Build a list of Satellite objects from the satellite dictionary."""
    sats: list[Satellite] = []
    for sat_id, sat_data in selected_satellites.items():
        orbital_type = OrbitalType[sat_data[0]]
        orbital_params = OrbitalParameters(
            semi_major_axis=sat_data[1],
            eccentricity=sat_data[2],
            inclination=sat_data[3],
            argument_of_perigee=sat_data[4],
            right_ascension_of_ascending_node=sat_data[5],
            mean_anomaly=sat_data[6],
            initial_representation_Epoch=sat_data[7],
        )

        obs_sensor = ObservationSensor(
            sensor_id=f"{sat_id}_sensor",
            resolution=SensorResolution.MEDIUM,
            sensor_mode=SensorMode.VISIBLE,
            field_of_view=45.0,
            observation_swath_width=50.0,
            data_rate=0.1,
            power_consumption=50.0,
            min_elevation_angle=0.0,
        )
        observation_capability = ObservationCapability(sensors=[obs_sensor])

        maneuverability_capability = ManeuverabilityCapability(
            maneuverability_type=ManeuverabilityType.AGILE,
            slew_rate=1.0,
            max_pitch_angle=45.0,
            max_yaw_angle=45.0,
            max_roll_angle=45.0,
            stabilization_time=10.0,
        )

        comm_antenna = CommunicationAntenna(
            antenna_id=f"{sat_id}_x_antenna",
            data_rate=100.0,
            antenna_type=AntennaType.DIRECTIONAL,
            frequency=AntennaFrequency.X_BAND,
            power_consumption=20.0,
            can_concurrent=False,
        )
        communication_capability = CommunicationCapability(antennas=[comm_antenna])

        satellite_specs = SatelliteSpecs(
            max_data_storage=1000.0,
            max_operation_time=4500.0,
            max_power=500.0,
            max_battery_capacity=2000.0,
            max_fuel=100.0,
        )

        sats.append(
            Satellite(
                satellite_id=sat_id,
                orbital_type=orbital_type,
                orbital_params=orbital_params,
                satellite_specs=satellite_specs,
                observation_capability=observation_capability,
                maneuverability_capability=maneuverability_capability,
                communication_capability=communication_capability,
            )
        )

    return sats


def build_missions_from_dict(missions_dict: dict) -> list[Mission]:
    """根据 generate_missions_dict 生成的任务字典构建 Mission 列表。
       / Build a list of Mission objects from the mission dictionary generated by generate_missions_dict."""
    missions: list[Mission] = []
    for mission_id, mission_data in missions_dict.items():
        latitude, longitude, priority, obs_req = mission_data
        target_location = TargetLocation(latitude=latitude, longitude=longitude, altitude=0.0)
        missions.append(
            Mission(
                mission_id=mission_id,
                target_location=target_location,
                priority=priority,
                observation_requirement=obs_req,
            )
        )
    return missions


def build_ground_stations_from_dict(ground_stations_dict: dict) -> list[GroundStation]:
    """简单地面站字典 -> GroundStation 列表。
       / Simple ground station dictionary -> List of GroundStation objects."""
    stations: list[GroundStation] = []
    for station_id, station_data in ground_stations_dict.items():
        latitude, longitude = station_data
        location = GroundStationLocation(latitude=latitude, longitude=longitude, altitude=0.0)
        stations.append(GroundStation(station_id=station_id, location=location, communication_capability=None))
    return stations


# ----------------------------------------------------------------------------
# 2) 目标文件路径解析 / Target file path resolution
# ----------------------------------------------------------------------------


def resolve_cities_file(cities_file_name: str) -> Path:
    """把用户输入的文件名解析成真实路径。 / Resolve the user-input filename into an actual path.

    支持： / Supports:
    - 直接给绝对/相对路径 / Passing absolute/relative paths directly
    - 只给文件名（例如 cities_01.json 或 cities_01），默认在 input/cities_data/ 下找
      / Passing only the filename (e.g., cities_01.json or cities_01), defaults to searching in input/cities_data/
    """
    p = Path(cities_file_name)
    if p.exists():
        return p

    name = cities_file_name
    if not name.lower().endswith(".json"):
        name = name + ".json"

    candidate = CITIES_DIR / name
    if candidate.exists():
        return candidate

    raise FileNotFoundError(
        f"找不到目标文件: {cities_file_name}。尝试路径: {p} / {candidate} "
        f"/ Target file not found: {cities_file_name}. Attempted paths: {p} / {candidate}"
    )


# ----------------------------------------------------------------------------
# 3) 子进程任务函数 / Subprocess task function
# ----------------------------------------------------------------------------


def _run_single_scenario_task(task_params: dict) -> dict:
    """ProcessPoolExecutor 的子进程入口。 / Subprocess entry point for ProcessPoolExecutor."""
    scenario_name: str = task_params["scenario_name"]
    sat_json_path: str = task_params["sat_json_path"]
    n_sat: int = task_params["n_sat"]
    n_missions: int = int(task_params.get("n_missions", 0))
    time_period_days: float = task_params["time_period_days"]
    mission_info: MissionInfo = task_params["mission_info"]
    ground_stations_dict: dict = task_params["ground_stations_dict"]
    ensure_each_mission_has_window: bool = task_params["ensure_each_mission_has_window"]
    targets_file_path: str | None = task_params.get("targets_file_path")
    targets_tag: str | None = task_params.get("targets_tag")

    try:
        t0 = time.time()

        # 1) 生成任务点 / Generate mission points
        missions_dict = generate_missions_dict(
            num_missions=n_missions,
            mission_info=mission_info,
            priority_range=(1, 10),
            duration_range=(5, 15),
            targets_file_path=targets_file_path,
        )

        # 实际任务数：读取目标文件模式下，以文件为准（忽略 missions_number）
        # / Actual number of missions: In target file mode, the file prevails (missions_number is ignored)
        n_missions = len(missions_dict)

        # 2) 构建对象（卫星在进程内缓存） / Construct objects (satellites are cached in the process)
        satellites = _load_satellites_from_json_path(sat_json_path)
        missions = build_missions_from_dict(missions_dict)
        ground_stations = build_ground_stations_from_dict(ground_stations_dict)

        metadata = ScenarioMetadata(
            name=scenario_name,
            creation_time=datetime(2025, 11, 18, 12, 0, 0),
            duration=3600 * 24 * time_period_days,
            time_step=1.0,  # 若 JSON 太大/太慢，可把它设为 5~30 秒 / If the JSON is too large/slow, this can be set to 5~30 seconds
        )

        scenario = Scenario(
            scenario_id=scenario_name,
            satellites=satellites,
            missions=missions,
            ground_stations=ground_stations,
            mission_info=mission_info,
            metadata=metadata,
            scenario_type=None,
        )

        # 3) 计算窗口（Orekit） / Calculate windows (Orekit)
        windows_dict = scenario.calculate_visibility_windows(
            ensure_each_mission_has_window=ensure_each_mission_has_window
        )
        obs_windows = windows_dict["observation_windows"]
        comm_windows = windows_dict["communication_windows"]

        elapsed = time.time() - t0

        # 文件名 tag / Filename tag
        if targets_file_path:
            tag = targets_tag or Path(targets_file_path).stem
        else:
            tag = f"dist{mission_info.distribution_type}"

        out_filename = f"{scenario_name}_Sats{n_sat}_M{n_missions}_T{time_period_days}d_{tag}.json"
        out_path = OUTPUT_DIR / out_filename

        scenario.export_to_json(filename=str(out_path), include_windows=True)

        print(
            f"[OK] {scenario_name}: sats={n_sat}, missions={n_missions}, T={time_period_days}d, "
            f"tag={tag}, obs={len(obs_windows)}, comm={len(comm_windows)}, "
            f"ensure={ensure_each_mission_has_window}, time={elapsed:.2f}s"
        )

        return {
            "scenario_name": scenario_name,
            "out_filename": out_filename,
            "n_sat": n_sat,
            "n_missions": n_missions,
            "time_period_days": time_period_days,
            "distribution_type": tag,  # summary 里复用该列 / Reuse this column in the summary
            "elapsed": elapsed,
            "ok": True,
            "error": None,
        }

    except Exception as e:
        print(f"[ERR] {scenario_name} execution failed: {e}")
        return {
            "scenario_name": scenario_name,
            "out_filename": None,
            "n_sat": n_sat,
            "n_missions": n_missions,
            "time_period_days": time_period_days,
            "distribution_type": task_params.get("targets_tag") or mission_info.distribution_type,
            "elapsed": 0.0,
            "ok": False,
            "error": repr(e),
        }


# ----------------------------------------------------------------------------
# 4) 主流程：构造任务组合 + 多进程并行 / Main workflow: Construct task combinations + multi-process parallelization
# ----------------------------------------------------------------------------


def run_all_scenarios(
    satellite_files: list[str],
    time_period_days_list: list[float],
    ground_stations_dict: dict,
    missions_number: list | None = None,
    targets_file_name: str | list[str] | None = None,
    max_workers: int | None = None,
) -> None:
    """并行批量生成场景。 / Generate scenarios in parallel batches.

    - targets_file_name=None : 随机生成任务点（会遍历 RANDOM_MISSION_INFOS），并强制 ensure_each_mission_has_window=True。
      / Randomly generate mission points (will iterate through RANDOM_MISSION_INFOS), and force ensure_each_mission_has_window=True.
    - targets_file_name 为 str 或 list[str] : 读取 input/cities_data/*.json 目标文件；任务数以文件为准；ensure_each_mission_has_window=False。
      / targets_file_name as str or list[str] : Read target files from input/cities_data/*.json; mission count depends on the file; ensure_each_mission_has_window=False.

    注意：读取目标文件模式下，不需要 missions_number；传了也会被忽略。
    / Note: In target file mode, missions_number is not needed; it will be ignored even if passed.
    """

    use_cities = targets_file_name is not None
    ensure_each_mission_has_window: bool = not use_cities

    # 目标文件列表（仅在 use_cities=True 时使用） / List of target files (used only when use_cities=True)
    targets_file_paths: list[str] = []
    targets_tags: list[str] = []

    if use_cities:
        if isinstance(targets_file_name, (list, tuple)):
            names = [str(x) for x in targets_file_name]
        else:
            names = [str(targets_file_name)]

        for name in names:
            p = resolve_cities_file(name)
            targets_file_paths.append(str(p))
            # tag 使用 stem（cities_01.json -> cities_01） / tag uses stem (cities_01.json -> cities_01)
            targets_tags.append(p.stem)

        print(f"[INFO] Using target file mode: Total {len(targets_file_paths)} sets of targets, ensure_each_mission_has_window=False")
        for p in targets_file_paths:
            print(f"       - {p}")
    else:
        print("[INFO] Using random generation mode, ensure_each_mission_has_window=True")

    # 1) 构造所有任务参数 / Construct all task parameters
    tasks: list[dict] = []
    total_scenarios = 0

    for sat_idx, sat_file_tag in enumerate(satellite_files):
        sat_json = INPUT_DIR / f"{sat_file_tag}.json"
        if not sat_json.exists():
            print(f"[WARN] Satellite data file does not exist: {sat_json}, skipping")
            continue

        # 在主进程读取一次仅用于拿 n_sat（不再把大字典传给子进程）
        # / Read once in the main process just to get n_sat (no longer passing the large dictionary to subprocesses)
        with open(sat_json, "r", encoding="utf-8") as f:
            sat_dict = json.load(f)
        n_sat = len(sat_dict)
        print(f"[INFO] Read satellite file {sat_json.name}, containing {n_sat} satellites")

        if use_cities:
            # 目标文件模式：任务数=文件内目标数；不再使用 missions_number，也不遍历 distribution_type
            # / Target file mode: Number of missions = number of targets in file; missions_number is no longer used, and distribution_type is not iterated
            for tf_path, tf_tag in zip(targets_file_paths, targets_tags):
                for time_period_days in time_period_days_list:
                    total_scenarios += 1
                    scenario_name = f"Scenario_S{total_scenarios}"
                    tasks.append(
                        {
                            "scenario_name": scenario_name,
                            "sat_json_path": str(sat_json),
                            "n_sat": n_sat,
                            # 占位：子进程会以 targets 文件为准覆盖 n_missions
                            # / Placeholder: Subprocess will overwrite n_missions based on the targets file
                            "n_missions": 0,
                            "time_period_days": float(time_period_days),
                            "mission_info": DUMMY_MISSION_INFO_FOR_CITIES,
                            "ground_stations_dict": ground_stations_dict,
                            "ensure_each_mission_has_window": False,
                            "targets_file_path": tf_path,
                            "targets_tag": tf_tag,
                        }
                    )
        else:
            # 随机生成模式：需要 missions_number / Random generation mode: requires missions_number
            if missions_number is None:
                # 兜底：避免用户忘记传 missions_number / Fallback: to prevent user from forgetting to pass missions_number
                mission_numbers_for_this_sat = [50]
            else:
                mission_numbers_for_this_sat = list(missions_number[sat_idx])

            for n_missions in mission_numbers_for_this_sat:
                for time_period_days in time_period_days_list:
                    for mission_info in RANDOM_MISSION_INFOS:
                        total_scenarios += 1
                        scenario_name = f"Scenario_S{total_scenarios}"

                        tasks.append(
                            {
                                "scenario_name": scenario_name,
                                "sat_json_path": str(sat_json),
                                "n_sat": n_sat,
                                "n_missions": int(n_missions),
                                "time_period_days": float(time_period_days),
                                "mission_info": mission_info,
                                "ground_stations_dict": ground_stations_dict,
                                "ensure_each_mission_has_window": True,
                                "targets_file_path": None,
                                "targets_tag": None,
                            }
                        )

    if not tasks:
        print("[WARN] No scenarios to execute, returning.")
        return

    # 2) 写 summary 文件头 / Write summary file header
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(SUMMARY_TXT, "w", encoding="utf-8") as f_sum:
        f_sum.write("Scenario File Record\n")
        f_sum.write(f"Generated at: {now_str}\n")
        f_sum.write(f"mode: {'cities' if use_cities else 'random'}\n")
        if use_cities:
            f_sum.write(f"targets_files: {targets_file_paths}\n")
        f_sum.write(f"ensure_each_mission_has_window: {ensure_each_mission_has_window}\n")
        f_sum.write("=" * 90 + "\n")
        header = (
            f"{'Filename':<52}"
            f"{'Satellites':>12}"
            f"{'Missions':>12}"
            f"{'Time(Days)':>14}"
            f"{'Dist/Target':>14}"
            f"{'Runtime(s)':>16}"
        )
        f_sum.write(header + "\n")
        f_sum.write("=" * 90 + "\n")

    # 3) 并行执行 / Parallel execution
    if max_workers is None:
        max_workers = max(1, (os.cpu_count() or 1) - 1)

    print("=" * 90)
    print(
        f"[INFO] Will calculate {len(tasks)} scenarios in parallel using {max_workers} processes, "
        f"mode={'cities' if use_cities else 'random'}, ensure={ensure_each_mission_has_window}"
    )

    finished = 0
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(_run_single_scenario_task, t) for t in tasks]

        for fut in as_completed(futures):
            result = fut.result()
            finished += 1

            if result["ok"] and result["out_filename"] is not None:
                with open(SUMMARY_TXT, "a", encoding="utf-8") as f_sum:
                    f_sum.write(
                        f"{result['out_filename']:<52}"
                        f"{result['n_sat']:>12}"
                        f"{result['n_missions']:>12}"
                        f"{result['time_period_days']:>14.2f}"
                        f"{str(result['distribution_type']):>14}"
                        f"{result['elapsed']:>16.3f}\n"
                    )
            else:
                print(f"[WARN] {result['scenario_name']} failed: {result['error']}")

            if finished % 10 == 0 or finished == len(tasks):
                print(f"[INFO] Progress: {finished}/{len(tasks)}")

    print("=" * 90)
    print(f"[INFO] All scenarios calculation finished, total scenarios: {len(tasks)}")
    print(f"[INFO] Scenario statistics TXT written to: {SUMMARY_TXT}")


# ----------------------------------------------------------------------------
# 5) 入口（你只需要改这几个变量即可） / Entry point (you only need to change these variables)
# ----------------------------------------------------------------------------

if __name__ == "__main__":

    satellite_files = [
        "1_satellites",
        # "3_satellites",
        # "5_satellites",
        # "10_satellites",
        # "20_satellites",
        # "50_satellites",
        # "100_satellites",
        # "200_satellites",
        # "500_satellites",
        # "1000_satellites",
        # "50_satellites_5_10",
        # "50_satellites_25_2",
        # "100_satellites_4_25",
        # "100_satellites_20_5",
        # "200_satellites_10_20",
        # "200_satellites_40_5",
        # "500_satellites_10_50",
        # "500_satellites_50_10",
        # "1000_satellites_10_100",
        # "1000_satellites_100_10",
    ]

    # 仿真时长（单位：天） / Simulation duration (Unit: days)
    time_period_days_list = [1]

    # 对应每种卫星数的任务数量组 / Number of missions corresponding to each satellite count
    missions_number = [
        (10, ), # 对应卫星数 1 / Corresponds to 1 satellite
        # (100, ), # 对应卫星数 3 / Corresponds to 3 satellites
        # (100, ), # 对应卫星数 5 / Corresponds to 5 satellites
        # (100, ), # 对应卫星数 10 / Corresponds to 10 satellites
        # (100, ), # 对应卫星数 20 / Corresponds to 20 satellites
        # (100, ), # 对应卫星数 50 / Corresponds to 50 satellites
        # (100, ), # 对应卫星数 100 / Corresponds to 100 satellites
        # (100, ), # 对应卫星数 200 / Corresponds to 200 satellites
        # (100, ), # 对应卫星数 500 / Corresponds to 500 satellites
        # (100, ), # 对应卫星数 1000 / Corresponds to 1000 satellites
    ]

    # 地面站信息 / Ground station information
    # 如需“无地面站”场景，只需改成 {} 即可 / If a "no ground station" scenario is needed, just change it to {}
    ground_stations_dict = {}
    # ground_stations_dict = {
    #     "GS-001": [25.4, 43.96],
    # }

    # 为 None 则使用随机生成。 / If None, random generation is used.
    targets_file_name: str | list[str] | None = None

    # # 若希望从 input/cities_data 读取目标文件，在这里写文件名：
    # # / If you want to read target files from input/cities_data, write the filenames here:
    # targets_file_name = ["cities_01",
    #                      "cities_02",
    #                      "cities_03",
    #                      "cities_04",
    #                      "cities_05",
    #                      "cities_06",
    #                      "cities_07",
    #                      "cities_08",
    #                      "cities_09",
    #                      "cities_10",]
    # targets_file_name = "cities_01.json"

    run_all_scenarios(
        satellite_files=satellite_files,
        time_period_days_list=time_period_days_list,
        missions_number=missions_number,
        ground_stations_dict=ground_stations_dict,
        targets_file_name=targets_file_name,
        max_workers=1,
    )