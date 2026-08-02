# -*- coding: utf-8 -*-
"""
scenario_splitter.py

Single-file scenario splitting script.

Features:
1. Supports both single-scenario splitting and parallel batch splitting.
2. Accepts a full path, a path relative to the output directory, or a file name.
3. Writes results by default to a splits directory alongside output while
   preserving the relative directory structure.
4. Supports Low-, Standard-, High-, and Mixed-Capacity A/B/C modes.
5. Uses ijson to stream missions and observation_windows, making it suitable
   for large files.
6. Supports reproducible random splitting:
   - seed acts as the master seed;
   - when output_seeds is not supplied, the master seed reproducibly generates
     num_outputs distinct child seeds;
   - the corresponding split_seed is recorded in each output file name and in
     its metadata.

Dependencies:
- Required: ijson
- Optional: orjson

Usage:
- Edit the configuration under __main__ directly, then run:
    python scenario_splitter.py
"""

from __future__ import annotations

import contextlib
import copy
import decimal
import json
import math
import random
import re
import sys
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

try:
    import ijson
    HAS_IJSON = True
except ImportError:
    HAS_IJSON = False

try:
    import orjson
    HAS_ORJSON = True
except ImportError:
    HAS_ORJSON = False

PathLike = Union[str, Path]


# ============================================================
# Path and file-name utilities
# ============================================================
def sanitize_filename(name: str) -> str:
    name = name.replace(" ", "-")
    return re.sub(r'[\\/:*?"<>|]+', "_", name)


def locate_search_root(search_root: PathLike = "output") -> Path:
    """Locate the output root directory, prioritising the script directory and then the current working directory."""
    sr = Path(search_root)

    if sr.is_absolute():
        if sr.exists() and sr.is_dir():
            return sr.resolve()
        raise FileNotFoundError(f"search_root 绝对路径不存在：{sr}")

    script_dir = Path(__file__).resolve().parent
    cand = script_dir / sr
    if cand.exists() and cand.is_dir():
        return cand.resolve()

    cand2 = Path.cwd() / sr
    if cand2.exists() and cand2.is_dir():
        return cand2.resolve()

    raise FileNotFoundError(
        f"未找到 search_root='{search_root}'。\n"
        f"脚本目录: {script_dir}\n"
        f"当前工作目录: {Path.cwd()}"
    )


def resolve_scenario_path(scenario_name_or_path: PathLike, *, search_root: PathLike = "output") -> Path:
    """
    Resolve the input scenario path:
    - Full path: return it directly.
    - Path relative to output: join it to the output root and validate it.
    - File name only: search recursively beneath output.
    """
    p = Path(scenario_name_or_path)
    if p.is_file():
        return p.resolve()

    root = locate_search_root(search_root)

    p_json = p if p.suffix.lower() == ".json" else p.with_suffix(".json")
    direct = root / p_json
    if direct.is_file():
        return direct.resolve()

    hits = list(root.rglob(p_json.name))
    if not hits:
        raise FileNotFoundError(f"在 {root} 下未找到：{p_json.name}")
    if len(hits) > 1:
        hits.sort(key=lambda x: len(str(x)))
        print(f"[WARN] 找到多个同名文件 {p_json.name}，默认使用：{hits[0]}")
    return hits[0].resolve()


def find_output_root_from_inpath(in_path: Path, fallback_root: Path) -> Path:
    for anc in [in_path.parent] + list(in_path.parents):
        if anc.name.lower() == "output":
            return anc.resolve()
    return fallback_root.resolve()


def infer_splits_out_dir(in_path: Path, *, output_root: Path) -> Path:
    """Map output/A/B/file.json to the output directory splits/A/B/."""
    splits_root = output_root.parent / "splits"
    try:
        rel = in_path.parent.resolve().relative_to(output_root.resolve())
        return splits_root / rel
    except Exception:
        return splits_root / "_external" / in_path.parent.name


# ============================================================
# JSON and date-time utilities
# ============================================================
def _json_default(obj: Any) -> Any:
    if isinstance(obj, decimal.Decimal):
        return float(obj)
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


def json_dumps(obj: Any) -> str:
    if HAS_ORJSON:
        return orjson.dumps(obj, default=_json_default).decode("utf-8")
    return json.dumps(obj, ensure_ascii=False, default=_json_default)


def parse_iso_dt(value: str) -> datetime:
    value = value.strip()
    if value.endswith("Z"):
        value = value[:-1] + "+00:00"
    dt = datetime.fromisoformat(value)
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def parse_offset_seconds(value: str) -> float:
    value = value.strip().lower()
    if not value:
        return 0.0

    try:
        return float(value)
    except ValueError:
        pass

    unit = value[-1]
    number = float(value[:-1])
    if unit == "s":
        return number
    if unit == "m":
        return number * 60.0
    if unit == "h":
        return number * 3600.0
    if unit == "d":
        return number * 86400.0
    raise ValueError(f"无法解析偏移：{value}")


def format_days(days: float) -> str:
    if abs(days - round(days)) < 1e-9:
        return str(int(round(days)))
    return f"{days:.6f}".rstrip("0").rstrip(".")


def update_scenario_stem(stem: str, *, sat_count: int, mission_count: int, duration_seconds: float) -> str:
    out = stem
    out = re.sub(r"Sats\d+", f"Sats{sat_count}", out) if "Sats" in out else out + f"_Sats{sat_count}"
    out = re.sub(r"_M\d+", f"_M{mission_count}", out) if "_M" in out else out + f"_M{mission_count}"

    days = duration_seconds / 86400.0
    t_str = format_days(days)
    out = re.sub(r"_T\d+(?:\.\d+)?d", f"_T{t_str}d", out) if "_T" in out else out + f"_T{t_str}d"
    return out.replace(".0d", "d")


# ============================================================
# Random-seed utilities
# ============================================================
def build_output_seeds(
    *,
    num_outputs: int,
    master_seed: int,
    output_seeds: Optional[Sequence[int]] = None,
) -> List[int]:
    """
    Generate the random seed associated with each output file.

    Rules:
    1. When output_seeds is supplied explicitly, use it directly and derive the
       number of outputs from its length.
    2. When output_seeds is not supplied:
       - num_outputs == 1 -> [master_seed]
       - num_outputs > 1  -> use master_seed to reproducibly generate
         num_outputs distinct child seeds.
    """
    if output_seeds is not None:
        seeds = [int(x) for x in output_seeds]
        if not seeds:
            raise ValueError("output_seeds 不能为空。")
        if len(set(seeds)) != len(seeds):
            raise ValueError("output_seeds 中存在重复值，请提供互不相同的随机种子。")
        return seeds

    if num_outputs <= 0:
        raise ValueError("num_outputs 必须 >= 1")

    if num_outputs == 1:
        return [int(master_seed)]

    rng = random.Random(int(master_seed))
    seen = set()
    seeds: List[int] = []
    while len(seeds) < num_outputs:
        candidate = rng.randrange(0, 2**31 - 1)
        if candidate not in seen:
            seen.add(candidate)
            seeds.append(candidate)
    return seeds


# ============================================================
# Capacity-mode utilities
# ============================================================
CAPACITY_PATTERNS = {
    "Low-Capacity": {"type": "hom", "scale": 0.5},
    "Standard-Capacity": {"type": "hom", "scale": 1.0},
    "High-Capacity": {"type": "hom", "scale": 1.5},
    "Mixed-Capacity A": {"type": "mix", "p": {"high": 0.20, "std": 0.60, "low": 0.20}},
    "Mixed-Capacity B": {"type": "mix", "p": {"high": 0.25, "std": 0.50, "low": 0.25}},
    "Mixed-Capacity C": {"type": "mix", "p": {"high": 0.30, "std": 0.40, "low": 0.30}},
}
SCALE_MAP = {"low": 0.5, "std": 1.0, "high": 1.5}


def get_capacity_scales(n: int, pattern: str, seed: int) -> List[float]:
    if pattern not in CAPACITY_PATTERNS:
        return [1.0] * n

    info = CAPACITY_PATTERNS[pattern]
    if info["type"] == "hom":
        return [info["scale"]] * n

    rng = random.Random(seed)
    counts = {k: int(math.floor(v * n)) for k, v in info["p"].items()}
    remainder = n - sum(counts.values())
    for _ in range(remainder):
        counts["std"] += 1

    scales = (
        [SCALE_MAP["high"]] * counts["high"]
        + [SCALE_MAP["std"]] * counts["std"]
        + [SCALE_MAP["low"]] * counts["low"]
    )
    rng.shuffle(scales)
    return scales


def apply_capacity_specs(sat: Dict[str, Any], scale: float, base: Dict[str, float]) -> Dict[str, Any]:
    specs = sat.setdefault("satellite_specs", {})
    specs["max_data_storage_GB"] = base["max_data_storage_GB"] * scale
    specs["max_power_W"] = base["max_power_W"] * scale

    for sensor in sat.get("observation_capability", {}).get("sensors", []):
        sensor["data_rate_Mbps"] = base["data_rate_Mbps"] * scale
        sensor["power_consumption_W"] = base["power_consumption_W"] * scale

    sat.setdefault("extra", {})["capacity_scale"] = scale
    return sat


# ============================================================
# Core splitting logic
# ============================================================
def _read_header_and_ids(in_path: Path) -> Tuple[Any, Any, Any, List[str]]:
    if not HAS_IJSON:
        raise ImportError("Need ijson: pip install ijson")

    with in_path.open("rb") as f:
        metadata = next(ijson.items(f, "metadata"))

        f.seek(0)
        satellites = list(ijson.items(f, "satellites.item"))

        scenario_type = None
        f.seek(0)
        scenario_type_iter = ijson.items(f, "scenario_type")
        try:
            scenario_type = next(scenario_type_iter)
        except StopIteration:
            scenario_type = None

        f.seek(0)
        mission_ids = [sys.intern(str(m["id"])) for m in ijson.items(f, "missions.item")]

    return metadata, satellites, scenario_type, mission_ids


def split_scenario_file(
    scenario_name_or_path: PathLike,
    *,
    num_missions: int,
    start_offset: Optional[str] = None,
    end_offset: Optional[str] = None,
    duration: Optional[str] = None,
    start_iso: Optional[str] = None,
    end_iso: Optional[str] = None,
    search_root: PathLike = "output",
    out_dir: Optional[PathLike] = None,
    out_name: Optional[str] = None,
    num_outputs: int = 1,
    seed: int = 0,
    output_seeds: Optional[Sequence[int]] = None,
    # The following parameters are retained for backwards compatibility.
    ensure_sat_coverage: bool = True,
    strict_sat_coverage: bool = False,
    balance_visibility: bool = True,
    prefer_more_windows: bool = True,
    capacity_pattern: Optional[str] = None,
    base_max_data_storage_GB: Optional[float] = None,
    base_max_power_W: Optional[float] = None,
    base_data_rate_Mbps: Optional[float] = None,
    base_power_consumption_W: Optional[float] = None,
    capacity_seed: Optional[int] = None,
) -> Union[Path, List[Path]]:
    del ensure_sat_coverage, strict_sat_coverage, balance_visibility, prefer_more_windows

    if num_missions <= 0:
        raise ValueError("num_missions 必须 >= 1")

    in_path = resolve_scenario_path(scenario_name_or_path, search_root=search_root)
    root = locate_search_root(search_root)
    output_root = find_output_root_from_inpath(in_path, fallback_root=root)
    out_dir_path = Path(out_dir) if out_dir is not None else infer_splits_out_dir(in_path, output_root=output_root)

    print(f"[Split] Reading {in_path.name} ...")
    metadata, satellites, scenario_type, all_mission_ids = _read_header_and_ids(in_path)
    print(f"[Split] Found {len(all_mission_ids)} total missions.")

    creation_dt = parse_iso_dt(metadata["creation_time"])
    if start_iso and end_iso:
        start_dt = parse_iso_dt(start_iso)
        end_dt = parse_iso_dt(end_iso)
    else:
        start_dt = creation_dt + timedelta(seconds=parse_offset_seconds(start_offset or "0s"))
        if end_offset:
            end_dt = creation_dt + timedelta(seconds=parse_offset_seconds(end_offset))
        elif duration:
            end_dt = start_dt + timedelta(seconds=parse_offset_seconds(duration))
        else:
            raise ValueError("必须提供 start_iso+end_iso，或 start_offset+duration/end_offset。")

    if end_dt <= start_dt:
        raise ValueError("结束时间必须晚于开始时间。")

    out_dir_path.mkdir(parents=True, exist_ok=True)
    duration_seconds = (end_dt - start_dt).total_seconds()
    base_stem = update_scenario_stem(
        out_name or in_path.stem,
        sat_count=len(satellites),
        mission_count=num_missions,
        duration_seconds=duration_seconds,
    )

    split_seeds = build_output_seeds(
        num_outputs=num_outputs,
        master_seed=int(seed),
        output_seeds=output_seeds,
    )
    num_outputs = len(split_seeds)
    print(f"[Split] Master seed = {int(seed)}")
    print(f"[Split] Output seeds = {split_seeds}")

    base_cap: Dict[str, float] = {}
    if capacity_pattern:
        if base_max_data_storage_GB is None:
            sat0 = satellites[0]

            def to_float(x: Any) -> float:
                return float(x) if not isinstance(x, decimal.Decimal) else float(x)

            base_cap = {
                "max_data_storage_GB": to_float(sat0.get("satellite_specs", {}).get("max_data_storage_GB", 0)),
                "max_power_W": to_float(sat0.get("satellite_specs", {}).get("max_power_W", 0)),
                "data_rate_Mbps": to_float(
                    sat0.get("observation_capability", {}).get("sensors", [{}])[0].get("data_rate_Mbps", 0)
                ),
                "power_consumption_W": to_float(
                    sat0.get("observation_capability", {}).get("sensors", [{}])[0].get("power_consumption_W", 0)
                ),
            }
        else:
            if base_data_rate_Mbps is None or base_power_consumption_W is None:
                raise ValueError("当手动提供容量基准参数时，四个 base_* 参数都需要提供。")
            base_cap = {
                "max_data_storage_GB": float(base_max_data_storage_GB),
                "max_power_W": float(base_max_power_W),
                "data_rate_Mbps": float(base_data_rate_Mbps),
                "power_consumption_W": float(base_power_consumption_W),
            }

    output_contexts: List[Dict[str, Any]] = []
    for idx, split_seed in enumerate(split_seeds):
        rng = random.Random(split_seed)
        if num_missions >= len(all_mission_ids):
            selected = all_mission_ids[:]
        else:
            selected = rng.sample(all_mission_ids, num_missions)

        suffix_parts: List[str] = []
        if len(split_seeds) > 1:
            suffix_parts.append(f"p{idx + 1}")
        suffix_parts.append(f"seed{split_seed}")
        suffix = "_" + "_".join(suffix_parts)

        scenario_id = f"{base_stem}{suffix}" + (f"_{capacity_pattern}" if capacity_pattern else "")
        scenario_id = sanitize_filename(scenario_id)

        out_meta = copy.deepcopy(metadata)
        out_meta["creation_time"] = start_dt.isoformat()
        out_meta["duration"] = duration_seconds
        out_meta.setdefault("extra", {})["original_mission_ids"] = selected
        out_meta["extra"]["split_master_seed"] = int(seed)
        out_meta["extra"]["split_seed"] = int(split_seed)
        out_meta["extra"]["split_output_index"] = idx + 1
        out_meta["extra"]["split_num_outputs"] = len(split_seeds)
        if capacity_pattern:
            out_meta["extra"]["capacity_pattern"] = capacity_pattern

        out_sats = copy.deepcopy(satellites)
        current_capacity_seed = None
        if capacity_pattern:
            current_capacity_seed = int((capacity_seed if capacity_seed is not None else split_seed))
            scales = get_capacity_scales(len(out_sats), capacity_pattern, current_capacity_seed)
            for i, sat in enumerate(out_sats):
                apply_capacity_specs(sat, scales[i], base_cap)

        if current_capacity_seed is not None:
            out_meta["extra"]["capacity_seed"] = current_capacity_seed

        for sat in out_sats:
            if "epoch" in sat.get("orbital_params", {}):
                sat["orbital_params"]["epoch"] = start_dt.isoformat()

        output_contexts.append(
            {
                "path": out_dir_path / f"{scenario_id}.json",
                "scenario_id": scenario_id,
                "scenario_type": scenario_type,
                "metadata": out_meta,
                "satellites": out_sats,
                "selected_set": set(selected),
                "id_map": {},
                "mission_counter": 1,
                "fh": None,
                "is_first_mission": True,
                "is_first_window": True,
                "split_seed": int(split_seed),
            }
        )

    print(f"[Split] Streaming to {len(output_contexts)} file(s) ...")

    with contextlib.ExitStack() as stack:
        for ctx in output_contexts:
            fh = stack.enter_context(ctx["path"].open("w", encoding="utf-8"))
            ctx["fh"] = fh
            fh.write("{\n")
            fh.write(f'  "scenario_id": {json_dumps(ctx["scenario_id"])},\n')
            if ctx["scenario_type"] is None:
                fh.write('  "scenario_type": null,\n')
            else:
                fh.write(f'  "scenario_type": {json_dumps(ctx["scenario_type"])},\n')
            fh.write(f'  "metadata": {json_dumps(ctx["metadata"])},\n')
            fh.write(f'  "satellites": {json_dumps(ctx["satellites"])},\n')
            fh.write('  "missions": [\n')

        with in_path.open("rb") as f_in:
            for mission in ijson.items(f_in, "missions.item"):
                old_id = str(mission.get("id"))
                for ctx in output_contexts:
                    if old_id not in ctx["selected_set"]:
                        continue

                    new_id = f"M{ctx['mission_counter']:03d}"
                    ctx["id_map"][old_id] = new_id
                    ctx["mission_counter"] += 1

                    mission_copy = mission.copy()
                    mission_copy["id"] = new_id

                    fh = ctx["fh"]
                    if not ctx["is_first_mission"]:
                        fh.write(',\n')
                    fh.write('    ')
                    fh.write(json_dumps(mission_copy))
                    ctx["is_first_mission"] = False

        for ctx in output_contexts:
            ctx["fh"].write('\n  ],\n  "observation_windows": [\n')

        with in_path.open("rb") as f_in:
            for obs in ijson.items(f_in, "observation_windows.item"):
                old_mission_id = str(obs.get("mission_id"))
                relevant_contexts = [ctx for ctx in output_contexts if old_mission_id in ctx["id_map"]]
                if not relevant_contexts:
                    continue

                valid_windows = []
                for tw in obs.get("time_windows", []):
                    try:
                        t1 = parse_iso_dt(tw["start_time"])
                        t2 = parse_iso_dt(tw["end_time"])
                        if t1 >= start_dt and t2 <= end_dt:
                            valid_windows.append(tw)
                    except Exception:
                        continue

                if not valid_windows:
                    continue

                for ctx in relevant_contexts:
                    new_obs = {
                        "request_id": obs.get("request_id"),
                        "mission_id": ctx["id_map"][old_mission_id],
                        "satellite_id": obs.get("satellite_id"),
                        "sensor_id": obs.get("sensor_id"),
                        "time_windows": valid_windows,
                    }
                    for key, value in obs.items():
                        if key not in new_obs:
                            new_obs[key] = value

                    fh = ctx["fh"]
                    if not ctx["is_first_window"]:
                        fh.write(',\n')
                    fh.write('    ')
                    fh.write(json_dumps(new_obs))
                    ctx["is_first_window"] = False

        for ctx in output_contexts:
            ctx["fh"].write('\n  ]\n}')

    print(f"[Split] Done. Generated {len(output_contexts)} file(s).")
    return [ctx["path"] for ctx in output_contexts] if len(output_contexts) > 1 else output_contexts[0]["path"]


# ============================================================
# Parallel batch splitting
# ============================================================
def build_split_tasks(
    split_jobs: List[Dict[str, Any]],
    *,
    start_offset: Optional[str],
    start_iso: Optional[str],
    end_iso: Optional[str],
    search_root: PathLike,
    out_dir: Optional[PathLike],
    out_name: Optional[str],
    base_max_data_storage_GB: float,
    base_max_power_W: float,
    base_data_rate_Mbps: float,
    base_power_consumption_W: float,
    seed_offset_per_scenario_index: bool = True,
) -> List[Dict[str, Any]]:
    tasks: List[Dict[str, Any]] = []

    for job_idx, job in enumerate(split_jobs):
        scenario_field = job["scenario_file"]
        scenario_list = list(scenario_field) if isinstance(scenario_field, (list, tuple)) else [scenario_field]
        seed0 = int(job.get("seed", 0))

        for sf_i, scenario_file in enumerate(scenario_list):
            task_seed = seed0 + sf_i if seed_offset_per_scenario_index else seed0

            for num_missions in job["num_missions_s"]:
                for duration in job["duration_s"]:
                    for capacity_pattern in job["capacity_pattern_s"]:
                        base = out_name or Path(str(scenario_file)).stem
                        derived_out_name = sanitize_filename(
                            f"{base}_job{job_idx}_sf{sf_i}_M{num_missions}_T{duration}_{capacity_pattern}_masterseed{task_seed}"
                        )
                        tasks.append(
                            {
                                "scenario_name_or_path": scenario_file,
                                "num_missions": int(num_missions),
                                "start_offset": start_offset,
                                "duration": duration,
                                "start_iso": start_iso,
                                "end_iso": end_iso,
                                "search_root": search_root,
                                "out_dir": out_dir,
                                "out_name": derived_out_name,
                                "num_outputs": int(job["num_outputs"]),
                                "seed": int(task_seed),
                                "output_seeds": job.get("output_seeds"),
                                "capacity_pattern": capacity_pattern,
                                "base_max_data_storage_GB": base_max_data_storage_GB,
                                "base_max_power_W": base_max_power_W,
                                "base_data_rate_Mbps": base_data_rate_Mbps,
                                "base_power_consumption_W": base_power_consumption_W,
                                "capacity_seed": job.get("capacity_seed"),
                                "_job_idx": job_idx,
                                "_sf_i": sf_i,
                                "_scenario_display": str(scenario_file),
                            }
                        )
    return tasks


def _run_one_split_task(task: Dict[str, Any]) -> Tuple[Dict[str, Any], Any]:
    kwargs = {k: v for k, v in task.items() if not str(k).startswith("_")}
    result = split_scenario_file(**kwargs)
    return task, result


def run_split_tasks_parallel(tasks: List[Dict[str, Any]], *, max_workers: int = 4) -> Tuple[int, int]:
    ok, fail = 0, 0
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(_run_one_split_task, task) for task in tasks]
        for future in as_completed(futures):
            try:
                task, result = future.result()
                ok += 1
                print(
                    f"[OK {ok}/{len(tasks)}] "
                    f"job={task.get('_job_idx')} sf={task.get('_sf_i')} "
                    f"file={Path(task.get('_scenario_display')).name} "
                    f"M={task['num_missions']} T={task['duration']} pattern={task['capacity_pattern']} "
                    f"master_seed={task['seed']}"
                )
                print(f"    -> {result}")
            except Exception as exc:
                fail += 1
                print(f"[FAIL {fail}] {exc}")
                print(traceback.format_exc())
    return ok, fail


if __name__ == "__main__":
    # ========================================================
    # Execution mode
    # ========================================================
    MODE = "batch"   # Options: "single" or "batch"

    # ========================================================
    # Common parameters
    # ========================================================
    SEARCH_ROOT = "output"
    START_OFFSET = "0h"
    START_ISO = None
    END_ISO = None
    OUT_DIR = None
    OUT_NAME = None

    BASE_MAX_DATA_STORAGE_GB = 10.0
    BASE_MAX_POWER_W = 750.0
    BASE_DATA_RATE_MBPS = 400.0
    BASE_POWER_CONSUMPTION_W = 50.0

    # ========================================================
    # Single-scenario splitting configuration
    # Notes:
    # 1. If output_seeds is None, seed is used as the master seed to generate num_outputs distinct child seeds.
    # 2. If output_seeds is supplied explicitly, it takes precedence and num_outputs is derived from its length.
    # ========================================================
    SINGLE_CONFIG = {
        "scenario_name_or_path": "Scenario_Sats100_M600_T7d_dist1",
        "num_missions": 600,
        "start_offset": "0h",
        "duration": "72h",
        "search_root": SEARCH_ROOT,
        "out_dir": OUT_DIR,
        "out_name": OUT_NAME,
        "num_outputs": 10,
        "seed": 20260401,
        "output_seeds": None,
        # Example of specifying the seeds manually:
        # "output_seeds": [101, 102, 103, 104, 105, 106, 107, 108, 109, 110],
        "capacity_pattern": "Standard-Capacity",
        "base_max_data_storage_GB": BASE_MAX_DATA_STORAGE_GB,
        "base_max_power_W": BASE_MAX_POWER_W,
        "base_data_rate_Mbps": BASE_DATA_RATE_MBPS,
        "base_power_consumption_W": BASE_POWER_CONSUMPTION_W,
    }

    # ========================================================
    # Batch splitting configuration
    # Notes:
    # - seed: master seed
    # - output_seeds: optional; when supplied, each output in this job group uses one of these child seeds.
    # ========================================================
    SPLIT_JOBS = [
        {
            "scenario_file": [
                "Scenario_Sats500_M8360_T3d_dist0",
                "Scenario_Sats500_M10350_T3d_dist1",
            ],
            "num_missions_s": [2000, 5000],
            "duration_s": ["12h", "24h", "72h"],
            "capacity_pattern_s": ["Standard-Capacity"],
            "num_outputs": 10,
            "seed": 20260401,
            "output_seeds": None,
            # Example: "output_seeds": [1001,1002,1003,1004,1005,1006,1007,1008,1009,1010],
        },
    ]

    MAX_WORKERS = 4

    if MODE == "single":
        result = split_scenario_file(**SINGLE_CONFIG)
        print("Split ->", result)
    else:
        tasks = build_split_tasks(
            split_jobs=SPLIT_JOBS,
            start_offset=START_OFFSET,
            start_iso=START_ISO,
            end_iso=END_ISO,
            search_root=SEARCH_ROOT,
            out_dir=OUT_DIR,
            out_name=OUT_NAME,
            base_max_data_storage_GB=BASE_MAX_DATA_STORAGE_GB,
            base_max_power_W=BASE_MAX_POWER_W,
            base_data_rate_Mbps=BASE_DATA_RATE_MBPS,
            base_power_consumption_W=BASE_POWER_CONSUMPTION_W,
        )
        print(f"[Parallel split] tasks={len(tasks)}, max_workers={MAX_WORKERS}")
        ok, fail = run_split_tasks_parallel(tasks, max_workers=MAX_WORKERS)
        print(f"Done. Success={ok}, Failed={fail}")
