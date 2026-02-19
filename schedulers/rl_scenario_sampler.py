# -*- coding: utf-8 -*-
"""rl_scenario_sampler.py

为 PPO 训练生成“随机子场景”（临时 JSON）。
Generate "random sub-scenarios" (temporary JSONs) for PPO training.

用户需求对齐 / User Requirement Alignment
------------
训练数据来源：output/ 下所有 json（含子文件夹），但排除 output/schedules。
Training data source: All JSONs under output/ (including subfolders), excluding output/schedules.

每次 env.reset 时： / On each env.reset:
1) 先随机选一个 base json； / 1) Randomly select a base JSON;
2) 从文件名解析 base 规模：Sats(卫星数)、M(任务数)、T(周期天数，支持小数)； / 2) Parse base scale from filename: Sats (number of satellites), M (number of tasks), T (period in days, supports decimals);
3) 在 [1..base] 范围内随机生成 (S', M', T')，其中 T' 以 0.5 天为最小单位； / 3) Randomly generate (S', M', T') within [1..base], where T' has a minimum unit of 0.5 days;
4) 在 base json 里随机抽取 S' 颗卫星、M' 个任务，并过滤窗口数据，仅保留相关部分； / 4) Randomly sample S' satellites and M' tasks from the base JSON, filter window data, and keep only relevant parts;
5) 随机选择资源能力模式（Low/Standard/High/Mixed A/B/C）并注入到卫星 specs； / 5) Randomly select resource capacity mode (Low/Standard/High/Mixed A/B/C) and inject into satellite specs;
6) 随机选择敏捷姿态剖面（High/Standard/Low/Limited-Agility），作为训练环境的参数； / 6) Randomly select agile attitude profile (High/Standard/Low/Limited-Agility) as training environment parameters;
7) 将子场景写入临时文件（output/tmp_rl/ 目录），训练结束后可删除。 / 7) Write the sub-scenario to a temporary file (output/tmp_rl/ directory), which can be deleted after training.

说明 / Description
----
这里的“抽取子场景”尽量保持 JSON 结构不变（只做子集过滤 + 轻量修改），
The "sub-scenario extraction" here tries to keep the JSON structure unchanged (only subset filtering + lightweight modification),
从而复用你现有的 scenario_loader / constraint_model / candidate_pool 逻辑。
thereby reusing your existing scenario_loader / constraint_model / candidate_pool logic.
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
    # 与 scenario_loader._parse_iso_time 保持一致（这里避免循环依赖）
    # Consistent with scenario_loader._parse_iso_time (avoiding circular dependency here)
    s = (s or "").strip()
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    dt = datetime.fromisoformat(s)
    if dt.tzinfo is not None:
        dt = dt.astimezone(timezone.utc).replace(tzinfo=None)
    return dt


def _format_iso(dt: datetime) -> str:
    # 输出统一 +00:00 的字符串（简单起见）
    # Output unified +00:00 string (for simplicity)
    # 注意：你的 loader 会把 tzinfo 统一去掉，因此这里有没有 tzinfo 都无所谓。
    # Note: Your loader strips tzinfo uniformly, so it doesn't matter whether tzinfo is present here or not.
    return dt.replace(microsecond=0).isoformat() + "+00:00"


def _slice_angle_data(data: Any, start_idx: int, end_idx: int) -> Any:
    """对 agile_data / non_agile_data 做稳健切片。
       / Robust slicing for agile_data / non_agile_data.

    支持： / Supports:
    - list: 直接切片 / Direct slicing
    - dict: 对 dict 中的 list 值做同样切片（其余原样保留） / Apply the same slice to list values in dict (preserve others as is)
    - 其他类型：原样返回 / Other types: Return as is
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
    """从场景文件名（stem）解析 (Sats, M, T_days)。解析不到返回 None。
       / Parse (Sats, M, T_days) from scenario filename (stem). Return None if unparseable."""
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
    """对 raw json 的 satellites 列表就地修改：调整 per-orbit max capacity。
       / In-place modification of the satellites list in raw JSON: adjust per-orbit max capacity.

    注意：你真实 benchmark 的 capacity 模式可能有更精确的定义。
    / Note: Your real benchmark capacity modes might have more precise definitions.
    这里采用“缩放 max_data_storage_GB / max_power_W”的方式来制造多样性，
    / Here we use "scaling max_data_storage_GB / max_power_W" to create diversity,
    同时不改传感器消耗（这样约束会更紧/更松）。
    / while keeping sensor consumption unchanged (so constraints become tighter/looser).
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
        # 不改 / Do not change
        return
    if mode == "High-Capacity":
        for s in sats:
            scale_one(s, 1.6)
        return

    # Mixed：把卫星分组并按比例缩放 / Mixed: Group satellites and scale proportionally
    idx = list(range(n))
    rng.shuffle(idx)

    if mode == "Mixed-Capacity A":
        # 1/3 low, 1/3 std, 1/3 high
        a = n // 3
        b = 2 * n // 3
        low = idx[:a]
        std = idx[a:b]
        high = idx[b:]
    elif mode == "Mixed-Capacity B":
        # 25% low, 50% std, 25% high
        a = n // 4
        b = 3 * n // 4
        low = idx[:a]
        std = idx[a:b]
        high = idx[b:]
    else:  # Mixed-Capacity C
        # 50% low, 25% std, 25% high
        a = n // 2
        b = 3 * n // 4
        low = idx[:a]
        std = idx[a:b]
        high = idx[b:]

    for i in low:
        scale_one(sats[i], 0.6)
    for i in high:
        scale_one(sats[i], 1.6)
    # std 不变 / std remains unchanged


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
    """从 base_json 生成一个临时训练场景文件，并返回信息。
       / Generate a temporary training scenario file from base_json, and return its info."""
    raw = json.loads(Path(base_json).read_text(encoding="utf-8"))

    sats_base, tasks_base, days_base = parse_scale_from_name(Path(base_json).stem)
    # 兜底：解析不到就用实际数量 / Fallback: If unparseable, use actual counts
    sats_list = raw.get("satellites") or raw.get("satellite") or []
    missions_list = raw.get("missions") or raw.get("tasks") or []
    if sats_base is None:
        sats_base = len(sats_list)
    if tasks_base is None:
        tasks_base = len(missions_list)
    if days_base is None:
        # 从时间算 / Calculate from time
        try:
            st = _parse_iso(raw.get("start_time") or raw.get("scenario_start_time") or raw.get("start"))
            et = _parse_iso(raw.get("end_time") or raw.get("scenario_end_time") or raw.get("end"))
            days_base = max(0.5, (et - st).total_seconds() / 86400.0)
        except Exception:
            days_base = 1.0

    # 随机生成规模（范围：1..base） / Randomly generate scale (range: 1..base)
    sampled_sats = rng.randint(1, max(1, int(sats_base)))
    sampled_tasks = rng.randint(1, max(1, int(tasks_base)))
    # T'：0.5 天为步长 / T': 0.5 days as step size
    steps = max(1, int(round(days_base / 0.5)))
    sampled_days = rng.randint(1, steps) * 0.5

    # 随机选 capacity & agility / Randomly select capacity & agility
    capacity_mode = rng.choice(CAPACITY_MODES)
    agility_profile = rng.choice(AGILITY_PROFILES)

    # 选卫星 / Select satellites
    sat_ids = [s.get("id") for s in sats_list if isinstance(s, dict) and s.get("id") is not None]
    sat_ids = [str(x) for x in sat_ids]
    rng.shuffle(sat_ids)
    keep_sat_ids = set(sat_ids[:sampled_sats])
    sats_new = [s for s in sats_list if isinstance(s, dict) and str(s.get("id")) in keep_sat_ids]

    # 选任务(missions) / Select missions
    mission_ids = [m.get("id") for m in missions_list if isinstance(m, dict) and m.get("id") is not None]
    mission_ids = [str(x) for x in mission_ids]
    rng.shuffle(mission_ids)
    keep_mission_ids = set(mission_ids[:sampled_tasks])
    missions_new = [m for m in missions_list if isinstance(m, dict) and str(m.get("id")) in keep_mission_ids]

    # 时间范围裁剪 / Time range clipping
    # start_time/end_time 字段名以你 loader 支持的为准，这里尽量兼容。
    # The field names for start_time/end_time depend on your loader's support; trying to be compatible here.
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

    # 过滤 observation_windows / Filter observation_windows
    obs_wins = raw.get("observation_windows") or []
    obs_new: List[dict] = []
    for w in obs_wins:
        if not isinstance(w, dict):
            continue
        sid = str(w.get("satellite_id"))
        mid = str(w.get("mission_id"))
        if sid not in keep_sat_ids or mid not in keep_mission_ids:
            continue

        # 若有 st/et，则裁剪到 [st,et] / If st/et exist, clip to [st,et]
        if st is not None and et is not None:
            try:
                ws = _parse_iso(w.get("start_time"))
                we = _parse_iso(w.get("end_time"))
            except Exception:
                obs_new.append(w)
                continue

            # 无交集则跳过 / Skip if no intersection
            if we <= st or ws >= et:
                continue

            # clip / clip
            clip_s = max(ws, st)
            clip_e = min(we, et)
            if clip_e <= clip_s:
                continue

            time_step = float(w.get("time_step", 1.0) or 1.0)
            # 切片索引（按 step） / Slicing index (by step)
            start_idx = int(max(0.0, round((clip_s - ws).total_seconds() / time_step)))
            end_idx = int(max(0.0, round((clip_e - ws).total_seconds() / time_step)))
            if end_idx <= start_idx:
                continue

            # 更新时间 & 角度数据 / Update time & angle data
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

    # 过滤 comm windows / Filter comm windows
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

    # 写回 / Write back
    raw["satellites"] = sats_new
    raw["missions"] = missions_new
    raw["observation_windows"] = obs_new
    if "communication_windows" in raw:
        raw["communication_windows"] = comm_new

    # 应用 capacity mode / Apply capacity mode
    _apply_capacity_mode_to_satellites(raw["satellites"], capacity_mode, rng)

    # 记录 sampling 信息（便于训练日志排查） / Record sampling info (for easy troubleshooting in training logs)
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