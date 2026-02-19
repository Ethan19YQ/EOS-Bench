# algorithms/candidate_pool.py
# -*- coding: utf-8 -*-

from __future__ import annotations

from typing import Dict, List, Tuple, Optional
import random

from schedulers.scenario_loader import SchedulingProblem, SchedulingTask, CommWindow
from schedulers.constraint_model import Assignment, TimePlacementStrategy

from algorithms.random_utils import make_rng



def _estimate_data_volume_GB(problem: SchedulingProblem, w, sat_start, sat_end) -> float:
    """估计该候选的容量消耗（GB）。按 data_rate_Mbps * duration_s 转换。"""
    try:
        dur = (sat_end - sat_start).total_seconds()
    except Exception:
        dur = getattr(problem.tasks.get(w.mission_id, None), "required_duration", 0.0)
    sat = problem.satellites.get(w.satellite_id)
    if sat is None:
        return 0.0
    sensor_id = getattr(w, "sensor_id", "") or ""
    spec = sat.sensors.get(sensor_id)
    if spec is None:
        return 0.0
    mbits = float(spec.data_rate_Mbps) * float(dur)
    # Mb -> GB (1 GB = 1024 MB, 1 MB = 8 Mb)
    return mbits / (8.0 * 1024.0)

def _estimate_power_cost_W(problem: SchedulingProblem, w) -> float:
    """估计该候选的能量/功率消耗（按任务次数计，单位沿用 W 字段）。"""
    sat = problem.satellites.get(w.satellite_id)
    if sat is None:
        return 0.0
    sensor_id = getattr(w, "sensor_id", "") or ""
    spec = sat.sensors.get(sensor_id)
    if spec is None:
        return 0.0
    return float(spec.power_consumption_W)
def _overlap(a_start, a_end, b_start, b_end) -> bool:
    return not (a_end <= b_start or a_start >= b_end)


def _place_latest(window_start, window_end, required_duration):
    total = (window_end - window_start).total_seconds()
    if required_duration > total:
        return None
    end = window_end
    start = end - (window_end - window_start).__class__(seconds=required_duration)  # timedelta
    return start, end


def _place_random(window_start, window_end, required_duration, rng: random.Random):
    total = (window_end - window_start).total_seconds()
    if required_duration > total:
        return None
    slack = total - required_duration
    offset = rng.random() * slack
    start = window_start + (window_end - window_start).__class__(seconds=offset)
    end = start + (window_end - window_start).__class__(seconds=required_duration)
    return start, end


def enumerate_task_candidates(
    problem: SchedulingProblem,
    task: SchedulingTask,
    placement_mode: str = "earliest",
    downlink_duration_ratio: float = 1.0,
    max_candidates: int = 256,
    random_samples_per_window: int = 1,
    seed: Optional[int] = None,
    prefer_must_first: bool = True,
) -> List[Assignment]:
    """
    给单个任务生成候选 Assignment（不考虑与其它任务冲突，只保证窗口合法）。
    这版“多样化”候选：earliest/center/latest + 随机采样若干。
    """
    if task.required_duration <= 0 or not task.windows:
        return []

    rng = make_rng(seed)

    obs_windows = sorted(task.windows, key=lambda w: w.start_time)
    has_gs = len(problem.ground_stations) > 0

    comm_by_sat: Dict[str, List[CommWindow]] = {}
    if has_gs:
        for cw in problem.comm_windows:
            comm_by_sat.setdefault(cw.satellite_id, []).append(cw)
        for sid in comm_by_sat:
            comm_by_sat[sid].sort(key=lambda x: x.start_time)

    cands: List[Assignment] = []
    downlink_dur = task.required_duration * float(downlink_duration_ratio)

    def push_candidate(w, sat_start, sat_end, sat_angles=None, cw=None, gs_start=None, gs_end=None):
        cands.append(
            Assignment(
                task_id=task.id,
                satellite_id=w.satellite_id,
                sat_start_time=sat_start,
                sat_end_time=sat_end,
                sat_window_id=w.window_id,
                sensor_id=getattr(w, 'sensor_id', ''),
                orbit_number=int(getattr(w, 'orbit_number', 0) or 0),
                data_volume_GB=_estimate_data_volume_GB(problem, w, sat_start, sat_end),
                power_cost_W=_estimate_power_cost_W(problem, w),
                sat_angles=sat_angles,
                ground_station_id=(None if cw is None else cw.ground_station_id),
                gs_start_time=gs_start,
                gs_end_time=gs_end,
                gs_window_id=(None if cw is None else cw.window_id),
            )
        )

    # 观测窗口内的多种放置
    for w in obs_windows:
        if len(cands) >= max_candidates:
            break

        placements: List[tuple] = []

        sat = problem.satellites.get(w.satellite_id)
        sat_type = getattr(sat, "maneuverability_type", "agile") if sat is not None else "agile"

        total = (w.end_time - w.start_time).total_seconds()
        dur = task.required_duration
        step = float(getattr(w, "time_step", 1.0) or 1.0)

        if dur <= 0 or dur > total:
            continue

        max_offset = total - dur
        import math
        max_k = int(math.floor((max_offset + 1e-9) / step))
        if max_k < 0:
            continue

        def _slice_angles_payload(payload, start_idx: int, count: int):
            """对角度数据做“按时间步切片”。
            - list: 直接切片
            - dict: 对其中的 list 递归切片（常见：{roll:[...], pitch:[...]}）
            - 其它：原样返回
            """
            if payload is None:
                return None
            if isinstance(payload, list):
                return payload[start_idx : start_idx + count]
            if isinstance(payload, dict):
                out = {}
                for kk, vv in payload.items():
                    if isinstance(vv, (list, dict)):
                        out[kk] = _slice_angles_payload(vv, start_idx, count)
                    else:
                        out[kk] = vv
                return out
            return payload

        def _angles_for(k: int):
            # 角度数据：敏捷 -> 取子窗口对应的切片；非敏捷 -> 单组数据
            if sat_type == "non_agile":
                return getattr(w, "non_agile_data", None)

            ad = getattr(w, "agile_data", None)
            # 用 ceil 保证覆盖 duration（避免 round 导致短一格）
            n = max(1, int(math.ceil((dur / step) - 1e-9)))
            return _slice_angles_payload(ad, k, n)

        from datetime import timedelta


        # 非敏捷：只允许取“中间那一段”满足任务时长的窗口（对齐 time_step）
        if sat_type == "non_agile":
            center_k = int(round((max_offset / 2.0) / step))
            center_k = max(0, min(max_k, center_k))
            sat_start = w.start_time + timedelta(seconds=center_k * step)
            sat_end = sat_start + timedelta(seconds=dur)
            placements.append((sat_start, sat_end, _angles_for(center_k)))
        else:
            # 敏捷：按 time_step 枚举所有可行子窗口开始位置（必要时打散顺序以提供随机性）
            ks = list(range(0, max_k + 1))

            # 先保证 earliest/center/latest 都在前面（更容易得到“合理”解）
            must = [0, int(round((max_offset / 2.0) / step)), max_k]
            must = [k for k in must if 0 <= k <= max_k]
            must_uniq = []
            for k in must:
                if k not in must_uniq:
                    must_uniq.append(k)

            rest = [k for k in ks if k not in must_uniq]
            if prefer_must_first:
                rng.shuffle(rest)
                ks = must_uniq + rest
            else:
                ks = must_uniq + rest
                rng.shuffle(ks)

            # 逐个生成（会被 max_candidates 截断）
            for k in ks:
                sat_start = w.start_time + timedelta(seconds=k * step)
                sat_end = sat_start + timedelta(seconds=dur)
                placements.append((sat_start, sat_end, _angles_for(k)))

        # 去重（同一个窗口可能产生重复）
        uniq = []
        seen = set()
        for s, e, ang in placements:
            key = (s, e)
            if key in seen:
                continue
            seen.add(key)
            uniq.append((s, e, ang))

        for sat_start, sat_end, sat_angles in uniq:
            if len(cands) >= max_candidates:
                break

            if not has_gs:
                push_candidate(w, sat_start, sat_end, sat_angles)
                continue

            for cw in comm_by_sat.get(w.satellite_id, []):
                if len(cands) >= max_candidates:
                    break

                earliest = max(cw.start_time, sat_end)
                avail = (cw.end_time - earliest).total_seconds()
                if avail < downlink_dur:
                    continue

                # downlink 多样化放置
                dl_places = []
                p = TimePlacementStrategy.place(earliest, cw.end_time, downlink_dur, mode="earliest")
                if p: dl_places.append(p)
                p = TimePlacementStrategy.place(earliest, cw.end_time, downlink_dur, mode="center")
                if p: dl_places.append(p)

                total_dl = (cw.end_time - earliest).total_seconds()
                if downlink_dur <= total_dl:
                    gs_end = cw.end_time
                    gs_start = gs_end - timedelta(seconds=downlink_dur)
                    if gs_start >= earliest:
                        dl_places.append((gs_start, gs_end))

                for _ in range(max(0, int(random_samples_per_window))):
                    slack_dl = total_dl - downlink_dur
                    if slack_dl <= 1e-9:
                        break
                    offset = rng.random() * slack_dl
                    gs_start = earliest + timedelta(seconds=offset)
                    gs_end = gs_start + timedelta(seconds=downlink_dur)
                    dl_places.append((gs_start, gs_end))

                # 去重
                dl_seen = set()
                for gs_start, gs_end in dl_places:
                    key = (gs_start, gs_end)
                    if key in dl_seen:
                        continue
                    dl_seen.add(key)
                    push_candidate(w, sat_start, sat_end, sat_angles, cw=cw, gs_start=gs_start, gs_end=gs_end)

                    if len(cands) >= max_candidates:
                        break

    return cands


def assignment_sat_intervals(a: Assignment) -> List[Tuple[str, object, object]]:
    res = [(a.satellite_id, a.sat_start_time, a.sat_end_time)]
    if a.gs_start_time is not None and a.gs_end_time is not None:
        res.append((a.satellite_id, a.gs_start_time, a.gs_end_time))
    return res


def assignments_conflict(a: Assignment, b: Assignment) -> bool:
    if a.satellite_id == b.satellite_id:
        for _, s1, e1 in assignment_sat_intervals(a):
            for _, s2, e2 in assignment_sat_intervals(b):
                if _overlap(s1, e1, s2, e2):
                    return True

    if a.ground_station_id and b.ground_station_id and a.ground_station_id == b.ground_station_id:
        if a.gs_start_time and a.gs_end_time and b.gs_start_time and b.gs_end_time:
            if _overlap(a.gs_start_time, a.gs_end_time, b.gs_start_time, b.gs_end_time):
                return True

    return False
