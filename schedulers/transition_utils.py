# -*- coding: utf-8 -*-
"""transition_utils.py
姿态转换时间模型（敏捷/非敏捷）。
/ Attitude transition time model (agile/non-agile).



敏捷卫星： / Agile satellites:
Δg = |Δroll| + |Δpitch| + |Δyaw| （度 / degrees）
Trans(Δg) 分段模型（来自用户给定公式）：
/ Trans(Δg) piecewise model (from user-provided formula):
- Δg <= 10: 11.66
- 10 < Δg <= 30: a1 + Δg/v1, a1=5
- 30 < Δg <= 60: a2 + Δg/v2, a2=10
- 60 < Δg <= 90: a3 + Δg/v3, a3=16
- Δg > 90:  a4 + Δg/v4，其中 a4 为常数（用户给定）
  / where a4 is a constant (user-provided)

非敏捷卫星： / Non-agile satellites:
固定时间 non_agile_transition_s（默认 10 秒），由 main/main_scheduler 配置。
/ Fixed time non_agile_transition_s (default 10 seconds), configured by main/main_scheduler.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple


AGILITY_PROFILES: Dict[str, Tuple[float, float, float, float]] = {
    # profile name -> (v1, v2, v3, v4)  (deg/s)
    "High-Agility": (3.00, 4.00, 5.00, 6.00),
    "Standard-Agility": (1.50, 2.00, 2.50, 3.00),
    "Low-Agility": (0.75, 1.00, 1.25, 1.50),
    "Limited-Agility": (0.50, 0.67, 0.83, 1.00),
}

# 分段常数 / Piecewise constants
A1 = 5.0
A2 = 10.0
A3 = 16.0
A4 = 22.0
C_SMALL = 11.66  # Δg<=10 的常数时间（秒） / Constant time (seconds) for Δg<=10


def _normalize_profile_name(name: str) -> str:
    if not name:
        return "Standard-Agility"
    n = name.strip()
    # allow convenient aliases / 允许方便的别名
    low = n.lower().replace("_", "-")
    if low in {"high", "high-agility", "highagility"}:
        return "High-Agility"
    if low in {"standard", "standard-agility", "standardagility"}:
        return "Standard-Agility"
    if low in {"low", "low-agility", "lowagility"}:
        return "Low-Agility"
    if low in {"limited", "limited-agility", "limitedagility"}:
        return "Limited-Agility"
    # keep original / 保持原样
    return n


def get_profile_velocities(profile_name: str) -> Tuple[float, float, float, float]:
    key = _normalize_profile_name(profile_name)
    return AGILITY_PROFILES.get(key, AGILITY_PROFILES["Standard-Agility"])


def compute_transition_time_agile(delta_g_deg: float, profile_name: str) -> float:
    """敏捷卫星转换时间（秒），输入 delta_g（度）。
       / Agile satellite transition time (seconds), input delta_g (degrees)."""
    dg = float(delta_g_deg)
    if dg <= 10.0:
        return C_SMALL

    v1, v2, v3, v4 = get_profile_velocities(profile_name)

    if dg <= 30.0:
        return A1 + dg / v1
    if dg <= 60.0:
        return A2 + dg / v2
    if dg <= 90.0:
        return A3 + dg / v3

    return A4 + dg / v4


def _get_first_last_from_list(x: Any, want_first: bool) -> Any:
    if not isinstance(x, list) or not x:
        return None
    return x[0] if want_first else x[-1]


def extract_attitude_from_angles(angles: Any, want_first: bool) -> Optional[Dict[str, float]]:
    """从 sat_angles（可能是 list/dict/嵌套结构）提取第一/最后一帧姿态。
       返回 dict: {roll,pitch,yaw}，单位度。提取失败返回 None。
       / Extract the first/last frame attitude from sat_angles (might be list/dict/nested structure).
       Returns dict: {roll, pitch, yaw} in degrees. Returns None if extraction fails.
    """
    if angles is None:
        return None

    # Case 1: list of records / 记录列表
    if isinstance(angles, list):
        rec = _get_first_last_from_list(angles, want_first)
        if rec is None:
            return None
        # record as dict / 字典形式的记录
        if isinstance(rec, dict):
            return _attitude_from_mapping(rec)
        # record as list/tuple length>=3 / 列表/元组形式的记录，长度 >= 3
        if isinstance(rec, (list, tuple)) and len(rec) >= 3:
            try:
                roll, pitch, yaw = float(rec[0]), float(rec[1]), float(rec[2])
                return {"roll": roll, "pitch": pitch, "yaw": yaw}
            except Exception:
                return None
        return None

    # Case 2: dict of arrays / nested
    #         / 数组字典 / 嵌套结构
    if isinstance(angles, dict):
        # common: {'roll':[...], 'pitch':[...], 'yaw':[...]} or {'gamma':[], 'pi':[], 'psi':[]}
        # 常见形式：{'roll':[...], 'pitch':[...], 'yaw':[...]} 或 {'gamma':[], 'pi':[], 'psi':[]}
        for keyset in (("roll", "pitch", "yaw"), ("gamma", "pi", "psi")):
            if all(k in angles for k in keyset):
                try:
                    r_list = angles[keyset[0]]
                    p_list = angles[keyset[1]]
                    y_list = angles[keyset[2]]
                    r = _get_first_last_from_list(r_list, want_first)
                    p = _get_first_last_from_list(p_list, want_first)
                    y = _get_first_last_from_list(y_list, want_first)
                    if r is None or p is None or y is None:
                        return None
                    roll = float(r); pitch = float(p); yaw = float(y)
                    # map gamma->roll, pi->pitch, psi->yaw if using gamma/pi/psi
                    # 如果使用 gamma/pi/psi，则映射 gamma->roll, pi->pitch, psi->yaw
                    return {"roll": roll, "pitch": pitch, "yaw": yaw}
                except Exception:
                    return None

        # nested dict: {'attitude':{'roll':...}}
        # try to find an inner mapping with keys
        # 嵌套字典: {'attitude':{'roll':...}} 尝试寻找带有键值的内部映射
        for v in angles.values():
            if isinstance(v, dict):
                att = _attitude_from_mapping(v)
                if att is not None:
                    return att
        return None

    return None


def _attitude_from_mapping(m: Dict[str, Any]) -> Optional[Dict[str, float]]:
    # accept both roll/pitch/yaw and gamma/pi/psi
    # 接受 roll/pitch/yaw 和 gamma/pi/psi 两种命名
    if all(k in m for k in ("roll", "pitch", "yaw")):
        try:
            return {"roll": float(m["roll"]), "pitch": float(m["pitch"]), "yaw": float(m["yaw"])}
        except Exception:
            return None
    if all(k in m for k in ("gamma", "pi", "psi")):
        try:
            # gamma=roll, pi=pitch, psi=yaw
            return {"roll": float(m["gamma"]), "pitch": float(m["pi"]), "yaw": float(m["psi"])}
        except Exception:
            return None
    return None


def delta_g_between(prev_angles: Any, next_angles: Any) -> Optional[float]:
    """计算 Δg = |Δroll| + |Δpitch| + |Δyaw|（度）。提取失败返回 None。
       / Calculate Δg = |Δroll| + |Δpitch| + |Δyaw| (degrees). Returns None if extraction fails."""
    a1 = extract_attitude_from_angles(prev_angles, want_first=False)
    a2 = extract_attitude_from_angles(next_angles, want_first=True)
    if a1 is None or a2 is None:
        return None
    dr = abs(a2["roll"] - a1["roll"])
    dp = abs(a2["pitch"] - a1["pitch"])
    dy = abs(a2["yaw"] - a1["yaw"])
    return float(dr + dp + dy)