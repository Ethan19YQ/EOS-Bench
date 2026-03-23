# -*- coding: utf-8 -*-
"""
transition_utils.py

Main functionality:
This module provides utility functions for satellite attitude transition modeling.
It defines agility profiles, computes agile-satellite transition time from angular
change, extracts attitude values from multiple angle-data formats, and calculates
the total angular change between two task attitudes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple


AGILITY_PROFILES: Dict[str, Tuple[float, float, float, float]] = {
    # profile name -> (v1, v2, v3, v4) in deg/s
    "High-Agility": (3.00, 4.00, 5.00, 6.00),
    "Standard-Agility": (1.50, 2.00, 2.50, 3.00),
    "Low-Agility": (0.75, 1.00, 1.25, 1.50),
    "Limited-Agility": (0.50, 0.67, 0.83, 1.00),
}

# Piecewise constants
A1 = 5.0
A2 = 10.0
A3 = 16.0
A4 = 22.0
C_SMALL = 11.66  # Constant time in seconds for Δg <= 10


def _normalize_profile_name(name: str) -> str:
    if not name:
        return "Standard-Agility"
    n = name.strip()
    # Allow convenient aliases
    low = n.lower().replace("_", "-")
    if low in {"high", "high-agility", "highagility"}:
        return "High-Agility"
    if low in {"standard", "standard-agility", "standardagility"}:
        return "Standard-Agility"
    if low in {"low", "low-agility", "lowagility"}:
        return "Low-Agility"
    if low in {"limited", "limited-agility", "limitedagility"}:
        return "Limited-Agility"
    # Keep the original name
    return n


def get_profile_velocities(profile_name: str) -> Tuple[float, float, float, float]:
    key = _normalize_profile_name(profile_name)
    return AGILITY_PROFILES.get(key, AGILITY_PROFILES["Standard-Agility"])


def compute_transition_time_agile(delta_g_deg: float, profile_name: str) -> float:
    """Agile satellite transition time in seconds, with delta_g in degrees."""
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
    """Extract the first or last frame attitude from sat_angles.

    Returns a dictionary {roll, pitch, yaw} in degrees.
    Returns None if extraction fails.
    """
    if angles is None:
        return None

    # Case 1: list of records
    if isinstance(angles, list):
        rec = _get_first_last_from_list(angles, want_first)
        if rec is None:
            return None
        # Record as dict
        if isinstance(rec, dict):
            return _attitude_from_mapping(rec)
        # Record as list/tuple with length >= 3
        if isinstance(rec, (list, tuple)) and len(rec) >= 3:
            try:
                roll, pitch, yaw = float(rec[0]), float(rec[1]), float(rec[2])
                return {"roll": roll, "pitch": pitch, "yaw": yaw}
            except Exception:
                return None
        return None

    # Case 2: dict of arrays or nested structure
    if isinstance(angles, dict):
        # Common cases:
        # {'roll':[...], 'pitch':[...], 'yaw':[...]}
        # {'gamma':[], 'pi':[], 'psi':[]}
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
                    roll = float(r)
                    pitch = float(p)
                    yaw = float(y)
                    # Map gamma -> roll, pi -> pitch, psi -> yaw when using gamma/pi/psi
                    return {"roll": roll, "pitch": pitch, "yaw": yaw}
                except Exception:
                    return None

        # Nested dict case, such as {'attitude': {'roll': ...}}
        # Try to find an inner mapping with valid keys
        for v in angles.values():
            if isinstance(v, dict):
                att = _attitude_from_mapping(v)
                if att is not None:
                    return att
        return None

    return None


def _attitude_from_mapping(m: Dict[str, Any]) -> Optional[Dict[str, float]]:
    # Accept both roll/pitch/yaw and gamma/pi/psi
    if all(k in m for k in ("roll", "pitch", "yaw")):
        try:
            return {"roll": float(m["roll"]), "pitch": float(m["pitch"]), "yaw": float(m["yaw"])}
        except Exception:
            return None
    if all(k in m for k in ("gamma", "pi", "psi")):
        try:
            # gamma = roll, pi = pitch, psi = yaw
            return {"roll": float(m["gamma"]), "pitch": float(m["pi"]), "yaw": float(m["psi"])}
        except Exception:
            return None
    return None


def delta_g_between(prev_angles: Any, next_angles: Any) -> Optional[float]:
    """Calculate Δg = |Δroll| + |Δpitch| + |Δyaw| in degrees.

    Returns None if extraction fails.
    """
    a1 = extract_attitude_from_angles(prev_angles, want_first=False)
    a2 = extract_attitude_from_angles(next_angles, want_first=True)
    if a1 is None or a2 is None:
        return None
    dr = abs(a2["roll"] - a1["roll"])
    dp = abs(a2["pitch"] - a1["pitch"])
    dy = abs(a2["yaw"] - a1["yaw"])
    return float(dr + dp + dy)