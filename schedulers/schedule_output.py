# -*- coding: utf-8 -*-
"""
schedule_output.py

Main functionality:
This module exports scheduling results to JSON and generates a Gantt chart
visualization for scheduled assignments, including optional evaluation metrics.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import json
from datetime import datetime

import matplotlib.pyplot as plt
import matplotlib

from .scenario_loader import SchedulingProblem
from .constraint_model import Schedule
from .evaluation_metrics import EvaluationMetrics

# Set fonts to improve cross-platform text rendering in plots
matplotlib.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "Arial Unicode MS"]
matplotlib.rcParams["axes.unicode_minus"] = False


# ==============================
# 1. JSON output
# ==============================

def save_schedule_to_json(
    schedule: Schedule,
    problem: SchedulingProblem,
    output_path: str | Path,
    metrics: Optional[EvaluationMetrics] = None,
) -> None:

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    has_gs = len(problem.ground_stations) > 0

    assigned_task_ids = {a.task_id for a in schedule.assignments}
    all_task_ids = set(problem.tasks.keys())
    unassigned_tasks = sorted(list(all_task_ids - assigned_task_ids))

    result = {
        "scenario_id": problem.scenario_id,
        "start_time": problem.start_time.isoformat(),
        "end_time": problem.end_time.isoformat(),
        "assignments": [],
        "unassigned_tasks": unassigned_tasks,
    }

    for a in schedule.assignments:
        task = problem.tasks[a.task_id]

        if has_gs:
            # For scenarios with ground stations, ground station fields must be included
            result["assignments"].append(
                {
                    "task_id": a.task_id,
                    "satellite_id": a.satellite_id,
                    "ground_station_id": a.ground_station_id,
                    "sat_start_time": a.sat_start_time.isoformat(),
                    "sat_end_time": a.sat_end_time.isoformat(),
                    "sensor_id": getattr(a, "sensor_id", ""),
                    "orbit_number": int(getattr(a, "orbit_number", 0) or 0),
                    "data_volume_GB": float(getattr(a, "data_volume_GB", 0.0) or 0.0),
                    "power_cost_W": float(getattr(a, "power_cost_W", 0.0) or 0.0),
                    "sat_angles": a.sat_angles,
                    "gs_start_time": a.gs_start_time.isoformat() if a.gs_start_time else None,
                    "gs_end_time": a.gs_end_time.isoformat() if a.gs_end_time else None,
                    "priority": task.priority,
                }
            )
        else:
            # For scenarios without ground stations, exclude all ground station-related fields
            result["assignments"].append(
                {
                    "task_id": a.task_id,
                    "satellite_id": a.satellite_id,
                    "sat_start_time": a.sat_start_time.isoformat(),
                    "sat_end_time": a.sat_end_time.isoformat(),
                    "sensor_id": getattr(a, "sensor_id", ""),
                    "orbit_number": int(getattr(a, "orbit_number", 0) or 0),
                    "data_volume_GB": float(getattr(a, "data_volume_GB", 0.0) or 0.0),
                    "power_cost_W": float(getattr(a, "power_cost_W", 0.0) or 0.0),
                    "sat_angles": a.sat_angles,
                    "priority": task.priority,
                }
            )

    # Metrics section
    if metrics is not None:
        result["metrics"] = {
            "TP": metrics.task_profit,
            "TCR": metrics.task_completion_rate,
            "BD": metrics.balance_degree,
            "TM": metrics.timeliness_metric,
            "RT": metrics.runtime_efficiency,
            "RV": metrics.robustness_variance,
        }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)


# ==============================
# 2. Gantt visualization
# ==============================

def plot_schedule_gantt(
    schedule: Schedule,
    problem: SchedulingProblem,
    output_path: str | Path,
    metrics: Optional[EvaluationMetrics] = None,
) -> None:

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not schedule.assignments:
        # No assigned tasks, so skip plotting
        return

    has_gs = len(problem.ground_stations) > 0

    sat_ids = sorted(problem.satellites.keys())
    sat_index = {sid: idx for idx, sid in enumerate(sat_ids)}

    fig, ax = plt.subplots(figsize=(12, 6))

    for a in schedule.assignments:
        idx = sat_index[a.satellite_id]
        start_min = (a.sat_start_time - problem.start_time).total_seconds() / 60.0
        end_min = (a.sat_end_time - problem.start_time).total_seconds() / 60.0
        width = end_min - start_min

        ax.barh(
            y=idx,
            width=width,
            left=start_min,
            height=0.4,
            align="center",
        )

        # Label: task ID plus optional ground station
        if has_gs and a.ground_station_id:
            label = f"{a.task_id}->{a.ground_station_id}"
        else:
            label = a.task_id

        ax.text(
            start_min + width / 2,
            idx,
            label,
            va="center",
            ha="center",
            fontsize=8,
        )

    ax.set_yticks(list(range(len(sat_ids))))
    ax.set_yticklabels(sat_ids)
    ax.set_xlabel("Time (minutes, relative to scenario start)")
    ax.set_ylabel("Satellite")
    ax.set_title(f"Scheduling Result Gantt Chart - {problem.scenario_id}")
    ax.grid(True, axis="x", linestyle="--", linewidth=0.5)

    # Display metrics in the chart
    if metrics is not None:
        lines = [
            f"TP  (Task Profit): {metrics.task_profit:.3f}",
            f"TCR (Completion Rate): {metrics.task_completion_rate:.3f}",
            f"BD  (Balance Degree): {metrics.balance_degree:.3f}",
            f"TM  (Timeliness): {metrics.timeliness_metric:.3f}",
            f"RT  (Runtime): {metrics.runtime_efficiency:.3f} s",
        ]
        if metrics.robustness_variance is not None:
            lines.append(f"RV  (Robustness): {metrics.robustness_variance:.3f}")

        text = "\n".join(lines)
        # Use axis coordinates (0~1), top-right corner
        ax.text(
            0.99,
            0.99,
            text,
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=9,
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.7),
        )

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)