# -*- coding: utf-8 -*-
"""
schedule_output.py
调度结果输出与可视化模块 / Scheduling result output and visualization module

本文件在整个项目中的角色 / Role of this file in the entire project
--------------------------------
1. 将调度结果 Schedule 按统一格式导出到 JSON 文件，便于后续统计分析；
   / Export the scheduling result Schedule to a JSON file in a unified format for subsequent statistical analysis;
2. 使用 matplotlib 绘制简单的 Gantt 图，展示每颗卫星的任务时间安排，
   并输出为 PNG 图片；
   / Use matplotlib to draw a simple Gantt chart showing the task schedule of each satellite, and output it as a PNG image;
3. 同时在 JSON 与图中展示评价指标数据（TP, TCR, BD, TM, RT, RV）。
   / Simultaneously display evaluation metric data (TP, TCR, BD, TM, RT, RV) in the JSON and the chart.

特别说明： / Special Notes:
----------
- 若场景中没有地面站信息（problem.ground_stations 为空），
  则输出的 JSON assignments 中不包含任何地面站相关字段。
  / If there is no ground station information in the scenario (problem.ground_stations is empty), the output JSON assignments will not contain any ground station-related fields.
- 若场景中存在地面站，则所有被调度的任务必须带有地面站安排，
  且 JSON assignments 中包含对应字段。
  / If ground stations exist in the scenario, all scheduled tasks must have ground station arrangements, and the corresponding fields will be included in the JSON assignments.
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

# 设置中文字体，避免图中中文乱码
# / Set Chinese font to avoid garbled Chinese characters in the plot
matplotlib.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "Arial Unicode MS"]
matplotlib.rcParams["axes.unicode_minus"] = False


# ==============================
# 1. JSON 输出 / 1. JSON Output
# ==============================

def save_schedule_to_json(
    schedule: Schedule,
    problem: SchedulingProblem,
    output_path: str | Path,
    metrics: Optional[EvaluationMetrics] = None,
) -> None:
    """
    将调度结果保存为 JSON。
    / Save the scheduling result as JSON.

    若 problem.ground_stations 非空，则 assignments 中包含：
    / If problem.ground_stations is not empty, assignments include:
        task_id, satellite_id,
        ground_station_id, sat_start_time, sat_end_time,
        gs_start_time, gs_end_time, priority

    若 problem.ground_stations 为空，则 assignments 中仅包含：
    / If problem.ground_stations is empty, assignments only include:
        task_id, satellite_id, sat_start_time, sat_end_time, priority

    JSON 顶层结构示例： / JSON top-level structure example:

    {
      "scenario_id": "Scenario_S1",
      "start_time": "...",
      "end_time": "...",
      "assignments": [...],
      "unassigned_tasks": [...],
      "metrics": {
        "TP": ...,
        "TCR": ...,
        "BD": ...,
        "TM": ...,
        "RT": ...,
        "RV": ...
      }
    }
    """
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
            # 有地面站的场景：必须输出地面站字段
            # / Scenarios with ground stations: ground station fields must be outputted
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
            # 无地面站：不输出任何地面站相关字段
            # / No ground stations: do not output any ground station-related fields
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

    # 指标部分 / Metrics section
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
# 2. Gantt 可视化 / 2. Gantt Visualization
# ==============================


def plot_schedule_gantt(
    schedule: Schedule,
    problem: SchedulingProblem,
    output_path: str | Path,
    metrics: Optional[EvaluationMetrics] = None,
) -> None:
    """
    绘制简单 Gantt 图： / Draw a simple Gantt chart:
    - 横轴为时间（相对场景开始的分钟数）； / X-axis is time (minutes relative to the start of the scenario);
    - 纵轴为卫星； / Y-axis is satellite;
    - 每个任务在对应卫星行上画出“观测段”， / Draw the "observation segment" for each task on the corresponding satellite row,
      标签中附带任务 ID 以及地面站 ID（如有）。 / with the task ID and ground station ID (if any) attached to the label.

    同时在图右上角展示主要评价指标（若提供）。
    / Simultaneously display the main evaluation metrics in the upper right corner of the chart (if provided).
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not schedule.assignments:
        # 没有任何任务被分配，不画图
        # / No tasks were assigned, do not draw the chart
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

        # 标签：任务 ID + 可选地面站
        # / Label: Task ID + optional ground station
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
    ax.set_xlabel("时间（分钟，相对场景开始） / Time (minutes, relative to scenario start)")
    ax.set_ylabel("卫星 / Satellite")
    ax.set_title(f"调度结果 Gantt 图 / Scheduling Result Gantt Chart - {problem.scenario_id}")
    ax.grid(True, axis="x", linestyle="--", linewidth=0.5)

    # 在图中展示指标
    # / Display metrics in the chart
    if metrics is not None:
        lines = [
            f"TP  (任务收益 / Task Profit): {metrics.task_profit:.3f}",
            f"TCR (完成率 / Completion Rate)  : {metrics.task_completion_rate:.3f}",
            f"BD  (均衡度 / Balance Degree)  : {metrics.balance_degree:.3f}",
            f"TM  (时效性 / Timeliness)  : {metrics.timeliness_metric:.3f}",
            f"RT  (运行时间 / Runtime): {metrics.runtime_efficiency:.3f} s",
        ]
        if metrics.robustness_variance is not None:
            lines.append(f"RV  (鲁棒性 / Robustness): {metrics.robustness_variance:.3f}")

        text = "\n".join(lines)
        # 坐标使用轴坐标 (0~1)，右上角
        # / Coordinates use axis coordinates (0~1), upper right corner
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