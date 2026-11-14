"""Evaluate a saved plan at several cloud levels without running a solver."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from schedulers.constraint_model import Assignment, Schedule
from schedulers.scenario_loader import load_scheduling_problem_from_json, _parse_iso_time
from schedulers.weather_mask import evaluate_cloud_fractions, WEATHER_MASK_SCHEMA


def load_saved_schedule(path, problem):
    """Match saved assignments to the access windows of their source instance."""
    data = json.loads(Path(path).read_text(encoding="utf8"))
    if data.get("scenario_id") != problem.scenario_id:
        raise ValueError("The schedule and scenario have different scenario IDs")
    for field, expected in (("start_time", problem.start_time), ("end_time", problem.end_time)):
        if field in data and _parse_iso_time(data[field]) != expected:
            raise ValueError("The schedule and scenario have different planning horizons")
    old_mask = data.get("weather_mask", {})
    if old_mask and old_mask.get("schema") != WEATHER_MASK_SCHEMA:
        raise ValueError("Use a plan constructed from the full opportunity set")
    schedule = Schedule(metadata=data.get("solver_metadata", {}))
    for row in data["assignments"]:
        task = problem.tasks.get(row["task_id"])
        if task is None:
            raise ValueError("Scheduled task is absent from the source scenario")
        start = _parse_iso_time(row["sat_start_time"])
        end = _parse_iso_time(row["sat_end_time"])
        # Older exports omit window IDs. Accept them only with a unique match.
        candidates = [w for w in task.windows
            if w.satellite_id == row["satellite_id"]
            and w.start_time <= start < end <= w.end_time
            and ("sat_window_id" not in row or w.window_id == row["sat_window_id"])
            and ("sensor_id" not in row or w.sensor_id == row["sensor_id"])
            and ("orbit_number" not in row or w.orbit_number == row["orbit_number"])]
        if len(candidates) != 1:
            raise ValueError(f"Task {task.id}: expected one matching source window, found {len(candidates)}")
        w = candidates[0]
        schedule.assignments.append(Assignment(
            task_id=task.id, satellite_id=w.satellite_id,
            sat_start_time=start, sat_end_time=end, sat_window_id=w.window_id,
            sensor_id=w.sensor_id, orbit_number=w.orbit_number,
            data_volume_GB=float(row.get("data_volume_GB", 0)),
            power_cost_W=float(row.get("power_cost_W", 0)),
            sat_angles=row.get("sat_angles"), ground_station_id=row.get("ground_station_id"),
            gs_start_time=_parse_iso_time(row["gs_start_time"]) if row.get("gs_start_time") else None,
            gs_end_time=_parse_iso_time(row["gs_end_time"]) if row.get("gs_end_time") else None,
            gs_window_id=row.get("gs_window_id"),
        ))
    if len(schedule.assignments) != len(schedule.assigned_task_ids):
        raise ValueError("A saved plan must assign each task at most once")
    return schedule


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", type=Path, required=True, help="Original scenario JSON")
    parser.add_argument("--schedule", type=Path, required=True, help="Saved schedule JSON")
    parser.add_argument("--fractions", type=float, nargs="+", default=[0, .25, .5, .75])
    parser.add_argument("--weather_seed", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True, help="Separate cloud-results JSON")
    parser.add_argument("--overwrite", action="store_true", help="Replace an existing cloud-results file")
    args = parser.parse_args()
    if args.output.resolve() in {args.scenario.resolve(), args.schedule.resolve()}:
        parser.error("The output must be separate from the source scenario and schedule")
    if args.output.exists() and not args.overwrite:
        parser.error("The output already exists; choose another path or use --overwrite")
    problem = load_scheduling_problem_from_json(args.scenario)
    schedule = load_saved_schedule(args.schedule, problem)
    report = evaluate_cloud_fractions(problem, schedule, args.fractions, args.weather_seed)
    report.update({"scenario_id": problem.scenario_id,
                   "source_scenario": str(args.scenario.resolve()),
                   "source_schedule": str(args.schedule.resolve())})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w" if args.overwrite else "x", encoding="utf8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
    for result in report["results"]:
        print(f"cloud={result['blocked_fraction']:.0%} "
              f"TP={result['realised_TP']:.6g} TCR={result['realised_TCR']:.6g}")
    print(f"Cloud results saved to: {args.output.resolve()}")


if __name__ == "__main__":
    main()
