"""Deterministic cloud masks for evaluating a completed observation plan."""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass

WEATHER_MASK_SCHEMA = "deterministic_cloud_outcomes_v2"
WEATHER_SEMANTICS = "post_schedule_loss_no_replanning"


@dataclass(frozen=True)
class WeatherMaskConfig:
    enabled: bool = False
    blocked_fraction: float = 0.2
    seed: int = 0

    def __post_init__(self):
        if not isinstance(self.enabled, bool):
            raise ValueError("weather enabled must be a boolean")
        if isinstance(self.blocked_fraction, bool) or not isinstance(self.blocked_fraction, (int, float)):
            raise ValueError("blocked_fraction must be a number in [0, 1]")
        if not math.isfinite(self.blocked_fraction) or not 0 <= self.blocked_fraction <= 1:
            raise ValueError("blocked_fraction must be finite and in [0, 1]")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise ValueError("weather seed must be an integer")
        object.__setattr__(self, "blocked_fraction", float(self.blocked_fraction))

    @property
    def output_tag(self):
        return f"cloud_post_f{float(self.blocked_fraction)}_s{self.seed}"


def resolve_weather_config(config=None):
    if config is None:
        return WeatherMaskConfig()
    if isinstance(config, WeatherMaskConfig):
        return config
    if isinstance(config, dict):
        return WeatherMaskConfig(**config)
    raise TypeError("weather_config must be a dict, WeatherMaskConfig or None")


def _rank_windows(problem, seed):
    """Return a source fingerprint and one ordering shared by all cloud levels."""
    inventory = []
    for tid in sorted(problem.tasks):
        for w in problem.tasks[tid].windows:
            inventory.append((w.window_id, tid, w.satellite_id, w.sensor_id,
                w.orbit_number, w.start_time.isoformat(), w.end_time.isoformat()))
    inventory.sort()
    source = json.dumps({"scenario_id": problem.scenario_id,
        "start": problem.start_time.isoformat(), "end": problem.end_time.isoformat(),
        "windows": inventory}, ensure_ascii=True, separators=(",", ":"))
    fingerprint = hashlib.sha256(source.encode("utf8")).hexdigest()
    ranked = sorted(inventory, key=lambda row: (
        hashlib.sha256(f"{fingerprint}|{seed}|{row[0]}".encode("utf8")).hexdigest(), row[0]))
    return fingerprint, ranked


def _mask_metadata(problem, cfg, fingerprint, ranked):
    total = len(ranked)
    n_blocked = min(total, int(math.floor(float(cfg.blocked_fraction) * total + 0.5)))
    blocked = {row[0] for row in ranked[:n_blocked]}
    blocked_ids = sorted(blocked)
    mask_digest = hashlib.sha256(json.dumps({
        "schema": WEATHER_MASK_SCHEMA, "source": fingerprint,
        "config": asdict(cfg), "blocked": blocked_ids},
        sort_keys=True, separators=(",", ":")).encode("utf8")).hexdigest()
    return {
        "schema": WEATHER_MASK_SCHEMA, "semantics": WEATHER_SEMANTICS,
        "config": asdict(cfg), "source_window_fingerprint": fingerprint,
        "mask_sha256": mask_digest, "blocked_window_ids": blocked_ids,
        "original_windows": total, "blocked_windows": n_blocked,
        "unblocked_windows": total - n_blocked,
        "actual_blocked_fraction": n_blocked / total if total else 0.0,
        "affected_tasks": sum(any(w.window_id in blocked for w in t.windows)
                              for t in problem.tasks.values()),
        "total_tasks": len(problem.tasks),
        "tasks_without_windows": sum(not t.windows for t in problem.tasks.values()),
        "tasks_without_clear_windows": sum(
            not any(w.window_id not in blocked for w in t.windows)
            for t in problem.tasks.values()),
    }


def apply_weather_mask(problem, config=None):
    """Attach an outcome mask while preserving every planning window."""
    cfg = resolve_weather_config(config)
    if not cfg.enabled:
        return problem
    if problem.weather_metadata:
        raise ValueError("Weather mask already applied; reload the original instance")
    fingerprint, ranked = _rank_windows(problem, cfg.seed)
    problem.weather_metadata = _mask_metadata(problem, cfg, fingerprint, ranked)
    return problem


def evaluate_cloud_outcomes(problem, schedule, metadata=None):
    """Report realised TP/TCR without changing assignments or resource usage."""
    mask = problem.weather_metadata if metadata is None else metadata
    if mask and mask.get("schema") != WEATHER_MASK_SCHEMA:
        raise ValueError("Reload the original scenario to evaluate post-scheduling cloud outcomes")
    blocked = set(mask.get("blocked_window_ids", []))
    windows = {w.window_id: w for task in problem.tasks.values() for w in task.windows}
    successful, planned, failed = set(), set(), []
    for a in schedule.assignments:
        w = windows.get(a.sat_window_id)
        if (a.task_id not in problem.tasks or w is None
                or w.mission_id != a.task_id or w.satellite_id != a.satellite_id
                or not w.start_time <= a.sat_start_time < a.sat_end_time <= w.end_time):
            raise ValueError("Scheduled observation has no matching source window")
        planned.add(a.task_id)
        if a.sat_window_id in blocked:
            failed.append({"task_id": a.task_id, "satellite_id": a.satellite_id,
                           "window_id": a.sat_window_id})
        else:
            successful.add(a.task_id)
    total = len(problem.tasks)
    return {
        "planned_TP": sum(float(problem.tasks[t].priority) for t in sorted(planned)),
        "planned_TCR": len(planned) / total if total else 0.0,
        "realised_TP": sum(float(problem.tasks[t].priority) for t in sorted(successful)),
        "realised_TCR": len(successful) / total if total else 0.0,
        "successful_tasks": len(successful),
        "successful_task_ids": sorted(successful),
        "failed_scheduled_observations": failed,
    }


def evaluate_cloud_fractions(problem, schedule, fractions=(0, .25, .5, .75), seed=0):
    """Evaluate the same plan at several fractions with one shared ordering."""
    if (problem.weather_metadata
            and problem.weather_metadata.get("schema") != WEATHER_MASK_SCHEMA):
        raise ValueError("Reload the original scenario before evaluating cloud fractions")
    configs = [WeatherMaskConfig(True, fraction, seed) for fraction in fractions]
    if not configs:
        raise ValueError("Provide at least one cloud fraction")
    fingerprint, ranked = _rank_windows(problem, seed)
    results = []
    for cfg in configs:
        mask = _mask_metadata(problem, cfg, fingerprint, ranked)
        results.append({
            "blocked_fraction": cfg.blocked_fraction,
            "actual_blocked_fraction": mask["actual_blocked_fraction"],
            "blocked_window_count": mask["blocked_windows"],
            "blocked_window_ids": mask["blocked_window_ids"],
            "mask_sha256": mask["mask_sha256"],
            **evaluate_cloud_outcomes(problem, schedule, mask),
        })
    return {"schema": WEATHER_MASK_SCHEMA, "semantics": WEATHER_SEMANTICS,
            "source_window_fingerprint": fingerprint, "cloud_seed": seed,
            "total_window_count": len(ranked), "results": results}
