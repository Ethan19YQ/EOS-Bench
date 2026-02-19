# -*- coding: utf-8 -*-
"""main.py
说明 / Description
----
- class_id=1: MIP (混合整数规划 / Mixed Integer Programming)
- class_id=2: Heuristic (completion/profit/balance/timeliness) (启发式算法 / Heuristic algorithms)
- class_id=3: Meta-heuristics (sa/ga/aco) (元启发式算法 / Meta-heuristic algorithms)
- class_id=4: PPO（强化学习，分 Train 与 Test 两阶段 / Reinforcement Learning, divided into Train and Test phases）
"""

from __future__ import annotations

import os
import time
import re
import hashlib
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Tuple, Any, Optional

from schedulers.scenario_loader import load_scheduling_problem_from_json
from schedulers.constraint_model import ConstraintModel, Schedule
from schedulers.engine import SchedulingEngine
from schedulers.schedule_output import save_schedule_to_json, plot_schedule_gantt
from schedulers.evaluation_metrics import compute_evaluation_metrics, EvaluationMetrics

from algorithms.objectives import ObjectiveWeights
from algorithms.factory import create_algorithm

BASE_DIR = Path(__file__).resolve().parent
SCENARIO_DIR = BASE_DIR / "output"
SCHEDULE_DIR = BASE_DIR / "output" / "schedules"
MODELS_DIR = BASE_DIR / "output" / "models"


def _get_torch_device_prefer_cuda() -> str:
    """尽量返回可用的 device 字符串（不强依赖 torch）。 / Try to return an available device string (without strictly depending on torch)."""
    try:
        import torch  # type: ignore
        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def _objective_desc(w: ObjectiveWeights | None, algo_name: str, class_id: int) -> str:
    """把优化目标写成可读字符串，用于日志输出。 / Format the optimization objective into a readable string for logging."""
    if class_id in (1, 3, 4) and w is not None:
        return (
            f"profit={w.w_profit:g},completion={w.w_completion:g},"
            f"timeliness={w.w_timeliness:g},balance={w.w_balance:g}"
        )
    if class_id == 2:
        return f"implicit({algo_name})"
    return "N/A"


def _parse_weights_from_model_name(name: str) -> ObjectiveWeights | None:
    """从模型文件名解析目标权重（例如 _p1_c0_t0_b0）。 / Parse objective weights from the model filename (e.g., _p1_c0_t0_b0)."""
    m = re.search(r"_p([0-9.]+)_c([0-9.]+)_t([0-9.]+)_b([0-9.]+)", name)
    if not m:
        return None
    try:
        return ObjectiveWeights(float(m.group(1)), float(m.group(2)), float(m.group(3)), float(m.group(4)))
    except Exception:
        return None


def _sat_count_from_scenario_name(scenario_name: str) -> int | None:
    """从场景文件名中解析卫星数量。解析不到返回 None。 / Parse the number of satellites from the scenario filename. Return None if unparseable."""
    m = re.search(r"Sats(\d+)", scenario_name)
    if not m:
        return None
    try:
        return int(m.group(1))
    except Exception:
        return None


def _stable_seed(text: str) -> int:
    """生成稳定的 31-bit seed（同一 text 恒定）。 / Generate a stable 31-bit seed (constant for the same text)."""
    h = hashlib.md5(text.encode("utf-8"), usedforsecurity=False).hexdigest()
    return int(h[:8], 16) & 0x7FFFFFFF


def _list_scenario_jsons(output_dir: Path) -> list[Path]:
    """递归扫描 output_dir 下所有场景 json，跳过 schedules 目录。 / Recursively scan all scenario JSONs under output_dir, skipping the schedules directory."""
    candidates: list[Path] = []
    for p in output_dir.rglob("Scenario_*.json"):
        if not p.is_file():
            continue
        try:
            rel = p.relative_to(output_dir)
        except Exception:
            rel = p
        if "schedules" in rel.parts:
            continue
        candidates.append(p)

    uniq = sorted({c.resolve() for c in candidates}, key=lambda x: str(x))
    return [Path(p) for p in uniq]


def _runlog_columns() -> list[tuple[str, str, str, int]]:
    return [
        ("Scenario", "scenario_name", "<", 26),
        ("Algo", "algo_name", "<", 16),
        ("Class", "class_id", ">", 5),
        ("Objective", "objective", "<", 44),
        ("TP", "TP", ">", 12),
        ("TCR", "TCR", ">", 12),
        ("BD", "BD", ">", 12),
        ("TM", "TM", ">", 12),
        ("RT(s)", "RT", ">", 12),
        ("Gap", "GAP", ">", 10),
        ("Plan", "plan", "<", 42),
        ("Note", "note", "<", 60),
    ]


def _fmt_cell(val: object, width: int, key: str = "") -> str:
    if val is None:
        s = ""
    else:
        if key == "class_id":
            try:
                s = str(int(val))
            except Exception:
                s = str(val)
        elif isinstance(val, bool):
            s = str(val)
        elif isinstance(val, int):
            s = str(val)
        elif isinstance(val, float):
            s = f"{val:.3f}"
        else:
            s = str(val)

    if len(s) > width:
        if width <= 3:
            s = s[:width]
        else:
            s = s[: width - 3] + "..."
    return s


def _format_row(row: dict) -> str:
    cols = _runlog_columns()
    parts: list[str] = []
    for _, key, align, width in cols:
        cell = _fmt_cell(row.get(key, ""), width, key)
        parts.append(f" {cell:{align}{width}} ")
    return "|" + "|".join(parts) + "|"


def _format_header_line() -> tuple[str, str]:
    cols = _runlog_columns()
    header_cells = [f" {name:{align}{width}} " for (name, _, align, width) in cols]
    header = "|" + "|".join(header_cells) + "|"
    sep = "+" + "+".join(["-" * (width + 2) for (_, _, _, width) in cols]) + "+"
    return header, sep


def _append_runlog_row(log_path: Path, row: dict) -> None:
    scenario_file = str(row.get("scenario_file", ""))
    line = _format_row(row)
    if scenario_file:
        line = f"{line}  scenario_json={scenario_file}"
    with log_path.open("a", encoding="utf-8") as f:
        f.write(line + "\n")
        f.flush()


def _train_one_process(
        run_id: int,
        device: str,
        model_path: Path,
        episodes: int,
        rollout_steps: int,
        max_actions: int,
        placement_mode: str,
        downlink_ratio: float,
        unassigned_penalty: float,
        objective_weights: ObjectiveWeights,
        reward_scale: float,
        seed: int,
        non_agile_transition_s: float,
        resample_every_episodes: int,
        save_every_episodes: int,
        resume_if_exists: bool,
        cuda_visible_devices: Optional[str] = None,
) -> str:
    """RL 单次训练（单进程，避免 GPU Pickle 问题）。 / Single RL training (single process, to avoid GPU Pickle issues)."""
    if cuda_visible_devices is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(cuda_visible_devices)

    dev = device
    if device.startswith("cuda") and ":" not in device:
        dev = "cuda:0"

    from algorithms.ppo.learning import PPOLearningScheduler, PPORunConfig

    cfg = PPORunConfig(
        episodes=int(episodes),
        resample_every_episodes=int(resample_every_episodes),
        save_every_episodes=int(save_every_episodes),
        resume_if_exists=bool(resume_if_exists),
        rollout_steps=int(rollout_steps),
        max_actions=int(max_actions),
        placement_mode=str(placement_mode),
        unassigned_penalty=float(unassigned_penalty),
        downlink_duration_ratio=float(downlink_ratio),
        device=str(dev),
        seed=int(seed),
        objective_weights=objective_weights,
        reward_scale=float(reward_scale),
        non_agile_transition_s=float(non_agile_transition_s),
    )

    learner = PPOLearningScheduler(model_path=str(model_path), run_cfg=cfg)
    mpth = learner.train(BASE_DIR)
    return str(mpth)

def run_single_scheduling(
        problem_json: Path,
        class_id: int,
        algo_name: str,
        placement_mode: str = "earliest",
        unassigned_penalty: float = 1000.0,
        downlink_duration_ratio: float = 1.0,
        agility_profile: str = "Standard-Agility",
        non_agile_transition_s: float = 10.0,
        objective_weights: ObjectiveWeights = ObjectiveWeights(1.0, 0.0, 0.0, 0.0),
        cfg_overrides: dict | None = None,
        schedule_dir: Path | None = None,
        rl_model_path: str = "",
        rl_device: str = "cpu",
        rl_max_actions: int = 256,
        rl_unassigned_penalty: float = 1.0,
) -> Tuple[EvaluationMetrics, Path, Path]:
    """单次运行评测 + 输出结果文件（JSON + Gantt + 指标）。 / Single evaluation run + output result files (JSON + Gantt + metrics)."""

    problem = load_scheduling_problem_from_json(problem_json)

    cm = ConstraintModel(
        problem=problem,
        placement_mode=placement_mode,
        unassigned_penalty=unassigned_penalty,
        downlink_duration_ratio=downlink_duration_ratio,
        agility_profile=agility_profile,
        non_agile_transition_s=non_agile_transition_s,
    )

    algo_name_norm = (algo_name or "").lower().strip()

    if class_id in (1, 2, 3):
        cfg_overrides = dict(cfg_overrides or {})
        cfg_overrides.setdefault("agility_profile", agility_profile)
        cfg_overrides.setdefault("non_agile_transition_s", non_agile_transition_s)

        algorithm = create_algorithm(
            algo_name=algo_name_norm,
            objective_weights=objective_weights,
            cfg_overrides=cfg_overrides or {},
        )
        engine = SchedulingEngine(problem=problem, constraint_model=cm, algorithm=algorithm)
        t0 = time.time()
        best_schedule: Schedule = engine.run()
        t1 = time.time()

    elif class_id == 4:
        if algo_name_norm not in ("ppo",):
            raise ValueError("class_id=4 仅支持 algo_name='ppo' / class_id=4 only supports algo_name='ppo'")
        from algorithms.ppo.learning import PPOLearningScheduler, PPORunConfig

        rl_algo = PPOLearningScheduler(
            model_path=rl_model_path if rl_model_path else None,
            run_cfg=PPORunConfig(
                max_actions=rl_max_actions,
                placement_mode=placement_mode,
                unassigned_penalty=rl_unassigned_penalty,
                downlink_duration_ratio=downlink_duration_ratio,
                device=rl_device,
                objective_weights=objective_weights,
                agility_profile=agility_profile,
                non_agile_transition_s=non_agile_transition_s,
            ),
        )
        t0 = time.time()
        best_schedule = rl_algo.search(
            problem=problem,
            constraint_model=cm,
            initial_schedule=cm.build_initial_schedule(),
            base_dir=BASE_DIR,
        )
        t1 = time.time()

    else:
        raise ValueError("class_id 必须为 1~4 / class_id must be between 1 and 4")

    runtime = t1 - t0

    metrics = compute_evaluation_metrics(
        problem=problem,
        schedule=best_schedule,
        runtime_seconds=runtime,
        robustness_tp_samples=None,
        mip_gap=(best_schedule.metadata.get("mip_gap") if hasattr(best_schedule, "metadata") else None),
    )

    schedule_dir = schedule_dir or SCHEDULE_DIR
    schedule_dir.mkdir(parents=True, exist_ok=True)

    if class_id in (1, 3):
        ow = objective_weights
        obj_tag = f"p{ow.w_profit:g}_c{ow.w_completion:g}_t{ow.w_timeliness:g}_b{ow.w_balance:g}"
        output_basename = f"scheduler_{problem_json.stem}_c{class_id}_{algo_name_norm}_{obj_tag}"
    elif class_id == 4:
        model_tag = Path(rl_model_path).stem if rl_model_path else "unknown"
        output_basename = f"scheduler_{problem_json.stem}_c{class_id}_{algo_name_norm}_{model_tag}"
    else:
        obj_tag = "implicit"
        output_basename = f"scheduler_{problem_json.stem}_c{class_id}_{algo_name_norm}_{obj_tag}"

    schedule_json_path = schedule_dir / f"{output_basename}.json"
    gantt_png_path = schedule_dir / f"{output_basename}.png"

    save_schedule_to_json(best_schedule, problem, schedule_json_path, metrics)
    plot_schedule_gantt(best_schedule, problem, gantt_png_path, metrics)

    gap_str = "" if metrics.mip_gap is None else f" GAP={metrics.mip_gap:.6g}"
    print(
        f"[INFO] algo={algo_name_norm} class={class_id} "
        f"RT={runtime:.3f}s TP={metrics.task_profit:.3f} "
        f"TCR={metrics.task_completion_rate:.3f} "
        f"BD={metrics.balance_degree:.3f} TM={metrics.timeliness_metric:.3f}{gap_str}"
    )
    print(f"[INFO] saved: {schedule_json_path.name}")

    return metrics, schedule_json_path, gantt_png_path


def _worker_run_single(job: dict[str, Any]) -> dict[str, Any]:
    """子进程工作函数：运行一次算法并返回日志行字典。 / Subprocess worker function: run the algorithm once and return a log row dictionary."""
    scenario_json = Path(job["scenario_json"])
    scenario_name = job["scenario_name"]
    scenario_file = job["scenario_file"]
    class_id = int(job["class_id"])
    algo_name = str(job["algo_name"])
    placement_mode = str(job["placement_mode"])
    downlink_duration_ratio = float(job["downlink_duration_ratio"])
    unassigned_penalty = float(job["unassigned_penalty"])
    agility_profile = str(job.get("agility_profile", "Standard-Agility"))
    non_agile_transition_s = float(job.get("non_agile_transition_s", 10.0))
    cfg_overrides = dict(job.get("cfg_overrides", {}) or {})
    schedule_dir = Path(job.get("schedule_dir", str(SCHEDULE_DIR)))

    # RL config / 强化学习配置
    rl_model_path = str(job.get("rl_model_path", ""))
    rl_device = str(job.get("rl_device", "cpu"))
    rl_max_actions = int(job.get("rl_max_actions", 256))

    ow = job.get("objective_weights")
    if isinstance(ow, ObjectiveWeights):
        objective_weights = ow
    else:
        objective_weights = ObjectiveWeights(*ow)

    if class_id in (1, 3, 4):
        objective_desc = _objective_desc(objective_weights, algo_name, class_id)
    else:
        objective_desc = _objective_desc(ObjectiveWeights(0, 0, 0, 0), algo_name, class_id)

    try:
        print(
            f"[RUN] scenario={scenario_name} algo={algo_name} class={class_id} obj={objective_desc}",
            flush=True,
        )
        metrics, schedule_json_path, _ = run_single_scheduling(
            problem_json=scenario_json,
            class_id=class_id,
            algo_name=algo_name,
            placement_mode=placement_mode,
            downlink_duration_ratio=downlink_duration_ratio,
            unassigned_penalty=unassigned_penalty,
            agility_profile=agility_profile,
            non_agile_transition_s=non_agile_transition_s,
            objective_weights=objective_weights,
            cfg_overrides=cfg_overrides,
            schedule_dir=schedule_dir,
            rl_model_path=rl_model_path,
            rl_device=rl_device,
            rl_max_actions=rl_max_actions,
            rl_unassigned_penalty=unassigned_penalty,
        )

        plan_rel = schedule_json_path.name
        try:
            plan_rel = str(schedule_json_path.relative_to(SCHEDULE_DIR))
        except Exception:
            pass

        row = {
            "scenario_name": scenario_name,
            "algo_name": algo_name,
            "class_id": class_id,
            "objective": objective_desc,
            "TP": metrics.task_profit,
            "TCR": metrics.task_completion_rate,
            "BD": metrics.balance_degree,
            "TM": metrics.timeliness_metric,
            "RT": metrics.runtime_efficiency,
            "GAP": metrics.mip_gap,
            "plan": plan_rel,
            "note": "",
            "scenario_file": scenario_file,
        }
        return row

    except Exception as e:
        return {
            "scenario_name": scenario_name,
            "algo_name": algo_name,
            "class_id": class_id,
            "objective": objective_desc,
            "TP": "",
            "TCR": "",
            "BD": "",
            "TM": "",
            "RT": "",
            "GAP": "",
            "plan": "",
            "note": f"ERROR: {e}",
            "scenario_file": scenario_file,
        }

def main():
    placement_mode = "earliest"  # earliest / center / latest (放置模式 / Placement mode)
    downlink_duration_ratio = 1.0
    unassigned_penalty = 1000.0
    agility_profile = "Standard-Agility"  # High-Agility / Standard-Agility / Low-Agility / Limited-Agility
    non_agile_transition_s = 10.0

    # ---------------------------------------------
    # RL (PPO) 训练与测试专属配置 / RL (PPO) exclusive config for Training and Testing
    # ---------------------------------------------
    rl_do_train = False  # 是否开启训练阶段 / Whether to enable the training phase
    rl_do_test = True  # 是否开启评测推断阶段 / Whether to enable the evaluation/inference phase

    rl_train_device = _get_torch_device_prefer_cuda()
    rl_train_gpu_id = 0
    rl_train_episodes = 3000
    rl_resample_every_episodes = 10
    rl_save_every_episodes = 100
    rl_resume_if_exists = True
    rl_rollout_steps = 2048
    rl_max_actions = 1024
    rl_reward_scale = 10.0
    rl_train_base_seed = 0
    rl_model_prefix = "ppo_model"
    rl_train_for_each_objective = False  # 为每一个 objective 独立训练一个模型 / Train an independent model for each objective

    rl_test_device = _get_torch_device_prefer_cuda()
    test_model_paths: list[Path] = [
        MODELS_DIR / "ppo_model_p1_c0_t0_b0.pt"
    ]
    # ---------------------------------------------

    # 并行跑分配置（三个维度：MIP、RL、启发式及元启发式） / Parallel benchmarking config (MIP, RL, Heuristics and Meta-heuristics)
    max_workers_mip = 1
    max_workers_rl = 1
    max_workers_other = 1

    algo_specs = [
        {"class_id": 1, "algo_name": "mip", "cfg_overrides": {}},

        {"class_id": 2, "algo_name": "completion_first", "cfg_overrides": {}},
        {"class_id": 2, "algo_name": "profit_first", "cfg_overrides": {}},
        {"class_id": 2, "algo_name": "balance_first", "cfg_overrides": {}},
        {"class_id": 2, "algo_name": "timeliness_first", "cfg_overrides": {}},

        {"class_id": 3, "algo_name": "sa", "cfg_overrides": {}},
        {"class_id": 3, "algo_name": "ga", "cfg_overrides": {}},
        {"class_id": 3, "algo_name": "aco", "cfg_overrides": {}},

        {"class_id": 4, "algo_name": "ppo", "cfg_overrides": {}},
    ]

    objective_loop = [
        ObjectiveWeights(w_profit=1.0, w_completion=0.0, w_timeliness=0.0, w_balance=0.0),
        # ObjectiveWeights(w_profit=0.0, w_completion=1.0, w_timeliness=0.0, w_balance=0.0),
        # ObjectiveWeights(w_profit=0.0, w_completion=0.0, w_timeliness=0.0, w_balance=1.0),
        # ObjectiveWeights(w_profit=0.0, w_completion=0.0, w_timeliness=1.0, w_balance=0.0),
        # ObjectiveWeights(w_profit=0.25, w_completion=0.25, w_timeliness=0.25, w_balance=0.25),
    ]

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    SCHEDULE_DIR.mkdir(parents=True, exist_ok=True)

    # =========================================================
    # 阶段一：强化学习 (PPO) 训练流程 (按需执行) / Phase 1: RL (PPO) Training Workflow (Execute as needed)
    # =========================================================
    rl_enabled_in_specs = any(spec["class_id"] == 4 for spec in algo_specs)

    if rl_enabled_in_specs and rl_do_train:
        print("[INFO] ================= RL TRAINING START =================")
        weights_list = objective_loop if rl_train_for_each_objective else [ObjectiveWeights(1.0, 0.0, 0.0, 0.0)]
        trained_models = []

        for idx, ow in enumerate(weights_list):
            ow_n = ow.normalized()
            obj_desc = _objective_desc(ow_n, "ppo", 4)
            seed = int(rl_train_base_seed) + idx * 1000

            model_path = MODELS_DIR / (
                f"{rl_model_prefix}_p{ow_n.w_profit:g}_c{ow_n.w_completion:g}_t{ow_n.w_timeliness:g}_b{ow_n.w_balance:g}.pt"
            )

            print(f"[PPO][TRAIN] obj={obj_desc} model={model_path.name}")
            # 同步训练，以防止多进程导致 GPU 资源抢占 / Synchronous training to prevent multi-process GPU resource contention
            mpth = _train_one_process(
                run_id=idx,
                device=rl_train_device,
                model_path=model_path,
                episodes=rl_train_episodes,
                rollout_steps=rl_rollout_steps,
                max_actions=rl_max_actions,
                placement_mode=placement_mode,
                downlink_ratio=downlink_duration_ratio,
                unassigned_penalty=unassigned_penalty,
                objective_weights=ow_n,
                reward_scale=rl_reward_scale,
                seed=seed,
                non_agile_transition_s=non_agile_transition_s,
                resample_every_episodes=rl_resample_every_episodes,
                save_every_episodes=rl_save_every_episodes,
                resume_if_exists=rl_resume_if_exists,
                cuda_visible_devices=str(int(rl_train_gpu_id)) if rl_train_device.startswith("cuda") else None,
            )
            trained_models.append(Path(mpth))
            print(f"[PPO][TRAIN] done: {mpth}")

        print("[INFO] ================= RL TRAINING END =================")

        # 训练出来的模型自动纳入随后的跑分测试环节 / The trained models are automatically included in the subsequent benchmarking test phase
        if rl_do_test:
            for mp in trained_models:
                if mp not in test_model_paths:
                    test_model_paths.append(mp)

    # =========================================================
    # 阶段二：场景加载与算法评测 (Test / Benchmarking) / Phase 2: Scenario Loading & Algorithm Evaluation (Test / Benchmarking)
    # =========================================================
    scenario_paths = _list_scenario_jsons(SCENARIO_DIR)
    if not scenario_paths:
        raise FileNotFoundError(f"No Scenario_*.json files found in {SCENARIO_DIR}")

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    log_path = SCHEDULE_DIR / f"runlog_batch_{timestamp}.txt"

    header, sep = _format_header_line()
    with log_path.open("w", encoding="utf-8") as f:
        f.write("Satellite Benchmark Batch Run Log\n")
        f.write(f"Generated at: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Scenario dir: {SCENARIO_DIR}\n")
        f.write(f"Placement mode: {placement_mode}\n")
        f.write(f"Downlink ratio: {downlink_duration_ratio}\n")
        f.write(f"Unassigned penalty: {unassigned_penalty}\n")
        f.write(f"Parallel: mip={max_workers_mip}, rl={max_workers_rl}, other={max_workers_other}\n")
        f.write("\n")
        f.write(sep + "\n")
        f.write(header + "\n")
        f.write(sep + "\n")

    mip_jobs: list[dict[str, Any]] = []
    rl_jobs: list[dict[str, Any]] = []
    other_jobs: list[dict[str, Any]] = []

    rl_skipped_warned = False

    for scenario_json in scenario_paths:
        scenario_name = scenario_json.stem

        rel_parent = scenario_json.parent.relative_to(SCENARIO_DIR) if scenario_json.parent != SCENARIO_DIR else Path(
            ".")
        scenario_schedule_dir = SCHEDULE_DIR / rel_parent
        scenario_rl_dir = SCHEDULE_DIR /  rel_parent

        for spec in algo_specs:
            class_id = int(spec["class_id"])
            algo_name = str(spec["algo_name"])
            cfg_overrides = dict(spec.get("cfg_overrides", {}) or {})

            if class_id in (1, 3):
                if class_id == 1:
                    sat_count = _sat_count_from_scenario_name(scenario_name)
                    if sat_count is not None and sat_count > 20:
                        for ow in objective_loop:
                            objective_desc = _objective_desc(ow, algo_name, class_id)
                            note = f"SKIPPED: Sats={sat_count} > 20 (MIP disabled)"
                            print(
                                f"[SKIP] scenario={scenario_name} algo={algo_name} class={class_id} obj={objective_desc} -> {note}")
                            _append_runlog_row(
                                log_path,
                                {
                                    "scenario_name": scenario_name,
                                    "algo_name": algo_name,
                                    "class_id": class_id,
                                    "objective": objective_desc,
                                    "TP": "", "TCR": "", "BD": "", "TM": "", "RT": "", "GAP": "", "plan": "",
                                    "note": note,
                                    "scenario_file": str(scenario_json),
                                },
                            )
                        continue

                base_seed = _stable_seed(f"{scenario_name}|{algo_name}")

                for ow in objective_loop:
                    cfg_overrides_job = dict(cfg_overrides)
                    cfg_overrides_job.setdefault("seed", base_seed)

                    job = {
                        "scenario_json": str(scenario_json), "scenario_name": scenario_name,
                        "scenario_file": str(scenario_json),
                        "class_id": class_id, "algo_name": algo_name, "placement_mode": placement_mode,
                        "downlink_duration_ratio": downlink_duration_ratio, "unassigned_penalty": unassigned_penalty,
                        "agility_profile": agility_profile, "non_agile_transition_s": non_agile_transition_s,
                        "objective_weights": (ow.w_profit, ow.w_completion, ow.w_timeliness, ow.w_balance),
                        "cfg_overrides": cfg_overrides_job, "schedule_dir": str(scenario_schedule_dir),
                    }
                    if class_id == 1:
                        mip_jobs.append(job)
                    else:
                        other_jobs.append(job)

            elif class_id == 2:
                other_jobs.append(
                    {
                        "scenario_json": str(scenario_json), "scenario_name": scenario_name,
                        "scenario_file": str(scenario_json),
                        "class_id": class_id, "algo_name": algo_name, "placement_mode": placement_mode,
                        "downlink_duration_ratio": downlink_duration_ratio, "unassigned_penalty": unassigned_penalty,
                        "agility_profile": agility_profile, "non_agile_transition_s": non_agile_transition_s,
                        "objective_weights": (0.0, 0.0, 0.0, 0.0), "cfg_overrides": cfg_overrides,
                        "schedule_dir": str(scenario_schedule_dir),
                    }
                )

            elif class_id == 4:
                if not rl_do_test:
                    continue

                if not test_model_paths:
                    if not rl_skipped_warned:
                        print("[WARN] algo_specs includes PPO (class_id=4), but test_model_paths is not configured. Skipping RL test phase.")
                        rl_skipped_warned = True
                    continue

                for model_path in test_model_paths:
                    if not model_path.exists():
                        print(f"[SKIP] RL model file not found: {model_path}")
                        continue

                    ow = _parse_weights_from_model_name(model_path.name)
                    if ow is None:
                        ow = ObjectiveWeights(1.0, 0.0, 0.0, 0.0)

                    rl_jobs.append(
                        {
                            "scenario_json": str(scenario_json), "scenario_name": scenario_name,
                            "scenario_file": str(scenario_json),
                            "class_id": class_id, "algo_name": algo_name, "placement_mode": placement_mode,
                            "downlink_duration_ratio": downlink_duration_ratio,
                            "unassigned_penalty": unassigned_penalty,
                            "agility_profile": agility_profile, "non_agile_transition_s": non_agile_transition_s,
                            "objective_weights": (ow.w_profit, ow.w_completion, ow.w_timeliness, ow.w_balance),
                            "cfg_overrides": cfg_overrides, "schedule_dir": str(scenario_rl_dir),
                            "rl_model_path": str(model_path), "rl_device": rl_test_device,
                            "rl_max_actions": rl_max_actions,
                        }
                    )

    future_to_pool: dict[Any, str] = {}
    ex_mip = ProcessPoolExecutor(max_workers=max_workers_mip) if (max_workers_mip > 0 and mip_jobs) else None
    ex_rl = ProcessPoolExecutor(max_workers=max_workers_rl) if (max_workers_rl > 0 and rl_jobs) else None
    ex_other = ProcessPoolExecutor(max_workers=max_workers_other) if (max_workers_other > 0 and other_jobs) else None

    try:
        if ex_mip is not None:
            for job in mip_jobs:
                fut = ex_mip.submit(_worker_run_single, job)
                future_to_pool[fut] = "mip"
        if ex_rl is not None:
            for job in rl_jobs:
                fut = ex_rl.submit(_worker_run_single, job)
                future_to_pool[fut] = "rl"
        if ex_other is not None:
            for job in other_jobs:
                fut = ex_other.submit(_worker_run_single, job)
                future_to_pool[fut] = "other"

        for fut in as_completed(list(future_to_pool.keys())):
            row = fut.result()
            _append_runlog_row(log_path, row)

    finally:
        if ex_mip is not None:
            ex_mip.shutdown(wait=True, cancel_futures=False)
        if ex_rl is not None:
            ex_rl.shutdown(wait=True, cancel_futures=False)
        if ex_other is not None:
            ex_other.shutdown(wait=True, cancel_futures=False)

    print(f"[INFO] run log saved: {log_path}")

if __name__ == "__main__":
    try:
        import multiprocessing as _mp

        _mp.freeze_support()
    except Exception:
        pass
    main()