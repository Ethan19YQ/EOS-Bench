# -*- coding: utf-8 -*-
"""
rl_env.py
强化学习调度环境（PPO训练/测试共用） / Reinforcement Learning Scheduling Environment (Shared for PPO Train/Test)

本模块功能 / Module Functionality
----------
将 SchedulingProblem + ConstraintModel 封装成一个“按任务逐个决策”的 RL 环境：
Encapsulate SchedulingProblem + ConstraintModel into a "task-by-task decision" RL environment:
- reset(problem): 初始化一个空 Schedule； / Initialize an empty Schedule;
- step(action): 为当前任务选择一个可行 Assignment（或选择跳过）； / Choose a feasible Assignment for the current task (or choose to skip);
- reward: 以“任务优先级收益 - 未分配惩罚/冲突惩罚”为核心，可扩展； / Reward based on "task priority profit - unassigned penalty/conflict penalty", extensible;
- 支持 action_mask：屏蔽不可行动作（PPO训练必备）。 / Support action_mask: mask infeasible actions (essential for PPO training).

与现有框架的关系 / Relationship with Existing Framework
----------------
- 输入问题 / Input Problem：schedulers.scenario_loader.SchedulingProblem
- 约束与构造 / Constraints and Construction：schedulers.constraint_model.ConstraintModel / Assignment / Schedule
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple, Any

import numpy as np

from .scenario_loader import SchedulingProblem, SchedulingTask, TaskWindow, CommWindow
from .constraint_model import ConstraintModel, Assignment, Schedule, TimePlacementStrategy
from algorithms.objectives import ObjectiveWeights, ObjectiveModel
from algorithms.candidate_pool import enumerate_task_candidates


@dataclass
class ActionCandidate:
    """一个候选动作：给当前任务安排一个具体的 Assignment（含可选地面站数传段）
       / An action candidate: arrange a specific Assignment (including optional ground station data transmission segment) for the current task"""
    assignment: Optional[Assignment]  # None 表示“跳过任务” / None means "skip task"


class RLSchedulingEnv:
    """
    环境定义：
    Environment definition:
    - 任务按 priority 从高到低排序（训练/测试一致，稳定且合理）
      / Tasks are sorted by priority from high to low (consistent for train/test, stable and reasonable)
    - 动作：从候选 Assignment 列表里选一个（index），另预留 index=0 为“跳过”
      / Action: Select one (index) from the list of candidate Assignments, reserving index=0 for "skip"
    """

    def __init__(
        self,
        max_actions: int = 256,
        placement_mode: str = "earliest",
        unassigned_penalty: float = 1.0,
        downlink_duration_ratio: float = 1.0,
        objective_weights: ObjectiveWeights | None = None,
        reward_scale: float = 10.0,
        agility_profile: str = "Standard-Agility",
        non_agile_transition_s: float = 10.0,
    ) -> None:
        self.max_actions = int(max_actions)
        self.placement_mode = placement_mode
        self.unassigned_penalty = float(unassigned_penalty)
        self.downlink_duration_ratio = float(downlink_duration_ratio)

        # 目标权重（与其它算法一致）：用于 reward 计算
        # Objective weights (consistent with other algorithms): used for reward calculation
        self.objective_weights = (objective_weights or ObjectiveWeights(1.0, 0.0, 0.0, 0.0)).normalized()
        # reward 缩放：ObjectiveModel.score 是 0~1，直接用 delta 会很小，缩放后更利于训练
        # Reward scaling: ObjectiveModel.score is 0~1, using delta directly would be very small, scaling is better for training
        self.reward_scale = float(reward_scale)

        # 与当前规划需求保持一致：姿态机动模型参数
        # Keep consistent with current planning requirements: attitude maneuver model parameters
        self.agility_profile = str(agility_profile)
        self.non_agile_transition_s = float(non_agile_transition_s)

        self.problem: Optional[SchedulingProblem] = None
        self.cm: Optional[ConstraintModel] = None
        self.obj_model: Optional[ObjectiveModel] = None
        self._last_score: float = 0.0

        self.sorted_tasks: List[SchedulingTask] = []
        self.task_index: int = 0

        self.schedule: Schedule = Schedule()

        # 每步动作候选缓存（含 action_mask 对应顺序）
        # Per-step action candidate cache (including action_mask corresponding order)
        self._candidates: List[ActionCandidate] = []

    # -------------------------
    # reset / step
    # -------------------------

    def reset(self, problem: SchedulingProblem) -> Dict[str, Any]:
        self.problem = problem
        self.cm = ConstraintModel(
            problem=problem,
            placement_mode=self.placement_mode,
            unassigned_penalty=1000.0,          # 注意：这里是目标函数用的；RL reward 用 self.unassigned_penalty / Note: used for objective function; RL reward uses self.unassigned_penalty
            downlink_duration_ratio=self.downlink_duration_ratio,
            agility_profile=self.agility_profile,
            non_agile_transition_s=self.non_agile_transition_s,
        )

        # 任务排序（固定策略，便于训练稳定）
        # Task sorting (fixed strategy, facilitating training stability)
        self.sorted_tasks = sorted(problem.tasks.values(), key=lambda t: t.priority, reverse=True)
        self.task_index = 0
        self.schedule = Schedule()
        self._candidates = []

        # 目标模型：用于 reward = delta(score)
        # Objective model: used for reward = delta(score)
        self.obj_model = ObjectiveModel(problem, self.objective_weights)
        self._last_score = float(self.obj_model.score(self.schedule))

        return {
            "state": self._get_state(),
            "action_mask": self._get_action_mask(),
        }

    def step(self, action: int) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        assert self.problem is not None and self.cm is not None and self.obj_model is not None

        if self.task_index >= len(self.sorted_tasks):
            # 已结束 / Done
            return {"state": self._get_state(), "action_mask": self._get_action_mask()}, 0.0, True, {"status": "done"}

        # 更新候选（保证与 mask 同步） / Update candidates (ensure synchronization with mask)
        self._candidates = self._build_candidates_for_current_task()

        # action 安全裁剪 / Safe clipping of action
        action = int(action)
        action = max(0, min(action, self.max_actions - 1))

        # 若 action 不可用，则视为跳过（惩罚） / If action is unavailable, treat as skip (penalty)
        mask = self._get_action_mask()
        if mask[action] < 0.5:
            # 无效动作：负奖励（并推进到下一个任务，避免卡死）
            # Invalid action: negative reward (and advance to the next task to avoid getting stuck)
            reward = -self.unassigned_penalty
            done = self._advance()
            return {"state": self._get_state(), "action_mask": self._get_action_mask()}, reward, done, {"status": "invalid_action"}

        cand = self._candidates[action]
        task = self.sorted_tasks[self.task_index]

        # 跳过任务 / Skip task
        if cand.assignment is None:
            # 跳过：给一个明确惩罚（并推进） / Skip: give an explicit penalty (and advance)
            reward = -self.unassigned_penalty
            done = self._advance()
            return {"state": self._get_state(), "action_mask": self._get_action_mask()}, reward, done, {"status": "skip"}

        # 可行性检查（再次检查） / Feasibility check (double check)
        if not self.cm.is_feasible_assignment(cand.assignment, self.schedule):
            reward = -self.unassigned_penalty
            done = False
            return {"state": self._get_state(), "action_mask": self._get_action_mask()}, reward, done, {"status": "infeasible"}

        # 落地 / Apply Assignment
        self.schedule.assignments.append(cand.assignment)

        # reward：与其它算法一致的“目标驱动”——用 ObjectiveModel.score 的增量
        # reward: "Objective-driven" consistent with other algorithms -- use the increment of ObjectiveModel.score
        new_score = float(self.obj_model.score(self.schedule))
        reward = self.reward_scale * (new_score - self._last_score)
        self._last_score = new_score

        done = self._advance()
        return {"state": self._get_state(), "action_mask": self._get_action_mask()}, reward, done, {"status": "ok"}

    def get_schedule(self) -> Schedule:
        return self.schedule

    # -------------------------
    # internal helpers
    # -------------------------

    def _advance(self) -> bool:
        self.task_index += 1
        return self.task_index >= len(self.sorted_tasks)

    def _get_state(self) -> np.ndarray:
        """
        简单但稳定的向量 state：
        Simple but stable vector state:
        [task_priority_norm, task_duration_norm, remaining_ratio,
         sat_load_mean, sat_load_std,
         (gs_load_mean, gs_load_std 可选 / optional)]
        """
        assert self.problem is not None

        T = (self.problem.end_time - self.problem.start_time).total_seconds()
        T = max(T, 1.0)

        if self.task_index >= len(self.sorted_tasks):
            dim = 5 + (2 if len(self.problem.ground_stations) > 0 else 0)
            return np.zeros((dim,), dtype=np.float32)

        task = self.sorted_tasks[self.task_index]
        pr = float(task.priority)
        dur = float(task.required_duration)

        # 卫星负载：每颗卫星分配任务数 / Satellite load: number of tasks assigned per satellite
        sat_ids = list(self.problem.satellites.keys())
        sat_counts = []
        for sid in sat_ids:
            c = sum(1 for a in self.schedule.assignments if a.satellite_id == sid)
            sat_counts.append(c)
        sat_counts = np.array(sat_counts, dtype=np.float32) if sat_counts else np.zeros((1,), dtype=np.float32)
        sat_mean = float(np.mean(sat_counts))
        sat_std = float(np.std(sat_counts))

        base = [
            pr / 10.0,
            dur / T,
            1.0 - float(self.task_index) / max(len(self.sorted_tasks), 1),
            sat_mean / max(len(self.sorted_tasks), 1),
            sat_std / max(len(self.sorted_tasks), 1),
        ]

        # 地面站负载（仅当有地面站） / Ground station load (only if ground stations exist)
        if len(self.problem.ground_stations) > 0:
            gs_ids = list(self.problem.ground_stations.keys())
            gs_counts = []
            for gid in gs_ids:
                c = sum(1 for a in self.schedule.assignments if a.ground_station_id == gid)
                gs_counts.append(c)
            gs_counts = np.array(gs_counts, dtype=np.float32) if gs_counts else np.zeros((1,), dtype=np.float32)
            gs_mean = float(np.mean(gs_counts))
            gs_std = float(np.std(gs_counts))
            base.extend([gs_mean / max(len(self.sorted_tasks), 1), gs_std / max(len(self.sorted_tasks), 1)])

        return np.array(base, dtype=np.float32)

    def _get_action_mask(self) -> np.ndarray:
        """
        mask 长度固定 max_actions。 / mask length is fixed to max_actions.
        action=0 永远可用（跳过任务）。 / action=0 is always available (skip task).
        其余 action 对应候选 assignment（可行才置1）。 / Other actions correspond to candidate assignments (set to 1 only if feasible).
        """
        self._candidates = self._build_candidates_for_current_task()
        mask = np.zeros((self.max_actions,), dtype=np.float32)

        # 0号动作：跳过（永远可选） / Action 0: Skip (always selectable)
        mask[0] = 1.0

        # 从1开始填可行动作 / Fill feasible actions starting from 1
        k = min(len(self._candidates) - 1, self.max_actions - 1)
        if k > 0:
            mask[1:1 + k] = 1.0

        return mask

    def _build_candidates_for_current_task(self) -> List[ActionCandidate]:
        """
        构造候选动作列表： / Construct candidate action list:
        index=0: skip
        index>=1: feasible assignments
        """
        assert self.problem is not None and self.cm is not None

        # 结束态 / Terminal state
        if self.task_index >= len(self.sorted_tasks):
            return [ActionCandidate(None)]

        task = self.sorted_tasks[self.task_index]

        cands: List[ActionCandidate] = [ActionCandidate(None)]  # 0: skip

        # 按窗口开始时间排序（稳定） / Sort by window start time (stable)
        windows = sorted(task.windows, key=lambda w: w.start_time)

        has_gs = len(self.problem.ground_stations) > 0

        # 统一候选生成：使用与其它算法一致的离散子窗口规则（duration_s + time_step + agile/non_agile）
        # Unified candidate generation: use discrete sub-window rules consistent with other algorithms (duration_s + time_step + agile/non_agile)
        cand_assignments = enumerate_task_candidates(
            problem=self.problem,
            task=task,
            placement_mode=self.placement_mode,
            downlink_duration_ratio=self.downlink_duration_ratio,
            max_candidates=max(0, self.max_actions - 1),  # 预留 index=0 为 skip / Reserve index=0 for skip
            random_samples_per_window=0,
            seed=None,
        )

        for a in cand_assignments:
            if self.cm.is_feasible_assignment(a, self.schedule):
                cands.append(ActionCandidate(a))
                if len(cands) >= self.max_actions:
                    break

        return cands