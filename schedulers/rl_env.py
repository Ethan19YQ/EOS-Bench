# -*- coding: utf-8 -*-
"""
rl_env.py

Main functionality:
This module defines the reinforcement learning environment for satellite scheduling.
It provides task ordering, action masking, candidate assignment generation,
state construction, reward calculation based on objective score improvement,
and schedule rollout for PPO-style training and inference.
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
    """A candidate action that assigns a specific Assignment to the current task,
    including an optional ground station downlink segment."""
    assignment: Optional[Assignment]  # None means "skip task"


class RLSchedulingEnv:
    """
    Environment definition:
    - Tasks are sorted by priority from high to low, consistently for training and testing,
      which is stable and reasonable.
    - Action: select one candidate Assignment by index, with index=0 reserved for "skip".
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

        # Objective weights, consistent with other algorithms, used for reward calculation
        self.objective_weights = (objective_weights or ObjectiveWeights(1.0, 0.0, 0.0, 0.0)).normalized()
        # Reward scaling: ObjectiveModel.score is in [0, 1], so using the delta directly would be too small
        self.reward_scale = float(reward_scale)

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

        # Per-step action candidate cache, aligned with action_mask ordering
        self._candidates: List[ActionCandidate] = []

    # -------------------------
    # reset / step
    # -------------------------

    def reset(self, problem: SchedulingProblem) -> Dict[str, Any]:
        self.problem = problem
        self.cm = ConstraintModel(
            problem=problem,
            placement_mode=self.placement_mode,
            unassigned_penalty=1000.0,  # Used for the objective function; RL reward uses self.unassigned_penalty
            downlink_duration_ratio=self.downlink_duration_ratio,
            agility_profile=self.agility_profile,
            non_agile_transition_s=self.non_agile_transition_s,
        )

        # Fixed task ordering for stable training
        self.sorted_tasks = sorted(problem.tasks.values(), key=lambda t: t.priority, reverse=True)
        self.task_index = 0
        self.schedule = Schedule()
        self._candidates = []

        # Objective model used for reward = delta(score)
        self.obj_model = ObjectiveModel(problem, self.objective_weights)
        self._last_score = float(self.obj_model.score(self.schedule))

        return {
            "state": self._get_state(),
            "action_mask": self._get_action_mask(),
        }

    def step(self, action: int) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        assert self.problem is not None and self.cm is not None and self.obj_model is not None

        if self.task_index >= len(self.sorted_tasks):
            # Done
            return {"state": self._get_state(), "action_mask": self._get_action_mask()}, 0.0, True, {"status": "done"}

        # Update candidates to keep them synchronized with the mask
        self._candidates = self._build_candidates_for_current_task()

        # Safe clipping of action
        action = int(action)
        action = max(0, min(action, self.max_actions - 1))

        # If the action is unavailable, treat it as skip with a penalty
        mask = self._get_action_mask()
        if mask[action] < 0.5:
            # Invalid action: negative reward and advance to the next task to avoid getting stuck
            reward = -self.unassigned_penalty
            done = self._advance()
            return {"state": self._get_state(), "action_mask": self._get_action_mask()}, reward, done, {"status": "invalid_action"}

        cand = self._candidates[action]
        task = self.sorted_tasks[self.task_index]

        # Skip task
        if cand.assignment is None:
            # Skip: apply an explicit penalty and advance
            reward = -self.unassigned_penalty
            done = self._advance()
            return {"state": self._get_state(), "action_mask": self._get_action_mask()}, reward, done, {"status": "skip"}

        # Feasibility check performed again
        if not self.cm.is_feasible_assignment(cand.assignment, self.schedule):
            reward = -self.unassigned_penalty
            done = False
            return {"state": self._get_state(), "action_mask": self._get_action_mask()}, reward, done, {"status": "infeasible"}

        # Apply assignment
        self.schedule.assignments.append(cand.assignment)

        # Objective-driven reward consistent with other algorithms:
        # use the increment of ObjectiveModel.score
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
        A simple but stable state vector:
        [task_priority_norm, task_duration_norm, remaining_ratio,
         sat_load_mean, sat_load_std,
         (gs_load_mean, gs_load_std optional)]
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

        # Satellite load: number of assigned tasks per satellite
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

        # Ground station load, only when ground stations exist
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
        The mask length is fixed to max_actions.
        action=0 is always available and means "skip task".
        Other actions correspond to candidate assignments and are set to 1 only if feasible.
        """
        self._candidates = self._build_candidates_for_current_task()
        mask = np.zeros((self.max_actions,), dtype=np.float32)

        # Action 0: skip, always available
        mask[0] = 1.0

        # Fill feasible actions starting from index 1
        k = min(len(self._candidates) - 1, self.max_actions - 1)
        if k > 0:
            mask[1:1 + k] = 1.0

        return mask

    def _build_candidates_for_current_task(self) -> List[ActionCandidate]:
        """
        Construct the candidate action list:
        index=0: skip
        index>=1: feasible assignments
        """
        assert self.problem is not None and self.cm is not None

        # Terminal state
        if self.task_index >= len(self.sorted_tasks):
            return [ActionCandidate(None)]

        task = self.sorted_tasks[self.task_index]

        cands: List[ActionCandidate] = [ActionCandidate(None)]  # 0: skip

        # Sort by window start time for stability
        windows = sorted(task.windows, key=lambda w: w.start_time)

        has_gs = len(self.problem.ground_stations) > 0

        # Unified candidate generation:
        # use the same discrete sub-window rules as other algorithms
        cand_assignments = enumerate_task_candidates(
            problem=self.problem,
            task=task,
            placement_mode=self.placement_mode,
            downlink_duration_ratio=self.downlink_duration_ratio,
            max_candidates=max(0, self.max_actions - 1),  # Reserve index=0 for skip
            random_samples_per_window=0,
            seed=None,
        )

        for a in cand_assignments:
            if self.cm.is_feasible_assignment(a, self.schedule):
                cands.append(ActionCandidate(a))
                if len(cands) >= self.max_actions:
                    break

        return cands