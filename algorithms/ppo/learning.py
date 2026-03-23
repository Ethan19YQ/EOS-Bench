# -*- coding: utf-8 -*-
"""
ppo_learning.py

Main functionality:
This module implements a PPO-based learning scheduler for satellite scheduling.
It supports training with scenario resampling, checkpoint saving and resuming,
episode return logging, and greedy inference for testing on a single scheduling problem.

Class 4: Learning-based scheduling algorithm (PPO example)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Dict, Any
import time
import random
import json
import hashlib

import numpy as np



def _config_signature(run_cfg: 'PPORunConfig', ppo_cfg: 'PPOConfig') -> str:
    """Generate a stable signature based on key hyperparameters for model naming and resume-training matching."""
    d = {
        "episodes": int(run_cfg.episodes),
        "resample_every_episodes": int(getattr(run_cfg, "resample_every_episodes", 1) or 1),
        "rollout_steps": int(run_cfg.rollout_steps),
        "max_actions": int(run_cfg.max_actions),
        "placement_mode": str(run_cfg.placement_mode),
        "unassigned_penalty": float(run_cfg.unassigned_penalty),
        "downlink_duration_ratio": float(run_cfg.downlink_duration_ratio),
        "device": str(run_cfg.device),
        "seed": int(run_cfg.seed),
        "objective": {
            "p": float(run_cfg.objective_weights.w_profit),
            "c": float(run_cfg.objective_weights.w_completion),
            "t": float(run_cfg.objective_weights.w_timeliness),
            "b": float(run_cfg.objective_weights.w_balance),
        },
        "reward_scale": float(run_cfg.reward_scale),
        "agility_profile": str(run_cfg.agility_profile),
        "non_agile_transition_s": float(run_cfg.non_agile_transition_s),
        # PPO hyperparameters
        "ppo": {
            "gamma": float(ppo_cfg.gamma),
            "gae_lambda": float(ppo_cfg.gae_lambda),
            "clip_eps": float(ppo_cfg.clip_eps),
            "lr": float(ppo_cfg.lr),
            "ent_coef": float(ppo_cfg.ent_coef),
            "vf_coef": float(ppo_cfg.vf_coef),
            "max_grad_norm": float(ppo_cfg.max_grad_norm),
            "update_epochs": int(ppo_cfg.update_epochs),
            "minibatch": int(ppo_cfg.minibatch),
        },
    }
    s = json.dumps(d, sort_keys=True, ensure_ascii=False)
    h = hashlib.md5(s.encode("utf-8"), usedforsecurity=False).hexdigest()[:10]
    return h


def _format_model_tag(run_cfg: 'PPORunConfig', sig: str) -> str:
    """Format the model tag."""
    ow = run_cfg.objective_weights
    return f"p{ow.w_profit:g}_c{ow.w_completion:g}_t{ow.w_timeliness:g}_b{ow.w_balance:g}_N{int(getattr(run_cfg,'resample_every_episodes',1) or 1)}_{sig}"


def _save_checkpoint(path: Path, agent: 'PPOAgent', run_cfg: 'PPORunConfig', ppo_cfg: 'PPOConfig', episode: int) -> None:
    """Save checkpoint."""
    import torch
    ckpt = {
        "policy_state": agent.policy.state_dict(),
        "opt_state": agent.opt.state_dict(),
        "episode": int(episode),
        "run_cfg": run_cfg.__dict__,
        "ppo_cfg": ppo_cfg.__dict__,
        "timestamp": time.time(),
        "rng_python": random.getstate(),
        "rng_numpy": np.random.get_state(),
        "rng_torch": torch.random.get_rng_state(),
        "rng_torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(ckpt, str(path))


def _try_load_checkpoint(path: Path, agent: 'PPOAgent') -> dict | None:
    """Return the checkpoint dict. If it is the old state_dict format, return {'legacy': True}."""
    try:
        import torch
        obj = torch.load(str(path), map_location=agent.device)
        if isinstance(obj, dict) and "policy_state" in obj:
            agent.policy.load_state_dict(obj["policy_state"])
            if "opt_state" in obj and isinstance(obj["opt_state"], dict):
                try:
                    agent.opt.load_state_dict(obj["opt_state"])
                except Exception:
                    pass
            return obj
        # Compatible with old format: only state_dict
        if isinstance(obj, dict):
            agent.policy.load_state_dict(obj)
            return {"legacy": True, "episode": 0}
        return None
    except Exception:
        return None


from schedulers.rl_env import RLSchedulingEnv
from schedulers.rl_utils import scan_scenario_jsons, default_model_path
from schedulers.scenario_loader import load_scheduling_problem_from_json, SchedulingProblem
from schedulers.constraint_model import ConstraintModel, Schedule
from algorithms.ppo.policy import PPOPolicy
from algorithms.ppo.agent import PPOAgent, PPOConfig
from algorithms.objectives import ObjectiveWeights


@dataclass
class PPORunConfig:
    """PPO run configuration."""
    episodes: int = 1000
    # Resample or regenerate the training scenario only every N episodes
    # (default=1, meaning a new scenario every episode)
    resample_every_episodes: int = 1
    rollout_steps: int = 2048
    max_actions: int = 256
    placement_mode: str = "earliest"
    unassigned_penalty: float = 1.0
    downlink_duration_ratio: float = 1.0
    device: str = "cpu"
    seed: int = 0

    # ===== Resume training / periodic saving =====
    save_every_episodes: int = 50  # Save the model every N episodes (0 means no periodic saving)
    resume_if_exists: bool = True  # Continue training from an existing model with the same name if it exists

    # ===== Newly added: multi-objective weights consistent with other algorithms =====
    # Note: RL reward uses the increment of ObjectiveModel.score (score ∈ [0,1]),
    # so the weights are passed directly here.
    objective_weights: ObjectiveWeights = ObjectiveWeights(1.0, 0.0, 0.0, 0.0)
    reward_scale: float = 10.0

    # Align with current planning requirements: attitude transition model parameters
    # (can be randomly overridden by the sampler during training)
    agility_profile: str = "Standard-Agility"
    non_agile_transition_s: float = 10.0


class PPOLearningScheduler:
    """
    PPO scheduler:
    - Training phase: train(output_dir) -> save model
    - Testing phase: search(problem, ...) -> output Schedule
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        run_cfg: Optional[PPORunConfig] = None,
    ) -> None:
        self.run_cfg = run_cfg or PPORunConfig()
        self.model_path = model_path  # Can be None: a default path will be generated automatically during training

        self._agent: Optional[PPOAgent] = None
        self._state_dim: Optional[int] = None
        self._max_actions: int = self.run_cfg.max_actions

    # -------------------------
    # Train
    # -------------------------

    def train(self, base_dir: Path) -> Path:
        """
        Automatically read scenario JSON files under output/, train PPO,
        and save the model.
        """
        try:
            import torch  # noqa
        except Exception as e:
            raise RuntimeError("PyTorch must be installed to train PPO: pip install torch") from e
        # ===== Optional acceleration settings for GPU training =====
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            try:
                torch.set_float32_matmul_precision("high")
            except Exception:
                pass
        output_dir = (base_dir / "output").resolve()
        scenario_paths = scan_scenario_jsons(output_dir)
        if not scenario_paths:
            raise FileNotFoundError(f"No scenario JSONs available for training found in {output_dir} (schedules excluded)")

        # Read one scenario to get state_dim
        sample_problem = load_scheduling_problem_from_json(scenario_paths[0])
        env = RLSchedulingEnv(
            max_actions=self._max_actions,
            placement_mode=self.run_cfg.placement_mode,
            unassigned_penalty=self.run_cfg.unassigned_penalty,
            downlink_duration_ratio=self.run_cfg.downlink_duration_ratio,
            objective_weights=self.run_cfg.objective_weights,
            reward_scale=self.run_cfg.reward_scale,
            agility_profile=self.run_cfg.agility_profile,
            non_agile_transition_s=self.run_cfg.non_agile_transition_s,
        )
        obs0 = env.reset(sample_problem)
        state_dim = int(obs0["state"].shape[0])
        self._state_dim = state_dim

        # Initialize PPO
        policy = PPOPolicy(state_dim=state_dim, max_actions=self._max_actions, hidden=128)
        ppo_cfg = PPOConfig()
        agent = PPOAgent(policy=policy, cfg=ppo_cfg, device=self.run_cfg.device)
        self._agent = agent

        # ===== Model naming + resume training =====
        sig = _config_signature(self.run_cfg, ppo_cfg)
        model_tag = _format_model_tag(self.run_cfg, sig)
        model_dir = (base_dir / "output" / "models").resolve()
        model_dir.mkdir(parents=True, exist_ok=True)
        model_path = Path(self.model_path) if self.model_path else (model_dir / f"ppo_{model_tag}.pt")
        meta_path = model_path.with_suffix(model_path.suffix + ".meta.json")
        # Start training from 0. If a checkpoint exists and resuming is allowed,
        # it will be overwritten by the logic below.
        start_episode = 0
        if getattr(self.run_cfg, "resume_if_exists", True) and model_path.exists():
            ck = _try_load_checkpoint(model_path, agent)
            if ck is not None and isinstance(ck, dict):
                start_episode = int(ck.get("episode", 0))
                print(f"[PPO][RESUME] loaded {model_path.name}, start_episode={start_episode}", flush=True)
        # Write or update meta for easier manual inspection
        try:
            meta = {
                "model_path": str(model_path),
                "model_tag": model_tag,
                "signature": sig,
                "run_cfg": self.run_cfg.__dict__,
                "ppo_cfg": ppo_cfg.__dict__,
            }
            meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception:
            pass


        # Training loop
        rng = random.Random(int(self.run_cfg.seed))
        t_train_start = time.time()

        # ===== Newly added: episode return records =====
        episode_returns = []
        episode_lengths = []
        log_path = (base_dir / "output" / "models" / "ppo_train_returns.csv").resolve()
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(log_path, "w", encoding="utf-8") as f:
            f.write("episode,return,length,elapsed_sec\n")

        # During training, each reset generates a temporary sub-scenario JSON,
        # satisfying the user's random sampling requirement
        from schedulers.rl_scenario_sampler import build_temp_training_scenario
        tmp_dir = (base_dir / "output" / "tmp_rl").resolve()
        # Scenario reuse: only resample every N episodes
        # (N=run_cfg.resample_every_episodes)
        cached_problem = None
        cached_sampled = None
        cached_sampled_at_episode = -1

        def _get_problem_for_episode(ep: int):
            """Get or reuse the problem for the current episode."""
            nonlocal cached_problem, cached_sampled, cached_sampled_at_episode
            N = int(getattr(self.run_cfg, "resample_every_episodes", 1) or 1)
            if N < 1:
                N = 1
            if cached_problem is None or (ep % N == 0):
                base_json = Path(rng.choice(scenario_paths))
                sampled = build_temp_training_scenario(base_json, out_dir=tmp_dir, rng=rng)
                p = load_scheduling_problem_from_json(sampled.tmp_json)
                try:
                    sampled.tmp_json.unlink(missing_ok=True)
                except Exception:
                    pass
                cached_problem = p
                cached_sampled = sampled
                cached_sampled_at_episode = ep
                print(
                    f"[PPO-RESET] (resample) ep={ep} base={sampled.base_json.name} -> sats={sampled.sampled_sats} "
                    f"tasks={sampled.sampled_tasks} days={sampled.sampled_days:g} "
                    f"cap={sampled.capacity_mode} agility={sampled.agility_profile} (every {N} eps)",
                    flush=True,
                )
                return p, sampled
            else:
                p = cached_problem
                sampled = cached_sampled
                print(
                    f"[PPO-RESET] (reuse) ep={ep} reuse_from_ep={cached_sampled_at_episode} (every {N} eps) "
                    f"base={getattr(sampled, 'base_json', 'N/A')}",
                    flush=True,
                )
                return p, sampled

        episode = start_episode
        while episode < self.run_cfg.episodes:
            # rollout buffer
            states, masks, actions, rewards, dones, logps, values = [], [], [], [], [], [], []

            # ===== Accumulate for current episode =====
            ep_return = 0.0
            ep_len = 0

            # -------- Generate or reuse training scenario and reset --------
            p, sampled = _get_problem_for_episode(episode)

            env = RLSchedulingEnv(
                max_actions=self._max_actions,
                placement_mode=self.run_cfg.placement_mode,
                unassigned_penalty=self.run_cfg.unassigned_penalty,
                downlink_duration_ratio=self.run_cfg.downlink_duration_ratio,
                objective_weights=self.run_cfg.objective_weights,
                reward_scale=self.run_cfg.reward_scale,
                agility_profile=getattr(sampled, "agility_profile", self.run_cfg.agility_profile),
                non_agile_transition_s=self.run_cfg.non_agile_transition_s,
            )
            obs = env.reset(p)



            for _ in range(self.run_cfg.rollout_steps):
                a, logp, v = agent.act(obs["state"], obs["action_mask"])
                next_obs, r, done, info = env.step(a)

                # ===== Newly added: accumulate episode return and length =====
                ep_return += float(r)
                ep_len += 1

                states.append(obs["state"])
                masks.append(obs["action_mask"])
                actions.append(a)
                rewards.append(r)
                dones.append(1.0 if done else 0.0)
                logps.append(logp)
                values.append(v)

                obs = next_obs

                if done:
                    episode += 1

                    # ===== Newly added: print and write episode return =====
                    elapsed = time.time() - t_train_start
                    episode_returns.append(ep_return)
                    episode_lengths.append(ep_len)

                    print(f"[PPO-EP] ep={episode}/{self.run_cfg.episodes} "
                          f"return={ep_return:.3f} len={ep_len} elapsed={elapsed:.1f}s")

                    with open(log_path, "a", encoding="utf-8") as f:
                        f.write(f"{episode},{ep_return:.6f},{ep_len},{elapsed:.3f}\n")

                    # ===== Checkpoint saving: save every save_every_episodes episodes =====
                    se = int(getattr(self.run_cfg, "save_every_episodes", 0) or 0)
                    if se > 0 and (episode % se == 0):
                        try:
                            _save_checkpoint(model_path, agent, self.run_cfg, ppo_cfg, episode)
                            print(f"[PPO][CKPT] saved ep={episode} -> {model_path.name}", flush=True)
                        except Exception as _e:
                            print(f"[PPO][CKPT][WARN] save failed: {_e}", flush=True)
                    # Start a new episode following the resample_every_episodes reuse rule
                    p, sampled = _get_problem_for_episode(episode)
                    env = RLSchedulingEnv(
                        max_actions=self._max_actions,
                        placement_mode=self.run_cfg.placement_mode,
                        unassigned_penalty=self.run_cfg.unassigned_penalty,
                        downlink_duration_ratio=self.run_cfg.downlink_duration_ratio,
                        objective_weights=self.run_cfg.objective_weights,
                        reward_scale=self.run_cfg.reward_scale,
                        agility_profile=getattr(sampled, "agility_profile", self.run_cfg.agility_profile),
                        non_agile_transition_s=self.run_cfg.non_agile_transition_s,
                    )
                    obs = env.reset(p)

                    if episode >= self.run_cfg.episodes:
                        break

            # bootstrap & PPO update (the original logic remains unchanged)
            import torch
            with torch.no_grad():
                st = torch.tensor(obs["state"], dtype=torch.float32, device=self.run_cfg.device).unsqueeze(0)
                mk = torch.tensor(obs["action_mask"], dtype=torch.float32, device=self.run_cfg.device).unsqueeze(0)
                logits, last_v = agent.policy(st)
                _ = agent.policy.apply_action_mask(logits, mk)
                last_value = float(last_v.item())

            adv, ret = agent.compute_gae(
                rewards=np.array(rewards, dtype=np.float32),
                values=np.array(values, dtype=np.float32),
                dones=np.array(dones, dtype=np.float32),
                last_value=last_value,
            )

            stats = agent.update({
                "state": np.array(states, dtype=np.float32),
                "mask": np.array(masks, dtype=np.float32),
                "action": np.array(actions, dtype=np.int64),
                "logp": np.array(logps, dtype=np.float32),
                "adv": adv.astype(np.float32),
                "ret": ret.astype(np.float32),
            })

            if episode % 50 == 0:
                elapsed = time.time() - t_train_start
                print(f"[PPO-TRAIN] ep={episode}/{self.run_cfg.episodes} "
                      f"pi_loss={stats['pi_loss']:.4f} vf_loss={stats['vf_loss']:.4f} "
                      f"ent={stats['entropy']:.4f} elapsed={elapsed:.1f}s")

        print(f"[PPO-TRAIN] Episode return log written to: {log_path}")
        # Save model in checkpoint format, including optimizer, episode, and config
        # to support resume training
        try:
            _save_checkpoint(model_path, agent, self.run_cfg, ppo_cfg, episode)
            print(f"[PPO-TRAIN] Model saved: {model_path}")
        except Exception as e:
            # Fallback: at least save weights
            try:
                agent.save(str(model_path))
                print(f"[PPO-TRAIN][WARN] checkpoint save failed, only weights saved: {model_path} err={e}")
            except Exception:
                raise
        return model_path

    # -------------------------
    # Test (as algorithm.search)
    # -------------------------

    def _ensure_loaded(self, base_dir: Path) -> None:
        """Ensure the model is loaded."""
        if self._agent is not None:
            return

        try:
            import torch  # noqa
        except Exception as e:
            raise RuntimeError("PyTorch must be installed to load PPO model: pip install torch") from e

        if self.model_path is None:
            mp = default_model_path(base_dir, "ppo")
        else:
            mp = Path(self.model_path).resolve()

        if not mp.exists():
            raise FileNotFoundError(f"PPO model does not exist: {mp} (please train first or specify correct --model_path)")

        # To create the policy, state_dim must be known first
        # It will be calculated once using problem
        # Since problem is passed into search(), initialization can be done inside search()
        self.model_path = str(mp)

    def search(
        self,
        problem: SchedulingProblem,
        constraint_model: ConstraintModel,
        initial_schedule: Schedule,
        base_dir: Optional[Path] = None,
    ) -> Schedule:
        """
        Test-phase entry point: output a Schedule for a single problem.
        Note: initial_schedule is not used here. RL decoding directly generates
        the schedule to keep the process independent.
        """
        base_dir = base_dir or Path(__file__).resolve().parents[1]
        self._ensure_loaded(base_dir)

        # Build env consistent with training
        env = RLSchedulingEnv(
            max_actions=self._max_actions,
            placement_mode=self.run_cfg.placement_mode,
            unassigned_penalty=self.run_cfg.unassigned_penalty,
            downlink_duration_ratio=self.run_cfg.downlink_duration_ratio,
            objective_weights=self.run_cfg.objective_weights,
            reward_scale=self.run_cfg.reward_scale,
            agility_profile=self.run_cfg.agility_profile,
            non_agile_transition_s=self.run_cfg.non_agile_transition_s,
        )
        obs = env.reset(problem)

        # Initialize policy and agent
        # state_dim comes from env
        state_dim = int(obs["state"].shape[0])
        policy = PPOPolicy(state_dim=state_dim, max_actions=self._max_actions, hidden=128)
        agent = PPOAgent(policy=policy, cfg=PPOConfig(), device=self.run_cfg.device)
        agent.load(self.model_path)
        self._agent = agent

        done = False
        while not done:
            a = agent.greedy(obs["state"], obs["action_mask"])  # Greedy selection is more stable for testing
            obs, r, done, info = env.step(a)

        return env.get_schedule()