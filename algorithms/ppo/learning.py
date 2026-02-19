# -*- coding: utf-8 -*-
"""
ppo_learning.py
第4类：基于学习的调度算法（PPO示例） / Class 4: Learning-based scheduling algorithm (PPO example)

本文件提供两部分 / This file provides two parts:
1) train(): 自动从 output/ 读取场景 json，训练 PPO 并保存模型
   / Automatically read scenario JSONs from output/, train PPO, and save the model
2) search(): 作为调度算法接口（测试阶段），加载模型后对单个问题输出 Schedule
   / Serve as the scheduling algorithm interface (test phase), output Schedule for a single problem after loading the model

与现有框架关系 / Relationship with existing framework
--------------
- 训练：直接使用 schedulers.rl_env.RLSchedulingEnv 进行 rollout
  / Training: directly use schedulers.rl_env.RLSchedulingEnv for rollout
- 测试：实现 BaseSchedulerAlgorithm.search(problem, constraint_model, initial_schedule) -> Schedule
  / Testing: implement BaseSchedulerAlgorithm.search(problem, constraint_model, initial_schedule) -> Schedule
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
    """根据关键超参数生成稳定签名，用于模型命名与断点续训匹配。
       / Generate a stable signature based on key hyperparameters for model naming and resuming training."""
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
        # PPO 超参数 / PPO Hyperparameters
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
    """格式化模型标签 / Format the model tag"""
    ow = run_cfg.objective_weights
    return f"p{ow.w_profit:g}_c{ow.w_completion:g}_t{ow.w_timeliness:g}_b{ow.w_balance:g}_N{int(getattr(run_cfg,'resample_every_episodes',1) or 1)}_{sig}"


def _save_checkpoint(path: Path, agent: 'PPOAgent', run_cfg: 'PPORunConfig', ppo_cfg: 'PPOConfig', episode: int) -> None:
    """保存断点 / Save checkpoint"""
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
    """返回 checkpoint dict（若是旧格式 state_dict 则返回 {'legacy': True}）。
       / Return the checkpoint dict (if it is the old state_dict format, return {'legacy': True})."""
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
        # 兼容旧格式：仅 state_dict / Compatible with old format: only state_dict
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
    """PPO 运行配置 / PPO Run Configuration"""
    episodes: int = 1000
    # 每 N 个 episode 才重新采样/生成一次训练场景（默认=1，即每回合换场景）
    # / Resample/generate training scenario only every N episodes (default=1, i.e., change scenario every episode)
    resample_every_episodes: int = 1
    rollout_steps: int = 2048
    max_actions: int = 256
    placement_mode: str = "earliest"
    unassigned_penalty: float = 1.0
    downlink_duration_ratio: float = 1.0
    device: str = "cpu"
    seed: int = 0

    # ===== 断点续训/周期保存 / Resuming Training/Periodic Saving =====
    save_every_episodes: int = 50  # 每多少个 episode 保存一次模型（0 表示不定期保存） / Save model every N episodes (0 means no periodic saving)
    resume_if_exists: bool = True  # 若同名模型存在则在其基础上继续训练 / If a model with the same name exists, continue training on it

    # ===== 新增：与其它算法一致的多目标权重 / Newly added: Multi-objective weights consistent with other algorithms =====
    # 说明：RL 的 reward 采用 ObjectiveModel.score 的增量（score∈[0,1]），因此这里直接传权重。
    # / Note: RL reward uses the increment of ObjectiveModel.score (score∈[0,1]), so weights are passed directly here.
    objective_weights: ObjectiveWeights = ObjectiveWeights(1.0, 0.0, 0.0, 0.0)
    reward_scale: float = 10.0

    # 与当前规划需求对齐：姿态转换模型参数（训练时可由 sampler 随机覆盖）
    # / Align with current planning requirements: attitude transition model parameters (can be randomly overridden by sampler during training)
    agility_profile: str = "Standard-Agility"
    non_agile_transition_s: float = 10.0


class PPOLearningScheduler:
    """
    PPO 调度器： / PPO Scheduler:
    - 训练阶段：train(output_dir) -> 保存模型 / Training phase: train(output_dir) -> save model
    - 测试阶段：search(problem, ...) -> 输出 Schedule / Testing phase: search(problem, ...) -> output Schedule
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        run_cfg: Optional[PPORunConfig] = None,
    ) -> None:
        self.run_cfg = run_cfg or PPORunConfig()
        self.model_path = model_path  # 可为空：训练时会自动生成默认路径 / Can be None: default path will be generated automatically during training

        self._agent: Optional[PPOAgent] = None
        self._state_dim: Optional[int] = None
        self._max_actions: int = self.run_cfg.max_actions

    # -------------------------
    # Train
    # -------------------------

    def train(self, base_dir: Path) -> Path:
        """
        自动读取 output/ 下的场景 json，训练 PPO 并保存模型。
        / Automatically read scenario JSONs from output/, train PPO, and save the model.
        """
        try:
            import torch  # noqa
        except Exception as e:
            raise RuntimeError("PyTorch must be installed to train PPO: pip install torch") from e
        # ===== GPU 训练可选加速设置（放这里） / Optional acceleration settings for GPU training (placed here) =====
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

        # 读取一个场景拿 state_dim / Read one scenario to get state_dim
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

        # 初始化 PPO / Initialize PPO
        policy = PPOPolicy(state_dim=state_dim, max_actions=self._max_actions, hidden=128)
        ppo_cfg = PPOConfig()
        agent = PPOAgent(policy=policy, cfg=ppo_cfg, device=self.run_cfg.device)
        self._agent = agent

        # ===== 模型命名 + 断点续训 / Model naming + Resuming training =====
        sig = _config_signature(self.run_cfg, ppo_cfg)
        model_tag = _format_model_tag(self.run_cfg, sig)
        model_dir = (base_dir / "output" / "models").resolve()
        model_dir.mkdir(parents=True, exist_ok=True)
        model_path = Path(self.model_path) if self.model_path else (model_dir / f"ppo_{model_tag}.pt")
        meta_path = model_path.with_suffix(model_path.suffix + ".meta.json")
        # 从 0 开始训练；若存在 checkpoint 且允许续训，则会被下面逻辑覆盖。
        # / Start training from 0; if a checkpoint exists and resuming is allowed, it will be overwritten by the logic below.
        start_episode = 0
        if getattr(self.run_cfg, "resume_if_exists", True) and model_path.exists():
            ck = _try_load_checkpoint(model_path, agent)
            if ck is not None and isinstance(ck, dict):
                start_episode = int(ck.get("episode", 0))
                print(f"[PPO][RESUME] loaded {model_path.name}, start_episode={start_episode}", flush=True)
        # 写入/更新 meta，便于人工检查 / Write/update meta, facilitating manual inspection
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


        # 训练循环 / Training loop
        rng = random.Random(int(self.run_cfg.seed))
        t_train_start = time.time()

        # ===== 新增：回合收益记录 / Newly added: Episode return records =====
        episode_returns = []
        episode_lengths = []
        log_path = (base_dir / "output" / "models" / "ppo_train_returns.csv").resolve()
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(log_path, "w", encoding="utf-8") as f:
            f.write("episode,return,length,elapsed_sec\n")

        # 训练时：每次 reset 都生成一个“临时子场景 json”，满足用户的随机抽取要求
        # / During training: generate a "temporary sub-scenario JSON" every reset, satisfying the user's random extraction requirement
        from schedulers.rl_scenario_sampler import build_temp_training_scenario
        tmp_dir = (base_dir / "output" / "tmp_rl").resolve()
        # 场景复用：每 N 个 episode 才重新采样一次（N=run_cfg.resample_every_episodes）
        # / Scenario reuse: resample only every N episodes (N=run_cfg.resample_every_episodes)
        cached_problem = None
        cached_sampled = None
        cached_sampled_at_episode = -1

        def _get_problem_for_episode(ep: int):
            """获取/复用当前 episode 的问题 / Get/reuse the problem for the current episode"""
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

            # ===== 新增：当前回合累计 / Newly added: Accumulate for current episode =====
            ep_return = 0.0
            ep_len = 0            # -------- 生成/复用训练场景并 reset / Generate/reuse training scenario and reset --------
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

                # ===== 新增：累计回合收益/长度 / Newly added: Accumulate episode return/length =====
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

                    # ===== 新增：打印/写入每回合收益 / Newly added: Print/write episode return =====
                    elapsed = time.time() - t_train_start
                    episode_returns.append(ep_return)
                    episode_lengths.append(ep_len)

                    print(f"[PPO-EP] ep={episode}/{self.run_cfg.episodes} "
                          f"return={ep_return:.3f} len={ep_len} elapsed={elapsed:.1f}s")

                    with open(log_path, "a", encoding="utf-8") as f:
                        f.write(f"{episode},{ep_return:.6f},{ep_len},{elapsed:.3f}\n")

                    # ===== 断点保存：每 save_every_episodes 个 episode 保存一次 / Checkpoint saving: Save every save_every_episodes episodes =====
                    se = int(getattr(self.run_cfg, "save_every_episodes", 0) or 0)
                    if se > 0 and (episode % se == 0):
                        try:
                            _save_checkpoint(model_path, agent, self.run_cfg, ppo_cfg, episode)
                            print(f"[PPO][CKPT] saved ep={episode} -> {model_path.name}", flush=True)
                        except Exception as _e:
                            print(f"[PPO][CKPT][WARN] save failed: {_e}", flush=True)
                    # 开新回合（遵循 resample_every_episodes 复用规则） / Start new episode (following resample_every_episodes reuse rules)
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

            # bootstrap & PPO update（你原来的逻辑保持不变 / Your original logic remains unchanged）
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
        # 保存模型（断点格式，包含 optimizer/episode/config，支持继续训练）
        # / Save model (checkpoint format, includes optimizer/episode/config, supports resuming training)
        try:
            _save_checkpoint(model_path, agent, self.run_cfg, ppo_cfg, episode)
            print(f"[PPO-TRAIN] Model saved: {model_path}")
        except Exception as e:
            # 兜底：至少保存权重 / Fallback: at least save weights
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
        """确保模型已加载 / Ensure model is loaded"""
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

        # 为了创建 policy，需要先知道 state_dim：用 problem 计算一次
        # / To create policy, state_dim needs to be known first: calculated once using problem
        # 这里在 search() 内部会传入 problem，所以 search() 里再初始化即可
        # / Since problem is passed into search(), initialization can be done inside search()
        self.model_path = str(mp)

    def search(
        self,
        problem: SchedulingProblem,
        constraint_model: ConstraintModel,
        initial_schedule: Schedule,
        base_dir: Optional[Path] = None,
    ) -> Schedule:
        """
        测试阶段入口：对单个 problem 输出一个 Schedule。
        / Test phase entry point: output a Schedule for a single problem.
        注意：这里不使用 initial_schedule，直接用 RL 解码生成方案（保持独立）。
        / Note: initial_schedule is not used here, RL decoding directly generates the plan (kept independent).
        """
        base_dir = base_dir or Path(__file__).resolve().parents[1]
        self._ensure_loaded(base_dir)

        # 构建 env（与训练一致） / Build env (consistent with training)
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

        # 初始化 policy/agent / Initialize policy/agent
        # state_dim 来自 env / state_dim comes from env
        state_dim = int(obs["state"].shape[0])
        policy = PPOPolicy(state_dim=state_dim, max_actions=self._max_actions, hidden=128)
        agent = PPOAgent(policy=policy, cfg=PPOConfig(), device=self.run_cfg.device)
        agent.load(self.model_path)
        self._agent = agent

        done = False
        while not done:
            a = agent.greedy(obs["state"], obs["action_mask"])  # 测试用贪心更稳定 / Greedy is more stable for testing
            obs, r, done, info = env.step(a)

        return env.get_schedule()