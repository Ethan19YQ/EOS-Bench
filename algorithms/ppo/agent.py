# -*- coding: utf-8 -*-
"""
ppo_agent.py
PPO 训练核心（采样 + GAE + clip 更新） / PPO Training Core (Sampling + GAE + Clip Update)

特点 / Features
----
- 不依赖 stable-baselines3，便于与你的 action_mask / 约束模型耦合；
  / Does not rely on stable-baselines3, facilitating coupling with your action_mask / constraint model;
- 支持 train/test 分离：训练输出模型文件；测试加载模型贪心解码。
  / Supports train/test separation: training outputs model files; testing loads the model for greedy decoding.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from algorithms.ppo.policy import PPOPolicy


@dataclass
class PPOConfig:
    """PPO 超参数配置 / PPO Hyperparameter Configuration"""
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_eps: float = 0.2
    lr: float = 3e-4
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    update_epochs: int = 4
    minibatch: int = 256


class PPOAgent:
    """
    PPO 智能体：负责动作采样、优势估计与网络更新。
    / PPO Agent: Responsible for action sampling, advantage estimation, and network updates.
    """

    def __init__(self, policy: PPOPolicy, cfg: PPOConfig, device: str = "cpu") -> None:
        self.policy = policy.to(device)
        self.cfg = cfg
        self.device = device
        self.opt = optim.Adam(self.policy.parameters(), lr=cfg.lr)

    def act(self, state: np.ndarray, action_mask: np.ndarray) -> Tuple[int, float, float]:
        """
        根据策略网络进行随机采样（用于训练）。
        / Stochastic sampling based on the policy network (used for training).

        返回 / Returns:
            (action, log_prob, value)
        """
        st = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        mk = torch.tensor(action_mask, dtype=torch.float32, device=self.device).unsqueeze(0)

        with torch.no_grad():
            logits, v = self.policy(st)
            logits = self.policy.apply_action_mask(logits, mk)
            dist = torch.distributions.Categorical(logits=logits)
            a = dist.sample()
            logp = dist.log_prob(a)

        return int(a.item()), float(logp.item()), float(v.item())

    def greedy(self, state: np.ndarray, action_mask: np.ndarray) -> int:
        """
        贪心选择概率最大的动作（用于测试/推理）。
        / Greedily select the action with the highest probability (used for testing/inference).
        """
        st = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        mk = torch.tensor(action_mask, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            logits, _ = self.policy(st)
            logits = self.policy.apply_action_mask(logits, mk)
            return int(torch.argmax(logits, dim=-1).item())

    def compute_gae(self, rewards, values, dones, last_value):
        """
        计算广义优势估计 (GAE) 和目标价值 (Returns)。
        / Compute Generalized Advantage Estimation (GAE) and target values (Returns).
        """
        adv = np.zeros_like(rewards, dtype=np.float32)
        last_gae = 0.0
        for t in reversed(range(len(rewards))):
            nonterminal = 1.0 - dones[t]
            next_v = last_value if t == len(rewards) - 1 else values[t + 1]
            delta = rewards[t] + self.cfg.gamma * next_v * nonterminal - values[t]
            last_gae = delta + self.cfg.gamma * self.cfg.gae_lambda * nonterminal * last_gae
            adv[t] = last_gae
        ret = adv + values
        return adv, ret

    def update(self, batch: Dict[str, np.ndarray]) -> Dict[str, float]:
        """
        执行 PPO 的网络更新逻辑 (Clip Loss, Value Loss, Entropy Bonus)。
        / Execute PPO network update logic (Clip Loss, Value Loss, Entropy Bonus).
        """
        states = torch.tensor(batch["state"], dtype=torch.float32, device=self.device)
        masks = torch.tensor(batch["mask"], dtype=torch.float32, device=self.device)
        actions = torch.tensor(batch["action"], dtype=torch.int64, device=self.device)
        old_logp = torch.tensor(batch["logp"], dtype=torch.float32, device=self.device)
        adv = torch.tensor(batch["adv"], dtype=torch.float32, device=self.device)
        ret = torch.tensor(batch["ret"], dtype=torch.float32, device=self.device)

        # 优势标准化 / Advantage normalization
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        n = states.shape[0]
        idx = np.arange(n)

        pi_loss_sum = 0.0
        vf_loss_sum = 0.0
        ent_sum = 0.0
        steps = 0

        for _ in range(self.cfg.update_epochs):
            np.random.shuffle(idx)
            for s in range(0, n, self.cfg.minibatch):
                mb = idx[s:s + self.cfg.minibatch]
                logits, v = self.policy(states[mb])
                logits = self.policy.apply_action_mask(logits, masks[mb])
                dist = torch.distributions.Categorical(logits=logits)

                logp = dist.log_prob(actions[mb])
                ratio = torch.exp(logp - old_logp[mb])

                # 策略裁剪损失 / Policy clipping loss
                surr1 = ratio * adv[mb]
                surr2 = torch.clamp(ratio, 1 - self.cfg.clip_eps, 1 + self.cfg.clip_eps) * adv[mb]
                pi_loss = -torch.min(surr1, surr2).mean()

                # 价值网络损失 / Value network loss
                vf_loss = (ret[mb] - v).pow(2).mean()
                # 熵奖励 / Entropy bonus
                ent = dist.entropy().mean()

                loss = pi_loss + self.cfg.vf_coef * vf_loss - self.cfg.ent_coef * ent

                self.opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), self.cfg.max_grad_norm)
                self.opt.step()

                pi_loss_sum += float(pi_loss.item())
                vf_loss_sum += float(vf_loss.item())
                ent_sum += float(ent.item())
                steps += 1

        return {
            "pi_loss": pi_loss_sum / max(steps, 1),
            "vf_loss": vf_loss_sum / max(steps, 1),
            "entropy": ent_sum / max(steps, 1),
        }

    def save(self, path: str) -> None:
        """保存模型 / Save the model"""
        torch.save(self.policy.state_dict(), path)

    def load(self, path: str) -> None:
        """加载模型 / Load the model"""
        # self.policy.load_state_dict(torch.load(path, map_location=self.device))
        ckpt = torch.load(path, map_location=self.device)

        # 兼容两种格式： / Compatible with two formats:
        # 1) 旧：直接保存 state_dict / 1) Old: Directly save state_dict
        # 2) 新：完整 checkpoint / 2) New: Full checkpoint
        if isinstance(ckpt, dict) and "policy_state" in ckpt:
            state_dict = ckpt["policy_state"]
        else:
            state_dict = ckpt

        self.policy.load_state_dict(state_dict)
        self.policy.to(self.device)
        self.policy.eval()