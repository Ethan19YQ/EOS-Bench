# -*- coding: utf-8 -*-
"""
ppo_agent.py

Main functionality:
This module defines the PPO agent and its hyperparameter configuration.
It handles action selection, greedy inference, generalized advantage estimation,
policy and value network updates, and model save/load operations for PPO-based
reinforcement learning.
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
    """PPO hyperparameter configuration."""
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
    PPO agent responsible for action sampling, advantage estimation,
    and network updates.
    """

    def __init__(self, policy: PPOPolicy, cfg: PPOConfig, device: str = "cpu") -> None:
        self.policy = policy.to(device)
        self.cfg = cfg
        self.device = device
        self.opt = optim.Adam(self.policy.parameters(), lr=cfg.lr)

    def act(self, state: np.ndarray, action_mask: np.ndarray) -> Tuple[int, float, float]:
        """
        Perform stochastic sampling from the policy network during training.

        Returns:
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
        Greedily select the action with the highest probability
        during testing or inference.
        """
        st = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        mk = torch.tensor(action_mask, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            logits, _ = self.policy(st)
            logits = self.policy.apply_action_mask(logits, mk)
            return int(torch.argmax(logits, dim=-1).item())

    def compute_gae(self, rewards, values, dones, last_value):
        """
        Compute Generalized Advantage Estimation (GAE)
        and target values (returns).
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
        Execute the PPO network update logic, including
        clip loss, value loss, and entropy bonus.
        """
        states = torch.tensor(batch["state"], dtype=torch.float32, device=self.device)
        masks = torch.tensor(batch["mask"], dtype=torch.float32, device=self.device)
        actions = torch.tensor(batch["action"], dtype=torch.int64, device=self.device)
        old_logp = torch.tensor(batch["logp"], dtype=torch.float32, device=self.device)
        adv = torch.tensor(batch["adv"], dtype=torch.float32, device=self.device)
        ret = torch.tensor(batch["ret"], dtype=torch.float32, device=self.device)

        # Advantage normalization
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

                # Policy clipping loss
                surr1 = ratio * adv[mb]
                surr2 = torch.clamp(ratio, 1 - self.cfg.clip_eps, 1 + self.cfg.clip_eps) * adv[mb]
                pi_loss = -torch.min(surr1, surr2).mean()

                # Value network loss
                vf_loss = (ret[mb] - v).pow(2).mean()
                # Entropy bonus
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
        """Save the model."""
        torch.save(self.policy.state_dict(), path)

    def load(self, path: str) -> None:
        """Load the model."""
        # self.policy.load_state_dict(torch.load(path, map_location=self.device))
        ckpt = torch.load(path, map_location=self.device)

        if isinstance(ckpt, dict) and "policy_state" in ckpt:
            state_dict = ckpt["policy_state"]
        else:
            state_dict = ckpt

        self.policy.load_state_dict(state_dict)
        self.policy.to(self.device)
        self.policy.eval()