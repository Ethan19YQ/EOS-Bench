# -*- coding: utf-8 -*-
"""
algorithms/ppo/policy.py
PPO 策略网络（支持 action_mask） / PPO Policy Network (supports action_mask)

说明 / Description
----
- 输入 / Input：env 输出的 state 向量 / state vector from the environment
- 输出 / Output：动作 logits（固定 max_actions） + 价值函数 V(s) / action logits (fixed max_actions) + Value function V(s)
- action_mask 用于把不可行动作的 logits 设为极小值，从而不会被采样
  / action_mask is used to set the logits of infeasible actions to a very small value, preventing them from being sampled
"""

from __future__ import annotations

import torch
import torch.nn as nn



class PPOPolicy(nn.Module):
    """
    基于 Actor-Critic 架构的 PPO 策略网络。
    / PPO Policy network based on the Actor-Critic architecture.
    """
    def __init__(self, state_dim: int, max_actions: int, hidden: int = 128) -> None:
        super().__init__()
        self.max_actions = max_actions

        # 共享特征提取骨干网络 / Shared feature extraction backbone
        self.backbone = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
        )

        # Actor 头部：输出动作概率的 Logits / Actor Head: output logits for action probabilities
        self.actor = nn.Linear(hidden, max_actions)
        # Critic 头部：输出状态价值 V(s) / Critic Head: output state value V(s)
        self.critic = nn.Linear(hidden, 1)

    def forward(self, state: torch.Tensor):
        """
        前向传播 / Forward pass
        返回 / Returns: (logits, value)
        """
        h = self.backbone(state)
        logits = self.actor(h)
        value = self.critic(h).squeeze(-1)
        return logits, value

    @staticmethod
    def apply_action_mask(logits: torch.Tensor, action_mask: torch.Tensor) -> torch.Tensor:
        """
        动作掩码处理：将不可行动作的 Logits 替换为极小值。
        / Action Masking: Replace logits of infeasible actions with a very small value.
        """
        neg = torch.tensor(-1e9, device=logits.device, dtype=logits.dtype)
        # mask > 0.5 时保留原值，否则置为极小负数
        # / Keep original value if mask > 0.5, otherwise set to a very small negative number
        return torch.where(action_mask > 0.5, logits, neg)