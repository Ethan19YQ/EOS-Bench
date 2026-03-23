# -*- coding: utf-8 -*-
"""
algorithms/ppo/policy.py

Main functionality:
This module defines the PPO policy network based on an Actor-Critic architecture.
It supports action masking so that infeasible actions can be suppressed during
training and inference.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class PPOPolicy(nn.Module):
    """
    PPO policy network based on the Actor-Critic architecture.
    """
    def __init__(self, state_dim: int, max_actions: int, hidden: int = 128) -> None:
        super().__init__()
        self.max_actions = max_actions

        # Shared feature extraction backbone
        self.backbone = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
        )

        # Actor head: output logits for action probabilities
        self.actor = nn.Linear(hidden, max_actions)
        # Critic head: output state value V(s)
        self.critic = nn.Linear(hidden, 1)

    def forward(self, state: torch.Tensor):
        """
        Forward pass.

        Returns:
            (logits, value)
        """
        h = self.backbone(state)
        logits = self.actor(h)
        value = self.critic(h).squeeze(-1)
        return logits, value

    @staticmethod
    def apply_action_mask(logits: torch.Tensor, action_mask: torch.Tensor) -> torch.Tensor:
        """
        Action masking: replace logits of infeasible actions with a very small value.
        """
        neg = torch.tensor(-1e9, device=logits.device, dtype=logits.dtype)
        # Keep the original value if mask > 0.5; otherwise set it to a very small negative number
        return torch.where(action_mask > 0.5, logits, neg)