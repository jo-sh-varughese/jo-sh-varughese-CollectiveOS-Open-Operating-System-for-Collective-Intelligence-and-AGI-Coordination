"""
baselines/commnet.py
====================
CommNet-style communication baseline [Sukhbaatar et al., 2016].

Each agent broadcasts a continuous message vector derived from its
local observation.  All agents receive the mean message before
computing their action.  This is a minimal communication baseline —
it tests whether message-passing improves coordination without
requiring explicit coordination protocols.

Reference:
    Sukhbaatar, S., Szlam, A., & Fergus, R. (2016).
    Learning Multiagent Communication with Backpropagation. NeurIPS.
"""

from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
from typing import List, Optional

from agents.base import BaseAgent, RolloutBuffer


class CommNetActorCritic(nn.Module):
    """
    Architecture:
        obs → encoder → message
        [obs, mean_message] → policy_trunk → actor / critic heads
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        msg_dim: int = 16,
        hidden: int = 128,
    ):
        super().__init__()
        self.msg_dim = msg_dim

        # Message encoder: obs → msg
        self.encoder = nn.Sequential(
            nn.Linear(obs_dim, hidden // 2),
            nn.Tanh(),
            nn.Linear(hidden // 2, msg_dim),
            nn.Tanh(),
        )

        # Policy trunk: [obs, mean_msg] → features
        self.trunk = nn.Sequential(
            nn.Linear(obs_dim + msg_dim, hidden),
            nn.LayerNorm(hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
        )
        self.actor_mean = nn.Linear(hidden, action_dim)
        self.actor_log_std = nn.Parameter(torch.zeros(action_dim))
        self.critic = nn.Linear(hidden, 1)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.zeros_(m.bias)
        nn.init.orthogonal_(self.actor_mean.weight, gain=0.01)

    def encode(self, obs: torch.Tensor) -> torch.Tensor:
        """Produce message from local observation."""
        return self.encoder(obs)

    def forward(
        self, obs: torch.Tensor, mean_msg: torch.Tensor
    ) -> tuple:
        aug = torch.cat([obs, mean_msg], dim=-1)
        feat = self.trunk(aug)
        mean = torch.sigmoid(self.actor_mean(feat))
        std = self.actor_log_std.exp().clamp(1e-4, 1.0)
        value = self.critic(feat).squeeze(-1)
        return mean, std, value

    def evaluate(
        self, obs: torch.Tensor, mean_msg: torch.Tensor, actions: torch.Tensor
    ) -> tuple:
        mean, std, value = self.forward(obs, mean_msg)
        dist = Normal(mean, std)
        log_prob = dist.log_prob(actions).sum(-1)
        entropy = dist.entropy().sum(-1)
        return log_prob, entropy, value


class CommNetAgent(BaseAgent):
    """
    CommNet agent.  Must be used in a group — agents share their
    message tensors via the CommNetGroup coordinator.

    Do not instantiate directly; use CommNetGroup instead.
    """

    def __init__(
        self,
        agent_id: int,
        obs_dim: int,
        action_dim: int,
        msg_dim: int = 16,
        lr: float = 3e-4,
        clip_eps: float = 0.2,
        value_coeff: float = 0.5,
        entropy_coeff: float = 0.01,
        n_epochs: int = 4,
        hidden: int = 128,
        device: str = "cpu",
    ):
        super().__init__(agent_id, obs_dim, action_dim)
        self.msg_dim = msg_dim
        self.clip_eps = clip_eps
        self.value_coeff = value_coeff
        self.entropy_coeff = entropy_coeff
        self.n_epochs = n_epochs
        self.device = torch.device(device)

        self.net = CommNetActorCritic(obs_dim, action_dim, msg_dim, hidden).to(self.device)
        self.optimizer = optim.Adam(self.net.parameters(), lr=lr)
        self.buffer = RolloutBuffer()

        self._last_lp: float = 0.0
        self._last_val: float = 0.0
        self._last_msg: Optional[torch.Tensor] = None

    def encode_message(self, obs: np.ndarray) -> np.ndarray:
        obs_t = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            msg = self.net.encode(obs_t)
        return msg.squeeze(0).cpu().numpy()

    def select_action(
        self, obs: np.ndarray, mean_msg: Optional[np.ndarray] = None, **kwargs
    ) -> np.ndarray:
        if mean_msg is None:
            mean_msg = np.zeros(self.msg_dim, dtype=np.float32)

        obs_t = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        msg_t = torch.tensor(mean_msg, dtype=torch.float32, device=self.device).unsqueeze(0)

        with torch.no_grad():
            mean, std, value = self.net(obs_t, msg_t)
            dist = Normal(mean, std)
            action = dist.sample()
            lp = dist.log_prob(action).sum(-1)

        self._last_lp = lp.item()
        self._last_val = value.item()
        return action.squeeze(0).clamp(0, 1).cpu().numpy()

    def store(self, obs, action, reward, done):
        self.buffer.obs.append(obs)
        self.buffer.actions.append(action)
        self.buffer.rewards.append(reward)
        self.buffer.log_probs.append(self._last_lp)
        self.buffer.values.append(self._last_val)
        self.buffer.dones.append(done)

    def update(
        self,
        mean_msgs: Optional[np.ndarray] = None,
        last_obs: Optional[np.ndarray] = None,
        last_mean_msg: Optional[np.ndarray] = None,
    ) -> float:
        if last_obs is not None and last_mean_msg is not None:
            obs_t = torch.tensor(last_obs, dtype=torch.float32, device=self.device).unsqueeze(0)
            msg_t = torch.tensor(last_mean_msg, dtype=torch.float32, device=self.device).unsqueeze(0)
            with torch.no_grad():
                _, _, last_val = self.net(obs_t, msg_t)
            last_value = last_val.item()
        else:
            last_value = 0.0

        returns, advantages = self.buffer.compute_returns_and_advantages(last_value)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        T = len(self.buffer.obs)
        if mean_msgs is None:
            mean_msgs = np.zeros((T, self.msg_dim), dtype=np.float32)

        obs_t = torch.tensor(np.array(self.buffer.obs), dtype=torch.float32, device=self.device)
        act_t = torch.tensor(np.array(self.buffer.actions), dtype=torch.float32, device=self.device)
        msg_t = torch.tensor(mean_msgs, dtype=torch.float32, device=self.device)
        ret_t = torch.tensor(returns, dtype=torch.float32, device=self.device)
        adv_t = torch.tensor(advantages, dtype=torch.float32, device=self.device)
        old_lp = torch.tensor(np.array(self.buffer.log_probs), dtype=torch.float32, device=self.device)

        total_loss = 0.0
        for _ in range(self.n_epochs):
            lp, ent, val = self.net.evaluate(obs_t, msg_t, act_t)
            ratio = torch.exp(lp - old_lp)
            surr = torch.min(
                ratio * adv_t,
                torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * adv_t,
            )
            loss = (
                -surr.mean()
                + self.value_coeff * (ret_t - val).pow(2).mean()
                - self.entropy_coeff * ent.mean()
            )
            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.net.parameters(), 0.5)
            self.optimizer.step()
            total_loss += loss.item()

        self.buffer.clear()
        return total_loss / self.n_epochs

    def reset(self):
        self.buffer.clear()

    def save(self, path: str):
        torch.save(self.net.state_dict(), path)

    def load(self, path: str):
        self.net.load_state_dict(torch.load(path, map_location=self.device))


# ======================================================================
# Group coordinator
# ======================================================================

class CommNetGroup:
    """
    Coordinates a group of CommNet agents.

    Usage:
        group = CommNetGroup(n_agents=5, obs_dim=4, action_dim=1)
        actions = group.act(obs_list)
        group.store(obs_list, actions, rewards, dones)
        group.update(obs_list)
    """

    def __init__(self, n_agents: int, obs_dim: int, action_dim: int, **agent_kwargs):
        self.n_agents = n_agents
        self.agents = [
            CommNetAgent(i, obs_dim, action_dim, **agent_kwargs)
            for i in range(n_agents)
        ]

    def act(self, obs_list: List[np.ndarray]) -> List[np.ndarray]:
        # Round 1: encode messages
        messages = np.array([
            self.agents[i].encode_message(obs_list[i])
            for i in range(self.n_agents)
        ])
        mean_msg = messages.mean(axis=0)

        # Round 2: act with aggregated message
        actions = [
            self.agents[i].select_action(obs_list[i], mean_msg=mean_msg)
            for i in range(self.n_agents)
        ]
        self._last_msgs = messages
        self._last_mean_msg = mean_msg
        return actions

    def store(
        self,
        obs_list: List[np.ndarray],
        actions: List[np.ndarray],
        rewards: List[float],
        dones: List[bool],
    ):
        for i in range(self.n_agents):
            self.agents[i].store(obs_list[i], actions[i], rewards[i], dones[i])

    def update(self, last_obs_list: List[np.ndarray]) -> List[float]:
        # Encode final messages
        msgs = np.array([
            self.agents[i].encode_message(last_obs_list[i])
            for i in range(self.n_agents)
        ])
        last_mean = msgs.mean(axis=0)

        # Retrieve stored mean messages for buffer (approximate: use last)
        losses = []
        for i in range(self.n_agents):
            T = len(self.agents[i].buffer.obs)
            stored_mean_msgs = np.tile(self._last_mean_msg, (T, 1))
            loss = self.agents[i].update(
                mean_msgs=stored_mean_msgs,
                last_obs=last_obs_list[i],
                last_mean_msg=last_mean,
            )
            losses.append(loss)
        return losses

    def reset(self):
        for a in self.agents:
            a.reset()
