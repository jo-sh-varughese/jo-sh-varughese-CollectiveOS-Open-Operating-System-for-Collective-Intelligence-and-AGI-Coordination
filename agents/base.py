"""
agents/base.py
==============
Base classes for all CollectiveOS-Bench agents.

Hierarchy:
    BaseAgent
        ├── RandomAgent
        ├── GreedyAgent
        └── PolicyAgent (PyTorch neural net + PPO-style update)

Communication:
    MessageBus  – lightweight broadcast channel for partial observability
                  experiments.  Not used by all baselines.
"""

from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal, Categorical
from typing import Any, Dict, List, Optional, Tuple
import abc


# ======================================================================
# Message passing
# ======================================================================

class MessageBus:
    """
    Simple synchronous broadcast bus.
    Agents push observations or intent signals; all agents can read.
    """

    def __init__(self, n_agents: int, msg_dim: int):
        self.n_agents = n_agents
        self.msg_dim = msg_dim
        self._buffer: List[Optional[np.ndarray]] = [None] * n_agents

    def send(self, agent_id: int, message: np.ndarray):
        assert message.shape == (self.msg_dim,)
        self._buffer[agent_id] = message.copy()

    def receive_all(self) -> np.ndarray:
        """Returns (n_agents, msg_dim) array; zeros for silent agents."""
        out = np.zeros((self.n_agents, self.msg_dim), dtype=np.float32)
        for i, m in enumerate(self._buffer):
            if m is not None:
                out[i] = m
        return out

    def clear(self):
        self._buffer = [None] * self.n_agents


# ======================================================================
# Base interface
# ======================================================================

class BaseAgent(abc.ABC):
    """Abstract agent interface."""

    def __init__(self, agent_id: int, obs_dim: int, action_dim: int):
        self.agent_id = agent_id
        self.obs_dim = obs_dim
        self.action_dim = action_dim

    @abc.abstractmethod
    def select_action(self, obs: np.ndarray, **kwargs) -> np.ndarray:
        """Return an action given an observation."""

    def update(self, *args, **kwargs):
        """Optional gradient update step."""

    def reset(self):
        """Called at episode start."""


# ======================================================================
# Baselines
# ======================================================================

class RandomAgent(BaseAgent):
    """Samples uniformly from action space."""

    def __init__(self, agent_id: int, obs_dim: int, action_dim: int,
                 action_low: float = 0.0, action_high: float = 1.0,
                 seed: Optional[int] = None):
        super().__init__(agent_id, obs_dim, action_dim)
        self.low = action_low
        self.high = action_high
        self.rng = np.random.default_rng(seed)

    def select_action(self, obs: np.ndarray, **kwargs) -> np.ndarray:
        return self.rng.uniform(self.low, self.high, self.action_dim).astype(np.float32)


class GreedyAgent(BaseAgent):
    """Always extracts at maximum; no cooperation."""

    def select_action(self, obs: np.ndarray, **kwargs) -> np.ndarray:
        return np.ones(self.action_dim, dtype=np.float32)


class CooperativeAgent(BaseAgent):
    """Always complies with rules and extracts conservatively."""

    def select_action(self, obs: np.ndarray, **kwargs) -> np.ndarray:
        action = np.zeros(self.action_dim, dtype=np.float32)
        action[0] = 0.3   # conservative extraction
        if self.action_dim > 1:
            action[1] = 1.0   # comply
        return action


# ======================================================================
# Actor-Critic network (shared trunk)
# ======================================================================

class ActorCritic(nn.Module):
    """
    Shared trunk → actor head (Gaussian policy) + critic head (value fn).

    For continuous action spaces of dimension action_dim.
    """

    def __init__(self, obs_dim: int, action_dim: int, hidden: int = 128):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.LayerNorm(hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
        )
        self.actor_mean = nn.Linear(hidden, action_dim)
        self.actor_log_std = nn.Parameter(torch.zeros(action_dim))
        self.critic = nn.Linear(hidden, 1)

        # Orthogonal initialisation (common in RL)
        for layer in self.trunk:
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight, gain=np.sqrt(2))
                nn.init.zeros_(layer.bias)
        nn.init.orthogonal_(self.actor_mean.weight, gain=0.01)
        nn.init.orthogonal_(self.critic.weight, gain=1.0)

    def forward(self, obs: torch.Tensor):
        feat = self.trunk(obs)
        mean = torch.sigmoid(self.actor_mean(feat))  # ∈ (0,1)
        std = torch.exp(self.actor_log_std).clamp(1e-4, 1.0)
        value = self.critic(feat).squeeze(-1)
        return mean, std, value

    def evaluate(self, obs: torch.Tensor, actions: torch.Tensor):
        mean, std, value = self.forward(obs)
        dist = Normal(mean, std)
        log_prob = dist.log_prob(actions).sum(-1)
        entropy = dist.entropy().sum(-1)
        return log_prob, entropy, value


# ======================================================================
# PPO Agent
# ======================================================================

class RolloutBuffer:
    """Stores one episode of experience for a single agent."""

    def __init__(self):
        self.obs: List[np.ndarray] = []
        self.actions: List[np.ndarray] = []
        self.rewards: List[float] = []
        self.log_probs: List[float] = []
        self.values: List[float] = []
        self.dones: List[bool] = []

    def clear(self):
        self.__init__()

    def compute_returns_and_advantages(
        self, last_value: float, gamma: float = 0.99, lam: float = 0.95
    ) -> Tuple[np.ndarray, np.ndarray]:
        """GAE-Lambda returns."""
        T = len(self.rewards)
        advantages = np.zeros(T, dtype=np.float32)
        returns = np.zeros(T, dtype=np.float32)
        gae = 0.0
        values = np.array(self.values + [last_value], dtype=np.float32)
        for t in reversed(range(T)):
            delta = (
                self.rewards[t]
                + gamma * values[t + 1] * (1 - self.dones[t])
                - values[t]
            )
            gae = delta + gamma * lam * (1 - self.dones[t]) * gae
            advantages[t] = gae
            returns[t] = gae + values[t]
        return returns, advantages


class PPOAgent(BaseAgent):
    """
    Proximal Policy Optimisation agent with clipped objective.

    Hyperparameters follow Schulman et al. (2017) defaults.
    """

    def __init__(
        self,
        agent_id: int,
        obs_dim: int,
        action_dim: int,
        lr: float = 3e-4,
        clip_eps: float = 0.2,
        value_coeff: float = 0.5,
        entropy_coeff: float = 0.01,
        n_epochs: int = 4,
        hidden: int = 128,
        device: str = "cpu",
    ):
        super().__init__(agent_id, obs_dim, action_dim)
        self.clip_eps = clip_eps
        self.value_coeff = value_coeff
        self.entropy_coeff = entropy_coeff
        self.n_epochs = n_epochs
        self.device = torch.device(device)

        self.net = ActorCritic(obs_dim, action_dim, hidden).to(self.device)
        self.optimizer = optim.Adam(self.net.parameters(), lr=lr)
        self.buffer = RolloutBuffer()
        self._last_log_prob: float = 0.0
        self._last_value: float = 0.0

    def select_action(self, obs: np.ndarray, **kwargs) -> np.ndarray:
        obs_t = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            mean, std, value = self.net(obs_t)
            dist = Normal(mean, std)
            action = dist.sample()
            log_prob = dist.log_prob(action).sum(-1)

        action_np = action.squeeze(0).clamp(0, 1).cpu().numpy()
        self._last_log_prob = log_prob.item()
        self._last_value = value.item()
        return action_np

    def store(self, obs, action, reward, done):
        self.buffer.obs.append(obs)
        self.buffer.actions.append(action)
        self.buffer.rewards.append(reward)
        self.buffer.log_probs.append(self._last_log_prob)
        self.buffer.values.append(self._last_value)
        self.buffer.dones.append(done)

    def update(self, last_obs: Optional[np.ndarray] = None):
        if last_obs is not None:
            obs_t = torch.tensor(last_obs, dtype=torch.float32, device=self.device).unsqueeze(0)
            with torch.no_grad():
                _, _, last_val = self.net(obs_t)
            last_value = last_val.item()
        else:
            last_value = 0.0

        returns, advantages = self.buffer.compute_returns_and_advantages(last_value)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        obs_t = torch.tensor(np.array(self.buffer.obs), dtype=torch.float32, device=self.device)
        act_t = torch.tensor(np.array(self.buffer.actions), dtype=torch.float32, device=self.device)
        ret_t = torch.tensor(returns, dtype=torch.float32, device=self.device)
        adv_t = torch.tensor(advantages, dtype=torch.float32, device=self.device)
        old_lp_t = torch.tensor(np.array(self.buffer.log_probs), dtype=torch.float32, device=self.device)

        total_loss = 0.0
        for _ in range(self.n_epochs):
            log_prob, entropy, value = self.net.evaluate(obs_t, act_t)
            ratio = torch.exp(log_prob - old_lp_t)
            surr1 = ratio * adv_t
            surr2 = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * adv_t
            actor_loss = -torch.min(surr1, surr2).mean()
            critic_loss = (ret_t - value).pow(2).mean()
            entropy_loss = -entropy.mean()
            loss = actor_loss + self.value_coeff * critic_loss + self.entropy_coeff * entropy_loss

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
