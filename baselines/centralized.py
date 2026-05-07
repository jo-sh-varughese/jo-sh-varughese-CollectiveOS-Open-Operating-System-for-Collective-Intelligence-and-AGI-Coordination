"""
baselines/centralized.py
========================
Centralised Controller Baseline.

A single omniscient planner observes the full joint state and outputs
actions for all agents simultaneously.  This is the strongest possible
baseline — it provides the social optimum reference point for CEI.

Two variants:
    CentralizedOptimalController  — analytic optimal policy (Commons only)
    CentralizedPPOController      — single PPO net trained on joint state
"""

from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
from typing import List, Optional, Tuple


# ======================================================================
# Analytic optimal controller (Commons environment)
# ======================================================================

class CentralizedOptimalController:
    """
    Analytically derived optimal controller for the Commons environment.

    Optimum strategy: each agent extracts 1/N of the sustainable yield,
    distributed equally to maximise social welfare while preserving the
    resource above collapse threshold.

    This provides the true upper bound (social optimum) for CEI
    computation.
    """

    def __init__(self, n_agents: int, regen_rate: float = 0.3, K: float = 100.0):
        self.n_agents = n_agents
        self.r = regen_rate
        self.K = K

    def act(self, obs_list: List[np.ndarray], env_state: dict) -> List[np.ndarray]:
        """
        obs_list  : per-agent observations (unused; uses env_state)
        env_state : must contain 'resource_level'
        """
        R = env_state.get("resource_level", self.K * 0.8)
        # Optimal: share sustainable yield equally, take slightly below
        # to maintain resource above collapse threshold.
        sustainable_yield = self.r * R * (1.0 - R / self.K)
        # Conservative fraction — each agent gets equal share at 90% of optimum
        per_agent_frac = min(0.9 / self.n_agents, 1.0)
        action = np.array([per_agent_frac], dtype=np.float32)
        return [action.copy() for _ in range(self.n_agents)]

    def optimal_social_welfare_estimate(self, T: int) -> float:
        """
        Estimate total social welfare under optimal play.
        Used as SW_optimal in CEI calculation.
        """
        R = self.K * 0.8
        total = 0.0
        for _ in range(T):
            sy = self.r * R * (1.0 - R / self.K)
            extracted_total = 0.9 * sy
            total += extracted_total
            R = min(self.K, max(0.0, R - extracted_total + sy))
        return total


# ======================================================================
# Centralised PPO (joint-state, joint-action)
# ======================================================================

class JointActorCritic(nn.Module):
    """
    Actor-critic network operating on the concatenated joint observation.
    Outputs actions for all N agents simultaneously.

    obs_dim_total = n_agents * obs_dim_per_agent
    act_dim_total = n_agents * action_dim_per_agent
    """

    def __init__(
        self,
        joint_obs_dim: int,
        joint_act_dim: int,
        hidden: int = 256,
    ):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(joint_obs_dim, hidden),
            nn.LayerNorm(hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden // 2),
            nn.Tanh(),
        )
        self.actor_mean = nn.Linear(hidden // 2, joint_act_dim)
        self.actor_log_std = nn.Parameter(torch.zeros(joint_act_dim))
        self.critic = nn.Linear(hidden // 2, 1)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.zeros_(m.bias)
        nn.init.orthogonal_(self.actor_mean.weight, gain=0.01)

    def forward(self, joint_obs: torch.Tensor):
        feat = self.trunk(joint_obs)
        mean = torch.sigmoid(self.actor_mean(feat))
        std = self.actor_log_std.exp().clamp(1e-4, 1.0)
        value = self.critic(feat).squeeze(-1)
        return mean, std, value


class CentralizedPPOController:
    """
    Single PPO agent operating on the joint state-action space.

    Centralised training with centralised execution (CTCE).
    Provides upper bound for what PPO-class algorithms can achieve
    with full observability.
    """

    def __init__(
        self,
        n_agents: int,
        obs_dim: int,
        action_dim: int,
        lr: float = 3e-4,
        clip_eps: float = 0.2,
        value_coeff: float = 0.5,
        entropy_coeff: float = 0.01,
        n_epochs: int = 4,
        hidden: int = 256,
        device: str = "cpu",
    ):
        self.n_agents = n_agents
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.clip_eps = clip_eps
        self.value_coeff = value_coeff
        self.entropy_coeff = entropy_coeff
        self.n_epochs = n_epochs
        self.device = torch.device(device)

        self.net = JointActorCritic(
            n_agents * obs_dim, n_agents * action_dim, hidden
        ).to(self.device)
        self.optimizer = optim.Adam(self.net.parameters(), lr=lr)

        # Rollout buffer (joint)
        self._obs_buf: List[np.ndarray] = []
        self._act_buf: List[np.ndarray] = []
        self._rew_buf: List[float] = []
        self._lp_buf: List[float] = []
        self._val_buf: List[float] = []
        self._done_buf: List[bool] = []
        self._last_lp: float = 0.0
        self._last_val: float = 0.0

    def act(
        self, obs_list: List[np.ndarray], env_state: Optional[dict] = None
    ) -> List[np.ndarray]:
        joint_obs = np.concatenate(obs_list).astype(np.float32)
        obs_t = torch.tensor(joint_obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            mean, std, value = self.net(obs_t)
            dist = Normal(mean, std)
            action = dist.sample()
            lp = dist.log_prob(action).sum(-1)

        joint_action = action.squeeze(0).clamp(0, 1).cpu().numpy()
        self._last_lp = lp.item()
        self._last_val = value.item()

        # Split joint action back to per-agent
        actions = [
            joint_action[i * self.action_dim: (i + 1) * self.action_dim]
            for i in range(self.n_agents)
        ]
        return actions

    def store(
        self, obs_list: List[np.ndarray], actions: List[np.ndarray],
        reward: float, done: bool
    ):
        self._obs_buf.append(np.concatenate(obs_list))
        self._act_buf.append(np.concatenate(actions))
        self._rew_buf.append(reward)
        self._lp_buf.append(self._last_lp)
        self._val_buf.append(self._last_val)
        self._done_buf.append(done)

    def update(self, last_obs_list: Optional[List[np.ndarray]] = None) -> float:
        if last_obs_list is not None:
            joint = np.concatenate(last_obs_list).astype(np.float32)
            obs_t = torch.tensor(joint, device=self.device).unsqueeze(0)
            with torch.no_grad():
                _, _, last_val = self.net(obs_t)
            last_value = last_val.item()
        else:
            last_value = 0.0

        # GAE
        T = len(self._rew_buf)
        advantages = np.zeros(T, dtype=np.float32)
        returns = np.zeros(T, dtype=np.float32)
        values = np.array(self._val_buf + [last_value], dtype=np.float32)
        gae = 0.0
        gamma, lam = 0.99, 0.95
        for t in reversed(range(T)):
            delta = self._rew_buf[t] + gamma * values[t+1] * (1 - self._done_buf[t]) - values[t]
            gae = delta + gamma * lam * (1 - self._done_buf[t]) * gae
            advantages[t] = gae
            returns[t] = gae + values[t]

        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        obs_t = torch.tensor(np.array(self._obs_buf), dtype=torch.float32, device=self.device)
        act_t = torch.tensor(np.array(self._act_buf), dtype=torch.float32, device=self.device)
        ret_t = torch.tensor(returns, dtype=torch.float32, device=self.device)
        adv_t = torch.tensor(advantages, dtype=torch.float32, device=self.device)
        old_lp = torch.tensor(np.array(self._lp_buf), dtype=torch.float32, device=self.device)

        total_loss = 0.0
        for _ in range(self.n_epochs):
            mean, std, value = self.net(obs_t)
            dist = Normal(mean, std)
            lp = dist.log_prob(act_t).sum(-1)
            entropy = dist.entropy().sum(-1)
            ratio = torch.exp(lp - old_lp)
            surr = torch.min(ratio * adv_t,
                             torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * adv_t)
            loss = (-surr.mean()
                    + self.value_coeff * (ret_t - value).pow(2).mean()
                    - self.entropy_coeff * entropy.mean())
            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.net.parameters(), 0.5)
            self.optimizer.step()
            total_loss += loss.item()

        # Clear buffer
        self._obs_buf.clear(); self._act_buf.clear(); self._rew_buf.clear()
        self._lp_buf.clear(); self._val_buf.clear(); self._done_buf.clear()
        return total_loss / self.n_epochs

    def save(self, path: str):
        torch.save(self.net.state_dict(), path)

    def load(self, path: str):
        self.net.load_state_dict(torch.load(path, map_location=self.device))
