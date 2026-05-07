"""
CommonsEnvironment
==================
Models a shared renewable resource pool subject to over-exploitation
(the classic Tragedy of the Commons). Agents observe the pool state
and choose extraction quantities; the pool regenerates logistically.

State  : [resource_level, agent_i_stock, ...consumption_history...]
Action : continuous extraction fraction in [0, 1]
Reward : individual gain - social collapse penalty
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
import gym
from gym import spaces


class CommonsEnvironment(gym.Env):
    """
    Shared renewable resource environment.

    Parameters
    ----------
    n_agents        : number of agents
    max_resource    : carrying capacity of the commons (K)
    regen_rate      : logistic growth rate r
    collapse_thresh : resource level below which collapse is declared
    max_steps       : episode length
    """

    metadata = {"render.modes": ["human"]}

    def __init__(
        self,
        n_agents: int = 5,
        max_resource: float = 100.0,
        regen_rate: float = 0.3,
        collapse_thresh: float = 5.0,
        max_steps: int = 200,
        seed: Optional[int] = None,
    ):
        super().__init__()
        self.n_agents = n_agents
        self.K = max_resource
        self.r = regen_rate
        self.collapse_thresh = collapse_thresh
        self.max_steps = max_steps
        self.rng = np.random.default_rng(seed)

        # Each agent observes: [resource_level / K, own_stock / K,
        #                        mean_last_extraction, step / max_steps]
        obs_dim = 4
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(obs_dim,), dtype=np.float32
        )
        # Fraction of sustainable yield to extract [0, 1]
        self.action_space = spaces.Box(
            low=0.0, high=1.0, shape=(1,), dtype=np.float32
        )

        self._resource = None
        self._stocks = None
        self._step_count = 0
        self._extraction_history: List[np.ndarray] = []

    # ------------------------------------------------------------------
    # Core gym interface
    # ------------------------------------------------------------------

    def reset(self) -> List[np.ndarray]:
        self._resource = self.K * 0.8  # start at 80 % capacity
        self._stocks = np.zeros(self.n_agents, dtype=np.float32)
        self._step_count = 0
        self._extraction_history = []
        return self._get_obs()

    def step(
        self, actions: List[np.ndarray]
    ) -> Tuple[List[np.ndarray], List[float], bool, dict]:
        """
        actions : list of length n_agents, each shape (1,) in [0, 1].
                  Fraction of the *sustainable yield* each agent takes.
        """
        actions_arr = np.array([np.clip(a, 0, 1).item() for a in actions])

        # Sustainable yield = r * R * (1 - R/K)
        sustainable_yield = self.r * self._resource * (1.0 - self._resource / self.K)
        # Each agent's demand
        demanded = actions_arr * sustainable_yield
        total_demanded = demanded.sum()

        # Cannot extract more than available resource
        if total_demanded > self._resource:
            scale = self._resource / (total_demanded + 1e-8)
            extracted = demanded * scale
        else:
            extracted = demanded.copy()

        self._resource = max(0.0, self._resource - extracted.sum())
        self._stocks += extracted
        self._extraction_history.append(extracted.copy())
        if len(self._extraction_history) > 10:
            self._extraction_history.pop(0)

        # Regeneration (logistic)
        growth = self.r * self._resource * (1.0 - self._resource / self.K)
        self._resource = min(self.K, self._resource + growth)

        # Collapse detection
        collapsed = self._resource < self.collapse_thresh

        # Rewards
        rewards = self._compute_rewards(extracted, collapsed)

        self._step_count += 1
        done = collapsed or (self._step_count >= self.max_steps)

        info = {
            "resource_level": self._resource,
            "extracted": extracted,
            "collapsed": collapsed,
            "sustainable_yield": sustainable_yield,
        }

        return self._get_obs(), rewards, done, info

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _compute_rewards(
        self, extracted: np.ndarray, collapsed: bool
    ) -> List[float]:
        """
        Reward_i = extracted_i  -  lambda * collapse_penalty
                                 -  mu    * inequality_term

        Social collapse destroys all future value: heavy one-time penalty.
        """
        lam = 50.0   # collapse weight
        mu = 0.1     # inequality aversion
        collapse_pen = lam if collapsed else 0.0
        mean_ext = extracted.mean() + 1e-8
        inequality = np.abs(extracted - mean_ext).mean() / mean_ext
        rewards = extracted - collapse_pen - mu * inequality
        return rewards.tolist()

    def _get_obs(self) -> List[np.ndarray]:
        mean_last = (
            np.mean(self._extraction_history[-1])
            if self._extraction_history
            else 0.0
        )
        base = np.array(
            [
                self._resource / self.K,
                mean_last / (self.K + 1e-8),
                self._step_count / self.max_steps,
            ],
            dtype=np.float32,
        )
        obs_list = []
        for i in range(self.n_agents):
            own_stock = self._stocks[i] / (self.K + 1e-8)
            obs_i = np.concatenate([[own_stock], base]).astype(np.float32)
            obs_list.append(obs_i)
        return obs_list

    def render(self, mode: str = "human"):
        print(
            f"Step {self._step_count:3d} | Resource: {self._resource:7.2f} / {self.K:.1f}"
        )

    def cooperation_rate(self) -> float:
        """
        Fraction of steps where total extraction ≤ sustainable yield.
        Computed from history; call after episode.
        """
        if not self._extraction_history:
            return 0.0
        sustainable = self.r * self.K / 4  # max sustainable yield ≈ rK/4
        rates = [
            1.0 if h.sum() <= sustainable else 0.0
            for h in self._extraction_history
        ]
        return float(np.mean(rates))
