"""
InstitutionEnvironment
======================
The KEY NOVELTY of CollectiveOS-Bench.

Agents operate under a dynamic rule system (the "institution").  Rules
govern resource allocation, permitted actions, and sanctioning.  Agents
may comply, deviate, or — once per episode — propose a rule change.
Governance quality emerges from the interplay between rule quality,
compliance, enforcement, and collective adaptation.

State  (per agent, dim=10):
  [resource_frac, own_stock_frac, n_violations_norm, compliance_rate,
   current_rule_threshold, sanction_level, proposal_pending,
   step_frac, public_good_level, institutional_trust]

Action (per agent, 4 dims):
  [extraction_frac,      ∈ [0,1]
   comply_flag,          ∈ {0,1}   (follow the rule or deviate)
   propose_rule_change,  ∈ {0,1}   (propose new threshold this step)
   new_rule_value]       ∈ [0,1]   (only used when propose=1)

Reward:
  extraction gain
  - sanction (if non-compliant and caught)
  + public_good_bonus * public_good_level
  + institutional_trust_bonus

Novel metrics supported:
  Institutional Emergence Score (IES)
  Collective Adaptation Rate   (CAR)
  Coordination Efficiency Index (CEI)
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
import gym
from gym import spaces
from collections import deque


class Rule:
    """
    A single institutional rule.

    threshold    : max fraction of sustainable yield any agent may extract.
    sanction     : fraction of extraction forfeited if caught violating.
    enforcement  : probability that a violation is detected.
    """

    def __init__(
        self,
        threshold: float = 0.5,
        sanction: float = 0.3,
        enforcement: float = 0.7,
    ):
        self.threshold = float(np.clip(threshold, 0.05, 1.0))
        self.sanction = float(np.clip(sanction, 0.0, 1.0))
        self.enforcement = float(np.clip(enforcement, 0.0, 1.0))

    def __repr__(self):
        return (
            f"Rule(threshold={self.threshold:.2f}, "
            f"sanction={self.sanction:.2f}, "
            f"enforcement={self.enforcement:.2f})"
        )


class InstitutionEnvironment(gym.Env):
    metadata = {"render.modes": ["human"]}

    # Voting: majority required to adopt a rule proposal
    VOTE_THRESHOLD = 0.5

    def __init__(
        self,
        n_agents: int = 6,
        max_resource: float = 100.0,
        regen_rate: float = 0.3,
        collapse_thresh: float = 5.0,
        max_steps: int = 300,
        initial_rule: Optional[Rule] = None,
        allow_rule_proposals: bool = True,
        seed: Optional[int] = None,
    ):
        super().__init__()
        self.n_agents = n_agents
        self.K = max_resource
        self.r = regen_rate
        self.collapse_thresh = collapse_thresh
        self.max_steps = max_steps
        self.allow_proposals = allow_rule_proposals
        self.rng = np.random.default_rng(seed)

        self.initial_rule = initial_rule or Rule(0.5, 0.3, 0.7)
        self.current_rule: Rule = Rule()
        self._resource = 0.0
        self._stocks = np.zeros(n_agents)
        self._step_count = 0
        self._violations = np.zeros(n_agents, dtype=int)
        self._compliance_history: deque = deque(maxlen=50)
        self._institutional_trust = 0.5
        self._public_good = 0.0
        self._rule_change_log: List[dict] = []
        self._adaptation_events = 0

        obs_dim = 10
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(obs_dim,), dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32),
            high=np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32),
        )

    # ------------------------------------------------------------------
    # Gym interface
    # ------------------------------------------------------------------

    def reset(self) -> List[np.ndarray]:
        self._resource = self.K * 0.8
        self._stocks = np.zeros(self.n_agents, dtype=np.float32)
        self._violations = np.zeros(self.n_agents, dtype=int)
        self._step_count = 0
        self._compliance_history.clear()
        self._institutional_trust = 0.5
        self._public_good = 0.0
        self._rule_change_log = []
        self._adaptation_events = 0
        self.current_rule = Rule(
            self.initial_rule.threshold,
            self.initial_rule.sanction,
            self.initial_rule.enforcement,
        )
        return self._get_obs()

    def step(
        self, actions: List[np.ndarray]
    ) -> Tuple[List[np.ndarray], List[float], bool, dict]:
        """
        actions : list of n_agents arrays, each shape (4,).
        """
        actions_arr = np.array([np.clip(a, 0, 1) for a in actions])

        extraction_fracs = actions_arr[:, 0]
        comply_flags = (actions_arr[:, 1] > 0.5).astype(bool)
        propose_flags = (actions_arr[:, 2] > 0.5).astype(bool) if self.allow_proposals else np.zeros(self.n_agents, dtype=bool)
        new_rule_vals = actions_arr[:, 3]

        # --- Rule proposals & democratic voting ---
        proposers = np.where(propose_flags)[0]
        if len(proposers) > 0:
            # Average of proposals weighted by compliance history
            prop_threshold = np.mean(new_rule_vals[proposers])
            votes_in_favour = np.sum(
                [1 for i in range(self.n_agents) if comply_flags[i]]
            )
            vote_frac = votes_in_favour / self.n_agents
            if vote_frac >= self.VOTE_THRESHOLD:
                old_rule = Rule(
                    self.current_rule.threshold,
                    self.current_rule.sanction,
                    self.current_rule.enforcement,
                )
                # Adapt threshold toward proposed value
                self.current_rule.threshold = (
                    0.6 * self.current_rule.threshold + 0.4 * prop_threshold
                )
                self._rule_change_log.append(
                    {
                        "step": self._step_count,
                        "old": old_rule.threshold,
                        "new": self.current_rule.threshold,
                        "vote_frac": vote_frac,
                    }
                )
                self._adaptation_events += 1

        # --- Extraction ---
        sustainable_yield = (
            self.r * self._resource * (1.0 - self._resource / self.K)
        )
        rule_max_frac = self.current_rule.threshold

        effective_frac = np.where(
            comply_flags,
            np.minimum(extraction_fracs, rule_max_frac),
            extraction_fracs,           # deviators ignore the cap
        )
        demanded = effective_frac * sustainable_yield
        total_demanded = demanded.sum()
        if total_demanded > self._resource:
            scale = self._resource / (total_demanded + 1e-8)
            extracted = demanded * scale
        else:
            extracted = demanded.copy()

        self._resource = max(0.0, self._resource - extracted.sum())

        # --- Sanctioning ---
        is_violator = (~comply_flags) & (extraction_fracs > rule_max_frac)
        caught = (
            is_violator
            & (self.rng.random(self.n_agents) < self.current_rule.enforcement)
        )
        sanctions = np.where(caught, extracted * self.current_rule.sanction, 0.0)
        net_extracted = extracted - sanctions

        self._stocks += net_extracted
        self._violations += is_violator.astype(int)

        # --- Compliance & trust update ---
        compliance_rate = comply_flags.mean()
        self._compliance_history.append(compliance_rate)
        self._institutional_trust = (
            0.95 * self._institutional_trust
            + 0.05 * (compliance_rate - is_violator.mean())
        )
        self._institutional_trust = float(
            np.clip(self._institutional_trust, 0, 1)
        )

        # --- Public good (taxes + fines fund a shared pool) ---
        self._public_good += sanctions.sum() * 0.5
        self._public_good *= 0.98  # slow decay

        # --- Regeneration ---
        growth = self.r * self._resource * (1.0 - self._resource / self.K)
        self._resource = min(self.K, self._resource + growth)

        collapsed = self._resource < self.collapse_thresh

        rewards = self._compute_rewards(
            net_extracted, sanctions, caught, collapsed
        )

        self._step_count += 1
        done = collapsed or (self._step_count >= self.max_steps)

        info = {
            "resource_level": self._resource,
            "compliance_rate": compliance_rate,
            "violations": is_violator.sum(),
            "caught": caught.sum(),
            "current_rule": str(self.current_rule),
            "institutional_trust": self._institutional_trust,
            "adaptation_events": self._adaptation_events,
            "collapsed": collapsed,
        }

        return self._get_obs(), rewards, done, info

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _compute_rewards(
        self,
        net_extracted: np.ndarray,
        sanctions: np.ndarray,
        caught: np.ndarray,
        collapsed: bool,
    ) -> List[float]:
        collapse_pen = 100.0 if collapsed else 0.0
        pg_bonus = self._public_good / (self.K + 1e-8)
        trust_bonus = self._institutional_trust * 0.5
        rewards = (
            net_extracted
            - collapse_pen
            + pg_bonus
            + trust_bonus
        )
        return rewards.tolist()

    def _get_obs(self) -> List[np.ndarray]:
        comp_rate = (
            np.mean(self._compliance_history)
            if self._compliance_history
            else 0.5
        )
        obs = []
        for i in range(self.n_agents):
            o = np.array(
                [
                    self._resource / self.K,
                    self._stocks[i] / (self.K * self.max_steps + 1e-8),
                    min(self._violations[i] / 10.0, 1.0),
                    comp_rate,
                    self.current_rule.threshold,
                    self.current_rule.sanction,
                    1.0 if self.allow_proposals else 0.0,
                    self._step_count / self.max_steps,
                    min(self._public_good / (self.K + 1e-8), 1.0),
                    self._institutional_trust,
                ],
                dtype=np.float32,
            )
            obs.append(o)
        return obs

    def render(self, mode: str = "human"):
        comp = (
            np.mean(self._compliance_history) if self._compliance_history else 0.0
        )
        print(
            f"Step {self._step_count:3d} | Resource: {self._resource:6.2f} "
            f"| Rule: {self.current_rule} "
            f"| Compliance: {comp:.2%} "
            f"| Trust: {self._institutional_trust:.2f}"
        )

    # ------------------------------------------------------------------
    # Novel metric helpers
    # ------------------------------------------------------------------

    def institutional_emergence_score(self) -> float:
        """
        IES = (adaptation_events / max_possible_adaptations)
              * mean_compliance
              * (1 - collapse_indicator)

        Captures whether meaningful rule adaptation occurred AND was
        associated with high compliance.
        """
        max_adapt = max(1, self.max_steps // 10)
        adapt_rate = min(self._adaptation_events / max_adapt, 1.0)
        mean_comp = (
            float(np.mean(self._compliance_history))
            if self._compliance_history
            else 0.0
        )
        collapse_ind = 1.0 if self._resource < self.collapse_thresh else 0.0
        ies = adapt_rate * mean_comp * (1.0 - collapse_ind)
        return float(ies)

    def collective_adaptation_rate(self) -> float:
        """
        CAR = |rule_change_log| weighted by vote_frac and magnitude.

        Measures how effectively the collective updated its institution.
        """
        if not self._rule_change_log:
            return 0.0
        scores = []
        for ev in self._rule_change_log:
            magnitude = abs(ev["new"] - ev["old"])
            scores.append(magnitude * ev["vote_frac"])
        return float(np.mean(scores))
