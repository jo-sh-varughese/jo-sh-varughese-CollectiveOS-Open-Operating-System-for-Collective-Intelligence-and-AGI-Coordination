"""
metrics/core.py
===============
CollectiveOS-Bench Metric Suite.

Standard Metrics:
    cooperation_rate(...)            → CR  ∈ [0,1]
    social_welfare(...)              → SW  ∈ ℝ
    gini_coefficient(...)            → G   ∈ [0,1]
    stability_score(...)             → SS  ∈ [0,1]

Novel Metrics (key contribution):
    institutional_emergence_score()  → IES ∈ [0,1]
    collective_adaptation_rate()     → CAR ∈ [0,1]
    coordination_efficiency_index()  → CEI ∈ [0,1]

All functions operate on numpy arrays and return Python floats.
Units and formulas are documented inline.
"""

import numpy as np
from typing import List, Optional, Sequence


# ======================================================================
# Standard metrics
# ======================================================================

def cooperation_rate(
    actions: np.ndarray,
    threshold: float,
    axis: int = -1,
) -> float:
    """
    Cooperation Rate (CR).

    Fraction of agent-steps where an agent's chosen action is ≤ threshold.

        CR = (1 / N·T) · Σ_{i,t} 𝟙[a_{i,t} ≤ threshold]

    Parameters
    ----------
    actions   : array of shape (T, N) or (N,)
    threshold : cooperative action ceiling

    Returns
    -------
    CR ∈ [0, 1]
    """
    actions = np.asarray(actions, dtype=np.float32)
    return float(np.mean(actions <= threshold))


def social_welfare(rewards: np.ndarray) -> float:
    """
    Social Welfare (SW).

    Utilitarian sum of individual returns across agents and time.

        SW = Σ_{i=1}^{N} Σ_{t=1}^{T} r_{i,t}

    Parameters
    ----------
    rewards : array of shape (T, N) or (N,)

    Returns
    -------
    SW ∈ ℝ
    """
    return float(np.sum(rewards))


def gini_coefficient(values: np.ndarray) -> float:
    """
    Gini Coefficient (G) — inequality index.

    Standard formula using sorted values:

        G = (2 · Σ_{i=1}^{N} i · x_i) / (N · Σ x_i) - (N+1)/N

    where x_i are sorted in ascending order.

    Parameters
    ----------
    values : 1-D array of non-negative values (e.g. cumulative rewards)

    Returns
    -------
    G ∈ [0, 1]  (0 = perfect equality, 1 = maximum inequality)
    """
    values = np.asarray(values, dtype=np.float64).flatten()
    values = values[values >= 0]
    if values.sum() == 0:
        return 0.0
    values = np.sort(values)
    n = len(values)
    idx = np.arange(1, n + 1)
    g = (2 * (idx * values).sum()) / (n * values.sum()) - (n + 1) / n
    return float(np.clip(g, 0, 1))


def stability_score(
    resource_levels: Sequence[float],
    collapse_thresh: float,
    K: float,
) -> float:
    """
    Stability Score (SS).

    Measures how consistently the commons stayed above collapse and
    close to carrying capacity.

        SS = (1/T) · Σ_t max(0, R_t - collapse_thresh) / K

    Normalised to [0, 1].

    Parameters
    ----------
    resource_levels : sequence of resource levels over time
    collapse_thresh : minimum viable resource level
    K               : carrying capacity

    Returns
    -------
    SS ∈ [0, 1]
    """
    rl = np.asarray(resource_levels, dtype=np.float64)
    above = np.maximum(0.0, rl - collapse_thresh) / (K + 1e-8)
    return float(np.mean(above))


# ======================================================================
# Novel metrics
# ======================================================================

def institutional_emergence_score(
    adaptation_events: int,
    max_possible_adaptations: int,
    compliance_history: Sequence[float],
    collapsed: bool,
) -> float:
    """
    Institutional Emergence Score (IES).

    Captures whether a functioning institution emerged — one that was
    collectively adapted AND associated with high compliance, without
    triggering collapse.

        IES = A_rate · C̄ · (1 - collapse)

    where
        A_rate = min(adaptation_events / max_possible_adaptations, 1)
        C̄      = mean compliance rate over the episode

    Parameters
    ----------
    adaptation_events          : number of accepted rule changes
    max_possible_adaptations   : maximum feasible changes (e.g. T // 10)
    compliance_history         : per-step mean compliance rates
    collapsed                  : whether the commons collapsed

    Returns
    -------
    IES ∈ [0, 1]
    """
    if max_possible_adaptations <= 0:
        return 0.0
    a_rate = min(adaptation_events / max_possible_adaptations, 1.0)
    c_bar = float(np.mean(compliance_history)) if len(compliance_history) > 0 else 0.0
    collapse_ind = 1.0 if collapsed else 0.0
    return float(a_rate * c_bar * (1.0 - collapse_ind))


def collective_adaptation_rate(
    rule_change_log: List[dict],
    max_magnitude: float = 1.0,
) -> float:
    """
    Collective Adaptation Rate (CAR).

    Measures how effectively agents updated their institution over time,
    weighting each change by democratic legitimacy (vote fraction) and
    magnitude.

        CAR = (1 / |Δ|) · Σ_{k∈Δ} |Δthreshold_k| · vote_frac_k

    where Δ is the set of accepted rule-change events.

    Parameters
    ----------
    rule_change_log  : list of dicts with keys 'old', 'new', 'vote_frac'
    max_magnitude    : normalisation constant (default 1.0)

    Returns
    -------
    CAR ∈ [0, 1]
    """
    if not rule_change_log:
        return 0.0
    scores = [
        abs(ev["new"] - ev["old"]) * ev.get("vote_frac", 1.0)
        for ev in rule_change_log
    ]
    raw = float(np.mean(scores))
    return float(np.clip(raw / (max_magnitude + 1e-8), 0, 1))


def coordination_efficiency_index(
    actual_social_welfare: float,
    optimal_social_welfare: float,
    nash_social_welfare: float,
) -> float:
    """
    Coordination Efficiency Index (CEI).

    Normalises actual performance between the Nash equilibrium baseline
    (fully selfish, zero coordination) and the social optimum.

        CEI = (SW_actual - SW_nash) / (SW_optimal - SW_nash + ε)

    CEI = 0  ↔  agents achieved no better than Nash
    CEI = 1  ↔  agents achieved the social optimum

    Parameters
    ----------
    actual_social_welfare  : Σ r_i across agents and episode
    optimal_social_welfare : theoretical maximum (cooperative)
    nash_social_welfare    : expected value under Nash play

    Returns
    -------
    CEI ∈ [0, 1]  (clipped)
    """
    denom = optimal_social_welfare - nash_social_welfare
    if abs(denom) < 1e-8:
        return 1.0 if actual_social_welfare >= optimal_social_welfare else 0.0
    cei = (actual_social_welfare - nash_social_welfare) / denom
    return float(np.clip(cei, 0.0, 1.0))


# ======================================================================
# Aggregate summary
# ======================================================================

class MetricSummary:
    """
    Convenience wrapper: accumulates episode data and computes all
    metrics at episode end.
    """

    def __init__(
        self,
        n_agents: int,
        action_threshold: float = 0.5,
        collapse_thresh: float = 5.0,
        K: float = 100.0,
        optimal_sw: float = 1000.0,
        nash_sw: float = 100.0,
    ):
        self.n_agents = n_agents
        self.action_threshold = action_threshold
        self.collapse_thresh = collapse_thresh
        self.K = K
        self.optimal_sw = optimal_sw
        self.nash_sw = nash_sw

        self._actions: List[np.ndarray] = []
        self._rewards: List[np.ndarray] = []
        self._resource_levels: List[float] = []
        self._cumulative_rewards = np.zeros(n_agents)

    def record(
        self,
        actions: List[np.ndarray],
        rewards: List[float],
        resource_level: float,
    ):
        self._actions.append(np.array(actions).flatten())
        self._rewards.append(np.array(rewards))
        self._resource_levels.append(resource_level)
        self._cumulative_rewards += np.array(rewards)

    def compute(
        self,
        adaptation_events: int = 0,
        max_adaptations: int = 30,
        compliance_history: Optional[List[float]] = None,
        rule_change_log: Optional[List[dict]] = None,
        collapsed: bool = False,
    ) -> dict:
        if compliance_history is None:
            compliance_history = []
        if rule_change_log is None:
            rule_change_log = []

        all_actions = np.concatenate(self._actions)
        all_rewards = np.vstack(self._rewards)

        sw = social_welfare(all_rewards)

        return {
            "cooperation_rate": cooperation_rate(all_actions, self.action_threshold),
            "social_welfare": sw,
            "gini_coefficient": gini_coefficient(self._cumulative_rewards),
            "stability_score": stability_score(
                self._resource_levels, self.collapse_thresh, self.K
            ),
            "institutional_emergence_score": institutional_emergence_score(
                adaptation_events, max_adaptations, compliance_history, collapsed
            ),
            "collective_adaptation_rate": collective_adaptation_rate(rule_change_log),
            "coordination_efficiency_index": coordination_efficiency_index(
                sw, self.optimal_sw, self.nash_sw
            ),
        }
