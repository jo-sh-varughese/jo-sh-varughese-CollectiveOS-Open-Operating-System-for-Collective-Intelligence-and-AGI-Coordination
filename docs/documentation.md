# CollectiveOS-Bench — Documentation

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Environment API Reference](#environment-api-reference)
3. [Agent API Reference](#agent-api-reference)
4. [Metrics Reference](#metrics-reference)
5. [Experiment Configuration](#experiment-configuration)
6. [Adding New Environments](#adding-new-environments)
7. [Adding New Agents](#adding-new-agents)
8. [API Reference (REST)](#api-reference-rest)
9. [Reproducibility Guide](#reproducibility-guide)
10. [Troubleshooting](#troubleshooting)

---

## 1. Architecture Overview

```
CollectiveOS-Bench
│
├── environments/          Gym-compatible multi-agent environments
│   ├── CommonsEnvironment     Renewable resource dilemma
│   ├── MarketEnvironment      Decentralised double-auction market
│   └── InstitutionEnvironment Dynamic governance (KEY NOVELTY)
│
├── agents/                Agent base classes + PPO implementation
│   ├── BaseAgent              Abstract interface
│   ├── PPOAgent               Independent PPO (PyTorch)
│   ├── GreedyAgent            Always-defect baseline
│   ├── CooperativeAgent       Fixed cooperative policy
│   └── RandomAgent            Uniform random baseline
│
├── baselines/             Extended baselines
│   ├── CentralizedOptimalController   Analytic optimum (Commons)
│   ├── CentralizedPPOController       Joint-state PPO (CTCE)
│   └── CommNetAgent / CommNetGroup    Communication baseline
│
├── metrics/               Metric implementations
│   ├── Standard:  CR, SW, Gini, SS
│   └── Novel:     IES, CAR, CEI
│
├── experiments/           Experiment management
│   ├── ExperimentRunner       Standard runner (independent agents)
│   ├── CentralizedRunner      CTCE runner
│   └── CommNetRunner          CommNet runner
│
├── api/                   FastAPI REST layer
├── plots/                 Matplotlib visualisation utilities
└── reproducibility/       Fixed-seed reproduce scripts
```

**Data flow per episode:**

```
env.reset() → obs_list (per-agent)
    ↓
agent.select_action(obs_i) → action_i   [for each agent i]
    ↓
env.step(actions) → (obs_next, rewards, done, info)
    ↓
agent.store(obs, action, reward, done)  [for PPO-class agents]
    ↓
[if done] agent.update() → gradient step
    ↓
MetricSummary.record(actions, rewards, resource)
    ↓
[end episode] MetricSummary.compute() → metrics dict
```

---

## 2. Environment API Reference

All environments implement the `gym.Env` interface, extended to return
**lists** of per-agent observations/rewards instead of scalars.

### CommonsEnvironment

```python
env = CommonsEnvironment(
    n_agents=5,           # number of agents
    max_resource=100.0,   # carrying capacity K
    regen_rate=0.3,       # logistic growth rate r
    collapse_thresh=5.0,  # resource below this → collapse
    max_steps=200,        # episode length
    seed=42,              # RNG seed
)
```

**Observation** (per agent, shape `(4,)`):
```
[own_stock/K, resource/K, mean_last_extraction/K, step/max_steps]
```

**Action** (per agent, shape `(1,)`):
```
extraction_fraction ∈ [0, 1]  (fraction of sustainable yield to take)
```

**Info dict keys:**
- `resource_level` — current resource level
- `extracted` — array of per-agent extractions
- `collapsed` — bool
- `sustainable_yield` — r·R·(1 - R/K)

**Additional methods:**
```python
env.cooperation_rate()  → float  # fraction of steps with total extraction ≤ sustainable yield
```

---

### MarketEnvironment

```python
env = MarketEnvironment(
    n_agents=6,
    init_inv=50.0,    # initial inventory per good per agent
    init_cash=200.0,  # initial cash per agent
    max_steps=300,
    seed=42,
)
```

**Observation** (per agent, shape `(6,)`):
```
[inv_A/inv_max, inv_B/inv_max, price_A/p_max, price_B/p_max,
 cash/cash_max, step/max_steps]
```

**Action** (per agent, shape `(4,)`):
```
[offer_type ∈ {0=hold, 1=buy, 2=sell},
 good ∈ {0=A, 1=B},
 quantity_fraction ∈ [0,1],
 price_fraction ∈ [0,1]]   (maps to [0.5·ref, 2.0·ref])
```

**Info dict keys:**
- `prices` — current reference prices array `[p_A, p_B]`
- `n_trades` — trades cleared this step
- `total_volume` — total quantity traded

```python
env.market_efficiency()  → float  # VWAP convergence proxy
```

---

### InstitutionEnvironment

```python
env = InstitutionEnvironment(
    n_agents=6,
    max_resource=100.0,
    regen_rate=0.3,
    collapse_thresh=5.0,
    max_steps=300,
    initial_rule=Rule(threshold=0.5, sanction=0.3, enforcement=0.7),
    allow_rule_proposals=True,
    seed=42,
)
```

**Observation** (per agent, shape `(10,)`):
```
[resource/K, own_stock/(K·T), violations/10, compliance_rate,
 rule_threshold, rule_sanction, proposals_allowed,
 step/T, public_good/K, institutional_trust]
```

**Action** (per agent, shape `(4,)`):
```
[extraction_fraction ∈ [0,1],
 comply_flag ∈ {0,1},
 propose_rule_change ∈ {0,1},
 new_rule_value ∈ [0,1]]
```

**Info dict keys:**
- `resource_level`, `compliance_rate`, `violations`, `caught`
- `current_rule` — string representation
- `institutional_trust` — running trust score
- `adaptation_events` — cumulative rule changes accepted

**Novel metric methods:**
```python
env.institutional_emergence_score()  → float   # IES
env.collective_adaptation_rate()     → float   # CAR
```

---

## 3. Agent API Reference

### BaseAgent (abstract)

```python
class BaseAgent:
    def select_action(self, obs: np.ndarray, **kwargs) -> np.ndarray: ...
    def update(self, *args, **kwargs): ...
    def reset(self): ...
```

### PPOAgent

```python
agent = PPOAgent(
    agent_id=0,
    obs_dim=10,
    action_dim=4,
    lr=3e-4,           # Adam learning rate
    clip_eps=0.2,      # PPO clip epsilon
    value_coeff=0.5,   # value loss weight
    entropy_coeff=0.01,# entropy bonus weight
    n_epochs=4,        # gradient epochs per update
    hidden=128,        # hidden layer width
    device="cpu",
)

action = agent.select_action(obs)
agent.store(obs, action, reward, done)
loss = agent.update(last_obs=final_obs)   # call at episode end
agent.save("checkpoints/agent_0.pt")
agent.load("checkpoints/agent_0.pt")
```

### CommNetGroup

```python
group = CommNetGroup(n_agents=5, obs_dim=10, action_dim=4, msg_dim=16)
actions = group.act(obs_list)            # message + action in one call
group.store(obs_list, actions, rewards, dones)
group.update(last_obs_list)
group.reset()
```

---

## 4. Metrics Reference

### Standard Metrics

| Metric | Symbol | Range | Formula |
|--------|--------|-------|---------|
| Cooperation Rate | CR | [0,1] | `mean(actions ≤ threshold)` |
| Social Welfare | SW | ℝ | `sum(all rewards)` |
| Gini Coefficient | G | [0,1] | Lorenz-based inequality |
| Stability Score | SS | [0,1] | `mean(max(0, R-R_min) / K)` |

### Novel Metrics

| Metric | Symbol | Range | Key Property |
|--------|--------|-------|--------------|
| Institutional Emergence Score | IES | [0,1] | Zero on collapse; requires both adaptation AND compliance |
| Collective Adaptation Rate | CAR | [0,1] | Weighted by vote legitimacy and magnitude |
| Coordination Efficiency Index | CEI | [0,1] | Normalised between Nash and social optimum |

### Usage

```python
from metrics.core import MetricSummary

summary = MetricSummary(n_agents=5, K=100.0)

# During episode:
summary.record(actions, rewards, resource_level)

# At episode end:
metrics = summary.compute(
    adaptation_events=env._adaptation_events,
    max_adaptations=30,
    compliance_history=list(env._compliance_history),
    rule_change_log=env._rule_change_log,
    collapsed=info["collapsed"],
)
# Returns dict with all 7 metric values
```

---

## 5. Experiment Configuration

YAML config fields:

```yaml
name: my_experiment         # experiment name (used for save directory)
environment: institution    # commons | market | institution
env_kwargs:                 # passed to environment __init__
  n_agents: 6
  max_resource: 100.0
  allow_rule_proposals: true
agent_type: ppo             # random | greedy | cooperative | ppo
agent_kwargs:               # passed to agent __init__
  lr: 0.0003
  hidden: 128
n_episodes: 500             # training episodes
max_steps: 300              # steps per episode
seed: 42                    # RNG seed (fixed for reproducibility)
log_interval: 25            # log every N episodes
save_dir: results           # results output directory
device: cpu                 # cpu | cuda
```

Run from command line:
```bash
python -m experiments.runner --config configs/institution_ppo.yaml
```

---

## 6. Adding New Environments

1. Create `environments/myenv.py` inheriting from `gym.Env`.
2. Implement `reset()`, `step()`, `_get_obs()`, `render()`.
3. Return **lists** (one per agent) from `reset()` and `step()`.
4. Register in `environments/__init__.py`.
5. Add to `ENV_MAP` in `experiments/runner.py`.

Minimum required:
```python
class MyEnvironment(gym.Env):
    def __init__(self, n_agents, max_steps, seed=None, **kwargs):
        self.n_agents = n_agents
        self.observation_space = spaces.Box(...)
        self.action_space = spaces.Box(...)

    def reset(self) -> List[np.ndarray]: ...
    def step(self, actions) -> Tuple[List, List, bool, dict]: ...
```

---

## 7. Adding New Agents

Subclass `BaseAgent`:

```python
from agents.base import BaseAgent
import numpy as np

class MyAgent(BaseAgent):
    def select_action(self, obs: np.ndarray, **kwargs) -> np.ndarray:
        # Your policy here
        return np.zeros(self.action_dim, dtype=np.float32)

    def update(self, **kwargs):
        # Optional gradient update
        pass
```

Register in `AGENT_MAP` in `experiments/runner.py`.

---

## 8. API Reference (REST)

Start server:
```bash
uvicorn api.app:app --reload --port 8000
```

**Endpoints:**

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Health check |
| POST | `/experiments/run` | Launch experiment (async) |
| GET | `/experiments/{name}` | Fetch full results JSON |
| GET | `/experiments/{name}/metrics` | Fetch aggregated metrics only |
| GET | `/experiments` | List all experiments and statuses |

**Example POST body:**
```json
{
  "name": "my_run",
  "environment": "institution",
  "env_kwargs": {"n_agents": 6, "allow_rule_proposals": true},
  "agent_type": "ppo",
  "agent_kwargs": {"lr": 0.0003},
  "n_episodes": 200,
  "max_steps": 300,
  "seed": 42
}
```

---

## 9. Reproducibility Guide

All results in the paper can be reproduced with:

```bash
python reproducibility/reproduce_all.py
```

This script:
1. Runs all 3 environments × 4 baselines × 5 seeds
2. Saves per-run results to `results/{name}/results.json`
3. Aggregates across seeds to `results/all_results_summary.json`
4. Prints the summary table matching paper Table 1

For a quick test (30 episodes):
```bash
python reproducibility/reproduce_all.py --quick
```

**Seed specification:** Seeds 42, 123, 2024, 7, 999 are used throughout.  
Setting `PYTHONHASHSEED=<seed>` is handled automatically by the runner.

---

## 10. Troubleshooting

**`ModuleNotFoundError: No module named 'environments'`**  
→ Run from the `collectiveos-bench/` root directory, or `pip install -e .`

**`gym.error.Error: Box bound error`**  
→ Ensure actions are clipped to `[0,1]` before passing to `env.step()`.

**PPO divergence (NaN rewards)**  
→ Reduce learning rate to `1e-4`; check that rewards are bounded.

**Market environment: no trades clearing**  
→ Agents may all be posting the same offer type. Ensure `action_space`  
   sampling covers all offer types (0, 1, 2).

**Slow training on CPU**  
→ Reduce `n_agents` and `max_steps`, or set `device: cuda` if available.  
   The CommNet baseline is ~2× slower than standard PPO due to message encoding.
