# CollectiveOS-Bench

**CollectiveOS-Bench: A Benchmark Suite for Emergent Cooperation, Institutions, and Collective Intelligence in Multi-Agent Systems**

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.9%2B-green.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange.svg)](https://pytorch.org)

---

## Overview

CollectiveOS-Bench is a reproducible, research-grade benchmark for studying **emergent cooperation**, **decentralised coordination**, **institutional dynamics**, and **adaptive governance** in multi-agent systems.

Existing benchmarks largely focus on competitive settings or fully cooperative tasks with fixed rules. CollectiveOS-Bench fills the gap by introducing environments where institutional structure itself is a dynamic, emergent property of agent behaviour—and providing a principled metric suite to measure it.

---

## Environments

| Environment | Core Dynamic | Key Challenge |
|---|---|---|
| **Commons** | Shared renewable resource pool | Avoid tragedy of the commons |
| **Market** | Decentralised double-auction | Emergent price discovery, specialisation |
| **Institution** ★ | Dynamic rule system with voting | Governance emergence, compliance, adaptation |

★ = Key novelty of this benchmark.

---

## Novel Metrics

Beyond standard cooperation and social welfare metrics, CollectiveOS-Bench introduces:

- **Institutional Emergence Score (IES)** — quantifies whether agents collectively formed and maintained a functioning institution
- **Collective Adaptation Rate (CAR)** — measures the rate and quality of democratic rule updates
- **Coordination Efficiency Index (CEI)** — normalises performance between the Nash equilibrium and the social optimum

---

## Quick Start

```bash
git clone https://github.com/johanvarughese/collectiveos-bench
cd collectiveos-bench
pip install -r requirements.txt
# or
pip install -e .
```

**Run a single experiment:**
```bash
python -m experiments.runner --config configs/commons_ppo.yaml
```

**Reproduce all paper results:**
```bash
python reproducibility/reproduce_all.py
# Quick smoke test:
python reproducibility/reproduce_all.py --quick
```

**Generate plots:**
```bash
python plots/visualize.py
```

**Launch REST API:**
```bash
uvicorn api.app:app --reload --port 8000
# POST http://localhost:8000/experiments/run
# GET  http://localhost:8000/experiments/{name}
```

---

## Project Structure

```
collectiveos-bench/
 ├── environments/          # Commons, Market, Institution gym envs
 ├── agents/                # BaseAgent, PPOAgent, baselines
 ├── metrics/               # Standard + novel metric implementations
 ├── experiments/           # Config-driven experiment runner + logger
 ├── configs/               # YAML experiment configs (reproducible)
 ├── results/               # Output JSON — auto-generated
 ├── plots/                 # Matplotlib visualisation scripts
 ├── reproducibility/       # Fixed-seed scripts to reproduce paper results
 ├── api/                   # FastAPI REST interface
 ├── docs/                  # Extended documentation
 └── paper/                 # NeurIPS-style paper draft (LaTeX + PDF)
```

---

## Baselines

| Baseline | Description |
|---|---|
| `random` | Uniform random actions — lower bound |
| `greedy` | Always maximise extraction — Nash defection |
| `cooperative` | Fixed cooperative policy — upper bound reference |
| `ppo` | Independent PPO agents (no centralised training) |

---

## Reproducibility

All experiments use fixed NumPy, PyTorch, and Python random seeds.  
Configs are versioned YAML files in `configs/`.  
The `reproducibility/reproduce_all.py` script regenerates all paper tables and figures end-to-end.

---

## Citation

If you use CollectiveOS-Bench in your research, please cite:

```bibtex
@misc{varughese2024collectiveos,
  title     = {CollectiveOS-Bench: A Benchmark Suite for Emergent Cooperation,
               Institutions, and Collective Intelligence in Multi-Agent Systems},
  author    = {Varughese, Johan},
  year      = {2024},
  url       = {https://github.com/johanvarughese/collectiveos-bench},
  note      = {Original benchmark design, environments, and metrics
               attributed to Johan Varughese. Open-source under Apache 2.0.}
}
```

Academic use without citation is a violation of the terms of this repository.

---

## License

Copyright © 2024 Johan Varughese.

Licensed under the [Apache License 2.0](LICENSE).

The original benchmark design, environment architectures, metric formulations  
(IES, CAR, CEI), and experimental protocol are the intellectual contribution  
of Johan Varughese. Derivative works must retain attribution.

---

## Contributing

Issues and pull requests are welcome. Please open an issue before submitting large changes.
