#!/usr/bin/env python
"""
reproducibility/reproduce_all.py
==================================
Reproduce all key results from the CollectiveOS-Bench paper.

This script runs all three environments across four baselines with
five random seeds each.  Expected wall-clock time on a modern CPU:
~25 minutes.  Set N_EPISODES to a smaller value for a quick smoke test.

Usage:
    python reproducibility/reproduce_all.py
    python reproducibility/reproduce_all.py --quick   # 50 episodes
"""

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from experiments.runner import ExperimentRunner, ExperimentConfig

SEEDS = [42, 123, 2024, 7, 999]
ENVIRONMENTS = ["commons", "institution", "market"]
AGENTS = ["ppo", "cooperative", "greedy", "random"]

FULL_EPISODES = 300
QUICK_EPISODES = 30

ENV_KWARGS = {
    "commons":     {"n_agents": 5, "max_resource": 100.0, "regen_rate": 0.3},
    "institution": {"n_agents": 6, "max_resource": 100.0, "allow_rule_proposals": True},
    "market":      {"n_agents": 6, "init_inv": 50.0, "init_cash": 200.0},
}
AGENT_KWARGS = {
    "ppo":         {"lr": 3e-4, "clip_eps": 0.2, "n_epochs": 4, "hidden": 128},
    "cooperative": {},
    "greedy":      {},
    "random":      {},
}
MAX_STEPS = {"commons": 200, "institution": 300, "market": 300}


def run_all(n_episodes: int):
    all_results = {}
    for env in ENVIRONMENTS:
        for agent in AGENTS:
            seed_results = []
            for seed in SEEDS:
                name = f"{env}_{agent}_seed{seed}"
                print(f"\n{'='*60}\nRunning: {name}\n{'='*60}")
                cfg = ExperimentConfig(
                    name=name,
                    environment=env,
                    env_kwargs=ENV_KWARGS[env],
                    agent_type=agent,
                    agent_kwargs=AGENT_KWARGS[agent],
                    n_episodes=n_episodes,
                    max_steps=MAX_STEPS[env],
                    seed=seed,
                    log_interval=max(1, n_episodes // 10),
                    save_dir="results",
                )
                runner = ExperimentRunner(cfg)
                result = runner.run()
                seed_results.append(result)

            key = f"{env}_{agent}"
            metrics_across_seeds = {}
            for metric_key in seed_results[0]["metrics"]:
                means = [r["metrics"][metric_key]["mean"] for r in seed_results]
                import numpy as np
                metrics_across_seeds[metric_key] = {
                    "mean": float(np.mean(means)),
                    "std": float(np.std(means)),
                    "per_seed": means,
                }
            all_results[key] = metrics_across_seeds

    out_path = Path("results") / "all_results_summary.json"
    out_path.parent.mkdir(exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n\nAll results saved to {out_path}")

    _print_summary_table(all_results)
    return all_results


def _print_summary_table(results: dict):
    from textwrap import shorten
    KEY_METRICS = [
        "cooperation_rate",
        "stability_score",
        "institutional_emergence_score",
        "coordination_efficiency_index",
    ]
    ABBREV = {"cooperation_rate": "CR", "stability_score": "SS",
              "institutional_emergence_score": "IES",
              "coordination_efficiency_index": "CEI"}

    header = f"{'Experiment':<30}" + "".join(f"{ABBREV[m]:>10}" for m in KEY_METRICS)
    print("\n" + "=" * len(header))
    print("RESULTS SUMMARY")
    print("=" * len(header))
    print(header)
    print("-" * len(header))
    for key, metrics in results.items():
        row = f"{key:<30}"
        for m in KEY_METRICS:
            v = metrics.get(m, {}).get("mean", 0.0)
            row += f"{v:>10.3f}"
        print(row)
    print("=" * len(header))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="Run fewer episodes for testing")
    args = parser.parse_args()
    n_ep = QUICK_EPISODES if args.quick else FULL_EPISODES
    run_all(n_ep)
