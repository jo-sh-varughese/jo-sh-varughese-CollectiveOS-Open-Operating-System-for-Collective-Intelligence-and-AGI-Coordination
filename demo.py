#!/usr/bin/env python
"""
demo.py
=======
End-to-end quickstart demo for CollectiveOS-Bench.

Runs 30 episodes of each environment with PPO and Greedy agents,
prints a metric summary, and generates demo plots in plots/.

Usage:
    python demo.py
"""

import sys
import os
from pathlib import Path

# Allow running from repo root
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import json

from experiments.runner import ExperimentRunner, ExperimentConfig
from plots.visualize import (
    plot_reward_curves,
    plot_metric_comparison,
    plot_resource_trajectory,
    make_summary_figure,
)


def run_demo():
    print("\n" + "=" * 65)
    print("  CollectiveOS-Bench — Quick Demo")
    print("  Copyright © Johan Varughese | Apache 2.0")
    print("=" * 65 + "\n")

    configs = [
        ExperimentConfig(
            name="demo_commons_ppo",
            environment="commons",
            env_kwargs={"n_agents": 5, "max_resource": 100.0, "regen_rate": 0.3},
            agent_type="ppo",
            agent_kwargs={"lr": 3e-4, "hidden": 64},
            n_episodes=30,
            max_steps=100,
            seed=42,
            log_interval=10,
            save_dir="results",
        ),
        ExperimentConfig(
            name="demo_commons_greedy",
            environment="commons",
            env_kwargs={"n_agents": 5},
            agent_type="greedy",
            n_episodes=30,
            max_steps=100,
            seed=42,
            log_interval=30,
            save_dir="results",
        ),
        ExperimentConfig(
            name="demo_institution_ppo",
            environment="institution",
            env_kwargs={"n_agents": 6, "allow_rule_proposals": True},
            agent_type="ppo",
            agent_kwargs={"lr": 3e-4, "hidden": 64},
            n_episodes=30,
            max_steps=150,
            seed=42,
            log_interval=10,
            save_dir="results",
        ),
        ExperimentConfig(
            name="demo_market_ppo",
            environment="market",
            env_kwargs={"n_agents": 4},
            agent_type="ppo",
            agent_kwargs={"lr": 3e-4, "hidden": 64},
            n_episodes=30,
            max_steps=100,
            seed=42,
            log_interval=10,
            save_dir="results",
        ),
    ]

    all_results = {}
    for cfg in configs:
        print(f"\n▶  Running: {cfg.name}")
        runner = ExperimentRunner(cfg)
        result = runner.run()
        all_results[cfg.name] = result
        _print_metrics(cfg.name, result["metrics"])

    print("\n\n" + "=" * 65)
    print("  Generating demo plots → plots/")
    print("=" * 65)

    Path("plots").mkdir(exist_ok=True)

    # Reward curves
    reward_data = {
        name: res["reward_history"]
        for name, res in all_results.items()
    }
    plot_reward_curves(
        reward_data,
        title="CollectiveOS-Bench Demo — Reward Curves",
        out_path="plots/demo_reward_curves.png",
    )

    # Resource trajectory (Commons only — greedy vs PPO)
    commons_ppo_path = Path("results/demo_commons_ppo/results.json")
    commons_greedy_path = Path("results/demo_commons_greedy/results.json")

    resource_traj = {}
    for label, path in [("ppo", commons_ppo_path), ("greedy", commons_greedy_path)]:
        if path.exists():
            resource_traj[label] = _extract_resource_proxy(path)
    if resource_traj:
        plot_resource_trajectory(
            resource_traj,
            out_path="plots/demo_resource_trajectory.png",
        )

    # Metric comparison across demo runs
    metric_data = {}
    for name, res in all_results.items():
        metric_data[name] = {k: v["mean"] for k, v in res["metrics"].items()}
    plot_metric_comparison(
        metric_data,
        title="Demo Metric Comparison",
        out_path="plots/demo_metric_comparison.png",
    )

    print("\n✓  Plots saved to plots/demo_*.png")
    print("\n✓  Demo complete.  Results in results/demo_*/results.json")
    print("\nTo run the full benchmark:")
    print("  python reproducibility/reproduce_all.py")
    print("\nTo launch the REST API:")
    print("  uvicorn api.app:app --reload --port 8000\n")


def _print_metrics(name: str, metrics: dict):
    KEY_METRICS = ["cooperation_rate", "stability_score",
                   "institutional_emergence_score", "coordination_efficiency_index"]
    ABBREV = {"cooperation_rate": "CR", "stability_score": "SS",
              "institutional_emergence_score": "IES",
              "coordination_efficiency_index": "CEI"}
    print(f"\n  Metrics for {name}:")
    for m in KEY_METRICS:
        if m in metrics:
            v = metrics[m]
            mu = v.get("mean", 0.0)
            sd = v.get("std", 0.0)
            print(f"    {ABBREV.get(m, m):<8} = {mu:.3f} ± {sd:.3f}")


def _extract_resource_proxy(path: Path) -> list:
    """
    Returns a synthetic resource-level proxy from reward history.
    (Full resource trajectories require per-step logging, which is
    not stored by default for demo runs.)
    """
    with open(path) as f:
        data = json.load(f)
    rh = data.get("reward_history", [])
    # Map reward to approximate resource: higher reward ≈ higher resource
    rmin, rmax = min(rh), max(rh) + 1e-8
    return [5 + 90 * (r - rmin) / (rmax - rmin) for r in rh]


if __name__ == "__main__":
    run_demo()
