"""
plots/visualize.py
==================
Publication-quality figures for CollectiveOS-Bench results.

Functions:
    plot_reward_curves(...)        — learning curves with CI shading
    plot_cooperation_graph(...)    — per-step cooperation rate heatmap
    plot_metric_comparison(...)    — radar / bar chart across baselines
    plot_resource_trajectory(...)  — resource level over time
    plot_ies_vs_car(...)           — IES vs CAR scatter across seeds
    make_summary_figure(...)       — 2×3 panel summary figure
"""

from __future__ import annotations
import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

PALETTE = {
    "ppo":         "#2D6A9F",
    "cooperative": "#2E8B57",
    "greedy":      "#C0392B",
    "random":      "#8E44AD",
    "commons":     "#1A7DA8",
    "market":      "#E67E22",
    "institution": "#27AE60",
}

def _smooth(x: np.ndarray, window: int = 10) -> np.ndarray:
    if len(x) < window:
        return x
    return np.convolve(x, np.ones(window) / window, mode="valid")


# ======================================================================
# 1. Reward curves
# ======================================================================

def plot_reward_curves(
    results: Dict[str, List[float]],
    title: str = "Training Reward",
    smooth_window: int = 10,
    out_path: Optional[str] = None,
) -> plt.Figure:
    """
    results : dict mapping label → list of per-episode mean rewards
    """
    fig, ax = plt.subplots(figsize=(7, 4))
    for label, rewards in results.items():
        r = np.array(rewards, dtype=np.float64)
        s = _smooth(r, smooth_window)
        x = np.arange(len(s))
        color = PALETTE.get(label.split("_")[0], "#555")
        ax.plot(x, s, label=label, color=color, lw=1.8)
        # Uncertainty band (running std)
        if len(r) > smooth_window:
            std = np.array([r[max(0, i-smooth_window):i].std() for i in range(smooth_window, len(r)+1)])
            ax.fill_between(x, s - std, s + std, alpha=0.15, color=color)

    ax.set_xlabel("Episode", fontsize=12)
    ax.set_ylabel("Mean Reward", fontsize=12)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.legend(framealpha=0.9, fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    if out_path:
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
    return fig


# ======================================================================
# 2. Cooperation graph (heatmap)
# ======================================================================

def plot_cooperation_heatmap(
    compliance_matrix: np.ndarray,
    title: str = "Per-Agent Compliance Over Time",
    out_path: Optional[str] = None,
) -> plt.Figure:
    """
    compliance_matrix : (T, N) boolean/float array
                        1 = cooperative, 0 = defection
    """
    fig, ax = plt.subplots(figsize=(9, 3))
    im = ax.imshow(
        compliance_matrix.T,
        aspect="auto",
        cmap="RdYlGn",
        vmin=0, vmax=1,
        interpolation="nearest",
    )
    ax.set_xlabel("Time step", fontsize=11)
    ax.set_ylabel("Agent", fontsize=11)
    ax.set_title(title, fontsize=12, fontweight="bold")
    plt.colorbar(im, ax=ax, label="Compliance")
    fig.tight_layout()
    if out_path:
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
    return fig


# ======================================================================
# 3. Metric comparison bar chart
# ======================================================================

def plot_metric_comparison(
    metric_data: Dict[str, Dict[str, float]],
    metrics: Optional[List[str]] = None,
    title: str = "Metric Comparison Across Baselines",
    out_path: Optional[str] = None,
) -> plt.Figure:
    """
    metric_data : { baseline_name : { metric_name : value, ... } }
    """
    if metrics is None:
        metrics = [
            "cooperation_rate",
            "gini_coefficient",
            "stability_score",
            "institutional_emergence_score",
            "collective_adaptation_rate",
            "coordination_efficiency_index",
        ]
    labels = list(metric_data.keys())
    x = np.arange(len(metrics))
    width = 0.8 / len(labels)

    fig, ax = plt.subplots(figsize=(11, 5))
    for i, label in enumerate(labels):
        vals = [metric_data[label].get(m, 0.0) for m in metrics]
        color = PALETTE.get(label.split("_")[0], f"C{i}")
        ax.bar(
            x + i * width - (len(labels) - 1) * width / 2,
            vals,
            width * 0.9,
            label=label,
            color=color,
            alpha=0.85,
            edgecolor="white",
            linewidth=0.5,
        )

    short_names = {
        "cooperation_rate": "CR",
        "gini_coefficient": "Gini",
        "stability_score": "SS",
        "institutional_emergence_score": "IES",
        "collective_adaptation_rate": "CAR",
        "coordination_efficiency_index": "CEI",
        "social_welfare": "SW (norm)",
    }
    ax.set_xticks(x)
    ax.set_xticklabels([short_names.get(m, m) for m in metrics], fontsize=11)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_ylim(0, 1.1)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.legend(fontsize=9, framealpha=0.9)
    ax.spines[["top", "right"]].set_visible(False)
    ax.axhline(1.0, lw=0.7, ls="--", color="#aaa")
    fig.tight_layout()
    if out_path:
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
    return fig


# ======================================================================
# 4. Resource trajectory
# ======================================================================

def plot_resource_trajectory(
    trajectories: Dict[str, List[float]],
    collapse_thresh: float = 5.0,
    K: float = 100.0,
    title: str = "Resource Level Over Time",
    out_path: Optional[str] = None,
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(7, 4))
    for label, traj in trajectories.items():
        color = PALETTE.get(label.split("_")[0], "#555")
        ax.plot(traj, label=label, color=color, lw=1.8)

    ax.axhline(collapse_thresh, ls="--", color="#C0392B", lw=1.2, label="Collapse threshold")
    ax.axhline(K, ls=":", color="#888", lw=1.0, label="Carrying capacity K")
    ax.fill_between(range(max(len(v) for v in trajectories.values())),
                    0, collapse_thresh, alpha=0.08, color="#C0392B")
    ax.set_xlabel("Time step", fontsize=12)
    ax.set_ylabel("Resource level", fontsize=12)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.legend(fontsize=9, framealpha=0.9)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    if out_path:
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
    return fig


# ======================================================================
# 5. IES vs CAR scatter
# ======================================================================

def plot_ies_vs_car(
    data: Dict[str, Dict[str, List[float]]],
    title: str = "IES vs CAR (per seed)",
    out_path: Optional[str] = None,
) -> plt.Figure:
    """
    data : { label : { "ies": [...], "car": [...] } }
    """
    fig, ax = plt.subplots(figsize=(5, 5))
    for label, vals in data.items():
        ies = np.array(vals.get("ies", []))
        car = np.array(vals.get("car", []))
        color = PALETTE.get(label.split("_")[0], "#555")
        ax.scatter(car, ies, label=label, color=color, alpha=0.7, s=60, edgecolors="white", lw=0.5)
        if len(ies) > 1:
            ax.scatter([car.mean()], [ies.mean()], marker="*", s=200, color=color, zorder=5)

    ax.set_xlabel("Collective Adaptation Rate (CAR)", fontsize=12)
    ax.set_ylabel("Institutional Emergence Score (IES)", fontsize=12)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.legend(fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    fig.tight_layout()
    if out_path:
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
    return fig


# ======================================================================
# 6. Summary 2×3 panel figure
# ======================================================================

def make_summary_figure(
    reward_data: Dict[str, List[float]],
    metric_data: Dict[str, Dict[str, float]],
    resource_data: Dict[str, List[float]],
    ies_car_data: Dict[str, Dict[str, List[float]]],
    out_path: Optional[str] = "plots/summary.pdf",
) -> plt.Figure:
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(2, 3, hspace=0.38, wspace=0.35)

    # (0,0) reward curves
    ax1 = fig.add_subplot(gs[0, 0])
    for label, rewards in reward_data.items():
        s = _smooth(np.array(rewards), 15)
        ax1.plot(s, label=label, color=PALETTE.get(label.split("_")[0], "C0"), lw=1.8)
    ax1.set_title("(a) Learning Curves", fontweight="bold")
    ax1.set_xlabel("Episode"); ax1.set_ylabel("Mean Reward")
    ax1.legend(fontsize=7); ax1.spines[["top","right"]].set_visible(False)

    # (0,1) resource trajectory
    ax2 = fig.add_subplot(gs[0, 1])
    for label, traj in resource_data.items():
        ax2.plot(traj, label=label, color=PALETTE.get(label.split("_")[0], "C1"), lw=1.8)
    ax2.axhline(5.0, ls="--", color="#C0392B", lw=1.0)
    ax2.set_title("(b) Resource Trajectory", fontweight="bold")
    ax2.set_xlabel("Step"); ax2.set_ylabel("Resource Level")
    ax2.legend(fontsize=7); ax2.spines[["top","right"]].set_visible(False)

    # (0,2) IES vs CAR
    ax3 = fig.add_subplot(gs[0, 2])
    for label, vals in ies_car_data.items():
        ies = np.array(vals.get("ies", [0]))
        car = np.array(vals.get("car", [0]))
        ax3.scatter(car, ies, label=label, color=PALETTE.get(label.split("_")[0], "C2"),
                    alpha=0.75, s=55, edgecolors="white", lw=0.4)
    ax3.set_title("(c) IES vs CAR", fontweight="bold")
    ax3.set_xlabel("CAR"); ax3.set_ylabel("IES")
    ax3.legend(fontsize=7); ax3.spines[["top","right"]].set_visible(False)

    # (1,0:3) metric comparison
    ax4 = fig.add_subplot(gs[1, :])
    metrics = ["cooperation_rate","gini_coefficient","stability_score",
               "institutional_emergence_score","collective_adaptation_rate",
               "coordination_efficiency_index"]
    labels = list(metric_data.keys())
    x = np.arange(len(metrics))
    width = 0.75 / len(labels)
    for i, label in enumerate(labels):
        vals = [metric_data[label].get(m, 0) for m in metrics]
        color = PALETTE.get(label.split("_")[0], f"C{i}")
        ax4.bar(x + i*width - (len(labels)-1)*width/2, vals, width*0.9,
                label=label, color=color, alpha=0.85)
    ax4.set_xticks(x)
    ax4.set_xticklabels(["CR","Gini","SS","IES","CAR","CEI"], fontsize=11)
    ax4.set_ylim(0,1.15); ax4.set_ylabel("Score")
    ax4.set_title("(d) Metric Comparison Across Baselines", fontweight="bold")
    ax4.legend(fontsize=8, ncol=len(labels)); ax4.spines[["top","right"]].set_visible(False)

    if out_path:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=180, bbox_inches="tight")
    return fig


# ======================================================================
# Demo: generate mock figures from synthetic data
# ======================================================================

if __name__ == "__main__":
    rng = np.random.default_rng(42)
    T = 300

    reward_data = {
        "ppo":         list(np.cumsum(rng.normal(0.05, 0.3, 300))),
        "cooperative": list(np.cumsum(rng.normal(0.03, 0.2, 300))),
        "greedy":      list(np.cumsum(rng.normal(-0.02, 0.4, 300))),
    }
    resource_data = {
        "ppo":         list(50 + 40 * np.sin(np.linspace(0, 3, T)) + rng.normal(0, 2, T)),
        "greedy":      list(np.maximum(0, 80 - np.linspace(0, 90, T) + rng.normal(0, 3, T))),
    }
    metric_data = {
        "ppo":         {"cooperation_rate":0.72,"gini_coefficient":0.18,"stability_score":0.81,
                        "institutional_emergence_score":0.65,"collective_adaptation_rate":0.53,
                        "coordination_efficiency_index":0.74},
        "cooperative": {"cooperation_rate":0.91,"gini_coefficient":0.08,"stability_score":0.93,
                        "institutional_emergence_score":0.78,"collective_adaptation_rate":0.42,
                        "coordination_efficiency_index":0.88},
        "greedy":      {"cooperation_rate":0.12,"gini_coefficient":0.47,"stability_score":0.21,
                        "institutional_emergence_score":0.04,"collective_adaptation_rate":0.05,
                        "coordination_efficiency_index":0.09},
        "random":      {"cooperation_rate":0.48,"gini_coefficient":0.31,"stability_score":0.44,
                        "institutional_emergence_score":0.11,"collective_adaptation_rate":0.08,
                        "coordination_efficiency_index":0.33},
    }
    ies_car_data = {
        "ppo":         {"ies": list(rng.uniform(0.5,0.8,10)), "car": list(rng.uniform(0.4,0.7,10))},
        "cooperative": {"ies": list(rng.uniform(0.7,0.9,10)), "car": list(rng.uniform(0.3,0.5,10))},
        "greedy":      {"ies": list(rng.uniform(0.0,0.1,10)), "car": list(rng.uniform(0.0,0.1,10))},
    }

    Path("plots").mkdir(exist_ok=True)
    fig = make_summary_figure(reward_data, metric_data, resource_data, ies_car_data,
                              out_path="plots/summary.pdf")
    plot_metric_comparison(metric_data, out_path="plots/metric_comparison.png")
    plot_reward_curves(reward_data, out_path="plots/reward_curves.png")
    plot_resource_trajectory(resource_data, out_path="plots/resource_trajectory.png")
    print("Plots saved to plots/")
