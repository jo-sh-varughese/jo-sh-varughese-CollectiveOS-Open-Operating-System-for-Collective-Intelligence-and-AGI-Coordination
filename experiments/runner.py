"""
experiments/runner.py
=====================
Config-driven experiment runner for CollectiveOS-Bench.

Usage
-----
    from experiments.runner import ExperimentRunner
    runner = ExperimentRunner.from_yaml("configs/commons_ppo.yaml")
    results = runner.run()

Or from CLI:
    python -m experiments.runner --config configs/commons_ppo.yaml
"""

from __future__ import annotations
import argparse
import json
import logging
import os
import random
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import yaml

from environments import CommonsEnvironment, MarketEnvironment, InstitutionEnvironment
from agents.base import RandomAgent, GreedyAgent, CooperativeAgent, PPOAgent
from metrics.core import MetricSummary

logger = logging.getLogger(__name__)


# ======================================================================
# Config dataclass
# ======================================================================

@dataclass
class ExperimentConfig:
    name: str = "unnamed"
    environment: str = "commons"        # commons | market | institution
    env_kwargs: Dict[str, Any] = field(default_factory=dict)
    agent_type: str = "ppo"             # random | greedy | cooperative | ppo
    agent_kwargs: Dict[str, Any] = field(default_factory=dict)
    n_episodes: int = 200
    max_steps: int = 300
    seed: int = 42
    log_interval: int = 10
    save_dir: str = "results"
    device: str = "cpu"


# ======================================================================
# Runner
# ======================================================================

ENV_MAP = {
    "commons": CommonsEnvironment,
    "market": MarketEnvironment,
    "institution": InstitutionEnvironment,
}

AGENT_MAP = {
    "random": RandomAgent,
    "greedy": GreedyAgent,
    "cooperative": CooperativeAgent,
    "ppo": PPOAgent,
}


class ExperimentRunner:

    def __init__(self, config: ExperimentConfig):
        self.cfg = config
        self._set_seeds(config.seed)
        self._setup_logging()

        # Build environment
        env_cls = ENV_MAP[config.environment]
        env_kw = dict(config.env_kwargs)
        env_kw.setdefault("n_agents", 5)
        env_kw.setdefault("max_steps", config.max_steps)
        env_kw["seed"] = config.seed
        self.env = env_cls(**env_kw)

        self.n_agents = self.env.n_agents

        obs_dim = self.env.observation_space.shape[0]
        action_dim = self.env.action_space.shape[0]

        # Build agents
        agent_cls = AGENT_MAP[config.agent_type]
        agent_kw = dict(config.agent_kwargs)
        if config.agent_type == "ppo":
            agent_kw["device"] = config.device
        self.agents: List = [
            agent_cls(
                agent_id=i,
                obs_dim=obs_dim,
                action_dim=action_dim,
                **agent_kw,
            )
            for i in range(self.n_agents)
        ]

        # Metric accumulator
        self.metrics = MetricSummary(
            n_agents=self.n_agents,
            K=env_kw.get("max_resource", 100.0),
        )

        self.save_dir = Path(config.save_dir) / config.name
        self.save_dir.mkdir(parents=True, exist_ok=True)

        # Results history
        self.episode_rewards: List[float] = []
        self.episode_metrics: List[dict] = []

    # ------------------------------------------------------------------

    @classmethod
    def from_yaml(cls, path: str) -> "ExperimentRunner":
        with open(path) as f:
            raw = yaml.safe_load(f)
        cfg = ExperimentConfig(**raw)
        return cls(cfg)

    @classmethod
    def from_dict(cls, d: dict) -> "ExperimentRunner":
        return cls(ExperimentConfig(**d))

    # ------------------------------------------------------------------

    def run(self) -> dict:
        logger.info(f"Starting experiment: {self.cfg.name}")
        t0 = time.time()

        for episode in range(self.cfg.n_episodes):
            ep_result = self._run_episode(episode)
            self.episode_rewards.append(ep_result["total_reward"])
            self.episode_metrics.append(ep_result["metrics"])

            if (episode + 1) % self.cfg.log_interval == 0:
                recent = np.mean(self.episode_rewards[-self.cfg.log_interval:])
                logger.info(
                    f"Episode {episode+1:4d}/{self.cfg.n_episodes} | "
                    f"Mean reward (last {self.cfg.log_interval}): {recent:8.3f}"
                )

        elapsed = time.time() - t0
        summary = self._aggregate()
        summary["elapsed_seconds"] = elapsed
        summary["config"] = asdict(self.cfg)

        out_path = self.save_dir / "results.json"
        with open(out_path, "w") as f:
            json.dump(summary, f, indent=2)
        logger.info(f"Results saved to {out_path}")

        return summary

    # ------------------------------------------------------------------

    def _run_episode(self, episode_idx: int) -> dict:
        obs_list = self.env.reset()
        for agent in self.agents:
            agent.reset()

        total_reward = np.zeros(self.n_agents)
        step_metrics = MetricSummary(
            n_agents=self.n_agents,
            K=getattr(self.env, "K", 100.0),
        )

        done = False
        step = 0

        while not done:
            actions = [
                agent.select_action(obs_list[i])
                for i, agent in enumerate(self.agents)
            ]
            obs_next, rewards, done, info = self.env.step(actions)

            # Store for PPO
            for i, agent in enumerate(self.agents):
                if isinstance(agent, PPOAgent):
                    agent.store(obs_list[i], actions[i], rewards[i], done)

            resource = info.get("resource_level", 0.0)
            step_metrics.record(actions, rewards, resource)
            total_reward += np.array(rewards)
            obs_list = obs_next
            step += 1

        # PPO update at episode end
        if isinstance(self.agents[0], PPOAgent):
            losses = []
            for i, agent in enumerate(self.agents):
                loss = agent.update(last_obs=obs_list[i])
                losses.append(loss)

        # Compute episode metrics
        adapt = getattr(self.env, "_adaptation_events", 0)
        comp_hist = list(getattr(self.env, "_compliance_history", []))
        rule_log = getattr(self.env, "_rule_change_log", [])
        collapsed = info.get("collapsed", False)

        ep_metrics = step_metrics.compute(
            adaptation_events=adapt,
            max_adaptations=max(1, self.cfg.max_steps // 10),
            compliance_history=comp_hist,
            rule_change_log=rule_log,
            collapsed=collapsed,
        )

        return {
            "episode": episode_idx,
            "total_reward": float(total_reward.mean()),
            "metrics": ep_metrics,
            "steps": step,
        }

    # ------------------------------------------------------------------

    def _aggregate(self) -> dict:
        rewards = np.array(self.episode_rewards)
        metric_keys = list(self.episode_metrics[0].keys()) if self.episode_metrics else []
        agg_metrics = {}
        for k in metric_keys:
            vals = [m[k] for m in self.episode_metrics]
            agg_metrics[k] = {
                "mean": float(np.mean(vals)),
                "std": float(np.std(vals)),
                "min": float(np.min(vals)),
                "max": float(np.max(vals)),
            }
        return {
            "experiment": self.cfg.name,
            "n_episodes": self.cfg.n_episodes,
            "reward_mean": float(rewards.mean()),
            "reward_std": float(rewards.std()),
            "reward_history": rewards.tolist(),
            "metrics": agg_metrics,
        }

    # ------------------------------------------------------------------

    @staticmethod
    def _set_seeds(seed: int):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        os.environ["PYTHONHASHSEED"] = str(seed)

    @staticmethod
    def _setup_logging():
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            datefmt="%H:%M:%S",
        )


# ======================================================================
# CLI entry point
# ======================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CollectiveOS-Bench Experiment Runner")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    args = parser.parse_args()
    runner = ExperimentRunner.from_yaml(args.config)
    results = runner.run()
    print(json.dumps(results["metrics"], indent=2))
