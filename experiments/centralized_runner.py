"""
experiments/centralized_runner.py
==================================
Experiment runner variant for centralized and CommNet baselines.

These baselines require modified training loops (joint observations,
message passing) not supported by the standard ExperimentRunner.
"""

from __future__ import annotations
import json
import logging
import os
import random
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import yaml

from environments import CommonsEnvironment, MarketEnvironment, InstitutionEnvironment
from baselines.centralized import CentralizedPPOController, CentralizedOptimalController
from baselines.commnet import CommNetGroup
from metrics.core import MetricSummary
from experiments.runner import ExperimentConfig, ENV_MAP

logger = logging.getLogger(__name__)


class CentralizedRunner:
    """
    Runner for the CentralizedPPOController baseline.

    Differs from ExperimentRunner:
    - Single network sees joint observations
    - Mean reward (social welfare / N) used for training signal
    """

    def __init__(self, config: ExperimentConfig):
        self.cfg = config
        _set_seeds(config.seed)

        env_cls = ENV_MAP[config.environment]
        env_kw = dict(config.env_kwargs)
        env_kw.setdefault("n_agents", 5)
        env_kw.setdefault("max_steps", config.max_steps)
        env_kw["seed"] = config.seed
        self.env = env_cls(**env_kw)
        self.n_agents = self.env.n_agents

        obs_dim = self.env.observation_space.shape[0]
        action_dim = self.env.action_space.shape[0]

        self.controller = CentralizedPPOController(
            n_agents=self.n_agents,
            obs_dim=obs_dim,
            action_dim=action_dim,
            device=config.device,
            **config.agent_kwargs,
        )
        self.metrics = MetricSummary(
            n_agents=self.n_agents,
            K=env_kw.get("max_resource", 100.0),
        )
        self.save_dir = Path(config.save_dir) / config.name
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.episode_rewards: List[float] = []
        self.episode_metrics: List[dict] = []

    def run(self) -> dict:
        for episode in range(self.cfg.n_episodes):
            obs_list = self.env.reset()
            total_rew = np.zeros(self.n_agents)
            step_metrics = MetricSummary(
                n_agents=self.n_agents,
                K=getattr(self.env, "K", 100.0),
            )
            done = False

            while not done:
                actions = self.controller.act(obs_list)
                obs_next, rewards, done, info = self.env.step(actions)
                rew_arr = np.array(rewards)
                mean_rew = float(rew_arr.mean())
                self.controller.store(obs_list, actions, mean_rew, done)
                step_metrics.record(actions, rewards, info.get("resource_level", 0.0))
                total_rew += rew_arr
                obs_list = obs_next

            loss = self.controller.update(last_obs_list=obs_list)

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
            self.episode_rewards.append(float(total_rew.mean()))
            self.episode_metrics.append(ep_metrics)

            if (episode + 1) % self.cfg.log_interval == 0:
                recent = np.mean(self.episode_rewards[-self.cfg.log_interval:])
                logger.info(f"[CentralPPO] Ep {episode+1}/{self.cfg.n_episodes} | Reward: {recent:.3f}")

        return self._aggregate()

    def _aggregate(self) -> dict:
        rewards = np.array(self.episode_rewards)
        metric_keys = list(self.episode_metrics[0].keys()) if self.episode_metrics else []
        agg = {}
        for k in metric_keys:
            vals = [m[k] for m in self.episode_metrics]
            agg[k] = {"mean": float(np.mean(vals)), "std": float(np.std(vals))}
        result = {
            "experiment": self.cfg.name,
            "reward_mean": float(rewards.mean()),
            "reward_std": float(rewards.std()),
            "reward_history": rewards.tolist(),
            "metrics": agg,
        }
        with open(self.save_dir / "results.json", "w") as f:
            json.dump(result, f, indent=2)
        return result


class CommNetRunner:
    """
    Runner for CommNet communication baseline.
    Agents pass messages before acting each step.
    """

    def __init__(self, config: ExperimentConfig):
        self.cfg = config
        _set_seeds(config.seed)

        env_cls = ENV_MAP[config.environment]
        env_kw = dict(config.env_kwargs)
        env_kw.setdefault("n_agents", 5)
        env_kw.setdefault("max_steps", config.max_steps)
        env_kw["seed"] = config.seed
        self.env = env_cls(**env_kw)
        self.n_agents = self.env.n_agents

        obs_dim = self.env.observation_space.shape[0]
        action_dim = self.env.action_space.shape[0]

        self.group = CommNetGroup(
            n_agents=self.n_agents,
            obs_dim=obs_dim,
            action_dim=action_dim,
            **config.agent_kwargs,
        )
        self.save_dir = Path(config.save_dir) / config.name
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.episode_rewards: List[float] = []
        self.episode_metrics: List[dict] = []

    def run(self) -> dict:
        for episode in range(self.cfg.n_episodes):
            obs_list = self.env.reset()
            self.group.reset()
            total_rew = np.zeros(self.n_agents)
            step_metrics = MetricSummary(
                n_agents=self.n_agents,
                K=getattr(self.env, "K", 100.0),
            )
            done = False

            while not done:
                actions = self.group.act(obs_list)
                obs_next, rewards, done, info = self.env.step(actions)
                self.group.store(obs_list, actions, rewards, [done] * self.n_agents)
                step_metrics.record(actions, rewards, info.get("resource_level", 0.0))
                total_rew += np.array(rewards)
                obs_list = obs_next

            self.group.update(obs_list)

            adapt = getattr(self.env, "_adaptation_events", 0)
            comp_hist = list(getattr(self.env, "_compliance_history", []))
            rule_log = getattr(self.env, "_rule_change_log", [])

            ep_metrics = step_metrics.compute(
                adaptation_events=adapt,
                max_adaptations=max(1, self.cfg.max_steps // 10),
                compliance_history=comp_hist,
                rule_change_log=rule_log,
                collapsed=info.get("collapsed", False),
            )
            self.episode_rewards.append(float(total_rew.mean()))
            self.episode_metrics.append(ep_metrics)

            if (episode + 1) % self.cfg.log_interval == 0:
                recent = np.mean(self.episode_rewards[-self.cfg.log_interval:])
                logger.info(f"[CommNet] Ep {episode+1}/{self.cfg.n_episodes} | Reward: {recent:.3f}")

        return self._aggregate()

    def _aggregate(self) -> dict:
        rewards = np.array(self.episode_rewards)
        metric_keys = list(self.episode_metrics[0].keys()) if self.episode_metrics else []
        agg = {}
        for k in metric_keys:
            vals = [m[k] for m in self.episode_metrics]
            agg[k] = {"mean": float(np.mean(vals)), "std": float(np.std(vals))}
        result = {
            "experiment": self.cfg.name,
            "reward_mean": float(rewards.mean()),
            "reward_history": rewards.tolist(),
            "metrics": agg,
        }
        with open(self.save_dir / "results.json", "w") as f:
            json.dump(result, f, indent=2)
        return result


def _set_seeds(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
