"""
api/app.py
==========
FastAPI REST interface for CollectiveOS-Bench.

Endpoints
---------
POST /experiments/run         → launch a new experiment (async)
GET  /experiments/{name}      → fetch results JSON
GET  /experiments             → list all saved experiments
GET  /health                  → health check

Usage
-----
    uvicorn api.app:app --reload --port 8000
"""

from __future__ import annotations
import json
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel

# Relative import fallback for standalone usage
try:
    from experiments.runner import ExperimentRunner, ExperimentConfig
except ImportError:
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from experiments.runner import ExperimentRunner, ExperimentConfig

app = FastAPI(
    title="CollectiveOS-Bench API",
    description="REST interface for multi-agent benchmark experiments.",
    version="0.1.0",
)

RESULTS_DIR = Path("results")
_running: Dict[str, str] = {}   # name → "running" | "done" | "error"


# ======================================================================
# Schemas
# ======================================================================

class RunRequest(BaseModel):
    name: str
    environment: str = "commons"
    env_kwargs: Dict[str, Any] = {}
    agent_type: str = "ppo"
    agent_kwargs: Dict[str, Any] = {}
    n_episodes: int = 100
    max_steps: int = 200
    seed: int = 42
    device: str = "cpu"


class StatusResponse(BaseModel):
    name: str
    status: str


# ======================================================================
# Background task
# ======================================================================

def _run_experiment(cfg: ExperimentConfig):
    name = cfg.name
    _running[name] = "running"
    try:
        runner = ExperimentRunner(cfg)
        runner.run()
        _running[name] = "done"
    except Exception as exc:
        _running[name] = f"error: {exc}"


# ======================================================================
# Endpoints
# ======================================================================

@app.get("/health")
def health():
    return {"status": "ok", "service": "CollectiveOS-Bench API"}


@app.post("/experiments/run", response_model=StatusResponse)
def run_experiment(req: RunRequest, background_tasks: BackgroundTasks):
    """Launch an experiment in a background thread."""
    if req.name in _running and _running[req.name] == "running":
        raise HTTPException(400, f"Experiment '{req.name}' is already running.")

    cfg = ExperimentConfig(
        name=req.name,
        environment=req.environment,
        env_kwargs=req.env_kwargs,
        agent_type=req.agent_type,
        agent_kwargs=req.agent_kwargs,
        n_episodes=req.n_episodes,
        max_steps=req.max_steps,
        seed=req.seed,
        device=req.device,
        save_dir="results",
    )
    background_tasks.add_task(_run_experiment, cfg)
    _running[req.name] = "queued"
    return StatusResponse(name=req.name, status="queued")


@app.get("/experiments/{name}")
def get_results(name: str):
    """Return results JSON for a completed experiment."""
    result_path = RESULTS_DIR / name / "results.json"
    if not result_path.exists():
        status = _running.get(name, "not_found")
        raise HTTPException(
            404,
            detail={"name": name, "status": status,
                    "message": "Results not yet available."},
        )
    with open(result_path) as f:
        return json.load(f)


@app.get("/experiments", response_model=List[StatusResponse])
def list_experiments():
    """List all experiments (running or completed)."""
    out = []
    for name, status in _running.items():
        out.append(StatusResponse(name=name, status=status))
    # Also scan results dir for any not in memory
    if RESULTS_DIR.exists():
        for d in RESULTS_DIR.iterdir():
            if d.is_dir() and d.name not in _running:
                rp = d / "results.json"
                out.append(StatusResponse(
                    name=d.name,
                    status="done" if rp.exists() else "incomplete",
                ))
    return out


@app.get("/experiments/{name}/metrics")
def get_metrics(name: str):
    """Return only the aggregated metrics for a completed experiment."""
    result_path = RESULTS_DIR / name / "results.json"
    if not result_path.exists():
        raise HTTPException(404, detail=f"No results for '{name}'.")
    with open(result_path) as f:
        data = json.load(f)
    return data.get("metrics", {})
