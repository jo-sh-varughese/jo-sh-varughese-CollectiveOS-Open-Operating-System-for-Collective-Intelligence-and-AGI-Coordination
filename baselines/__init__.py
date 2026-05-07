"""
baselines/__init__.py + commnet.py
===================================
Baseline suite for CollectiveOS-Bench.

Available baselines:
    CentralizedOptimalController  — analytic optimum (Commons)
    CentralizedPPOController      — joint-state PPO (CTCE)
    CommNetAgent                  — communication-augmented agent
    IndependentPPOBaseline        — wrapper for experiment runner
"""

from .centralized import CentralizedOptimalController, CentralizedPPOController
from .commnet import CommNetAgent

__all__ = [
    "CentralizedOptimalController",
    "CentralizedPPOController",
    "CommNetAgent",
]
