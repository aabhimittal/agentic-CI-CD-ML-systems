"""Agentic CI/CD for ML + Robotics — deployments as decision trees, not scripts.

Public surface: the data model plus the high-level orchestrator entry point.
"""

from __future__ import annotations

from .memory import DeploymentMemory
from .models import (
    Decision,
    DeploymentReport,
    DeploymentRequest,
    GateThresholds,
    HealthSnapshot,
    RolloutStep,
    RolloutStrategy,
    RolloutStrategyPlan,
    StepRecord,
    Verdict,
)

__all__ = [
    "Decision",
    "DeploymentMemory",
    "DeploymentReport",
    "DeploymentRequest",
    "GateThresholds",
    "HealthSnapshot",
    "RolloutStep",
    "RolloutStrategy",
    "RolloutStrategyPlan",
    "StepRecord",
    "Verdict",
]

__version__ = "0.2.0"
