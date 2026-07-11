"""End-to-end walkthrough of the agentic deployment engine.

Runs the three shipped scenarios (healthy canary, quality regression, latent
safety incident) plus a programmatic low-risk change that the agent routes to a
blue/green cut-over. No API key required — the deterministic agent is used unless
ANTHROPIC_API_KEY is set.

Usage:
    python examples/run_demo.py
"""

from __future__ import annotations

from pathlib import Path

from agentic_deploy.agent import Agent
from agentic_deploy.cli import _render_plain, load_scenario
from agentic_deploy.models import DeploymentRequest, GateThresholds
from agentic_deploy.orchestrator import deploy
from agentic_deploy.robotics_sim import BehaviorProfile

SCENARIOS = Path(__file__).resolve().parents[1] / "scenarios"


def _run_scenario(name: str) -> None:
    request, profile, seed = load_scenario(SCENARIOS / name)
    report = deploy(request, profile, agent=Agent(use_llm=False), seed=seed)
    print(_render_plain(report))
    print()


def _run_blue_green() -> None:
    # Low blast radius + non-critical task → the agent picks blue/green.
    request = DeploymentRequest(
        service="pallet-router",
        model_version="route-net:v1.4.0",
        previous_version="route-net:v1.3.9",
        fleet_size=8,
        task_criticality=0.2,
        change_summary="config tweak; no weight change",
        thresholds=GateThresholds(),
    )
    profile = BehaviorProfile(
        cand_error_rate=0.009,
        cand_task_success_rate=0.985,
        cand_latency_p99_ms=170,
        cand_cycle_time_s=7.9,
        noise=0.01,
    )
    report = deploy(request, profile, agent=Agent(use_llm=False), seed=3)
    print(_render_plain(report))
    print()


def main() -> None:
    for name in ("healthy_canary.yaml", "regression_rollback.yaml", "safety_incident.yaml"):
        _run_scenario(name)
    _run_blue_green()


if __name__ == "__main__":
    main()
