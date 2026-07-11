"""End-to-end tests driving full scenarios through the orchestrator."""

from __future__ import annotations

from pathlib import Path

from agentic_deploy.agent import Agent
from agentic_deploy.cli import load_scenario
from agentic_deploy.orchestrator import deploy

REPO = Path(__file__).resolve().parents[1]
SCENARIOS = REPO / "scenarios"


def _run(name: str):
    request, profile, seed = load_scenario(SCENARIOS / name)
    return deploy(request, profile, agent=Agent(use_llm=False), seed=seed)


def test_healthy_scenario_promotes():
    report = _run("healthy_canary.yaml")
    assert report.outcome == "promoted"
    assert report.state_machine_path[-1] == "Promoted"
    # Reached full traffic and finalized.
    assert any(s.step.traffic_pct >= 100.0 for s in report.steps)
    assert report.cluster_state["candidate_weight"] == 100.0


def test_regression_scenario_rolls_back_before_full_fleet():
    report = _run("regression_rollback.yaml")
    assert report.outcome == "rolled_back"
    assert report.state_machine_path[-1] == "RolledBack"
    # The whole point: caught before reaching 100% of the fleet.
    assert all(s.step.traffic_pct < 100.0 for s in report.steps)
    assert report.cluster_state["candidate_weight"] == 0.0
    assert report.rollback_reasoning  # agent produced a narrative


def test_safety_scenario_rolls_back_on_safety_signal():
    report = _run("safety_incident.yaml")
    assert report.outcome == "rolled_back"
    last = report.steps[-1]
    assert "safety_incidents" in last.decision.breached
    assert last.snapshot.safety_incidents >= 1


def test_report_is_json_serializable():
    report = _run("regression_rollback.yaml")
    d = report.to_dict()
    assert d["outcome"] == "rolled_back"
    assert d["plan"]["strategy"] in ("canary", "blue_green")
    assert isinstance(d["steps"], list) and d["steps"]


def test_canary_chosen_for_safety_critical_change():
    report = _run("healthy_canary.yaml")
    assert report.plan.strategy.value == "canary"
    assert len(report.plan.steps) > 1
