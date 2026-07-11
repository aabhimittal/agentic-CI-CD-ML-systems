"""Tests for the deterministic (no-LLM) agent path."""

from __future__ import annotations

from agentic_deploy.agent import Agent
from agentic_deploy.models import (
    DeploymentRequest,
    GateThresholds,
    HealthSnapshot,
    RolloutStrategy,
)


def _request(**overrides) -> DeploymentRequest:
    base = dict(
        service="grasp-planner",
        model_version="cand",
        previous_version="base",
        fleet_size=50,
        task_criticality=0.5,
        change_summary="",
        thresholds=GateThresholds(),
    )
    base.update(overrides)
    return DeploymentRequest(**base)


def test_high_risk_change_gets_canary():
    agent = Agent(use_llm=False)
    plan = agent.plan(_request(fleet_size=90, task_criticality=0.9))
    assert plan.strategy == RolloutStrategy.CANARY
    assert plan.steps[-1].traffic_pct == 100.0
    assert len(plan.steps) > 1
    assert plan.source == "heuristic"


def test_low_risk_change_gets_blue_green():
    agent = Agent(use_llm=False)
    plan = agent.plan(_request(fleet_size=6, task_criticality=0.15))
    assert plan.strategy == RolloutStrategy.BLUE_GREEN
    assert [s.traffic_pct for s in plan.steps] == [100.0]


def test_risk_score_monotonic_in_criticality():
    agent = Agent(use_llm=False)
    low = agent.plan(_request(task_criticality=0.1)).risk_score
    high = agent.plan(_request(task_criticality=0.9)).risk_score
    assert high > low
    assert 0.0 <= low <= 1.0 and 0.0 <= high <= 1.0


def test_change_keywords_raise_risk():
    agent = Agent(use_llm=False)
    plain = agent.plan(_request(change_summary="tweak")).risk_score
    scary = agent.plan(_request(change_summary="major architecture retrain")).risk_score
    assert scary > plain


def test_plan_is_deterministic():
    agent = Agent(use_llm=False)
    r = _request(fleet_size=70, task_criticality=0.6)
    assert agent.plan(r).to_dict() == agent.plan(r).to_dict()


def test_rollback_explanation_mentions_signal_and_revert_target():
    agent = Agent(use_llm=False)
    request = _request(fleet_size=40, previous_version="grasp-net:v2.2.1")
    snap = HealthSnapshot(
        traffic_pct=50.0,
        service_error_rate=0.2,
        service_latency_p99_ms=180.0,
        task_success_rate=0.7,
        task_cycle_time_s=8.0,
        safety_incidents=0,
    )
    text = agent.explain_rollback(request, snap, ["task_success_rate"])
    assert "task_success_rate" in text
    assert "grasp-net:v2.2.1" in text


def test_agent_never_uses_llm_when_forced_off():
    agent = Agent(use_llm=False)
    assert agent.use_llm is False
    assert agent._client is None
