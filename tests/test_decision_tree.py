"""Tests for the promotion decision-tree engine."""

from __future__ import annotations

from agentic_deploy.decision_tree import build_promotion_tree, evaluate
from agentic_deploy.models import GateThresholds, HealthSnapshot, Verdict


def _snapshot(**overrides) -> HealthSnapshot:
    base = dict(
        traffic_pct=50.0,
        service_error_rate=0.01,
        service_latency_p99_ms=180.0,
        task_success_rate=0.98,
        task_cycle_time_s=8.0,
        safety_incidents=0,
    )
    base.update(overrides)
    return HealthSnapshot(**base)


def test_healthy_snapshot_promotes():
    tree = build_promotion_tree(GateThresholds())
    decision = evaluate(tree, _snapshot())
    assert decision.verdict == Verdict.PROMOTE
    assert decision.breached == []
    assert decision.path[-1] == "[promote]"


def test_safety_incident_is_terminal_rollback():
    tree = build_promotion_tree(GateThresholds())
    # Even with every other signal perfect, a safety incident rolls back.
    decision = evaluate(tree, _snapshot(safety_incidents=1))
    assert decision.verdict == Verdict.ROLLBACK
    assert "safety_incidents" in decision.breached
    # Safety is checked first, so it appears at the front of the path.
    assert decision.path[0].startswith("safety_incident=")


def test_task_success_below_floor_rolls_back():
    tree = build_promotion_tree(GateThresholds())
    decision = evaluate(tree, _snapshot(task_success_rate=0.80))
    assert decision.verdict == Verdict.ROLLBACK
    assert "task_success_rate" in decision.breached


def test_error_rate_breach_rolls_back():
    tree = build_promotion_tree(GateThresholds())
    decision = evaluate(tree, _snapshot(service_error_rate=0.20))
    assert decision.verdict == Verdict.ROLLBACK
    assert "service_error_rate" in decision.breached


def test_latency_breach_rolls_back():
    tree = build_promotion_tree(GateThresholds())
    decision = evaluate(tree, _snapshot(service_latency_p99_ms=900.0))
    assert decision.verdict == Verdict.ROLLBACK
    assert "service_latency_p99_ms" in decision.breached


def test_cycle_time_regression_holds_not_rollback():
    tree = build_promotion_tree(GateThresholds())
    decision = evaluate(tree, _snapshot(task_cycle_time_s=20.0))
    assert decision.verdict == Verdict.HOLD
    assert "task_cycle_time_s" in decision.breached


def test_thresholds_are_honored():
    strict = GateThresholds(min_task_success_rate=0.99)
    tree = build_promotion_tree(strict)
    # 0.985 passes the default 0.90 floor but fails the strict 0.99 floor.
    decision = evaluate(tree, _snapshot(task_success_rate=0.985))
    assert decision.verdict == Verdict.ROLLBACK


def test_decision_serializes():
    tree = build_promotion_tree(GateThresholds())
    decision = evaluate(tree, _snapshot(safety_incidents=2))
    d = decision.to_dict()
    assert d["verdict"] == "rollback"
    assert isinstance(d["path"], list)
