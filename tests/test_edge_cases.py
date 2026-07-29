"""Industrial edge cases: the situations that break naive deployment pipelines.

Each test encodes a failure mode seen in real fleets — telemetry outages, faults
that pass every canary gate, warm-up transients, boundary-valued metrics, tiny
fleets — and asserts the engine's fail-safe behavior.
"""

from __future__ import annotations

from pathlib import Path

from agentic_deploy.agent import Agent
from agentic_deploy.cli import load_scenario
from agentic_deploy.decision_tree import build_promotion_tree, evaluate
from agentic_deploy.models import (
    DeploymentRequest,
    GateThresholds,
    HealthSnapshot,
    Verdict,
)
from agentic_deploy.orchestrator import deploy
from agentic_deploy.robotics_sim import BehaviorProfile, FleetSimulator

SCENARIOS = Path(__file__).resolve().parents[1] / "scenarios"


def _request(**overrides) -> DeploymentRequest:
    base = dict(
        service="grasp-planner",
        model_version="cand",
        previous_version="base",
        fleet_size=60,
        task_criticality=0.8,
        change_summary="",
        thresholds=GateThresholds(),
    )
    base.update(overrides)
    return DeploymentRequest(**base)


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


# ---- telemetry loss: unknown must never read as healthy ---------------------


def test_missing_telemetry_holds_never_promotes():
    tree = build_promotion_tree(GateThresholds())
    # Perfect-looking (but stale) numbers with telemetry down → HOLD, not promote.
    decision = evaluate(tree, _snapshot(telemetry_ok=False))
    assert decision.verdict == Verdict.HOLD
    assert "telemetry_ok" in decision.breached
    assert decision.path[0] == "telemetry_missing=yes"


def test_sustained_telemetry_outage_escalates_to_rollback():
    request, profile, seed = load_scenario(SCENARIOS / "telemetry_dropout.yaml")
    report = deploy(request, profile, agent=Agent(use_llm=False), seed=seed)
    assert report.outcome == "rolled_back"
    # The rollout held (fail-safe) before escalating — it never promoted blind.
    verdicts = [str(r.decision.verdict) for r in report.steps]
    assert "hold" in verdicts
    assert report.cluster_state["candidate_weight"] == 0.0


def test_brief_telemetry_blip_recovers_and_promotes():
    # A one-window outage: the gate holds (fail-safe), time advances, telemetry
    # returns on re-observation, and the healthy candidate still gets promoted.
    # A blip must not fail a good deployment.
    request, profile, seed = load_scenario(SCENARIOS / "telemetry_dropout.yaml")
    profile.telemetry_dropout_steps = (1,)  # dark for a single window only
    report = deploy(request, profile, agent=Agent(use_llm=False), seed=seed)
    assert report.outcome == "promoted"
    verdicts = [str(r.decision.verdict) for r in report.steps]
    assert "hold" in verdicts  # the blip was held on, not ignored


# ---- faults that pass every canary gate: the soak (bake) catches them -------


def test_late_latent_fault_caught_in_soak():
    request, profile, seed = load_scenario(SCENARIOS / "late_latent_soak.yaml")
    report = deploy(request, profile, agent=Agent(use_llm=False), seed=seed)
    assert report.outcome == "rolled_back"
    # Every rollout gate promoted; the breach was found during soak.
    rollout_records = [r for r in report.steps if r.step.phase == "rollout"]
    soak_records = [r for r in report.steps if r.step.phase == "soak"]
    assert all(r.decision.verdict == Verdict.PROMOTE for r in rollout_records)
    assert soak_records and str(soak_records[-1].decision.verdict) == "rollback"
    assert "SoakTest" in report.state_machine_path
    assert report.cluster_state["candidate_weight"] == 0.0


def test_healthy_run_soaks_then_promotes():
    request, profile, seed = load_scenario(SCENARIOS / "healthy_canary.yaml")
    report = deploy(request, profile, agent=Agent(use_llm=False), seed=seed)
    assert report.outcome == "promoted"
    soak_records = [r for r in report.steps if r.step.phase == "soak"]
    assert len(soak_records) >= 2  # the bake spans multiple observation windows
    assert all(r.decision.verdict == Verdict.PROMOTE for r in soak_records)
    assert report.state_machine_path[-1] == "Promoted"


def test_blue_green_fault_contained_and_reverted():
    # Blue/green cuts over atomically, so a faulty candidate does briefly serve
    # 100% — the gate must catch it at the first observation and fully revert.
    request = _request(fleet_size=6, task_criticality=0.2)  # low risk → blue/green
    profile = BehaviorProfile(cand_task_success_rate=0.70, cand_error_rate=0.15, noise=0.01)
    report = deploy(request, profile, agent=Agent(use_llm=False), seed=9)
    assert report.plan.strategy.value == "blue_green"
    assert report.outcome == "rolled_back"
    state = report.cluster_state
    assert state["candidate_weight"] == 0.0
    assert state["active_color"] == "blue"
    assert state["deployments"]["green"]["replicas"] == 0


# ---- transients: hold, recover, promote (no false rollback) -----------------


def test_warmup_transient_holds_then_recovers_and_promotes():
    # Cycle time spikes when the fleet first goes to 100% (cache-cold warm-up,
    # window 3) then clears — a soft HOLD signal. The gate holds, re-measures a
    # window later, and promotes once it recovers, without ever rolling back a
    # healthy candidate.
    profile = BehaviorProfile(
        cand_cycle_time_s=15.0,             # over the 12s guardrail...
        regression_activates_at_step=3,     # ...at the 100% window
        regression_deactivates_at_step=4,   # ...for exactly one window
        noise=0.0,
    )
    request = _request()
    report = deploy(request, profile, agent=Agent(use_llm=False), seed=2)
    assert report.outcome == "promoted"
    verdicts = [str(r.decision.verdict) for r in report.steps]
    assert "hold" in verdicts       # the transient was observed and held on
    assert "rollback" not in verdicts


def test_persistent_soft_regression_escalates_to_rollback():
    # A cycle-time regression that never recovers: HOLD is not a loophole —
    # after the hold budget it escalates to a rollback.
    profile = BehaviorProfile(cand_cycle_time_s=15.0, noise=0.0)
    report = deploy(_request(), profile, agent=Agent(use_llm=False), seed=2)
    assert report.outcome == "rolled_back"
    last = report.steps[-1]
    assert "escalating to rollback" in last.decision.reason


# ---- boundary values: documented comparator semantics -----------------------


def test_metric_exactly_at_threshold_passes():
    # Guardrails are strict inequalities: exactly-at-threshold is compliant.
    # This is deliberate and documented — assert it so a refactor to >= is caught.
    thr = GateThresholds()
    tree = build_promotion_tree(thr)
    decision = evaluate(
        tree,
        _snapshot(
            service_error_rate=thr.max_error_rate,
            service_latency_p99_ms=thr.max_latency_p99_ms,
            task_success_rate=thr.min_task_success_rate,
            task_cycle_time_s=thr.max_cycle_time_s,
            safety_incidents=thr.max_safety_incidents,
        ),
    )
    assert decision.verdict == Verdict.PROMOTE


def test_epsilon_over_threshold_fails():
    thr = GateThresholds()
    tree = build_promotion_tree(thr)
    decision = evaluate(tree, _snapshot(service_error_rate=thr.max_error_rate + 1e-9))
    assert decision.verdict == Verdict.ROLLBACK


# ---- degenerate fleets ------------------------------------------------------


def test_single_robot_cell_deploys():
    # A one-robot work cell: blast radius is minimal but safety still gates.
    request = _request(fleet_size=1, task_criticality=0.9)
    profile = BehaviorProfile(noise=0.01)
    report = deploy(request, profile, agent=Agent(use_llm=False), seed=4)
    assert report.outcome == "promoted"
    assert report.plan.strategy.value == "canary"  # criticality alone forces canary


def test_zero_fleet_still_runs_and_reports():
    # A fleet of zero robots (e.g. pre-provisioned line) must not crash the
    # planner or the pipeline; it just produces a trivially promotable rollout.
    request = _request(fleet_size=0, task_criticality=0.1)
    profile = BehaviorProfile(noise=0.0)
    report = deploy(request, profile, agent=Agent(use_llm=False), seed=1)
    assert report.outcome in ("promoted", "rolled_back")
    assert 0.0 <= report.plan.risk_score <= 1.0
