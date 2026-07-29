"""Tests for deployment memory and experience-informed planning."""

from __future__ import annotations

from pathlib import Path

from agentic_deploy.agent import Agent
from agentic_deploy.cli import load_scenario
from agentic_deploy.memory import DeploymentMemory
from agentic_deploy.models import DeploymentRequest, GateThresholds, RolloutStrategy
from agentic_deploy.orchestrator import deploy

SCENARIOS = Path(__file__).resolve().parents[1] / "scenarios"


def _request(**overrides) -> DeploymentRequest:
    base = dict(
        service="grasp-planner",
        model_version="cand",
        previous_version="base",
        fleet_size=20,
        task_criticality=0.3,
        change_summary="",
        thresholds=GateThresholds(),
    )
    base.update(overrides)
    return DeploymentRequest(**base)


def test_memory_records_outcomes(tmp_path):
    memory = DeploymentMemory(tmp_path / "mem.json")
    request, profile, seed = load_scenario(SCENARIOS / "regression_rollback.yaml")
    deploy(request, profile, agent=Agent(use_llm=False, memory=memory), seed=seed, memory=memory)
    assert len(memory) == 1
    entry = memory.history("grasp-planner")[0]
    assert entry["outcome"] == "rolled_back"
    assert "task_success_rate" in entry["breached"]
    # Persisted across a reload.
    reloaded = DeploymentMemory(tmp_path / "mem.json")
    assert reloaded.last_outcome("grasp-planner") == "rolled_back"


def test_rollback_history_makes_next_plan_more_cautious(tmp_path):
    memory = DeploymentMemory(tmp_path / "mem.json")

    # Fresh service, low risk → blue/green.
    naive = Agent(use_llm=False).plan(_request())
    assert naive.strategy == RolloutStrategy.BLUE_GREEN

    # Record a rollback for the service, then re-plan the same change.
    request, profile, seed = load_scenario(SCENARIOS / "regression_rollback.yaml")
    deploy(request, profile, agent=Agent(use_llm=False, memory=memory), seed=seed, memory=memory)

    informed = Agent(use_llm=False, memory=memory).plan(_request())
    # The same low-risk change is now canaried with a 5% bake-in first step.
    assert informed.strategy == RolloutStrategy.CANARY
    assert informed.steps[0].traffic_pct == 5.0
    assert informed.risk_score > naive.risk_score
    assert "rolled back" in informed.rationale


def test_bake_in_step_prepended_to_existing_canary(tmp_path):
    memory = DeploymentMemory(tmp_path / "mem.json")
    request, profile, seed = load_scenario(SCENARIOS / "regression_rollback.yaml")
    deploy(request, profile, agent=Agent(use_llm=False, memory=memory), seed=seed, memory=memory)

    plan = Agent(use_llm=False, memory=memory).plan(
        _request(fleet_size=90, task_criticality=0.9)  # already canary-worthy
    )
    assert plan.strategy == RolloutStrategy.CANARY
    assert [s.traffic_pct for s in plan.steps][:2] == [5.0, 10.0]
    # Step indices stay contiguous after the prepend.
    assert [s.index for s in plan.steps] == list(range(len(plan.steps)))


def test_memory_scoped_per_service(tmp_path):
    memory = DeploymentMemory(tmp_path / "mem.json")
    request, profile, seed = load_scenario(SCENARIOS / "regression_rollback.yaml")
    deploy(request, profile, agent=Agent(use_llm=False, memory=memory), seed=seed, memory=memory)

    # A different service is unaffected by grasp-planner's history.
    other = Agent(use_llm=False, memory=memory).plan(_request(service="pallet-router"))
    assert other.strategy == RolloutStrategy.BLUE_GREEN
    assert memory.failure_rate("pallet-router") == 0.0


def test_clean_streak_restores_confidence(tmp_path):
    memory = DeploymentMemory(tmp_path / "mem.json")
    reg_request, reg_profile, reg_seed = load_scenario(SCENARIOS / "regression_rollback.yaml")
    deploy(reg_request, reg_profile, agent=Agent(use_llm=False, memory=memory),
           seed=reg_seed, memory=memory)

    ok_request, ok_profile, ok_seed = load_scenario(SCENARIOS / "healthy_canary.yaml")
    deploy(ok_request, ok_profile, agent=Agent(use_llm=False, memory=memory),
           seed=ok_seed, memory=memory)

    # Last outcome is now a promotion → no 5% bake-in forced anymore.
    assert memory.last_outcome("grasp-planner") == "promoted"
    plan = Agent(use_llm=False, memory=memory).plan(_request(fleet_size=90, task_criticality=0.9))
    assert plan.steps[0].traffic_pct == 10.0


def test_corrupt_store_does_not_block_deployments(tmp_path):
    path = tmp_path / "mem.json"
    path.write_text("{not valid json")
    memory = DeploymentMemory(path)  # must not raise
    assert len(memory) == 0
    # The corrupt file is preserved for inspection.
    assert path.with_suffix(".json.corrupt").exists()
