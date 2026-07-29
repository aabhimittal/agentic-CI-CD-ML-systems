"""The orchestrator — wires every component behind the state machine.

It loads the ASL deployment definition, binds Python task handlers to a per-run
:class:`_RunState`, and executes it. The handlers are where the agent, robotics
simulator, Prometheus provider, decision tree, and mock cluster meet:

    plan_strategy → provision → (rollout_step → gate)* → promote | rollback

The result is a fully populated :class:`DeploymentReport` — the explainable record
of what strategy was chosen, what each gate saw, which path the decision tree took,
and, on failure, the agent's rollback reasoning.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .agent import Agent
from .decision_tree import Node, build_promotion_tree, evaluate
from .kubernetes import MockCluster
from .models import (
    Decision,
    DeploymentReport,
    DeploymentRequest,
    RolloutStep,
    RolloutStrategyPlan,
    StepRecord,
    Verdict,
)
from .prometheus import PrometheusProvider
from .robotics_sim import BehaviorProfile, FleetSimulator
from .state_machine import Execution, StateMachine
from . import strategies

# Simulated seconds from a gate breach to the fleet reaching a safe state.
# The agentic path detects at the gate, so this is small and fixed.
_ROLLBACK_DRAIN_SECONDS = 15.0
# How many "hold" verdicts to tolerate on one step before escalating to rollback.
_MAX_HOLDS = 2
# Simulated bake time at 100% traffic before the candidate is finalized, and how
# many observation windows the bake spans. Multiple windows matter: a fault that
# needs sustained full-fleet load (thermal, memory growth, cache churn) may only
# appear in the second window.
_SOAK_SECONDS = 60.0
_SOAK_ROUNDS = 2


def _repo_root() -> Path:
    # src/agentic_deploy/orchestrator.py → parents[2] is the repo root (editable install).
    return Path(__file__).resolve().parents[2]


def default_asl_path() -> Path:
    return _repo_root() / "deploy" / "stepfunctions" / "deployment.asl.json"


def default_k8s_dir() -> Path:
    return _repo_root() / "deploy" / "k8s"


@dataclass
class _RunState:
    """Mutable state shared by the task handlers for a single deployment."""

    request: DeploymentRequest
    agent: Agent
    simulator: FleetSimulator
    prometheus: PrometheusProvider
    cluster: MockCluster
    tree: Node
    k8s_dir: Path | None
    plan: RolloutStrategyPlan | None = None
    report: DeploymentReport | None = None
    next_index: int = 0
    holds: int = 0
    soak_rounds: int = 0
    window: int = 0     # monotonic observation-window counter — time advances
    elapsed: float = 0.0


class Orchestrator:
    """Runs one deployment end-to-end via the ASL state machine."""

    def __init__(self, asl_path: str | Path | None = None) -> None:
        self.asl_path = Path(asl_path) if asl_path else default_asl_path()
        with self.asl_path.open() as fh:
            self.definition = json.load(fh)

    def deploy(
        self,
        request: DeploymentRequest,
        profile: BehaviorProfile,
        *,
        agent: Agent | None = None,
        seed: int = 0,
        k8s_dir: str | Path | None = None,
        memory=None,
    ) -> DeploymentReport:
        # When a memory store is given without an explicit agent, build the agent
        # around it so plans reflect the service's deployment history.
        agent = agent or Agent(memory=memory)
        run = _RunState(
            request=request,
            agent=agent,
            simulator=FleetSimulator(profile, request.fleet_size, seed=seed),
            prometheus=PrometheusProvider(request.service, request.model_version),
            cluster=MockCluster(request.service, request.previous_version, request.model_version),
            tree=build_promotion_tree(request.thresholds),
            k8s_dir=Path(k8s_dir) if k8s_dir else _maybe(default_k8s_dir()),
        )
        machine = StateMachine(self.definition, self._handlers(run))
        execution = machine.run()
        report = self._finalize(run, execution)
        if memory is not None:
            memory.record(report)  # future plans for this service learn from this run
        return report

    # ---- task handlers -----------------------------------------------------

    def _handlers(self, run: _RunState) -> dict[str, Any]:
        return {
            "plan_strategy": lambda ctx: self._plan_strategy(run, ctx),
            "provision": lambda ctx: self._provision(run, ctx),
            "rollout_step": lambda ctx: self._rollout_step(run, ctx),
            "soak_test": lambda ctx: self._soak_test(run, ctx),
            "rollback": lambda ctx: self._rollback(run, ctx),
            "promote": lambda ctx: self._promote(run, ctx),
        }

    def _plan_strategy(self, run: _RunState, ctx: dict[str, Any]) -> dict[str, Any]:
        run.plan = run.agent.plan(run.request)
        run.report = DeploymentReport(request=run.request, plan=run.plan)
        return {"strategy": str(run.plan.strategy), "risk_score": run.plan.risk_score}

    def _provision(self, run: _RunState, ctx: dict[str, Any]) -> dict[str, Any]:
        # Apply the Service + baseline manifests if they're on disk.
        if run.k8s_dir is not None and run.k8s_dir.exists():
            for name in ("service.yaml", "deployment-blue.yaml", "deployment-green.yaml"):
                path = run.k8s_dir / name
                if path.exists():
                    run.cluster.apply(path)
        return {}

    def _rollout_step(self, run: _RunState, ctx: dict[str, Any]) -> dict[str, Any]:
        assert run.plan is not None and run.report is not None
        idx = run.next_index
        step = run.plan.steps[idx]

        # 1. Shift traffic / cut over in the cluster.
        strategies.apply_step(run.cluster, run.plan.strategy, step)

        # 2-4. Observe, evaluate the tree, and apply hold-escalation (dwell time
        # is accounted inside the gate).
        decision = self._gate(run, step)

        # 5. Compute routing flags for the ASL Choice state.
        has_more = False
        if decision.verdict == Verdict.PROMOTE:
            if idx + 1 < len(run.plan.steps):
                run.next_index = idx + 1
                has_more = True
        # On HOLD we re-run the same step (next_index unchanged); on ROLLBACK we stop.
        return {"verdict": str(decision.verdict), "has_more_steps": has_more}

    def _soak_test(self, run: _RunState, ctx: dict[str, Any]) -> dict[str, Any]:
        """Bake at 100%: re-observe after full rollout to catch late-activating faults.

        Every gate promoted, so the whole fleet now runs the candidate — but a
        latent fault can surface only after full exposure (thermal load, cache
        churn, rare SKUs). The soak observes at a *later* step index so such
        faults activate, and routes through the same decision tree. Holds re-soak
        with the same escalation cap as rollout steps.
        """

        assert run.plan is not None and run.report is not None
        last_index = run.plan.steps[-1].index
        soak_step = RolloutStep(
            index=last_index + 1 + run.soak_rounds,
            traffic_pct=100.0,
            dwell_seconds=_SOAK_SECONDS,
            phase="soak",
        )
        run.soak_rounds += 1
        decision = self._gate(run, soak_step)
        return {
            "verdict": str(decision.verdict),
            "has_more_steps": False,
            "soak_complete": run.soak_rounds >= _SOAK_ROUNDS,
        }

    def _gate(self, run: _RunState, step: RolloutStep) -> Decision:
        """Shared gate: observe fleet health, evaluate the tree, escalate stuck holds.

        Observations are keyed by a monotonic *window* counter, not the step
        index: time advances with every observation, including re-observations
        after a hold and soak rounds. That is what lets transient conditions
        (telemetry outages, warm-up spikes) genuinely recover — and latent
        faults genuinely activate — while a step is being re-measured.
        """

        assert run.report is not None
        run.elapsed += step.dwell_seconds

        snapshot = run.simulator.observe(step.traffic_pct, run.window)
        run.window += 1
        run.prometheus.record(snapshot)
        decision = evaluate(run.tree, snapshot)

        # Escalate a stuck "hold" to a rollback so we never loop forever.
        if decision.verdict == Verdict.HOLD:
            run.holds += 1
            if run.holds > _MAX_HOLDS:
                decision = Decision(
                    verdict=Verdict.ROLLBACK,
                    reason=f"held {run.holds - 1}x without recovery; escalating to rollback",
                    path=decision.path + ["[escalated→rollback]"],
                    breached=decision.breached or ["task_cycle_time_s"],
                )
        else:
            run.holds = 0

        run.report.steps.append(StepRecord(step=step, snapshot=snapshot, decision=decision))
        return decision

    def _rollback(self, run: _RunState, ctx: dict[str, Any]) -> dict[str, Any]:
        assert run.report is not None
        strategies.rollback(run.cluster)
        last = run.report.steps[-1]
        run.report.rollback_seconds = _ROLLBACK_DRAIN_SECONDS
        run.report.rollback_reasoning = run.agent.explain_rollback(
            run.request, last.snapshot, last.decision.breached
        )
        run.report.outcome = "rolled_back"
        return {}

    def _promote(self, run: _RunState, ctx: dict[str, Any]) -> dict[str, Any]:
        assert run.report is not None
        strategies.promote(run.cluster)
        run.report.outcome = "promoted"
        return {}

    # ---- finalize ----------------------------------------------------------

    def _finalize(self, run: _RunState, execution: Execution) -> DeploymentReport:
        assert run.report is not None
        run.report.duration_seconds = run.elapsed
        # Attach the state-machine trace and final cluster state for the report.
        run.report.state_machine_path = execution.visited
        run.report.cluster_state = run.cluster.state()
        return run.report


def _maybe(path: Path) -> Path | None:
    return path if path.exists() else None


def deploy(
    request: DeploymentRequest,
    profile: BehaviorProfile,
    *,
    agent: Agent | None = None,
    seed: int = 0,
    asl_path: str | Path | None = None,
    k8s_dir: str | Path | None = None,
    memory=None,
) -> DeploymentReport:
    """Convenience wrapper: build an Orchestrator and run one deployment."""

    return Orchestrator(asl_path=asl_path).deploy(
        request, profile, agent=agent, seed=seed, k8s_dir=k8s_dir, memory=memory
    )
