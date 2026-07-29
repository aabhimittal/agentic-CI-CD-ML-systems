"""Core data model for the agentic deployment engine.

These types are the vocabulary shared by every module: the agent plans a
:class:`RolloutStrategy`, the state machine executes :class:`RolloutStep` s,
the simulator/prometheus provider emit :class:`HealthSnapshot` s, the decision
tree turns a snapshot into a :class:`Decision`, and the whole run is captured in
a :class:`DeploymentReport`.

Everything is a plain dataclass so reports serialize cleanly to JSON and stay
diff-friendly in tests.
"""

from __future__ import annotations

import dataclasses
import enum
import time
from dataclasses import dataclass, field
from typing import Any


class RolloutStrategy(str, enum.Enum):
    """How new model weights reach the fleet."""

    CANARY = "canary"          # shift traffic in graduated steps, gate each step
    BLUE_GREEN = "blue_green"  # stand up a parallel stack, cut over atomically

    def __str__(self) -> str:  # nicer CLI/report output than "RolloutStrategy.CANARY"
        return self.value


class Verdict(str, enum.Enum):
    """The go/no-go outcome the decision tree returns at a gate."""

    PROMOTE = "promote"    # health is good, advance to the next step
    HOLD = "hold"          # inconclusive, wait and re-measure
    ROLLBACK = "rollback"  # health breached a guardrail, abort

    def __str__(self) -> str:
        return self.value


@dataclass
class HealthSnapshot:
    """A point-in-time reading of the signals that gate a promotion.

    ``service_*`` fields come from the Prometheus provider; ``task_*`` and
    ``safety_incidents`` come from the robotics simulator. ``traffic_pct`` is the
    share of the fleet currently running the candidate model.
    """

    traffic_pct: float
    service_error_rate: float      # fraction of inference requests that errored (0..1)
    service_latency_p99_ms: float  # 99th percentile inference latency
    task_success_rate: float       # fraction of robot tasks completed successfully (0..1)
    task_cycle_time_s: float       # mean seconds per robot task
    safety_incidents: int          # count of safety-envelope violations in the window
    telemetry_ok: bool = True      # False when fleet telemetry is stale/lost this window
    step_index: int = 0
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


@dataclass
class GateThresholds:
    """Guardrails the decision tree compares a :class:`HealthSnapshot` against.

    Defaults are deliberately conservative; scenarios override them.
    """

    max_error_rate: float = 0.05
    max_latency_p99_ms: float = 500.0
    min_task_success_rate: float = 0.90
    max_cycle_time_s: float = 12.0
    max_safety_incidents: int = 0

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


@dataclass
class Decision:
    """The result of evaluating one gate, with the reasoning that produced it.

    ``path`` is the ordered list of decision-tree node labels that were visited
    to reach ``verdict`` — this is what makes the decision auditable rather than
    a black box.
    """

    verdict: Verdict
    reason: str
    path: list[str] = field(default_factory=list)
    breached: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        d = dataclasses.asdict(self)
        d["verdict"] = str(self.verdict)
        return d


@dataclass
class RolloutStep:
    """One planned increment of a rollout (e.g. shift traffic to 25%)."""

    index: int
    traffic_pct: float
    dwell_seconds: float  # how long to observe before evaluating the gate
    phase: str = "rollout"  # "rollout" | "soak" (post-100% bake observation)

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


@dataclass
class RolloutStrategyPlan:
    """The agent's chosen strategy plus the reasoning behind it."""

    strategy: RolloutStrategy
    steps: list[RolloutStep]
    risk_score: float           # 0..1, higher = riskier change
    rationale: str
    source: str = "heuristic"   # "claude" when produced by the LLM, else "heuristic"

    def to_dict(self) -> dict[str, Any]:
        return {
            "strategy": str(self.strategy),
            "steps": [s.to_dict() for s in self.steps],
            "risk_score": self.risk_score,
            "rationale": self.rationale,
            "source": self.source,
        }


@dataclass
class DeploymentRequest:
    """The change to be deployed and the context the agent needs to plan."""

    service: str
    model_version: str            # candidate, e.g. "grasp-net:v2.3.0"
    previous_version: str         # currently serving, for rollback
    fleet_size: int               # number of robots affected
    task_criticality: float       # 0..1, higher = more safety-critical task
    change_summary: str = ""      # human note about what changed in the model
    thresholds: GateThresholds = field(default_factory=GateThresholds)

    def to_dict(self) -> dict[str, Any]:
        d = dataclasses.asdict(self)
        d["thresholds"] = self.thresholds.to_dict()
        return d


@dataclass
class StepRecord:
    """Everything that happened at a single rollout step."""

    step: RolloutStep
    snapshot: HealthSnapshot
    decision: Decision

    def to_dict(self) -> dict[str, Any]:
        return {
            "step": self.step.to_dict(),
            "snapshot": self.snapshot.to_dict(),
            "decision": self.decision.to_dict(),
        }


@dataclass
class DeploymentReport:
    """The explainable record of a full deployment run."""

    request: DeploymentRequest
    plan: RolloutStrategyPlan
    steps: list[StepRecord] = field(default_factory=list)
    outcome: str = "pending"          # "promoted" | "rolled_back" | "failed"
    rollback_reasoning: str = ""      # agent's narrative when outcome == rolled_back
    duration_seconds: float = 0.0     # simulated wall-clock of the rollout
    rollback_seconds: float = 0.0     # simulated time from breach detection to safe state
    state_machine_path: list[str] = field(default_factory=list)  # ASL states visited
    cluster_state: dict[str, Any] = field(default_factory=dict)  # final mock-cluster state

    def to_dict(self) -> dict[str, Any]:
        return {
            "request": self.request.to_dict(),
            "plan": self.plan.to_dict(),
            "steps": [s.to_dict() for s in self.steps],
            "outcome": self.outcome,
            "rollback_reasoning": self.rollback_reasoning,
            "duration_seconds": round(self.duration_seconds, 2),
            "rollback_seconds": round(self.rollback_seconds, 2),
            "state_machine_path": list(self.state_machine_path),
            "cluster_state": self.cluster_state,
        }
