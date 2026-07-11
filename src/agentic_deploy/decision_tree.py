"""The decision-tree engine — the core idea of this project.

A GitOps promotion gate is usually a script: a linear sequence of `if metric >
threshold: rollback` checks buried in pipeline YAML. Here the gate is an explicit
**decision tree**. Each internal node asks one question of a
:class:`~agentic_deploy.models.HealthSnapshot`; each leaf is a
:class:`~agentic_deploy.models.Verdict`. Evaluating the tree records the exact
path of node labels visited, so every promote / hold / rollback comes with a
machine- and human-readable justification instead of an opaque exit code.

The tree is data, not control flow, which is what lets the agent *tune* it (via
:class:`~agentic_deploy.models.GateThresholds`) and lets a report replay exactly
why a decision was made.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from .models import Decision, GateThresholds, HealthSnapshot, Verdict

# A predicate answers one yes/no question about a snapshot.
Predicate = Callable[[HealthSnapshot], bool]


@dataclass
class Node:
    """A node in the promotion decision tree.

    A node is a *leaf* when ``verdict`` is set. Otherwise it is a *branch*: it
    evaluates ``predicate`` and descends into ``if_true`` or ``if_false``.
    ``breach_signal``, when present on a branch whose predicate detects a
    guardrail breach, is recorded so the report can name the offending signal.
    """

    label: str
    predicate: Predicate | None = None
    if_true: "Node | None" = None
    if_false: "Node | None" = None
    verdict: Verdict | None = None
    reason: str = ""
    breach_signal: str | None = None

    @property
    def is_leaf(self) -> bool:
        return self.verdict is not None


def evaluate(root: Node, snapshot: HealthSnapshot) -> Decision:
    """Walk ``root`` against ``snapshot`` and return an explainable Decision.

    The returned :class:`Decision` carries the ordered ``path`` of visited node
    labels and the list of ``breached`` guardrail signals, so callers never have
    to guess why a verdict was reached.
    """

    path: list[str] = []
    breached: list[str] = []
    node = root

    while node is not None and not node.is_leaf:
        if node.predicate is None:  # malformed tree — a branch must have a predicate
            raise ValueError(f"branch node {node.label!r} has no predicate")
        outcome = node.predicate(snapshot)
        path.append(f"{node.label}={'yes' if outcome else 'no'}")
        if outcome and node.breach_signal:
            breached.append(node.breach_signal)
        node = node.if_true if outcome else node.if_false

    if node is None or node.verdict is None:
        raise ValueError("decision tree did not terminate in a leaf verdict")

    path.append(f"[{node.verdict}]")
    return Decision(
        verdict=node.verdict,
        reason=node.reason or f"reached {node.verdict} leaf",
        path=path,
        breached=breached,
    )


def build_promotion_tree(thresholds: GateThresholds) -> Node:
    """Construct the standard promotion gate as a decision tree.

    The tree encodes the safety-first ordering an SRE would use by hand:

    1. **Safety incidents** are terminal — any violation rolls back immediately.
    2. **Task success rate** is the robotics-specific health signal; below the
       floor is a rollback.
    3. **Service error rate** and **latency p99** are classic service guardrails.
    4. **Cycle-time regression** is softer: it *holds* (wait and re-measure)
       rather than rolling back outright, because it may be transient warm-up.
    5. If nothing breached, **promote**.
    """

    promote = Node(
        label="all_clear",
        verdict=Verdict.PROMOTE,
        reason="all guardrails within thresholds",
    )
    hold_on_cycle = Node(
        label="cycle_time_regressed",
        verdict=Verdict.HOLD,
        reason="cycle time elevated but non-critical; hold and re-measure",
    )

    # 4. Cycle time (soft): hold vs promote.
    cycle_node = Node(
        label="cycle_time_over_max",
        predicate=lambda s: s.task_cycle_time_s > thresholds.max_cycle_time_s,
        breach_signal="task_cycle_time_s",
        if_true=hold_on_cycle,
        if_false=promote,
    )

    # 3b. Latency guardrail.
    latency_node = Node(
        label="latency_p99_over_max",
        predicate=lambda s: s.service_latency_p99_ms > thresholds.max_latency_p99_ms,
        breach_signal="service_latency_p99_ms",
        if_true=Node(
            label="rollback_latency",
            verdict=Verdict.ROLLBACK,
            reason="p99 inference latency breached guardrail",
        ),
        if_false=cycle_node,
    )

    # 3a. Error-rate guardrail.
    error_node = Node(
        label="error_rate_over_max",
        predicate=lambda s: s.service_error_rate > thresholds.max_error_rate,
        breach_signal="service_error_rate",
        if_true=Node(
            label="rollback_error_rate",
            verdict=Verdict.ROLLBACK,
            reason="inference error rate breached guardrail",
        ),
        if_false=latency_node,
    )

    # 2. Robot task success floor.
    task_node = Node(
        label="task_success_below_min",
        predicate=lambda s: s.task_success_rate < thresholds.min_task_success_rate,
        breach_signal="task_success_rate",
        if_true=Node(
            label="rollback_task_success",
            verdict=Verdict.ROLLBACK,
            reason="robot task success rate fell below floor",
        ),
        if_false=error_node,
    )

    # 1. Safety incidents — terminal, evaluated first.
    root = Node(
        label="safety_incident",
        predicate=lambda s: s.safety_incidents > thresholds.max_safety_incidents,
        breach_signal="safety_incidents",
        if_true=Node(
            label="rollback_safety",
            verdict=Verdict.ROLLBACK,
            reason="safety-envelope violation detected",
        ),
        if_false=task_node,
    )
    return root


def render_tree(node: Node, indent: int = 0) -> str:
    """Return an ASCII rendering of the tree (used by the CLI to show the gate)."""

    pad = "  " * indent
    if node.is_leaf:
        return f"{pad}└─ [{node.verdict}] {node.reason}\n"
    out = f"{pad}? {node.label}\n"
    if node.if_true is not None:
        out += f"{pad}  yes →\n" + render_tree(node.if_true, indent + 2)
    if node.if_false is not None:
        out += f"{pad}  no  →\n" + render_tree(node.if_false, indent + 2)
    return out
