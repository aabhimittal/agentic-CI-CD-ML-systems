"""Rollout strategy adapters.

Translate an abstract :class:`~agentic_deploy.models.RolloutStep` into concrete
:class:`~agentic_deploy.kubernetes.MockCluster` operations. The state machine calls
:func:`apply_step` at each step, then :func:`promote` or :func:`rollback` at the end.

The two strategies differ only in *how a step maps to cluster mechanics*:

* **canary** — each step shifts the candidate traffic weight to ``step.traffic_pct``.
* **blue_green** — the single 100% step is an atomic Service cut-over to green.
"""

from __future__ import annotations

from .kubernetes import MockCluster
from .models import RolloutStep, RolloutStrategy


def apply_step(cluster: MockCluster, strategy: RolloutStrategy, step: RolloutStep) -> None:
    """Perform the cluster action for one rollout step."""

    if strategy == RolloutStrategy.CANARY:
        cluster.set_candidate_weight(step.traffic_pct)
    elif strategy == RolloutStrategy.BLUE_GREEN:
        # Blue/green validates green, then cuts over on the (only) 100% step.
        if step.traffic_pct >= 100.0:
            cluster.cut_over()
        else:
            cluster.stand_up_green()
    else:  # pragma: no cover - enum is exhaustive
        raise ValueError(f"unknown strategy {strategy!r}")


def promote(cluster: MockCluster) -> None:
    """Finalize a successful rollout."""

    cluster.promote()


def rollback(cluster: MockCluster) -> None:
    """Abort a rollout and return to the baseline."""

    cluster.rollback()
