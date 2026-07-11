"""A mock Kubernetes controller that drives the rollout mechanics.

There is no live cluster here, but the mechanics are faithful: a blue and a green
Deployment, a Service selecting one color, and a candidate traffic weight that a
canary shifts in steps. Every mutation is appended to an event log so a report can
show exactly what the controller did — apply, weight-shift, cut-over, rollback.

Real manifests live in ``deploy/k8s/``; :meth:`MockCluster.apply` will read them
if a path is given, but the controller works with in-memory defaults too.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class DeploymentState:
    color: str          # "blue" or "green"
    version: str
    replicas: int
    ready: bool = False


@dataclass
class MockCluster:
    """In-memory stand-in for the cluster's rollout-relevant state."""

    service: str
    baseline_version: str
    candidate_version: str
    replicas: int = 4
    active_color: str = "blue"
    candidate_weight: float = 0.0  # percent of traffic to the candidate (green)
    events: list[str] = field(default_factory=list)
    deployments: dict[str, DeploymentState] = field(default_factory=dict)

    def __post_init__(self) -> None:
        # Blue is the currently-serving baseline; green is where the candidate lands.
        self.deployments = {
            "blue": DeploymentState("blue", self.baseline_version, self.replicas, ready=True),
            "green": DeploymentState("green", self.candidate_version, 0, ready=False),
        }
        self._log(f"cluster initialized: blue={self.baseline_version} serving 100%")

    # ---- manifest handling -------------------------------------------------

    def apply(self, manifest_path: str | Path) -> None:
        """Record that a manifest was applied (parsing it if PyYAML is available)."""

        path = Path(manifest_path)
        kind = "manifest"
        try:
            import yaml

            with path.open() as fh:
                docs = [d for d in yaml.safe_load_all(fh) if d]
            kinds = ",".join(sorted({d.get("kind", "?") for d in docs}))
            kind = kinds or "manifest"
        except Exception:
            pass
        self._log(f"kubectl apply -f {path.name} ({kind})")

    # ---- rollout mechanics -------------------------------------------------

    def stand_up_green(self) -> None:
        """Scale up the green Deployment with the candidate model."""

        green = self.deployments["green"]
        green.replicas = self.replicas
        green.ready = True
        self._log(f"scaled green to {self.replicas} replicas ({self.candidate_version})")

    def set_candidate_weight(self, pct: float) -> None:
        """Shift ``pct`` percent of traffic to the candidate (canary step)."""

        if not self.deployments["green"].ready and pct > 0:
            self.stand_up_green()
        self.candidate_weight = pct
        self._log(f"canary weight → {pct:.0f}% candidate / {100 - pct:.0f}% baseline")

    def cut_over(self) -> None:
        """Blue/green atomic switch: Service now selects green."""

        if not self.deployments["green"].ready:
            self.stand_up_green()
        self.active_color = "green"
        self.candidate_weight = 100.0
        self._log(f"service selector → green; {self.candidate_version} now serving 100%")

    def promote(self) -> None:
        """Finalize: candidate is the new baseline; retire the old stack."""

        self.candidate_weight = 100.0
        self.deployments["blue"].ready = False
        self.deployments["blue"].replicas = 0
        self._log(f"promoted {self.candidate_version}; retired previous stack")

    def rollback(self) -> None:
        """Revert all traffic to the baseline and tear down the candidate."""

        self.candidate_weight = 0.0
        self.active_color = "blue"
        green = self.deployments["green"]
        green.replicas = 0
        green.ready = False
        self._log(f"ROLLBACK: traffic → 100% baseline ({self.baseline_version}); green torn down")

    # ---- reporting ---------------------------------------------------------

    def state(self) -> dict[str, Any]:
        return {
            "service": self.service,
            "active_color": self.active_color,
            "candidate_weight": self.candidate_weight,
            "deployments": {
                c: {"version": d.version, "replicas": d.replicas, "ready": d.ready}
                for c, d in self.deployments.items()
            },
            "events": list(self.events),
        }

    def _log(self, msg: str) -> None:
        self.events.append(msg)
