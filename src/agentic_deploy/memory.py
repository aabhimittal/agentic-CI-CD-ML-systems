"""Deployment memory — the agent learns from what happened before.

A scripted pipeline treats every deployment as the first one. An *agentic* system
should not: a service whose last release rolled back deserves a more cautious plan
than one with a clean streak. :class:`DeploymentMemory` is a small append-only
JSON store of past deployment outcomes; the agent consults it when planning:

* recent failures raise the risk score, and
* a service whose **last** deployment rolled back gets an extra low-traffic
  bake-in step (5%) prepended to its canary.

The store is a plain file so it survives across runs, diffs cleanly, and can be
inspected (or edited) by an operator. Concurrency is out of scope — one deployer
process owns the file, which matches how a deployment controller runs.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .models import DeploymentReport


class DeploymentMemory:
    """Append-only JSON history of deployment outcomes, keyed by service."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._entries: list[dict[str, Any]] = []
        if self.path.exists():
            try:
                self._entries = json.loads(self.path.read_text()) or []
            except (json.JSONDecodeError, OSError):
                # A corrupt store must not block deployments; start fresh but
                # keep the bad file aside for inspection.
                backup = self.path.with_suffix(self.path.suffix + ".corrupt")
                try:
                    self.path.rename(backup)
                except OSError:
                    pass
                self._entries = []

    # ---- recording ---------------------------------------------------------

    def record(self, report: "DeploymentReport") -> None:
        last = report.steps[-1] if report.steps else None
        self._entries.append(
            {
                "service": report.request.service,
                "model_version": report.request.model_version,
                "strategy": str(report.plan.strategy),
                "risk_score": report.plan.risk_score,
                "outcome": report.outcome,
                "breached": list(last.decision.breached) if last else [],
                "max_traffic_pct": max((s.step.traffic_pct for s in report.steps), default=0.0),
                "timestamp": time.time(),
            }
        )
        self._save()

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        tmp.write_text(json.dumps(self._entries, indent=2))
        tmp.replace(self.path)

    # ---- queries the agent uses -------------------------------------------

    def history(self, service: str) -> list[dict[str, Any]]:
        return [e for e in self._entries if e["service"] == service]

    def failure_rate(self, service: str, window: int = 10) -> float:
        """Fraction of the last ``window`` deployments of ``service`` that rolled back."""

        recent = self.history(service)[-window:]
        if not recent:
            return 0.0
        failures = sum(1 for e in recent if e["outcome"] != "promoted")
        return failures / len(recent)

    def last_outcome(self, service: str) -> str | None:
        recent = self.history(service)
        return recent[-1]["outcome"] if recent else None

    def recent_breaches(self, service: str, window: int = 10) -> list[str]:
        """Distinct guardrail signals that caused recent rollbacks (most recent last)."""

        seen: list[str] = []
        for e in self.history(service)[-window:]:
            for sig in e.get("breached", []):
                if sig not in seen:
                    seen.append(sig)
        return seen

    def __len__(self) -> int:
        return len(self._entries)
