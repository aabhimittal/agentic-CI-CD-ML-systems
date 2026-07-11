"""Prometheus-style health-signal provider.

Prometheus doesn't *generate* metrics — it stores and exposes them and answers
queries. This module mirrors that role: the :class:`FleetSimulator` produces
:class:`HealthSnapshot` s, the provider records the latest one, exposes every
signal in valid Prometheus **text exposition format**, and answers the simple
threshold queries the decision tree needs.

Exposition text is generated directly (no dependency required). If the optional
``prometheus_client`` package is installed, :func:`exposition_is_parseable` can be
used to confirm the output round-trips through a real parser.
"""

from __future__ import annotations

from typing import Iterable

from .models import HealthSnapshot

# metric name -> (help text, prometheus type, snapshot attribute)
_METRICS: list[tuple[str, str, str, str]] = [
    ("inference_error_rate", "Fraction of inference requests that errored.", "gauge", "service_error_rate"),
    ("inference_latency_p99_ms", "99th percentile inference latency in ms.", "gauge", "service_latency_p99_ms"),
    ("robot_task_success_rate", "Fraction of robot tasks completed successfully.", "gauge", "task_success_rate"),
    ("robot_task_cycle_time_seconds", "Mean seconds per robot task.", "gauge", "task_cycle_time_s"),
    ("robot_safety_incidents_total", "Safety-envelope violations in the window.", "counter", "safety_incidents"),
    ("candidate_traffic_percent", "Share of fleet traffic on the candidate model.", "gauge", "traffic_pct"),
]


class PrometheusProvider:
    """Records fleet health and exposes it the way Prometheus would."""

    def __init__(self, service: str, version: str) -> None:
        self.service = service
        self.version = version
        self._latest: HealthSnapshot | None = None

    def record(self, snapshot: HealthSnapshot) -> None:
        self._latest = snapshot

    @property
    def latest(self) -> HealthSnapshot | None:
        return self._latest

    def query(self, metric: str) -> float:
        """Return the current value of an exposed metric by its Prometheus name."""

        if self._latest is None:
            raise RuntimeError("no snapshot recorded yet")
        for name, _help, _type, attr in _METRICS:
            if name == metric:
                return float(getattr(self._latest, attr))
        raise KeyError(f"unknown metric {metric!r}")

    def exposition(self) -> str:
        """Render the latest snapshot as Prometheus text exposition format."""

        if self._latest is None:
            return ""
        labels = f'service="{self.service}",version="{self.version}"'
        lines: list[str] = []
        for name, help_text, mtype, attr in _METRICS:
            value = getattr(self._latest, attr)
            lines.append(f"# HELP {name} {help_text}")
            lines.append(f"# TYPE {name} {mtype}")
            lines.append(f"{name}{{{labels}}} {_format_value(value)}")
        return "\n".join(lines) + "\n"


def _format_value(value: float | int) -> str:
    if isinstance(value, int):
        return str(value)
    # Prometheus wants plain decimal, no scientific notation for these ranges.
    return f"{value:.6g}"


def exposition_is_parseable(text: str) -> bool:
    """True if ``prometheus_client`` can parse ``text`` (optional dependency).

    Returns True when the package is absent so callers can treat it as a
    best-effort check rather than a hard requirement.
    """

    try:
        from prometheus_client.parser import text_string_to_metric_families
    except Exception:
        return True
    try:
        list(text_string_to_metric_families(text))
        return True
    except Exception:
        return False


def signals() -> Iterable[str]:
    """The Prometheus metric names this provider exposes."""

    return [name for name, _, _, _ in _METRICS]
