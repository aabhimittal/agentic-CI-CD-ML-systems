"""Robotic task-impact simulator — the "world" the deployment acts on.

A real system would read these numbers from a fleet of robots and their serving
stack. Here a :class:`FleetSimulator` synthesizes them from a
:class:`BehaviorProfile` that describes how the *candidate* model differs from the
currently-serving *baseline*. Because the fleet is partly on the candidate
(``traffic_pct``) and partly on the baseline, aggregate service metrics are
traffic-weighted blends — which is exactly why a graduated canary can catch a bad
model before it reaches the whole fleet.

Safety incidents are generated **only** from candidate robots, so they can surface
at low traffic even when blended averages still look fine. That is the signal the
decision tree checks first.

The simulator is fully seeded, so a scenario + seed always produces the same run —
which is what makes the benchmark reproducible.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Any

from .models import HealthSnapshot


@dataclass
class BehaviorProfile:
    """How the candidate model behaves relative to the healthy baseline."""

    # Baseline (previous model) — healthy operating point.
    base_error_rate: float = 0.01
    base_latency_p99_ms: float = 180.0
    base_task_success_rate: float = 0.98
    base_cycle_time_s: float = 8.0

    # Candidate operating point (absolute values it converges to at 100% traffic).
    cand_error_rate: float = 0.01
    cand_latency_p99_ms: float = 190.0
    cand_task_success_rate: float = 0.98
    cand_cycle_time_s: float = 8.2

    # Expected safety-envelope violations per observation window at 100% candidate.
    cand_safety_rate: float = 0.0

    # Step/window indices below are *observation windows* — monotonic in time,
    # advancing on every observation (including re-observations after a hold and
    # post-promotion soak rounds).

    # A latent regression stays dormant until this window (0 = active from start).
    regression_activates_at_step: int = 0

    # A transient fault: the regression clears again from this window onward
    # (-1 = permanent). Models warm-up effects like cache-cold cycle-time spikes.
    regression_deactivates_at_step: int = -1

    # Fleet telemetry is dark during these windows — the observation has
    # stale/missing data. Industrially this must read as "unknown", never "healthy".
    telemetry_dropout_steps: tuple[int, ...] = ()

    # Relative gaussian noise applied to each reading.
    noise: float = 0.02

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "BehaviorProfile":
        known = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        kwargs = {k: v for k, v in d.items() if k in known}
        if "telemetry_dropout_steps" in kwargs:  # YAML gives a list; the field is a tuple
            kwargs["telemetry_dropout_steps"] = tuple(kwargs["telemetry_dropout_steps"])
        return cls(**kwargs)


class FleetSimulator:
    """Generates :class:`HealthSnapshot` s for a fleet under a rollout."""

    def __init__(self, profile: BehaviorProfile, fleet_size: int, seed: int = 0) -> None:
        self.profile = profile
        self.fleet_size = fleet_size
        self.rng = random.Random(seed)

    def observe(self, traffic_pct: float, step_index: int) -> HealthSnapshot:
        """Produce the fleet's health at a given candidate traffic share."""

        p = self.profile
        frac = max(0.0, min(traffic_pct / 100.0, 1.0))

        # Telemetry dropout: the window has no trustworthy data. Freeze readings at
        # the healthy baseline but mark them untrusted — the gate must not treat
        # stale-but-green numbers as evidence of health.
        if step_index in p.telemetry_dropout_steps:
            return HealthSnapshot(
                traffic_pct=traffic_pct,
                service_error_rate=p.base_error_rate,
                service_latency_p99_ms=p.base_latency_p99_ms,
                task_success_rate=p.base_task_success_rate,
                task_cycle_time_s=p.base_cycle_time_s,
                safety_incidents=0,
                telemetry_ok=False,
                step_index=step_index,
            )

        # A latent regression behaves like the baseline until it activates, and a
        # transient one (e.g. warm-up) clears again at deactivation.
        active = step_index >= p.regression_activates_at_step and (
            p.regression_deactivates_at_step < 0
            or step_index < p.regression_deactivates_at_step
        )
        cand_error = p.cand_error_rate if active else p.base_error_rate
        cand_latency = p.cand_latency_p99_ms if active else p.base_latency_p99_ms
        cand_success = p.cand_task_success_rate if active else p.base_task_success_rate
        cand_cycle = p.cand_cycle_time_s if active else p.base_cycle_time_s
        safety_rate = p.cand_safety_rate if active else 0.0

        # Traffic-weighted blend of baseline and candidate populations.
        error = self._blend(p.base_error_rate, cand_error, frac)
        latency = self._blend(p.base_latency_p99_ms, cand_latency, frac)
        success = self._blend(p.base_task_success_rate, cand_success, frac)
        cycle = self._blend(p.base_cycle_time_s, cand_cycle, frac)

        # Safety incidents come only from candidate robots; sample a small count.
        incidents = self._poisson(safety_rate * frac)

        return HealthSnapshot(
            traffic_pct=traffic_pct,
            service_error_rate=self._clamp_rate(self._noisy(error)),
            service_latency_p99_ms=max(0.0, self._noisy(latency)),
            task_success_rate=self._clamp_rate(self._noisy(success)),
            task_cycle_time_s=max(0.0, self._noisy(cycle)),
            safety_incidents=incidents,
            step_index=step_index,
        )

    # ---- helpers -----------------------------------------------------------

    @staticmethod
    def _blend(base: float, cand: float, frac: float) -> float:
        return base * (1.0 - frac) + cand * frac

    def _noisy(self, value: float) -> float:
        return value * (1.0 + self.rng.gauss(0.0, self.profile.noise))

    @staticmethod
    def _clamp_rate(x: float) -> float:
        return max(0.0, min(x, 1.0))

    def _poisson(self, lam: float) -> int:
        """Knuth's algorithm — a seeded Poisson sample for incident counts."""

        if lam <= 0.0:
            return 0
        limit = math.exp(-lam)
        k, prod = 0, 1.0
        while True:
            k += 1
            prod *= self.rng.random()
            if prod <= limit:
                return k - 1
