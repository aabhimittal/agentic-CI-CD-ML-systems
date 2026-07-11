"""The deployment agent — the "decision maker" of the pipeline.

The agent has two jobs:

1. :meth:`Agent.plan` — inspect a :class:`~agentic_deploy.models.DeploymentRequest`
   and choose a rollout **strategy** (canary vs blue/green) with graduated steps
   and a risk score, explaining *why*.
2. :meth:`Agent.explain_rollback` — when a gate rolls back, write an operator-facing
   narrative of what breached and what the system did about it.

Both jobs are backed by Claude (``claude-opus-4-8``) when the ``anthropic`` SDK and
credentials are available, and by a deterministic heuristic otherwise. The two paths
return the *same shapes*, so the rest of the system — and the test suite — never has
to care which one ran. Every LLM call is wrapped so any failure silently degrades to
the heuristic rather than breaking a deployment.
"""

from __future__ import annotations

import json
import os
from typing import Any

from .models import (
    DeploymentRequest,
    HealthSnapshot,
    RolloutStep,
    RolloutStrategy,
    RolloutStrategyPlan,
)

MODEL = "claude-opus-4-8"

# JSON schema the LLM must fill for a strategy plan (structured outputs).
_PLAN_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "strategy": {"type": "string", "enum": ["canary", "blue_green"]},
        "risk_score": {"type": "number"},
        "rationale": {"type": "string"},
        "traffic_steps": {
            "type": "array",
            "items": {"type": "number"},
            "description": "Ordered candidate traffic percentages, ending at 100.",
        },
    },
    "required": ["strategy", "risk_score", "rationale", "traffic_steps"],
    "additionalProperties": False,
}


def _llm_available() -> bool:
    """True only if the SDK imports *and* some credential is plausibly present."""

    try:
        import anthropic  # noqa: F401
    except Exception:
        return False
    # The SDK also resolves `ant auth login` profiles, but an API key is the
    # common CI signal; treat its presence (or an explicit auth token) as opt-in.
    return bool(os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN"))


class Agent:
    """Plans rollouts and reasons about rollbacks."""

    def __init__(self, use_llm: bool | None = None, model: str = MODEL) -> None:
        # ``None`` means auto-detect; callers (and tests) can force either path.
        self.use_llm = _llm_available() if use_llm is None else use_llm
        self.model = model
        self._client = None
        if self.use_llm:
            try:
                import anthropic

                self._client = anthropic.Anthropic()
            except Exception:
                # Import/construction failed after detection — fall back cleanly.
                self.use_llm = False

    # ---- strategy planning -------------------------------------------------

    def plan(self, request: DeploymentRequest) -> RolloutStrategyPlan:
        if self.use_llm and self._client is not None:
            plan = self._plan_with_llm(request)
            if plan is not None:
                return plan
        return self._plan_heuristic(request)

    def _plan_heuristic(self, request: DeploymentRequest) -> RolloutStrategyPlan:
        """Rule-based planner. Deterministic given a request — safe for tests/CI."""

        risk = _risk_score(request)
        # High risk or safety-critical work → graduated canary to cap blast radius.
        # Low risk → atomic blue/green cutover for speed.
        if risk >= 0.5 or request.task_criticality >= 0.7:
            strategy = RolloutStrategy.CANARY
            traffic = [10.0, 25.0, 50.0, 100.0]
            rationale = (
                f"Risk score {risk:.2f} (fleet={request.fleet_size}, "
                f"criticality={request.task_criticality:.2f}). Graduated canary "
                f"limits blast radius and gates each step on robot task health."
            )
        else:
            strategy = RolloutStrategy.BLUE_GREEN
            traffic = [100.0]
            rationale = (
                f"Risk score {risk:.2f} is low and the task is not safety-critical. "
                f"Blue/green validates a parallel green stack, then cuts over "
                f"atomically for a fast, reversible switch."
            )
        return RolloutStrategyPlan(
            strategy=strategy,
            steps=_steps_from_traffic(traffic, strategy),
            risk_score=risk,
            rationale=rationale,
            source="heuristic",
        )

    def _plan_with_llm(self, request: DeploymentRequest) -> RolloutStrategyPlan | None:
        """Ask Claude for a strategy. Returns None on any error (caller falls back)."""

        system = (
            "You are a deployment strategist for an ML + robotics fleet CI/CD system. "
            "Choose between a graduated 'canary' rollout (safer, limits blast radius) "
            "and an atomic 'blue_green' cutover (faster, fully reversible). Favor canary "
            "for high blast radius or safety-critical robot tasks. Return traffic_steps "
            "as ordered candidate-traffic percentages ending at 100."
        )
        user = json.dumps(request.to_dict())
        try:
            resp = self._client.messages.create(
                model=self.model,
                max_tokens=2048,
                thinking={"type": "adaptive"},
                system=system,
                output_config={"format": {"type": "json_schema", "schema": _PLAN_SCHEMA}},
                messages=[{"role": "user", "content": user}],
            )
            text = next(b.text for b in resp.content if b.type == "text")
            data = json.loads(text)
            strategy = RolloutStrategy(data["strategy"])
            traffic = [float(x) for x in data["traffic_steps"]] or [100.0]
            if traffic[-1] != 100.0:
                traffic.append(100.0)
            return RolloutStrategyPlan(
                strategy=strategy,
                steps=_steps_from_traffic(traffic, strategy),
                risk_score=float(data["risk_score"]),
                rationale=str(data["rationale"]),
                source="claude",
            )
        except Exception:
            return None

    # ---- rollback reasoning ------------------------------------------------

    def explain_rollback(
        self,
        request: DeploymentRequest,
        snapshot: HealthSnapshot,
        breached: list[str],
    ) -> str:
        if self.use_llm and self._client is not None:
            text = self._explain_with_llm(request, snapshot, breached)
            if text:
                return text
        return _explain_heuristic(request, snapshot, breached)

    def _explain_with_llm(
        self,
        request: DeploymentRequest,
        snapshot: HealthSnapshot,
        breached: list[str],
    ) -> str | None:
        system = (
            "You are the on-call reasoning agent for a robotics deployment system. "
            "In 2-4 sentences, explain plainly why the rollout was rolled back, which "
            "health signal breached its guardrail, and confirm the fleet was reverted "
            "to the previous model version. Be specific and non-alarmist."
        )
        payload = {
            "service": request.service,
            "candidate": request.model_version,
            "reverted_to": request.previous_version,
            "breached_signals": breached,
            "snapshot": snapshot.to_dict(),
            "thresholds": request.thresholds.to_dict(),
        }
        try:
            resp = self._client.messages.create(
                model=self.model,
                max_tokens=1024,
                thinking={"type": "adaptive"},
                system=system,
                messages=[{"role": "user", "content": json.dumps(payload)}],
            )
            return "".join(b.text for b in resp.content if b.type == "text").strip() or None
        except Exception:
            return None


# ---- shared heuristics (module-level so the baseline benchmark can reuse them) ----


def _risk_score(request: DeploymentRequest) -> float:
    """Blend blast radius, task criticality, and change hints into a 0..1 score."""

    # Blast radius: saturating on fleet size (a 100-robot fleet is "full" risk).
    blast = min(request.fleet_size / 100.0, 1.0)
    crit = max(0.0, min(request.task_criticality, 1.0))
    # Change-summary keywords bump risk for known-scary changes.
    text = request.change_summary.lower()
    change_penalty = 0.0
    for kw, bump in (("architecture", 0.2), ("retrain", 0.15), ("major", 0.15), ("weights", 0.1)):
        if kw in text:
            change_penalty += bump
    score = 0.45 * blast + 0.40 * crit + change_penalty
    return round(min(score, 1.0), 3)


def _steps_from_traffic(traffic: list[float], strategy: RolloutStrategy) -> list[RolloutStep]:
    # Canary observes each step briefly; blue/green validates the green stack longer.
    dwell = 30.0 if strategy == RolloutStrategy.CANARY else 60.0
    return [RolloutStep(index=i, traffic_pct=p, dwell_seconds=dwell) for i, p in enumerate(traffic)]


def _explain_heuristic(
    request: DeploymentRequest,
    snapshot: HealthSnapshot,
    breached: list[str],
) -> str:
    thr = request.thresholds
    observed = {
        "safety_incidents": (snapshot.safety_incidents, f"> {thr.max_safety_incidents}"),
        "task_success_rate": (snapshot.task_success_rate, f"< {thr.min_task_success_rate}"),
        "service_error_rate": (snapshot.service_error_rate, f"> {thr.max_error_rate}"),
        "service_latency_p99_ms": (snapshot.service_latency_p99_ms, f"> {thr.max_latency_p99_ms}"),
        "task_cycle_time_s": (snapshot.task_cycle_time_s, f"> {thr.max_cycle_time_s}"),
    }
    if breached:
        parts = []
        for sig in breached:
            if sig in observed:
                val, bound = observed[sig]
                parts.append(f"{sig}={val} (guardrail {bound})")
        detail = "; ".join(parts) if parts else ", ".join(breached)
    else:
        detail = "an aggregate health check failed"
    return (
        f"Rolled back {request.service} candidate {request.model_version} at "
        f"{snapshot.traffic_pct:.0f}% traffic: {detail}. The fleet of "
        f"{request.fleet_size} robots was reverted to {request.previous_version}; "
        f"no further traffic was shifted to the candidate."
    )
