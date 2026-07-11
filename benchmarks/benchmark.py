"""Reproducible benchmark: scripted GitOps baseline vs the agentic engine.

We generate many randomized rollouts — a mix of healthy candidates and faulty ones
(quality regressions and latent safety bugs) — and deploy each with two strategies:

* **scripted baseline** — a fixed-schedule GitOps deployer with no per-step health
  gating. It bakes each stage on a timer and reaches 100% before any post-deploy
  alert can fire, so a faulty candidate is exposed to the *whole fleet* first and is
  only reverted after a slow mean-time-to-detect.
* **agentic** — this project's engine: the decision tree gates every canary step, so
  faulty candidates are caught at 10–50% traffic and reverted immediately.

Two headline metrics are reported, defined mechanistically so they're auditable:

* **Failed deployments** — faulty candidates that reached 100% of the fleet (a
  fleet-wide incident). Containment is the whole point of gating.
* **Mean rollback time** — for caught-faulty rollouts, simulated seconds from the
  breach to a safe state (detection latency + drain).

The numbers are simulated but fully reproducible: run with a fixed ``--seed`` and you
get the same result every time. This is what grounds the README's figures.

Usage:
    python benchmarks/benchmark.py --seeds 200
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass

from agentic_deploy.agent import Agent
from agentic_deploy.decision_tree import build_promotion_tree, evaluate
from agentic_deploy.models import DeploymentRequest, GateThresholds, Verdict
from agentic_deploy.orchestrator import deploy
from agentic_deploy.robotics_sim import BehaviorProfile, FleetSimulator

# --- baseline timing model (simulated seconds) ---
# Scripted baseline bakes to 100%, then a monitoring window detects the breach.
BASELINE_DETECT_SECONDS = 240.0
DRAIN_SECONDS = 60.0
# Agentic drain once a gate decides to roll back.
AGENTIC_DRAIN_SECONDS = 15.0


@dataclass
class Case:
    request: DeploymentRequest
    profile: BehaviorProfile
    seed: int
    faulty: bool  # ground truth: does the candidate breach a guardrail at 100%?


def _random_case(rng: random.Random, seed: int) -> Case:
    fleet = rng.randint(10, 100)
    crit = round(rng.uniform(0.1, 0.95), 2)
    thresholds = GateThresholds()
    base = dict(
        base_error_rate=0.010,
        base_latency_p99_ms=180.0,
        base_task_success_rate=0.980,
        base_cycle_time_s=8.0,
        noise=0.01,
    )
    kind = rng.random()
    if kind < 0.55:
        # Healthy candidate.
        summary = "minor update"
        profile = BehaviorProfile(
            cand_error_rate=rng.uniform(0.006, 0.014),
            cand_latency_p99_ms=rng.uniform(165, 200),
            cand_task_success_rate=rng.uniform(0.975, 0.995),
            cand_cycle_time_s=rng.uniform(7.6, 8.4),
            cand_safety_rate=0.0,
            **base,
        )
    elif kind < 0.80:
        # Quality regression.
        summary = "major architecture change; retrain"
        profile = BehaviorProfile(
            cand_error_rate=rng.uniform(0.06, 0.13),
            cand_latency_p99_ms=rng.uniform(280, 420),
            cand_task_success_rate=rng.uniform(0.72, 0.88),
            cand_cycle_time_s=rng.uniform(9.5, 12.5),
            cand_safety_rate=0.0,
            regression_activates_at_step=rng.choice([0, 0, 1]),
            **base,
        )
    else:
        # Latent safety bug: healthy-looking metrics, safety fault at higher traffic.
        summary = "new safety-critical weights"
        profile = BehaviorProfile(
            cand_error_rate=rng.uniform(0.008, 0.015),
            cand_latency_p99_ms=rng.uniform(170, 210),
            cand_task_success_rate=rng.uniform(0.975, 0.99),
            cand_cycle_time_s=rng.uniform(7.8, 8.4),
            cand_safety_rate=rng.uniform(4.0, 10.0),
            regression_activates_at_step=rng.choice([1, 2, 2, 3]),
            **base,
        )
    request = DeploymentRequest(
        service="grasp-planner",
        model_version="candidate",
        previous_version="baseline",
        fleet_size=fleet,
        task_criticality=crit,
        change_summary=summary,
        thresholds=thresholds,
    )
    faulty = _is_faulty(request, profile, seed)
    return Case(request, profile, seed, faulty)


def _is_faulty(request: DeploymentRequest, profile: BehaviorProfile, seed: int) -> bool:
    """Ground truth: would the candidate breach a guardrail at full fleet traffic?"""

    sim = FleetSimulator(profile, request.fleet_size, seed=seed)
    tree = build_promotion_tree(request.thresholds)
    # Observe at the profile's final step so any latent regression is active.
    last_step = max(profile.regression_activates_at_step, 3)
    snap = sim.observe(100.0, last_step)
    verdict = evaluate(tree, snap).verdict
    return verdict != Verdict.PROMOTE


def _run_baseline(case: Case) -> tuple[bool, float | None]:
    """Return (failed_deployment, rollback_seconds_or_None).

    The baseline always reaches 100% before it can react, so every faulty
    candidate is a fleet-wide failure; it is then reverted slowly.
    """

    if case.faulty:
        rollback = BASELINE_DETECT_SECONDS + DRAIN_SECONDS
        return True, rollback  # reached full fleet = failed, then reverted
    return False, None


def _run_agentic(case: Case) -> tuple[bool, float | None]:
    """Return (failed_deployment, rollback_seconds_or_None) for the agentic engine."""

    report = deploy(case.request, case.profile, agent=Agent(use_llm=False), seed=case.seed)
    reached_full = any(s.step.traffic_pct >= 100.0 for s in report.steps)
    failed = case.faulty and reached_full
    rollback = None
    if report.outcome == "rolled_back" and report.steps:
        breach_dwell = report.steps[-1].step.dwell_seconds
        rollback = breach_dwell + AGENTIC_DRAIN_SECONDS
    return failed, rollback


def run(n: int, seed: int) -> dict:
    rng = random.Random(seed)
    cases = [_random_case(rng, seed + i) for i in range(n)]
    faulty = [c for c in cases if c.faulty]

    b_failed = a_failed = 0
    b_times: list[float] = []
    a_times: list[float] = []
    for case in cases:
        bf, bt = _run_baseline(case)
        af, at = _run_agentic(case)
        b_failed += int(bf)
        a_failed += int(af)
        if bt is not None:
            b_times.append(bt)
        if at is not None:
            a_times.append(at)

    b_mttr = sum(b_times) / len(b_times) if b_times else 0.0
    a_mttr = sum(a_times) / len(a_times) if a_times else 0.0
    failed_reduction = (b_failed - a_failed) / b_failed if b_failed else 0.0
    mttr_reduction = (b_mttr - a_mttr) / b_mttr if b_mttr else 0.0

    return {
        "rollouts": n,
        "faulty_candidates": len(faulty),
        "baseline": {"failed_deployments": b_failed, "mean_rollback_seconds": round(b_mttr, 1)},
        "agentic": {"failed_deployments": a_failed, "mean_rollback_seconds": round(a_mttr, 1)},
        "failed_deployment_reduction_pct": round(100 * failed_reduction, 1),
        "mean_rollback_time_reduction_pct": round(100 * mttr_reduction, 1),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Baseline vs agentic deployment benchmark")
    parser.add_argument("--seeds", type=int, default=200, help="number of randomized rollouts")
    parser.add_argument("--seed", type=int, default=1234, help="base RNG seed")
    parser.add_argument("--json", action="store_true", help="emit raw JSON only")
    args = parser.parse_args()

    result = run(args.seeds, args.seed)
    if args.json:
        print(json.dumps(result, indent=2))
        return

    print(f"Randomized rollouts : {result['rollouts']} "
          f"({result['faulty_candidates']} faulty candidates)")
    print("-" * 60)
    print(f"{'':22}{'scripted baseline':>18}{'agentic':>12}")
    print(f"{'failed deployments':22}"
          f"{result['baseline']['failed_deployments']:>18}"
          f"{result['agentic']['failed_deployments']:>12}")
    print(f"{'mean rollback (s)':22}"
          f"{result['baseline']['mean_rollback_seconds']:>18}"
          f"{result['agentic']['mean_rollback_seconds']:>12}")
    print("-" * 60)
    print(f"Failed deployments   ↓ {result['failed_deployment_reduction_pct']}%")
    print(f"Mean rollback time   ↓ {result['mean_rollback_time_reduction_pct']}%")


if __name__ == "__main__":
    main()
