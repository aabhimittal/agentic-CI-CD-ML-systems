"""Command-line entry point: ``agentic-deploy``.

Subcommands:

* ``run <scenario.yaml>``  — execute a full deployment and print the explainable
  report (decision-tree path per gate, agent rationale, rollback reasoning).
* ``tree <scenario.yaml>`` — render the promotion decision tree for a scenario.

Uses ``rich`` for pretty output when installed, and falls back to plain text.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import yaml

from .agent import Agent
from .decision_tree import build_promotion_tree, render_tree
from .models import DeploymentReport, DeploymentRequest, GateThresholds
from .orchestrator import Orchestrator
from .robotics_sim import BehaviorProfile


def load_scenario(path: str | Path) -> tuple[DeploymentRequest, BehaviorProfile, int]:
    """Parse a scenario YAML into a request, a behavior profile, and a seed."""

    with Path(path).open() as fh:
        data: dict[str, Any] = yaml.safe_load(fh) or {}

    thr = GateThresholds(**(data.get("thresholds") or {}))
    request = DeploymentRequest(
        service=data["service"],
        model_version=data["model_version"],
        previous_version=data["previous_version"],
        fleet_size=int(data["fleet_size"]),
        task_criticality=float(data.get("task_criticality", 0.5)),
        change_summary=data.get("change_summary", ""),
        thresholds=thr,
    )
    profile = BehaviorProfile.from_dict(data.get("behavior") or {})
    seed = int(data.get("seed", 0))
    return request, profile, seed


def _make_agent(args: argparse.Namespace, memory=None) -> Agent:
    if getattr(args, "force_heuristic", False):
        return Agent(use_llm=False, memory=memory)
    if getattr(args, "force_llm", False):
        return Agent(use_llm=True, memory=memory)
    return Agent(memory=memory)


# ---- rendering -------------------------------------------------------------


def _render_plain(report: DeploymentReport) -> str:
    r, p = report.request, report.plan
    out: list[str] = []
    out.append("=" * 72)
    out.append(f"DEPLOYMENT  {r.service}  {r.previous_version} → {r.model_version}")
    out.append("=" * 72)
    out.append(f"Agent plan   : {p.strategy}  (risk={p.risk_score:.2f}, source={p.source})")
    out.append(f"Rationale    : {p.rationale}")
    out.append(f"State machine: {' → '.join(report.state_machine_path)}")
    out.append("-" * 72)
    for rec in report.steps:
        s, d = rec.snapshot, rec.decision
        label = "soak  " if rec.step.phase == "soak" else f"step {rec.step.index}"
        telemetry = "" if s.telemetry_ok else "  TELEMETRY-LOST"
        out.append(
            f"{label}  traffic={s.traffic_pct:>5.0f}%  "
            f"err={s.service_error_rate:.3f}  p99={s.service_latency_p99_ms:>5.0f}ms  "
            f"task_ok={s.task_success_rate:.3f}  cycle={s.task_cycle_time_s:.1f}s  "
            f"safety={s.safety_incidents}{telemetry}  →  {str(d.verdict).upper()}"
        )
        out.append(f"          tree: {' / '.join(d.path)}")
        if d.breached:
            out.append(f"          breached: {', '.join(d.breached)}")
    out.append("-" * 72)
    out.append(f"OUTCOME      : {report.outcome.upper()}")
    out.append(f"Rollout time : {report.duration_seconds:.0f}s (simulated)")
    if report.outcome == "rolled_back":
        out.append(f"Rollback time: {report.rollback_seconds:.0f}s (breach → safe state)")
        out.append(f"Agent says   : {report.rollback_reasoning}")
    out.append("=" * 72)
    return "\n".join(out)


def _render_rich(report: DeploymentReport) -> bool:
    try:
        from rich.console import Console
        from rich.panel import Panel
        from rich.table import Table
    except Exception:
        return False

    console = Console()
    r, p = report.request, report.plan
    header = (
        f"[bold]{r.service}[/bold]  {r.previous_version} → [bold]{r.model_version}[/bold]\n"
        f"strategy: [cyan]{p.strategy}[/cyan]   risk: {p.risk_score:.2f}   "
        f"agent: [magenta]{p.source}[/magenta]\n{p.rationale}"
    )
    console.print(Panel(header, title="Deployment plan", expand=False))

    table = Table(show_header=True, header_style="bold")
    for col in ("step", "traffic", "err", "p99 ms", "task ok", "cycle", "safety", "verdict"):
        table.add_column(col)
    for rec in report.steps:
        s, d = rec.snapshot, rec.decision
        color = {"promote": "green", "hold": "yellow", "rollback": "red"}.get(str(d.verdict), "white")
        table.add_row(
            "soak" if rec.step.phase == "soak" else str(rec.step.index),
            f"{s.traffic_pct:.0f}%",
            f"{s.service_error_rate:.3f}",
            f"{s.service_latency_p99_ms:.0f}",
            f"{s.task_success_rate:.3f}",
            f"{s.task_cycle_time_s:.1f}",
            str(s.safety_incidents),
            f"[{color}]{str(d.verdict).upper()}[/{color}]",
        )
    console.print(table)

    outcome_color = "green" if report.outcome == "promoted" else "red"
    footer = f"[{outcome_color}]{report.outcome.upper()}[/{outcome_color}]  "
    footer += f"rollout {report.duration_seconds:.0f}s"
    if report.outcome == "rolled_back":
        footer += f"  •  rollback {report.rollback_seconds:.0f}s\n\n{report.rollback_reasoning}"
    console.print(Panel(footer, title="Outcome", expand=False))
    console.print(
        f"state machine: {' → '.join(report.state_machine_path)}", style="dim"
    )
    return True


def _emit(report: DeploymentReport, as_json: bool) -> None:
    if as_json:
        print(json.dumps(report.to_dict(), indent=2))
    elif not _render_rich(report):
        print(_render_plain(report))


# ---- subcommands -----------------------------------------------------------


def _cmd_run(args: argparse.Namespace) -> int:
    request, profile, seed = load_scenario(args.scenario)
    if args.seed is not None:
        seed = args.seed
    memory = None
    if args.memory:
        from .memory import DeploymentMemory

        memory = DeploymentMemory(args.memory)
    report = Orchestrator(asl_path=args.asl).deploy(
        request, profile, agent=_make_agent(args, memory), seed=seed, memory=memory
    )
    _emit(report, args.json)
    # Exit non-zero on rollback so CI pipelines can gate on it.
    return 0 if report.outcome == "promoted" else 2


def _cmd_tree(args: argparse.Namespace) -> int:
    request, _profile, _seed = load_scenario(args.scenario)
    tree = build_promotion_tree(request.thresholds)
    print(f"Promotion decision tree for {request.service} (thresholds shown):")
    print(json.dumps(request.thresholds.to_dict(), indent=2))
    print()
    print(render_tree(tree))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="agentic-deploy",
        description="Agent-driven CI/CD for ML + Robotics — deployments as decision trees.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="run a deployment scenario end-to-end")
    run.add_argument("scenario", help="path to a scenario YAML")
    run.add_argument("--json", action="store_true", help="emit the report as JSON")
    run.add_argument("--seed", type=int, default=None, help="override the scenario RNG seed")
    run.add_argument("--asl", default=None, help="override the Step Functions ASL path")
    run.add_argument("--force-heuristic", action="store_true", help="never call the LLM")
    run.add_argument("--force-llm", action="store_true", help="require the LLM agent")
    run.add_argument(
        "--memory",
        default=None,
        metavar="PATH",
        help="JSON deployment-memory store; plans get more cautious after rollbacks",
    )
    run.set_defaults(func=_cmd_run)

    tree = sub.add_parser("tree", help="render the promotion decision tree for a scenario")
    tree.add_argument("scenario", help="path to a scenario YAML")
    tree.set_defaults(func=_cmd_tree)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
