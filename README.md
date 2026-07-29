# Agentic CI/CD for ML + Robotics (Beyond GitOps)

**Deployments are decision trees, not scripts.**

A GitOps pipeline runs a fixed sequence: shift traffic on a timer, hope the alerts
fire, page a human when they don't. This project replaces that script with an
**agent-driven deployment engine** that *plans* a rollout strategy, *simulates* the
impact on a robot fleet, *chooses* canary vs blue/green, and *rolls back with
explainable reasoning* — gated at every step by a decision tree over live health
signals.

It is a **faithful, self-contained simulation**. It ships the real infrastructure
artifacts you would deploy (a Dockerfile, Kubernetes manifests, an AWS Step Functions
state machine, Prometheus rules) *and* a local engine that actually executes them end
to end — so the whole thing runs anywhere, with **no cloud, no cluster, and no API
key**. A reproducible benchmark compares it against a scripted GitOps baseline.

```
                 ┌──────────────────────────────────────────────────────────┐
                 │  Deployment Agent  (Claude claude-opus-4-8 · or heuristic) │
   change  ────► │  plans strategy (canary vs blue/green) + reasons rollback │
                 └───────────────┬──────────────────────────────────────────┘
                                 │ strategy + steps
                 ┌───────────────▼──────────────────────────────────────────┐
                 │  Step Functions state machine (deployment.asl.json)       │
                 │  Plan → Provision → (RolloutStep → Gate)* → Promote/Roll  │
                 └───────────────┬──────────────────────────────────────────┘
             per step │ shift traffic (mock K8s: canary weight / blue-green cutover)
                      ▼
     ┌────────────────────────────┐        ┌──────────────────────────────┐
     │  Robotics fleet simulator  │──────► │  Prometheus health signals    │
     │  task success, cycle time, │        │  error rate, p99, task_ok,    │
     │  safety incidents          │        │  safety_incidents (exposition)│
     └────────────────────────────┘        └───────────────┬──────────────┘
                                                            │ HealthSnapshot
                                            ┌───────────────▼──────────────┐
                                            │  Promotion DECISION TREE      │
                                            │  → promote / hold / rollback  │
                                            │  (records the path it took)   │
                                            └───────────────────────────────┘
```

---

## Quickstart

```bash
pip install -e .            # core only (PyYAML). Add [all] for Claude + rich + prometheus_client.

# Run a scenario end-to-end (explainable report):
agentic-deploy run scenarios/healthy_canary.yaml       # → PROMOTED
agentic-deploy run scenarios/regression_rollback.yaml  # → ROLLED_BACK (caught at 50%)
agentic-deploy run scenarios/safety_incident.yaml      # → ROLLED_BACK (latent safety fault)
agentic-deploy run scenarios/late_latent_soak.yaml     # → ROLLED_BACK (caught in soak/bake)
agentic-deploy run scenarios/telemetry_dropout.yaml    # → ROLLED_BACK (fail-safe on lost telemetry)

# Let the agent learn from history (plans get cautious after rollbacks):
agentic-deploy run scenarios/healthy_canary.yaml --memory deploy-history.json

# Inspect the promotion decision tree for a scenario:
agentic-deploy tree scenarios/safety_incident.yaml

# Reproduce the benchmark (scripted GitOps vs agentic):
python benchmarks/benchmark.py --seeds 200

# Full walkthrough of all scenarios + a blue/green example:
python examples/run_demo.py
```

`agentic-deploy run` exits non-zero on a rollback, so it drops straight into a CI gate.

---

## What a run looks like

```
DEPLOYMENT  grasp-planner  grasp-net:v2.2.1 → grasp-net:v3.0.0
Agent plan   : canary  (risk=1.00, source=heuristic)
Rationale    : Graduated canary limits blast radius and gates each step on robot task health.
State machine: PlanStrategy → ProvisionGreen → RolloutStep → EvaluateGate → ... → Rollback → RolledBack
------------------------------------------------------------------------
step 0  traffic= 10%  err=0.018  task_ok=0.970  safety=0  →  PROMOTE
step 1  traffic= 25%  err=0.030  task_ok=0.934  safety=0  →  PROMOTE
step 2  traffic= 50%  err=0.049  task_ok=0.886  safety=0  →  ROLLBACK
          tree: safety_incident=no / task_success_below_min=yes / [rollback]
          breached: task_success_rate
------------------------------------------------------------------------
OUTCOME      : ROLLED_BACK      Rollback time: 15s (breach → safe state)
Agent says   : Rolled back grasp-net:v3.0.0 at 50% traffic: task_success_rate=0.886
               (guardrail < 0.9). The fleet of 80 robots was reverted to grasp-net:v2.2.1.
```

The candidate regressed robot task success. Blended across a partly-canaried fleet the
drop stayed hidden at 10% and 25% traffic — then crossed the floor at 50%, and the
decision tree rolled back **before the whole fleet was exposed**. Every gate prints the
exact path it walked, so the decision is auditable, not a black box.

---

## The innovation: the gate is a decision tree

`src/agentic_deploy/decision_tree.py` models each promotion gate as an explicit tree of
`(condition → branch)` nodes over a `HealthSnapshot`, terminating in a
`promote` / `hold` / `rollback` leaf. Evaluating it records the **ordered path of nodes
visited** and the **breached guardrails** — that trace *is* the explanation.

The default tree encodes a safety-first ordering:

0. **Telemetry integrity** — the fail-safe root. If the fleet's metrics pipeline is
   dark, stale-but-green numbers are *not* evidence of health: the gate **holds**
   instead of promoting on unknown state, and a sustained outage escalates to a
   rollback. A one-window blip recovers and the rollout continues.
1. **Safety incidents** — terminal. Any violation rolls back immediately.
2. **Robot task success rate** below floor → rollback.
3. **Inference error rate** / **p99 latency** guardrails → rollback.
4. **Cycle-time regression** → *hold* (wait and re-measure) rather than roll back; a
   capped number of holds escalates to a rollback.
5. Otherwise → **promote**.

Because the tree is *data*, the agent can tune its thresholds per change, and a report
can replay precisely why any decision was made. Render it for any scenario with
`agentic-deploy tree <scenario>`.

---

## The agent: Claude, with a deterministic fallback

`src/agentic_deploy/agent.py` does two things — **choose a rollout strategy** and
**explain a rollback** — backed by Claude (`claude-opus-4-8`, adaptive thinking,
structured outputs) when the `anthropic` SDK and credentials are present, and by a
deterministic heuristic otherwise. Both paths return the *same shapes*, so the engine
and the test suite never care which ran, and every LLM call degrades gracefully to the
heuristic on any error.

- High blast radius or safety-critical task → **graduated canary** (limit exposure).
- Low risk, non-critical task → **blue/green** (fast, atomic, reversible cutover).

Set `ANTHROPIC_API_KEY` (or run `ant auth login`) and install `pip install -e .[llm]`
to use the real model; force either path with `--force-llm` / `--force-heuristic`.

### Deployment memory: the agent learns from history

Pass `--memory deploy-history.json` and the engine keeps an append-only record of
every deployment outcome. The agent consults it when planning:

- recent rollbacks of a service **raise its risk score** (and the rationale says so),
- a service whose *last* deployment rolled back gets a **5% bake-in step** prepended —
  the cheapest possible probe before real traffic is at stake — and a low-risk change
  that would have been blue/green is canaried instead,
- a clean streak restores normal confidence.

```
Rationale: ... History: 100% of recent deployments of grasp-planner rolled back
(breached: task_success_rate); raising risk accordingly. Last deployment of this
service rolled back — prepending a 5% bake-in step before shifting real traffic.
```

### Soak (bake) gate: catching faults that pass every canary step

The classic incident is the fault that *only* appears after full rollout — memory
growth, thermal load, cache churn under sustained 100% traffic. After the last
rollout step, the state machine enters a **SoakTest → EvaluateSoak** loop: the fleet
bakes at 100% across multiple observation windows and every window re-runs the same
decision tree. A late-activating breach is caught during the bake and rolled back
*before* the candidate is finalized as the new baseline — turning a
"promoted-then-fleet-wide-incident" into a contained, explained rollback. See
`scenarios/late_latent_soak.yaml`.

---

## Faithful infrastructure, actually executed

The `deploy/` artifacts are real, and the engine drives them rather than mocking around
them:

| Concern | Artifact | How the engine uses it |
| --- | --- | --- |
| **Step Functions** | `deploy/stepfunctions/deployment.asl.json` | Valid Amazon States Language; `state_machine.py` interprets it (`Task`/`Choice`/`Succeed`/`Fail`) to drive the actual run. |
| **Kubernetes** | `deploy/k8s/*.yaml` | Blue/green Deployments, a Service, and an Argo Rollouts canary CR; the mock controller applies them and shifts traffic/cuts over. |
| **Prometheus** | `deploy/prometheus/rules.yaml` | Recording + alerting rules; `prometheus.py` emits the same metric names in valid text **exposition format** and answers the gate's queries. |
| **Docker** | `deploy/Dockerfile` | Buildable model-serving image parameterized by `MODEL_VERSION` (blue = baseline, green = candidate). |

The canary `AnalysisTemplate` queries (`robot_task_success_rate`,
`inference_error_rate`, `robot_safety_incidents_total`) mirror the guardrails the
decision tree enforces — the declarative and the agentic gate agree.

---

## Scenarios

| Scenario | Candidate behavior | Outcome |
| --- | --- | --- |
| `healthy_canary.yaml` | Small improvement across all signals | **Promoted** through 10→25→50→100% + soak |
| `regression_rollback.yaml` | Quality regression (task success + error rate) | **Rolled back** at 50% — before full-fleet exposure |
| `safety_incident.yaml` | Healthy metrics but a **latent** safety fault at 50% | **Rolled back** on the safety signal |
| `late_latent_soak.yaml` | Fault that activates only *after* full rollout | Passes every canary gate; **rolled back in soak** |
| `telemetry_dropout.yaml` | Healthy candidate, but the metrics pipeline goes dark | **Holds** (fail-safe), then **rolled back** when telemetry doesn't recover |

Scenarios are plain YAML: a `DeploymentRequest`, gate `thresholds`, and a
`BehaviorProfile` describing how the candidate differs from the baseline (including a
latent regression that only activates at a later step). Runs are fully seeded and
reproducible.

---

## Benchmark: agentic vs scripted GitOps

`benchmarks/benchmark.py` generates many randomized rollouts — a mix of healthy
candidates, quality regressions, and latent safety bugs — and deploys each with two
strategies:

- **scripted baseline** — fixed-schedule GitOps with no per-step gating; reaches 100%
  before any post-deploy alert fires, so a faulty model hits the whole fleet first and
  is reverted only after a slow mean-time-to-detect.
- **agentic** — this engine; the decision tree gates every step.

Representative run (`--seeds 200 --seed 1234`, fully reproducible):

```
Randomized rollouts : 200 (82 faulty candidates)
                       scripted baseline     agentic
failed deployments                    82          19
mean rollback (s)                  300.0        46.9
------------------------------------------------------------
Failed deployments   ↓ 76.8%
Mean rollback time   ↓ 84.4%
```

Of the faulty candidates the agentic engine *does* let reach 100%, the late-activating
ones are now caught by the **soak gate** and reverted — the scripted baseline promotes
them and ships a fleet-wide incident.

A *failed deployment* is a faulty candidate that reached **100% of the fleet** (a
fleet-wide incident); *rollback time* is simulated seconds from breach to a safe state.

> These are **simulation** figures against a deliberately naive baseline, so they read
> optimistically. In the real world the target outcomes are more modest — on the order
> of **↓35% failed deployments** and **↓60% mean rollback time** — but the mechanism is
> the same: gate on fleet health at every step and contain blast radius instead of
> discovering problems after full rollout. The value the project always delivers is
> **explainable** deployment decisions.

---

## Project layout

```
src/agentic_deploy/
  models.py         data model (DeploymentRequest, HealthSnapshot, Decision, ...)
  decision_tree.py  the promotion decision-tree engine  ← core idea
  agent.py          strategy planning + rollback reasoning (Claude / heuristic)
  memory.py         deployment memory — plans learn from past outcomes
  robotics_sim.py   seeded fleet simulator (latent/transient faults, telemetry loss)
  prometheus.py     health-signal provider + text exposition
  kubernetes.py     mock cluster: canary weight / blue-green cutover
  strategies.py     step → cluster-op adapters
  state_machine.py  minimal AWS Step Functions (ASL) interpreter
  orchestrator.py   wires it all behind the state machine (incl. the soak gate)
  cli.py            `agentic-deploy` entry point (--memory for history-aware plans)
deploy/             Dockerfile, k8s manifests, Step Functions ASL, Prometheus rules
scenarios/          five end-to-end scenarios
benchmarks/         reproducible baseline-vs-agentic benchmark
examples/           run_demo.py — scripted walkthrough
tests/              pytest suite incl. industrial edge cases (no API key required)
```

---

## Testing

```bash
pip install -e .[dev]
pytest -q
```

The suite covers the decision tree, the deterministic agent, the fleet simulator, the
ASL interpreter, full-scenario orchestration, deployment memory, and a battery of
**industrial edge cases**: sustained vs transient telemetry outages, faults that pass
every canary gate (caught in soak), warm-up transients that hold-then-recover,
persistent soft regressions that escalate, exact-boundary metric values, blue/green
fault containment, and single-robot / zero-robot fleets. It runs entirely offline —
the agent tests exercise the heuristic path, so no credentials are needed.

---

## License

MIT — see [LICENSE](LICENSE).
