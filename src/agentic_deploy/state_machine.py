"""A minimal AWS Step Functions (ASL) interpreter.

The deployment *logic* — plan, provision, step, gate, promote/rollback — is
defined declaratively in ``deploy/stepfunctions/deployment.asl.json`` as a real
Amazon States Language document. This interpreter executes that document against
Python task handlers, so the same artifact you could hand to AWS Step Functions is
what actually drives the local demo. That is the "Step Functions for deployment
logic" piece, made runnable.

Supported subset: ``Task`` (dispatches to a named handler), ``Choice`` (with the
common numeric/string/boolean comparators and ``And``/``Or``/``Not``), ``Pass``,
``Succeed``, and ``Fail``. That is enough to express a gated canary loop.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

Handler = Callable[[dict[str, Any]], dict[str, Any]]

_COMPARATORS: dict[str, Callable[[Any, Any], bool]] = {
    "StringEquals": lambda a, b: a == b,
    "BooleanEquals": lambda a, b: bool(a) == bool(b),
    "NumericEquals": lambda a, b: float(a) == float(b),
    "NumericGreaterThan": lambda a, b: float(a) > float(b),
    "NumericLessThan": lambda a, b: float(a) < float(b),
    "NumericGreaterThanEquals": lambda a, b: float(a) >= float(b),
    "NumericLessThanEquals": lambda a, b: float(a) <= float(b),
}


@dataclass
class Execution:
    """The result of running a state machine."""

    status: str                       # "succeeded" | "failed"
    context: dict[str, Any]
    visited: list[str] = field(default_factory=list)
    error: str | None = None
    cause: str | None = None


class StateMachine:
    """Executes an ASL definition against a set of task handlers."""

    def __init__(self, definition: dict[str, Any], handlers: dict[str, Handler]) -> None:
        self.definition = definition
        self.handlers = handlers
        self.states = definition["States"]

    def run(self, initial_context: dict[str, Any] | None = None, max_transitions: int = 1000) -> Execution:
        context: dict[str, Any] = dict(initial_context or {})
        visited: list[str] = []
        name = self.definition["StartAt"]

        for _ in range(max_transitions):
            state = self.states[name]
            visited.append(name)
            stype = state["Type"]

            if stype == "Task":
                handler = self.handlers.get(state["Resource"])
                if handler is None:
                    raise KeyError(f"no handler registered for Task resource {state['Resource']!r}")
                context.update(handler(context) or {})
                if state.get("End"):
                    return Execution("succeeded", context, visited)
                name = state["Next"]

            elif stype == "Pass":
                if "Result" in state:
                    context.update(state["Result"])
                if state.get("End"):
                    return Execution("succeeded", context, visited)
                name = state["Next"]

            elif stype == "Choice":
                name = self._choose(state, context)

            elif stype == "Succeed":
                return Execution("succeeded", context, visited)

            elif stype == "Fail":
                return Execution(
                    "failed", context, visited,
                    error=state.get("Error"), cause=state.get("Cause"),
                )

            else:  # pragma: no cover - defensive
                raise ValueError(f"unsupported state type {stype!r} in state {name!r}")

        raise RuntimeError("state machine exceeded max transitions (possible loop)")

    # ---- Choice evaluation -------------------------------------------------

    def _choose(self, state: dict[str, Any], context: dict[str, Any]) -> str:
        for rule in state.get("Choices", []):
            nxt = rule["Next"]
            if self._match(rule, context):
                return nxt
        if "Default" not in state:
            raise ValueError("Choice state matched no rule and has no Default")
        return state["Default"]

    def _match(self, rule: dict[str, Any], context: dict[str, Any]) -> bool:
        if "And" in rule:
            return all(self._match(r, context) for r in rule["And"])
        if "Or" in rule:
            return any(self._match(r, context) for r in rule["Or"])
        if "Not" in rule:
            return not self._match(rule["Not"], context)
        var = _resolve(rule["Variable"], context)
        for op, fn in _COMPARATORS.items():
            if op in rule:
                return fn(var, rule[op])
        raise ValueError(f"choice rule has no supported comparator: {rule!r}")


def _resolve(path: str, context: dict[str, Any]) -> Any:
    """Resolve a simple JSONPath like ``$.a.b`` against the context dict."""

    if not path.startswith("$"):
        return path
    node: Any = context
    for part in path.lstrip("$").lstrip(".").split("."):
        if part == "":
            continue
        node = node[part]
    return node
