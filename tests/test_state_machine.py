"""Tests for the ASL (Step Functions) interpreter."""

from __future__ import annotations

import json
from pathlib import Path

from agentic_deploy.state_machine import StateMachine

REPO = Path(__file__).resolve().parents[1]


def test_task_choice_succeed_flow():
    definition = {
        "StartAt": "Set",
        "States": {
            "Set": {"Type": "Task", "Resource": "set", "Next": "Gate"},
            "Gate": {
                "Type": "Choice",
                "Choices": [{"Variable": "$.ok", "BooleanEquals": True, "Next": "Win"}],
                "Default": "Lose",
            },
            "Win": {"Type": "Succeed"},
            "Lose": {"Type": "Fail", "Error": "Nope"},
        },
    }
    sm = StateMachine(definition, {"set": lambda ctx: {"ok": True}})
    result = sm.run()
    assert result.status == "succeeded"
    assert result.visited == ["Set", "Gate", "Win"]


def test_fail_state_reports_error():
    definition = {
        "StartAt": "Gate",
        "States": {
            "Gate": {
                "Type": "Choice",
                "Choices": [{"Variable": "$.v", "NumericGreaterThan": 10, "Next": "Ok"}],
                "Default": "Bad",
            },
            "Ok": {"Type": "Succeed"},
            "Bad": {"Type": "Fail", "Error": "TooLow", "Cause": "v <= 10"},
        },
    }
    sm = StateMachine(definition, {})
    result = sm.run(initial_context={"v": 5})
    assert result.status == "failed"
    assert result.error == "TooLow"


def test_compound_and_choice():
    definition = {
        "StartAt": "Gate",
        "States": {
            "Gate": {
                "Type": "Choice",
                "Choices": [
                    {
                        "And": [
                            {"Variable": "$.verdict", "StringEquals": "promote"},
                            {"Variable": "$.more", "BooleanEquals": True},
                        ],
                        "Next": "Loop",
                    }
                ],
                "Default": "Done",
            },
            "Loop": {"Type": "Succeed"},
            "Done": {"Type": "Succeed"},
        },
    }
    sm = StateMachine(definition, {})
    assert sm.run({"verdict": "promote", "more": True}).visited[-1] == "Loop"
    assert sm.run({"verdict": "promote", "more": False}).visited[-1] == "Done"


def test_shipped_asl_is_valid_and_loads():
    asl = REPO / "deploy" / "stepfunctions" / "deployment.asl.json"
    definition = json.loads(asl.read_text())
    # A no-op handler set is enough to validate the interpreter accepts the doc.
    handlers = {
        r: (lambda ctx: {}) for r in ("plan_strategy", "provision", "rollout_step", "rollback", "promote")
    }
    sm = StateMachine(definition, handlers)
    assert sm.definition["StartAt"] == "PlanStrategy"
    assert "EvaluateGate" in sm.states
