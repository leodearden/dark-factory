"""The typed-plan read the Plan-model test files share (``orchestrator/artifacts.py::Plan``)."""

from __future__ import annotations

from orchestrator.artifacts import Plan, TaskArtifacts


def read_model(artifacts: TaskArtifacts) -> Plan:
    """The plan on disk as a Plan; fails the test when plan.json is absent."""
    plan = artifacts.read_plan_model()
    assert plan is not None, f'{artifacts.artifact_path("plan.json")} is absent'
    return plan
