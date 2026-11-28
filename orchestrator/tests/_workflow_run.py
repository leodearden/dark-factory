"""A real ``TaskWorkflow.run()`` in a fresh repo, driven only through the workflow's public entry point.

FakeCurator answers the curator's tools. The defect-record tools stay with the
FakeScheduler, whose FakeDefectRecords folds them into its task store.
"""

from __future__ import annotations

from pathlib import Path
from typing import Generic, TypeVar
from unittest.mock import AsyncMock

from _curator_fake import FakeCurator
from _info_l0_router_fixtures import PASSED, run_config
from _workflow_helpers import (
    AgentStub,
    FakeBriefing,
    _build_workflow_with_escalation,
    _init_repo,
    _make_git_ops,
)

from orchestrator.agents.review_briefing import ReviewBriefing
from orchestrator.event_store import EventStore
from orchestrator.scheduler import TaskAssignment
from orchestrator.workflow import WorkflowOutcome

_Recorder = TypeVar('_Recorder', bound=FakeBriefing)


class WorkflowRun(Generic[_Recorder]):
    """A real ``TaskWorkflow.run()`` of *assignment* in a fresh repo: *stub* answers every agent, *recorder* every briefing."""

    def __init__(
        self, tmp_path: Path, monkeypatch, stub: AgentStub, recorder: _Recorder,
        assignment: TaskAssignment, *, briefing: ReviewBriefing, router_enabled: bool,
        steward_factory: type | None = None,
    ) -> None:
        repo = tmp_path / 'repo'
        repo.mkdir()
        _init_repo(repo)
        config = run_config(repo, briefing=briefing, router_enabled=router_enabled)
        self.workflow, self.scheduler, self.queue = _build_workflow_with_escalation(
            config, _make_git_ops(config.project_root), assignment, stub, tmp_path,
            steward_factory=steward_factory,
        )
        self.briefing = recorder
        self.workflow.briefing = recorder
        self._monkeypatch = monkeypatch
        monkeypatch.setattr('orchestrator.workflow.invoke_agent', stub.invoke_agent)
        monkeypatch.setattr(
            'orchestrator.workflow.run_scoped_verification', AsyncMock(return_value=PASSED),
        )
        self.event_store = EventStore(tmp_path / 'runs.db', run_id='run-test')
        self.workflow.event_store = self.event_store
        self.curator = FakeCurator()
        self._scheduler_dispatch = self.scheduler.dispatch_tool
        self.workflow.scheduler.dispatch_tool = self._dispatch_tool

    async def _dispatch_tool(self, name: str, arguments: dict, *, timeout: float = 15) -> dict:
        if self.scheduler.defect_records.handles(name):
            return await self._scheduler_dispatch(name, arguments, timeout=timeout)
        return await self.curator(name, arguments, timeout=timeout)

    async def run(self) -> WorkflowOutcome:
        return (await self.workflow.run()).outcome
