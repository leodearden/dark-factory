"""The implementer is handed a worktree whose ``.task/plan.json`` reads this task's plan.

Task 3608.  Between planning and EXECUTE the workflow checks the lane's
agent-facing plan link, relinks it once when it is broken, and blocks the task
as an ``infra_issue`` rather than dispatch an implementer whose first read
would fail.  Both cases drive the public ``TaskWorkflow.run()`` in a fresh
repo, breaking the lane link right after the architect writes the plan.
"""

from __future__ import annotations

import json
import os
from collections.abc import Callable, Iterator
from pathlib import Path
from unittest.mock import AsyncMock

import pytest
from _curator_fake import FakeCurator
from _info_l0_router_fixtures import PASSED, TASK_ID, assignment, run_config
from _workflow_helpers import (
    AgentStub,
    _build_workflow_with_escalation,
    _derive_meta_root_like_production,  # noqa: F401  autouse fixture, see its docstring
    _init_repo,
    _make_git_ops,
)
from shared.cli_invoke import AgentResult

from orchestrator.agents.review_briefing import ReviewBriefing
from orchestrator.event_store import EventStore
from orchestrator.workflow import WorkflowOutcome


def _lane_plan(cwd: Path) -> Path:
    return cwd / '.task' / 'plan.json'


class _LaneBreakingStub(AgentStub):
    """Breaks the lane plan link after each plan, and records what the first implementer read there."""

    def __init__(self, break_lane: Callable[[Path], None]) -> None:
        super().__init__()
        self._break_lane = break_lane
        self.lane_paths: list[Path] = []
        self.first_implementer_read: tuple[bool, str] | None = None

    async def _architect(self, cwd: Path) -> AgentResult:
        result = await super()._architect(cwd)
        self.lane_paths.append(_lane_plan(cwd))
        self._break_lane(_lane_plan(cwd))
        return result

    async def _implementer(self, cwd: Path) -> AgentResult:
        if self.first_implementer_read is None:
            lane = _lane_plan(cwd)
            self.first_implementer_read = (lane.is_symlink(), lane.read_text())
        return await super()._implementer(cwd)


async def _run(tmp_path: Path, monkeypatch, stub: AgentStub):
    repo = tmp_path / 'repo'
    repo.mkdir()
    _init_repo(repo)
    config = run_config(repo, briefing=ReviewBriefing.DIFF_ONLY, router_enabled=True)
    workflow, _scheduler, queue = _build_workflow_with_escalation(
        config, _make_git_ops(config.project_root), assignment(), stub, tmp_path,
    )
    monkeypatch.setattr('orchestrator.workflow.invoke_agent', stub.invoke_agent)
    monkeypatch.setattr(
        'orchestrator.workflow.run_scoped_verification', AsyncMock(return_value=PASSED),
    )
    workflow.event_store = EventStore(tmp_path / 'runs.db', run_id='run-test')
    workflow.scheduler.dispatch_tool = FakeCurator()
    outcome = (await workflow.run()).outcome
    return outcome, queue


def _replace_with_empty_file(lane: Path) -> None:
    lane.unlink()
    lane.write_text('')


@pytest.fixture
def unremovable_dirs() -> Iterator[list[Path]]:
    """Directories a test makes read-only; each is made writable again at teardown."""
    locked: list[Path] = []
    yield locked
    for path in locked:
        if path.exists():
            path.chmod(0o700)


@pytest.mark.asyncio
async def test_a_lane_plan_broken_after_planning_is_relinked_before_the_implementer_reads_it(
    tmp_path, monkeypatch,
):
    stub = _LaneBreakingStub(_replace_with_empty_file)

    outcome, _queue = await _run(tmp_path, monkeypatch, stub)

    assert outcome == WorkflowOutcome.DONE
    assert stub.first_implementer_read is not None
    was_link, read = stub.first_implementer_read
    assert was_link
    assert [step['id'] for step in json.loads(read)['steps']] == ['step-1', 'step-2']


@pytest.mark.asyncio
@pytest.mark.skipif(os.geteuid() == 0, reason='root ignores the directory mode that makes the relink fail')
async def test_a_lane_plan_that_cannot_be_relinked_blocks_before_any_implementer_runs(
    tmp_path, monkeypatch, unremovable_dirs,
):
    def replace_with_unremovable_dir(lane: Path) -> None:
        lane.unlink()
        locked = lane / 'locked'
        locked.mkdir(parents=True)
        (locked / 'held.txt').write_text('held')
        locked.chmod(0o500)
        unremovable_dirs.append(locked)

    stub = _LaneBreakingStub(replace_with_unremovable_dir)

    outcome, queue = await _run(tmp_path, monkeypatch, stub)

    assert outcome == WorkflowOutcome.BLOCKED
    assert 'implementer' not in stub.calls
    lane = str(stub.lane_paths[-1])
    refusals = [
        esc for esc in queue.get_by_task(TASK_ID, status='pending')
        if esc.category == 'infra_issue' and (lane in esc.summary or lane in esc.detail)
    ]
    assert refusals, [(esc.category, esc.summary) for esc in queue.get_by_task(TASK_ID)]
