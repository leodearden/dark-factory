"""Tests for the ``StewardOutcome`` sum type (W9-δ).

Replaces the workflow's forensic re-read of escalation-queue files with a
typed, in-process channel carrying one of five frozen-dataclass variants.
Mirrors the ``TerminalReport`` frozen-dataclass precedent (task W9-γ / 2247,
see ``test_workflow_terminal_report.py``): each variant is frozen, carries a
typed payload, and is re-exported through the ``orchestrator.workflow`` shim.

PRD ``plans/workflow-state-machine-prd.md`` §10 / Contract §8.1-8.2 SO-1 /
Resolved decision D6.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest
from _orch_helpers import pydantic_spec

from orchestrator import workflow_types
from orchestrator.artifacts import TaskArtifacts
from orchestrator.config import OrchestratorConfig
from orchestrator.workflow import TaskWorkflow
from orchestrator.workflow_types import (
    StewardBudgetExhausted,
    StewardInterrupted,
    StewardOutcome,
    StewardReescalatedL1,
    StewardResolved,
    StewardTerminalDecision,
)


class TestStewardOutcomeExports:
    """Every variant + the StewardOutcome alias is public and shim-reexported."""

    @pytest.mark.parametrize('name', [
        'StewardResolved',
        'StewardReescalatedL1',
        'StewardTerminalDecision',
        'StewardInterrupted',
        'StewardBudgetExhausted',
        'StewardOutcome',
    ])
    def test_name_in_workflow_types_all(self, name):
        assert name in workflow_types.__all__

    def test_reexported_from_workflow_module(self):
        from orchestrator.workflow import (
            StewardBudgetExhausted as ShimBudgetExhausted,
        )
        from orchestrator.workflow import (
            StewardInterrupted as ShimInterrupted,
        )
        from orchestrator.workflow import (
            StewardOutcome as ShimOutcome,
        )
        from orchestrator.workflow import (
            StewardReescalatedL1 as ShimReescalatedL1,
        )
        from orchestrator.workflow import (
            StewardResolved as ShimResolved,
        )
        from orchestrator.workflow import (
            StewardTerminalDecision as ShimTerminalDecision,
        )

        assert ShimResolved is StewardResolved
        assert ShimReescalatedL1 is StewardReescalatedL1
        assert ShimTerminalDecision is StewardTerminalDecision
        assert ShimInterrupted is StewardInterrupted
        assert ShimBudgetExhausted is StewardBudgetExhausted
        assert ShimOutcome is StewardOutcome

    def test_steward_outcome_alias_covers_every_variant(self):
        """StewardOutcome is the PEP-604 union of all five variants."""
        import types

        assert isinstance(StewardOutcome, types.UnionType)
        members = set(StewardOutcome.__args__)
        assert members == {
            StewardResolved,
            StewardReescalatedL1,
            StewardTerminalDecision,
            StewardInterrupted,
            StewardBudgetExhausted,
        }


def _make_workflow(*, tmp_path: Path, task_id: str = '2248') -> TaskWorkflow:
    """Minimal ``TaskWorkflow`` harness — mirrors ``test_workflow.py``'s own
    ``_make_workflow``.  Duplicated here (rather than imported) because this
    amendment pass's module lock covers ``test_steward_outcome.py`` but not
    ``test_workflow.py`` — see the repo-wide convention of a per-test-file
    ``_make_workflow`` copy (26 test files already do this).
    """
    assignment = MagicMock()
    assignment.task_id = task_id
    assignment.task = {'id': task_id, 'title': 'T', 'description': 'd'}
    assignment.modules = ['mod_a']

    _spec = pydantic_spec(OrchestratorConfig)
    config = MagicMock(spec_set=_spec)
    config.fused_memory.project_id = 'dark_factory'
    config.fused_memory.url = 'http://localhost:8002'
    config.max_review_cycles = 2
    config.max_amendment_rounds = 1
    config.lock_depth = 2
    config.steward_completion_timeout = 300.0
    config.project_root = tmp_path / 'proj'
    config.merge_train_former_enabled = False
    config.merge_train_max_members = 3

    scheduler = MagicMock()
    git_ops = MagicMock()

    wf = TaskWorkflow(
        assignment=assignment,
        config=config,
        git_ops=git_ops,
        scheduler=scheduler,
        briefing=MagicMock(),
        mcp=MagicMock(),
    )
    worktree = tmp_path / 'wt'
    worktree.mkdir(parents=True, exist_ok=True)
    (worktree / '.task').mkdir(exist_ok=True)
    wf.artifacts = TaskArtifacts(worktree)
    wf.worktree = worktree
    wf.merge_queue = MagicMock()
    wf.plan = {'files': ['a.py', 'b.py', 'c.py']}
    wf._base_commit = 'base_sha'
    wf._module_configs = []
    wf.git_ops.rebind_branch_to_head = AsyncMock(return_value=True)
    return wf


@pytest.mark.asyncio
class TestAwaitStewardCompletionCancelSafety:
    """Amendment (review fix — ``robustness_behavior_change``): a soft-cancel
    must never be routed into the task-2060 resume-plan branch.

    Before this fix, ``_await_steward_completion`` derived
    ``wip_commits_present`` from the worktree probe unconditionally whenever
    the channel produced nothing — including when ``self._cancel_event`` was
    (or became) set.  If the worktree happened to carry WIP,
    ``_mark_blocked``'s ``StewardInterrupted(wip=True)`` branch would REQUEUE
    the task and dismiss its pending L0 — even though this workflow slot is
    merely being soft-cancelled/preempted, not failing.  The fix forces
    ``wip_commits_present=False`` whenever cancellation is in play, so the
    outcome always routes to the no-requeue BLOCKED+L1 fall-through instead
    (mirrors the pre-W9-delta behaviour for a cancel-interrupted wait, where
    the L0 was still pending and fell through to the generic BLOCKED+L1
    path rather than being treated as a resolved/resumable case).
    """

    async def test_cancel_already_set_on_entry_forces_no_wip(
        self, tmp_path: Path,
    ):
        """``_cancel_event`` set BEFORE the call skips the wait outright —
        the synthesized outcome must still carry wip=False even though the
        worktree probe would say True."""
        wf = _make_workflow(tmp_path=tmp_path)
        wf.config.steward_completion_timeout = 5.0
        wf._steward_outcome_channel = asyncio.Queue()
        wf.scheduler.get_status = AsyncMock(return_value='blocked')
        wf._worktree_has_wip_commits = AsyncMock(return_value=True)
        wf._cancel_event.set()

        outcome = await asyncio.wait_for(
            wf._await_steward_completion(), timeout=2,
        )

        assert outcome == StewardInterrupted('timeout', wip_commits_present=False)

    async def test_cancel_fired_during_wait_forces_no_wip(
        self, tmp_path: Path,
    ):
        """``_cancel_event`` fires WHILE waiting (nothing published on the
        channel) — same guarantee: wip is forced False, not derived."""
        wf = _make_workflow(tmp_path=tmp_path)
        wf.config.steward_completion_timeout = 5.0
        wf._steward_outcome_channel = asyncio.Queue()
        wf.scheduler.get_status = AsyncMock(return_value='blocked')
        wf._worktree_has_wip_commits = AsyncMock(return_value=True)

        async def _do_cancel():
            await asyncio.sleep(0.01)
            wf._cancel_event.set()

        asyncio.create_task(_do_cancel())
        outcome = await asyncio.wait_for(
            wf._await_steward_completion(), timeout=2,
        )

        assert outcome == StewardInterrupted('timeout', wip_commits_present=False)

    async def test_genuine_timeout_without_cancel_still_derives_wip(
        self, tmp_path: Path,
    ):
        """Control: a REAL grace-timeout (no cancellation involved) is
        unaffected by this fix — wip is still derived from the probe."""
        wf = _make_workflow(tmp_path=tmp_path)
        wf.config.steward_completion_timeout = 0.05
        wf._steward_outcome_channel = asyncio.Queue()
        wf.scheduler.get_status = AsyncMock(return_value='blocked')
        wf._worktree_has_wip_commits = AsyncMock(return_value=True)

        outcome = await asyncio.wait_for(
            wf._await_steward_completion(), timeout=2,
        )

        assert outcome == StewardInterrupted('timeout', wip_commits_present=True)


@pytest.mark.asyncio
class TestAwaitStewardCompletionOverrideDismissesOrphanL0:
    """Amendment (review fix — ``orphan_resource_leak``): the ``has_open_l1``
    override that substitutes a requeue-producing outcome with
    ``StewardReescalatedL1`` must dismiss the still-pending L0 when the
    substituted outcome was a wip-present ``StewardInterrupted`` — that
    publisher deliberately skips ``_auto_escalate_to_human`` (task-2060
    fix), so nothing else dismisses it, and ``_mark_blocked``'s
    ``StewardReescalatedL1`` branch assumes it is already gone (true only
    for a steward-PUBLISHED ``StewardReescalatedL1``).
    """

    async def test_wip_interrupted_override_dismisses_pending_l0(
        self, tmp_path: Path,
    ):
        wf = _make_workflow(tmp_path=tmp_path)
        wf.config.steward_completion_timeout = 5.0
        wf._steward_outcome_channel = asyncio.Queue()
        wf._steward_outcome_channel.put_nowait(
            StewardInterrupted('attempt_cap', wip_commits_present=True),
        )
        wf.scheduler.get_status = AsyncMock(return_value='blocked')
        wf.escalation_queue = MagicMock()
        wf.escalation_queue.has_open_l1 = MagicMock(return_value=True)
        pending_l0 = MagicMock(id='esc-orig-0')

        def _get_by_task(task_id, *, status, level):  # noqa: ARG001
            if level == 1:
                return [MagicMock(id='L1')]
            assert level == 0
            assert status == 'pending'
            return [pending_l0]

        wf.escalation_queue.get_by_task = MagicMock(side_effect=_get_by_task)

        outcome = await wf._await_steward_completion()

        assert outcome == StewardReescalatedL1(esc_id='L1')
        wf.escalation_queue.resolve.assert_called_once()
        call = wf.escalation_queue.resolve.call_args
        assert call.args[0] == 'esc-orig-0'
        assert call.kwargs.get('dismiss') is True
        assert call.kwargs.get('resolved_by') == 'auto-dismissed'

    async def test_resolved_override_does_not_touch_l0(
        self, tmp_path: Path,
    ):
        """A plain ``StewardResolved`` override (not wip-present-interrupted)
        must NOT run the new dismissal loop — that outcome's own L0 is
        already resolved by the steward itself; this amendment is scoped to
        the wip-present ``StewardInterrupted`` case only."""
        wf = _make_workflow(tmp_path=tmp_path)
        wf.config.steward_completion_timeout = 5.0
        wf._steward_outcome_channel = asyncio.Queue()
        wf._steward_outcome_channel.put_nowait(
            StewardResolved(resolution_text='fixed it'),
        )
        wf.scheduler.get_status = AsyncMock(return_value='blocked')
        wf.escalation_queue = MagicMock()
        wf.escalation_queue.has_open_l1 = MagicMock(return_value=True)
        wf.escalation_queue.get_by_task = MagicMock(
            return_value=[MagicMock(id='L1')],
        )

        outcome = await wf._await_steward_completion()

        assert outcome == StewardReescalatedL1(esc_id='L1')
        wf.escalation_queue.resolve.assert_not_called()
