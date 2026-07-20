"""Tests for durable merge-retry intent on merge-phase resume (task 2795).

When a merge-phase escalation is resolved via ``resume``, ``_requeue`` returns
``WorkflowOutcome.REQUEUED`` while leaving the task ``in-progress`` for an
in-RAM in-place merge retry — a restart mid-retry silently loses that
obligation (Reify 5166). This module unit-tests the durable fix:

* ``_stamp_merge_retry_pending`` / ``_clear_merge_retry_pending`` — best-effort
  metadata stamp/clear helpers (mirroring ``_stamp_first_merge_enqueue``).
* the ``_requeue`` wiring — the merge_phase=True StewardResolved path stamps
  before returning REQUEUED; the merge_phase=False path does not.
* ``_merge_and_finalise`` — the extracted MERGE+SUCCESS tail shared by the
  normal ``_drive`` path and the resume guard.
* ``_resume_merge_retry_if_pending`` — the ``_drive`` early guard that jumps
  straight back to the merge phase when the post-rebase worktree HEAD still
  matches the stamped ``branch_head``.

Follows the mock-fixture style of ``test_workflow_merge_thrash.py``: a local
``_make`` builds a minimal :class:`TaskWorkflow` with mocks, and the
module-level ``_run`` (git rev-parse) is controlled by monkeypatching
``orchestrator.workflow._run``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from _orch_helpers import pydantic_spec

from orchestrator.config import OrchestratorConfig
from orchestrator.workflow import TaskWorkflow, WorkflowOutcome, WorkflowState
from orchestrator.workflow_types import StewardResolved


@dataclass
class _Fixture:
    wf: TaskWorkflow
    update_task: AsyncMock
    scheduler: MagicMock
    git_ops: MagicMock
    queue: MagicMock


def _make(
    *,
    task_id: str = '77',
    metadata: dict | None = None,
    worktree: Path = Path('/tmp/wt-2795'),
    main_sha: str = 'BASE-SHA',
    backend_metadata: dict | None = None,
    update_task_raises: bool = False,
) -> _Fixture:
    assignment = MagicMock()
    assignment.task_id = task_id
    assignment.task = {
        'id': task_id, 'title': 'T', 'description': 'd',
        'metadata': metadata or {},
    }
    assignment.modules = ['mod_a']

    _spec = pydantic_spec(OrchestratorConfig)
    config = MagicMock(spec_set=_spec)
    config.fused_memory.project_id = 'dark_factory'
    config.fused_memory.url = 'http://localhost:8002'
    config.lock_depth = 2
    config.steward_completion_timeout = 300.0
    config.project_root = Path('/tmp/non-existent-for-test')

    if update_task_raises:
        update_task = AsyncMock(side_effect=RuntimeError('mcp down'))
    else:
        update_task = AsyncMock(return_value=True)

    scheduler = MagicMock()
    scheduler.update_task = update_task
    scheduler.set_task_status = AsyncMock()
    scheduler.get_status = AsyncMock(return_value='in-progress')
    # _merge_fresh_metadata reads backend metadata before every stamp/clear.
    scheduler.get_task = AsyncMock(
        return_value={'metadata': backend_metadata if backend_metadata is not None else {}}
    )

    git_ops = MagicMock()
    git_ops.get_main_sha = AsyncMock(return_value=main_sha)

    queue = MagicMock()
    queue.get_by_task = MagicMock(return_value=[])

    wf = TaskWorkflow(
        assignment=assignment,
        config=config,
        git_ops=git_ops,
        scheduler=scheduler,
        briefing=MagicMock(),
        mcp=MagicMock(),
        escalation_queue=queue,  # type: ignore[arg-type]
    )
    wf.artifacts = MagicMock()
    wf.worktree = worktree

    return _Fixture(
        wf=wf, update_task=update_task, scheduler=scheduler,
        git_ops=git_ops, queue=queue,
    )


def _persisted_metadata(update_task: AsyncMock) -> dict:
    assert update_task.await_args is not None
    args, kwargs = update_task.await_args
    return kwargs.get('metadata') or args[1]


def _fake_run(*, head: str = 'HEAD-SHA', rc: int = 0):
    """Return an async stand-in for orchestrator.workflow._run (git rev-parse)."""

    async def _run(cmd, cwd=None, **kwargs):  # noqa: ARG001
        if rc == 0:
            return (0, head + '\n', '')
        return (rc, '', 'fatal: bad revision')

    return _run


_STAMP = {'branch_head': 'HEAD-SHA', 'base_sha': 'BASE-SHA', 'resolved_at': '2026-07-20T05:00:00+00:00'}


# ---------------------------------------------------------------------------
# step-3: stamp / clear helpers
# ---------------------------------------------------------------------------


class TestStampClearHelpers:
    @pytest.mark.asyncio
    async def test_stamp_persists_branch_head_base_sha_resolved_at(self, monkeypatch):
        f = _make(metadata={'retry_ledger': {'x': 1}}, main_sha='BASE-SHA')
        monkeypatch.setattr('orchestrator.workflow._run', _fake_run(head='HEAD-SHA'))

        await f.wf._stamp_merge_retry_pending()

        meta = _persisted_metadata(f.update_task)
        mrp = meta['merge_retry_pending']
        assert mrp['branch_head'] == 'HEAD-SHA'
        assert mrp['base_sha'] == 'BASE-SHA'
        assert isinstance(mrp['resolved_at'], str) and mrp['resolved_at']
        # Sibling metadata keys are preserved (read-modify-write, not clobber).
        assert meta['retry_ledger'] == {'x': 1}
        # In-memory task metadata mirrors the persisted stamp.
        assert f.wf.task['metadata']['merge_retry_pending'] == mrp

    @pytest.mark.asyncio
    async def test_stamp_is_best_effort_and_does_not_raise_on_persist_failure(
        self, monkeypatch,
    ):
        f = _make(update_task_raises=True)
        monkeypatch.setattr('orchestrator.workflow._run', _fake_run(head='HEAD-SHA'))

        # scheduler.update_task raises — the helper must swallow it.
        await f.wf._stamp_merge_retry_pending()

        # In-memory metadata is still updated (persist is the last, best-effort step).
        assert f.wf.task['metadata']['merge_retry_pending']['branch_head'] == 'HEAD-SHA'

    @pytest.mark.asyncio
    async def test_clear_removes_key_from_persisted_and_in_memory(self, monkeypatch):
        f = _make(metadata={'merge_retry_pending': dict(_STAMP), 'retry_ledger': {'x': 1}})

        await f.wf._clear_merge_retry_pending()

        meta = _persisted_metadata(f.update_task)
        assert 'merge_retry_pending' not in meta
        # Clearing one key leaves siblings intact.
        assert meta['retry_ledger'] == {'x': 1}
        assert 'merge_retry_pending' not in f.wf.task['metadata']

    @pytest.mark.asyncio
    async def test_clear_is_best_effort_and_does_not_raise_on_persist_failure(self):
        f = _make(
            metadata={'merge_retry_pending': dict(_STAMP)},
            update_task_raises=True,
        )
        # Must not propagate the update_task failure.
        await f.wf._clear_merge_retry_pending()
        assert 'merge_retry_pending' not in f.wf.task['metadata']
