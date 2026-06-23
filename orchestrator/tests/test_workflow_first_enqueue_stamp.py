"""Tests for α substrate — write-once first-submission timestamp merge_first_enqueued_at.

Covers:
  1. MergeRequest dataclass carries the new carrier field (step-1/2).
  2. _stamp_first_merge_enqueue stamps and persists on first call (step-3/4).
  3. Write-once: resubmit with in-memory value already set → no re-stamp (step-5/6).
  4. Write-once: backend has value, in-memory is stale → adopt backend, no re-stamp (step-7/8).
  5. Persistence failure is non-fatal (step-9/10).
  6. _submit_to_merge_queue threads merge_first_enqueued_at onto the MergeRequest (step-11/12).
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from _orch_helpers import pydantic_spec

from orchestrator.config import OrchestratorConfig
from orchestrator.merge_queue import MergeOutcome, MergeRequest
from orchestrator.workflow import TaskWorkflow, WorkflowOutcome

# ---------------------------------------------------------------------------
# Shared fixture builder (mirrors test_workflow_merge_thrash._make)
# ---------------------------------------------------------------------------


@dataclass
class _Fixture:
    wf: TaskWorkflow
    update_task: AsyncMock
    get_task: AsyncMock
    mark_blocked: AsyncMock


def _make(
    *,
    task_id: str = '99',
    metadata: dict | None = None,
    update_task_raises: bool = False,
    get_task_metadata: dict | None = None,
) -> _Fixture:
    assignment = MagicMock()
    assignment.task_id = task_id
    assignment.task = {
        'id': task_id, 'title': 'T', 'description': 'd',
        'metadata': metadata if metadata is not None else {},
    }
    assignment.modules = ['mod_a']

    _spec = pydantic_spec(OrchestratorConfig)
    config = MagicMock(spec_set=_spec)
    config.fused_memory.project_id = 'dark_factory'
    config.fused_memory.url = 'http://localhost:8002'
    config.lock_depth = 2
    config.steward_completion_timeout = 300.0
    config.project_root = Path('/tmp/non-existent-for-test')
    config.max_consecutive_merge_thrash = 2

    if update_task_raises:
        update_task = AsyncMock(side_effect=RuntimeError('mcp down'))
    else:
        update_task = AsyncMock(return_value=True)

    # scheduler.get_task returns a task dict with the given metadata
    _backend_meta = get_task_metadata if get_task_metadata is not None else {}
    get_task = AsyncMock(return_value={'id': task_id, 'metadata': _backend_meta})

    scheduler = MagicMock()
    scheduler.update_task = update_task
    scheduler.get_task = get_task
    scheduler.set_task_status = AsyncMock()
    scheduler.get_status = AsyncMock(return_value='in-progress')

    git_ops = MagicMock()
    git_ops.get_main_sha = AsyncMock(return_value='SHA-A')

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

    mark_blocked = AsyncMock(return_value=WorkflowOutcome.BLOCKED)
    wf._mark_blocked = mark_blocked  # type: ignore[method-assign]

    return _Fixture(wf=wf, update_task=update_task, get_task=get_task, mark_blocked=mark_blocked)


def _persisted_metadata(update_task: AsyncMock) -> dict:
    assert update_task.await_args is not None
    args, kwargs = update_task.await_args
    return kwargs.get('metadata') or args[1]


# ---------------------------------------------------------------------------
# Step 1/2 — MergeRequest carrier field
# ---------------------------------------------------------------------------


def test_merge_request_carries_first_enqueued_at_field():
    """MergeRequest has merge_first_enqueued_at field; defaults to None.

    RED: MergeRequest has no such field today (TypeError on construction).
    GREEN after step-2 adds the carrier field.
    """
    config = MagicMock()
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
    future: asyncio.Future[MergeOutcome] = loop.create_future()

    # Round-trip an explicit value
    req_explicit = MergeRequest(
        task_id='1',
        branch='1',
        worktree=Path('/tmp/wt'),
        pre_rebased=False,
        task_files=None,
        module_configs=[],
        config=config,
        result=future,
        merge_first_enqueued_at=123.0,
    )
    assert req_explicit.merge_first_enqueued_at == 123.0

    # Default is None when not supplied
    req_default = MergeRequest(
        task_id='2',
        branch='2',
        worktree=Path('/tmp/wt'),
        pre_rebased=False,
        task_files=None,
        module_configs=[],
        config=config,
        result=future,
    )
    assert req_default.merge_first_enqueued_at is None


# ---------------------------------------------------------------------------
# Step 3/4 — _stamp_first_merge_enqueue: fresh stamp + persist
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_first_submission_stamps_and_persists():
    """First call stamps time.time() into metadata, persists, and returns the value.

    RED: _stamp_first_merge_enqueue does not exist yet.
    GREEN after step-4 adds the minimal implementation.
    """
    f = _make(metadata={}, get_task_metadata={})

    with patch('orchestrator.workflow.time') as mock_time:
        mock_time.time.return_value = 1000.0
        result = await f.wf._stamp_first_merge_enqueue()

    assert result == 1000.0
    assert f.wf.task['metadata']['merge_first_enqueued_at'] == 1000.0

    persisted = _persisted_metadata(f.update_task)
    assert persisted['merge_first_enqueued_at'] == 1000.0
    f.update_task.assert_awaited_once()


# ---------------------------------------------------------------------------
# Step 5/6 — write-once: in-memory value already set → no re-stamp
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_write_once_resubmit_preserves_original():
    """In-memory value already set → return original, no re-stamp, no persist.

    Models a task re-dispatch after block/resolve where scheduler has reloaded
    metadata from backend so the in-memory dict already carries the original value.

    RED: minimal step-4 helper unconditionally re-stamps and persists.
    GREEN after step-6 adds the fast-path guard.
    """
    f = _make(metadata={'merge_first_enqueued_at': 111.0}, get_task_metadata={})

    with patch('orchestrator.workflow.time') as mock_time:
        mock_time.time.return_value = 999.0
        result = await f.wf._stamp_first_merge_enqueue()

    assert result == 111.0
    assert f.wf.task['metadata']['merge_first_enqueued_at'] == 111.0
    f.update_task.assert_not_awaited()


# ---------------------------------------------------------------------------
# Step 7/8 — backend value adopted when in-memory is stale
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_backend_value_adopted_not_restamped():
    """Backend has merge_first_enqueued_at; stale in-memory copy does not.

    Slow path: in-memory metadata is empty but the backend already persisted a
    value.  _merge_fresh_metadata should surface it and the helper should adopt
    it without re-stamping or re-persisting.

    RED: step-6 helper, on empty in-memory metadata, ignores backend and stamps.
    GREEN after step-8 inserts the _merge_fresh_metadata backend-read.
    """
    f = _make(metadata={}, get_task_metadata={'merge_first_enqueued_at': 555.0})

    with patch('orchestrator.workflow.time') as mock_time:
        mock_time.time.return_value = 999.0
        result = await f.wf._stamp_first_merge_enqueue()

    assert result == 555.0
    assert f.wf.task['metadata']['merge_first_enqueued_at'] == 555.0
    f.update_task.assert_not_awaited()


# ---------------------------------------------------------------------------
# Step 9/10 — persistence failure is non-fatal
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_persistence_failure_is_non_fatal():
    """update_task raising RuntimeError must not propagate; in-memory stamp applied.

    RED: step-8 helper lets the RuntimeError propagate.
    GREEN after step-10 wraps update_task in a logged try/except.
    """
    f = _make(
        metadata={},
        get_task_metadata={},
        update_task_raises=True,
    )

    with patch('orchestrator.workflow.time') as mock_time:
        mock_time.time.return_value = 1000.0
        # Must NOT raise even though update_task raises RuntimeError
        result = await f.wf._stamp_first_merge_enqueue()

    assert result == 1000.0
    assert f.wf.task['metadata']['merge_first_enqueued_at'] == 1000.0


# ---------------------------------------------------------------------------
# Step 11/12 — _submit_to_merge_queue threads value onto MergeRequest
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_submit_to_merge_queue_threads_first_enqueued_at_onto_request(
    tmp_path: Path,
    monkeypatch,
):
    """_submit_to_merge_queue passes merge_first_enqueued_at=111.0 to MergeRequest.

    Metadata pre-seeded with merge_first_enqueued_at=111.0 (write-once fast-path),
    so update_task should NOT be awaited (no re-stamp).

    RED: _submit_to_merge_queue constructs MergeRequest without the new field.
    GREEN after step-12 calls _stamp_first_merge_enqueue and threads the result in.
    """
    f = _make(
        metadata={'merge_first_enqueued_at': 111.0},
        get_task_metadata={'merge_first_enqueued_at': 111.0},
    )
    wf = f.wf
    wf.worktree = tmp_path / 'wt'
    wf.worktree.mkdir(parents=True, exist_ok=True)
    wf.merge_queue = MagicMock()
    wf.merge_inflight_registry = None  # skip attach branch
    wf.plan = {'files': []}
    wf._module_configs = []

    captured: list[MergeRequest] = []

    async def fake_enqueue(queue, req: MergeRequest, event_store, **_kwargs):
        captured.append(req)
        req.result.set_result(MergeOutcome('blocked', reason='generic'))

    monkeypatch.setattr(
        'orchestrator.merge_queue.enqueue_merge_request', fake_enqueue,
    )

    await wf._submit_to_merge_queue('99', pre_rebased=False)

    assert len(captured) == 1
    assert captured[0].merge_first_enqueued_at == 111.0
    # Write-once fast-path: no persist call
    f.update_task.assert_not_awaited()
