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
    config = MagicMock(spec_set=pydantic_spec(OrchestratorConfig))
    # Use a MagicMock stand-in for result — the dataclass stores it but never
    # awaits it during construction, so no real asyncio.Future is needed.
    result_mock = MagicMock()

    # Round-trip an explicit value
    req_explicit = MergeRequest(
        task_id='1',
        branch='1',
        worktree=Path('/tmp/wt'),
        pre_rebased=False,
        task_files=None,
        module_configs=[],
        config=config,
        result=result_mock,
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
        result=result_mock,
    )
    assert req_default.merge_first_enqueued_at is None


# ---------------------------------------------------------------------------
# Step 3/4 — _stamp_first_merge_enqueue: fresh stamp + persist
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_first_submission_stamps_and_persists():
    """First call stamps time.time(), persists under retry_ledger, returns a float.

    α typed ``RetryLedger.merge_first_enqueued_at`` as ``str | None`` (the
    runtime value is a float epoch); the persisted/in-memory ledger carries
    ``str(epoch)`` while the method's return value stays a float
    (``float(str(x)) == x`` round-trip).

    RED: _stamp_first_merge_enqueue still writes the legacy top-level float key.
    GREEN after step-12 migrates it onto metadata['retry_ledger'].
    """
    f = _make(metadata={}, get_task_metadata={})

    with patch('orchestrator.workflow.time') as mock_time:
        mock_time.time.return_value = 1000.0
        result = await f.wf._stamp_first_merge_enqueue()

    assert result == 1000.0
    assert isinstance(result, float)
    assert f.wf.task['metadata']['retry_ledger']['merge_first_enqueued_at'] == str(1000.0)

    persisted = _persisted_metadata(f.update_task)
    assert persisted['retry_ledger']['merge_first_enqueued_at'] == str(1000.0)
    f.update_task.assert_awaited_once()


# ---------------------------------------------------------------------------
# Step 5/6 — write-once: in-memory value already set → no re-stamp
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_write_once_resubmit_preserves_original():
    """In-memory retry_ledger value already set → return original, no re-stamp, no persist.

    Models a task re-dispatch after block/resolve where scheduler has reloaded
    metadata from backend so the in-memory dict already carries the original
    value (as ``str(epoch)``, the typed ledger's on-disk shape).

    RED: current implementation reads the legacy top-level key, not the ledger.
    GREEN after step-12 adds the retry_ledger fast-path guard.
    """
    f = _make(
        metadata={'retry_ledger': {'merge_first_enqueued_at': str(111.0)}},
        get_task_metadata={},
    )

    with patch('orchestrator.workflow.time') as mock_time:
        mock_time.time.return_value = 999.0
        result = await f.wf._stamp_first_merge_enqueue()

    assert result == 111.0
    assert isinstance(result, float)
    assert f.wf.task['metadata']['retry_ledger']['merge_first_enqueued_at'] == str(111.0)
    f.update_task.assert_not_awaited()


# ---------------------------------------------------------------------------
# Step 7/8 — backend value adopted when in-memory is stale
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_backend_value_adopted_not_restamped():
    """Backend has merge_first_enqueued_at under retry_ledger; stale in-memory copy does not.

    Slow path: in-memory metadata is empty but the backend already persisted a
    value (as ``str(epoch)``).  _merge_fresh_metadata should surface it and the
    helper should adopt it (returning a float) without re-stamping or re-persisting.

    RED: current implementation checks the legacy top-level backend key, not
    the nested ledger.
    GREEN after step-12 checks fresh.get('retry_ledger').
    """
    f = _make(
        metadata={},
        get_task_metadata={'retry_ledger': {'merge_first_enqueued_at': str(555.0)}},
    )

    with patch('orchestrator.workflow.time') as mock_time:
        mock_time.time.return_value = 999.0
        result = await f.wf._stamp_first_merge_enqueue()

    assert result == 555.0
    assert isinstance(result, float)
    assert f.wf.task['metadata']['retry_ledger']['merge_first_enqueued_at'] == str(555.0)
    f.update_task.assert_not_awaited()


# ---------------------------------------------------------------------------
# Legacy top-level fallback — in-flight task stamped by pre-migration code
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_legacy_top_level_first_enqueued_at_adopted():
    """A legacy top-level float (no retry_ledger at all) is adopted, not re-stamped.

    Models an in-flight task whose ``merge_first_enqueued_at`` was stamped by
    pre-migration code at the old top-level metadata key, with no
    ``retry_ledger`` blob present yet.  The fast-path legacy fallback must
    return the original epoch as a float, restore it under
    ``retry_ledger`` in memory (so subsequent calls take the ledger
    fast-path), and must not re-stamp or persist — the value is already
    durable on the backend under the old key.

    RED: current implementation has no retry_ledger concept at all — it reads
    this exact top-level key directly, so this test alone would not
    distinguish old from new behaviour; it starts failing once step-12's
    ledger-first fast-path is added without a legacy fallback (which would
    otherwise ignore the top-level key and re-stamp).
    """
    f = _make(metadata={'merge_first_enqueued_at': 222.0}, get_task_metadata={})

    with patch('orchestrator.workflow.time') as mock_time:
        mock_time.time.return_value = 999.0
        result = await f.wf._stamp_first_merge_enqueue()

    assert result == 222.0
    assert isinstance(result, float)
    assert f.wf.task['metadata']['retry_ledger']['merge_first_enqueued_at'] == str(222.0)
    f.update_task.assert_not_awaited()


# ---------------------------------------------------------------------------
# Step 9/10 — persistence failure is non-fatal
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_persistence_failure_is_non_fatal():
    """update_task raising RuntimeError must not propagate; in-memory stamp applied.

    Unlike the three anti-thrash guards (which now escalate to a human on
    persist failure), the first-enqueue stamp keeps the old best-effort
    behaviour: losing the epoch only perturbs merge-queue aging order
    (self-correcting), it is not a money-burning loop guard. No escalation.

    RED: current implementation persists the legacy top-level key.
    GREEN after step-12 persists metadata['retry_ledger'] and keeps the same
    non-fatal try/except (no _mark_blocked call).
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
    assert f.wf.task['metadata']['retry_ledger']['merge_first_enqueued_at'] == str(1000.0)
    f.mark_blocked.assert_not_awaited()


# ---------------------------------------------------------------------------
# Step 11/12 — _submit_to_merge_queue threads value onto MergeRequest
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_submit_to_merge_queue_threads_first_enqueued_at_onto_request(
    tmp_path: Path,
    monkeypatch,
):
    """_submit_to_merge_queue passes merge_first_enqueued_at=111.0 to MergeRequest.

    Metadata pre-seeded with retry_ledger.merge_first_enqueued_at=str(111.0)
    (write-once fast-path), so update_task should NOT be awaited (no re-stamp).

    RED: _submit_to_merge_queue constructs MergeRequest without the new field.
    GREEN after step-12 calls _stamp_first_merge_enqueue and threads the result in.
    """
    f = _make(
        metadata={'retry_ledger': {'merge_first_enqueued_at': str(111.0)}},
        get_task_metadata={'retry_ledger': {'merge_first_enqueued_at': str(111.0)}},
    )
    wf = f.wf
    wf.worktree = tmp_path / 'wt'
    wf.worktree.mkdir(parents=True, exist_ok=True)
    wf.merge_queue = MagicMock()
    wf.merge_inflight_registry = None  # skip attach branch
    wf.plan = {'files': []}
    wf._module_configs = []
    # task-1923: _submit_to_merge_queue now calls rebind_branch_to_head
    # (belt-and-braces rebind before enqueue); must be AsyncMock for await.
    wf.git_ops.rebind_branch_to_head = AsyncMock(return_value=True)

    captured: list[MergeRequest] = []

    async def fake_register_and_enqueue(
        queue, req: MergeRequest, event_store, registry, *, retention=None,
    ):
        # Intercept the function actually called on this path so the patch
        # stays valid even if register_and_enqueue_merge_request is ever
        # refactored to stop delegating to enqueue_merge_request internally.
        captured.append(req)
        req.result.set_result(MergeOutcome('blocked', reason='generic'))
        return True

    monkeypatch.setattr(
        'orchestrator.merge_queue.register_and_enqueue_merge_request',
        fake_register_and_enqueue,
    )

    await wf._submit_to_merge_queue('99', pre_rebased=False)

    assert len(captured) == 1
    assert captured[0].merge_first_enqueued_at == 111.0
    # Write-once fast-path: no persist call
    f.update_task.assert_not_awaited()
