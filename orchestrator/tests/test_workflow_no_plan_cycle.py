"""Tests for ``TaskWorkflow._handle_no_plan_failure`` (Fix C).

The helper increments ``consecutive_no_plan_failures`` keyed by
``last_no_plan_main_sha``, both carried inside the typed
``metadata.retry_ledger`` blob (:class:`shared.task_metadata.RetryLedger`).
When the counter hits ``>= 2`` with the same main SHA, the workflow
short-circuits to BLOCKED + L1 instead of letting the steward attempt
resolution again. A ``retry_ledger`` that fails RetryLedger validation
(corrupt/legacy shape) is treated as a fresh all-zero ledger rather than
crashing.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest
from _orch_helpers import pydantic_spec

from orchestrator.config import OrchestratorConfig
from orchestrator.workflow import TaskWorkflow, WorkflowOutcome


@dataclass
class _Fixture:
    wf: TaskWorkflow
    update_task: AsyncMock
    get_main_sha: AsyncMock
    mark_blocked: AsyncMock


def _make(
    *,
    project_root: Path,
    task_id: str = '99',
    main_sha: str = 'mainSHA-1',
    metadata: dict | None = None,
    backend_metadata: dict | None = None,
    update_task_raises: bool = False,
    get_main_sha_raises: bool = False,
    get_task_raises: bool = False,
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
    config.project_root = project_root

    if update_task_raises:
        update_task = AsyncMock(side_effect=RuntimeError('mcp down'))
    else:
        update_task = AsyncMock(return_value=True)
    if get_main_sha_raises:
        get_main_sha = AsyncMock(side_effect=RuntimeError('git down'))
    else:
        get_main_sha = AsyncMock(return_value=main_sha)

    # backend_metadata: what get_task returns (defaults to metadata if not set).
    _backend_md = backend_metadata if backend_metadata is not None else (metadata or {})

    scheduler = MagicMock()
    scheduler.update_task = update_task
    scheduler.set_task_status = AsyncMock()
    scheduler.get_status = AsyncMock(return_value='in-progress')
    if get_task_raises:
        scheduler.get_task = AsyncMock(side_effect=RuntimeError('mcp down'))
    else:
        scheduler.get_task = AsyncMock(return_value={'id': task_id, 'metadata': _backend_md})

    git_ops = MagicMock()
    git_ops.get_main_sha = get_main_sha

    wf = TaskWorkflow(
        assignment=assignment,
        config=config,
        git_ops=git_ops,
        scheduler=scheduler,
        briefing=MagicMock(),
        mcp=MagicMock(),
    )

    # Stub _mark_blocked — we only assert how _handle_no_plan_failure routes.
    mark_blocked = AsyncMock(return_value=WorkflowOutcome.BLOCKED)
    wf._mark_blocked = mark_blocked  # type: ignore[method-assign]

    return _Fixture(
        wf=wf,
        update_task=update_task,
        get_main_sha=get_main_sha,
        mark_blocked=mark_blocked,
    )


def _persisted_metadata(update_task: AsyncMock) -> dict:
    """Return the metadata dict passed to the most recent update_task call."""
    assert update_task.await_args is not None
    args, kwargs = update_task.await_args
    return kwargs.get('metadata') or args[1]


@pytest.mark.asyncio
async def test_first_failure_increments_counter_to_one_and_routes_normally(
    tmp_path: Path,
):
    f = _make(project_root=tmp_path, main_sha='SHA-A')

    outcome = await f.wf._handle_no_plan_failure(
        'no plan.json produced', detail='boom',
    )

    assert outcome == WorkflowOutcome.BLOCKED
    # _mark_blocked called WITHOUT escalate_to_human (normal steward path).
    f.mark_blocked.assert_awaited_once()
    _, kwargs = f.mark_blocked.await_args
    assert kwargs.get('escalate_to_human') is not True
    # Counter persisted to 1 with current main SHA, under retry_ledger.
    f.update_task.assert_awaited_once()
    metadata = _persisted_metadata(f.update_task)
    ledger = metadata['retry_ledger']
    assert ledger['consecutive_no_plan_failures'] == 1
    assert ledger['last_no_plan_main_sha'] == 'SHA-A'


@pytest.mark.asyncio
async def test_second_failure_same_main_sha_escalates_to_human(tmp_path: Path):
    f = _make(
        project_root=tmp_path,
        main_sha='SHA-A',
        metadata={
            'retry_ledger': {
                'last_no_plan_main_sha': 'SHA-A',
                'consecutive_no_plan_failures': 1,
            },
        },
    )

    outcome = await f.wf._handle_no_plan_failure(
        'plan missing "steps"', detail='dump',
    )

    assert outcome == WorkflowOutcome.BLOCKED
    # _mark_blocked called with escalate_to_human=True (Fix C path).
    f.mark_blocked.assert_awaited_once()
    args, kwargs = f.mark_blocked.await_args
    assert kwargs.get('escalate_to_human') is True
    assert 'counter=2' in args[0] or 'Repeated no-plan' in args[0]
    metadata = _persisted_metadata(f.update_task)
    assert metadata['retry_ledger']['consecutive_no_plan_failures'] == 2


@pytest.mark.asyncio
async def test_main_sha_changed_resets_counter_to_one(tmp_path: Path):
    """If main advanced between failures, the counter resets — the missing
    premise may now exist."""
    f = _make(
        project_root=tmp_path,
        main_sha='SHA-NEW',
        metadata={
            'retry_ledger': {
                'last_no_plan_main_sha': 'SHA-OLD',
                'consecutive_no_plan_failures': 5,
            },
        },
    )

    outcome = await f.wf._handle_no_plan_failure(
        'no plan.json produced', detail='boom',
    )

    assert outcome == WorkflowOutcome.BLOCKED
    _, kwargs = f.mark_blocked.await_args
    assert kwargs.get('escalate_to_human') is not True
    metadata = _persisted_metadata(f.update_task)
    ledger = metadata['retry_ledger']
    assert ledger['consecutive_no_plan_failures'] == 1
    assert ledger['last_no_plan_main_sha'] == 'SHA-NEW'


@pytest.mark.asyncio
async def test_third_consecutive_failure_still_escalates(tmp_path: Path):
    """Counter ≥ 2 always routes to escalate_to_human, not just exactly 2."""
    f = _make(
        project_root=tmp_path,
        main_sha='SHA-A',
        metadata={
            'retry_ledger': {
                'last_no_plan_main_sha': 'SHA-A',
                'consecutive_no_plan_failures': 2,
            },
        },
    )

    await f.wf._handle_no_plan_failure('still no plan', detail='')

    _, kwargs = f.mark_blocked.await_args
    assert kwargs.get('escalate_to_human') is True


@pytest.mark.asyncio
async def test_persistence_failure_escalates_to_human(tmp_path: Path):
    """If scheduler.update_task raises, the counter can't be trusted to have
    landed — escalate to a human immediately rather than logging and
    proceeding (a lost increment would let the no-plan loop under-fire).

    This is true even on what would otherwise be a first, non-escalating
    failure (main_sha='SHA-A' with no prior ledger): persist failure always
    escalates, regardless of the counter values themselves.
    """
    f = _make(
        project_root=tmp_path, main_sha='SHA-A',
        update_task_raises=True,
    )

    outcome = await f.wf._handle_no_plan_failure('no plan', detail='')

    assert outcome == WorkflowOutcome.BLOCKED
    f.mark_blocked.assert_awaited_once()
    _, kwargs = f.mark_blocked.await_args
    assert kwargs.get('escalate_to_human') is True


@pytest.mark.asyncio
async def test_get_main_sha_failure_treats_as_unknown_main(tmp_path: Path):
    """If we can't read main SHA, counter resets to 1 (normal path)."""
    f = _make(
        project_root=tmp_path,
        metadata={
            'retry_ledger': {
                'last_no_plan_main_sha': 'SHA-A',
                'consecutive_no_plan_failures': 5,
            },
        },
        get_main_sha_raises=True,
    )

    outcome = await f.wf._handle_no_plan_failure('no plan', detail='')

    assert outcome == WorkflowOutcome.BLOCKED
    _, kwargs = f.mark_blocked.await_args
    assert kwargs.get('escalate_to_human') is not True


@pytest.mark.asyncio
async def test_total_counter_escalates_at_3(tmp_path: Path):
    """Total counter (no SHA reset) escalates after 3 failures across
    different main SHAs.

    Backstops the per-SHA counter for the busy-orchestrator case where
    main keeps moving between attempts, so the per-SHA counter never
    reaches 2.  Task 917 burned 16 Opus calls because of this — the
    total counter caps it at 3.
    """
    # Failure 1 on SHA-A: counter resets to 1, total=1 → normal path.
    f = _make(project_root=tmp_path, main_sha='SHA-A')
    await f.wf._handle_no_plan_failure('no plan', detail='')
    _, kwargs = f.mark_blocked.await_args
    assert kwargs.get('escalate_to_human') is not True
    metadata = _persisted_metadata(f.update_task)
    assert metadata['retry_ledger']['consecutive_no_plan_failures'] == 1
    assert metadata['retry_ledger']['total_no_plan_failures'] == 1

    # Failure 2 on SHA-B (main moved): counter resets to 1, total=2 → still normal.
    f = _make(
        project_root=tmp_path, main_sha='SHA-B',
        metadata={
            'retry_ledger': {
                'last_no_plan_main_sha': 'SHA-A',
                'consecutive_no_plan_failures': 1,
                'total_no_plan_failures': 1,
            },
        },
    )
    await f.wf._handle_no_plan_failure('no plan', detail='')
    _, kwargs = f.mark_blocked.await_args
    assert kwargs.get('escalate_to_human') is not True
    metadata = _persisted_metadata(f.update_task)
    assert metadata['retry_ledger']['consecutive_no_plan_failures'] == 1
    assert metadata['retry_ledger']['total_no_plan_failures'] == 2

    # Failure 3 on SHA-C: counter resets to 1, total=3 → escalates via total.
    f = _make(
        project_root=tmp_path, main_sha='SHA-C',
        metadata={
            'retry_ledger': {
                'last_no_plan_main_sha': 'SHA-B',
                'consecutive_no_plan_failures': 1,
                'total_no_plan_failures': 2,
            },
        },
    )
    await f.wf._handle_no_plan_failure('no plan', detail='')
    args, kwargs = f.mark_blocked.await_args
    assert kwargs.get('escalate_to_human') is True
    # The reason string must mention the total counter and which trigger fired.
    assert 'total=3' in args[0]
    assert 'total counter' in args[0]
    metadata = _persisted_metadata(f.update_task)
    assert metadata['retry_ledger']['consecutive_no_plan_failures'] == 1
    assert metadata['retry_ledger']['total_no_plan_failures'] == 3


@pytest.mark.asyncio
async def test_total_counter_increments_on_same_sha_path_too(tmp_path: Path):
    """The total counter must increment regardless of which counter fires
    the escalate; it tracks every no-plan failure unconditionally."""
    f = _make(
        project_root=tmp_path, main_sha='SHA-A',
        metadata={
            'retry_ledger': {
                'last_no_plan_main_sha': 'SHA-A',
                'consecutive_no_plan_failures': 1,
                'total_no_plan_failures': 1,
            },
        },
    )
    await f.wf._handle_no_plan_failure('no plan', detail='')
    args, kwargs = f.mark_blocked.await_args
    # Same-SHA counter trigger fires (counter=2), but total still increments.
    assert kwargs.get('escalate_to_human') is True
    metadata = _persisted_metadata(f.update_task)
    assert metadata['retry_ledger']['consecutive_no_plan_failures'] == 2
    assert metadata['retry_ledger']['total_no_plan_failures'] == 2
    assert 'same-SHA counter' in args[0]


@pytest.mark.asyncio
async def test_corrupt_total_metadata_treated_as_zero(tmp_path: Path):
    """A non-int total_no_plan_failures must not crash the helper.

    RetryLedger validation fails on the whole blob (not just the bad field),
    so the ledger resets to all-zeros rather than raising — same outcome as
    the old per-field ``int()`` parsing, reached via whole-ledger reset.
    """
    f = _make(
        project_root=tmp_path, main_sha='SHA-A',
        metadata={
            'retry_ledger': {
                'last_no_plan_main_sha': 'SHA-A',
                'consecutive_no_plan_failures': 1,
                'total_no_plan_failures': 'lots',  # corrupt -> whole ledger resets
            },
        },
    )
    await f.wf._handle_no_plan_failure('no plan', detail='')
    metadata = _persisted_metadata(f.update_task)
    assert metadata['retry_ledger']['total_no_plan_failures'] == 1


@pytest.mark.asyncio
async def test_corrupt_counter_metadata_treated_as_zero(tmp_path: Path):
    """If metadata has a non-int counter (e.g. legacy task), we don't crash."""
    f = _make(
        project_root=tmp_path,
        main_sha='SHA-A',
        metadata={
            'retry_ledger': {
                'last_no_plan_main_sha': 'SHA-A',
                'consecutive_no_plan_failures': 'one',  # corrupt -> whole ledger resets
            },
        },
    )

    outcome = await f.wf._handle_no_plan_failure('no plan', detail='')

    assert outcome == WorkflowOutcome.BLOCKED
    _, kwargs = f.mark_blocked.await_args
    assert kwargs.get('escalate_to_human') is not True
    metadata = _persisted_metadata(f.update_task)
    assert metadata['retry_ledger']['consecutive_no_plan_failures'] == 1


@pytest.mark.asyncio
async def test_non_dict_retry_ledger_treated_as_fresh_ledger(tmp_path: Path):
    """A non-dict ``retry_ledger`` blob must not crash via ``TypeError``.

    ``RetryLedger(**raw)`` raises ``TypeError`` (not ``ValidationError``) if
    ``raw`` isn't a mapping — must reset to a fresh all-zero ledger instead
    of propagating the exception.
    """
    f = _make(
        project_root=tmp_path,
        main_sha='SHA-A',
        metadata={'retry_ledger': 'garbage'},
    )

    outcome = await f.wf._handle_no_plan_failure('no plan', detail='')

    assert outcome == WorkflowOutcome.BLOCKED
    _, kwargs = f.mark_blocked.await_args
    assert kwargs.get('escalate_to_human') is not True
    metadata = _persisted_metadata(f.update_task)
    assert metadata['retry_ledger']['consecutive_no_plan_failures'] == 1
    assert metadata['retry_ledger']['total_no_plan_failures'] == 1


@pytest.mark.asyncio
async def test_persists_memory_hints_from_fresh_backend_metadata(tmp_path: Path):
    """memory_hints added by Stage-2 reconciliation after load survive the write.

    The in-memory copy (self.task['metadata']) does NOT have memory_hints;
    the backend's current metadata (scheduler.get_task) DOES.  After
    _handle_no_plan_failure the persisted dict must contain BOTH the
    incremented counter AND memory_hints from the fresh backend read.
    """
    in_memory_md: dict = {}  # no memory_hints in-memory
    backend_md = {
        'memory_hints': {'entities': ['E1'], 'queries': ['q1']},
    }
    f = _make(
        project_root=tmp_path,
        main_sha='SHA-A',
        metadata=in_memory_md,
        backend_metadata=backend_md,
    )

    outcome = await f.wf._handle_no_plan_failure('no plan', detail='')

    assert outcome == WorkflowOutcome.BLOCKED
    metadata = _persisted_metadata(f.update_task)
    assert metadata['retry_ledger']['consecutive_no_plan_failures'] == 1, (
        f'Counter must have incremented to 1; got {metadata}'
    )
    assert metadata.get('memory_hints') == {'entities': ['E1'], 'queries': ['q1']}, (
        f'memory_hints from backend read must survive the write; got {metadata}'
    )


@pytest.mark.asyncio
async def test_get_task_failure_falls_back_to_in_memory_metadata_and_warns(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """If scheduler.get_task raises, fall back to in-memory metadata.

    The counter must still advance (no exception escapes), in-memory memory_hints
    are preserved on the fallback path, and a WARNING is emitted mentioning the
    failure so operators can see the degraded path.
    """
    import logging

    f = _make(
        project_root=tmp_path,
        main_sha='SHA-A',
        metadata={
            'memory_hints': {'entities': ['E1']},
        },
        get_task_raises=True,
    )

    with caplog.at_level(logging.WARNING):
        outcome = await f.wf._handle_no_plan_failure('no plan', detail='')

    # Must not raise; normal blocked outcome.
    assert outcome == WorkflowOutcome.BLOCKED
    # update_task must still be called once — persistence happens on the fallback path.
    f.update_task.assert_awaited_once()
    metadata = _persisted_metadata(f.update_task)
    assert metadata['retry_ledger']['consecutive_no_plan_failures'] == 1, (
        f'Counter must advance on fallback path; got {metadata}'
    )
    # In-memory memory_hints survive because we fell back to the in-memory copy.
    assert metadata.get('memory_hints') == {'entities': ['E1']}, (
        f'In-memory memory_hints must survive on fallback path; got {metadata}'
    )
    # A WARNING with the specific refresh-failure message must be logged.
    warning_texts = [r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING]
    assert any(
        'failed to refresh metadata before no-plan counter' in t
        for t in warning_texts
    ), (
        f'Expected warning about get_task refresh failure; got: {warning_texts}'
    )


@pytest.mark.asyncio
async def test_backend_wins_on_collision(tmp_path: Path):
    """When the same key exists in both in-memory and backend metadata,
    the backend value wins.

    This pins the merge direction: {**in_memory, **backend}.  A one-character
    inversion to {**backend, **in_memory} would silently flip this contract.
    """
    f = _make(
        project_root=tmp_path,
        main_sha='SHA-A',
        metadata={'memory_hints': {'entities': ['OLD']}},
        backend_metadata={'memory_hints': {'entities': ['NEW']}},
    )

    await f.wf._handle_no_plan_failure('no plan', detail='')

    persisted = _persisted_metadata(f.update_task)
    assert persisted.get('memory_hints') == {'entities': ['NEW']}, (
        f'Backend value must win on collision; got {persisted.get("memory_hints")!r}'
    )
