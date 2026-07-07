"""Tests for Fix 2 — anti-thrash guard on repeated infra-issue resumes.

Mirrors the style of ``test_workflow_no_plan_cycle.py``: builds a minimal
``TaskWorkflow`` with mocks and drives ``_check_infra_resume_thrash``
directly to assert state transitions. Counters live inside the typed
``metadata.retry_ledger`` blob (:class:`shared.task_metadata.RetryLedger`).

The counter is keyed by iteration-log entry count (canonical "agent ran
real work" signal).  Steward fix-commits will reset the counter via
iteration-log growth — that is intentional and the counter resets to 1.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest
from _orch_helpers import pydantic_spec
from escalation.models import Escalation

from orchestrator.config import OrchestratorConfig
from orchestrator.workflow import TaskWorkflow, WorkflowOutcome


@dataclass
class _Fixture:
    wf: TaskWorkflow
    update_task: AsyncMock
    mark_blocked: AsyncMock
    iteration_log: list[dict]
    queue: MagicMock


def _esc(
    *,
    task_id: str = '99',
    category: str = 'infra_issue',
    status: str = 'resolved',
    level: int = 0,
    resolved_at: str | None = '2026-04-27T12:00:00Z',
) -> Escalation:
    return Escalation(
        id=f'esc-{task_id}-1',
        task_id=task_id,
        agent_role='implementer',
        severity='blocking',
        category=category,
        summary='infra blocker',
        detail='infra detail',
        status=status,
        level=level,
        resolved_at=resolved_at,
    )


def _make(
    *,
    task_id: str = '99',
    metadata: dict | None = None,
    backend_metadata: dict | None = None,
    iteration_log: list[dict] | None = None,
    resolved_l0s: list[Escalation] | None = None,
    update_task_raises: bool = False,
    get_task_raises: bool = False,
    max_consecutive_infra_resumes: int = 3,
    no_queue: bool = False,
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
    config.max_consecutive_infra_resumes = max_consecutive_infra_resumes

    if update_task_raises:
        update_task = AsyncMock(side_effect=RuntimeError('mcp down'))
    else:
        update_task = AsyncMock(return_value=True)

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
    git_ops.get_main_sha = AsyncMock(return_value='SHA-A')

    if no_queue:
        queue = None
    else:
        queue = MagicMock()
        queue.get_by_task = MagicMock(return_value=resolved_l0s or [])

    wf = TaskWorkflow(
        assignment=assignment,
        config=config,
        git_ops=git_ops,
        scheduler=scheduler,
        briefing=MagicMock(),
        mcp=MagicMock(),
        escalation_queue=queue,  # type: ignore[arg-type]
    )

    # Stub artifacts.read_iteration_log() — no real .task/ directory needed.
    iter_log = list(iteration_log or [])
    wf.artifacts = MagicMock()
    wf.artifacts.read_iteration_log = MagicMock(return_value=(iter_log, []))

    # Stub _mark_blocked — we only care about how _check_infra_resume_thrash
    # routes.
    mark_blocked = AsyncMock(return_value=WorkflowOutcome.BLOCKED)
    wf._mark_blocked = mark_blocked  # type: ignore[method-assign]

    return _Fixture(
        wf=wf,
        update_task=update_task,
        mark_blocked=mark_blocked,
        iteration_log=iter_log,
        queue=queue,  # type: ignore[arg-type]
    )


def _persisted_metadata(update_task: AsyncMock) -> dict:
    assert update_task.await_args is not None
    args, kwargs = update_task.await_args
    return kwargs.get('metadata') or args[1]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_counter_increments_on_consecutive_infra_resumes_no_iter_growth():
    """Same iteration-log size + infra_issue category → counter increments."""
    f = _make(
        metadata={
            'retry_ledger': {
                'consecutive_infra_resume_failures': 1,
                'last_infra_resume_iteration_count': 5,
            },
        },
        iteration_log=[{'iteration': i} for i in range(5)],  # still 5
        resolved_l0s=[_esc(category='infra_issue')],
    )

    outcome = await f.wf._check_infra_resume_thrash()

    # Below threshold → fall through (None).
    assert outcome is None
    md = _persisted_metadata(f.update_task)
    ledger = md['retry_ledger']
    assert ledger['consecutive_infra_resume_failures'] == 2
    assert ledger['last_infra_resume_iteration_count'] == 5
    f.mark_blocked.assert_not_awaited()


@pytest.mark.asyncio
async def test_counter_resets_to_one_when_iteration_log_grows():
    """Steward fix-commit advanced the iteration log → counter resets."""
    f = _make(
        metadata={
            'retry_ledger': {
                'consecutive_infra_resume_failures': 2,
                'last_infra_resume_iteration_count': 5,
            },
        },
        iteration_log=[{'iteration': i} for i in range(8)],  # grew 5 → 8
        resolved_l0s=[_esc(category='infra_issue')],
    )

    outcome = await f.wf._check_infra_resume_thrash()

    assert outcome is None
    md = _persisted_metadata(f.update_task)
    ledger = md['retry_ledger']
    assert ledger['consecutive_infra_resume_failures'] == 1, (
        f'Counter must reset to 1 on iteration-log growth: {md}'
    )
    assert ledger['last_infra_resume_iteration_count'] == 8


@pytest.mark.asyncio
async def test_counter_resets_to_zero_on_non_infra_category():
    """task_failure / design_concern / etc. → reset to zero."""
    f = _make(
        metadata={
            'retry_ledger': {
                'consecutive_infra_resume_failures': 2,
                'last_infra_resume_iteration_count': 5,
            },
        },
        iteration_log=[{'iteration': i} for i in range(5)],  # unchanged
        resolved_l0s=[_esc(category='task_failure')],
    )

    outcome = await f.wf._check_infra_resume_thrash()

    assert outcome is None
    md = _persisted_metadata(f.update_task)
    ledger = md['retry_ledger']
    assert ledger['consecutive_infra_resume_failures'] == 0
    assert ledger['last_infra_resume_iteration_count'] == 5


@pytest.mark.asyncio
async def test_counter_promotes_to_l1_at_threshold():
    """Counter reaches max_consecutive_infra_resumes → escalate_to_human=True."""
    f = _make(
        metadata={
            'retry_ledger': {
                'consecutive_infra_resume_failures': 2,  # one below default 3
                'last_infra_resume_iteration_count': 5,
            },
        },
        iteration_log=[{'iteration': i} for i in range(5)],  # unchanged
        resolved_l0s=[_esc(category='infra_issue')],
    )

    outcome = await f.wf._check_infra_resume_thrash()

    assert outcome == WorkflowOutcome.BLOCKED
    f.mark_blocked.assert_awaited_once()
    args, kwargs = f.mark_blocked.await_args
    assert kwargs.get('escalate_to_human') is True
    assert 'counter=3' in args[0] or 'thrash' in args[0].lower()


@pytest.mark.asyncio
async def test_threshold_is_configurable_below_default():
    """Lowering max_consecutive_infra_resumes promotes earlier."""
    f = _make(
        metadata={
            'retry_ledger': {
                'consecutive_infra_resume_failures': 1,  # one below threshold=2
                'last_infra_resume_iteration_count': 0,
            },
        },
        iteration_log=[],
        resolved_l0s=[_esc(category='infra_issue')],
        max_consecutive_infra_resumes=2,
    )

    outcome = await f.wf._check_infra_resume_thrash()

    assert outcome == WorkflowOutcome.BLOCKED
    _, kwargs = f.mark_blocked.await_args
    assert kwargs.get('escalate_to_human') is True


@pytest.mark.asyncio
async def test_no_queue_skips_classification_and_resets_counter():
    """Eval mode (no escalation queue) cannot classify the L0 → reset."""
    f = _make(
        metadata={
            'retry_ledger': {
                'consecutive_infra_resume_failures': 2,
                'last_infra_resume_iteration_count': 5,
            },
        },
        iteration_log=[{'iteration': i} for i in range(5)],
        no_queue=True,
    )

    outcome = await f.wf._check_infra_resume_thrash()

    assert outcome is None
    md = _persisted_metadata(f.update_task)
    assert md['retry_ledger']['consecutive_infra_resume_failures'] == 0


@pytest.mark.asyncio
async def test_persistence_failure_escalates_to_human():
    """If scheduler.update_task raises, the counter can't be trusted to have
    landed — escalate to a human immediately rather than logging and
    proceeding (a lost increment would let the infra-resume loop under-fire).

    This is true even below threshold: persist failure always escalates,
    regardless of the counter values themselves.
    """
    f = _make(
        metadata={
            'retry_ledger': {
                'consecutive_infra_resume_failures': 0,
                'last_infra_resume_iteration_count': 0,
            },
        },
        iteration_log=[{'iteration': 1}],
        resolved_l0s=[_esc(category='infra_issue')],
        update_task_raises=True,
    )

    outcome = await f.wf._check_infra_resume_thrash()

    assert outcome == WorkflowOutcome.BLOCKED
    f.mark_blocked.assert_awaited_once()
    _, kwargs = f.mark_blocked.await_args
    assert kwargs.get('escalate_to_human') is True


@pytest.mark.asyncio
async def test_corrupt_counter_metadata_treated_as_zero():
    """Non-int counter (e.g. legacy task) must not crash the helper.

    RetryLedger validation fails on the whole blob (not just the bad field),
    so the ledger resets to all-zeros rather than raising — same outcome as
    the old per-field ``int()`` parsing, reached via whole-ledger reset.
    """
    f = _make(
        metadata={
            'retry_ledger': {
                'consecutive_infra_resume_failures': 'three',  # corrupt
                'last_infra_resume_iteration_count': 5,
            },
        },
        iteration_log=[{'iteration': i} for i in range(5)],
        resolved_l0s=[_esc(category='infra_issue')],
    )

    outcome = await f.wf._check_infra_resume_thrash()

    assert outcome is None
    md = _persisted_metadata(f.update_task)
    assert md['retry_ledger']['consecutive_infra_resume_failures'] == 1


@pytest.mark.asyncio
async def test_picks_most_recent_resolved_l0_by_resolved_at():
    """Multiple resolved L0s: use the most recent one for category."""
    older = _esc(category='task_failure', resolved_at='2026-04-27T10:00:00Z')
    newer = _esc(category='infra_issue', resolved_at='2026-04-27T12:00:00Z')
    f = _make(
        metadata={
            'retry_ledger': {
                'consecutive_infra_resume_failures': 2,  # threshold=3 default
                'last_infra_resume_iteration_count': 5,
            },
        },
        iteration_log=[{'iteration': i} for i in range(5)],
        resolved_l0s=[older, newer],  # in any order
    )

    outcome = await f.wf._check_infra_resume_thrash()

    # Newer == infra_issue → counter increments to 3 → threshold hit.
    assert outcome == WorkflowOutcome.BLOCKED
    f.mark_blocked.assert_awaited_once()
    _, kwargs = f.mark_blocked.await_args
    assert kwargs.get('escalate_to_human') is True


@pytest.mark.asyncio
async def test_metadata_round_trips_via_scheduler_update():
    """Persisted metadata is the same dict layout the helper expects on
    the next invocation — defends against typo regressions on the metadata
    keys."""
    f = _make(
        metadata={},
        iteration_log=[{'iteration': i} for i in range(3)],
        resolved_l0s=[_esc(category='infra_issue')],
    )

    await f.wf._check_infra_resume_thrash()

    md = _persisted_metadata(f.update_task)
    ledger = md['retry_ledger']
    # Both keys must be present and machine-readable on the next call.
    assert isinstance(ledger['consecutive_infra_resume_failures'], int)
    assert isinstance(ledger['last_infra_resume_iteration_count'], int)
    assert ledger['consecutive_infra_resume_failures'] == 1
    assert ledger['last_infra_resume_iteration_count'] == 3


@pytest.mark.asyncio
async def test_persists_memory_hints_from_fresh_backend_metadata():
    """memory_hints added by Stage-2 reconciliation after load survive the write.

    The in-memory copy (self.task['metadata']) does NOT have memory_hints;
    the backend's current metadata (scheduler.get_task) DOES.  After
    _check_infra_resume_thrash the persisted dict must contain BOTH the
    incremented counter AND memory_hints from the fresh backend read.
    """
    in_memory_md = {
        'retry_ledger': {
            'consecutive_infra_resume_failures': 1,
            'last_infra_resume_iteration_count': 5,
        },
    }
    backend_md = {
        'retry_ledger': {
            'consecutive_infra_resume_failures': 1,
            'last_infra_resume_iteration_count': 5,
        },
        'memory_hints': {'entities': ['E1'], 'queries': ['q1']},
    }
    f = _make(
        metadata=in_memory_md,
        backend_metadata=backend_md,
        iteration_log=[{'iteration': i} for i in range(5)],  # unchanged → infra_issue increments
        resolved_l0s=[_esc(category='infra_issue')],
    )

    outcome = await f.wf._check_infra_resume_thrash()

    # Below threshold (2 < 3) → fall through.
    assert outcome is None
    md = _persisted_metadata(f.update_task)
    assert md['retry_ledger']['consecutive_infra_resume_failures'] == 2, (
        f'Counter must have incremented to 2; got {md}'
    )
    assert md.get('memory_hints') == {'entities': ['E1'], 'queries': ['q1']}, (
        f'memory_hints from backend read must survive the write; got {md}'
    )


@pytest.mark.asyncio
async def test_get_task_failure_falls_back_to_in_memory_metadata_and_warns(caplog: pytest.LogCaptureFixture) -> None:
    """If scheduler.get_task raises, fall back to in-memory metadata.

    The counter must still advance (no exception escapes), and a WARNING must
    be emitted mentioning the failure so operators can see the degraded path.
    """
    import logging

    f = _make(
        metadata={
            'retry_ledger': {
                'consecutive_infra_resume_failures': 0,
                'last_infra_resume_iteration_count': 0,
            },
            'memory_hints': {'entities': ['E1']},
        },
        get_task_raises=True,
        iteration_log=[{'iteration': 1}],
        resolved_l0s=[_esc(category='infra_issue')],
    )

    with caplog.at_level(logging.WARNING):
        outcome = await f.wf._check_infra_resume_thrash()

    # Must not raise; must fall through (1 < 3 threshold).
    assert outcome is None
    # update_task must still be called once — persistence happens on the fallback path.
    f.update_task.assert_awaited_once()
    md = _persisted_metadata(f.update_task)
    assert md['retry_ledger']['consecutive_infra_resume_failures'] == 1, (
        f'Counter must advance on fallback path; got {md}'
    )
    # In-memory memory_hints survive because we fell back to the in-memory copy.
    assert md.get('memory_hints') == {'entities': ['E1']}, (
        f'In-memory memory_hints must survive on fallback path; got {md}'
    )
    # A WARNING with the specific refresh-failure message must be logged.
    warning_texts = [r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING]
    assert any(
        'failed to refresh metadata before infra-resume thrash' in t
        for t in warning_texts
    ), (
        f'Expected warning about get_task refresh failure; got: {warning_texts}'
    )
