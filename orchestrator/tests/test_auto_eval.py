"""Tests for the harness auto-eval hook (``Harness._maybe_auto_eval``).

When the optimistic path (Lever B revalidation skip or Lever C SIMPLE_TASK)
blocks at a phase the auto-eval cares about, the hook renames the original
branch + worktree and submits a sibling task with ``force_full_path=True``.
"""

from __future__ import annotations

import json as _json
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest
from _orch_helpers import _init_harness_state_for_test, assert_update_wire_mode, pydantic_spec

from orchestrator.config import OrchestratorConfig
from orchestrator.event_store import EventType
from orchestrator.harness import Harness, TaskReport
from orchestrator.scheduler import Scheduler, TaskAssignment
from orchestrator.task_status import ACTIVE_TASK_STATUSES
from orchestrator.workflow import WorkflowOutcome


@dataclass
class _Fixture:
    harness: Harness
    assignment: TaskAssignment
    git_ops_rename: AsyncMock
    dispatch_tool: AsyncMock
    update_task: AsyncMock
    event_emit: MagicMock
    submit_calls: list[tuple[str, dict]]


def _make(
    *,
    project_root: Path,
    optimistic_path: str | None = 'simple_task',
    auto_eval_redo: bool = False,
    block_phase: str = 'verify',
    outcome: WorkflowOutcome = WorkflowOutcome.BLOCKED,
    auto_eval_enabled: bool = True,
    auto_eval_phases: set[str] | None = None,
    redo_budget_usd: float = 50.0,
    submit_returns: dict | None = None,
    rename_raises: Exception | None = None,
    worktree_exists: bool = True,
    seen_redo_costs: list[float] | None = None,
) -> _Fixture:
    if auto_eval_phases is None:
        auto_eval_phases = {'plan', 'execute', 'verify', 'review'}

    config = MagicMock(spec_set=pydantic_spec(OrchestratorConfig))
    config.project_root = project_root
    config.auto_eval_enabled = auto_eval_enabled
    config.auto_eval_redo_budget_usd = redo_budget_usd
    config.auto_eval_phases = auto_eval_phases

    harness = Harness.__new__(Harness)  # bypass full Harness init
    _init_harness_state_for_test(harness)  # task 1449: initialise digest counters
    harness.config = config
    harness.event_store = MagicMock()
    harness.event_store.emit = MagicMock()
    harness.cost_store = None
    harness._auto_eval_redo_task_ids = set()

    # git_ops with rename_worktree as AsyncMock + a synthetic worktree_base.
    project_root.mkdir(parents=True, exist_ok=True)
    worktree_base = project_root / '.worktrees'
    worktree_base.mkdir(parents=True, exist_ok=True)
    git_ops = MagicMock()
    git_ops.worktree_base = worktree_base
    git_ops_rename = AsyncMock(side_effect=rename_raises)
    git_ops.rename_worktree = git_ops_rename
    harness.git_ops = git_ops

    # Pre-create the worktree directory so the hook actually invokes rename.
    if worktree_exists:
        (worktree_base / 'orig-task').mkdir(parents=True, exist_ok=True)

    if submit_returns is None:
        submit_returns = {'task_id': 'redo-1', 'status': 'deferred'}

    dispatch_tool = AsyncMock()

    # First call submit_task → submit_returns
    # Second call set_task_status → OK
    async def _dispatch(name, args, *, timeout=15):
        return await dispatch_tool(name, args, timeout=timeout)

    dispatch_tool.return_value = submit_returns
    update_task = AsyncMock(return_value=True)

    scheduler = MagicMock()
    scheduler.dispatch_tool = _dispatch
    scheduler.update_task = update_task
    # Default: no prior auto-eval redo siblings. Tests exercising the
    # dedupe/supersede behaviour override this per-test.
    scheduler.get_tasks = AsyncMock(return_value=[])
    harness.scheduler = scheduler

    metadata: dict = {
        'modules': ['mod_a'],
        'files': ['mod_a/f.py'],
    }
    if optimistic_path:
        metadata['optimistic_path'] = optimistic_path
    if auto_eval_redo:
        metadata['auto_eval_redo'] = True

    assignment = TaskAssignment(
        task_id='orig-task',
        task={
            'id': 'orig-task',
            'title': 'Document foo',
            'description': 'd',
            'priority': 'medium',
            'metadata': metadata,
        },
        modules=['mod_a'],
    )

    # Multi-call dispatch — distinguish submit_task vs set_task_status.
    submit_calls: list[tuple[str, dict]] = []

    async def _dispatch_recording(name, args, *, timeout=15):
        submit_calls.append((name, args))
        if name == 'submit_task':
            return submit_returns
        return {'ok': True}

    scheduler.dispatch_tool = _dispatch_recording

    return _Fixture(
        harness=harness,
        assignment=assignment,
        git_ops_rename=git_ops_rename,
        dispatch_tool=dispatch_tool,
        update_task=update_task,
        event_emit=harness.event_store.emit,
        submit_calls=submit_calls,
    )


def _make_report(
    *,
    outcome: WorkflowOutcome = WorkflowOutcome.BLOCKED,
    block_phase: str = 'verify',
) -> TaskReport:
    return TaskReport(
        task_id='orig-task',
        title='Document foo',
        outcome=outcome,
        block_phase=block_phase,
        cost_usd=0.0,
    )


# ---------------------------------------------------------------------------
# Harness._find_prior_auto_eval_redos
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_find_prior_auto_eval_redos_filters_siblings():
    harness = Harness.__new__(Harness)
    harness.scheduler = MagicMock()
    harness.scheduler.get_tasks = AsyncMock(return_value=[
        # (a) matching siblings — same spawned_from, auto_eval_redo truthy.
        {
            'id': 'redo-A',
            'status': 'deferred',
            'metadata': {'auto_eval_redo': True, 'spawned_from': 'orig-task'},
        },
        {
            'id': 'redo-B',
            'status': 'pending',
            'metadata': {'auto_eval_redo': True, 'spawned_from': 'orig-task'},
        },
        # (b) non-redo task sharing spawned_from — must be excluded.
        {
            'id': 'other',
            'metadata': {'spawned_from': 'orig-task'},
        },
        # (c) redo for a different parent — must be excluded.
        {
            'id': 'redo-X',
            'metadata': {'auto_eval_redo': True, 'spawned_from': 'someone-else'},
        },
        # (d) the original task itself — must be excluded.
        {
            'id': 'orig-task',
            'metadata': {},
        },
    ])

    result = await harness._find_prior_auto_eval_redos('orig-task')

    assert set(result) == {'redo-A', 'redo-B'}
    harness.scheduler.get_tasks.assert_awaited_once_with(
        statuses=ACTIVE_TASK_STATUSES,
    )


@pytest.mark.asyncio
async def test_find_prior_auto_eval_redos_fail_open():
    harness = Harness.__new__(Harness)
    harness.scheduler = MagicMock()
    harness.scheduler.get_tasks = AsyncMock(side_effect=RuntimeError('boom'))

    result = await harness._find_prior_auto_eval_redos('orig-task')

    assert result == []


# ---------------------------------------------------------------------------
# Harness._filter_cancellable_redos (task 2075 review amendment)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_filter_cancellable_redos_excludes_in_progress_and_blocked():
    """Only still-pending/deferred candidates are safe to auto-cancel.

    'in-progress' may be real agent work in flight; 'blocked' may be an
    open human-facing escalation. Neither should be silently superseded.
    """
    harness = Harness.__new__(Harness)
    harness.scheduler = MagicMock()
    harness.scheduler.get_tasks = AsyncMock(return_value=[
        {'id': 'redo-pending', 'status': 'pending'},
        {'id': 'redo-deferred', 'status': 'deferred'},
        {'id': 'redo-in-progress', 'status': 'in-progress'},
        {'id': 'redo-blocked', 'status': 'blocked'},
    ])

    result = await harness._filter_cancellable_redos(
        ['redo-pending', 'redo-deferred', 'redo-in-progress', 'redo-blocked'],
    )

    assert set(result) == {'redo-pending', 'redo-deferred'}


@pytest.mark.asyncio
async def test_filter_cancellable_redos_excludes_raced_to_terminal():
    """A candidate that completed/was cancelled since the snapshot must not
    be re-cancelled — it is simply absent from the fresh active-only fetch.
    """
    harness = Harness.__new__(Harness)
    harness.scheduler = MagicMock()
    harness.scheduler.get_tasks = AsyncMock(return_value=[
        {'id': 'redo-still-here', 'status': 'deferred'},
        # 'redo-done' is absent: it raced to a terminal state and the
        # server-side ACTIVE_TASK_STATUSES filter excludes it.
    ])

    result = await harness._filter_cancellable_redos(
        ['redo-still-here', 'redo-done'],
    )

    assert result == ['redo-still-here']


@pytest.mark.asyncio
async def test_filter_cancellable_redos_fail_open():
    harness = Harness.__new__(Harness)
    harness.scheduler = MagicMock()
    harness.scheduler.get_tasks = AsyncMock(side_effect=RuntimeError('boom'))

    result = await harness._filter_cancellable_redos(['redo-1'])

    assert result == []


@pytest.mark.asyncio
async def test_filter_cancellable_redos_empty_candidates_skips_query():
    harness = Harness.__new__(Harness)
    harness.scheduler = MagicMock()
    harness.scheduler.get_tasks = AsyncMock(return_value=[])

    result = await harness._filter_cancellable_redos([])

    assert result == []
    harness.scheduler.get_tasks.assert_not_awaited()


@pytest.mark.asyncio
async def test_dispatches_redo_on_block(tmp_path: Path):
    f = _make(project_root=tmp_path / 'proj')
    await f.harness._maybe_auto_eval(f.assignment, _make_report())

    f.git_ops_rename.assert_awaited_once()
    rename_kwargs = f.git_ops_rename.await_args.kwargs
    assert rename_kwargs['old_branch'] == 'orig-task'
    assert rename_kwargs['new_branch'] == 'orig-task-skip-attempt'

    # submit_task + set_task_status both fired
    submitted = [c for c in f.submit_calls if c[0] == 'submit_task']
    flipped = [c for c in f.submit_calls if c[0] == 'set_task_status']
    assert len(submitted) == 1
    assert len(flipped) == 1
    submit_args = submitted[0][1]
    assert submit_args['planning_mode'] is True
    assert submit_args['metadata']['force_full_path'] is True
    assert submit_args['metadata']['auto_eval_redo'] is True
    assert submit_args['metadata']['auto_eval_pair'] == 'orig-task'

    # Cross-link back on original
    f.update_task.assert_awaited_once()
    update_args = f.update_task.await_args
    assert update_args.args[0] == 'orig-task'
    assert update_args.kwargs['metadata']['auto_eval_pair'] == 'redo-1'

    # Event emitted
    emit_calls = [
        c for c in f.event_emit.call_args_list
        if c.args and c.args[0] == EventType.auto_eval_dispatched
    ]
    assert len(emit_calls) == 1
    data = emit_calls[0].kwargs['data']
    assert data['original_task_id'] == 'orig-task'
    assert data['optimistic_path'] == 'simple_task'

    assert 'redo-1' in f.harness._auto_eval_redo_task_ids


@pytest.mark.asyncio
async def test_skipped_when_no_optimistic_path(tmp_path: Path):
    f = _make(project_root=tmp_path / 'proj', optimistic_path=None)
    await f.harness._maybe_auto_eval(f.assignment, _make_report())
    f.git_ops_rename.assert_not_awaited()
    assert f.submit_calls == []


@pytest.mark.asyncio
async def test_skipped_for_redo_of_redo(tmp_path: Path):
    f = _make(project_root=tmp_path / 'proj', auto_eval_redo=True)
    await f.harness._maybe_auto_eval(f.assignment, _make_report())
    f.git_ops_rename.assert_not_awaited()


@pytest.mark.asyncio
async def test_skipped_for_excluded_phase(tmp_path: Path):
    f = _make(project_root=tmp_path / 'proj', block_phase='merge')
    await f.harness._maybe_auto_eval(
        f.assignment, _make_report(block_phase='merge'),
    )
    f.git_ops_rename.assert_not_awaited()


@pytest.mark.asyncio
async def test_skipped_for_non_blocked_outcome(tmp_path: Path):
    f = _make(project_root=tmp_path / 'proj')
    await f.harness._maybe_auto_eval(
        f.assignment,
        _make_report(outcome=WorkflowOutcome.DONE),
    )
    f.git_ops_rename.assert_not_awaited()


@pytest.mark.asyncio
async def test_disabled_via_config(tmp_path: Path):
    f = _make(project_root=tmp_path / 'proj', auto_eval_enabled=False)
    await f.harness._maybe_auto_eval(f.assignment, _make_report())
    f.git_ops_rename.assert_not_awaited()


@pytest.mark.asyncio
async def test_budget_exhausted_skips_redo(tmp_path: Path):
    f = _make(project_root=tmp_path / 'proj', redo_budget_usd=10.0)

    async def _high_used() -> float:
        return 100.0

    f.harness._auto_eval_budget_used_24h = _high_used  # type: ignore[assignment]

    await f.harness._maybe_auto_eval(f.assignment, _make_report())
    f.git_ops_rename.assert_not_awaited()


@pytest.mark.asyncio
async def test_rename_failure_does_not_block_redo(tmp_path: Path):
    f = _make(
        project_root=tmp_path / 'proj',
        rename_raises=RuntimeError('worktree move failed'),
    )

    await f.harness._maybe_auto_eval(f.assignment, _make_report())

    # rename was attempted but failed — redo still dispatched
    f.git_ops_rename.assert_awaited_once()
    submitted = [c for c in f.submit_calls if c[0] == 'submit_task']
    assert len(submitted) == 1

    emit_calls = [
        c for c in f.event_emit.call_args_list
        if c.args and c.args[0] == EventType.auto_eval_dispatched
    ]
    assert emit_calls
    assert emit_calls[0].kwargs['data']['rename_succeeded'] is False


@pytest.mark.asyncio
async def test_no_worktree_no_rename(tmp_path: Path):
    f = _make(project_root=tmp_path / 'proj', worktree_exists=False)
    await f.harness._maybe_auto_eval(f.assignment, _make_report())
    f.git_ops_rename.assert_not_awaited()
    submitted = [c for c in f.submit_calls if c[0] == 'submit_task']
    assert len(submitted) == 1


@pytest.mark.asyncio
async def test_submit_returns_no_task_id(tmp_path: Path):
    f = _make(
        project_root=tmp_path / 'proj',
        submit_returns={'error': 'curator full'},
    )
    await f.harness._maybe_auto_eval(f.assignment, _make_report())
    submitted = [c for c in f.submit_calls if c[0] == 'submit_task']
    flipped = [c for c in f.submit_calls if c[0] == 'set_task_status']
    assert len(submitted) == 1
    assert len(flipped) == 0  # status flip not called
    assert f.harness._auto_eval_redo_task_ids == set()


@pytest.mark.asyncio
async def test_extract_task_id_from_structured_content(tmp_path: Path):
    """submit_task may return MCP envelope shapes."""
    f = _make(
        project_root=tmp_path / 'proj',
        submit_returns={
            'content': [{'type': 'text', 'text': '{"task_id": "redo-99"}'}],
        },
    )
    await f.harness._maybe_auto_eval(f.assignment, _make_report())
    assert 'redo-99' in f.harness._auto_eval_redo_task_ids


@pytest.mark.asyncio
async def test_extract_task_id_handles_structured_content_dict(tmp_path: Path):
    f = _make(
        project_root=tmp_path / 'proj',
        submit_returns={'structuredContent': {'task_id': 'redo-77'}},
    )
    await f.harness._maybe_auto_eval(f.assignment, _make_report())
    assert 'redo-77' in f.harness._auto_eval_redo_task_ids


@pytest.mark.asyncio
async def test_auto_eval_back_link_forwards_merge_mode(tmp_path: Path, monkeypatch):
    """Back-link update_task wire call must carry metadata_mode='merge' (#4271 fix).

    The auto-eval redo back-link writes the full task_metadata blob enriched
    with auto_eval_pair.  Under merge (shallow last-write-wins) sibling keys
    present in the DB but absent from task_metadata survive; under replace they
    are silently deleted.

    RED today: wrapper forwards nothing instead of metadata_mode='merge'.
    GREEN after step-5 impl (scheduler.py update_task gains metadata_mode param).
    """
    # Seed task_metadata with a sibling key so the full-blob assertion can
    # verify it is present in the wire payload.
    f = _make(project_root=tmp_path / 'proj')
    f.assignment.task['metadata']['memory_hints'] = ['hint-A']

    # Capture update_task wire calls via a REAL Scheduler + monkeypatched mcp_call.
    captured_update_payloads: list[dict] = []

    async def mock_mcp_call(url, method, payload, **kwargs):
        captured_update_payloads.append(payload)
        return {}

    real_scheduler = Scheduler(OrchestratorConfig(max_per_module=1))
    monkeypatch.setattr('orchestrator.scheduler.mcp_call', mock_mcp_call)

    # Replace only update_task on the mock scheduler; dispatch_tool stays on the
    # MagicMock so submit_task/set_task_status return the stubbed fixture responses.
    f.harness.scheduler.update_task = real_scheduler.update_task

    await f.harness._maybe_auto_eval(f.assignment, _make_report())

    # update_task must have been called exactly once (the back-link write).
    arguments = assert_update_wire_mode(captured_update_payloads, 'merge')

    # Full-blob assertions: auto_eval_pair and seeded sibling memory_hints both present.
    blob = _json.loads(arguments['metadata'])
    assert blob.get('auto_eval_pair') == 'redo-1', (
        f"auto_eval_pair must be 'redo-1' in blob; got: {blob}"
    )
    assert blob.get('memory_hints') == ['hint-A'], (
        f"Sibling memory_hints must survive in written blob; got: {blob}"
    )


# ---------------------------------------------------------------------------
# Supersede prior auto-eval redo siblings on repeated re-block (task 2075)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_supersedes_prior_redo_siblings(tmp_path: Path):
    f = _make(project_root=tmp_path / 'proj')
    f.harness.scheduler.get_tasks = AsyncMock(return_value=[
        {
            'id': 'redo-old-1',
            'status': 'deferred',
            'metadata': {'auto_eval_redo': True, 'spawned_from': 'orig-task'},
        },
        {
            'id': 'redo-old-2',
            'status': 'pending',
            'metadata': {'auto_eval_redo': True, 'spawned_from': 'orig-task'},
        },
    ])

    await f.harness._maybe_auto_eval(f.assignment, _make_report())

    submitted = [c for c in f.submit_calls if c[0] == 'submit_task']
    status_calls = [c for c in f.submit_calls if c[0] == 'set_task_status']
    cancelled = [c for c in status_calls if c[1].get('status') == 'cancelled']
    flipped = [c for c in status_calls if c[1].get('status') == 'pending']

    assert len(submitted) == 1
    assert {c[1].get('id') for c in cancelled} == {'redo-old-1', 'redo-old-2'}
    # The new redo's own deferred → pending flip must still fire.
    assert len(flipped) == 1
    assert flipped[0][1].get('id') == 'redo-1'

    emit_calls = [
        c for c in f.event_emit.call_args_list
        if c.args and c.args[0] == EventType.auto_eval_dispatched
    ]
    assert len(emit_calls) == 1
    data = emit_calls[0].kwargs['data']
    assert set(data['superseded_redo_ids']) == {'redo-old-1', 'redo-old-2'}


@pytest.mark.asyncio
async def test_no_supersede_when_no_prior_redos(tmp_path: Path):
    """Regression guard: happy path unchanged when there are no prior redos."""
    f = _make(project_root=tmp_path / 'proj')
    # _make's default scheduler.get_tasks stub returns [] (no prior siblings).

    await f.harness._maybe_auto_eval(f.assignment, _make_report())

    status_calls = [c for c in f.submit_calls if c[0] == 'set_task_status']
    cancelled = [c for c in status_calls if c[1].get('status') == 'cancelled']
    flipped = [c for c in status_calls if c[1].get('status') == 'pending']

    assert cancelled == []
    assert len(flipped) == 1

    emit_calls = [
        c for c in f.event_emit.call_args_list
        if c.args and c.args[0] == EventType.auto_eval_dispatched
    ]
    assert len(emit_calls) == 1
    assert emit_calls[0].kwargs['data']['superseded_redo_ids'] == []


@pytest.mark.asyncio
async def test_supersede_excludes_in_progress_and_blocked_siblings(tmp_path: Path):
    """in-progress (active agent work) and blocked (open human escalation)
    prior redos must be left alone — only pending/deferred are superseded.
    """
    f = _make(project_root=tmp_path / 'proj')
    f.harness.scheduler.get_tasks = AsyncMock(return_value=[
        {
            'id': 'redo-old-deferred',
            'status': 'deferred',
            'metadata': {'auto_eval_redo': True, 'spawned_from': 'orig-task'},
        },
        {
            'id': 'redo-old-in-progress',
            'status': 'in-progress',
            'metadata': {'auto_eval_redo': True, 'spawned_from': 'orig-task'},
        },
        {
            'id': 'redo-old-blocked',
            'status': 'blocked',
            'metadata': {'auto_eval_redo': True, 'spawned_from': 'orig-task'},
        },
    ])

    await f.harness._maybe_auto_eval(f.assignment, _make_report())

    status_calls = [c for c in f.submit_calls if c[0] == 'set_task_status']
    cancelled_ids = {
        c[1].get('id') for c in status_calls if c[1].get('status') == 'cancelled'
    }
    assert cancelled_ids == {'redo-old-deferred'}

    emit_calls = [
        c for c in f.event_emit.call_args_list
        if c.args and c.args[0] == EventType.auto_eval_dispatched
    ]
    assert len(emit_calls) == 1
    assert emit_calls[0].kwargs['data']['superseded_redo_ids'] == [
        'redo-old-deferred',
    ]


@pytest.mark.asyncio
async def test_supersede_cancel_failure_is_isolated(tmp_path: Path):
    """One stale sibling's cancel dispatch raising must not abort the other
    cancellation, the new redo's pending flip, back-link, or event emission.
    """
    f = _make(project_root=tmp_path / 'proj')
    f.harness.scheduler.get_tasks = AsyncMock(return_value=[
        {
            'id': 'redo-old-1',
            'status': 'deferred',
            'metadata': {'auto_eval_redo': True, 'spawned_from': 'orig-task'},
        },
        {
            'id': 'redo-old-2',
            'status': 'pending',
            'metadata': {'auto_eval_redo': True, 'spawned_from': 'orig-task'},
        },
    ])

    calls: list[tuple[str, dict]] = []

    async def _flaky_dispatch(name, args, *, timeout=15):
        calls.append((name, args))
        if (
            name == 'set_task_status'
            and args.get('id') == 'redo-old-1'
            and args.get('status') == 'cancelled'
        ):
            raise RuntimeError('transient dispatch failure')
        if name == 'submit_task':
            return {'task_id': 'redo-1', 'status': 'deferred'}
        return {'ok': True}

    f.harness.scheduler.dispatch_tool = _flaky_dispatch

    # Must not raise despite the failing cancel dispatch.
    await f.harness._maybe_auto_eval(f.assignment, _make_report())

    cancelled_ids = {
        c[1].get('id') for c in calls
        if c[0] == 'set_task_status' and c[1].get('status') == 'cancelled'
    }
    assert cancelled_ids == {'redo-old-1', 'redo-old-2'}

    flipped = [
        c for c in calls
        if c[0] == 'set_task_status' and c[1].get('status') == 'pending'
    ]
    assert len(flipped) == 1
    assert flipped[0][1].get('id') == 'redo-1'

    f.update_task.assert_awaited_once()

    emit_calls = [
        c for c in f.event_emit.call_args_list
        if c.args and c.args[0] == EventType.auto_eval_dispatched
    ]
    assert len(emit_calls) == 1
    assert set(emit_calls[0].kwargs['data']['superseded_redo_ids']) == {
        'redo-old-1', 'redo-old-2',
    }
