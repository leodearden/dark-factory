"""Tests for the WP-D backlog escalation policy + orchestrator detector."""

from __future__ import annotations

import asyncio
import json
import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, MagicMock

import pytest
import pytest_asyncio
from _fm_helpers import pydantic_spec, submit_and_resolve

from fused_memory.config.schema import FusedMemoryConfig
from fused_memory.reconciliation.backlog_policy import BacklogPolicy
from fused_memory.reconciliation.event_buffer import EventBuffer
from fused_memory.services.orchestrator_detector import (
    is_orchestrator_live_for,
    orchestrator_started_at,
)

if TYPE_CHECKING:
    pass


# ── Fixtures ──────────────────────────────────────────────────────────────


@pytest_asyncio.fixture
async def event_buffer(tmp_path):
    buf = EventBuffer(db_path=tmp_path / 'backlog_eb.db', buffer_size_threshold=100)
    await buf.initialize()
    yield buf
    await buf.close()


class _StubQueue:
    """Minimal EventQueue-like stub exposing ``stats()``."""

    def __init__(self, queue_depth: int = 0, retry_in_flight: int = 0) -> None:
        self._depth = queue_depth
        self._retry = retry_in_flight

    def stats(self) -> dict:
        return {
            'queue_depth': self._depth,
            'retry_in_flight': self._retry,
            'queue_capacity': 10_000,
            'last_commit_ts': None,
            'events_committed': 0,
            'overflow_drops': 0,
            'dead_letters': 0,
            'drainer_running': True,
        }


async def _seed_buffered(event_buffer: EventBuffer, project_id: str, n: int) -> None:
    """Insert ``n`` buffered events for ``project_id`` directly via the schema."""
    import uuid
    from datetime import UTC, datetime

    from fused_memory.models.reconciliation import (
        EventSource,
        EventType,
        ReconciliationEvent,
    )

    for _ in range(n):
        event = ReconciliationEvent(
            id=str(uuid.uuid4()),
            type=EventType.task_created,
            source=EventSource.agent,
            project_id=project_id,
            timestamp=datetime.now(UTC),
            payload={'seed': True},
        )
        await event_buffer.push(event)


# ── BacklogPolicy.check ───────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_ok_verdict_when_under_threshold(event_buffer):
    """Under the hard limit → ok verdict, no escalation, no rejection."""
    await _seed_buffered(event_buffer, 'proj', n=3)

    def detector(_root: str) -> bool:
        return False

    policy = BacklogPolicy(
        event_buffer,
        _StubQueue(),
        detector,
        hard_limit=10,
    )
    verdict = await policy.check('proj', project_root='/does/not/matter')
    assert verdict.outcome == 'ok'
    assert verdict.to_error_dict() == {}


@pytest.mark.asyncio
async def test_rejection_verdict_when_over_threshold_and_no_orchestrator(
    event_buffer, tmp_path,
):
    """Over threshold + no orchestrator → structured rejection, no file written."""
    await _seed_buffered(event_buffer, 'proj', n=6)
    project_root = tmp_path / 'proj_root'
    project_root.mkdir()

    def detector(_root: str) -> bool:
        return False

    policy = BacklogPolicy(
        event_buffer,
        _StubQueue(),
        detector,
        hard_limit=5,
    )
    verdict = await policy.check('proj', project_root=str(project_root))
    assert verdict.outcome == 'rejection'
    assert verdict.backlog == 6
    assert verdict.threshold == 5
    assert verdict.project_id == 'proj'
    err = verdict.to_error_dict()
    assert err['error_type'] == 'ReconciliationBacklogExceeded'
    assert 'backlog 6 > limit 5' in err['error']
    # No escalation file.
    esc_dir = project_root / 'data' / 'escalations'
    assert not esc_dir.exists() or not any(esc_dir.iterdir())


@pytest.mark.asyncio
async def test_escalation_when_over_threshold_and_orchestrator_live(
    event_buffer, tmp_path,
):
    """Over threshold + orchestrator live → escalation JSON on disk."""
    await _seed_buffered(event_buffer, 'proj', n=12)
    project_root = tmp_path / 'proj_root'
    project_root.mkdir()

    def detector(root: str) -> bool:
        return root == str(project_root)

    policy = BacklogPolicy(
        event_buffer,
        _StubQueue(),
        detector,
        hard_limit=10,
    )
    verdict = await policy.check('proj', project_root=str(project_root))
    assert verdict.outcome == 'escalated'
    assert verdict.escalation_path is not None
    path = Path(verdict.escalation_path)
    assert path.exists()
    body = json.loads(path.read_text())
    assert body['id'].startswith('esc-reconciliation-backlog-')
    assert body['severity'] == 'blocking'
    assert body['level'] == 1
    assert body['suggested_action'] == 'drain_reconciliation'
    assert body['backlog'] == 12
    assert body['threshold'] == 10
    assert body['project_id'] == 'proj'
    assert body['workflow_state'] == 'infra'
    assert body['category'] == 'infra_issue'


@pytest.mark.asyncio
async def test_rate_limit_prevents_spam(event_buffer, tmp_path):
    """Two triggers inside the rate window → one file only."""
    await _seed_buffered(event_buffer, 'proj', n=12)
    project_root = tmp_path / 'proj_root'
    project_root.mkdir()
    clock = {'now': 1_000_000.0}

    def now() -> float:
        return clock['now']

    policy = BacklogPolicy(
        event_buffer,
        _StubQueue(),
        lambda _: True,
        hard_limit=10,
        rate_limit_seconds=900.0,
        time_provider=now,
    )
    v1 = await policy.check('proj', project_root=str(project_root))
    clock['now'] += 60.0  # within window
    v2 = await policy.check('proj', project_root=str(project_root))

    assert v1.outcome == 'escalated'
    assert v2.outcome == 'escalated'
    # Only the first wrote a file; second returned escalated verdict with no path.
    assert v1.escalation_path is not None
    assert v2.escalation_path is None
    esc_files = list((project_root / 'data' / 'escalations').iterdir())
    assert len(esc_files) == 1


@pytest.mark.asyncio
async def test_rate_limit_allows_after_window(event_buffer, tmp_path):
    """Advance clock past the rate window → second trigger writes another file."""
    await _seed_buffered(event_buffer, 'proj', n=12)
    project_root = tmp_path / 'proj_root'
    project_root.mkdir()
    clock = {'now': 1_000_000.0}

    def now() -> float:
        return clock['now']

    policy = BacklogPolicy(
        event_buffer,
        _StubQueue(),
        lambda _: True,
        hard_limit=10,
        rate_limit_seconds=900.0,
        time_provider=now,
    )
    v1 = await policy.check('proj', project_root=str(project_root))
    clock['now'] += 901.0  # just past window
    v2 = await policy.check('proj', project_root=str(project_root))

    assert v1.escalation_path is not None
    assert v2.escalation_path is not None
    assert v1.escalation_path != v2.escalation_path
    esc_files = sorted((project_root / 'data' / 'escalations').iterdir())
    assert len(esc_files) == 2


@pytest.mark.asyncio
async def test_on_judge_halt_writes_escalation(event_buffer, tmp_path):
    """Judge halt routes through escalation path when orchestrator is live."""
    project_root = tmp_path / 'proj_root'
    project_root.mkdir()

    policy = BacklogPolicy(
        event_buffer,
        _StubQueue(),
        lambda _: True,
        hard_limit=500,
    )
    policy.register_project_root('proj', str(project_root))
    verdict = await policy.on_judge_halt('proj', reason='too many serious findings')

    assert verdict.outcome == 'escalated'
    assert verdict.error_type == 'ReconciliationJudgeHalted'
    files = list((project_root / 'data' / 'escalations').iterdir())
    assert len(files) == 1
    body = json.loads(files[0].read_text())
    assert body['error_type'] == 'ReconciliationJudgeHalted'
    assert 'too many serious findings' in body['detail']


@pytest.mark.asyncio
async def test_on_watchdog_wedge_writes_escalation_with_wedge_error_type(
    event_buffer, tmp_path,
):
    """Wedge payload → escalation with error_type=SqliteDrainerWedged."""
    project_root = tmp_path / 'proj_root'
    project_root.mkdir()

    policy = BacklogPolicy(
        event_buffer,
        _StubQueue(),
        lambda _: True,
        hard_limit=500,
    )
    policy.register_project_root('proj', str(project_root))
    verdicts = await policy.on_watchdog_wedge({
        'stale_for_seconds': 180.0,
        'queue_depth': 42,
        'retry_in_flight': 3,
    })

    assert len(verdicts) == 1
    v = verdicts[0]
    assert v.outcome == 'escalated'
    assert v.error_type == 'SqliteDrainerWedged'
    files = list((project_root / 'data' / 'escalations').iterdir())
    assert len(files) == 1
    body = json.loads(files[0].read_text())
    assert body['error_type'] == 'SqliteDrainerWedged'
    assert 'stale_for_seconds' in body['detail']


# ── Startup seeding of known project roots (task 2998 GAP A) ──────────────


@pytest.mark.asyncio
async def test_register_known_project_roots_seeds_every_root(event_buffer, tmp_path):
    """Every known root is locatable BEFORE any check()/mutating MCP call.

    GAP A: at startup the only caller of ``register_project_root`` was
    ``check()``, so a halt rehydrated by ``Judge.initialize()`` had no root to
    escalate against.
    """
    root_a = tmp_path / 'reify_root'
    root_b = tmp_path / 'df_root'
    root_a.mkdir()
    root_b.mkdir()

    policy = BacklogPolicy(event_buffer, _StubQueue(), lambda _: True)
    policy.register_known_project_roots({
        'reify': str(root_a),
        'dark_factory': str(root_b),
    })

    # No check() has run — the roots must already resolve.
    assert policy.project_root_for('reify') == str(root_a)
    assert policy.project_root_for('dark_factory') == str(root_b)


@pytest.mark.asyncio
async def test_explicit_registration_clears_startup_seeded_provenance(
    event_buffer, tmp_path,
):
    """A real mutating-call registration promotes a seeded id back to active."""
    root_a = tmp_path / 'reify_root'
    root_a.mkdir()

    policy = BacklogPolicy(event_buffer, _StubQueue(), lambda _: True)
    policy.register_known_project_roots({'reify': str(root_a)})
    assert 'reify' in policy._startup_seeded

    policy.register_project_root('reify', str(root_a))
    assert 'reify' not in policy._startup_seeded


@pytest.mark.asyncio
async def test_watchdog_wedge_does_not_fan_out_to_idle_startup_seeded_projects(
    event_buffer, tmp_path,
):
    """Startup seeding must not turn one drainer wedge into N escalations.

    Seeding populates ``_state`` for every KNOWN project, and
    ``_projects_with_backlog`` derives its fan-out set from ``_state``.  An
    idle startup-seeded project (zero buffered events) must be skipped; an
    explicitly-registered project is escalated unconditionally, preserving
    ``test_on_watchdog_wedge_writes_escalation_with_wedge_error_type``.

    The queue MUST be non-empty here.  ``SqliteWatchdog._tick`` computes
    ``outstanding = queue_depth + retry_in_flight`` and fires the wedge
    callback only inside the ``wedged = outstanding > 0 and stale_for >
    threshold`` branch (sqlite_watchdog.py:118-159), so a zero-depth queue
    is a state in which ``on_watchdog_wedge`` can NEVER be invoked in
    production.  Exercising the guard against ``_StubQueue()`` would
    green-light a filter that is dead code at runtime: ``current_backlog``
    folds the GLOBAL queue depth into every project's count, which is
    non-zero by construction whenever a real wedge fires.
    """
    roots = {}
    for name in ('active', 'idle', 'explicit'):
        root = tmp_path / f'{name}_root'
        root.mkdir()
        roots[name] = root

    policy = BacklogPolicy(event_buffer, _StubQueue(queue_depth=5), lambda _: True)
    policy.register_known_project_roots({
        'active': str(roots['active']),
        'idle': str(roots['idle']),
    })
    policy.register_project_root('explicit', str(roots['explicit']))
    await _seed_buffered(event_buffer, 'active', n=3)

    verdicts = await policy.on_watchdog_wedge({'stale_for_seconds': 180.0})

    assert {v.project_id for v in verdicts} == {'active', 'explicit'}
    assert not (roots['idle'] / 'data' / 'escalations').exists()


@pytest.mark.asyncio
async def test_watchdog_wedge_survives_count_buffered_failure(tmp_path, caplog):
    """An unavailable buffered count must not SUPPRESS the wedge alert.

    ``count_buffered`` raises when the db is not initialised and on any sqlite
    error ('database is locked', disk I/O) — precisely the conditions under
    which a drainer wedge fires. An unguarded raise propagates out of
    ``on_watchdog_wedge`` (the watchdog's ``except Exception`` merely logs
    'wedge_callback raised'), so ZERO escalations get written for ANY project,
    including explicitly-registered active ones. Fail loud AND inclusive.
    """
    logger_name = 'fused_memory.reconciliation.backlog_policy'
    roots = {}
    for name in ('explicit', 'seeded'):
        root = tmp_path / f'{name}_root'
        root.mkdir()
        roots[name] = root

    broken_buffer = MagicMock()
    broken_buffer.count_buffered = AsyncMock(
        side_effect=RuntimeError('database is locked'),
    )

    policy = BacklogPolicy(broken_buffer, _StubQueue(queue_depth=5), lambda _: True)
    policy.register_project_root('explicit', str(roots['explicit']))
    policy.register_known_project_roots({'seeded': str(roots['seeded'])})

    with caplog.at_level(logging.WARNING, logger=logger_name):
        verdicts = await policy.on_watchdog_wedge({'stale_for_seconds': 180.0})

    # Both projects are alerted — the seeded one is NOT dropped just because
    # its count was unavailable.
    assert {v.project_id for v in verdicts} == {'explicit', 'seeded'}
    assert all(v.outcome == 'escalated' for v in verdicts)
    for root in roots.values():
        files = list((root / 'data' / 'escalations').iterdir())
        assert len(files) == 1
        # buffered is None → the escalation reports the global queue pressure
        # alone (5) rather than a fabricated per-project count.
        assert json.loads(files[0].read_text())['backlog'] == 5

    text = '\n'.join(r.getMessage() for r in caplog.records if r.name == logger_name)
    assert 'count_buffered failed' in text
    assert 'database is locked' in text

    # The count is taken ONCE per project (the wedge path reuses the filter's
    # value instead of re-querying via current_backlog).
    assert broken_buffer.count_buffered.await_count == 2


@pytest.mark.asyncio
async def test_rejection_branch_warning_is_throttled(event_buffer, tmp_path, caplog):
    """The rejection WARNING must not repeat on every ~5s halted tick.

    GAP B deliberately stopped burning the dedupe token on a failed write, so
    a halted project with no live orchestrator re-enters the rejection branch
    every harness tick. Unthrottled that is ~17k WARNING lines/day/project,
    which drowns the signal the line exists to raise. First rejection warns;
    repeats inside the window are DEBUG; a rejection still happening a window
    later warns again.
    """
    logger_name = 'fused_memory.reconciliation.backlog_policy'
    project_root = tmp_path / 'proj_root'
    project_root.mkdir()

    clock = {'t': 1000.0}
    policy = BacklogPolicy(
        event_buffer,
        _StubQueue(),
        lambda _: False,  # no live orchestrator → rejection branch
        rate_limit_seconds=900.0,
        time_provider=lambda: clock['t'],
    )
    policy.register_project_root('proj', str(project_root))

    def _counts():
        warns = [
            r for r in caplog.records
            if r.name == logger_name and r.levelno == logging.WARNING
        ]
        debugs = [
            r for r in caplog.records
            if r.name == logger_name and r.levelno == logging.DEBUG
        ]
        return len(warns), len(debugs)

    with caplog.at_level(logging.DEBUG, logger=logger_name):
        for _ in range(4):
            assert (await policy.on_judge_halt('proj', reason='r')).is_rejection
        assert _counts() == (1, 3)

        # A DIFFERENT fault kind keeps its own clock — a throttled halt must
        # never mute a first backlog/wedge rejection.
        await policy.on_watchdog_wedge({'stale_for_seconds': 180.0})
        assert _counts() == (2, 3)

        # Still rejecting a full window later → loud again.
        clock['t'] += 901.0
        assert (await policy.on_judge_halt('proj', reason='r')).is_rejection
        assert _counts() == (3, 3)


@pytest.mark.asyncio
async def test_rejection_branch_logs_why_no_escalation_was_written(
    event_buffer, tmp_path, caplog,
):
    """The rejection branch must say WHY nothing was written.

    The incident's defining symptom was that a halted project produced NO
    backlog_policy log line of ANY kind, so the drop was invisible. The two
    causes must be discriminated: an unregistered project_root vs. no live
    orchestrator for a root we do have.
    """
    logger_name = 'fused_memory.reconciliation.backlog_policy'

    # (a) No registered root at all.
    policy = BacklogPolicy(event_buffer, _StubQueue(), lambda _: True)
    with caplog.at_level(logging.WARNING, logger=logger_name):
        verdict = await policy.on_judge_halt('unregistered', reason='r')
    assert verdict.outcome == 'rejection'
    text = '\n'.join(
        r.getMessage() for r in caplog.records if r.name == logger_name
    )
    assert 'unregistered' in text
    assert 'judge_halt' in text
    assert 'ReconciliationJudgeHalted' in text
    assert 'project_root' in text and 'not registered' in text
    assert 'orchestrator' not in text

    # (b) Root registered, but no live orchestrator.
    caplog.clear()
    project_root = tmp_path / 'proj_root'
    project_root.mkdir()
    policy_b = BacklogPolicy(event_buffer, _StubQueue(), lambda _: False)
    policy_b.register_project_root('proj', str(project_root))
    with caplog.at_level(logging.WARNING, logger=logger_name):
        verdict_b = await policy_b.on_judge_halt('proj', reason='r')
    assert verdict_b.outcome == 'rejection'
    text_b = '\n'.join(
        r.getMessage() for r in caplog.records if r.name == logger_name
    )
    assert 'proj' in text_b
    assert 'judge_halt' in text_b
    assert 'ReconciliationJudgeHalted' in text_b
    assert 'no live orchestrator' in text_b
    assert str(project_root) in text_b


# ── orchestrator_detector ─────────────────────────────────────────────────


def test_orchestrator_detector_stale_lock_pid_dead(tmp_path):
    """PID in lock points to a process that doesn't exist → not live."""
    project_root = tmp_path / 'proj_root'
    lock_dir = project_root / 'data' / 'orchestrator'
    lock_dir.mkdir(parents=True)
    # Very high PID unlikely to be live.
    lock_dir.joinpath('orchestrator.lock').write_text(
        'PID 2147483646 started 2026-04-13T00:00:00Z\n', encoding='utf-8',
    )
    assert is_orchestrator_live_for(project_root) is False


def test_orchestrator_detector_live_pid(tmp_path):
    """PID for the current process → live (os.kill(pid,0) succeeds)."""
    project_root = tmp_path / 'proj_root'
    lock_dir = project_root / 'data' / 'orchestrator'
    lock_dir.mkdir(parents=True)
    my_pid = os.getpid()
    lock_dir.joinpath('orchestrator.lock').write_text(
        f'PID {my_pid} started 2026-04-18T00:00:00Z\n', encoding='utf-8',
    )
    assert is_orchestrator_live_for(project_root) is True


def test_orchestrator_detector_no_lock_file(tmp_path):
    """No orchestrator.lock → not live."""
    project_root = tmp_path / 'proj_root'
    project_root.mkdir()
    assert is_orchestrator_live_for(project_root) is False


def test_orchestrator_detector_unparseable_lock(tmp_path):
    """Garbage in lock → not live (defensive)."""
    project_root = tmp_path / 'proj_root'
    lock_dir = project_root / 'data' / 'orchestrator'
    lock_dir.mkdir(parents=True)
    lock_dir.joinpath('orchestrator.lock').write_text('not a pid\n', encoding='utf-8')
    assert is_orchestrator_live_for(project_root) is False


# ── orchestrator_started_at (task 2963) ───────────────────────────────────


def _write_orchestrator_lock(project_root, first_line: str):
    """Write ``<project_root>/data/orchestrator/orchestrator.lock`` with *first_line*.

    Mirrors the ``PID <N> started <ISO>`` lock-write fixture pattern used by
    the ``is_orchestrator_live_for`` tests above.
    """
    lock_dir = project_root / 'data' / 'orchestrator'
    lock_dir.mkdir(parents=True, exist_ok=True)
    lock_dir.joinpath('orchestrator.lock').write_text(first_line + '\n', encoding='utf-8')


def test_orchestrator_started_at_parses_z_suffix_utc_aware(tmp_path):
    """First line ``PID 123 started 2026-04-13T00:00:00Z`` → tz-aware UTC datetime."""
    from datetime import UTC, datetime, timedelta

    project_root = tmp_path / 'proj_root'
    _write_orchestrator_lock(project_root, 'PID 123 started 2026-04-13T00:00:00Z')

    started = orchestrator_started_at(project_root)
    assert started == datetime(2026, 4, 13, 0, 0, 0, tzinfo=UTC)
    # Verify the `Z` suffix parsed and the result is tz-aware.
    assert started is not None
    assert started.tzinfo is not None
    assert started.utcoffset() == timedelta(0)


def test_orchestrator_started_at_missing_lock_file(tmp_path):
    """No orchestrator.lock file → None (fail-safe, no raise)."""
    project_root = tmp_path / 'proj_root'
    project_root.mkdir()
    assert orchestrator_started_at(project_root) is None


def test_orchestrator_started_at_no_started_token(tmp_path):
    """First line has no ``started`` token (e.g. ``PID 123``) → None."""
    project_root = tmp_path / 'proj_root'
    _write_orchestrator_lock(project_root, 'PID 123')
    assert orchestrator_started_at(project_root) is None


def test_orchestrator_started_at_unparseable_timestamp(tmp_path):
    """``started`` followed by a non-timestamp token → None."""
    project_root = tmp_path / 'proj_root'
    _write_orchestrator_lock(project_root, 'PID 123 started not-a-time')
    assert orchestrator_started_at(project_root) is None


def test_orchestrator_started_at_oserror_returns_none(tmp_path):
    """OSError on read (lock path is a directory) → None (no raise)."""
    project_root = tmp_path / 'proj_root'
    lock_dir = project_root / 'data' / 'orchestrator'
    lock_dir.mkdir(parents=True)
    # Make orchestrator.lock a DIRECTORY so read_text raises OSError (IsADirectoryError).
    lock_dir.joinpath('orchestrator.lock').mkdir()
    assert orchestrator_started_at(project_root) is None


# ── TaskInterceptor integration ───────────────────────────────────────────


@pytest.fixture
def _taskmaster_mock():
    tm = AsyncMock()
    tm.get_task = AsyncMock(return_value={'id': '1', 'status': 'pending', 'title': 'T'})
    tm.set_task_status = AsyncMock(return_value={'success': True})
    tm.get_tasks = AsyncMock(return_value={'tasks': []})
    tm.add_task = AsyncMock(return_value={'id': '2', 'title': 'New'})
    tm.update_task = AsyncMock(return_value={'success': True})
    tm.remove_tasks = AsyncMock(return_value={'success': True})
    tm.add_dependency = AsyncMock(return_value={'success': True})
    tm.remove_dependency = AsyncMock(return_value={'success': True})
    tm.ensure_connected = AsyncMock()
    return tm


@pytest.mark.asyncio
async def test_task_interceptor_add_task_rejects_when_over_limit(
    event_buffer, tmp_path, _taskmaster_mock,
):
    """When policy rejects, interceptor returns error dict without mutating state."""
    import contextlib

    from fused_memory.middleware.task_interceptor import TaskInterceptor
    from fused_memory.middleware.ticket_store import TicketStore

    await _seed_buffered(event_buffer, 'proj_root', n=20)

    policy = BacklogPolicy(
        event_buffer,
        _StubQueue(),
        lambda _: False,  # no orchestrator
        hard_limit=5,
    )
    store = TicketStore(tmp_path / 'reject_tickets.db')
    await store.initialize()
    interceptor = None
    try:
        interceptor = TaskInterceptor(
            _taskmaster_mock,
            targeted_reconciler=None,
            event_buffer=event_buffer,
            backlog_policy=policy,
            ticket_store=store,
        )
        project_root = str(tmp_path / 'proj_root')
        result = await interceptor.submit_task(
            project_root=project_root, title='Should be rejected',
        )
        assert result.get('error_type') == 'ReconciliationBacklogExceeded'
        assert result['backlog'] == 20
        assert result['threshold'] == 5
        # taskmaster.add_task must NOT be called — the whole point of the rejection.
        _taskmaster_mock.add_task.assert_not_called()
    finally:
        await store.close()
        if interceptor is not None:
            for _wt in list(interceptor._worker_tasks.values()):
                if not _wt.done():
                    _wt.cancel()
                    with contextlib.suppress(asyncio.CancelledError, Exception):
                        await _wt


@pytest.mark.asyncio
async def test_task_interceptor_add_task_ok_when_under_limit(
    event_buffer, tmp_path, _taskmaster_mock,
):
    """Under-threshold → normal add_task flow, taskmaster called."""
    import contextlib

    from fused_memory.middleware.task_interceptor import TaskInterceptor
    from fused_memory.middleware.ticket_store import TicketStore

    policy = BacklogPolicy(
        event_buffer,
        _StubQueue(),
        lambda _: False,
        hard_limit=500,
    )
    config = MagicMock(spec_set=pydantic_spec(FusedMemoryConfig))
    config.curator.enabled = False
    store = TicketStore(tmp_path / 'ok_tickets.db')
    await store.initialize()
    interceptor = None
    try:
        interceptor = TaskInterceptor(
            _taskmaster_mock,
            targeted_reconciler=None,
            event_buffer=event_buffer,
            backlog_policy=policy,
            config=config,
            ticket_store=store,
        )
        project_root = str(tmp_path / 'proj_root')
        result = await submit_and_resolve(
            interceptor, project_root,
            title='Under the limit',
            timeout_seconds=5.0,
        )
        assert result == {'id': '2', 'title': 'New'}
        _taskmaster_mock.add_task.assert_called_once()
    finally:
        await store.close()
        if interceptor is not None:
            for _wt in list(interceptor._worker_tasks.values()):
                if not _wt.done():
                    _wt.cancel()
                    with contextlib.suppress(asyncio.CancelledError, Exception):
                        await _wt


# ── Per-project override mechanism (step-1 / step-3) ─────────────────────


class TestBacklogPolicyPerProjectOverride:
    """RED tests for hard_limit_overrides kwarg and hard_limit_for() resolver."""

    @pytest.mark.asyncio
    async def test_hard_limit_for_returns_override_when_present(self, event_buffer):
        """hard_limit_for('reify') returns the override, not the global default."""
        policy = BacklogPolicy(
            event_buffer,
            _StubQueue(),
            lambda _: False,
            hard_limit=10,
            hard_limit_overrides={'reify': 100},
        )
        assert policy.hard_limit_for('reify') == 100

    @pytest.mark.asyncio
    async def test_hard_limit_for_returns_default_when_no_override(self, event_buffer):
        """hard_limit_for('unmapped') falls back to global hard_limit."""
        policy = BacklogPolicy(
            event_buffer,
            _StubQueue(),
            lambda _: False,
            hard_limit=10,
            hard_limit_overrides={'reify': 100},
        )
        assert policy.hard_limit_for('unmapped') == 10

    @pytest.mark.asyncio
    async def test_check_ok_when_backlog_under_override(self, event_buffer, tmp_path):
        """12 events for 'reify' with override=100 → ok (would be over flat 10)."""
        await _seed_buffered(event_buffer, 'reify', n=12)
        project_root = tmp_path / 'reify_root'
        project_root.mkdir()

        policy = BacklogPolicy(
            event_buffer,
            _StubQueue(),
            lambda _: False,
            hard_limit=10,
            hard_limit_overrides={'reify': 100},
        )
        verdict = await policy.check('reify', project_root=str(project_root))
        assert verdict.outcome == 'ok'

    @pytest.mark.asyncio
    async def test_check_rejection_for_unmapped_project_uses_flat_limit(
        self, event_buffer, tmp_path,
    ):
        """12 events for 'small' (no override) with flat=10 → rejection, threshold==10."""
        await _seed_buffered(event_buffer, 'small', n=12)
        project_root = tmp_path / 'small_root'
        project_root.mkdir()

        policy = BacklogPolicy(
            event_buffer,
            _StubQueue(),
            lambda _: False,
            hard_limit=10,
            hard_limit_overrides={'reify': 100},
        )
        verdict = await policy.check('small', project_root=str(project_root))
        assert verdict.outcome == 'rejection'
        assert verdict.threshold == 10

    # ── step-3: per-project effective limit reported in threshold / escalation ──

    @pytest.mark.asyncio
    async def test_escalation_threshold_reflects_override(self, event_buffer, tmp_path):
        """25 events for 'reify' (override=20), orchestrator live → escalation JSON threshold==20."""
        await _seed_buffered(event_buffer, 'reify', n=25)
        project_root = tmp_path / 'reify_root'
        project_root.mkdir()

        policy = BacklogPolicy(
            event_buffer,
            _StubQueue(),
            lambda _: True,  # orchestrator live
            hard_limit=10,
            hard_limit_overrides={'reify': 20},
        )
        verdict = await policy.check('reify', project_root=str(project_root))
        assert verdict.outcome == 'escalated'
        assert verdict.threshold == 20, (
            f'verdict.threshold should be override 20, got {verdict.threshold}'
        )
        assert verdict.escalation_path is not None
        body = json.loads(Path(verdict.escalation_path).read_text())
        assert body['threshold'] == 20, (
            f"escalation JSON threshold should be override 20, got {body['threshold']}"
        )

    @pytest.mark.asyncio
    async def test_rejection_threshold_reflects_override(self, event_buffer, tmp_path):
        """25 events for 'reify' (override=20), no orchestrator → rejection threshold==20."""
        await _seed_buffered(event_buffer, 'reify', n=25)
        project_root = tmp_path / 'reify_root'
        project_root.mkdir()

        policy = BacklogPolicy(
            event_buffer,
            _StubQueue(),
            lambda _: False,  # no orchestrator
            hard_limit=10,
            hard_limit_overrides={'reify': 20},
        )
        verdict = await policy.check('reify', project_root=str(project_root))
        assert verdict.outcome == 'rejection'
        assert verdict.threshold == 20, (
            f'verdict.threshold should be override 20, got {verdict.threshold}'
        )
        err = verdict.to_error_dict()
        assert err['threshold'] == 20
        assert 'limit 20' in err['error']


# ── Deliverable (a): distinct, un-suppressible judge-halt escalation ────────


class TestDistinctLoudHaltEscalation:
    """A judge halt must escalate LOUDLY and DURABLY, never absorbed into
    generic backlog-overflow noise. Two independent defects caused that
    absorption on the base branch: (1) the escalation id was hardcoded
    ``esc-reconciliation-backlog-`` for every kind, so a halt was literally
    filed as 'backlog'; (2) a single shared per-project rate-limit bucket
    meant a backlog escalation inside the 900s window silently suppressed the
    judge-halt escalation. These tests pin the fix: a distinct id prefix AND a
    distinct per-kind rate-limit bucket."""

    @pytest.mark.asyncio
    async def test_halt_escalation_has_distinct_id_prefix(self, event_buffer, tmp_path):
        """on_judge_halt writes an escalation whose id starts
        ``esc-reconciliation-halt-`` (NOT ``-backlog-``), and whose summary
        names both the halt and the reason."""
        project_root = tmp_path / 'proj_root'
        project_root.mkdir()

        policy = BacklogPolicy(
            event_buffer,
            _StubQueue(),
            lambda _: True,  # orchestrator live
            hard_limit=500,
        )
        policy.register_project_root('proj', str(project_root))
        reason = 'Serious verdict in run e87d8e4a'
        verdict = await policy.on_judge_halt('proj', reason=reason)

        assert verdict.outcome == 'escalated'
        assert verdict.escalation_path is not None
        body = json.loads(Path(verdict.escalation_path).read_text())
        assert body['id'].startswith('esc-reconciliation-halt-'), body['id']
        assert not body['id'].startswith('esc-reconciliation-backlog-'), body['id']
        assert 'halt' in body['summary'].lower(), body['summary']
        assert reason in body['summary'], body['summary']

    @pytest.mark.asyncio
    async def test_halt_not_rate_limited_by_backlog_bucket(self, event_buffer, tmp_path):
        """A backlog escalation must NOT suppress a judge-halt escalation raised
        within the same rate-limit window — they use independent per-kind
        buckets, so BOTH files land."""
        await _seed_buffered(event_buffer, 'proj', n=12)
        project_root = tmp_path / 'proj_root'
        project_root.mkdir()
        clock = {'now': 1_000_000.0}

        def now() -> float:
            return clock['now']

        policy = BacklogPolicy(
            event_buffer,
            _StubQueue(),
            lambda _: True,
            hard_limit=10,
            rate_limit_seconds=900.0,
            time_provider=now,
        )
        # 1) backlog escalation (writes an esc-reconciliation-backlog-* file)
        v_backlog = await policy.check('proj', project_root=str(project_root))
        # 2) judge halt WITHIN the same window (no clock advance) — must still write.
        v_halt = await policy.on_judge_halt(
            'proj', reason='Serious verdict in run e87d8e4a',
        )

        assert v_backlog.outcome == 'escalated'
        assert v_halt.outcome == 'escalated'
        assert v_halt.escalation_path is not None
        esc_files = sorted((project_root / 'data' / 'escalations').iterdir())
        assert len(esc_files) == 2, [p.name for p in esc_files]
        ids = [json.loads(p.read_text())['id'] for p in esc_files]
        assert any(i.startswith('esc-reconciliation-halt-') for i in ids), ids
        assert any(i.startswith('esc-reconciliation-backlog-') for i in ids), ids

    @pytest.mark.asyncio
    async def test_backlog_bucket_still_rate_limits(self, event_buffer, tmp_path):
        """Two backlog check()s within the window still collapse to ONE backlog
        file — per-kind buckets must PRESERVE per-project backlog rate-limiting."""
        await _seed_buffered(event_buffer, 'proj', n=12)
        project_root = tmp_path / 'proj_root'
        project_root.mkdir()
        clock = {'now': 1_000_000.0}

        def now() -> float:
            return clock['now']

        policy = BacklogPolicy(
            event_buffer,
            _StubQueue(),
            lambda _: True,
            hard_limit=10,
            rate_limit_seconds=900.0,
            time_provider=now,
        )
        v1 = await policy.check('proj', project_root=str(project_root))
        clock['now'] += 60.0  # within window
        v2 = await policy.check('proj', project_root=str(project_root))

        assert v1.escalation_path is not None
        assert v2.escalation_path is None
        esc_files = list((project_root / 'data' / 'escalations').iterdir())
        assert len(esc_files) == 1


@pytest.mark.asyncio
async def test_backlog_escalation_detail_names_correct_probe(event_buffer, tmp_path):
    """The backlog escalation detail must name the EXACT verification probe a
    watcher uses to confirm a drain (get_queue_stats → reconciliation_backlog)
    and contrast it with the durable-write-queue counts (a different subsystem
    that stays ~0). This is the anti-mis-triage contract — task 2920 (b)."""
    await _seed_buffered(event_buffer, 'proj', n=12)
    project_root = tmp_path / 'proj_root'
    project_root.mkdir()

    policy = BacklogPolicy(
        event_buffer,
        _StubQueue(),
        lambda _: True,  # orchestrator live
        hard_limit=10,
    )
    verdict = await policy.check('proj', project_root=str(project_root))
    assert verdict.outcome == 'escalated'
    assert verdict.escalation_path is not None
    body = json.loads(Path(verdict.escalation_path).read_text())
    detail = body['detail']
    assert 'reconciliation_backlog' in detail, detail
    assert 'get_queue_stats' in detail, detail
    # Explicitly contrasts against the durable-write-queue subsystem.
    assert 'durable' in detail.lower(), detail


# ── on_judge_unhalt: auto-close the halt escalation (task 2998 GAP 3) ──────


def _persisted_record(esc_dir: Path, esc_id: str) -> dict:
    """Read a record from disk after resolve() — root first, then the archive.

    ``resolve()`` MOVES the file into ``<esc_dir>/archive/<date>/``, but leaves
    it in the queue root when the archive move fails, so both are probed.
    """
    root = esc_dir / f'{esc_id}.json'
    if root.exists():
        return json.loads(root.read_text(encoding='utf-8'))
    archived = sorted(esc_dir.glob(f'archive/*/{esc_id}.json'))
    assert archived, f'{esc_id} not found under {esc_dir} or its archive'
    return json.loads(archived[-1].read_text(encoding='utf-8'))


class TestOnJudgeUnhalt:
    """Clearing a halt must close the escalation the halt opened.

    Before task 2998 the halt record was written by BacklogPolicy but nothing
    ever closed it: ``unhalt_reconciliation`` cleared the judge's in-memory +
    journal state only, leaving an ``esc-reconciliation-halt-*.json`` pending
    forever.
    """

    async def _setup(self, event_buffer, tmp_path):
        """Write one halt + one backlog escalation for 'proj', plus a
        foreign-project halt record in the SAME directory."""
        project_root = tmp_path / 'proj_root'
        project_root.mkdir()

        policy = BacklogPolicy(
            event_buffer, _StubQueue(), lambda _: True, hard_limit=10,
        )
        policy.register_project_root('proj', str(project_root))

        halt_verdict = await policy.on_judge_halt('proj', reason='seed halt')
        assert halt_verdict.escalation_path is not None
        halt_id = Path(halt_verdict.escalation_path).stem

        await _seed_buffered(event_buffer, 'proj', n=12)
        backlog_verdict = await policy.check('proj')
        assert backlog_verdict.outcome == 'escalated'
        assert backlog_verdict.escalation_path is not None
        backlog_id = Path(backlog_verdict.escalation_path).stem

        esc_dir = project_root / 'data' / 'escalations'
        other_id = 'esc-reconciliation-halt-other-project'
        (esc_dir / f'{other_id}.json').write_text(
            json.dumps({
                'id': other_id,
                'task_id': None,
                'agent_role': 'fused-memory',
                'severity': 'blocking',
                'category': 'infra_issue',
                'summary': 'Reconciliation HALTED for other',
                'detail': 'halt belonging to a DIFFERENT project',
                'suggested_action': 'inspect_judge_halt',
                'timestamp': '2026-07-28T00:00:00+00:00',
                'status': 'pending',
                'level': 1,
                'workflow_state': 'infra',
                'project_id': 'other',
                'error_type': 'ReconciliationJudgeHalted',
            }, indent=2),
            encoding='utf-8',
        )
        return policy, esc_dir, halt_id, backlog_id, other_id

    @pytest.mark.asyncio
    async def test_resolves_only_this_projects_halt_escalation(
        self, event_buffer, tmp_path,
    ):
        from escalation.queue import EscalationQueue

        policy, esc_dir, halt_id, backlog_id, other_id = await self._setup(
            event_buffer, tmp_path,
        )

        resolved = await policy.on_judge_unhalt('proj')
        assert resolved == [halt_id]

        # resolve() ARCHIVES the file — probe via get(), not path.exists().
        queue = EscalationQueue(esc_dir)
        closed = queue.get(halt_id)
        assert closed is not None
        assert closed.status == 'resolved'
        assert closed.resolution
        assert closed.resolved_by is not None
        assert 'unhalt' in closed.resolved_by

        # The backlog escalation and the foreign-project halt stay pending.
        still_pending = {e.id for e in queue.get_pending()}
        assert backlog_id in still_pending
        assert other_id in still_pending
        assert halt_id not in still_pending

        # The CLOSED record must still identify its project and fault kind.
        # resolve() persists Escalation.to_json(), and from_dict keeps only
        # dataclass fields, so without a re-merge these four policy-owned keys
        # are destroyed on close — and the forensic query that diagnosed this
        # incident ('which escalation files carry ReconciliationJudgeHalted')
        # stops working against every auto-closed record.
        persisted = _persisted_record(esc_dir, halt_id)
        assert persisted['project_id'] == 'proj'
        assert persisted['error_type'] == 'ReconciliationJudgeHalted'
        assert 'backlog' in persisted
        assert 'threshold' in persisted
        assert persisted['status'] == 'resolved'

    @pytest.mark.asyncio
    async def test_second_unhalt_is_idempotent(self, event_buffer, tmp_path):
        policy, _esc_dir, halt_id, _backlog_id, _other = await self._setup(
            event_buffer, tmp_path,
        )

        assert await policy.on_judge_unhalt('proj') == [halt_id]
        assert await policy.on_judge_unhalt('proj') == []

    @pytest.mark.asyncio
    async def test_resolve_no_op_is_not_reported_as_closed(
        self, event_buffer, tmp_path, caplog,
    ):
        """A resolve() that did NOT close anything must not be reported closed.

        ``queue.resolve()`` returns None when the id cannot be located — e.g. a
        record whose ``id`` key disagrees with its filename, since ``get()``
        derives the path from the id. Appending unconditionally makes the MCP
        tool tell the operator 'Auto-resolved 1 pending halt escalation(s)' for
        a record that stays pending forever: the exact silent rot this feature
        exists to remove, merely relabelled as success.
        """
        logger_name = 'fused_memory.reconciliation.backlog_policy'
        project_root = tmp_path / 'proj_root'
        esc_dir = project_root / 'data' / 'escalations'
        esc_dir.mkdir(parents=True)

        policy = BacklogPolicy(event_buffer, _StubQueue(), lambda _: True)
        policy.register_project_root('proj', str(project_root))

        # id key deliberately disagrees with the filename stem.
        path = esc_dir / 'esc-reconciliation-halt-ONDISK.json'
        path.write_text(
            json.dumps({
                'id': 'esc-reconciliation-halt-DIFFERENT',
                'task_id': None,
                'agent_role': 'fused-memory',
                'severity': 'blocking',
                'category': 'infra_issue',
                'summary': 'Reconciliation HALTED for proj',
                'detail': 'halt whose id does not match its filename',
                'suggested_action': 'inspect_judge_halt',
                'timestamp': '2026-07-28T00:00:00+00:00',
                'status': 'pending',
                'level': 1,
                'workflow_state': 'infra',
                'project_id': 'proj',
                'error_type': 'ReconciliationJudgeHalted',
            }, indent=2),
            encoding='utf-8',
        )

        with caplog.at_level(logging.WARNING, logger=logger_name):
            assert await policy.on_judge_unhalt('proj') == []

        text = '\n'.join(
            r.getMessage() for r in caplog.records if r.name == logger_name
        )
        assert 'no-op' in text
        assert 'esc-reconciliation-halt-DIFFERENT' in text
        # And the record on disk is still, in fact, pending.
        assert json.loads(path.read_text())['status'] == 'pending'

    @pytest.mark.asyncio
    async def test_unregistered_project_returns_empty_and_logs(
        self, event_buffer, caplog,
    ):
        logger_name = 'fused_memory.reconciliation.backlog_policy'
        policy = BacklogPolicy(event_buffer, _StubQueue(), lambda _: True)

        with caplog.at_level(logging.INFO, logger=logger_name):
            assert await policy.on_judge_unhalt('nope') == []

        text = '\n'.join(
            r.getMessage() for r in caplog.records if r.name == logger_name
        )
        assert 'nope' in text
