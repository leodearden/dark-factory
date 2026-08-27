"""Tests for Harness._dismiss_stale_escalations() — auto-dismiss stale escalations on startup."""

from __future__ import annotations

import logging
from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from _recording_event_store import _RecordingEventStore
from escalation.models import Escalation
from escalation.queue import EscalationQueue

from orchestrator.event_store import EventType
from orchestrator.harness import Harness

# The origin incident (task 3172): esc-5189-7 had been pending 20h58m with a
# workflow parked on it; esc-5685-1 had been pending ~90s.  The restart sweep
# closed both with one fixed string and one 'benign' class, so the strand was
# indistinguishable from ordinary restart noise afterwards.
STRAND_AGE_SECS = 75480.0  # 20h58m
FRESH_AGE_SECS = 90.0
THRESHOLD_SECS = 600.0


def _queue_of(harness: Harness) -> EscalationQueue:
    """Narrow ``Harness._escalation_queue`` (typed ``EscalationQueue | None``).

    The ``strand_harness`` fixture always wires a real queue, so this states an
    existing precondition for the type checker rather than weakening a check.
    """
    queue = harness._escalation_queue
    assert queue is not None, 'strand_harness must be wired with a real EscalationQueue'
    return queue


def _get(queue: EscalationQueue, esc_id: str) -> Escalation:
    """Narrow ``EscalationQueue.get`` (``Escalation | None``) at a seeded id."""
    esc = queue.get(esc_id)
    assert esc is not None, f'escalation {esc_id} missing from the queue'
    return esc


def _seed_escalation(
    queue: EscalationQueue,
    esc_id: str,
    task_id: str,
    *,
    age_secs: float,
    level: int = 0,
    severity: str = 'blocking',
) -> Escalation:
    """Submit a pending escalation stamped *age_secs* in the past."""
    esc = Escalation(
        id=esc_id,
        task_id=task_id,
        agent_role='implementer',
        severity=severity,
        category='task_failure',
        summary=f'stranded escalation for task {task_id}',
        level=level,
    )
    esc.timestamp = (datetime.now(UTC) - timedelta(seconds=age_secs)).isoformat()
    queue.submit(esc)
    return esc


def _strand_events(harness: Harness) -> list[dict]:
    """The recorded stale_l0_strand_dismissed emits, newest-last."""
    return [
        payload
        for name, payload in harness.event_store.events  # type: ignore[union-attr]
        if name == str(EventType.stale_l0_strand_dismissed)
    ]


@pytest.fixture
def harness(tmp_path: Path, mock_orch_config) -> Harness:
    """Create a Harness with mocked internals for unit testing stale escalation cleanup."""
    with (
        patch('orchestrator.harness.McpLifecycle'),
        patch('orchestrator.harness.OverrideStore'),
        patch('orchestrator.harness.Scheduler'),
        patch('orchestrator.harness.BriefingAssembler'),
    ):
        h = Harness(mock_orch_config)

    # Replace scheduler with async mocks
    h.scheduler = MagicMock()
    h.scheduler.get_tasks = AsyncMock(return_value=[])
    h.scheduler.set_task_status = AsyncMock()
    h.scheduler.get_statuses = AsyncMock(return_value=({}, None))

    # Neutralise merge-worker startup so no real SpeculativeMergeWorker
    # background task is spawned inside a unit test.  Mirrors the _neutralise
    # idiom in test_harness_park_stop.py.
    h._start_merge_worker = AsyncMock()
    # The periodic main-tip sweep (task 1832) and no-landings circuit-breaker
    # (task θ/1893) loops are neutralised at the config level — mock_orch_config
    # sets main_tip_sweep_enabled/no_landings_breaker_enabled to False, so
    # _build_lifecycle_registry() never registers their BackgroundService and
    # run() spawns no real asyncio.Task for either (task 2241, W10-η).

    return h


@pytest.mark.asyncio
class TestDismissStaleEscalations:
    """Harness._dismiss_stale_escalations() auto-dismisses pending escalations."""

    async def test_no_queue_is_noop(self, harness: Harness):
        """When _escalation_queue is None (no escalation support), method is a no-op."""
        harness._escalation_queue = None
        # Should not raise; count should be 0 (nothing happens)
        await harness._dismiss_stale_escalations()
        # No assertion needed beyond "no exception raised"

    async def test_has_escalation_false_is_noop(self, harness: Harness, caplog):
        """When HAS_ESCALATION is False, method is a no-op with no side effects."""
        harness._escalation_queue = None
        with patch('orchestrator.harness.HAS_ESCALATION', False), caplog.at_level(logging.INFO):
            await harness._dismiss_stale_escalations()

        assert 'dismissed' not in caplog.text.lower()

    async def test_empty_queue_is_noop(self, harness: Harness, caplog):
        """When queue has no pending escalations, method does nothing."""
        mock_queue = MagicMock()
        mock_queue.dismiss_all_pending.return_value = 0
        harness._escalation_queue = mock_queue

        with caplog.at_level(logging.INFO):
            await harness._dismiss_stale_escalations()

        mock_queue.dismiss_all_pending.assert_called_once()
        # No "dismissed N" log line for zero count
        assert 'dismissed 0' not in caplog.text.lower()

    async def test_pending_escalations_dismissed(self, harness: Harness, caplog):
        """Pending escalations are all dismissed with the correct resolution message."""
        mock_queue = MagicMock()
        mock_queue.dismiss_all_pending.return_value = 3
        harness._escalation_queue = mock_queue

        with caplog.at_level(logging.INFO):
            await harness._dismiss_stale_escalations()

        mock_queue.dismiss_all_pending.assert_called_once()
        call_args = mock_queue.dismiss_all_pending.call_args
        resolution_msg = call_args[0][0]
        assert 'stale' in resolution_msg.lower() or 'prior' in resolution_msg.lower()

    async def test_strand_age_threshold_anchored_to_orphan_l0_timeout(self, harness: Harness):
        """The strand threshold reuses the operator-tuned orphan_l0_timeout_secs.

        This is an ANCHOR, not a new guessed constant (task 3172): the repo
        already ships an operator-tuned "an unattended L0 this old is overdue"
        bound, and the sweep borrows its VALUE rather than minting a second
        knob that would need its own reload tier.
        """
        mock_queue = MagicMock()
        mock_queue.dismiss_all_pending.return_value = 0
        harness._escalation_queue = mock_queue

        await harness._dismiss_stale_escalations()

        kwargs = mock_queue.dismiss_all_pending.call_args.kwargs
        assert kwargs['strand_age_secs'] == harness.config.orphan_l0_timeout_secs
        assert kwargs['strand_age_secs'] is not None

    async def test_dismissal_count_logged(self, harness: Harness, caplog):
        """When escalations are dismissed, count is logged at INFO level."""
        mock_queue = MagicMock()
        mock_queue.dismiss_all_pending.return_value = 5
        harness._escalation_queue = mock_queue

        with caplog.at_level(logging.INFO):
            await harness._dismiss_stale_escalations()

        assert '5' in caplog.text

    # Pin to a single xdist worker: this test hard-codes localhost:9999 on
    # harness.mcp.url; concurrent siblings sharing the same port would race.
    @pytest.mark.xdist_group('fixed_mcp_port')
    async def test_called_after_start_escalation_server_before_task_loop(
        self, harness: Harness, tmp_path: Path
    ):
        """_dismiss_stale_escalations is called after _start_escalation_server
        but before _recover_crashed_tasks in Harness.run()."""
        call_order: list[str] = []

        async def mock_mcp_start():
            pass

        async def mock_mcp_stop():
            pass

        async def mock_start_escalation_server():
            call_order.append('start_escalation_server')

        async def mock_dismiss_stale_escalations():
            call_order.append('dismiss_stale_escalations')

        async def mock_recover_crashed_tasks():
            call_order.append('recover_crashed_tasks')

        # Mock the PRD path
        prd_path = tmp_path / 'test.prd'
        prd_path.write_text('# Test PRD')

        harness.mcp = MagicMock()
        harness.mcp.start = AsyncMock(side_effect=mock_mcp_start)
        harness.mcp.stop = AsyncMock(side_effect=mock_mcp_stop)
        harness.mcp.url = 'http://localhost:9999'

        harness._start_escalation_server = AsyncMock(
            side_effect=mock_start_escalation_server
        )
        harness._dismiss_stale_escalations = AsyncMock(
            side_effect=mock_dismiss_stale_escalations
        )
        harness._recover_crashed_tasks = AsyncMock(
            side_effect=mock_recover_crashed_tasks
        )
        harness._stop_escalation_server = AsyncMock()
        harness._tag_prd_metadata = AsyncMock()
        harness._tag_task_modules = AsyncMock()
        harness.scheduler.get_tasks = AsyncMock(return_value=[])
        # Seed a pending task so run() has work to count and reaches the
        # dry-run stop (task 1563 removed the 'No pending tasks found' guard;
        # any tree, even empty, now passes through to dry_run=True return).
        harness.scheduler.get_statuses = AsyncMock(
            return_value=({'1': 'pending'}, None),
        )

        # Run with dry_run to avoid the task execution loop
        await harness.run(prd_path, dry_run=True)

        # Verify ordering: start_escalation_server → dismiss_stale_escalations
        # (recover_crashed_tasks may not be in dry_run path, but server startup is)
        assert 'start_escalation_server' in call_order
        assert 'dismiss_stale_escalations' in call_order

        server_idx = call_order.index('start_escalation_server')
        dismiss_idx = call_order.index('dismiss_stale_escalations')
        assert dismiss_idx > server_idx, (
            f'dismiss_stale_escalations ({dismiss_idx}) must come after '
            f'start_escalation_server ({server_idx})'
        )

        # Also verify dismiss comes before recover_crashed_tasks when that is called
        if 'recover_crashed_tasks' in call_order:
            recover_idx = call_order.index('recover_crashed_tasks')
            assert dismiss_idx < recover_idx, (
                f'dismiss_stale_escalations ({dismiss_idx}) must come before '
                f'recover_crashed_tasks ({recover_idx})'
            )


@pytest.mark.asyncio
class TestDismissStaleEscalationsFatal:
    """_dismiss_stale_escalations() failure must not prevent harness cleanup."""

    @pytest.mark.xdist_group('fixed_mcp_port')
    async def test_dismiss_failure_does_not_prevent_finally(
        self, harness: Harness, tmp_path: Path
    ):
        """If _dismiss_stale_escalations() raises, the finally block still runs.

        Specifically: _stop_escalation_server() and mcp.stop() must be called
        even when _dismiss_stale_escalations() raises an OSError.

        With the task-1563 guard removed, an empty task tree ({}, None) no
        longer raises RuntimeError.  run() now completes cleanly via dry_run=True.
        The finally block runs regardless.
        """
        prd_path = tmp_path / 'test.prd'
        prd_path.write_text('# Test PRD')

        harness.mcp = MagicMock()
        harness.mcp.start = AsyncMock()
        harness.mcp.stop = AsyncMock()
        harness.mcp.url = 'http://localhost:9999'

        harness._start_escalation_server = AsyncMock()
        harness._dismiss_stale_escalations = AsyncMock(
            side_effect=OSError('disk full simulated failure')
        )
        harness._stop_escalation_server = AsyncMock()
        # Mock the startup steps that are now reached (guard removed,
        # empty tree proceeds through 2b-2e before the dry-run return).
        harness._tag_prd_metadata = AsyncMock()
        harness._tag_task_modules = AsyncMock()
        harness._recover_crashed_tasks = AsyncMock()
        harness._reconcile_stranded_in_progress = AsyncMock()

        # run() catches the OSError in its dismiss-stale-escalations try/except,
        # then proceeds through the startup steps and exits cleanly via dry_run=True.
        # A re-raise (not a clean return) would fail this test — that's intentional.
        await harness.run(prd_path, dry_run=True)

        # Finally block must have run
        harness._stop_escalation_server.assert_called_once()
        harness.mcp.stop.assert_called_once()

    @pytest.mark.xdist_group('fixed_mcp_port')
    async def test_dismiss_failure_logged_as_warning(
        self, harness: Harness, tmp_path: Path, caplog
    ):
        """If _dismiss_stale_escalations() raises, the exception is caught and logged,
        not re-raised as an unhandled exception that aborts the entire run."""
        prd_path = tmp_path / 'test.prd'
        prd_path.write_text('# Test PRD')

        harness.mcp = MagicMock()
        harness.mcp.start = AsyncMock()
        harness.mcp.stop = AsyncMock()
        harness.mcp.url = 'http://localhost:9999'

        harness._start_escalation_server = AsyncMock()
        harness._stop_escalation_server = AsyncMock()
        harness._tag_prd_metadata = AsyncMock()
        harness._tag_task_modules = AsyncMock()
        harness._recover_crashed_tasks = AsyncMock()
        # Seed a pending task so run() has work to count and reaches the
        # dry-run stop (task 1563 removed the 'No pending tasks found' guard;
        # any tree, even empty, now passes through to dry_run=True return).
        harness.scheduler.get_statuses = AsyncMock(
            return_value=({'1': 'pending'}, None),
        )

        error_msg = 'disk full simulated failure'
        harness._dismiss_stale_escalations = AsyncMock(
            side_effect=OSError(error_msg)
        )

        # Run should complete without re-raising (dry_run stops after task population)
        with caplog.at_level(logging.WARNING):
            # If the exception propagates, run() would raise. If it's caught
            # and logged, run() should complete normally.
            try:
                await harness.run(prd_path, dry_run=True)
                # If we get here, the exception was caught — good
                run_completed = True
            except OSError:
                run_completed = False

        assert run_completed, (
            '_dismiss_stale_escalations() exception should be caught and logged, '
            'not re-raised to abort the run'
        )

        # The warning should appear in logs
        assert error_msg in caplog.text or 'dismiss' in caplog.text.lower()


@pytest.fixture
def strand_harness(harness: Harness, tmp_path: Path) -> Harness:
    """Harness wired to a REAL EscalationQueue plus a recording event store."""
    harness._escalation_queue = EscalationQueue(tmp_path / 'strand_queue')
    harness.event_store = _RecordingEventStore()  # type: ignore[assignment]
    harness.config.orphan_l0_timeout_secs = THRESHOLD_SECS
    return harness


@pytest.mark.asyncio
class TestStaleL0StrandIsLoud:
    """A level-0 stranded across a restart is recorded as a strand, not as noise.

    ACCEPTANCE for task 3172 ASK A + ASK B: the sweep must (a) leave the strand
    and the fresh restart artifact distinguishable in the durable record, and
    (b) say so out loud rather than silently flattening both into 'benign'.
    """

    async def test_both_pending_l0s_are_still_dismissed(self, strand_harness: Harness):
        """Loudness must not cost the sweep its actual job."""
        queue = _queue_of(strand_harness)
        _seed_escalation(queue, 'esc-5189-7', '5189', age_secs=STRAND_AGE_SECS)
        _seed_escalation(queue, 'esc-5685-1', '5685', age_secs=FRESH_AGE_SECS)

        await strand_harness._dismiss_stale_escalations()

        assert _get(queue, 'esc-5189-7').status == 'dismissed'
        assert _get(queue, 'esc-5685-1').status == 'dismissed'

    async def test_strand_and_fresh_record_are_distinguishable_afterwards(
        self, strand_harness: Harness
    ):
        """The exact flattening the origin incident exposed is gone."""
        queue = _queue_of(strand_harness)
        _seed_escalation(queue, 'esc-5189-7', '5189', age_secs=STRAND_AGE_SECS)
        _seed_escalation(queue, 'esc-5685-1', '5685', age_secs=FRESH_AGE_SECS)

        await strand_harness._dismiss_stale_escalations()

        strand = _get(queue, 'esc-5189-7')
        fresh = _get(queue, 'esc-5685-1')
        assert strand.resolution_class == 'stale-strand'
        assert fresh.resolution_class == 'benign'
        assert strand.resolution is not None and fresh.resolution is not None
        assert 'pending_secs=' in strand.resolution
        assert 'pending_secs=' in fresh.resolution

    async def test_exactly_one_strand_event_keyed_on_the_real_task_id(
        self, strand_harness: Harness
    ):
        """One event per strand, keyed so it joins against task_completed."""
        queue = _queue_of(strand_harness)
        _seed_escalation(queue, 'esc-5189-7', '5189', age_secs=STRAND_AGE_SECS)
        _seed_escalation(queue, 'esc-5685-1', '5685', age_secs=FRESH_AGE_SECS)

        await strand_harness._dismiss_stale_escalations()

        events = _strand_events(strand_harness)
        assert len(events) == 1
        assert events[0]['task_id'] == '5189'
        assert events[0]['data']['escalation_id'] == 'esc-5189-7'

    async def test_strand_event_payload_carries_age_and_blocked_ness(
        self, strand_harness: Harness
    ):
        """The payload states how long it waited and that a workflow was on it."""
        queue = _queue_of(strand_harness)
        _seed_escalation(queue, 'esc-5189-7', '5189', age_secs=STRAND_AGE_SECS)

        await strand_harness._dismiss_stale_escalations()

        data = _strand_events(strand_harness)[0]['data']
        assert abs(data['pending_secs'] - STRAND_AGE_SECS) < 5
        assert data['workflow_blocked'] is True
        assert data['severity'] == 'blocking'
        assert data['resolution_class'] == 'stale-strand'

    async def test_fresh_l0_produces_no_strand_event(self, strand_harness: Harness):
        """An ordinary restart artifact stays ordinary — no strand telemetry."""
        queue = _queue_of(strand_harness)
        _seed_escalation(queue, 'esc-5685-1', '5685', age_secs=FRESH_AGE_SECS)

        await strand_harness._dismiss_stale_escalations()

        assert _strand_events(strand_harness) == []

    async def test_info_severity_strand_reports_workflow_not_blocked(
        self, strand_harness: Harness
    ):
        """Blocked-ness is DERIVED from the durable severity, never guessed.

        escalate_info files severity='info' and the agent keeps going;
        escalate_blocker files severity='blocking' and the filing workflow
        genuinely parks.  After a restart _escalation_events is empty, so the
        record's own severity is the only surviving signal.
        """
        queue = _queue_of(strand_harness)
        _seed_escalation(
            queue, 'esc-4242-1', '4242', age_secs=STRAND_AGE_SECS, severity='info'
        )

        await strand_harness._dismiss_stale_escalations()

        events = _strand_events(strand_harness)
        assert len(events) == 1
        assert events[0]['data']['workflow_blocked'] is False
        assert events[0]['data']['severity'] == 'info'

    async def test_strand_sweep_logs_at_warning(self, strand_harness: Harness, caplog):
        """journald is loud too — a swept strand is not an INFO-level footnote."""
        queue = _queue_of(strand_harness)
        _seed_escalation(queue, 'esc-5189-7', '5189', age_secs=STRAND_AGE_SECS)

        with caplog.at_level(logging.WARNING):
            await strand_harness._dismiss_stale_escalations()

        warnings = [r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING]
        assert any('strand' in m.lower() for m in warnings), warnings

    async def test_telemetry_failure_never_aborts_the_dismissal(
        self, strand_harness: Harness
    ):
        """An observability failure must not cost the sweep its primary action."""
        queue = _queue_of(strand_harness)
        _seed_escalation(queue, 'esc-5189-7', '5189', age_secs=STRAND_AGE_SECS)
        strand_harness.event_store.emit = MagicMock(  # type: ignore[union-attr]
            side_effect=RuntimeError('event store exploded')
        )

        await strand_harness._dismiss_stale_escalations()

        assert _get(queue, 'esc-5189-7').status == 'dismissed'


def _aggregate_escalations(harness: Harness) -> list:
    """Pending strand-sweep aggregate escalations currently in the queue."""
    return [
        e
        for e in harness._escalation_queue.get_pending()  # type: ignore[union-attr]
        if e.agent_role == 'harness-stale-l0-strand-sweep'
    ]


@pytest.mark.asyncio
class TestStaleL0StrandEscalationSurvivesRestart:
    """The strand record must outlive the restart that destroyed the evidence.

    Level is LOAD-BEARING (task 3172): dismiss_all_pending sweeps every pending
    L0 at startup, so an aggregate filed at L0 would be erased by the very next
    restart — reproducing exactly the evidence destruction this task exists to
    stop.  L1 is explicitly preserved across restart.
    """

    async def test_one_aggregate_escalation_for_several_strands(
        self, strand_harness: Harness
    ):
        """Two strands plus a fresh L0 file exactly ONE aggregate, not three."""
        queue = _queue_of(strand_harness)
        _seed_escalation(queue, 'esc-5189-7', '5189', age_secs=STRAND_AGE_SECS)
        _seed_escalation(queue, 'esc-5190-1', '5190', age_secs=STRAND_AGE_SECS + 600)
        _seed_escalation(queue, 'esc-5685-1', '5685', age_secs=FRESH_AGE_SECS)

        await strand_harness._dismiss_stale_escalations()

        assert len(_aggregate_escalations(strand_harness)) == 1

    async def test_aggregate_is_filed_at_level_1(self, strand_harness: Harness):
        """L0 would be swept by the next restart's own dismissal; L1 survives."""
        queue = _queue_of(strand_harness)
        _seed_escalation(queue, 'esc-5189-7', '5189', age_secs=STRAND_AGE_SECS)

        await strand_harness._dismiss_stale_escalations()

        agg = _aggregate_escalations(strand_harness)[0]
        assert agg.level == 1
        assert agg.status == 'pending'

    async def test_aggregate_survives_a_subsequent_restart_sweep(
        self, strand_harness: Harness
    ):
        """The whole point: a second restart must not erase the strand record."""
        queue = _queue_of(strand_harness)
        _seed_escalation(queue, 'esc-5189-7', '5189', age_secs=STRAND_AGE_SECS)
        await strand_harness._dismiss_stale_escalations()
        agg_id = _aggregate_escalations(strand_harness)[0].id

        await strand_harness._dismiss_stale_escalations()  # the next restart

        assert _get(queue, agg_id).status == 'pending'

    async def test_aggregate_detail_names_every_strand(self, strand_harness: Harness):
        """Per-strand detail is not lost to aggregation."""
        queue = _queue_of(strand_harness)
        _seed_escalation(queue, 'esc-5189-7', '5189', age_secs=STRAND_AGE_SECS)
        _seed_escalation(queue, 'esc-5190-1', '5190', age_secs=STRAND_AGE_SECS + 600)

        await strand_harness._dismiss_stale_escalations()

        detail = _aggregate_escalations(strand_harness)[0].detail
        assert 'esc-5189-7' in detail
        assert 'esc-5190-1' in detail
        assert '5189' in detail and '5190' in detail
        assert '75480' in detail  # the pending age, in seconds

    async def test_second_sweep_with_open_aggregate_files_no_duplicate(
        self, strand_harness: Harness
    ):
        """Anti-storm dedup: one OPEN aggregate at a time, however many strands."""
        queue = _queue_of(strand_harness)
        _seed_escalation(queue, 'esc-5189-7', '5189', age_secs=STRAND_AGE_SECS)
        await strand_harness._dismiss_stale_escalations()

        _seed_escalation(queue, 'esc-7777-1', '7777', age_secs=STRAND_AGE_SECS)
        await strand_harness._dismiss_stale_escalations()

        assert len(_aggregate_escalations(strand_harness)) == 1

    async def test_zero_strands_files_nothing(self, strand_harness: Harness):
        """An ordinary restart that strands nothing stays completely quiet."""
        queue = _queue_of(strand_harness)
        _seed_escalation(queue, 'esc-5685-1', '5685', age_secs=FRESH_AGE_SECS)

        await strand_harness._dismiss_stale_escalations()

        assert _aggregate_escalations(strand_harness) == []

    async def test_submit_failure_does_not_abort_startup(
        self, strand_harness: Harness, caplog
    ):
        """A failure to file the aggregate is logged, never raised into startup."""
        queue = _queue_of(strand_harness)
        _seed_escalation(queue, 'esc-5189-7', '5189', age_secs=STRAND_AGE_SECS)
        with (
            patch.object(queue, 'submit', side_effect=OSError('disk full')),
            caplog.at_level(logging.WARNING),
        ):
            await strand_harness._dismiss_stale_escalations()  # must not raise

        assert _get(queue, 'esc-5189-7').status == 'dismissed'
        assert any(
            r.levelno >= logging.WARNING for r in caplog.records
        ), 'the filing failure must be logged'
