"""Tests for reconciliation harness (pipeline orchestration)."""

import contextlib
import uuid
from datetime import UTC, datetime
from unittest.mock import AsyncMock

import pytest
import pytest_asyncio

from fused_memory.models.reconciliation import (
    EventSource,
    EventType,
    ReconciliationEvent,
    ReconciliationRun,
    RunStatus,
    RunType,
    StageReport,
)
from fused_memory.reconciliation.event_buffer import EventBuffer
from fused_memory.reconciliation.journal import ReconciliationJournal


@pytest_asyncio.fixture
async def journal(tmp_path):
    j = ReconciliationJournal(tmp_path / 'harness_test')
    await j.initialize()
    yield j
    await j.close()


@pytest_asyncio.fixture
async def event_buffer(tmp_path):
    buf = EventBuffer(
        db_path=tmp_path / 'harness_eb.db', buffer_size_threshold=2, max_staleness_seconds=3600
    )
    await buf.initialize()
    yield buf
    await buf.close()


@pytest.fixture
def mock_memory_service():
    svc = AsyncMock()
    svc.search = AsyncMock(return_value=[])
    svc.get_episodes = AsyncMock(return_value=[])
    svc.get_status = AsyncMock(
        return_value={'graphiti': {'connected': True}, 'mem0': {'connected': True}, 'projects': {}}
    )
    svc.get_entity = AsyncMock(return_value={'nodes': [], 'edges': []})
    svc.mem0 = AsyncMock()
    svc.mem0.get_all = AsyncMock(return_value={'results': []})
    return svc


def _make_event(project_id: str = 'test-project') -> ReconciliationEvent:
    return ReconciliationEvent(
        id=str(uuid.uuid4()),
        type=EventType.episode_added,
        source=EventSource.agent,
        project_id=project_id,
        timestamp=datetime.now(UTC),
        payload={},
    )


@pytest.mark.asyncio
async def test_event_buffer_trigger_starts_pipeline(journal, event_buffer, mock_memory_service):
    """When buffer triggers, the pipeline should run."""
    # Push enough events to trigger
    for _ in range(3):
        await event_buffer.push(_make_event())

    should, reason = await event_buffer.should_trigger('test-project')
    assert should


@pytest.mark.asyncio
async def test_drain_clears_buffer(event_buffer):
    """Drain should atomically clear the buffer."""
    await event_buffer.push(_make_event())
    await event_buffer.push(_make_event())

    events = await event_buffer.drain('test-project')
    assert len(events) == 2

    # Should be empty now
    assert (await event_buffer.get_buffer_stats('test-project'))['size'] == 0


@pytest.mark.asyncio
async def test_active_run_prevents_trigger(event_buffer):
    """Active run should prevent trigger."""
    for _ in range(3):
        await event_buffer.push(_make_event())

    await event_buffer.mark_run_active('test-project')
    should, _ = await event_buffer.should_trigger('test-project')
    assert not should


@pytest.mark.asyncio
async def test_journal_run_lifecycle(journal):
    """Test run start, complete, and query."""
    from fused_memory.models.reconciliation import ReconciliationRun

    run = ReconciliationRun(
        id=str(uuid.uuid4()),
        project_id='test-project',
        run_type=RunType.full,
        trigger_reason='buffer_size:3',
        started_at=datetime.now(UTC),
        events_processed=3,
        status=RunStatus.running,
    )
    await journal.start_run(run)
    assert await journal.is_run_active('test-project')

    await journal.complete_run(run.id, 'completed')
    assert not await journal.is_run_active('test-project')

    loaded = await journal.get_run(run.id)
    assert loaded.status == 'completed'


@pytest.mark.asyncio
async def test_run_full_cycle_restores_events_on_failure(
    journal, event_buffer, mock_memory_service
):
    """Failed stage should restore drained events to buffered."""
    from fused_memory.config.schema import FusedMemoryConfig, ReconciliationConfig
    from fused_memory.reconciliation.harness import ReconciliationHarness

    config = FusedMemoryConfig(
        reconciliation=ReconciliationConfig(
            enabled=True,
            explore_codebase_root='/tmp/test',
            agent_llm_provider='anthropic',
            agent_llm_model='claude-sonnet-4-20250514',
        )
    )

    await event_buffer.push(_make_event())
    await event_buffer.push(_make_event())

    harness = ReconciliationHarness(
        memory_service=mock_memory_service,
        taskmaster=AsyncMock(),
        journal=journal,
        event_buffer=event_buffer,
        config=config,
    )
    harness._make_stages = lambda: harness.stages

    # Make first stage raise
    harness.stages[0].run = AsyncMock(side_effect=RuntimeError('stage exploded'))

    with pytest.raises(RuntimeError, match='stage exploded'):
        await harness.run_full_cycle('test-project', 'buffer_size:2')

    # Events should be restored to buffered
    stats = await event_buffer.get_buffer_stats('test-project')
    assert stats['size'] == 2


# ── Tests for harness extracting project_root from events (step-9) ────


def _make_event_with_root(
    project_id: str = 'dark_factory',
    project_root: str = '/home/leo/src/dark-factory',
) -> ReconciliationEvent:
    return ReconciliationEvent(
        id=str(uuid.uuid4()),
        type=EventType.task_status_changed,
        source=EventSource.agent,
        project_id=project_id,
        timestamp=datetime.now(UTC),
        payload={'_project_root': project_root, 'task_id': '1'},
    )


@pytest.mark.asyncio
async def test_full_cycle_extracts_project_root_from_events(
    journal, event_buffer, mock_memory_service
):
    """Harness should set both stage.project_id and stage.project_root from drained events."""
    from unittest.mock import AsyncMock

    from fused_memory.config.schema import FusedMemoryConfig, ReconciliationConfig
    from fused_memory.models.reconciliation import StageReport

    config = FusedMemoryConfig(
        reconciliation=ReconciliationConfig(
            enabled=True,
            explore_codebase_root='/tmp/test',
            agent_llm_provider='anthropic',
            agent_llm_model='claude-sonnet-4-20250514',
        )
    )

    # Push events with _project_root in payload
    await event_buffer.push(_make_event_with_root())
    await event_buffer.push(_make_event_with_root())

    from fused_memory.reconciliation.harness import ReconciliationHarness

    harness = ReconciliationHarness(
        memory_service=mock_memory_service,
        taskmaster=AsyncMock(),
        journal=journal,
        event_buffer=event_buffer,
        config=config,
    )
    harness._make_stages = lambda: harness.stages

    # Mock each stage's run method and capture the stage state
    captured_stages = []
    for stage in harness.stages:
        original_stage = stage

        async def mock_run(events, watermark, prior_reports, run_id, model=None, _s=original_stage):
            # Capture state at call time
            captured_stages.append(
                {
                    'project_id': _s.project_id,
                    'project_root': _s.project_root,
                }
            )
            return StageReport(
                stage=_s.stage_id,
                started_at=datetime.now(UTC),
                completed_at=datetime.now(UTC),
                items_flagged=[],
                stats={},
                llm_calls=0,
                tokens_used=0,
            )

        stage.run = mock_run

    await harness.run_full_cycle('dark_factory', 'buffer_size:2')

    assert len(captured_stages) == 3
    for stage_state in captured_stages:
        assert stage_state['project_id'] == 'dark_factory'
        assert stage_state['project_root'] == '/home/leo/src/dark-factory'


# ── Tests for Task 74: Stage 3 findings feedback loop ────────────────


def _make_s3_findings():
    """Return a mix of actionable and non-actionable Stage 3 findings."""
    return [
        {
            'description': 'Stale edge: uses_framework→React on project_alpha',
            'severity': 'moderate',
            'actionable': True,
            'category': 'memory_stale',
            'affected_ids': ['edge-abc123', 'project_alpha'],
            'suggested_action': 'Delete stale edge',
        },
        {
            'description': 'Contradictory edges on deploy_target',
            'severity': 'serious',
            'actionable': True,
            'category': 'memory_contradiction',
            'affected_ids': ['edge-def456'],
            'suggested_action': 'Delete older contradictory edge',
        },
        {
            'description': 'Systemic pattern: growing divergence between stores',
            'severity': 'moderate',
            'actionable': False,
            'category': 'systemic_pattern',
            'affected_ids': [],
            'suggested_action': 'Requires human review of store sync strategy',
        },
    ]


def _make_test_harness(journal, event_buffer, mock_memory_service):
    """Build a ReconciliationHarness wired to test fixtures with minimal config.

    Callers must mock individual stage.run methods as needed.
    """
    from fused_memory.config.schema import FusedMemoryConfig, ReconciliationConfig
    from fused_memory.reconciliation.harness import ReconciliationHarness

    config = FusedMemoryConfig(
        reconciliation=ReconciliationConfig(
            enabled=True,
            explore_codebase_root='/tmp/test',
            agent_llm_provider='anthropic',
            agent_llm_model='claude-sonnet-4-20250514',
        )
    )

    harness = ReconciliationHarness(
        memory_service=mock_memory_service,
        taskmaster=AsyncMock(),
        journal=journal,
        event_buffer=event_buffer,
        config=config,
    )
    # Patch _make_stages so tests that mock harness.stages[N].run still work
    harness._make_stages = lambda: harness.stages
    return harness


def _mock_stage_run(stage, items_flagged=None, before_return=None, capture_call_args=None):
    """Replace stage.run with a mock that returns a StageReport.

    Args:
        stage: The stage whose .run method will be replaced.
        items_flagged: Optional list of findings to include in the StageReport.
        before_return: Optional async callable invoked with the stage object just
            before the StageReport is returned.  Use this to capture mutable stage
            state (e.g. episode_limit, memory_limit) at the moment .run() fires.
        capture_call_args: Optional dict.  If provided, the model kwarg passed to
            .run() is stored as capture_call_args['model'] so callers can assert
            that the correct model was forwarded by the harness.
    """

    async def mock_run(events, watermark, prior_reports, run_id, model=None, _s=stage):
        if capture_call_args is not None:
            capture_call_args['model'] = model
        if before_return is not None:
            await before_return(_s)
        return StageReport(
            stage=_s.stage_id,
            started_at=datetime.now(UTC),
            completed_at=datetime.now(UTC),
            items_flagged=items_flagged or [],
            stats={},
            llm_calls=0,
            tokens_used=0,
        )

    stage.run = mock_run


@pytest.mark.asyncio
async def test_mock_stage_run_before_return_callback(journal, event_buffer, mock_memory_service):
    """_mock_stage_run must invoke an optional async before_return callback with the stage."""
    harness = _make_test_harness(journal, event_buffer, mock_memory_service)
    stage = harness.stages[0]

    callback_args: list = []

    async def capture(s):
        callback_args.append(s)

    _mock_stage_run(stage, before_return=capture)

    from fused_memory.models.reconciliation import Watermark

    watermark = Watermark(project_id='test-project')
    await stage.run([], watermark, [], 'test-run-id')

    assert len(callback_args) == 1, (
        f'Expected before_return callback to be called once, got {len(callback_args)}'
    )
    assert callback_args[0] is stage, (
        'Expected before_return callback to receive the stage object as argument'
    )


@pytest.mark.asyncio
async def test_finding_partition_actionable_vs_non_actionable():
    """Partition logic: actionable findings trigger remediation, non-actionable get escalated."""
    findings = _make_s3_findings()
    actionable = [f for f in findings if f.get('actionable', False)]
    non_actionable = [f for f in findings if not f.get('actionable', False)]
    assert len(actionable) == 2
    assert len(non_actionable) == 1
    assert non_actionable[0]['category'] == 'systemic_pattern'


@pytest.mark.asyncio
async def test_remediation_payload_assembly():
    """MemoryConsolidator produces findings-only payload in remediation mode."""
    from fused_memory.reconciliation.stages.memory_consolidator import MemoryConsolidator

    stage = MemoryConsolidator.__new__(MemoryConsolidator)
    stage.project_id = 'test-project'
    stage.remediation_findings = _make_s3_findings()[:2]  # actionable only
    stage.prior_s3_findings = None

    payload = stage._assemble_remediation_payload()
    assert 'Remediation Run' in payload
    assert 'Targeted Memory Fixes' in payload
    assert 'Stale edge' in payload
    assert 'Contradictory edges' in payload
    assert 'Do NOT perform general consolidation' in payload


@pytest.mark.asyncio
async def test_normal_payload_includes_prior_s3_findings(mock_memory_service):
    """Normal assemble_payload includes prior S3 findings section when set."""
    from fused_memory.models.reconciliation import StageId, Watermark
    from fused_memory.reconciliation.stages.memory_consolidator import MemoryConsolidator

    stage = MemoryConsolidator(
        StageId.memory_consolidator,
        mock_memory_service,
        None,
        AsyncMock(),
        AsyncMock(),
    )
    stage.project_id = 'test-project'
    stage.episode_limit = 500
    stage.memory_limit = 1000
    stage.prior_s3_findings = [_make_s3_findings()[0]]

    watermark = Watermark(project_id='test-project')
    payload = await stage.assemble_payload([], watermark, [])

    assert 'Prior Stage 3 Findings' in payload
    assert 'Stale edge' in payload


@pytest.mark.asyncio
async def test_run_full_cycle_triggers_remediation(journal, event_buffer, mock_memory_service):
    """Full cycle with S3 actionable findings triggers a remediation pass."""
    harness = _make_test_harness(journal, event_buffer, mock_memory_service)

    await event_buffer.push(_make_event())
    await event_buffer.push(_make_event())

    s3_findings = _make_s3_findings()

    # Mock stages: S1 and S2 return empty reports, S3 returns findings
    _mock_stage_run(harness.stages[0])
    _mock_stage_run(harness.stages[1])
    _mock_stage_run(harness.stages[2], items_flagged=s3_findings)

    run = await harness.run_full_cycle('test-project', 'buffer_size:2')

    assert run.status == 'completed'

    # Verify remediation run was created
    recent_runs = await journal.get_recent_runs('test-project', limit=5)
    assert len(recent_runs) == 2  # parent + remediation

    remediation_run = next(r for r in recent_runs if r.run_type == 'remediation')
    assert remediation_run.triggered_by == run.id
    assert remediation_run.events_processed == 0
    assert remediation_run.status == 'completed'


@pytest.mark.asyncio
async def test_remediation_does_not_run_without_actionable_findings(
    journal,
    event_buffer,
    mock_memory_service,
):
    """No remediation pass when S3 has only non-actionable findings."""
    harness = _make_test_harness(journal, event_buffer, mock_memory_service)

    await event_buffer.push(_make_event())

    non_actionable_only = [
        {
            'description': 'Needs human review',
            'severity': 'moderate',
            'actionable': False,
            'category': 'systemic_pattern',
            'affected_ids': [],
            'suggested_action': 'Human review needed',
        },
    ]

    _mock_stage_run(harness.stages[0])
    _mock_stage_run(harness.stages[1])
    _mock_stage_run(harness.stages[2], items_flagged=non_actionable_only)

    await harness.run_full_cycle('test-project', 'buffer_size:1')

    recent_runs = await journal.get_recent_runs('test-project', limit=5)
    assert len(recent_runs) == 1  # Only the parent run, no remediation


@pytest.mark.asyncio
async def test_remediation_failure_does_not_fail_parent(
    journal,
    event_buffer,
    mock_memory_service,
):
    """If remediation pass fails, parent run remains completed."""
    harness = _make_test_harness(journal, event_buffer, mock_memory_service)

    await event_buffer.push(_make_event())

    s3_findings = _make_s3_findings()

    # Track call count to distinguish parent vs remediation stages
    call_count = {'s1': 0}

    async def s1_run_that_fails_on_second(events, watermark, prior_reports, run_id, model=None):
        call_count['s1'] += 1
        if call_count['s1'] == 2:  # Remediation pass
            raise RuntimeError('remediation exploded')
        return StageReport(
            stage=harness.stages[0].stage_id,
            started_at=datetime.now(UTC),
            completed_at=datetime.now(UTC),
            items_flagged=[],
            stats={},
            llm_calls=0,
            tokens_used=0,
        )

    harness.stages[0].run = s1_run_that_fails_on_second
    _mock_stage_run(harness.stages[1])
    _mock_stage_run(harness.stages[2], items_flagged=s3_findings)

    # Should NOT raise — remediation failure is swallowed
    run = await harness.run_full_cycle('test-project', 'buffer_size:1')
    assert run.status == 'completed'

    # Remediation run should be marked failed
    recent_runs = await journal.get_recent_runs('test-project', limit=5)
    remediation_run = next((r for r in recent_runs if r.run_type == 'remediation'), None)
    assert remediation_run is not None
    assert remediation_run.status == 'failed'


@pytest.mark.asyncio
async def test_journal_triggered_by_roundtrip(journal):
    """triggered_by field persists through start_run/get_run/get_recent_runs."""
    parent_id = str(uuid.uuid4())
    child_id = str(uuid.uuid4())

    parent_run = ReconciliationRun(
        id=parent_id,
        project_id='test-project',
        run_type=RunType.full,
        trigger_reason='buffer_size:5',
        started_at=datetime.now(UTC),
        events_processed=5,
        status=RunStatus.completed,
    )
    await journal.start_run(parent_run)

    child_run = ReconciliationRun(
        id=child_id,
        project_id='test-project',
        run_type=RunType.remediation,
        trigger_reason='integrity_findings:2',
        started_at=datetime.now(UTC),
        events_processed=0,
        status=RunStatus.running,
        triggered_by=parent_id,
    )
    await journal.start_run(child_run)

    loaded = await journal.get_run(child_id)
    assert loaded is not None
    assert loaded.triggered_by == parent_id

    recent = await journal.get_recent_runs('test-project', limit=5)
    child_from_recent = next(r for r in recent if r.id == child_id)
    assert child_from_recent.triggered_by == parent_id

    parent_from_recent = next(r for r in recent if r.id == parent_id)
    assert parent_from_recent.triggered_by is None


@pytest.mark.asyncio
async def test_timeout_marks_run_failed(journal, event_buffer, mock_memory_service):
    """When asyncio.wait_for cancels run_full_cycle on timeout, the run must be marked 'failed'.

    Bug 5: asyncio.wait_for timeout cancels run_full_cycle via asyncio.CancelledError.
    CancelledError is NOT caught by 'except Exception', so complete_run(run_id, 'failed')
    is never called, leaving the run stuck in 'running'.

    This test confirms that after a timeout:
    - The journal run has status 'failed' (not 'running')
    - The buffer events were restored (buffer size == original event count)
    """
    import asyncio

    harness = _make_test_harness(journal, event_buffer, mock_memory_service)

    # Make first stage sleep forever (simulating a long-running stage)
    async def slow_stage_run(
        events, watermark, prior_reports, run_id, model=None, _s=harness.stages[0]
    ):
        await asyncio.sleep(999)  # Will be cancelled by wait_for
        return StageReport(
            stage=_s.stage_id,
            started_at=datetime.now(UTC),
            completed_at=datetime.now(UTC),
            items_flagged=[],
            stats={},
            llm_calls=0,
            tokens_used=0,
        )

    harness.stages[0].run = slow_stage_run
    _mock_stage_run(harness.stages[1])
    _mock_stage_run(harness.stages[2])

    # Push events so drain returns something
    await event_buffer.push(_make_event())
    await event_buffer.push(_make_event())

    # Run with a tight timeout to force cancellation
    with pytest.raises((TimeoutError, asyncio.TimeoutError)):
        await asyncio.wait_for(
            harness.run_full_cycle('test-project', 'buffer_size:2'),
            timeout=0.1,
        )

    # Give the event loop a moment to process any cleanup
    await asyncio.sleep(0.05)

    # The run must be marked 'failed', not stuck in 'running'
    recent_runs = await journal.get_recent_runs('test-project', limit=5)
    assert len(recent_runs) >= 1
    run = recent_runs[0]
    assert run.status == 'failed', (
        f"Expected run.status='failed' after timeout, got '{run.status}'. "
        'Bug 5: CancelledError is not caught in run_full_cycle, so complete_run is never called.'
    )

    # Events must have been restored to the buffer
    stats = await event_buffer.get_buffer_stats('test-project')
    assert stats['size'] == 2, (
        f'Expected buffer size=2 after timeout, got {stats["size"]}. '
        'Bug 5: restore_drained is not called on CancelledError.'
    )


@pytest.mark.asyncio
async def test_run_full_cycle_accepts_pre_drained_events(
    journal, event_buffer, mock_memory_service
):
    """run_full_cycle() must accept an optional 'events' param to skip drain().

    Bug 4: BacklogIterator.run() drains a chunk via drain_oldest_chunk(), then calls
    run_full_cycle() which re-drains via drain(), getting different events — the chunk
    events are silently lost.  Fix: add optional events param to run_full_cycle so
    BacklogIterator can pass the already-drained chunk.

    This test confirms that passing events=[...] to run_full_cycle uses those events
    without calling buffer.drain(), and that events_processed reflects the passed count.
    """
    harness = _make_test_harness(journal, event_buffer, mock_memory_service)

    # Mock all stages to succeed
    for stage in harness.stages:
        _mock_stage_run(stage)

    # Do NOT push any events to the buffer (drain() would return 0 events)
    # Manually create 2 events
    evt1 = _make_event()
    evt2 = _make_event()

    # Call run_full_cycle with pre-drained events
    # This should fail currently because run_full_cycle doesn't accept an 'events' param
    run = await harness.run_full_cycle('test-project', 'backlog_chunk:1:2', events=[evt1, evt2])

    assert run.events_processed == 2, (
        f'Expected events_processed=2 from pre-drained events, got {run.events_processed}. '
        "Bug 4: run_full_cycle does not accept an 'events' parameter."
    )
    assert run.status == 'completed'


@pytest.mark.asyncio
async def test_halted_project_skips_cycle(journal, event_buffer, mock_memory_service):
    """run_loop() must skip run_full_cycle for projects that are halted by the judge.

    Bug 3: judge.is_halted() is never called in run_loop, so halted projects keep
    processing new cycles.  This test drives one run_loop iteration via a short
    asyncio.wait_for and confirms run_full_cycle is never called for a halted project.
    """
    import asyncio
    from unittest.mock import AsyncMock, patch

    harness = _make_test_harness(journal, event_buffer, mock_memory_service)

    # Wire a real judge with the project pre-halted
    from fused_memory.config.schema import ReconciliationConfig
    from fused_memory.reconciliation.judge import Judge

    judge_config = ReconciliationConfig(
        enabled=True,
        explore_codebase_root='/tmp/test',
        agent_llm_provider='anthropic',
        agent_llm_model='claude-sonnet-4-20250514',
    )
    mock_j = AsyncMock()
    mock_j.get_run = AsyncMock(return_value=None)
    harness.judge = Judge(config=judge_config, journal=mock_j)
    harness.judge._halted_projects.add('test-project')  # Pre-halt the project

    # Push enough events to trigger a cycle
    for _ in range(3):
        await event_buffer.push(_make_event())

    # Confirm trigger would fire
    should, _ = await event_buffer.should_trigger('test-project')
    assert should

    # Track whether run_full_cycle is called
    run_full_cycle_called = []
    original_rfc = harness.run_full_cycle

    async def spy_rfc(*args, **kwargs):
        run_full_cycle_called.append(args)
        return await original_rfc(*args, **kwargs)

    # Also make _recover_stale_runs and escalation server no-ops
    harness._recover_stale_runs = AsyncMock(return_value=None)
    harness._start_escalation_server = AsyncMock()
    harness._stop_escalation_server = AsyncMock()

    with (
        patch.object(harness, 'run_full_cycle', side_effect=spy_rfc),
        contextlib.suppress(TimeoutError),
    ):
        # Run loop for one sleep cycle (loop sleeps 5s; we wait 0.2s — enough for 1 iteration)
        await asyncio.wait_for(harness.run_loop(), timeout=0.2)

    # For a halted project, run_full_cycle must NOT have been called
    assert len(run_full_cycle_called) == 0, (
        f'run_full_cycle was called {len(run_full_cycle_called)} time(s) '
        'for a halted project — Bug 3: halt check not wired into run_loop.'
    )


@pytest.mark.asyncio
async def test_judge_unhalt_clears_halt_escalated(journal, event_buffer, mock_memory_service):
    """Judge.unhalt must clear harness._halt_escalated so the next halt re-fires.

    Without this wiring, a manual unhalt followed by another halt wouldn't
    produce a fresh escalation because _notify_judge_halt dedupes per-process.
    """
    harness = _make_test_harness(journal, event_buffer, mock_memory_service)
    assert harness.judge is not None

    # Simulate a prior halt that's been escalated once
    harness._halt_escalated.add('test-project')
    await harness.judge._apply_halt('test-project', reason='seed')
    assert harness.judge.is_halted('test-project')

    await harness.judge.unhalt('test-project')

    assert not harness.judge.is_halted('test-project')
    # Harness callback must have fired and cleared the sentinel
    assert 'test-project' not in harness._halt_escalated


@pytest.mark.asyncio
async def test_project_loop_consumes_unhalt_grace(journal, event_buffer, mock_memory_service):
    """_project_loop decrements post-unhalt grace before running a cycle."""
    import asyncio
    from unittest.mock import AsyncMock, patch

    harness = _make_test_harness(journal, event_buffer, mock_memory_service)
    assert harness.judge is not None

    # Seed grace counter (simulates a prior halt + unhalt with halt_grace_cycles>0)
    harness.judge._unhalt_grace_remaining['test-project'] = 2
    # The journal mock should return decremented values
    harness.journal.decrement_unhalt_grace = AsyncMock(return_value=1)

    for _ in range(3):
        await event_buffer.push(_make_event())

    # Short-circuit run_full_cycle so the loop returns quickly
    async def fake_rfc(*_a, **_k):
        from fused_memory.models.reconciliation import ReconciliationRun, RunStatus, RunType
        return ReconciliationRun(
            id=str(uuid.uuid4()),
            project_id='test-project',
            run_type=RunType.full,
            trigger_reason='buffer_size:3',
            started_at=datetime.now(UTC),
            events_processed=3,
            status=RunStatus.completed,
        )

    harness._recover_stale_runs = AsyncMock(return_value=None)
    harness._start_escalation_server = AsyncMock()
    harness._stop_escalation_server = AsyncMock()

    with (
        patch.object(harness, 'run_full_cycle', side_effect=fake_rfc),
        contextlib.suppress(TimeoutError),
    ):
        await asyncio.wait_for(harness.run_loop(), timeout=0.5)

    harness.journal.decrement_unhalt_grace.assert_awaited()
    assert harness.judge.unhalt_grace_remaining('test-project') == 1


@pytest.mark.asyncio
async def test_make_stages_returns_clean_instances(journal, event_buffer, mock_memory_service):
    """_make_stages() returns fresh stage instances with no leftover per-run state.

    Previously, shared stages needed explicit cleanup after remediation.  Now each
    run_full_cycle and _run_remediation_pass creates its own stages via _make_stages(),
    so no cleanup is needed — fresh instances are always clean.
    """
    from fused_memory.config.schema import FusedMemoryConfig, ReconciliationConfig
    from fused_memory.reconciliation.harness import ReconciliationHarness
    from fused_memory.reconciliation.stages.memory_consolidator import MemoryConsolidator
    from fused_memory.reconciliation.stages.task_knowledge_sync import TaskKnowledgeSync

    config = FusedMemoryConfig(
        reconciliation=ReconciliationConfig(
            enabled=True,
            explore_codebase_root='/tmp/test',
            agent_llm_provider='anthropic',
            agent_llm_model='claude-sonnet-4-20250514',
        )
    )
    harness = ReconciliationHarness(
        memory_service=mock_memory_service,
        taskmaster=AsyncMock(),
        journal=journal,
        event_buffer=event_buffer,
        config=config,
    )

    stages = harness._make_stages()
    stage1 = stages[0]
    stage2 = stages[1]
    assert isinstance(stage1, MemoryConsolidator)
    assert isinstance(stage2, TaskKnowledgeSync)
    assert stage1.remediation_findings is None
    assert stage1.prior_s3_findings is None
    assert stage1.cycle_fence_time is None
    assert stage1.assembled_payload is None
    assert stage2.remediation_mode is False


@pytest.mark.asyncio
async def test_cancellation_cleanup_failure_preserves_cancelled_error(
    journal,
    event_buffer,
    mock_memory_service,
):
    """Cleanup exception during CancelledError handler must not replace CancelledError.

    Review issue [exception_swallowing]: If journal.complete_run raises RuntimeError
    during the CancelledError cleanup, the exception currently replaces CancelledError
    before the re-raise — so the caller receives RuntimeError instead of TimeoutError,
    bypassing run_loop's timeout handler.

    Fix: wrap cleanup awaits in a nested try/except Exception so CancelledError is
    always re-raised regardless of cleanup failures.
    """
    import asyncio

    harness = _make_test_harness(journal, event_buffer, mock_memory_service)

    # Make first stage sleep forever (will be cancelled by wait_for timeout)
    async def slow_stage_run(
        events,
        watermark,
        prior_reports,
        run_id,
        model=None,
        _s=harness.stages[0],
    ):
        await asyncio.sleep(999)
        return StageReport(
            stage=_s.stage_id,
            started_at=datetime.now(UTC),
            completed_at=datetime.now(UTC),
            items_flagged=[],
            stats={},
            llm_calls=0,
            tokens_used=0,
        )

    harness.stages[0].run = slow_stage_run
    _mock_stage_run(harness.stages[1])
    _mock_stage_run(harness.stages[2])

    await event_buffer.push(_make_event())

    # Mock complete_run to raise RuntimeError when called during cleanup (status='failed')
    original_complete_run = journal.complete_run

    async def failing_complete_run(run_id, status):
        if status == 'failed':
            raise RuntimeError('DB connection lost')
        return await original_complete_run(run_id, status)

    journal.complete_run = failing_complete_run

    # The caller must receive TimeoutError, NOT RuntimeError from the cleanup failure.
    # Before the fix, RuntimeError replaces CancelledError and propagates to the caller.
    with pytest.raises((TimeoutError, asyncio.TimeoutError)):
        await asyncio.wait_for(
            harness.run_full_cycle('test-project', 'buffer_size:1'),
            timeout=0.1,
        )


@pytest.mark.asyncio
async def test_cancellation_cleanup_shielded_from_second_cancel(
    journal,
    event_buffer,
    mock_memory_service,
):
    """asyncio.shield() must protect cleanup from a second cancellation during shutdown.

    Review issue [async_cancellation_safety]: Without asyncio.shield(), a second
    cancellation (e.g., during server shutdown) arriving while complete_run is
    awaiting the DB write will interrupt complete_run — leaving the journal stuck
    in 'running'.  With asyncio.shield(), complete_run runs in its own Task and
    finishes even if the outer task is cancelled again.

    This test injects the second cancel FROM WITHIN complete_run by calling
    outer_task.cancel() before awaiting asyncio.sleep(0).  Without shield,
    complete_run runs inside the outer task so asyncio.sleep(0) raises
    CancelledError (second cancel fires), aborting the write.  With shield,
    complete_run runs in a separate inner Task unaffected by the outer cancel,
    so sleep(0) returns normally and the write completes.
    """
    import asyncio

    harness = _make_test_harness(journal, event_buffer, mock_memory_service)

    # Event set by slow_stage_run when it starts — ensures the first cancel fires
    # inside the try block (not during pre-try setup like _get_prior_s3_findings).
    stage_entered = asyncio.Event()

    # Make first stage sleep forever (cancelled by the first outer_task.cancel())
    async def slow_stage_run(
        events,
        watermark,
        prior_reports,
        run_id,
        model=None,
        _s=harness.stages[0],
    ):
        stage_entered.set()
        await asyncio.sleep(999)
        return StageReport(
            stage=_s.stage_id,
            started_at=datetime.now(UTC),
            completed_at=datetime.now(UTC),
            items_flagged=[],
            stats={},
            llm_calls=0,
            tokens_used=0,
        )

    harness.stages[0].run = slow_stage_run
    _mock_stage_run(harness.stages[1])
    _mock_stage_run(harness.stages[2])

    await event_buffer.push(_make_event())

    # Capture the outer task reference so the mock can cancel it from within
    outer_task_ref: list = [None]
    original_complete_run = journal.complete_run

    async def self_cancelling_complete_run(run_id, status):
        if status == 'failed':
            # Simulate a second external cancellation (e.g., server shutdown)
            # arriving while cleanup is in progress.
            outer_task_ref[0].cancel()
            # Without asyncio.shield: this await runs in the outer task context,
            # so the pending cancel fires here — CancelledError aborts the write.
            # With asyncio.shield: this runs in its own inner Task, so the cancel
            # on the outer task does not propagate here and sleep(0) completes.
            await asyncio.sleep(0)
        await original_complete_run(run_id, status)

    journal.complete_run = self_cancelling_complete_run

    outer_task = asyncio.create_task(
        harness.run_full_cycle('test-project', 'buffer_size:1'),
    )
    outer_task_ref[0] = outer_task

    # Wait until slow_stage_run has actually started (deterministic — no sleep-based race).
    # Using a fixed sleep could misfire on a loaded CI host: if _get_prior_s3_findings
    # takes longer than the sleep, the cancel arrives before the try block and
    # complete_run is never called, leaving the run stuck in 'running'.
    # Race the event against outer_task to avoid infinite hang if run_full_cycle
    # fails before slow_stage_run is invoked (e.g. journal/buffer setup error).
    done, _ = await asyncio.wait(
        [asyncio.ensure_future(stage_entered.wait()), outer_task],
        return_when=asyncio.FIRST_COMPLETED,
    )
    if outer_task in done and not stage_entered.is_set():
        exc = 'task was cancelled' if outer_task.cancelled() else repr(outer_task.exception())
        pytest.fail(f'outer_task completed before slow_stage_run was invoked: {exc}')

    # First cancellation: triggers CancelledError in slow_stage_run → cleanup starts
    outer_task.cancel()

    # Wait for the outer task to finish (it will raise CancelledError)
    with contextlib.suppress(asyncio.CancelledError):
        await outer_task

    # Give any shield-wrapped inner Task time to complete the DB write.
    # Without shield: no inner task exists; with shield: inner task runs here.
    await asyncio.sleep(0.1)

    # The journal run must be 'failed', not stuck in 'running'
    recent_runs = await journal.get_recent_runs('test-project', limit=5)
    assert len(recent_runs) >= 1
    run = recent_runs[0]
    assert run.status == 'failed', (
        f"Expected status='failed' after double cancellation, got '{run.status}'. "
        'Review issue [async_cancellation_safety]: cleanup must be wrapped with '
        'asyncio.shield() so a second cancel cannot abort the DB write.'
    )


# ── Tests for _select_tier ────────────────────────────────────────────────────


class TestSelectTier:
    """ReconciliationHarness._select_tier returns correct TierConfig based on buffer size."""

    @pytest.mark.asyncio
    async def test_select_tier_sonnet(self, journal, event_buffer, mock_memory_service):
        """Buffer below opus threshold returns sonnet TierConfig."""
        from fused_memory.config.schema import FusedMemoryConfig, ReconciliationConfig
        from fused_memory.reconciliation.harness import ReconciliationHarness, TierConfig

        config = FusedMemoryConfig(
            reconciliation=ReconciliationConfig(
                enabled=True,
                explore_codebase_root='/tmp/test',
                agent_llm_provider='anthropic',
                agent_llm_model='claude-sonnet-4-20250514',
            )
        )
        # Compute threshold from config so the test survives default changes
        opus_threshold = (
            config.reconciliation.buffer_size_threshold * config.reconciliation.opus_threshold_ratio
        )

        harness = ReconciliationHarness(
            memory_service=mock_memory_service,
            taskmaster=AsyncMock(),
            journal=journal,
            event_buffer=event_buffer,
            config=config,
        )
        # Buffer size well below opus_threshold
        harness.buffer.get_buffer_stats = AsyncMock(
            return_value={'size': int(opus_threshold) - 10, 'oldest_event_age_seconds': None}
        )

        tier = await harness._select_tier('test-project')

        harness.buffer.get_buffer_stats.assert_called_once_with('test-project')
        assert isinstance(tier, TierConfig)
        assert tier.model == 'sonnet'
        assert tier.episode_limit == 125
        assert tier.memory_limit == 250

    @pytest.mark.asyncio
    async def test_select_tier_opus(self, journal, event_buffer, mock_memory_service):
        """Buffer above opus threshold returns opus TierConfig."""
        from fused_memory.config.schema import FusedMemoryConfig, ReconciliationConfig
        from fused_memory.reconciliation.harness import ReconciliationHarness, TierConfig

        config = FusedMemoryConfig(
            reconciliation=ReconciliationConfig(
                enabled=True,
                explore_codebase_root='/tmp/test',
                agent_llm_provider='anthropic',
                agent_llm_model='claude-sonnet-4-20250514',
            )
        )
        # Compute threshold from config so the test survives default changes
        opus_threshold = (
            config.reconciliation.buffer_size_threshold * config.reconciliation.opus_threshold_ratio
        )

        harness = ReconciliationHarness(
            memory_service=mock_memory_service,
            taskmaster=AsyncMock(),
            journal=journal,
            event_buffer=event_buffer,
            config=config,
        )
        # Buffer size clearly above opus_threshold
        harness.buffer.get_buffer_stats = AsyncMock(
            return_value={'size': int(opus_threshold) + 5, 'oldest_event_age_seconds': 60.0}
        )

        tier = await harness._select_tier('test-project')

        harness.buffer.get_buffer_stats.assert_called_once_with('test-project')
        assert isinstance(tier, TierConfig)
        assert tier.model == 'opus'
        assert tier.episode_limit == 500
        assert tier.memory_limit == 1000

    @pytest.mark.asyncio
    async def test_select_tier_boundary(self, journal, event_buffer, mock_memory_service):
        """Buffer size exactly at threshold returns sonnet — condition is strictly greater-than."""
        from fused_memory.config.schema import FusedMemoryConfig, ReconciliationConfig
        from fused_memory.reconciliation.harness import ReconciliationHarness, TierConfig

        config = FusedMemoryConfig(
            reconciliation=ReconciliationConfig(
                enabled=True,
                explore_codebase_root='/tmp/test',
                agent_llm_provider='anthropic',
                agent_llm_model='claude-sonnet-4-20250514',
            )
        )
        # Compute threshold from config so the test survives default changes
        opus_threshold = (
            config.reconciliation.buffer_size_threshold * config.reconciliation.opus_threshold_ratio
        )

        harness = ReconciliationHarness(
            memory_service=mock_memory_service,
            taskmaster=AsyncMock(),
            journal=journal,
            event_buffer=event_buffer,
            config=config,
        )
        # Buffer size exactly at opus_threshold — NOT above (strictly >, not >=)
        harness.buffer.get_buffer_stats = AsyncMock(
            return_value={'size': int(opus_threshold), 'oldest_event_age_seconds': None}
        )

        tier = await harness._select_tier('test-project')

        harness.buffer.get_buffer_stats.assert_called_once_with('test-project')
        assert isinstance(tier, TierConfig)
        assert tier.model == 'sonnet', (
            f'size==opus_threshold ({int(opus_threshold)}) should return sonnet '
            '(condition is strictly >); if this fails, the boundary condition was changed to >='
        )
        assert tier.episode_limit == 125
        assert tier.memory_limit == 250

    @pytest.mark.asyncio
    async def test_select_tier_custom_config(self, journal, event_buffer, mock_memory_service):
        """Non-default config with buffer_size_threshold=20 and opus_threshold_ratio=2.0 (threshold=40)."""
        from fused_memory.config.schema import FusedMemoryConfig, ReconciliationConfig
        from fused_memory.reconciliation.harness import ReconciliationHarness, TierConfig

        config = FusedMemoryConfig(
            reconciliation=ReconciliationConfig(
                enabled=True,
                explore_codebase_root='/tmp/test',
                agent_llm_provider='anthropic',
                agent_llm_model='claude-sonnet-4-20250514',
                buffer_size_threshold=20,
                opus_threshold_ratio=2.0,
                # opus_threshold = 20 * 2.0 = 40
            )
        )
        opus_threshold = (
            config.reconciliation.buffer_size_threshold * config.reconciliation.opus_threshold_ratio
        )
        assert opus_threshold == 40, f'Expected opus_threshold=40, got {opus_threshold}'

        harness = ReconciliationHarness(
            memory_service=mock_memory_service,
            taskmaster=AsyncMock(),
            journal=journal,
            event_buffer=event_buffer,
            config=config,
        )

        # Sub-case (a): buffer size 30 is below threshold (40) → sonnet
        harness.buffer.get_buffer_stats = AsyncMock(
            return_value={'size': 30, 'oldest_event_age_seconds': None}
        )
        tier_a = await harness._select_tier('test-project')

        harness.buffer.get_buffer_stats.assert_called_once_with('test-project')
        assert isinstance(tier_a, TierConfig)
        assert tier_a.model == 'sonnet', (
            f'size=30 < opus_threshold=40 should return sonnet, got {tier_a.model}'
        )
        assert tier_a.episode_limit == 125
        assert tier_a.memory_limit == 250

        # Sub-case (b): buffer size 50 is above threshold (40) → opus
        harness.buffer.get_buffer_stats = AsyncMock(
            return_value={'size': 50, 'oldest_event_age_seconds': 30.0}
        )
        tier_b = await harness._select_tier('test-project')

        harness.buffer.get_buffer_stats.assert_called_once_with('test-project')
        assert isinstance(tier_b, TierConfig)
        assert tier_b.model == 'opus', (
            f'size=50 > opus_threshold=40 should return opus, got {tier_b.model}'
        )
        assert tier_b.episode_limit == 500
        assert tier_b.memory_limit == 1000

    @pytest.mark.asyncio
    async def test_run_full_cycle_propagates_tier_to_consolidator(
        self,
        journal,
        event_buffer,
        mock_memory_service,
    ):
        """run_full_cycle applies TierConfig limits onto MemoryConsolidator before stage runs."""
        from fused_memory.reconciliation.harness import TierConfig
        from fused_memory.reconciliation.stages.memory_consolidator import MemoryConsolidator

        harness = _make_test_harness(journal, event_buffer, mock_memory_service)

        assert any(isinstance(s, MemoryConsolidator) for s in harness.stages), (
            'MemoryConsolidator not found in harness.stages — stage ordering changed?'
        )

        # Capture the limits as seen by MemoryConsolidator when its run() is invoked
        captured: dict = {}

        async def capture_limits(stage):
            captured['episode_limit'] = stage.episode_limit
            captured['memory_limit'] = stage.memory_limit

        for stage in harness.stages:
            if isinstance(stage, MemoryConsolidator):
                _mock_stage_run(stage, before_return=capture_limits, capture_call_args=captured)
            else:
                _mock_stage_run(stage)

        tier = TierConfig(model='sonnet', episode_limit=125, memory_limit=250)
        await harness.run_full_cycle(
            'test-project',
            'tier-propagation-test',
            tier=tier,
            events=[_make_event()],
        )

        assert captured, (
            'MemoryConsolidator.run() was never invoked — '
            'run_full_cycle skipped it or stage list changed'
        )

        assert captured.get('episode_limit') == 125, (
            f'Expected episode_limit=125 propagated to consolidator, got {captured.get("episode_limit")}'
        )
        assert captured.get('memory_limit') == 250, (
            f'Expected memory_limit=250 propagated to consolidator, got {captured.get("memory_limit")}'
        )
        assert captured.get('model') == 'sonnet', (
            f"Expected model='sonnet' forwarded as kwarg to stage.run(), got {captured.get('model')!r}"
        )

    @pytest.mark.asyncio
    async def test_run_full_cycle_does_not_mutate_non_consolidator_stages(
        self,
        journal,
        event_buffer,
        mock_memory_service,
    ):
        """isinstance guard ensures episode_limit/memory_limit are NOT set on Stage 2 and 3."""
        from fused_memory.reconciliation.harness import TierConfig
        from fused_memory.reconciliation.stages.memory_consolidator import MemoryConsolidator

        harness = _make_test_harness(journal, event_buffer, mock_memory_service)

        # Capture hasattr state at the moment each non-consolidator stage.run() fires
        non_consolidator_hasattr: dict[str, dict] = {}

        for stage in harness.stages:
            if isinstance(stage, MemoryConsolidator):
                _mock_stage_run(stage)
            else:
                stage_name = type(stage).__name__

                async def capture_hasattr(s, _name=stage_name):
                    non_consolidator_hasattr[_name] = {
                        'episode_limit': hasattr(s, 'episode_limit'),
                        'memory_limit': hasattr(s, 'memory_limit'),
                    }

                _mock_stage_run(stage, before_return=capture_hasattr)

        tier = TierConfig(model='sonnet', episode_limit=125, memory_limit=250)
        await harness.run_full_cycle(
            'test-project',
            'isinstance-guard-test',
            tier=tier,
            events=[_make_event()],
        )

        assert non_consolidator_hasattr, (
            'No non-MemoryConsolidator stages were invoked — stage list changed?'
        )

        for stage_name, attrs in non_consolidator_hasattr.items():
            assert not attrs['episode_limit'], (
                f'{stage_name}.episode_limit was set by run_full_cycle — '
                'isinstance guard at harness.py:417 may have been removed or widened'
            )
            assert not attrs['memory_limit'], (
                f'{stage_name}.memory_limit was set by run_full_cycle — '
                'isinstance guard at harness.py:417 may have been removed or widened'
            )


# ── Tier selection boundary tests (parametrized) ───────────────────────────────


@pytest.mark.asyncio
@pytest.mark.parametrize(
    'buffer_size,expected_model,expected_episode_limit,expected_memory_limit',
    [
        (0, 'sonnet', 125, 250),  # well below threshold (0 is NOT > 15.0)
        (15, 'sonnet', 125, 250),  # exact boundary (15 is NOT > 15.0, so sonnet)
        (16, 'opus', 500, 1000),  # just above boundary (16 > 15.0, so opus)
    ],
)
async def test_select_tier_boundary(
    journal,
    event_buffer,
    mock_memory_service,
    buffer_size,
    expected_model,
    expected_episode_limit,
    expected_memory_limit,
):
    """Parametrized boundary test for _select_tier.

    ReconciliationConfig defaults: buffer_size_threshold=10, opus_threshold_ratio=1.5.
    Threshold = 10 * 1.5 = 15.0; the condition is strictly greater-than (>).
    size=0  → NOT > 15.0 → sonnet (well below)
    size=15 → NOT > 15.0 → sonnet (exact boundary, must not upgrade)
    size=16 →     > 15.0 → opus  (just above boundary)
    """
    harness = _make_test_harness(journal, event_buffer, mock_memory_service)

    # Patch get_buffer_stats to return controlled buffer size
    harness.buffer.get_buffer_stats = AsyncMock(return_value={'size': buffer_size})

    tier = await harness._select_tier('test-project')

    assert tier.model == expected_model
    assert tier.episode_limit == expected_episode_limit
    assert tier.memory_limit == expected_memory_limit


@pytest.mark.asyncio
async def test_opus_tier_propagates_limits_to_consolidator(
    journal,
    event_buffer,
    mock_memory_service,
):
    """run_full_cycle propagates opus limits (500/1000) to MemoryConsolidator.

    When buffer size is 16 (> threshold 15.0), _select_tier returns opus tier.
    run_full_cycle must set stage.episode_limit=500 and stage.memory_limit=1000
    on the MemoryConsolidator before calling stage.run().
    """
    from fused_memory.reconciliation.stages.memory_consolidator import MemoryConsolidator

    harness = _make_test_harness(journal, event_buffer, mock_memory_service)

    # Mock buffer stats to trigger opus tier (16 > 15.0)
    harness.buffer.get_buffer_stats = AsyncMock(return_value={'size': 16})

    # Push an event so drain() has something to process
    await event_buffer.push(_make_event())

    # Capture limits at the moment stage.run() is called
    captured: dict = {}
    stage0 = harness.stages[0]
    assert isinstance(stage0, MemoryConsolidator)

    async def capturing_run(
        events,
        watermark,
        prior_reports,
        run_id,
        model=None,
        _s=stage0,
    ):
        captured['episode_limit'] = _s.episode_limit
        captured['memory_limit'] = _s.memory_limit
        return StageReport(
            stage=_s.stage_id,
            started_at=datetime.now(UTC),
            completed_at=datetime.now(UTC),
            items_flagged=[],
            stats={},
            llm_calls=0,
            tokens_used=0,
        )

    stage0.run = capturing_run
    _mock_stage_run(harness.stages[1])
    _mock_stage_run(harness.stages[2])

    # Zero-sentinel: reset to values that differ from opus tier defaults (500/1000).
    # MemoryConsolidator class defaults are 500/1000 — identical to opus values.
    # Without this reset, deleting harness.py:418-419 would leave stage0 at its class
    # defaults, and the test would still pass.  By forcing to 0 first we guarantee the
    # test fails when propagation is absent.
    stage0.episode_limit = 0
    stage0.memory_limit = 0

    tier = await harness._select_tier('test-project')
    await harness.run_full_cycle('test-project', 'buffer_size:16', tier=tier)

    assert captured.get('episode_limit') == 500, (
        f'Expected episode_limit=500 for opus tier, got {captured.get("episode_limit")}'
    )
    assert captured.get('memory_limit') == 1000, (
        f'Expected memory_limit=1000 for opus tier, got {captured.get("memory_limit")}'
    )


# ── Remediation attribute-propagation tests ───────────────────────────


@pytest.mark.asyncio
async def test_remediation_propagates_tier_limits_to_consolidator(
    journal,
    event_buffer,
    mock_memory_service,
):
    """_run_remediation_pass applies TierConfig limits onto MemoryConsolidator."""
    from fused_memory.reconciliation.harness import TierConfig
    from fused_memory.reconciliation.stages.memory_consolidator import MemoryConsolidator

    harness = _make_test_harness(journal, event_buffer, mock_memory_service)
    stages = harness._make_stages()
    harness._make_stages = lambda: stages

    captured: dict = {}

    async def capture_attrs(stage):
        captured['episode_limit'] = stage.episode_limit
        captured['memory_limit'] = stage.memory_limit
        captured['remediation_findings'] = stage.remediation_findings

    stage1 = stages[0]
    assert isinstance(stage1, MemoryConsolidator)

    # Zero-sentinel to ensure test fails if propagation is absent
    stage1.episode_limit = 0
    stage1.memory_limit = 0

    _mock_stage_run(stage1, before_return=capture_attrs)
    _mock_stage_run(stages[1])
    _mock_stage_run(stages[2])

    findings = [_make_s3_findings()[0]]  # one actionable finding
    tier = TierConfig(model='sonnet', episode_limit=125, memory_limit=250)

    await harness._run_remediation_pass(
        'test-project',
        '/tmp/test',
        'parent-run-id',
        findings,
        tier,
    )

    assert captured.get('episode_limit') == 125, (
        f'Expected episode_limit=125, got {captured.get("episode_limit")}'
    )
    assert captured.get('memory_limit') == 250, (
        f'Expected memory_limit=250, got {captured.get("memory_limit")}'
    )
    assert captured.get('remediation_findings') == findings


@pytest.mark.asyncio
async def test_remediation_sets_project_id_and_root_on_all_stages(
    journal,
    event_buffer,
    mock_memory_service,
):
    """_run_remediation_pass sets project_id and project_root on every stage."""
    from fused_memory.reconciliation.harness import TierConfig

    harness = _make_test_harness(journal, event_buffer, mock_memory_service)
    stages = harness._make_stages()
    harness._make_stages = lambda: stages

    stage_attrs: dict[str, dict] = {}

    for stage in stages:
        stage_name = type(stage).__name__

        async def capture(s, _name=stage_name):
            stage_attrs[_name] = {
                'project_id': s.project_id,
                'project_root': s.project_root,
            }

        _mock_stage_run(stage, before_return=capture)

    findings = [_make_s3_findings()[0]]
    tier = TierConfig(model='sonnet', episode_limit=100, memory_limit=200)

    await harness._run_remediation_pass(
        'my-project',
        '/srv/my-project',
        'parent-run-id',
        findings,
        tier,
    )

    for name, attrs in stage_attrs.items():
        assert attrs['project_id'] == 'my-project', (
            f"{name}: expected project_id='my-project', got {attrs['project_id']!r}"
        )
        assert attrs['project_root'] == '/srv/my-project', (
            f"{name}: expected project_root='/srv/my-project', got {attrs['project_root']!r}"
        )


@pytest.mark.asyncio
async def test_remediation_sets_remediation_mode_on_task_knowledge_sync(
    journal,
    event_buffer,
    mock_memory_service,
):
    """_run_remediation_pass sets remediation_mode=True on TaskKnowledgeSync."""
    from fused_memory.reconciliation.harness import TierConfig
    from fused_memory.reconciliation.stages.task_knowledge_sync import TaskKnowledgeSync

    harness = _make_test_harness(journal, event_buffer, mock_memory_service)
    stages = harness._make_stages()
    harness._make_stages = lambda: stages

    captured: dict = {}

    stage2 = stages[1]
    assert isinstance(stage2, TaskKnowledgeSync)

    async def capture_mode(s):
        captured['remediation_mode'] = s.remediation_mode

    _mock_stage_run(stages[0])
    _mock_stage_run(stage2, before_return=capture_mode)
    _mock_stage_run(stages[2])

    findings = [_make_s3_findings()[0]]
    tier = TierConfig(model='sonnet', episode_limit=100, memory_limit=200)

    await harness._run_remediation_pass(
        'test-project',
        '/tmp/test',
        'parent-run-id',
        findings,
        tier,
    )

    assert captured.get('remediation_mode') is True


@pytest.mark.asyncio
async def test_remediation_forwards_tier_model_to_stage_run(
    journal,
    event_buffer,
    mock_memory_service,
):
    """_run_remediation_pass forwards tier.model as the model kwarg to each stage.run()."""
    from fused_memory.reconciliation.harness import TierConfig

    harness = _make_test_harness(journal, event_buffer, mock_memory_service)
    stages = harness._make_stages()
    harness._make_stages = lambda: stages

    models_seen: dict[str, dict] = {}

    for stage in stages:
        stage_name = type(stage).__name__
        call_args: dict = {}
        _mock_stage_run(stage, capture_call_args=call_args)
        models_seen[stage_name] = call_args  # populated after run

    findings = [_make_s3_findings()[0]]
    tier = TierConfig(model='opus', episode_limit=500, memory_limit=1000)

    await harness._run_remediation_pass(
        'test-project',
        '/tmp/test',
        'parent-run-id',
        findings,
        tier,
    )

    for name, call_args in models_seen.items():
        assert call_args.get('model') == 'opus', (
            f"{name}: expected model='opus', got {call_args.get('model')!r}"
        )


# ── Tests for task 455: harness._fetch_filtered_task_tree ──────────────────────


class TestHarnessFetchFilteredTaskTree:
    """ReconciliationHarness._fetch_filtered_task_tree returns filtered task trees."""

    @pytest.mark.asyncio
    async def test_fetches_and_filters_task_tree(self, journal, event_buffer, mock_memory_service):
        """_fetch_filtered_task_tree fetches tasks and returns a FilteredTaskTree."""
        from fused_memory.reconciliation.task_filter import FilteredTaskTree

        harness = _make_test_harness(journal, event_buffer, mock_memory_service)

        # Mock taskmaster to return a mix of active + done + cancelled tasks
        harness.taskmaster.get_tasks.return_value = {  # type: ignore[union-attr,attr-defined]
            'tasks': [
                {'id': 1, 'title': 'T1', 'status': 'in-progress', 'dependencies': []},
                {'id': 2, 'title': 'T2', 'status': 'pending', 'dependencies': []},
                {'id': 3, 'title': 'T3', 'status': 'blocked', 'dependencies': []},
                {'id': 4, 'title': 'T4', 'status': 'deferred', 'dependencies': []},
                {'id': 5, 'title': 'T5', 'status': 'done', 'dependencies': []},
                {'id': 6, 'title': 'T6', 'status': 'done', 'dependencies': []},
                {'id': 7, 'title': 'T7', 'status': 'cancelled', 'dependencies': []},
            ]
        }

        result = await harness._fetch_filtered_task_tree('/abs/path')

        assert isinstance(result, FilteredTaskTree)
        assert len(result.active_tasks) == 4
        assert result.done_count == 2
        assert len(result.done_tasks) == 2  # main's FilteredTaskTree retains done task dicts
        assert result.cancelled_count == 1
        harness.taskmaster.get_tasks.assert_called_once_with(project_root='/abs/path')  # type: ignore[union-attr,attr-defined]

    @pytest.mark.asyncio
    async def test_handles_fetch_exception(
        self,
        journal,
        event_buffer,
        mock_memory_service,
        caplog,
    ):
        """_fetch_filtered_task_tree returns empty FilteredTaskTree and logs warning on error."""
        import logging

        from fused_memory.reconciliation.task_filter import FilteredTaskTree

        harness = _make_test_harness(journal, event_buffer, mock_memory_service)
        harness.taskmaster.get_tasks.side_effect = RuntimeError('connection refused')  # type: ignore[union-attr,attr-defined]

        with caplog.at_level(logging.WARNING):
            result = await harness._fetch_filtered_task_tree('/abs/path')

        # Must NOT re-raise; must return empty tree
        assert isinstance(result, FilteredTaskTree)
        assert result.active_tasks == []
        assert result.total_count == 0

        # Must have logged a warning containing BOTH the project_root and exception message
        assert any(
            'connection refused' in r.message and '/abs/path' in r.message
            for r in caplog.records
            if r.levelno >= logging.WARNING
        )

    @pytest.mark.asyncio
    async def test_returns_empty_when_no_taskmaster(
        self,
        journal,
        event_buffer,
        mock_memory_service,
    ):
        """_fetch_filtered_task_tree returns empty tree when taskmaster is None."""
        from fused_memory.config.schema import FusedMemoryConfig, ReconciliationConfig
        from fused_memory.reconciliation.harness import ReconciliationHarness
        from fused_memory.reconciliation.task_filter import FilteredTaskTree

        config = FusedMemoryConfig(
            reconciliation=ReconciliationConfig(
                enabled=True,
                explore_codebase_root='/tmp/test',
                agent_llm_provider='anthropic',
                agent_llm_model='claude-sonnet-4-20250514',
            )
        )
        harness = ReconciliationHarness(
            memory_service=mock_memory_service,
            taskmaster=None,
            journal=journal,
            event_buffer=event_buffer,
            config=config,
        )

        result = await harness._fetch_filtered_task_tree('/abs/path')

        assert isinstance(result, FilteredTaskTree)
        assert result.active_tasks == []

    @pytest.mark.asyncio
    async def test_returns_empty_when_empty_project_root(
        self,
        journal,
        event_buffer,
        mock_memory_service,
    ):
        """_fetch_filtered_task_tree returns empty tree when project_root is empty."""
        from fused_memory.reconciliation.task_filter import FilteredTaskTree

        harness = _make_test_harness(journal, event_buffer, mock_memory_service)

        result = await harness._fetch_filtered_task_tree('')

        assert isinstance(result, FilteredTaskTree)
        assert result.active_tasks == []
        harness.taskmaster.get_tasks.assert_not_called()  # type: ignore[union-attr,attr-defined]


# ── Tests for task 455: harness wires filtered_task_tree into stages ──────────


class TestHarnessFilteredTaskTreeWiring:
    """run_full_cycle and _run_remediation_pass wire _fetch_filtered_task_tree into stages."""

    def _make_tree(self):
        """Return a small FilteredTaskTree for wiring assertions."""
        from fused_memory.reconciliation.task_filter import FilteredTaskTree

        return FilteredTaskTree(
            active_tasks=[
                {'id': 1, 'title': 'T1', 'status': 'in-progress', 'dependencies': []},
            ],
            done_tasks=[],
            done_count=0,
            cancelled_count=0,
            total_count=1,
        )

    @pytest.mark.asyncio
    async def test_run_full_cycle_calls_fetch_once_with_project_root(
        self,
        journal,
        event_buffer,
        mock_memory_service,
    ):
        """run_full_cycle calls _fetch_filtered_task_tree exactly once with the project_root."""
        from unittest.mock import AsyncMock

        from fused_memory.reconciliation.task_filter import FilteredTaskTree

        harness = _make_test_harness(journal, event_buffer, mock_memory_service)
        harness._fetch_filtered_task_tree = AsyncMock(return_value=FilteredTaskTree())

        for stage in harness.stages:
            _mock_stage_run(stage)

        # Embed _project_root in the event payload so run_full_cycle can extract it
        event = _make_event()
        event.payload['_project_root'] = '/my/project'

        await harness.run_full_cycle('test-project', 'test-trigger', events=[event])

        harness._fetch_filtered_task_tree.assert_called_once_with('/my/project')  # type: ignore[attr-defined]

    @pytest.mark.asyncio
    async def test_run_full_cycle_invokes_get_tasks_exactly_once(
        self,
        journal,
        event_buffer,
        mock_memory_service,
    ):
        """Regression guard: run_full_cycle issues exactly one taskmaster.get_tasks call via
        _fetch_filtered_task_tree.

        Stages are mocked, so this covers the harness-level orchestration path (including that
        remediation reuses the pre-fetched tree rather than re-fetching), not stage-internal
        bypasses. Catching a stage that bypasses the helper by calling taskmaster.get_tasks
        directly would require an integration test with real stage implementations.
        """
        harness = _make_test_harness(journal, event_buffer, mock_memory_service)

        # Set up taskmaster.get_tasks to return a valid task list so the real
        # _fetch_filtered_task_tree can produce a non-empty FilteredTaskTree.
        harness.taskmaster.get_tasks.return_value = {  # type: ignore[union-attr]
            'tasks': (
                [
                    {'id': i, 'title': f'T{i}', 'status': 'pending', 'dependencies': []}
                    for i in range(1, 4)
                ]
                + [
                    {'id': i, 'title': f'T{i}', 'status': 'done', 'dependencies': []}
                    for i in range(4, 9)
                ]
            )
        }

        for stage in harness.stages:
            _mock_stage_run(stage)

        event = _make_event()
        event.payload['_project_root'] = '/my/project'

        await harness.run_full_cycle('test-project', 'test-trigger', events=[event])

        harness.taskmaster.get_tasks.assert_called_once()  # type: ignore[union-attr,attr-defined]

    @pytest.mark.asyncio
    async def test_run_full_cycle_sets_filtered_task_tree_on_consolidator(
        self,
        journal,
        event_buffer,
        mock_memory_service,
    ):
        """run_full_cycle passes fetched filtered_task_tree to MemoryConsolidator via _configure_consolidator."""
        from unittest.mock import AsyncMock

        from fused_memory.reconciliation.stages.memory_consolidator import MemoryConsolidator

        harness = _make_test_harness(journal, event_buffer, mock_memory_service)
        expected_tree = self._make_tree()
        harness._fetch_filtered_task_tree = AsyncMock(return_value=expected_tree)

        captured: dict = {}

        stage1 = harness.stages[0]
        assert isinstance(stage1, MemoryConsolidator)

        async def capture_tree(stage):
            captured['filtered_task_tree'] = stage.filtered_task_tree

        _mock_stage_run(stage1, before_return=capture_tree)
        _mock_stage_run(harness.stages[1])
        _mock_stage_run(harness.stages[2])

        await harness.run_full_cycle('test-project', 'test-trigger', events=[_make_event()])

        assert captured.get('filtered_task_tree') is expected_tree, (
            f'Expected MemoryConsolidator.filtered_task_tree to be the fetched tree, '
            f'got {captured.get("filtered_task_tree")!r}. '
            'run_full_cycle must call _configure_consolidator with filtered_task_tree kwarg.'
        )

    @pytest.mark.asyncio
    async def test_run_full_cycle_sets_filtered_task_tree_on_task_knowledge_sync(
        self,
        journal,
        event_buffer,
        mock_memory_service,
    ):
        """run_full_cycle sets filtered_task_tree on TaskKnowledgeSync."""
        from unittest.mock import AsyncMock

        from fused_memory.reconciliation.stages.task_knowledge_sync import TaskKnowledgeSync

        harness = _make_test_harness(journal, event_buffer, mock_memory_service)
        expected_tree = self._make_tree()
        harness._fetch_filtered_task_tree = AsyncMock(return_value=expected_tree)

        captured: dict = {}

        stage2 = harness.stages[1]
        assert isinstance(stage2, TaskKnowledgeSync)

        async def capture_tree(stage):
            captured['filtered_task_tree'] = stage.filtered_task_tree

        _mock_stage_run(harness.stages[0])
        _mock_stage_run(stage2, before_return=capture_tree)
        _mock_stage_run(harness.stages[2])

        await harness.run_full_cycle('test-project', 'test-trigger', events=[_make_event()])

        assert captured.get('filtered_task_tree') is expected_tree, (
            f'Expected TaskKnowledgeSync.filtered_task_tree to be the fetched tree, '
            f'got {captured.get("filtered_task_tree")!r}. '
            'run_full_cycle must set stage.filtered_task_tree on TaskKnowledgeSync instances.'
        )

    @pytest.mark.asyncio
    async def test_remediation_sets_filtered_task_tree_on_consolidator(
        self,
        journal,
        event_buffer,
        mock_memory_service,
    ):
        """_run_remediation_pass wires filtered_task_tree to MemoryConsolidator."""
        from unittest.mock import AsyncMock

        from fused_memory.reconciliation.harness import TierConfig
        from fused_memory.reconciliation.stages.memory_consolidator import MemoryConsolidator

        harness = _make_test_harness(journal, event_buffer, mock_memory_service)
        stages = harness._make_stages()
        harness._make_stages = lambda: stages

        expected_tree = self._make_tree()
        harness._fetch_filtered_task_tree = AsyncMock(return_value=expected_tree)

        captured: dict = {}

        stage1 = stages[0]
        assert isinstance(stage1, MemoryConsolidator)

        async def capture_tree(stage):
            captured['filtered_task_tree'] = stage.filtered_task_tree

        _mock_stage_run(stage1, before_return=capture_tree)
        _mock_stage_run(stages[1])
        _mock_stage_run(stages[2])

        findings = [_make_s3_findings()[0]]
        tier = TierConfig(model='sonnet', episode_limit=100, memory_limit=200)

        await harness._run_remediation_pass(
            'test-project',
            '/my/project',
            'parent-run-id',
            findings,
            tier,
        )

        assert captured.get('filtered_task_tree') is expected_tree, (
            f'Expected MemoryConsolidator.filtered_task_tree to be the fetched tree in remediation, '
            f'got {captured.get("filtered_task_tree")!r}. '
            '_run_remediation_pass must also call _fetch_filtered_task_tree and wire the result.'
        )

    @pytest.mark.asyncio
    async def test_remediation_sets_filtered_task_tree_on_task_knowledge_sync(
        self,
        journal,
        event_buffer,
        mock_memory_service,
    ):
        """_run_remediation_pass wires filtered_task_tree to TaskKnowledgeSync."""
        from unittest.mock import AsyncMock

        from fused_memory.reconciliation.harness import TierConfig
        from fused_memory.reconciliation.stages.task_knowledge_sync import TaskKnowledgeSync

        harness = _make_test_harness(journal, event_buffer, mock_memory_service)
        stages = harness._make_stages()
        harness._make_stages = lambda: stages

        expected_tree = self._make_tree()
        harness._fetch_filtered_task_tree = AsyncMock(return_value=expected_tree)

        captured: dict = {}

        stage2 = stages[1]
        assert isinstance(stage2, TaskKnowledgeSync)

        async def capture_tree(stage):
            captured['filtered_task_tree'] = stage.filtered_task_tree

        _mock_stage_run(stages[0])
        _mock_stage_run(stage2, before_return=capture_tree)
        _mock_stage_run(stages[2])

        findings = [_make_s3_findings()[0]]
        tier = TierConfig(model='sonnet', episode_limit=100, memory_limit=200)

        await harness._run_remediation_pass(
            'test-project',
            '/my/project',
            'parent-run-id',
            findings,
            tier,
        )

        assert captured.get('filtered_task_tree') is expected_tree, (
            f'Expected TaskKnowledgeSync.filtered_task_tree to be the fetched tree in remediation, '
            f'got {captured.get("filtered_task_tree")!r}. '
            '_run_remediation_pass must set stage.filtered_task_tree on TaskKnowledgeSync instances.'
        )

    @pytest.mark.asyncio
    async def test_run_full_cycle_uses_configure_task_sync_for_stage2(
        self,
        journal,
        event_buffer,
        mock_memory_service,
    ):
        """run_full_cycle calls _configure_task_sync (not naked assignment) for Stage-2 wiring."""
        from unittest.mock import AsyncMock

        from fused_memory.reconciliation.harness import ReconciliationHarness
        from fused_memory.reconciliation.stages.task_knowledge_sync import TaskKnowledgeSync

        harness = _make_test_harness(journal, event_buffer, mock_memory_service)
        expected_tree = self._make_tree()
        harness._fetch_filtered_task_tree = AsyncMock(return_value=expected_tree)

        spy_calls: list = []
        # Accessing a staticmethod via the class already unwraps it to a plain function
        real_helper = ReconciliationHarness._configure_task_sync

        def spy(stage, *, filtered_task_tree=None, remediation_mode=False):
            spy_calls.append(
                {
                    'stage': stage,
                    'filtered_task_tree': filtered_task_tree,
                    'remediation_mode': remediation_mode,
                }
            )
            real_helper(
                stage, filtered_task_tree=filtered_task_tree, remediation_mode=remediation_mode
            )

        ReconciliationHarness._configure_task_sync = staticmethod(spy)  # type: ignore[method-assign]
        try:
            for stage in harness.stages:
                _mock_stage_run(stage)

            event = _make_event()
            event.payload['_project_root'] = '/my/project'
            await harness.run_full_cycle('test-project', 'test-trigger', events=[event])
        finally:
            ReconciliationHarness._configure_task_sync = staticmethod(real_helper)  # type: ignore[method-assign]

        assert len(spy_calls) == 1, (
            f'Expected _configure_task_sync called once, got {len(spy_calls)}'
        )
        call = spy_calls[0]
        stage2 = harness.stages[1]
        assert isinstance(stage2, TaskKnowledgeSync)
        assert call['stage'] is stage2
        assert call['filtered_task_tree'] is expected_tree
        assert call['remediation_mode'] is False

    @pytest.mark.asyncio
    async def test_remediation_uses_configure_task_sync_for_stage2(
        self,
        journal,
        event_buffer,
        mock_memory_service,
    ):
        """_run_remediation_pass calls _configure_task_sync with remediation_mode=True for Stage 2."""
        from unittest.mock import AsyncMock

        from fused_memory.reconciliation.harness import ReconciliationHarness, TierConfig
        from fused_memory.reconciliation.stages.task_knowledge_sync import TaskKnowledgeSync

        harness = _make_test_harness(journal, event_buffer, mock_memory_service)
        stages = harness._make_stages()
        harness._make_stages = lambda: stages

        expected_tree = self._make_tree()
        harness._fetch_filtered_task_tree = AsyncMock(return_value=expected_tree)

        spy_calls: list = []
        real_helper = ReconciliationHarness._configure_task_sync

        def spy(stage, *, filtered_task_tree=None, remediation_mode=False):
            spy_calls.append(
                {
                    'stage': stage,
                    'filtered_task_tree': filtered_task_tree,
                    'remediation_mode': remediation_mode,
                }
            )
            real_helper(
                stage, filtered_task_tree=filtered_task_tree, remediation_mode=remediation_mode
            )

        ReconciliationHarness._configure_task_sync = staticmethod(spy)  # type: ignore[method-assign]
        try:
            _mock_stage_run(stages[0])
            _mock_stage_run(stages[1])
            _mock_stage_run(stages[2])

            findings = [_make_s3_findings()[0]]
            tier = TierConfig(model='sonnet', episode_limit=100, memory_limit=200)
            await harness._run_remediation_pass(
                'test-project',
                '/my/project',
                'parent-run-id',
                findings,
                tier,
            )
        finally:
            ReconciliationHarness._configure_task_sync = staticmethod(real_helper)  # type: ignore[method-assign]

        assert len(spy_calls) == 1, (
            f'Expected _configure_task_sync called once, got {len(spy_calls)}'
        )
        call = spy_calls[0]
        stage2 = stages[1]
        assert isinstance(stage2, TaskKnowledgeSync)
        assert call['stage'] is stage2
        assert call['filtered_task_tree'] is expected_tree
        assert call['remediation_mode'] is True

    @pytest.mark.asyncio
    async def test_run_remediation_pass_accepts_prefetched_tree(
        self,
        journal,
        event_buffer,
        mock_memory_service,
    ):
        """_run_remediation_pass uses a supplied filtered_task_tree and skips _fetch_filtered_task_tree."""
        from unittest.mock import AsyncMock

        from fused_memory.reconciliation.harness import TierConfig

        harness = _make_test_harness(journal, event_buffer, mock_memory_service)
        stages = harness._make_stages()
        harness._make_stages = lambda: stages

        # Pre-fetched tree passed by caller — fetch should NOT be called
        prefetched_tree = self._make_tree()
        harness._fetch_filtered_task_tree = AsyncMock(return_value=self._make_tree())

        captured: dict = {}

        async def capture_s1(stage):
            captured['s1_tree'] = stage.filtered_task_tree

        async def capture_s2(stage):
            captured['s2_tree'] = stage.filtered_task_tree

        _mock_stage_run(stages[0], before_return=capture_s1)
        _mock_stage_run(stages[1], before_return=capture_s2)
        _mock_stage_run(stages[2])

        findings = [_make_s3_findings()[0]]
        tier = TierConfig(model='sonnet', episode_limit=100, memory_limit=200)
        await harness._run_remediation_pass(
            'test-project',
            '/my/project',
            'parent-run-id',
            findings,
            tier,
            filtered_task_tree=prefetched_tree,
        )

        harness._fetch_filtered_task_tree.assert_not_called()  # type: ignore[attr-defined]
        assert captured.get('s1_tree') is prefetched_tree
        assert captured.get('s2_tree') is prefetched_tree

    @pytest.mark.asyncio
    async def test_run_full_cycle_and_remediation_fetches_task_tree_once_total(
        self,
        journal,
        event_buffer,
        mock_memory_service,
    ):
        """run_full_cycle + remediation makes exactly one _fetch_filtered_task_tree call total."""
        from unittest.mock import AsyncMock

        harness = _make_test_harness(journal, event_buffer, mock_memory_service)
        expected_tree = self._make_tree()
        harness._fetch_filtered_task_tree = AsyncMock(return_value=expected_tree)

        # Stage 1 and 2 run normally; Stage 3 returns an actionable finding to trigger remediation
        _mock_stage_run(harness.stages[0])
        _mock_stage_run(harness.stages[1])
        _mock_stage_run(harness.stages[2], items_flagged=[_make_s3_findings()[0]])

        event = _make_event()
        event.payload['_project_root'] = '/my/project'
        await harness.run_full_cycle('test-project', 'test-trigger', events=[event])

        assert harness._fetch_filtered_task_tree.call_count == 1, (  # type: ignore[attr-defined]
            f'Expected exactly one _fetch_filtered_task_tree call across the full cycle + '
            f'remediation pass, got {harness._fetch_filtered_task_tree.call_count}. '  # type: ignore[attr-defined]
            'run_full_cycle must thread its pre-fetched tree into _maybe_remediate.'
        )

    @pytest.mark.asyncio
    async def test_remediation_falls_back_to_fetch_when_tree_is_none(
        self,
        journal,
        event_buffer,
        mock_memory_service,
    ):
        """_run_remediation_pass calls _fetch_filtered_task_tree exactly once when no tree supplied."""
        from unittest.mock import AsyncMock

        from fused_memory.reconciliation.harness import TierConfig

        harness = _make_test_harness(journal, event_buffer, mock_memory_service)
        stages = harness._make_stages()
        harness._make_stages = lambda: stages

        expected_tree = self._make_tree()
        harness._fetch_filtered_task_tree = AsyncMock(return_value=expected_tree)

        _mock_stage_run(stages[0])
        _mock_stage_run(stages[1])
        _mock_stage_run(stages[2])

        findings = [_make_s3_findings()[0]]
        tier = TierConfig(model='sonnet', episode_limit=100, memory_limit=200)
        # No filtered_task_tree kwarg — method must fall back to _fetch_filtered_task_tree
        await harness._run_remediation_pass(
            'test-project',
            '/my/project',
            'parent-run-id',
            findings,
            tier,
        )

        harness._fetch_filtered_task_tree.assert_called_once_with('/my/project')  # type: ignore[attr-defined]


class TestConfigureTaskSync:
    """Unit tests for the _configure_task_sync staticmethod on ReconciliationHarness."""

    def _make_tree(self):
        """Return a small FilteredTaskTree for assertions."""
        from fused_memory.reconciliation.task_filter import FilteredTaskTree

        return FilteredTaskTree(
            active_tasks=[
                {'id': 2, 'title': 'T2', 'status': 'in-progress', 'dependencies': []},
            ],
            done_tasks=[],
            done_count=0,
            cancelled_count=0,
            total_count=1,
        )

    def test_configure_task_sync_sets_filtered_task_tree(
        self,
        journal,
        event_buffer,
        mock_memory_service,
    ):
        """_configure_task_sync applies filtered_task_tree and remediation_mode=False to stage2."""
        from fused_memory.reconciliation.harness import ReconciliationHarness
        from fused_memory.reconciliation.stages.task_knowledge_sync import TaskKnowledgeSync

        harness = _make_test_harness(journal, event_buffer, mock_memory_service)
        stage2 = harness.stages[1]
        assert isinstance(stage2, TaskKnowledgeSync)

        tree = self._make_tree()
        ReconciliationHarness._configure_task_sync(
            stage2, filtered_task_tree=tree, remediation_mode=False
        )

        assert stage2.filtered_task_tree is tree
        assert stage2.remediation_mode is False

    def test_configure_task_sync_sets_remediation_mode(
        self,
        journal,
        event_buffer,
        mock_memory_service,
    ):
        """_configure_task_sync applies remediation_mode=True to stage2."""
        from fused_memory.reconciliation.harness import ReconciliationHarness
        from fused_memory.reconciliation.stages.task_knowledge_sync import TaskKnowledgeSync

        harness = _make_test_harness(journal, event_buffer, mock_memory_service)
        stage2 = harness.stages[1]
        assert isinstance(stage2, TaskKnowledgeSync)

        tree = self._make_tree()
        ReconciliationHarness._configure_task_sync(
            stage2, filtered_task_tree=tree, remediation_mode=True
        )

        assert stage2.remediation_mode is True
        assert stage2.filtered_task_tree is tree

    def test_configure_task_sync_is_staticmethod(self):
        """_configure_task_sync must be declared as a @staticmethod."""
        import inspect

        from fused_memory.reconciliation.harness import ReconciliationHarness

        assert isinstance(
            inspect.getattr_static(ReconciliationHarness, '_configure_task_sync'),
            staticmethod,
        ), '_configure_task_sync must be a @staticmethod on ReconciliationHarness'


# ── Deferred-write replay durability ─────────────────────────────────────────


@pytest.mark.asyncio
async def test_replay_deletes_successful_and_preserves_failed(
    journal, event_buffer, mock_memory_service
):
    """_replay_deferred_writes deletes only successful writes; failed write stays in SQLite."""
    harness = _make_test_harness(journal, event_buffer, mock_memory_service)

    # add_memory raises only when content == 'bad'
    call_log: list[str] = []

    async def add_memory_side_effect(**kwargs):
        content = kwargs.get('content', '')
        call_log.append(content)
        if content == 'bad':
            raise RuntimeError('boom')

    mock_memory_service.add_memory = AsyncMock(side_effect=add_memory_side_effect)

    # Defer three writes
    await event_buffer.defer_write('test-project', 'good-1', 'cat', {})
    await event_buffer.defer_write('test-project', 'bad', 'cat', {})
    await event_buffer.defer_write('test-project', 'good-3', 'cat', {})

    # Should not raise — per-item exception is swallowed
    await harness._replay_deferred_writes('test-project')

    # add_memory was called for every claimed row
    assert len(call_log) == 3
    assert set(call_log) == {'good-1', 'bad', 'good-3'}

    # Only the failed write remains in SQLite
    # Re-queue any still-claimed rows so we can re-claim them
    await event_buffer.release_stale_claims(0.0)
    remaining = await event_buffer.claim_deferred_writes('test-project')
    assert len(remaining) == 1
    assert remaining[0]['content'] == 'bad'


@pytest.mark.asyncio
async def test_replay_propagates_cancellation_and_preserves_claims(
    journal, event_buffer, mock_memory_service
):
    """CancelledError propagates out of _replay_deferred_writes; unprocessed rows survive."""
    import asyncio

    harness = _make_test_harness(journal, event_buffer, mock_memory_service)

    call_count = 0

    async def add_memory_side_effect(**kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 2:
            raise asyncio.CancelledError()

    mock_memory_service.add_memory = AsyncMock(side_effect=add_memory_side_effect)

    await event_buffer.defer_write('test-project', 'a', 'cat', {})
    await event_buffer.defer_write('test-project', 'b', 'cat', {})
    await event_buffer.defer_write('test-project', 'c', 'cat', {})

    with pytest.raises(asyncio.CancelledError):
        await harness._replay_deferred_writes('test-project')

    # 'a' was successfully written and deleted; 'b' and 'c' remain claimed
    await event_buffer.release_stale_claims(0.0)
    remaining = await event_buffer.claim_deferred_writes('test-project')
    assert len(remaining) == 2
    assert [r['content'] for r in remaining] == ['b', 'c']


@pytest.mark.asyncio
async def test_run_loop_releases_stale_claims_on_startup(
    journal, event_buffer, mock_memory_service
):
    """run_loop() must call release_stale_claims once during startup (not per iteration)."""
    import asyncio
    from unittest.mock import AsyncMock

    harness = _make_test_harness(journal, event_buffer, mock_memory_service)

    # Patch side-effect dependencies to avoid network/filesystem calls
    harness._recover_stale_runs = AsyncMock(return_value=None)
    harness._start_escalation_server = AsyncMock()
    harness._stop_escalation_server = AsyncMock()

    # Spy on release_stale_claims: side_effect passes through to the real method,
    # so return_value is intentionally omitted (side_effect takes precedence).
    original_release = event_buffer.release_stale_claims
    harness.buffer.release_stale_claims = AsyncMock(side_effect=original_release)

    with contextlib.suppress(TimeoutError):
        await asyncio.wait_for(harness.run_loop(), timeout=0.2)

    # Must be called exactly once (startup, not per loop iteration)
    harness.buffer.release_stale_claims.assert_called_once_with(
        harness.config.stale_run_recovery_seconds
    )
