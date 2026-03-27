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
    buf = EventBuffer(db_path=tmp_path / 'harness_eb.db', buffer_size_threshold=2, max_staleness_seconds=3600)
    await buf.initialize()
    yield buf
    await buf.close()


@pytest.fixture
def mock_memory_service():
    svc = AsyncMock()
    svc.search = AsyncMock(return_value=[])
    svc.get_episodes = AsyncMock(return_value=[])
    svc.get_status = AsyncMock(return_value={'graphiti': {'connected': True}, 'mem0': {'connected': True}, 'projects': {}})
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
async def test_run_full_cycle_restores_events_on_failure(journal, event_buffer, mock_memory_service):
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
async def test_full_cycle_extracts_project_root_from_events(journal, event_buffer, mock_memory_service):
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

    # Mock each stage's run method and capture the stage state
    captured_stages = []
    for stage in harness.stages:
        original_stage = stage

        async def mock_run(events, watermark, prior_reports, run_id, model=None, _s=original_stage):
            # Capture state at call time
            captured_stages.append({
                'project_id': _s.project_id,
                'project_root': _s.project_root,
            })
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


def _make_harness_with_mocked_stages(journal, event_buffer, mock_memory_service):
    """Build a harness with all stages mocked to return StageReports."""
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

    return ReconciliationHarness(
        memory_service=mock_memory_service,
        taskmaster=AsyncMock(),
        journal=journal,
        event_buffer=event_buffer,
        config=config,
    )


def _mock_stage_run(stage, items_flagged=None):
    """Replace stage.run with a mock that returns a StageReport."""
    async def mock_run(events, watermark, prior_reports, run_id, model=None, _s=stage):
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
    stage.prior_s3_findings = [_make_s3_findings()[0]]

    watermark = Watermark(project_id='test-project')
    payload = await stage.assemble_payload([], watermark, [])

    assert 'Prior Stage 3 Findings' in payload
    assert 'Stale edge' in payload


@pytest.mark.asyncio
async def test_run_full_cycle_triggers_remediation(journal, event_buffer, mock_memory_service):
    """Full cycle with S3 actionable findings triggers a remediation pass."""
    harness = _make_harness_with_mocked_stages(journal, event_buffer, mock_memory_service)

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
    journal, event_buffer, mock_memory_service,
):
    """No remediation pass when S3 has only non-actionable findings."""
    harness = _make_harness_with_mocked_stages(journal, event_buffer, mock_memory_service)

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
    journal, event_buffer, mock_memory_service,
):
    """If remediation pass fails, parent run remains completed."""
    harness = _make_harness_with_mocked_stages(journal, event_buffer, mock_memory_service)

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
            items_flagged=[], stats={},
            llm_calls=0, tokens_used=0,
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

    harness = _make_harness_with_mocked_stages(journal, event_buffer, mock_memory_service)

    # Make first stage sleep forever (simulating a long-running stage)
    async def slow_stage_run(events, watermark, prior_reports, run_id, model=None, _s=harness.stages[0]):
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
        "Bug 5: CancelledError is not caught in run_full_cycle, so complete_run is never called."
    )

    # Events must have been restored to the buffer
    stats = await event_buffer.get_buffer_stats('test-project')
    assert stats['size'] == 2, (
        f"Expected buffer size=2 after timeout, got {stats['size']}. "
        "Bug 5: restore_drained is not called on CancelledError."
    )


@pytest.mark.asyncio
async def test_run_full_cycle_accepts_pre_drained_events(journal, event_buffer, mock_memory_service):
    """run_full_cycle() must accept an optional 'events' param to skip drain().

    Bug 4: BacklogIterator.run() drains a chunk via drain_oldest_chunk(), then calls
    run_full_cycle() which re-drains via drain(), getting different events — the chunk
    events are silently lost.  Fix: add optional events param to run_full_cycle so
    BacklogIterator can pass the already-drained chunk.

    This test confirms that passing events=[...] to run_full_cycle uses those events
    without calling buffer.drain(), and that events_processed reflects the passed count.
    """
    harness = _make_harness_with_mocked_stages(journal, event_buffer, mock_memory_service)

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
        f"Expected events_processed=2 from pre-drained events, got {run.events_processed}. "
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

    harness = _make_harness_with_mocked_stages(journal, event_buffer, mock_memory_service)

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

    # Also make _recover_stale_runs a no-op
    harness._recover_stale_runs = AsyncMock(return_value=None)

    with patch.object(harness, 'run_full_cycle', side_effect=spy_rfc), contextlib.suppress(TimeoutError):
        # Run loop for one sleep cycle (loop sleeps 5s; we wait 0.2s — enough for 1 iteration)
        await asyncio.wait_for(harness.run_loop(), timeout=0.2)

    # For a halted project, run_full_cycle must NOT have been called
    assert len(run_full_cycle_called) == 0, (
        f"run_full_cycle was called {len(run_full_cycle_called)} time(s) "
        "for a halted project — Bug 3: halt check not wired into run_loop."
    )


@pytest.mark.asyncio
async def test_stage_state_reset_after_remediation(journal, event_buffer, mock_memory_service):
    """Stage state (remediation_findings, remediation_mode) is reset after remediation."""
    from fused_memory.reconciliation.stages.memory_consolidator import MemoryConsolidator
    from fused_memory.reconciliation.stages.task_knowledge_sync import TaskKnowledgeSync

    harness = _make_harness_with_mocked_stages(journal, event_buffer, mock_memory_service)

    await event_buffer.push(_make_event())

    _mock_stage_run(harness.stages[0])
    _mock_stage_run(harness.stages[1])
    _mock_stage_run(harness.stages[2], items_flagged=_make_s3_findings())

    await harness.run_full_cycle('test-project', 'buffer_size:1')

    # Verify state was cleaned up
    stage1 = harness.stages[0]
    stage2 = harness.stages[1]
    assert isinstance(stage1, MemoryConsolidator)
    assert isinstance(stage2, TaskKnowledgeSync)
    assert stage1.remediation_findings is None
    assert stage1.prior_s3_findings is None
    assert stage2.remediation_mode is False


@pytest.mark.asyncio
async def test_cancellation_cleanup_failure_preserves_cancelled_error(
    journal, event_buffer, mock_memory_service,
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

    harness = _make_harness_with_mocked_stages(journal, event_buffer, mock_memory_service)

    # Make first stage sleep forever (will be cancelled by wait_for timeout)
    async def slow_stage_run(
        events, watermark, prior_reports, run_id, model=None, _s=harness.stages[0],
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
    journal, event_buffer, mock_memory_service,
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

    harness = _make_harness_with_mocked_stages(journal, event_buffer, mock_memory_service)

    # Make first stage sleep forever (cancelled by the first outer_task.cancel())
    async def slow_stage_run(
        events, watermark, prior_reports, run_id, model=None, _s=harness.stages[0],
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

    # Let the task start and reach the slow stage
    await asyncio.sleep(0.05)

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
        "Review issue [async_cancellation_safety]: cleanup must be wrapped with "
        "asyncio.shield() so a second cancel cannot abort the DB write."
    )


# ── End-to-end tier selection test (Task 200) ────────────────────────


@pytest.mark.asyncio
async def test_sonnet_tier_selection_sets_consolidator_limits(journal, event_buffer, mock_memory_service):
    """Full path: ReconciliationConfig defaults → _select_tier() → run_full_cycle() → MemoryConsolidator limits.

    Verifies that a buffer count of 0 (below the opus threshold of 15) causes
    _select_tier() to return a sonnet TierConfig with episode_limit=125 and
    memory_limit=250, and that run_full_cycle() propagates those limits to the
    MemoryConsolidator stage before calling stage.run().
    """
    from fused_memory.config.schema import FusedMemoryConfig, ReconciliationConfig
    from fused_memory.reconciliation.harness import ReconciliationHarness, TierConfig
    from fused_memory.reconciliation.stages.memory_consolidator import MemoryConsolidator

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

    # Push events before mocking, so drain() has real events to return
    await event_buffer.push(_make_event())
    await event_buffer.push(_make_event())

    # Mock buffer stats to return size=0, ensuring sonnet tier
    # (opus requires count > buffer_size_threshold * opus_threshold_ratio = 10 * 1.5 = 15)
    event_buffer.get_buffer_stats = AsyncMock(return_value={'size': 0})

    # Part 1: verify _select_tier() returns correct sonnet TierConfig
    tier = await harness._select_tier('test-project')
    assert tier.model == 'sonnet', f"Expected model='sonnet', got '{tier.model}'"
    assert tier.episode_limit == 125, f"Expected episode_limit=125, got {tier.episode_limit}"
    assert tier.memory_limit == 250, f"Expected memory_limit=250, got {tier.memory_limit}"
    assert isinstance(tier, TierConfig)

    # Part 2: mock stage.run() to capture MemoryConsolidator limits at call time
    captured_consolidator_limits: dict = {}

    for stage in harness.stages:
        original_stage = stage

        async def mock_run(events, watermark, prior_reports, run_id, model=None, _s=original_stage):
            if isinstance(_s, MemoryConsolidator):
                captured_consolidator_limits['episode_limit'] = _s.episode_limit
                captured_consolidator_limits['memory_limit'] = _s.memory_limit
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

    # Part 3: run full cycle with the sonnet tier
    run = await harness.run_full_cycle('test-project', 'buffer_size:2', tier=tier)
    assert run.status == 'completed'

    # Part 4: assert MemoryConsolidator received sonnet limits
    assert captured_consolidator_limits, 'No MemoryConsolidator stage found — guard failed'
    assert captured_consolidator_limits.get('episode_limit') == 125, (
        f"Expected MemoryConsolidator.episode_limit=125 (sonnet tier), "
        f"got {captured_consolidator_limits.get('episode_limit')}. "
        "harness.py:417-421 must set stage.episode_limit = tier.episode_limit for MemoryConsolidator."
    )
    assert captured_consolidator_limits.get('memory_limit') == 250, (
        f"Expected MemoryConsolidator.memory_limit=250 (sonnet tier), "
        f"got {captured_consolidator_limits.get('memory_limit')}. "
        "harness.py:417-421 must set stage.memory_limit = tier.memory_limit for MemoryConsolidator."
    )
