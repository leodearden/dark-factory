"""Tests for reconciliation harness (pipeline orchestration)."""

import contextlib
import uuid
from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import pytest_asyncio

from fused_memory.models.reconciliation import (
    AssembledPayload,
    EventSource,
    EventType,
    ReconciliationEvent,
    ReconciliationRun,
    RunStatus,
    RunType,
    StageReport,
)
from fused_memory.reconciliation.event_buffer import EventBuffer
from fused_memory.reconciliation.harness import BacklogIterator
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


# ── Tests for Task 927: project_root fallback fix ────────────────────


def _make_config_927(project_root: str | None = '/abs/from/config'):
    """Build a FusedMemoryConfig for task-927 tests.

    Args:
        project_root: If a string, wraps it in ``TaskmasterConfig``; if ``None``,
            sets ``taskmaster=None`` so ``harness._project_root`` defaults to ``''``.
    """
    from fused_memory.config.schema import (
        FusedMemoryConfig,
        ReconciliationConfig,
        TaskmasterConfig,
    )

    recon_cfg = ReconciliationConfig(
        enabled=True,
        explore_codebase_root='/tmp/test',
        agent_llm_provider='anthropic',
        agent_llm_model='claude-sonnet-4-20250514',
    )
    taskmaster_cfg = TaskmasterConfig(project_root=project_root) if project_root is not None else None
    return FusedMemoryConfig(taskmaster=taskmaster_cfg, reconciliation=recon_cfg)


def _make_harness_927(journal, event_buffer, mock_memory_service, project_root: str | None = '/abs/from/config'):
    """Build a ReconciliationHarness with a configured project_root for task-927 tests."""
    from unittest.mock import AsyncMock

    from fused_memory.reconciliation.harness import ReconciliationHarness

    harness = ReconciliationHarness(
        memory_service=mock_memory_service,
        taskmaster=AsyncMock(),
        journal=journal,
        event_buffer=event_buffer,
        config=_make_config_927(project_root),
    )
    harness._make_stages = lambda: harness.stages
    return harness


@pytest.mark.asyncio
async def test_harness_init_stores_project_root_from_taskmaster_config(
    journal, event_buffer, mock_memory_service
):
    """ReconciliationHarness.__init__ should store _project_root from config.taskmaster."""

    # (a) With taskmaster configured: _project_root and property should come from config
    harness_a = _make_harness_927(journal, event_buffer, mock_memory_service, '/abs/from/config')
    assert harness_a._project_root == '/abs/from/config'
    assert harness_a.project_root == '/abs/from/config'

    # (b) With taskmaster=None: _project_root should default to ''
    harness_b = _make_harness_927(journal, event_buffer, mock_memory_service, None)
    assert harness_b._project_root == ''
    assert harness_b.project_root == ''


@pytest.mark.asyncio
async def test_harness_init_resolves_relative_project_root_to_absolute(
    journal, event_buffer, mock_memory_service
):
    """ReconciliationHarness.__init__ must resolve relative project_root values to absolute.

    Three cases:
    (a) Relative path '.' is resolved to str(Path('.').resolve()) — an absolute path.
    (b) Already-absolute '/abs/already' passes through unchanged (idempotent).
    (c) project_root=None (no taskmaster) stays '' (preserves task-927 short-circuit).

    Both _project_root attribute and the public project_root property must reflect
    the normalized value.
    """
    from pathlib import Path

    # (a) Relative '.' must be resolved to an absolute path
    harness_a = _make_harness_927(journal, event_buffer, mock_memory_service, '.')
    expected_resolved = str(Path('.').resolve())
    assert harness_a._project_root == expected_resolved, (
        f"Expected _project_root={expected_resolved!r} (resolved absolute path), "
        f"got {harness_a._project_root!r}"
    )
    assert harness_a._project_root != '.', "relative '.' must not remain as-is"
    assert harness_a.project_root == harness_a._project_root, (
        "project_root property must mirror _project_root"
    )

    # (b) Already-absolute path passes through unchanged
    harness_b = _make_harness_927(journal, event_buffer, mock_memory_service, '/abs/already')
    assert harness_b._project_root == '/abs/already', (
        f"Absolute path should pass through unchanged; got {harness_b._project_root!r}"
    )
    assert harness_b.project_root == '/abs/already'

    # (c) None (no taskmaster configured) → empty string — task-927 short-circuit preserved
    harness_c = _make_harness_927(journal, event_buffer, mock_memory_service, None)
    assert harness_c._project_root == '', (
        f"taskmaster=None should give _project_root=''; got {harness_c._project_root!r}"
    )
    assert harness_c.project_root == ''

    # (d) Empty-string project_root → empty string — distinct from None, exercises the
    # truthiness guard `if _raw_root` branch (not the `config.taskmaster is None` branch).
    # If the guard were removed, empty string would silently resolve to CWD, breaking the
    # task-927 short-circuit in _fetch_filtered_task_tree.
    harness_d = _make_harness_927(journal, event_buffer, mock_memory_service, '')
    assert harness_d._project_root == '', (
        f"empty-string project_root must stay ''; got {harness_d._project_root!r}"
    )
    assert harness_d.project_root == ''


@pytest.mark.asyncio
async def test_run_full_cycle_uses_configured_project_root_when_events_lack_override(
    journal, event_buffer, mock_memory_service
):
    """run_full_cycle should fall back to self._project_root, not project_id, when events lack _project_root."""
    harness = _make_harness_927(journal, event_buffer, mock_memory_service)

    # Push events with NO _project_root key in payload
    await event_buffer.push(_make_event('dark_factory'))
    await event_buffer.push(_make_event('dark_factory'))

    # Capture stage.project_root at the moment each stage.run fires
    captured_roots: list[str] = []

    async def capture_root(stage):
        captured_roots.append(stage.project_root)

    for stage in harness.stages:
        _mock_stage_run(stage, before_return=capture_root)

    await harness.run_full_cycle('dark_factory', 'buffer_size:2')

    assert len(captured_roots) == 3
    for root in captured_roots:
        assert root == '/abs/from/config', (
            f"Expected project_root='/abs/from/config' but got '{root}'"
            " — fallback should use self._project_root, not project_id"
        )


@pytest.mark.asyncio
async def test_run_full_cycle_event_project_root_wins_over_configured(
    journal, event_buffer, mock_memory_service
):
    """Event _project_root should override the configured project_root (precedence invariant).

    This pins the guarantee that a configured project_root is only a fallback:
    events that carry _project_root in their payload always take priority,
    regardless of what TaskmasterConfig.project_root says.
    """
    harness = _make_harness_927(journal, event_buffer, mock_memory_service, '/from/config')

    # Push events whose payload carries a DIFFERENT project root
    await event_buffer.push(_make_event_with_root('dark_factory', '/from/event'))
    await event_buffer.push(_make_event_with_root('dark_factory', '/from/event'))

    captured_roots: list[str] = []

    async def capture_root(stage):
        captured_roots.append(stage.project_root)

    for stage in harness.stages:
        _mock_stage_run(stage, before_return=capture_root)

    await harness.run_full_cycle('dark_factory', 'buffer_size:2')

    assert len(captured_roots) == 3
    for root in captured_roots:
        assert root == '/from/event', (
            f"Expected project_root='/from/event' (from event payload) but got '{root}'"
            " — event-provided root must win over the configured fallback"
        )


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

    @pytest.mark.asyncio
    async def test_fetch_filtered_task_tree_rejects_non_absolute_project_root(
        self,
        journal,
        event_buffer,
        mock_memory_service,
        caplog,
    ):
        """_fetch_filtered_task_tree pre-checks that project_root is absolute.

        When a non-absolute (relative) path is passed:
        (a) returns an empty FilteredTaskTree (degrades gracefully),
        (b) does NOT call taskmaster.get_tasks (pre-check short-circuits before
            any network call),
        (c) emits a WARNING containing the distinct marker 'non-absolute
            project_root' and the rejected path repr so operators can grep
            production logs to identify the failure mode.
        """
        import logging

        from fused_memory.reconciliation.task_filter import FilteredTaskTree

        harness = _make_test_harness(journal, event_buffer, mock_memory_service)

        with caplog.at_level(logging.WARNING):
            result = await harness._fetch_filtered_task_tree('.')

        # (a) graceful degradation — empty tree, no exception raised
        assert isinstance(result, FilteredTaskTree)
        assert result.active_tasks == []
        assert result.total_count == 0

        # (b) taskmaster.get_tasks must NOT be called (pre-check short-circuit)
        harness.taskmaster.get_tasks.assert_not_called()  # type: ignore[union-attr,attr-defined]

        # (c) distinct WARNING marker present in logs, including repr of the rejected path
        warning_msgs = [r.message for r in caplog.records if r.levelno >= logging.WARNING]
        assert any(
            'non-absolute project_root' in msg and repr('.') in msg
            for msg in warning_msgs
        ), (
            f"Expected WARNING containing 'non-absolute project_root' and repr(\".\") == \"'.'\";"
            f" got: {warning_msgs}"
        )

    @pytest.mark.asyncio
    async def test_fetch_filtered_task_tree_logs_raw_task_count(
        self,
        journal,
        event_buffer,
        mock_memory_service,
        caplog,
    ):
        """_fetch_filtered_task_tree emits a log with the raw task count after a successful fetch.

        The log record must contain the integer count of tasks returned by
        taskmaster and the project_root so operators can distinguish
        'get_tasks returned 0 raw tasks' (upstream Taskmaster issue) from
        'get_tasks returned N tasks but filter partitioned all into other'
        (task_filter regression).

        Updated in task-958: the log was promoted from DEBUG to INFO under the
        event marker 'reconciliation.task_tree_fetched' with raw_count and
        project_root in the extra dict.
        """
        import logging

        from fused_memory.reconciliation.task_filter import FilteredTaskTree

        harness = _make_test_harness(journal, event_buffer, mock_memory_service)

        # Four tasks with mixed statuses
        harness.taskmaster.get_tasks.return_value = {  # type: ignore[union-attr,attr-defined]
            'tasks': [
                {'id': 1, 'title': 'T1', 'status': 'in-progress', 'dependencies': []},
                {'id': 2, 'title': 'T2', 'status': 'done', 'dependencies': []},
                {'id': 3, 'title': 'T3', 'status': 'cancelled', 'dependencies': []},
                {'id': 4, 'title': 'T4', 'status': 'pending', 'dependencies': []},
            ]
        }

        with caplog.at_level(logging.DEBUG):
            result = await harness._fetch_filtered_task_tree('/abs/path')

        # (a) A log record at >= DEBUG level must contain the count 4 and project_root.
        #     The record is now at INFO level with raw_count=4 in the extra dict.
        fetched_records = [
            r for r in caplog.records
            if r.levelno >= logging.DEBUG
            and getattr(r, 'raw_count', None) == 4
            and getattr(r, 'project_root', None) == '/abs/path'
        ]
        assert fetched_records, (
            f"Expected a log record with raw_count=4 and project_root='/abs/path';"
            f" got records: {[r.__dict__ for r in caplog.records]}"
        )

        # (b) sanity: returned tree reflects the actual data
        assert isinstance(result, FilteredTaskTree)
        assert len(result.active_tasks) == 2   # in-progress + pending
        assert result.done_count == 1
        assert result.cancelled_count == 1

    @pytest.mark.asyncio
    async def test_fetch_filtered_task_tree_logs_info_when_taskmaster_none(
        self,
        journal,
        event_buffer,
        mock_memory_service,
        caplog,
    ):
        """_fetch_filtered_task_tree emits an INFO log when taskmaster is None.

        When taskmaster is None (disabled or not configured), the short-circuit
        must emit a distinct INFO-level marker so ops can grep logs and confirm
        the branch that fired rather than wondering why Stage 2 sees an empty tree.

        Asserts:
        (a) an INFO-level record exists with the marker
            'reconciliation.task_tree_taskmaster_disabled'
        (b) the project_root '/abs/path' appears in the record (via message or
            extra so structured-log tools can correlate it)
        (c) no WARNING-level records from this branch (the non-absolute-path
            warning must not fire here — different branch)
        """
        import logging

        from fused_memory.reconciliation.task_filter import FilteredTaskTree

        harness = _make_test_harness(journal, event_buffer, mock_memory_service)
        harness.taskmaster = None

        with caplog.at_level(logging.INFO):
            result = await harness._fetch_filtered_task_tree('/abs/path')

        # Still returns empty tree
        assert isinstance(result, FilteredTaskTree)
        assert result.active_tasks == []

        # (a) INFO record with distinct event marker
        info_records = [r for r in caplog.records if r.levelno == logging.INFO]
        assert any(
            'reconciliation.task_tree_taskmaster_disabled' in r.getMessage()
            for r in info_records
        ), (
            f"Expected INFO record containing 'reconciliation.task_tree_taskmaster_disabled';"
            f" got INFO messages: {[r.getMessage() for r in info_records]}"
        )

        # (b) project_root must appear somewhere in the record (message or extra)
        marker_record = next(
            r for r in info_records
            if 'reconciliation.task_tree_taskmaster_disabled' in r.getMessage()
        )
        record_repr = repr(marker_record.__dict__)
        assert '/abs/path' in record_repr or '/abs/path' in marker_record.getMessage(), (
            f"project_root '/abs/path' not found in record; record={record_repr}"
        )

        # (c) no WARNING from this branch (non-absolute-path warning belongs to a
        #     different code path)
        warning_records = [r for r in caplog.records if r.levelno >= logging.WARNING]
        assert not warning_records, (
            f"Expected no WARNING records from taskmaster-None branch; got: "
            f"{[r.getMessage() for r in warning_records]}"
        )

    @pytest.mark.asyncio
    async def test_fetch_filtered_task_tree_logs_info_when_project_root_empty(
        self,
        journal,
        event_buffer,
        mock_memory_service,
        caplog,
    ):
        """_fetch_filtered_task_tree emits an INFO log when project_root is empty string.

        When project_root is '' the short-circuit returns an empty tree without
        calling taskmaster.get_tasks.  Ops must be able to see this happening so
        they can distinguish 'project root never set' from a healthy-but-empty
        project in production logs.

        Asserts:
        (a) an INFO-level record with marker 'reconciliation.task_tree_empty_project_root'
        (b) taskmaster.get_tasks was NOT called (short-circuit still fires)
        """
        import logging

        from fused_memory.reconciliation.task_filter import FilteredTaskTree

        harness = _make_test_harness(journal, event_buffer, mock_memory_service)

        with caplog.at_level(logging.INFO):
            result = await harness._fetch_filtered_task_tree('')

        # Still returns empty tree
        assert isinstance(result, FilteredTaskTree)
        assert result.active_tasks == []

        # (a) INFO record with distinct event marker
        info_records = [r for r in caplog.records if r.levelno == logging.INFO]
        assert any(
            'reconciliation.task_tree_empty_project_root' in r.getMessage()
            for r in info_records
        ), (
            f"Expected INFO record containing 'reconciliation.task_tree_empty_project_root';"
            f" got INFO messages: {[r.getMessage() for r in info_records]}"
        )

        # (b) short-circuit must NOT call taskmaster.get_tasks
        harness.taskmaster.get_tasks.assert_not_called()  # type: ignore[union-attr,attr-defined]

    @pytest.mark.asyncio
    async def test_fetch_filtered_task_tree_logs_info_after_successful_fetch_with_both_counts(
        self,
        journal,
        event_buffer,
        mock_memory_service,
        caplog,
    ):
        """_fetch_filtered_task_tree emits an INFO log with both raw_count and total_count.

        After a successful get_tasks call the log must include:
        (a) the distinct event marker 'reconciliation.task_tree_fetched' at INFO level
        (b) raw_count = 4 (number of tasks before filtering)
        (c) total_count = 4 (post-filter total from FilteredTaskTree)
        (d) the project_root

        This gives ops the exact signal to distinguish:
          - raw=0, total=0  → Taskmaster returned empty (upstream issue)
          - raw>0, total=0  → filter_task_tree shape mismatch
          - raw>0, total>0  → data flowing correctly
        """
        import logging

        from fused_memory.reconciliation.task_filter import FilteredTaskTree

        harness = _make_test_harness(journal, event_buffer, mock_memory_service)

        # Four tasks with mixed statuses
        harness.taskmaster.get_tasks.return_value = {  # type: ignore[union-attr,attr-defined]
            'tasks': [
                {'id': 1, 'title': 'T1', 'status': 'in-progress', 'dependencies': []},
                {'id': 2, 'title': 'T2', 'status': 'done', 'dependencies': []},
                {'id': 3, 'title': 'T3', 'status': 'cancelled', 'dependencies': []},
                {'id': 4, 'title': 'T4', 'status': 'pending', 'dependencies': []},
            ]
        }

        with caplog.at_level(logging.INFO):
            result = await harness._fetch_filtered_task_tree('/abs/path')

        # Result is correct
        assert isinstance(result, FilteredTaskTree)
        assert len(result.active_tasks) == 2
        assert result.done_count == 1
        assert result.cancelled_count == 1

        # (a) INFO log with the distinct event marker
        info_records = [r for r in caplog.records if r.levelno == logging.INFO]
        fetched_records = [
            r for r in info_records
            if 'reconciliation.task_tree_fetched' in r.getMessage()
        ]
        assert fetched_records, (
            f"Expected INFO record containing 'reconciliation.task_tree_fetched';"
            f" got INFO messages: {[r.getMessage() for r in info_records]}"
        )

        rec = fetched_records[0]
        rec_dict = rec.__dict__

        # (b) raw_count = 4
        assert rec_dict.get('raw_count') == 4, (
            f"Expected raw_count=4 in log extra; got: {rec_dict}"
        )

        # (c) total_count = 4 (all tasks counted in total_count)
        assert rec_dict.get('total_count') == 4, (
            f"Expected total_count=4 in log extra; got: {rec_dict}"
        )

        # (d) project_root present
        assert rec_dict.get('project_root') == '/abs/path', (
            f"Expected project_root='/abs/path' in log extra; got: {rec_dict}"
        )

    @pytest.mark.asyncio
    async def test_fetch_filtered_task_tree_unwraps_data_wrapper_response_shape(
        self,
        journal,
        event_buffer,
        mock_memory_service,
        caplog,
    ):
        """_fetch_filtered_task_tree transparently unwraps the {'data': {...}} wrapper.

        Some Taskmaster MCP calls return {'data': {'tasks': [...]}} instead of
        the canonical {'tasks': [...]} shape (cf. _extract_task_dict in
        task_interceptor.py:2820).  If the harness does not unwrap this shape,
        filter_task_tree silently returns an empty FilteredTaskTree.

        After unwrapping, the behaviour must be:
        (a) result.active_tasks has the 3 active tasks (wrapper was unwrapped)
        (b) result.total_count == 3
        (c) INFO log fires with marker 'reconciliation.task_tree_fetched'
        (d) the log record carries extra field wrapper_unwrapped=True so ops can
            spot response-shape drift in production
        """
        import logging

        from fused_memory.reconciliation.task_filter import FilteredTaskTree

        harness = _make_test_harness(journal, event_buffer, mock_memory_service)

        # Wrapped response shape returned by some Taskmaster MCP calls
        harness.taskmaster.get_tasks.return_value = {  # type: ignore[union-attr,attr-defined]
            'data': {
                'tasks': [
                    {'id': 1, 'title': 'T1', 'status': 'in-progress', 'dependencies': []},
                    {'id': 2, 'title': 'T2', 'status': 'pending', 'dependencies': []},
                    {'id': 3, 'title': 'T3', 'status': 'in-progress', 'dependencies': []},
                ]
            }
        }

        with caplog.at_level(logging.INFO):
            result = await harness._fetch_filtered_task_tree('/abs/path')

        # (a) + (b) wrapper was unwrapped and filter_task_tree operated on inner dict
        assert isinstance(result, FilteredTaskTree)
        assert len(result.active_tasks) == 3, (
            f"Expected 3 active tasks after wrapper unwrap; got {len(result.active_tasks)}"
        )
        assert result.total_count == 3

        # (c) INFO log with fetch marker
        info_records = [r for r in caplog.records if r.levelno == logging.INFO]
        fetched_records = [
            r for r in info_records
            if 'reconciliation.task_tree_fetched' in r.getMessage()
        ]
        assert fetched_records, (
            f"Expected INFO record containing 'reconciliation.task_tree_fetched';"
            f" got INFO messages: {[r.getMessage() for r in info_records]}"
        )

        # (d) wrapper_unwrapped=True in extra dict
        rec = fetched_records[0]
        assert getattr(rec, 'wrapper_unwrapped', None) is True, (
            f"Expected wrapper_unwrapped=True in log extra; got: {rec.__dict__}"
        )

    @pytest.mark.asyncio
    async def test_fetch_filtered_task_tree_distinguishes_fetch_zero_from_filter_zero_in_logs(
        self,
        journal,
        event_buffer,
        mock_memory_service,
        caplog,
    ):
        """INFO log unambiguously distinguishes zero-from-upstream vs zero-from-filter.

        Two sub-scenarios:
        (a) Taskmaster returns genuinely empty tasks list:
            → raw_count=0, total_count=0 in log.
        (b) Taskmaster returns tasks but filter_task_tree partitions all into
            other_count (unknown status):
            → raw_count=1, total_count=1, result.other_count=1, active/done/cancelled empty.

        This is the exact signal Step 4 of the task description calls for:
        operators can read a single log line and know whether the zero came
        from upstream Taskmaster or from filter_task_tree's partitioning.
        """
        import logging

        harness = _make_test_harness(journal, event_buffer, mock_memory_service)

        # ── Sub-scenario (a): Taskmaster returned genuinely empty ──────────
        harness.taskmaster.get_tasks.return_value = {'tasks': []}  # type: ignore[union-attr,attr-defined]

        with caplog.at_level(logging.INFO):
            result_a = await harness._fetch_filtered_task_tree('/abs/path')

        info_records_a = [
            r for r in caplog.records
            if r.levelno == logging.INFO
            and 'reconciliation.task_tree_fetched' in r.getMessage()
        ]
        assert info_records_a, "Expected reconciliation.task_tree_fetched for empty-tasks scenario"
        rec_a = info_records_a[0]
        assert getattr(rec_a, 'raw_count', None) == 0, (
            f"Scenario (a): expected raw_count=0; got {rec_a.__dict__}"
        )
        assert getattr(rec_a, 'total_count', None) == 0, (
            f"Scenario (a): expected total_count=0; got {rec_a.__dict__}"
        )
        assert result_a.total_count == 0
        assert result_a.active_tasks == []

        # ── Sub-scenario (b): tasks returned but all unknown status ────────
        caplog.clear()
        harness.taskmaster.get_tasks.reset_mock()  # type: ignore[union-attr,attr-defined]
        harness.taskmaster.get_tasks.return_value = {  # type: ignore[union-attr,attr-defined]
            'tasks': [
                {'id': 1, 'title': 'T1', 'status': 'some-unknown-status', 'dependencies': []}
            ]
        }

        with caplog.at_level(logging.INFO):
            result_b = await harness._fetch_filtered_task_tree('/abs/path')

        info_records_b = [
            r for r in caplog.records
            if r.levelno == logging.INFO
            and 'reconciliation.task_tree_fetched' in r.getMessage()
        ]
        assert info_records_b, "Expected reconciliation.task_tree_fetched for unknown-status scenario"
        rec_b = info_records_b[0]
        assert getattr(rec_b, 'raw_count', None) == 1, (
            f"Scenario (b): expected raw_count=1; got {rec_b.__dict__}"
        )
        assert getattr(rec_b, 'total_count', None) == 1, (
            f"Scenario (b): expected total_count=1; got {rec_b.__dict__}"
        )
        assert result_b.total_count == 1
        assert result_b.other_count == 1
        assert result_b.active_tasks == []
        assert result_b.done_count == 0
        assert result_b.cancelled_count == 0


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
    """run_loop() calls release_stale_claims(0) once at startup (fast-restart safe).

    Cutoff is 0 (not a time-based horizon) so every currently-claimed row is
    released unconditionally.  The per-project reconciliation lock guarantees
    at most one active replayer per project, so there is nothing to race with at
    startup before any project loop has spawned.
    """
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
    # Cutoff must be 0 so even a freshly-claimed row is re-queued on fast restart.
    harness.buffer.release_stale_claims.assert_called_once_with(0.0)


@pytest.mark.asyncio
async def test_run_loop_fast_restart_releases_recent_claims(
    journal, event_buffer, mock_memory_service
):
    """A freshly-claimed row (claimed_at≈now) is still released on startup.

    cutoff=0 unconditionally re-queues every currently-claimed row, so a row
    claimed immediately before a crash is always available for the new process
    to pick up — a time-based cutoff would silently skip it.
    """
    import asyncio
    from unittest.mock import AsyncMock

    harness = _make_test_harness(journal, event_buffer, mock_memory_service)

    # Defer a write and immediately claim it — simulating what a dead process left
    # behind.  The row is claimed_at≈now; a time-based cutoff would not release
    # it, but cutoff=0 does.
    await event_buffer.defer_write('test-project', 'payload-a', 'cat', {})
    claimed_before = await event_buffer.claim_deferred_writes('test-project')
    assert len(claimed_before) == 1, 'precondition: row should be claimed'

    # Patch side-effect dependencies to avoid network/filesystem calls
    harness._recover_stale_runs = AsyncMock(return_value=None)
    harness._start_escalation_server = AsyncMock()
    harness._stop_escalation_server = AsyncMock()

    # Run the loop just long enough to execute the startup sweep, then let it
    # time out in the main loop body.
    with contextlib.suppress(TimeoutError):
        await asyncio.wait_for(harness.run_loop(), timeout=0.2)

    # The startup sweep must have released the freshly-claimed row so a new
    # claim_deferred_writes call returns it.
    reclaimed = await event_buffer.claim_deferred_writes('test-project')
    assert len(reclaimed) == 1, (
        'run_loop startup sweep should have re-queued the freshly-claimed row'
    )
    assert reclaimed[0]['content'] == 'payload-a'

    # release_stale_claims must also increment attempt_count so the poison-pill
    # mechanism (delete after _MAX_DEFERRED_WRITE_ATTEMPTS) works correctly across
    # restarts.  Use the debug accessor rather than a raw aiosqlite connection so
    # tests don't leak the _db_path attribute or the deferred_writes schema.
    _row = await event_buffer._debug_get_deferred_row(reclaimed[0]['id'])
    assert _row is not None
    assert _row['attempt_count'] == 1, (
        'release_stale_claims must increment attempt_count on re-queue '
        '(contract: event_buffer.py:702-707)'
    )


# ── Tests for AllAccountsCappedException deferral in run_full_cycle ────


@pytest.mark.asyncio
async def test_run_pipeline_defers_on_all_accounts_capped(
    journal, event_buffer, mock_memory_service, caplog
):
    """AllAccountsCappedException in run_full_cycle defers gracefully.

    Contract:
    (a) run_full_cycle returns without raising,
    (b) the run is marked 'failed' in the journal,
    (c) drained events are restored to the buffer,
    (d) no 'recon_failure' escalation is emitted,
    (e) a warning log contains 'all accounts capped'.

    Fails before impl because the generic `except Exception` handler currently
    re-raises and calls _escalate('recon_failure', ...).
    """
    import logging
    from unittest.mock import AsyncMock

    from shared.cli_invoke import AllAccountsCappedException

    harness = _make_test_harness(journal, event_buffer, mock_memory_service)

    # Seed the event buffer with enough events to trigger
    await event_buffer.push(_make_event())
    await event_buffer.push(_make_event())

    # Make stage 0 raise AllAccountsCappedException
    harness.stages[0].run = AsyncMock(
        side_effect=AllAccountsCappedException(
            retries=3, elapsed_secs=200.0, label='Reconciliation stage (sonnet)'
        )
    )

    # Spy on _escalate to capture all categories emitted
    escalate_calls: list[str] = []
    original_escalate = harness._escalate

    def capturing_escalate(category: str, *args, **kwargs) -> None:
        escalate_calls.append(category)
        original_escalate(category, *args, **kwargs)

    harness._escalate = capturing_escalate  # type: ignore[method-assign]

    with caplog.at_level(logging.WARNING):
        run = await harness.run_full_cycle('test-project', 'buffer_size:2')

    # (a) Call completes WITHOUT raising
    assert run is not None

    # (b) Run marked as 'failed'
    assert run.status in (RunStatus.failed, 'failed'), (
        f"Expected run.status='failed', got '{run.status}'"
    )

    # Also verify via the journal
    recent_runs = await journal.get_recent_runs('test-project', limit=5)
    assert len(recent_runs) >= 1
    assert recent_runs[0].status == 'failed'

    # (c) Events were restored via buffer.restore_drained
    stats = await event_buffer.get_buffer_stats('test-project')
    assert stats['size'] == 2, (
        f"Expected buffer size=2 after cap deferral (events restored), got {stats['size']}"
    )

    # (d) NO 'recon_failure' escalation
    assert 'recon_failure' not in escalate_calls, (
        f"Expected no 'recon_failure' escalation for cap deferral, got: {escalate_calls}"
    )

    # (e) Warning log includes 'all accounts capped'
    log_messages = [r.message.lower() for r in caplog.records]
    assert any('all accounts capped' in msg for msg in log_messages), (
        f'Expected log containing "all accounts capped", got: {log_messages}'
    )


# ── Tests for Task 927: BacklogIterator project_root fallback ─────────


@pytest.mark.asyncio
async def test_backlog_iterator_uses_harness_project_root_when_events_lack_override(
    journal, event_buffer, mock_memory_service
):
    """BacklogIterator.run should pass harness.project_root to ContextAssembler
    when peeked events carry no _project_root key in their payload."""
    # Push one event with NO _project_root in payload
    await event_buffer.push(_make_event('dark_factory'))

    harness = _make_harness_927(journal, event_buffer, mock_memory_service)

    captured: dict = {}

    # Stub ContextAssembler: records project_root kwarg; assemble returns events=[] to exit loop
    def fake_assembler_factory(memory_service, taskmaster, config, project_root=''):
        captured['project_root'] = project_root
        inst = MagicMock()
        inst.assemble = AsyncMock(return_value=AssembledPayload(events=[]))
        return inst

    with patch(
        'fused_memory.reconciliation.context_assembler.ContextAssembler',
        side_effect=fake_assembler_factory,
    ):
        iterator = BacklogIterator(harness.config, harness.journal, harness.buffer, harness)
        await iterator.run('dark_factory')

    assert 'project_root' in captured, 'ContextAssembler was never constructed — iterator may not have run'
    assert captured['project_root'] == '/abs/from/config', (
        f"Expected project_root='/abs/from/config' but got '{captured['project_root']}'"
        " — BacklogIterator fallback should use harness.project_root, not project_id"
    )


@pytest.mark.asyncio
async def test_backlog_iterator_peek_window_finds_later_project_root_override(
    journal, event_buffer, mock_memory_service
):
    """Regression guard: a ``_project_root`` override on a later buffered event must be
    found even when earlier events in the peek window lack the key (window >= N positions).

    Uses a 9+1 setup so the override sits at exactly the current limit — any reduction
    to the peek window will trip this test.
    """
    # peek_buffered orders by `timestamp ASC LIMIT ?` (FIFO). Push 9 events that
    # LACK _project_root with monotonically-increasing timestamps, then 1 event
    # carrying _project_root='/from/event' with the latest timestamp. With FIFO
    # peek, the override event is returned LAST — so the resolver only finds it
    # if the window is wide enough to accommodate all 10 buffered events.
    # Using 9+1=10 puts the override right at the current limit, so any
    # reduction of the peek window (even to 9) will cause this test to fail.
    # Anchor base_ts 60 seconds in the past so that peek_buffered's
    # `WHERE timestamp < cutoff` clause (cutoff ≈ datetime.now(UTC) at run() time)
    # includes all 10 events.  Explicit offsets avoid sub-microsecond tie flakiness.
    base_ts = datetime.now(UTC) - timedelta(seconds=60)
    for i in range(9):
        await event_buffer.push(ReconciliationEvent(
            id=str(uuid.uuid4()),
            type=EventType.episode_added,
            source=EventSource.agent,
            project_id='dark_factory',
            timestamp=base_ts + timedelta(seconds=i),
            payload={},  # NO _project_root key
        ))
    await event_buffer.push(ReconciliationEvent(
        id=str(uuid.uuid4()),
        type=EventType.task_status_changed,
        source=EventSource.agent,
        project_id='dark_factory',
        timestamp=base_ts + timedelta(seconds=20),  # latest — FIFO peek returns last
        payload={'_project_root': '/from/event', 'task_id': '1'},
    ))

    harness = _make_harness_927(journal, event_buffer, mock_memory_service)

    captured: dict = {}

    def fake_assembler_factory(memory_service, taskmaster, config, project_root=''):
        captured['project_root'] = project_root
        inst = MagicMock()
        inst.assemble = AsyncMock(return_value=AssembledPayload(events=[]))
        return inst

    with patch(
        'fused_memory.reconciliation.context_assembler.ContextAssembler',
        side_effect=fake_assembler_factory,
    ):
        iterator = BacklogIterator(harness.config, harness.journal, harness.buffer, harness)
        await iterator.run('dark_factory')

    assert 'project_root' in captured, (
        'ContextAssembler was never constructed — iterator may not have run'
    )
    assert captured['project_root'] == '/from/event', (
        f"Expected project_root='/from/event' but got '{captured['project_root']}'. "
        'The peek window must be wide enough to find a later-buffered _project_root '
        'override past 9 earlier events that lack the key. If this test now fails, '
        'the peek window has been tuned too narrow — consider whether 10 buffered '
        'events is an unreasonable lookback and adjust accordingly.'
    )


@pytest.mark.asyncio
async def test_backlog_iterator_event_project_root_wins_over_configured(
    journal, event_buffer, mock_memory_service
):
    """BacklogIterator.run should use the project_root from peeked events over
    the harness-configured fallback (regression guard for task-927 invariant).

    Parallel to test_backlog_iterator_uses_harness_project_root_when_events_lack_override
    but exercises the event-override path: events carry _project_root='/from/event'
    while the harness is configured with '/from/config'.  The event value must win.

    Peek-window semantics differ from run_full_cycle's full-drain coverage at
    test_harness.py:341, making this a distinct regression guard.
    """
    # Push two events WITH _project_root key — both within the peek window
    await event_buffer.push(_make_event_with_root('dark_factory', '/from/event'))
    await event_buffer.push(_make_event_with_root('dark_factory', '/from/event'))

    # Build harness with a different configured root
    harness = _make_harness_927(journal, event_buffer, mock_memory_service, '/from/config')

    captured: dict = {}

    def fake_assembler_factory(memory_service, taskmaster, config, project_root=''):
        captured['project_root'] = project_root
        inst = MagicMock()
        inst.assemble = AsyncMock(return_value=AssembledPayload(events=[]))
        return inst

    with patch(
        'fused_memory.reconciliation.context_assembler.ContextAssembler',
        side_effect=fake_assembler_factory,
    ):
        iterator = BacklogIterator(harness.config, harness.journal, harness.buffer, harness)
        await iterator.run('dark_factory')

    assert 'project_root' in captured, (
        'ContextAssembler was never constructed — iterator may not have run'
    )
    assert captured['project_root'] == '/from/event', (
        f"Expected project_root='/from/event' but got '{captured['project_root']}'"
        " — event-provided root must win over the configured fallback in BacklogIterator"
    )


@pytest.mark.asyncio
async def test_empty_fallback_resolves_and_short_circuits_filtered_task_tree(
    journal, event_buffer, mock_memory_service
):
    """_resolve_project_root with no-override events returns '' and
    _fetch_filtered_task_tree('') short-circuits without a taskmaster call.

    This chained assertion guards the full resolver → fetcher pipeline introduced
    by task-927.  task-927 replaced project_id (e.g. 'dark_factory') with '' as
    the fallback in _resolve_project_root.  That change is only 'strictly better'
    because downstream _fetch_filtered_task_tree short-circuits on empty strings.
    Splitting the assertion would leave a gap: a future refactor could reintroduce
    project_id as fallback and _fetch_filtered_task_tree would silently start
    hitting taskmaster again.

    test_harness.py:1827 (test_returns_empty_when_empty_project_root) covers the
    fetch-level short-circuit in isolation; this test pins the resolver ↔ fetcher
    contract so the full pipeline is protected.
    """
    from fused_memory.reconciliation.task_filter import FilteredTaskTree

    # project_root=None → config.taskmaster=None → harness._project_root == ''
    # harness.taskmaster = AsyncMock() (injected mock) so we can assert it was NOT called
    harness = _make_harness_927(journal, event_buffer, mock_memory_service, project_root=None)

    # Events with NO _project_root key — triggers the fallback path
    events = [_make_event('dark_factory'), _make_event('dark_factory')]

    # (a) resolver returns '' — pins post-927 invariant (NOT 'dark_factory')
    resolved = harness._resolve_project_root(events)
    assert resolved == '', (
        f"_resolve_project_root returned '{resolved}' but expected '' — "
        "regression to old project_id fallback detected (task-927 invariant violated)"
    )

    # (b) fetcher returns empty tree when project_root is ''
    result = await harness._fetch_filtered_task_tree('')
    assert isinstance(result, FilteredTaskTree)
    assert result.active_tasks == []

    # (c) taskmaster.get_tasks was never called (short-circuit on empty project_root)
    harness.taskmaster.get_tasks.assert_not_called()  # type: ignore[union-attr,attr-defined]


@pytest.mark.asyncio
async def test_run_full_cycle_with_relative_config_and_memory_only_events_still_fetches_tree(
    journal, event_buffer, mock_memory_service
):
    """End-to-end regression: relative project_root='.' in config + memory-style events
    (no _project_root payload) must still produce a non-empty FilteredTaskTree in Stage 2.

    Replicates the production symptom from task 958:
    - config.taskmaster.project_root resolves to '.' when PROJECT_ROOT env is unset,
    - memory-service events (episode_added/memory_added) never carry _project_root,
    - so _resolve_project_root falls back to self._project_root.
    Before this fix, self._project_root stored '.' (relative) which bypassed the
    empty-string short-circuit but was rejected by TaskmasterBackend's absolute-path
    validator, giving a silent empty tree to Stage 2.
    After this fix, __init__ resolves '.' to str(Path('.').resolve()) so the full
    pipeline succeeds.

    Pins the init-normalization → resolver → fetcher pipeline end-to-end.
    """
    from pathlib import Path

    from fused_memory.reconciliation.task_filter import FilteredTaskTree

    # Build harness with RELATIVE project_root='.' — init must resolve to absolute
    harness = _make_harness_927(journal, event_buffer, mock_memory_service, project_root='.')
    expected_abs = str(Path('.').resolve())
    assert harness._project_root == expected_abs, (
        "Pre-condition: harness._project_root must be absolute after init"
    )

    # Push two memory-style events (no _project_root in payload)
    await event_buffer.push(_make_event('dark_factory'))
    await event_buffer.push(_make_event('dark_factory'))

    # Taskmaster returns one active task when called with the resolved absolute path
    harness.taskmaster.get_tasks.return_value = {  # type: ignore[union-attr,attr-defined]
        'tasks': [
            {'id': 1, 'title': 'Task A', 'status': 'in-progress', 'dependencies': []},
        ]
    }

    # Capture Stage 2 (TaskKnowledgeSync) filtered_task_tree at call time
    captured_s2: dict = {}

    async def capture_stage2(stage):
        captured_s2['filtered_task_tree'] = getattr(stage, 'filtered_task_tree', None)

    from fused_memory.reconciliation.stages.task_knowledge_sync import TaskKnowledgeSync

    for stage in harness.stages:
        if isinstance(stage, TaskKnowledgeSync):
            _mock_stage_run(stage, before_return=capture_stage2)
        else:
            _mock_stage_run(stage)

    await harness.run_full_cycle('dark_factory', 'buffer_size:2')

    # (a) taskmaster.get_tasks was called exactly once
    harness.taskmaster.get_tasks.assert_called_once()  # type: ignore[union-attr,attr-defined]

    # (b) the project_root arg was the resolved absolute path (not '.')
    call_kwargs = harness.taskmaster.get_tasks.call_args.kwargs  # type: ignore[union-attr,attr-defined]
    assert call_kwargs.get('project_root') == expected_abs, (
        f"get_tasks called with project_root={call_kwargs.get('project_root')!r}, "
        f"expected {expected_abs!r} — init normalization must resolve relative '.' to absolute"
    )

    # (c) Stage 2's filtered_task_tree at run time is non-empty
    assert 'filtered_task_tree' in captured_s2, (
        "Stage 2 run was never captured — check _mock_stage_run wiring"
    )
    stage2_tree = captured_s2['filtered_task_tree']
    assert isinstance(stage2_tree, FilteredTaskTree), (
        f"Stage 2 filtered_task_tree must be FilteredTaskTree; got {type(stage2_tree)}"
    )
    assert stage2_tree.total_count > 0, (
        f"Stage 2 must receive a non-empty FilteredTaskTree (total_count={stage2_tree.total_count}); "
        "got empty tree — init normalization or fetcher pipeline is broken"
    )
