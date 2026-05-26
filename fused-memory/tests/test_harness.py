"""Tests for reconciliation harness (pipeline orchestration)."""

import asyncio
import contextlib
import logging
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
    # task 1143: pre-populate _known_projects so pre-flight does not raise before the stage
    harness._known_projects['test-project'] = '/tmp/test-project'

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
async def test_full_cycle_uses_registry_for_project_root(
    journal, event_buffer, mock_memory_service
):
    """Harness binds stage.project_root from _known_projects[project_id], not event payloads.

    The events here carry _project_root='/home/leo/src/dark-factory' which happens to equal
    the registry value for dark_factory — the assertion still holds under the task-1143
    contract because the registry (not the payload) is the authoritative source.
    See test_known_project_roots_wins_over_event_payload for the explicit payload-vs-registry
    conflict case.
    """
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
    # task 1143: pre-populate _known_projects so pre-flight does not raise.
    # Events carry _project_root='/home/leo/src/dark-factory' which matches
    # the registry value for dark_factory, so the assertion below still holds
    # under the new contract (registry value, not event payload, is the source).
    harness._known_projects['dark_factory'] = '/home/leo/src/dark-factory'

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
    """run_full_cycle uses the registry-derived root for the cycle's project_id,
    even when events carry no _project_root payload.

    task 1143: updated from '/abs/from/config' to '/abs/from/dark-factory' so that
    resolve_project_id derives 'dark_factory' from the basename and _known_projects
    contains the key.  The new contract is: registry wins (not event payload, not
    self._project_root fallback).  This test pins: events-without-payload still gets
    the correct root via registry.
    """
    harness = _make_harness_927(journal, event_buffer, mock_memory_service, '/abs/from/dark-factory')

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
        assert root == '/abs/from/dark-factory', (
            f"Expected project_root='/abs/from/dark-factory' (registry-derived) but got '{root}'"
            " — run_full_cycle must use the registry-bound root for the cycle's project_id"
        )


@pytest.mark.asyncio
async def test_known_project_roots_wins_over_event_payload(
    journal, event_buffer, mock_memory_service
):
    """KNOWN_PROJECT_ROOTS registry wins over event payload _project_root (precedence invariant).

    task 1143: replaces test_run_full_cycle_event_project_root_wins_over_configured.
    Under the new contract, the registry is the single source of truth.  Events
    carrying _project_root='/from/event' while the registry maps dark_factory to
    '/abs/from/dark-factory' must yield '/abs/from/dark-factory' — the event payload
    is informational only and must not override the registry.
    """
    harness = _make_harness_927(journal, event_buffer, mock_memory_service, '/abs/from/dark-factory')

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
        assert root == '/abs/from/dark-factory', (
            f"Expected project_root='/abs/from/dark-factory' (registry-bound) but got '{root}'"
            " — KNOWN_PROJECT_ROOTS must win over event payload _project_root (task 1143)"
        )
        assert root != '/from/event', (
            "Event payload _project_root='/from/event' must not override the registry"
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

    task 1143: pre-populates _known_projects so callers can invoke
    run_full_cycle('dark_factory', ...) or run_full_cycle('test-project', ...)
    without triggering the pre-flight ValueError from _known_project_root_for.
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
    # task 1143: inject known projects so run_full_cycle pre-flight does not raise
    # for the project_ids used across the existing test suite.
    harness._known_projects = {
        'dark_factory': '/home/leo/src/dark-factory',
        'test-project': '/tmp/test-project',
    }
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
        'parent-run-id',
        findings,
        tier,
        project_root='/tmp/test-project',
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
    # task 1143: inject registry entry so _known_project_root_for('my-project') succeeds.
    harness._known_projects['my-project'] = '/srv/my-project'

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
        'parent-run-id',
        findings,
        tier,
        project_root='/srv/my-project',
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
        'parent-run-id',
        findings,
        tier,
        project_root='/tmp/test-project',
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
        'parent-run-id',
        findings,
        tier,
        project_root='/tmp/test-project',
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
        empty_root_records = [
            r for r in info_records
            if 'reconciliation.task_tree_empty_project_root' in r.getMessage()
        ]
        assert empty_root_records, (
            f"Expected INFO record containing 'reconciliation.task_tree_empty_project_root';"
            f" got INFO messages: {[r.getMessage() for r in info_records]}"
        )
        assert len(empty_root_records) == 1, (
            f"Expected exactly one such record; got {len(empty_root_records)}"
        )

        # (a2) project_root_repr must be in extra dict; repr('') == "''" vs repr(None) == 'None'
        # disambiguates empty-string from None at the log level.
        rec = empty_root_records[0]
        _MISSING = object()
        assert getattr(rec, 'project_root_repr', _MISSING) == repr(''), (
            f"Expected project_root_repr={repr('')!r} in extra dict;"
            f" got record __dict__: {rec.__dict__}"
        )

        # (b) short-circuit must NOT call taskmaster.get_tasks
        harness.taskmaster.get_tasks.assert_not_called()  # type: ignore[union-attr,attr-defined]

    @pytest.mark.asyncio
    async def test_fetch_filtered_task_tree_logs_debug_after_successful_happy_path_fetch_with_both_counts(
        self,
        journal,
        event_buffer,
        mock_memory_service,
        caplog,
    ):
        """_fetch_filtered_task_tree emits a DEBUG log with both raw_count and total_count.

        After a successful non-anomalous get_tasks call the log must include:
        (a) the distinct event marker 'reconciliation.task_tree_fetched' at DEBUG level
        (b) raw_count = 4 (number of tasks before filtering)
        (c) total_count = 4 (post-filter total from FilteredTaskTree)
        (d) the project_root

        Under the task-985 policy, INFO is reserved for anomalies; healthy
        fetches (raw>0, total>0) stay at DEBUG.  The structured fields are
        still present so operators can grep them at DEBUG when needed.

        This gives ops the exact signal to distinguish:
          - raw=0, total=0  → Taskmaster returned empty (upstream issue)
          - raw>0, total=0  → filter_task_tree shape mismatch (anomaly → INFO)
          - raw>0, total>0  → data flowing correctly (happy-path → DEBUG)
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

        # Result is correct
        assert isinstance(result, FilteredTaskTree)
        assert len(result.active_tasks) == 2
        assert result.done_count == 1
        assert result.cancelled_count == 1

        # (a) DEBUG log with the distinct event marker
        debug_records = [r for r in caplog.records if r.levelno == logging.DEBUG]
        fetched_records = [
            r for r in debug_records
            if 'reconciliation.task_tree_fetched' in r.getMessage()
        ]
        assert fetched_records, (
            f"Expected DEBUG record containing 'reconciliation.task_tree_fetched';"
            f" got DEBUG messages: {[r.getMessage() for r in debug_records]}"
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
    async def test_fetch_filtered_task_tree_happy_path_logs_at_debug_not_info(
        self,
        journal,
        event_buffer,
        mock_memory_service,
        caplog,
    ):
        """Happy-path fetch (raw_count>0, total_count>0) emits DEBUG, not INFO.

        Under the task-985 policy, INFO is reserved for anomalies.  A healthy
        fetch (raw>0, total>0) is non-anomalous, so the log level must be DEBUG.

        Asserts:
        (a) at least one record with marker 'reconciliation.task_tree_fetched' at
            DEBUG level, carrying raw_count=4, total_count=4, project_root='/abs/path'
        (b) NO record with marker 'reconciliation.task_tree_fetched' at INFO level
        """
        import logging

        from fused_memory.reconciliation.task_filter import FilteredTaskTree

        harness = _make_test_harness(journal, event_buffer, mock_memory_service)

        # Four tasks with mixed statuses — raw_count=4, total_count=4, no anomaly
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

        assert isinstance(result, FilteredTaskTree)

        # (a) must have a DEBUG record with the marker and correct structured fields
        debug_fetched = [
            r for r in caplog.records
            if r.levelno == logging.DEBUG
            and 'reconciliation.task_tree_fetched' in r.getMessage()
        ]
        assert debug_fetched, (
            f"Expected DEBUG record with 'reconciliation.task_tree_fetched';"
            f" got records: {[(r.levelno, r.getMessage()) for r in caplog.records]}"
        )
        rec = debug_fetched[0]
        assert getattr(rec, 'raw_count', None) == 4, (
            f"Expected raw_count=4 in DEBUG record; got {rec.__dict__}"
        )
        assert getattr(rec, 'total_count', None) == 4, (
            f"Expected total_count=4 in DEBUG record; got {rec.__dict__}"
        )
        assert getattr(rec, 'project_root', None) == '/abs/path', (
            f"Expected project_root='/abs/path' in DEBUG record; got {rec.__dict__}"
        )

        # (b) no INFO record with the marker (INFO is reserved for anomalies)
        info_fetched = [
            r for r in caplog.records
            if r.levelno == logging.INFO
            and 'reconciliation.task_tree_fetched' in r.getMessage()
        ]
        assert not info_fetched, (
            f"Expected NO INFO record with 'reconciliation.task_tree_fetched';"
            f" got: {[r.getMessage() for r in info_fetched]}"
        )

    @pytest.mark.asyncio
    async def test_fetch_filtered_task_tree_raw_gt_zero_total_zero_logs_info_anomaly(
        self,
        journal,
        event_buffer,
        mock_memory_service,
        caplog,
    ):
        """raw_count>0 AND total_count==0 is an anomaly and must emit INFO.

        When taskmaster returns tasks but every top-level entry fails the
        defensive isinstance(task, dict) guard in filter_task_tree (task_filter.py:191
        — e.g. bare ints, malformed entries), raw_count>0 while total_count==0.
        This signals a complete dict-guard drop — an anomaly operators should see
        without enabling DEBUG logging.

        Scope is intentionally narrow per the task-985 policy: PARTIAL drops
        (raw_count >> total_count when only some entries are non-dict, or all
        survivors land in other_count via unknown status) remain at DEBUG by
        design.  See sibling test
        test_fetch_filtered_task_tree_distinguishes_fetch_zero_from_filter_zero_in_logs
        for the contrapositive — the anomaly predicate is False there even though
        raw_count>0, because total_count>0.

        Construction: pass bare ints as task elements so filter drops them all.
        len(tasks_data['tasks'])==3 → raw_count=3; filter skips all ints →
        total_count=0.

        Asserts:
        (a) one record with marker 'reconciliation.task_tree_fetched' at INFO
            level, carrying raw_count=3, total_count=0
        (b) no DEBUG record with that marker (DEBUG is for non-anomaly paths)
        """
        import logging

        harness = _make_test_harness(journal, event_buffer, mock_memory_service)

        # Bare ints — filter skips all of them via isinstance(task, dict) guard
        harness.taskmaster.get_tasks.return_value = {  # type: ignore[union-attr,attr-defined]
            'tasks': [1, 2, 3]
        }

        with caplog.at_level(logging.DEBUG):
            await harness._fetch_filtered_task_tree('/abs/path')

        # (a) INFO record with marker and correct structured fields
        info_fetched = [
            r for r in caplog.records
            if r.levelno == logging.INFO
            and 'reconciliation.task_tree_fetched' in r.getMessage()
        ]
        assert info_fetched, (
            f"Expected INFO record with 'reconciliation.task_tree_fetched' for anomaly;"
            f" got records: {[(r.levelno, r.getMessage()) for r in caplog.records]}"
        )
        rec = info_fetched[0]
        assert getattr(rec, 'raw_count', None) == 3, (
            f"Expected raw_count=3; got {rec.__dict__}"
        )
        assert getattr(rec, 'total_count', None) == 0, (
            f"Expected total_count=0; got {rec.__dict__}"
        )

        # (b) no DEBUG record with the marker
        debug_fetched = [
            r for r in caplog.records
            if r.levelno == logging.DEBUG
            and 'reconciliation.task_tree_fetched' in r.getMessage()
        ]
        assert not debug_fetched, (
            f"Expected NO DEBUG record with 'reconciliation.task_tree_fetched';"
            f" got: {[r.getMessage() for r in debug_fetched]}"
        )

    @pytest.mark.asyncio
    async def test_fetch_filtered_task_tree_distinguishes_fetch_zero_from_filter_zero_in_logs(
        self,
        journal,
        event_buffer,
        mock_memory_service,
        caplog,
    ):
        """DEBUG log unambiguously distinguishes zero-from-upstream vs zero-from-filter.

        Under the task-985 policy, both sub-scenarios below are non-anomalies
        (the anomaly predicate is raw_count>0 AND total_count==0), so both emit
        at DEBUG.  Structured fields (raw_count, total_count) carry the signal
        that differentiates them — operators can grep at DEBUG when needed.

        Two sub-scenarios:
        (a) Taskmaster returns genuinely empty tasks list:
            → raw_count=0, total_count=0 in log — empty-but-healthy (DEBUG).
        (b) Taskmaster returns tasks but filter_task_tree partitions all into
            other_count (unknown status):
            → raw_count=1, total_count=1, result.other_count=1, active/done/cancelled empty
            (anomaly predicate raw>0 AND total==0 is False here since total==1 → DEBUG).

        Operators can read a single log line and know whether the zero came
        from upstream Taskmaster or from filter_task_tree's partitioning
        because the extra dict carries both raw_count and total_count.
        """
        import logging

        harness = _make_test_harness(journal, event_buffer, mock_memory_service)

        # ── Sub-scenario (a): Taskmaster returned genuinely empty ──────────
        harness.taskmaster.get_tasks.return_value = {'tasks': []}  # type: ignore[union-attr,attr-defined]

        with caplog.at_level(logging.DEBUG):
            result_a = await harness._fetch_filtered_task_tree('/abs/path')

        debug_records_a = [
            r for r in caplog.records
            if r.levelno == logging.DEBUG
            and 'reconciliation.task_tree_fetched' in r.getMessage()
        ]
        assert debug_records_a, "Expected reconciliation.task_tree_fetched at DEBUG for empty-tasks scenario"
        rec_a = debug_records_a[0]
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

        with caplog.at_level(logging.DEBUG):
            result_b = await harness._fetch_filtered_task_tree('/abs/path')

        debug_records_b = [
            r for r in caplog.records
            if r.levelno == logging.DEBUG
            and 'reconciliation.task_tree_fetched' in r.getMessage()
        ]
        assert debug_records_b, "Expected reconciliation.task_tree_fetched at DEBUG for unknown-status scenario"
        rec_b = debug_records_b[0]
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
        """run_full_cycle calls _fetch_filtered_task_tree exactly once with the registry-bound root.

        task 1143: the project_root used by _fetch_filtered_task_tree comes from
        _known_project_root_for(project_id), not from event payload _project_root.
        Event payload _project_root is now informational-only.
        """
        from unittest.mock import AsyncMock

        from fused_memory.reconciliation.task_filter import FilteredTaskTree

        harness = _make_test_harness(journal, event_buffer, mock_memory_service)
        harness._fetch_filtered_task_tree = AsyncMock(return_value=FilteredTaskTree())

        for stage in harness.stages:
            _mock_stage_run(stage)

        # Event payload _project_root is now ignored; registry binding wins (task 1143).
        event = _make_event()
        event.payload['_project_root'] = '/my/project'  # intentionally wrong — must be ignored

        await harness.run_full_cycle('test-project', 'test-trigger', events=[event])

        # _fetch_filtered_task_tree must use the registry-bound root, not the event payload.
        expected_root = harness._known_projects['test-project']  # '/tmp/test-project'
        harness._fetch_filtered_task_tree.assert_called_once_with(expected_root)  # type: ignore[attr-defined]

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
            'parent-run-id',
            findings,
            tier,
            project_root='/tmp/test-project',
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
            'parent-run-id',
            findings,
            tier,
            project_root='/tmp/test-project',
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
                'parent-run-id',
                findings,
                tier,
                project_root='/tmp/test-project',
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
            'parent-run-id',
            findings,
            tier,
            project_root='/tmp/test-project',
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
        # task 1143: inject registry entry so _known_project_root_for('test-project') succeeds.
        harness._known_projects['test-project'] = '/my/project'

        expected_tree = self._make_tree()
        harness._fetch_filtered_task_tree = AsyncMock(return_value=expected_tree)

        _mock_stage_run(stages[0])
        _mock_stage_run(stages[1])
        _mock_stage_run(stages[2])

        findings = [_make_s3_findings()[0]]
        tier = TierConfig(model='sonnet', episode_limit=100, memory_limit=200)
        # No filtered_task_tree kwarg — method must fall back to _fetch_filtered_task_tree.
        # project_root='/my/project' is threaded explicitly (injected above as the registry value,
        # but now passed directly as the threaded kwarg rather than re-resolved at call time).
        await harness._run_remediation_pass(
            'test-project',
            'parent-run-id',
            findings,
            tier,
            project_root='/my/project',
        )

        # _fetch_filtered_task_tree must be called with the threaded project_root value.
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
    # judge.initialize() does a real SQLite query; mock it to avoid timing
    # flakiness in slow environments (freethreaded Python, heavy parallel runs).
    if harness.judge is not None:
        harness.judge.initialize = AsyncMock()

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


# ── Stale-run reaper: instance-scoped lock check ─────────────────────────────


@pytest.mark.asyncio
async def test_recover_stale_runs_skips_when_same_instance_holds_lock(
    journal, event_buffer, mock_memory_service
):
    """Reaper must NOT recover a run owned by the instance that still holds the lock.

    This is the legitimate long-running cycle case: same EventBuffer.instance_id
    on both the run row and the active reconciliation_locks row.
    """
    harness = _make_test_harness(journal, event_buffer, mock_memory_service)
    project_id = 'test-project'
    cutoff = harness.config.stale_run_recovery_seconds

    # Run started long enough ago to be considered stale by age, owned by the
    # currently-live EventBuffer instance.
    run = ReconciliationRun(
        id='run-same-instance',
        project_id=project_id,
        run_type=RunType.full,
        trigger_reason='unit-test',
        started_at=datetime.now(UTC) - timedelta(seconds=cutoff * 2),
        status=RunStatus.running,
        instance_id=event_buffer.instance_id,
    )
    await journal.start_run(run)

    # The same live instance currently holds the project lock.
    acquired = await event_buffer.mark_run_active(project_id)
    assert acquired is True

    await harness._recover_stale_runs()

    after = await journal.get_run('run-same-instance')
    assert after is not None
    assert after.status == RunStatus.running, (
        'Run owned by the same live instance must not be reaped'
    )
    assert '_error' not in after.stage_reports


@pytest.mark.asyncio
async def test_recover_stale_runs_recovers_when_different_instance_holds_lock(
    journal, event_buffer, mock_memory_service
):
    """Reaper must recover an orphan even when another instance now holds the lock.

    This is the 2026-05-06 incident scenario: the original instance died, a fresh
    instance acquired the lock for its own cycle, and the project-scoped reaper
    check used to shield the orphan forever.
    """
    harness = _make_test_harness(journal, event_buffer, mock_memory_service)
    project_id = 'test-project'
    cutoff = harness.config.stale_run_recovery_seconds

    run = ReconciliationRun(
        id='run-orphan-A',
        project_id=project_id,
        run_type=RunType.full,
        trigger_reason='unit-test',
        started_at=datetime.now(UTC) - timedelta(seconds=cutoff * 2),
        status=RunStatus.running,
        instance_id='dead-instance-A',
    )
    await journal.start_run(run)

    # The live instance — different from 'dead-instance-A' — holds the lock now.
    acquired = await event_buffer.mark_run_active(project_id)
    assert acquired is True
    assert event_buffer.instance_id != 'dead-instance-A'

    # Defer one write so we can assert _replay_deferred_writes ran (the reaper
    # invokes it after marking the run failed).
    await event_buffer.defer_write(project_id, 'replayed-content', 'cat', {})
    mock_memory_service.add_memory = AsyncMock()

    await harness._recover_stale_runs()

    after = await journal.get_run('run-orphan-A')
    assert after is not None
    assert after.status == RunStatus.failed
    err = after.stage_reports.get('_error')
    assert isinstance(err, dict)
    assert err.get('error_type') == 'StaleRunRecovery'

    # Deferred write was replayed.
    mock_memory_service.add_memory.assert_awaited()


@pytest.mark.asyncio
async def test_recover_stale_runs_recovers_pre_migration_run_with_null_instance(
    journal, event_buffer, mock_memory_service
):
    """Pre-migration runs (instance_id IS NULL) must be reaped even when locked.

    The conservative choice: a NULL instance_id means we cannot prove the run is
    owned by the current process, so we treat it as an orphan.  This is the
    precise behaviour we need to clean up legacy rows that motivated the fix.
    """
    harness = _make_test_harness(journal, event_buffer, mock_memory_service)
    project_id = 'test-project'
    cutoff = harness.config.stale_run_recovery_seconds

    run = ReconciliationRun(
        id='run-pre-migration',
        project_id=project_id,
        run_type=RunType.full,
        trigger_reason='unit-test',
        started_at=datetime.now(UTC) - timedelta(seconds=cutoff * 2),
        status=RunStatus.running,
        instance_id=None,
    )
    await journal.start_run(run)

    # Lock is held — by the live instance, doesn't matter who.
    acquired = await event_buffer.mark_run_active(project_id)
    assert acquired is True

    await harness._recover_stale_runs()

    after = await journal.get_run('run-pre-migration')
    assert after is not None
    assert after.status == RunStatus.failed
    err = after.stage_reports.get('_error')
    assert isinstance(err, dict)
    assert err.get('error_type') == 'StaleRunRecovery'


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


# ── Tests for Task 1143: BacklogIterator hard-bind via KNOWN_PROJECT_ROOTS ──


@pytest.mark.asyncio
async def test_backlog_iterator_run_hard_binds_via_known_projects(
    journal, event_buffer, mock_memory_service
):
    """BacklogIterator.run must derive project_root from _known_project_root_for, not from event payload.

    Build a harness with _known_projects containing autopilot_video and dark_factory.
    Push events for autopilot_video with _project_root='/wrong/path' to prove the
    event payload is ignored. Verify ContextAssembler receives the registry-bound root
    '/home/leo/src/autopilot-video', NOT '/wrong/path' and NOT dark-factory's path.

    This test fails before step-8 because BacklogIterator.run still calls
    self.harness._resolve_project_root(peeked_for_root) which returns the event payload.
    """
    harness = _make_harness_with_known_projects(
        journal, event_buffer, mock_memory_service,
        {
            'autopilot_video': '/home/leo/src/autopilot-video',
            'dark_factory': '/home/leo/src/dark-factory',
        },
    )

    wrong_path = '/wrong/path'
    # Push events with deliberately wrong _project_root in payload.
    for _ in range(3):
        await event_buffer.push(_make_event_with_root('autopilot_video', wrong_path))

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
        await iterator.run('autopilot_video')

    assert 'project_root' in captured, (
        'ContextAssembler was never constructed — BacklogIterator.run may not have run'
    )
    expected = '/home/leo/src/autopilot-video'
    assert captured['project_root'] == expected, (
        f"Expected project_root={expected!r} (registry-bound) "
        f"but got {captured['project_root']!r} — "
        "event payload _project_root must be ignored by BacklogIterator"
    )
    assert captured['project_root'] != '/home/leo/src/dark-factory', (
        "BacklogIterator must not receive dark-factory's path for autopilot_video"
    )
    assert captured['project_root'] != wrong_path, (
        "BacklogIterator must ignore event payload _project_root='/wrong/path'"
    )


# ── Tests for Task 927: BacklogIterator project_root fallback ─────────


@pytest.mark.asyncio
async def test_backlog_iterator_uses_harness_project_root_when_events_lack_override(
    journal, event_buffer, mock_memory_service
):
    """BacklogIterator.run uses the registry-bound root for dark_factory even when
    events carry no _project_root payload.

    task 1143: project_root is now hard-bound via _known_project_root_for(project_id).
    Event payload _project_root is informational only.  This test preserves the
    'no payload → still gets a valid root' invariant, but the source is now
    _known_projects, not harness.project_root.
    """
    # Push one event with NO _project_root in payload
    await event_buffer.push(_make_event('dark_factory'))

    harness = _make_harness_927(journal, event_buffer, mock_memory_service)
    # task 1143: inject dark_factory into _known_projects so _known_project_root_for
    # succeeds.  The value matches the harness-configured root to preserve the
    # original test's intent: when events lack the key, the configured root is used.
    harness._known_projects['dark_factory'] = '/abs/from/config'

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
        f"Expected project_root='/abs/from/config' (registry-bound) but got '{captured['project_root']}'"
    )


@pytest.mark.asyncio
async def test_backlog_iterator_registry_wins_regardless_of_mixed_event_payloads(
    journal, event_buffer, mock_memory_service
):
    """BacklogIterator uses the registry-bound root even when buffered events have
    mixed payloads (some with _project_root, some without).

    task 1143: replaced the old peek-window-based resolver with a direct lookup in
    _known_project_root_for.  Event payload _project_root is now informational only —
    the registry is always authoritative regardless of event content.

    This replaces the former test_backlog_iterator_peek_window_finds_later_project_root_override
    which guarded the dead-code peek-window width behavior.
    """
    base_ts = datetime.now(UTC) - timedelta(seconds=60)
    # Mix: 2 events without _project_root, 1 with a wrong value — registry must win.
    for i in range(2):
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
        timestamp=base_ts + timedelta(seconds=10),
        payload={'_project_root': '/from/event', 'task_id': '1'},
    ))

    harness = _make_harness_927(journal, event_buffer, mock_memory_service)
    # task 1143: inject the registry entry for dark_factory.
    harness._known_projects['dark_factory'] = '/abs/from/config'

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
    assert captured['project_root'] == '/abs/from/config', (
        f"Expected registry-bound project_root='/abs/from/config' but got '{captured['project_root']}'. "
        "Event payload _project_root must be ignored (task 1143 contract)."
    )
    assert captured['project_root'] != '/from/event', (
        "BacklogIterator must not use event payload _project_root='/from/event' — registry wins"
    )


@pytest.mark.asyncio
async def test_known_project_roots_wins_over_event_payload_in_backlog_iterator(
    journal, event_buffer, mock_memory_service
):
    """BacklogIterator.run uses the registry-bound root, ignoring event payload _project_root.

    task 1143: replaces test_backlog_iterator_event_project_root_wins_over_configured.
    Under the new KNOWN_PROJECT_ROOTS contract, the registry is the single source of truth.
    Events carrying _project_root='/from/event' while the registry maps dark_factory to
    '/from/registry' must yield '/from/registry' — the event payload is informational only.

    Parallel to test_known_project_roots_wins_over_event_payload for run_full_cycle.
    """
    # Push two events WITH _project_root key — event payload must be IGNORED
    await event_buffer.push(_make_event_with_root('dark_factory', '/from/event'))
    await event_buffer.push(_make_event_with_root('dark_factory', '/from/event'))

    # Build harness with a different configured root and inject registry with a third path
    harness = _make_harness_927(journal, event_buffer, mock_memory_service, '/from/config')
    harness._known_projects['dark_factory'] = '/from/registry'

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
    assert captured['project_root'] == '/from/registry', (
        f"Expected registry-bound project_root='/from/registry' but got '{captured['project_root']}'. "
        "KNOWN_PROJECT_ROOTS must win over both event payload and harness config (task 1143)."
    )
    assert captured['project_root'] != '/from/event', (
        "BacklogIterator must not use event payload _project_root='/from/event'"
    )
    assert captured['project_root'] != '/from/config', (
        "BacklogIterator must not use harness-configured project_root='/from/config'"
    )


@pytest.mark.asyncio
async def test_fetch_filtered_task_tree_short_circuits_on_empty_project_root(
    journal, event_buffer, mock_memory_service
):
    """_fetch_filtered_task_tree('') short-circuits without a taskmaster call.

    task 1143: _resolve_project_root was deleted; this test retains the fetcher
    short-circuit guard (parts (b)+(c) of the former
    test_empty_fallback_resolves_and_short_circuits_filtered_task_tree) which
    is still meaningful — if project_root is '' for any reason, the fetcher must
    not call taskmaster.get_tasks.

    test_harness.py:1827 (test_returns_empty_when_empty_project_root) covers the
    same in isolation; keeping both guards different entry paths.
    """
    from fused_memory.reconciliation.task_filter import FilteredTaskTree

    harness = _make_harness_927(journal, event_buffer, mock_memory_service, project_root=None)

    # fetcher returns empty tree when project_root is ''
    result = await harness._fetch_filtered_task_tree('')
    assert isinstance(result, FilteredTaskTree)
    assert result.active_tasks == []

    # taskmaster.get_tasks was never called (short-circuit on empty project_root)
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
    # task 1143: pre-populate _known_projects so _known_project_root_for('dark_factory')
    # returns the same resolved-absolute path.  This preserves the test's original intent
    # (init resolves '.' to absolute; that path is what reaches get_tasks) while satisfying
    # the new KNOWN_PROJECT_ROOTS contract.
    harness._known_projects['dark_factory'] = expected_abs

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


# ---------------------------------------------------------------------------
# Task 1053 — Harness.drain() idle short-circuit
# ---------------------------------------------------------------------------


class TestHarnessDrainIdleShortCircuit:
    """drain() must emit 'Harness fully drained' synchronously when idle."""

    def test_drain_emits_when_no_tasks_ever_spawned(
        self, journal, event_buffer, mock_memory_service, caplog
    ):
        """drain() must synchronously log 'Harness fully drained' when _project_tasks is empty.

        Case: no project loops have ever been spawned (constructor default: {}).
        """
        harness = _make_test_harness(journal, event_buffer, mock_memory_service)
        # _project_tasks starts as {} — no mutation needed.

        with caplog.at_level(logging.INFO, logger='fused_memory.reconciliation.harness'):
            harness.drain()

        drained_records = [
            r for r in caplog.records if 'Harness fully drained' in r.message
        ]
        assert drained_records, (
            f"Expected at least one log record containing 'Harness fully drained — safe to restart' "
            f"but got records: {[r.message for r in caplog.records]}"
        )

    def test_drain_emits_when_only_done_tasks(
        self, journal, event_buffer, mock_memory_service, caplog
    ):
        """drain() must synchronously log 'Harness fully drained' when all tasks are done.

        Case: _project_tasks contains one entry whose .done() returns True (loops ran
        but have been completed; not yet reaped by the main loop).
        """
        harness = _make_test_harness(journal, event_buffer, mock_memory_service)
        done_task = MagicMock(spec=asyncio.Task)
        done_task.done.return_value = True
        harness._project_tasks['some-project'] = done_task

        with caplog.at_level(logging.INFO, logger='fused_memory.reconciliation.harness'):
            harness.drain()

        drained_records = [
            r for r in caplog.records if 'Harness fully drained' in r.message
        ]
        assert drained_records, (
            f"Expected at least one log record containing 'Harness fully drained — safe to restart' "
            f"but got records: {[r.message for r in caplog.records]}"
        )

    def test_drain_suppresses_fully_drained_when_loop_active(
        self, journal, event_buffer, mock_memory_service, caplog
    ):
        """drain() must NOT emit 'Harness fully drained' when a project loop is still running.

        The main reconciliation loop emits the marker after loops finish (existing
        behaviour). Synchronous emission in drain() must be suppressed when at least
        one _project_tasks entry has .done() == False.
        """
        harness = _make_test_harness(journal, event_buffer, mock_memory_service)

        active_task = MagicMock(spec=asyncio.Task)
        active_task.done.return_value = False
        harness._project_tasks['some-project'] = active_task

        with caplog.at_level(logging.INFO, logger='fused_memory.reconciliation.harness'):
            harness.drain()

        drained_records = [
            r for r in caplog.records if 'Harness fully drained' in r.message
        ]
        assert not drained_records, (
            f"Expected NO 'Harness fully drained' log when a project loop is active, "
            f"but got: {[r.message for r in drained_records]}"
        )

    def test_drain_twice_idle_emits_exactly_one_marker(
        self, journal, event_buffer, mock_memory_service, caplog
    ):
        """drain() called twice on an idle harness must emit 'Harness fully drained' exactly once.

        The second call must hit the 'Harness already draining' early-return path,
        not re-emit the marker.  This pins the drain()-twice contract against future
        refactors that might move the marker emission above or outside the early-return.
        """
        harness = _make_test_harness(journal, event_buffer, mock_memory_service)

        with caplog.at_level(logging.INFO, logger='fused_memory.reconciliation.harness'):
            harness.drain()
            harness.drain()  # second call — must hit early return

        drained_records = [
            r for r in caplog.records if 'Harness fully drained' in r.message
        ]
        already_draining_records = [
            r for r in caplog.records if 'Harness already draining' in r.message
        ]
        assert len(drained_records) == 1, (
            f"Expected exactly 1 'Harness fully drained' record but got "
            f"{len(drained_records)}: {[r.message for r in drained_records]}"
        )
        assert len(already_draining_records) >= 1, (
            f"Expected at least 1 'Harness already draining' record but got "
            f"{len(already_draining_records)}: {[r.message for r in caplog.records]}"
        )

    @pytest.mark.asyncio
    async def test_main_loop_does_not_emit_drain_progress_after_idle_drain(
        self, journal, event_buffer, mock_memory_service, caplog, monkeypatch
    ):
        """After drain() fires the marker synchronously, run_loop() must stay silent.

        When drain() is called on an idle harness it emits the one-shot marker and
        sets _drain_complete_logged=True.  Subsequent run_loop() iterations must NOT
        emit any 'Harness draining:' progress messages — neither the fully-drained
        marker (gated by the one-shot flag) nor the 'N project loop(s) still running'
        progress message (gated by the else-branch that only fires while loops are
        active).

        We patch the module-local _sleep to yield immediately so the loop body runs
        many iterations within the 0.2 s window, maximising the chance of catching
        spurious emissions.
        """
        real_sleep = asyncio.sleep

        async def fast_sleep(seconds: float) -> None:
            await real_sleep(0)

        # Patch the module-local _sleep binding — true module-scoped patch that
        # does not leak to other asyncio users in the same process.
        monkeypatch.setattr('fused_memory.reconciliation.harness._sleep', fast_sleep)

        harness = _make_test_harness(journal, event_buffer, mock_memory_service)

        # Stub side-effect dependencies so the loop body runs without network calls
        harness._recover_stale_runs = AsyncMock(return_value=None)
        harness._start_escalation_server = AsyncMock()
        harness._stop_escalation_server = AsyncMock()
        harness.buffer.get_active_projects = AsyncMock(return_value=[])

        with caplog.at_level(logging.INFO, logger='fused_memory.reconciliation.harness'):
            # Idle path: drain() fires the marker synchronously; _drain_complete_logged=True
            harness.drain()
            # Run the main loop; with fast_sleep many iterations execute in 0.2 s.
            with contextlib.suppress(TimeoutError):
                await asyncio.wait_for(harness.run_loop(), timeout=0.2)

        # Guard: ensure the loop body actually ran enough iterations to make the
        # absence-assertions below meaningful.  On a heavily-loaded CI host the 0.2 s
        # window could expire before any iteration runs, making the absence-assertions
        # pass vacuously.  This fails loudly with a diagnostic when that happens.
        #
        # Why _recover_stale_runs.call_count is a reliable iteration proxy:
        # `await self._recover_stale_runs()` is the FIRST awaited call inside the
        # per-iteration try-block of run_loop()'s while-True loop (harness.py:588).
        # It runs unconditionally on every iteration regardless of self._draining —
        # unlike buffer.get_active_projects, which is gated by `if not self._draining:`
        # (harness.py:591) and would have call_count == 0 in this test (drain() is
        # called before run_loop()).  REFACTOR NOTE: if _recover_stale_runs is ever
        # made conditional or moved below another awaited call, update this proxy to
        # something that remains unconditional at the top of each iteration.
        assert harness._recover_stale_runs.call_count >= 3, (
            f"Loop body must run multiple times to make the absence assertion meaningful; "
            f"only ran {harness._recover_stale_runs.call_count} times"
        )
        # After an idle drain all subsequent 'Harness draining:' progress messages
        # must be absent — the gate restructuring ensures the else-branch (which emits
        # 'N project loop(s) still running') can only fire while loops are active.
        draining_progress_records = [
            r for r in caplog.records if 'Harness draining:' in r.message
        ]
        assert len(draining_progress_records) == 0, (
            f"Expected NO 'Harness draining:' progress records after idle drain() "
            f"but got {len(draining_progress_records)}: "
            f"{[r.message for r in draining_progress_records]}"
        )
        # The marker must have been emitted exactly once (by drain(), not by run_loop())
        drained_records = [
            r for r in caplog.records if 'Harness fully drained' in r.message
        ]
        assert len(drained_records) == 1, (
            f"Expected exactly 1 'Harness fully drained' record but got "
            f"{len(drained_records)}: {[r.message for r in drained_records]}"
        )


# ── Deferred-write replay deduplication (Fix 2) ───────────────────────────────


class TestReplayDeferredWritesCompletionSummaryDedup:
    """Tests for the completion-summary dedup check in _replay_deferred_writes."""

    @pytest.mark.asyncio
    async def test_skip_on_prior_match(
        self, journal, event_buffer, mock_memory_service, caplog
    ):
        """When a prior done-summary exists in Mem0, the deferred write is skipped.

        Integration test: asserts the harness wires through find_prior_memory correctly
        — a matching prior causes the write to be skipped and the row deleted.
        Coercion and kwargs-forwarding contracts are covered in test_mem0_dedup.py.
        """
        from unittest.mock import MagicMock

        prior_result = MagicMock()
        prior_result.metadata = {
            'task_id': '517',
            'transition': 'done',
            'source': 'targeted_reconciliation',
        }
        mock_memory_service.search = AsyncMock(return_value=[prior_result])

        harness = _make_test_harness(journal, event_buffer, mock_memory_service)

        await event_buffer.defer_write(
            'test-project',
            "Task 'X' completed. Summary here.",
            'observations_and_summaries',
            {
                'task_id': '517',
                'transition': 'done',
                'source': 'targeted_reconciliation',
                '_deferred': True,
            },
        )

        with caplog.at_level(logging.INFO):
            await harness._replay_deferred_writes('test-project')

        # (a) add_memory was NOT called — dedup should have skipped the write
        mock_memory_service.add_memory.assert_not_called()

        # (b) the row was deleted from event_buffer
        await event_buffer.release_stale_claims(0.0)
        remaining = await event_buffer.claim_deferred_writes('test-project')
        assert len(remaining) == 0, f'Expected no remaining rows but got {len(remaining)}'

        # (c) INFO log mentions skipping task 517
        skip_records = [
            r for r in caplog.records
            if '517' in r.message and r.levelno == logging.INFO
        ]
        assert skip_records, (
            f'Expected an INFO log mentioning task 517 but got: '
            f'{[r.message for r in caplog.records]}'
        )

    @pytest.mark.asyncio
    async def test_no_transition_bypasses_dedup(
        self, journal, event_buffer, mock_memory_service
    ):
        """Deferred writes with no transition field bypass the dedup check entirely."""
        harness = _make_test_harness(journal, event_buffer, mock_memory_service)

        await event_buffer.defer_write(
            'test-project',
            'Some non-done write',
            'observations_and_summaries',
            {},  # no task_id, no transition
        )

        await harness._replay_deferred_writes('test-project')

        # (a) search was NOT called — bypass condition triggered before any search
        mock_memory_service.search.assert_not_called()

        # (b) add_memory WAS called once with the deferred-write content
        mock_memory_service.add_memory.assert_called_once()
        call_content = mock_memory_service.add_memory.call_args.kwargs.get('content')
        assert call_content == 'Some non-done write'

        # (c) the row was deleted from event_buffer
        await event_buffer.release_stale_claims(0.0)
        remaining = await event_buffer.claim_deferred_writes('test-project')
        assert len(remaining) == 0

    @pytest.mark.asyncio
    async def test_task_id_only_no_transition_bypasses_dedup(
        self, journal, event_buffer, mock_memory_service
    ):
        """transition='done' AND task_id are both required; task_id alone must bypass dedup.

        Validates the `transition == 'done'` clause of the conjunction independently:
        a future refactor that drops the `transition` guard would be caught here.
        """
        harness = _make_test_harness(journal, event_buffer, mock_memory_service)

        await event_buffer.defer_write(
            'test-project',
            'Blocked write for task 1',
            'observations_and_summaries',
            {'task_id': '1'},  # transition absent
        )

        await harness._replay_deferred_writes('test-project')

        # search must NOT be called — only task_id present, transition guard fails
        mock_memory_service.search.assert_not_called()
        # write must proceed normally
        mock_memory_service.add_memory.assert_called_once()
        content = mock_memory_service.add_memory.call_args.kwargs.get('content')
        assert content == 'Blocked write for task 1'

    @pytest.mark.asyncio
    async def test_transition_done_no_task_id_bypasses_dedup(
        self, journal, event_buffer, mock_memory_service
    ):
        """transition='done' AND task_id are both required; transition alone must bypass dedup.

        Validates the `tid` clause of the conjunction independently:
        a future refactor that drops the `tid` guard would be caught here.
        """
        harness = _make_test_harness(journal, event_buffer, mock_memory_service)

        await event_buffer.defer_write(
            'test-project',
            'Done write without task_id',
            'observations_and_summaries',
            {'transition': 'done'},  # task_id absent
        )

        await harness._replay_deferred_writes('test-project')

        # search must NOT be called — only transition present, tid guard fails
        mock_memory_service.search.assert_not_called()
        # write must proceed normally
        mock_memory_service.add_memory.assert_called_once()
        content = mock_memory_service.add_memory.call_args.kwargs.get('content')
        assert content == 'Done write without task_id'

    @pytest.mark.asyncio
    async def test_metadata_predicate_filters_wrong_transition(
        self, journal, event_buffer, mock_memory_service
    ):
        """When the dedup search returns rows matching task_id='517' but with
        transition != 'done' (or missing entirely), the predicate filters them all out,
        so the deferred completion-summary write proceeds via add_memory.

        Regression coverage for harness.py:460-476 — without this test, dropping the
        transition guard from the kind dict passed to find_prior_memory would silently
        start skipping legitimate writes whenever any past row for the task existed.
        """
        from unittest.mock import MagicMock

        # Three rows whose task_id matches but whose transition value is wrong.
        # They exercise the clause independently and include the missing-key path.
        blocked_result = MagicMock()
        blocked_result.metadata = {
            'task_id': '517',
            'transition': 'blocked',
            'source': 'targeted_reconciliation',
        }
        cancelled_result = MagicMock()
        cancelled_result.metadata = {
            'task_id': '517',
            'transition': 'cancelled',
            'source': 'targeted_reconciliation',
        }
        no_transition_result = MagicMock()
        no_transition_result.metadata = {
            'task_id': '517',
            'source': 'targeted_reconciliation',
            # transition key absent — exercises the dict.get('transition', '') default
        }

        mock_memory_service.search = AsyncMock(
            return_value=[blocked_result, cancelled_result, no_transition_result]
        )

        harness = _make_test_harness(journal, event_buffer, mock_memory_service)

        await event_buffer.defer_write(
            'test-project',
            "Task 'X' completed. Summary here.",
            'observations_and_summaries',
            {
                'task_id': '517',
                'transition': 'done',
                'source': 'targeted_reconciliation',
                '_deferred': True,
            },
        )

        await harness._replay_deferred_writes('test-project')

        # (a) search WAS called — transition='done' AND tid present, so dedup branch entered
        mock_memory_service.search.assert_called_once()

        # (b) add_memory WAS called — predicate filtered all rows out, write proceeds
        mock_memory_service.add_memory.assert_called_once()
        call_content = mock_memory_service.add_memory.call_args.kwargs.get('content')
        assert call_content == "Task 'X' completed. Summary here."

        # (c) the deferred row was deleted from event_buffer
        await event_buffer.release_stale_claims(0.0)
        remaining = await event_buffer.claim_deferred_writes('test-project')
        assert len(remaining) == 0, f'Expected no remaining rows but got {len(remaining)}'

    @pytest.mark.asyncio
    async def test_no_prior_match_falls_through_to_add_memory(
        self, journal, event_buffer, mock_memory_service
    ):
        """When Mem0 search returns no prior done-summary, the write proceeds normally."""
        mock_memory_service.search = AsyncMock(return_value=[])
        harness = _make_test_harness(journal, event_buffer, mock_memory_service)

        await event_buffer.defer_write(
            'test-project',
            'Task 999 completed.',
            'observations_and_summaries',
            {'task_id': '999', 'transition': 'done'},
        )

        await harness._replay_deferred_writes('test-project')

        # (a) search WAS called with project_id and '999' and 'completion'
        mock_memory_service.search.assert_called_once()
        search_kwargs = mock_memory_service.search.call_args.kwargs
        assert search_kwargs.get('project_id') == 'test-project'
        query = search_kwargs.get('query', '')
        assert '999' in query and 'completion' in query, (
            f"Expected query to mention '999' and 'completion', got: {query!r}"
        )

        # (b) add_memory WAS called once (no prior match — write proceeds)
        mock_memory_service.add_memory.assert_called_once()
        call_content = mock_memory_service.add_memory.call_args.kwargs.get('content')
        assert call_content == 'Task 999 completed.'

        # (c) the row was deleted from event_buffer
        await event_buffer.release_stale_claims(0.0)
        remaining = await event_buffer.claim_deferred_writes('test-project')
        assert len(remaining) == 0

    @pytest.mark.asyncio
    async def test_search_exception_falls_through_to_add_memory(
        self, journal, event_buffer, mock_memory_service, caplog
    ):
        """If the dedup search raises, the write still proceeds (degrade to no-dedup)."""
        mock_memory_service.search = AsyncMock(side_effect=RuntimeError('Mem0 down'))
        harness = _make_test_harness(journal, event_buffer, mock_memory_service)

        await event_buffer.defer_write(
            'test-project',
            'Task 777 completed.',
            'observations_and_summaries',
            {'task_id': '777', 'transition': 'done'},
        )

        with caplog.at_level(logging.WARNING):
            # (a) must NOT raise
            await harness._replay_deferred_writes('test-project')

        # (b) add_memory WAS called once — search failure falls through to write
        mock_memory_service.add_memory.assert_called_once()
        call_content = mock_memory_service.add_memory.call_args.kwargs.get('content')
        assert call_content == 'Task 777 completed.'

        # (c) the row was deleted from event_buffer
        await event_buffer.release_stale_claims(0.0)
        remaining = await event_buffer.claim_deferred_writes('test-project')
        assert len(remaining) == 0

        # (d) WARNING log mentions task 777 and the search failure
        warn_records = [
            r for r in caplog.records
            if r.levelno == logging.WARNING and '777' in r.message
        ]
        assert warn_records, (
            f'Expected a WARNING log mentioning task 777 but got: '
            f'{[r.message for r in caplog.records]}'
        )


# ── Tests for Task 1143: KNOWN_PROJECT_ROOTS hard-bind ───────────────────────

_FIVE_PROJECT_MAP = {
    'autopilot_video': '/home/leo/src/autopilot-video',
    'reify': '/home/leo/src/reify',
    'dark_factory': '/home/leo/src/dark-factory',
    'autotrade': '/home/leo/src/autotrade',
    'know_live': '/home/leo/src/know-live',
}


def _make_harness_with_known_projects(
    journal, event_buffer, mock_memory_service, known_projects: dict
):
    """Build a harness and monkeypatch _known_projects for task-1143 tests."""
    harness = _make_test_harness(journal, event_buffer, mock_memory_service)
    harness._known_projects = dict(known_projects)
    return harness


class TestKnownProjectRootFor:
    """Tests for ReconciliationHarness._known_project_root_for (task 1143)."""

    @pytest.mark.asyncio
    async def test_known_project_root_for_returns_mapped_root(
        self, journal, event_buffer, mock_memory_service
    ):
        """_known_project_root_for returns the correct project_root for each project_id."""
        harness = _make_harness_with_known_projects(
            journal, event_buffer, mock_memory_service, _FIVE_PROJECT_MAP
        )

        assert harness._known_project_root_for('autopilot_video') == '/home/leo/src/autopilot-video'
        assert harness._known_project_root_for('reify') == '/home/leo/src/reify'
        assert harness._known_project_root_for('dark_factory') == '/home/leo/src/dark-factory'
        assert harness._known_project_root_for('autotrade') == '/home/leo/src/autotrade'
        assert harness._known_project_root_for('know_live') == '/home/leo/src/know-live'

    @pytest.mark.asyncio
    async def test_known_project_root_for_raises_for_unknown_project_id(
        self, journal, event_buffer, mock_memory_service
    ):
        """_known_project_root_for raises ValueError for an unknown project_id.

        The error message must include both the unknown project_id and the sorted list
        of known project_ids so the operator can immediately diagnose the misconfiguration.
        """
        harness = _make_harness_with_known_projects(
            journal, event_buffer, mock_memory_service, _FIVE_PROJECT_MAP
        )

        with pytest.raises(ValueError) as exc_info:
            harness._known_project_root_for('not_a_real_project')

        err_msg = str(exc_info.value)
        assert 'not_a_real_project' in err_msg, (
            f"Error message must include the unknown project_id; got: {err_msg!r}"
        )
        # At least one known project_id must appear so the operator can see what's registered.
        # We intentionally avoid pinning the exact message format (repr vs bullet list, etc.)
        # — only the *presence* of diagnostic context is the contract.
        assert any(pid in err_msg for pid in sorted(_FIVE_PROJECT_MAP)), (
            f"Error message must include at least one known project_id; got: {err_msg!r}"
        )


@pytest.mark.asyncio
async def test_run_full_cycle_pre_flight_raises_before_side_effects(
    journal, event_buffer, mock_memory_service
):
    """Pre-flight _known_project_root_for raises before any drain or journal mutation.

    Pins that a misconfigured project_id never causes a partial cycle:
    (a) ValueError is raised,
    (b) events remain in the buffer (drain was NOT called),
    (c) no journal run row was created (start_run was NOT called).
    """
    harness = _make_harness_with_known_projects(
        journal, event_buffer, mock_memory_service,
        {'dark_factory': '/home/leo/src/dark-factory'},  # no autopilot_video entry
    )

    # Push 2 events for autopilot_video (which is NOT in _known_projects).
    await event_buffer.push(_make_event('autopilot_video'))
    await event_buffer.push(_make_event('autopilot_video'))

    # (a) ValueError with the unknown project_id in the message
    with pytest.raises(ValueError, match='autopilot_video'):
        await harness.run_full_cycle('autopilot_video', 'buffer_size:2')

    # (b) Events NOT drained — buffer still shows size==2
    stats = await event_buffer.get_buffer_stats('autopilot_video')
    assert stats['size'] == 2, (
        f"Expected buffer size==2 (events not drained) but got {stats['size']}"
        " — pre-flight must run before drain"
    )

    # (c) No journal start_run side effect — get_recent_runs returns no row
    recent_runs = await journal.get_recent_runs('autopilot_video', limit=1)
    assert len(recent_runs) == 0, (
        f"Expected no journal row but got {len(recent_runs)} row(s)"
        " — pre-flight must run before journal.start_run"
    )


@pytest.mark.parametrize(
    'project_id,expected_root',
    [
        ('autopilot_video', '/home/leo/src/autopilot-video'),
        ('reify', '/home/leo/src/reify'),
        ('autotrade', '/home/leo/src/autotrade'),
        ('know_live', '/home/leo/src/know-live'),
        ('dark_factory', '/home/leo/src/dark-factory'),
    ],
)
@pytest.mark.asyncio
async def test_run_full_cycle_hard_binds_project_root_via_known_projects(
    project_id, expected_root, journal, event_buffer, mock_memory_service
):
    """run_full_cycle must use the registry-derived root regardless of event payload.

    The harness's _known_projects dict is the single source of truth (task 1143).
    Even when events carry _project_root='/some/other/path', the stage must receive
    the registry-bound root. Also verifies non-dark-factory projects never receive
    dark-factory's path.

    This test fails before step-4 because _resolve_project_root honours the event payload.
    """
    harness = _make_harness_with_known_projects(
        journal, event_buffer, mock_memory_service, _FIVE_PROJECT_MAP
    )
    # The harness's configured _project_root is dark-factory (typical multi-project deployment).
    harness._project_root = '/home/leo/src/dark-factory'

    wrong_path = '/some/other/wrong/path'

    # Push events with deliberately wrong _project_root to verify it is ignored.
    await event_buffer.push(_make_event_with_root(project_id, wrong_path))
    await event_buffer.push(_make_event_with_root(project_id, wrong_path))

    captured_roots: list[str] = []

    async def capture_root(stage):
        captured_roots.append(stage.project_root)

    for stage in harness.stages:
        _mock_stage_run(stage, before_return=capture_root)

    await harness.run_full_cycle(project_id, 'buffer_size:2')

    assert len(captured_roots) == 3, f"Expected 3 captured roots, got {len(captured_roots)}"
    for root in captured_roots:
        assert root == expected_root, (
            f"project_id={project_id!r}: expected root {expected_root!r} but got {root!r} "
            f"— registry binding must win over event payload"
        )
        # Non-dark-factory projects must not receive dark-factory's path
        if project_id != 'dark_factory':
            assert root != '/home/leo/src/dark-factory', (
                f"project_id={project_id!r} must not receive dark-factory's path; got {root!r}"
            )
        # Wrong event payload must not leak through
        assert root != wrong_path, (
            f"Event payload _project_root must be ignored; stage got {root!r}"
        )


# ── Tests for Task 1143 step-9: _run_remediation_pass defense-in-depth ───


@pytest.mark.asyncio
async def test_remediation_pass_uses_threaded_project_root_over_registry(
    journal, event_buffer, mock_memory_service
):
    """_run_remediation_pass uses the caller-threaded project_root, not the registry value.

    Task 1163 reverses the task-1143 defense-in-depth contract: the caller now threads
    project_root as a required kwarg.  This test verifies that the threaded value wins
    even when _known_projects contains a DIFFERENT value for the same project_id.

    The harness has _known_projects = {'reify': '/path/B', 'dark_factory': ...} (wrong
    registry value).  We call _run_remediation_pass with project_root='/path/A' (the
    correct threaded value from run_full_cycle's pre-cycle resolution).

    All three stage.project_root captures must equal '/path/A', proving the threaded
    value wins and the registry is not consulted by _run_remediation_pass itself.
    """
    from fused_memory.reconciliation.harness import TierConfig

    harness = _make_harness_with_known_projects(
        journal, event_buffer, mock_memory_service,
        {
            'reify': '/path/B',  # WRONG value — must NOT be used by _run_remediation_pass
            'dark_factory': '/home/leo/src/dark-factory',
        },
    )

    stages = harness._make_stages()
    harness._make_stages = lambda: stages

    captured_roots: dict[str, str] = {}
    for stage in stages:
        stage_name = type(stage).__name__

        async def capture(s, _name=stage_name):
            captured_roots[_name] = s.project_root

        _mock_stage_run(stage, before_return=capture)

    findings = [_make_s3_findings()[0]]  # one actionable finding
    tier = TierConfig(model='sonnet', episode_limit=100, memory_limit=200)

    await harness._run_remediation_pass(
        project_id='reify',
        parent_run_id='test-parent-run',
        findings=findings,
        tier=tier,
        project_root='/path/A',  # threaded caller value — must win over registry '/path/B'
    )

    assert len(captured_roots) == 3, (
        f"Expected 3 stage captures, got {len(captured_roots)}: {list(captured_roots)}"
    )
    expected = '/path/A'
    for stage_name, root in captured_roots.items():
        assert root == expected, (
            f"{stage_name}: expected threaded project_root={expected!r} "
            f"but got {root!r} — _run_remediation_pass must use the caller-supplied "
            f"project_root kwarg, not _known_project_root_for('reify') (task 1163)"
        )


@pytest.mark.asyncio
async def test_remediation_uses_threaded_project_root_not_mutated_registry(
    journal, event_buffer, mock_memory_service
):
    """Regression: remediation sees the project_root resolved at run_full_cycle entry,
    not the registry value that may have changed after cycle entry.

    Task 1163: _run_remediation_pass previously re-resolved project_root via
    _known_project_root_for(project_id) at its own call site (line 1143).  This
    meant a mid-cycle mutation of _known_projects could give remediation a different
    root than the primary cycle used — a cross-contamination risk.

    After the fix, project_root is resolved ONCE by run_full_cycle at line 868
    (before any side-effects) and threaded through _maybe_remediate →
    _run_remediation_pass as a required kwarg.

    Setup:
    - _known_projects = {'reify': '/path/A', 'dark_factory': ...}
    - Stage 3's before_return mutates _known_projects['reify'] = '/path/B'
      (simulates a mid-cycle registry update) AND returns one actionable finding
      to trigger the remediation pass.
    - Stages 0 and 1 capture stage.project_root when mutation_done is True
      (i.e. during the remediation pass only).
    - Stage 3 captures stage.project_root when mutation_done is already True
      (i.e. the remediation-pass Stage 3 run).

    Assertion: all three remediation-pass captures equal '/path/A'
    (the pre-mutation resolution), NOT '/path/B' (the mutated registry value).
    """
    harness = _make_harness_with_known_projects(
        journal, event_buffer, mock_memory_service,
        {
            'reify': '/path/A',
            'dark_factory': '/home/leo/src/dark-factory',
        },
    )

    # Pin the same stage instances so before_return callbacks registered here
    # remain effective during the remediation pass (same _make_stages shim used
    # by ~10 other tests in this file).
    stages = harness._make_stages()
    harness._make_stages = lambda: stages

    # Push one event so run_full_cycle has something to drain for 'reify'.
    await event_buffer.push(_make_event('reify'))

    mutation_done: list[bool] = [False]
    remediation_roots: list[str] = []

    async def capture_if_post_mutation(stage):
        """Stages 0 and 1: capture project_root only during the remediation pass."""
        if mutation_done[0]:
            remediation_roots.append(stage.project_root)

    async def stage3_callback(stage):
        """Stage 3: on the PRIMARY pass mutate the registry; on REMEDIATION pass capture."""
        if not mutation_done[0]:
            # Primary-cycle Stage 3: simulate mid-cycle registry mutation and flag it.
            mutation_done[0] = True
            harness._known_projects['reify'] = '/path/B'
        else:
            # Remediation-pass Stage 3: capture the project_root that was threaded in.
            remediation_roots.append(stage.project_root)

    _mock_stage_run(stages[0], before_return=capture_if_post_mutation)
    _mock_stage_run(stages[1], before_return=capture_if_post_mutation)
    # Stage 3 returns one actionable finding to trigger _maybe_remediate.
    _mock_stage_run(stages[2], items_flagged=[_make_s3_findings()[0]], before_return=stage3_callback)

    await harness.run_full_cycle('reify', 'buffer_size:1')

    assert len(remediation_roots) == 3, (
        f'Expected 3 project_root captures from the remediation pass '
        f'(one per stage), got {len(remediation_roots)}: {remediation_roots!r}'
    )
    for root in remediation_roots:
        assert root == '/path/A', (
            f'Remediation stage saw project_root={root!r} but expected '
            f"'/path/A' (run_full_cycle's pre-mutation resolution at line 868). "
            f"Got '/path/B' means _run_remediation_pass re-resolved from the "
            f'mutated registry (task 1163 regression).'
        )


@pytest.mark.asyncio
async def test_dark_factory_full_cycle_no_regression(
    journal, event_buffer, mock_memory_service
):
    """End-to-end smoke: dark_factory cycle uses correct project_root across all 3 stages.

    Acceptance criterion (task 1143 step-15): "No regression for dark_factory Stage 1, 2, or 3."

    Build a harness with all five projects in _known_projects (realistic deployment).
    Push events for 'dark_factory' — some with _project_root payload (matching registry),
    some without — to verify neither variant causes contamination.
    Mock stage.run and capture project_id, project_root, known_projects at call time.
    Assert all three stages see:
      - project_id='dark_factory'
      - project_root='/home/leo/src/dark-factory'
      - known_projects containing all five entries
    """
    harness = _make_harness_with_known_projects(
        journal, event_buffer, mock_memory_service, _FIVE_PROJECT_MAP
    )

    # Push a mix: some events with correct _project_root, some without
    await event_buffer.push(_make_event_with_root('dark_factory', '/home/leo/src/dark-factory'))
    await event_buffer.push(_make_event('dark_factory'))  # no _project_root payload

    captured_states: list[dict] = []

    async def capture_stage_state(stage):
        captured_states.append({
            'stage_type': type(stage).__name__,
            'project_id': stage.project_id,
            'project_root': stage.project_root,
            'known_projects': dict(getattr(stage, 'known_projects', {})),
        })

    for stage in harness.stages:
        _mock_stage_run(stage, before_return=capture_stage_state)

    run = await harness.run_full_cycle('dark_factory', 'buffer_size:2')

    assert run.status == 'completed', f'Expected completed run, got status={run.status!r}'
    assert len(captured_states) == 3, (
        f'Expected 3 stage captures (one per stage), got {len(captured_states)}: '
        f'{[s["stage_type"] for s in captured_states]}'
    )

    expected_root = '/home/leo/src/dark-factory'
    for state in captured_states:
        assert state['project_id'] == 'dark_factory', (
            f'{state["stage_type"]}: project_id must be dark_factory, '
            f'got {state["project_id"]!r}'
        )
        assert state['project_root'] == expected_root, (
            f'{state["stage_type"]}: project_root must be {expected_root!r}, '
            f'got {state["project_root"]!r} — registry-bind regression check'
        )
        # Stage 2 (TaskKnowledgeSync) receives known_projects; verify map is intact
        if state['known_projects']:
            for pid, proot in _FIVE_PROJECT_MAP.items():
                assert pid in state['known_projects'], (
                    f'{state["stage_type"]}: known_projects must contain {pid!r}'
                )
                assert state['known_projects'][pid] == proot, (
                    f'{state["stage_type"]}: known_projects[{pid!r}] must be '
                    f'{proot!r}, got {state["known_projects"][pid]!r}'
                )


@pytest.mark.asyncio
@pytest.mark.parametrize('project_id,expected_root', [
    ('reify', '/home/leo/src/reify'),
    ('autopilot_video', '/home/leo/src/autopilot-video'),
])
async def test_stage3_payload_for_reify_emits_reify_root(
    journal, event_buffer, mock_memory_service, project_id, expected_root
):
    """Integration-level pin: Stage 3 assembled payload contains Use project_root="<correct>".

    Acceptance criterion (task 1143 step-17): Stage 3 integrity_check for
    project_id='reify' uses project_root='/home/leo/src/reify' for all
    Taskmaster calls.

    Approach:
    - Stages 0 and 1 are mocked normally (no CLI call).
    - Stage 2 (IntegrityCheck) has a custom run that calls the real
      assemble_payload() to capture the assembled string, then returns
      a StageReport without invoking the CLI subprocess.
    This means we test the actual payload logic end-to-end through run_full_cycle.
    """
    from fused_memory.models.reconciliation import StageReport

    harness = _make_harness_with_known_projects(
        journal, event_buffer, mock_memory_service, _FIVE_PROJECT_MAP
    )

    # Push events for the target project
    await event_buffer.push(_make_event(project_id))
    await event_buffer.push(_make_event(project_id))

    # Mock Stage 0 and 1 normally
    _mock_stage_run(harness.stages[0])
    _mock_stage_run(harness.stages[1])

    # Stage 2 (IntegrityCheck): custom run captures the payload from assemble_payload
    captured_payload: list[str] = []
    stage2 = harness.stages[2]

    async def capture_and_return(
        events, watermark, prior_reports, run_id, model=None, _s=stage2
    ):
        payload = await _s.assemble_payload(events, watermark, prior_reports)
        captured_payload.append(payload)
        return StageReport(
            stage=_s.stage_id,
            started_at=__import__('datetime').datetime.now(__import__('datetime').timezone.utc),
            completed_at=__import__('datetime').datetime.now(__import__('datetime').timezone.utc),
            items_flagged=[],
            stats={},
            llm_calls=0,
            tokens_used=0,
        )

    stage2.run = capture_and_return

    await harness.run_full_cycle(project_id, 'buffer_size:2')

    assert len(captured_payload) == 1, (
        f'Expected stage 2 (IntegrityCheck) to be called exactly once; '
        f'got {len(captured_payload)} calls'
    )
    payload = captured_payload[0]

    expected_directive = f'Use project_root="{expected_root}"'
    assert expected_directive in payload, (
        f'Stage 3 payload for project_id={project_id!r} must contain '
        f'{expected_directive!r} (task 1143 step-12). '
        f'Payload (first 600 chars):\n{payload[:600]}'
    )
    assert '/home/leo/src/dark-factory' not in payload or project_id == 'dark_factory', (
        f'Stage 3 payload for project_id={project_id!r} must not contain '
        f'dark-factory path — cross-contamination guard (task 1143)'
    )


# ── step-19: UnknownProjectError tests ────────────────────────────────────────

class TestUnknownProjectError:
    """Tests for the UnknownProjectError exception class (task 1143 step-19/20).

    These tests fail before step-20 with ImportError because UnknownProjectError
    does not yet exist in fused_memory.reconciliation.harness.
    """

    def test_unknown_project_error_is_value_error_subclass(self):
        """UnknownProjectError must subclass ValueError for backward-compat.

        Any existing or future ``except ValueError`` callsite (test code, callers
        of ``_known_project_root_for``) must continue to match after step-20.
        """
        from fused_memory.reconciliation.harness import UnknownProjectError  # noqa: F401

        assert issubclass(UnknownProjectError, ValueError) is True, (
            'UnknownProjectError must subclass ValueError for backward-compat '
            '(task 1143 step-20)'
        )

    @pytest.mark.asyncio
    async def test_known_project_root_for_raises_unknown_project_error_specifically(
        self, journal, event_buffer, mock_memory_service
    ):
        """_known_project_root_for raises UnknownProjectError (not bare ValueError).

        After step-20, the narrow exception type lets _project_loop distinguish
        a KNOWN_PROJECT_ROOTS misconfiguration from generic ValueErrors raised by
        stages (e.g. watermark/stage project_id mismatch, unset limits).
        """
        from fused_memory.reconciliation.harness import UnknownProjectError

        harness = _make_harness_with_known_projects(
            journal, event_buffer, mock_memory_service,
            {'dark_factory': '/home/leo/src/dark-factory'},
        )

        with pytest.raises(UnknownProjectError) as exc_info:
            harness._known_project_root_for('not_a_real_project')

        # Subclass relationship — must also satisfy bare ValueError catches
        assert isinstance(exc_info.value, ValueError) is True, (
            'UnknownProjectError must be an instance of ValueError '
            '(task 1143 step-20 backward-compat)'
        )

        err_msg = str(exc_info.value)
        assert 'not_a_real_project' in err_msg, (
            f'Error message must include the unknown project_id; got: {err_msg!r}'
        )
        assert 'dark_factory' in err_msg, (
            f'Error message must include known project_ids; got: {err_msg!r}'
        )


# ── step-21: _project_loop narrow exception handling tests ────────────────────

class TestProjectLoopNarrowsExceptionHandling:
    """Tests that _project_loop handles UnknownProjectError and plain ValueError differently.

    Before step-22, the bare ``except ValueError`` swallows ALL ValueErrors as
    misconfiguration and aborts the loop — including the transient ``ValueError``s
    raised by stages (watermark/stage project_id mismatch, unset limits).

    After step-22, the narrow ``except UnknownProjectError`` only catches registry
    misconfiguration; plain ``ValueError``s fall through to ``except Exception``
    and trigger the normal 5-second cooldown retry.

    Subcase (a) passes both before and after step-22 (same behavior — UnknownProjectError
    aborts loop cleanly in both cases).
    Subcase (b) fails BEFORE step-22 (bare ValueError catches and aborts rather than
    retrying) and passes AFTER step-22 (falls through to retry path).
    """

    @pytest.mark.asyncio
    async def test_project_loop_aborts_on_unknown_project_error_no_retry(
        self, journal, event_buffer, mock_memory_service
    ):
        """_project_loop must abort cleanly on UnknownProjectError — no 5-second retry.

        Pins that a KNOWN_PROJECT_ROOTS misconfiguration causes the loop to exit
        immediately (logs once, returns), rather than spam-retrying every 5 seconds.
        """
        import asyncio
        from unittest.mock import AsyncMock, patch

        # Harness with NO entry for 'autopilot_video' — any attempt to run a
        # full cycle for this project_id should raise UnknownProjectError.
        harness = _make_harness_with_known_projects(
            journal, event_buffer, mock_memory_service,
            {'dark_factory': '/home/leo/src/dark-factory'},
        )

        # Push enough events to trigger the cycle (buffer_size_threshold=2 in fixture)
        for _ in range(5):
            await event_buffer.push(_make_event('autopilot_video'))

        sleep_mock = AsyncMock()

        with patch('fused_memory.reconciliation.harness._sleep', sleep_mock):
            # Should return cleanly — no exception escapes (loop catches and returns)
            await asyncio.wait_for(
                harness._project_loop('autopilot_video'),
                timeout=10.0,
            )

        # The 5-second cooldown sleep must NOT have been called — the narrow catch
        # should abort the loop without ever reaching the bottom `await _sleep(5)`.
        cooldown_calls = [
            c for c in sleep_mock.await_args_list if c.args and c.args[0] == 5
        ]
        assert len(cooldown_calls) == 0, (
            f'_sleep(5) was called {len(cooldown_calls)} time(s) — '
            f'UnknownProjectError should abort the loop without any retry cooldown '
            f'(task 1143 step-22). Calls: {sleep_mock.await_args_list}'
        )

    @pytest.mark.asyncio
    async def test_project_loop_retries_on_generic_value_error_from_stage(
        self, journal, event_buffer, mock_memory_service
    ):
        """A plain ValueError from a stage must NOT abort the loop — it must retry.

        Before step-22, the bare ``except ValueError`` silently swallows stage
        ValueErrors (e.g. watermark↔stage project_id mismatch) and returns,
        causing the loop to never recover. After step-22, the narrow
        ``except UnknownProjectError`` lets the plain ValueError fall through to
        ``except Exception``, which logs it and continues with the 5-second cooldown.

        This test FAILS before step-22 (run_full_cycle called once, no _sleep(5))
        and PASSES after step-22 (run_full_cycle called twice, _sleep(5) called).
        """
        import asyncio
        from contextlib import suppress
        from unittest.mock import AsyncMock, patch

        # Harness WITH 'autopilot_video' so _known_project_root_for succeeds
        harness = _make_harness_with_known_projects(
            journal, event_buffer, mock_memory_service,
            {
                'autopilot_video': '/home/leo/src/autopilot-video',
                'dark_factory': '/home/leo/src/dark-factory',
            },
        )

        # Push enough events to trigger should_trigger → True
        for _ in range(5):
            await event_buffer.push(_make_event('autopilot_video'))

        # First call raises a plain ValueError (simulates transient stage error,
        # e.g. base.py:108 watermark↔stage project_id mismatch).
        # Second call raises CancelledError to terminate the loop in the test.
        rfc_mock = AsyncMock(
            side_effect=[
                ValueError('simulated watermark mismatch — not a misconfig'),
                asyncio.CancelledError(),
            ]
        )

        sleep_mock = AsyncMock()

        with (
            patch.object(harness, 'run_full_cycle', rfc_mock),
            patch('fused_memory.reconciliation.harness._sleep', sleep_mock),
            suppress(asyncio.CancelledError),
        ):
            # CancelledError on the 2nd run_full_cycle call propagates out — suppress it.
            await harness._project_loop('autopilot_video')

        # run_full_cycle must have been called at least twice — proof that the
        # plain ValueError was NOT loop-fatal (retry path was taken).
        assert rfc_mock.await_count >= 2, (
            f'run_full_cycle must be retried after a plain ValueError; '
            f'got {rfc_mock.await_count} call(s). '
            f'If the count is 1, the bare except ValueError is still swallowing '
            f'stage errors as misconfiguration (task 1143 step-22).'
        )

        # The 5-second cooldown must have been called between the two cycles.
        cooldown_calls = [
            c for c in sleep_mock.await_args_list if c.args and c.args[0] == 5
        ]
        assert len(cooldown_calls) >= 1, (
            f'_sleep(5) (retry cooldown) must be called after the plain ValueError; '
            f'got calls: {sleep_mock.await_args_list}. '
            f'If 0 calls, the except ValueError is aborting instead of falling '
            f'through to except Exception (task 1143 step-22).'
        )


class TestKnownProjectsInjection:
    """Tests for the known_projects DI kwarg on ReconciliationHarness (task 1164)."""

    def test_harness_accepts_known_projects_kwarg_and_uses_it(
        self, journal, event_buffer, mock_memory_service, monkeypatch
    ):
        """Injected known_projects dict wins; build_known_projects_map is NOT called.

        Patching build_known_projects_map to raise proves that the DI path skips
        the env-var-consulting factory entirely — not just that the result differs.
        """
        from fused_memory.config.schema import FusedMemoryConfig, ReconciliationConfig
        from fused_memory.reconciliation.harness import ReconciliationHarness

        def _must_not_be_called(*args, **kwargs):
            raise AssertionError(
                "build_known_projects_map must not be called when known_projects is injected"
            )

        monkeypatch.setattr(
            'fused_memory.reconciliation.harness.build_known_projects_map',
            _must_not_be_called,
        )

        injected = {'pid_a': '/path/a', 'pid_b': '/path/b'}
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
            known_projects=injected,
        )
        # The harness stores a defensive copy equal to the injected dict.
        assert harness._known_projects == {'pid_a': '/path/a', 'pid_b': '/path/b'}

    def test_harness_default_known_projects_kwarg_falls_back_to_build_known_projects_map(
        self, journal, event_buffer, mock_memory_service
    ):
        """When known_projects kwarg is omitted, harness falls back to build_known_projects_map.

        This verifies back-compat: all existing tests that construct ReconciliationHarness
        without the kwarg continue to get the env-derived map.
        """
        from fused_memory.config.schema import FusedMemoryConfig, ReconciliationConfig
        from fused_memory.models.scope import build_known_projects_map
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
            taskmaster=None,
            journal=journal,
            event_buffer=event_buffer,
            config=config,
        )
        expected = build_known_projects_map(harness._project_root)
        assert harness._known_projects == expected


# ── Deferred-write ordering tests (task 1474) ──────────────────────────────


@pytest.mark.asyncio
async def test_project_loop_replays_deferred_writes_before_releasing_lock(
    journal, event_buffer, mock_memory_service
):
    """_project_loop (finally path) must replay deferred writes WHILE the lock is held.

    Bug: mark_run_complete (which deletes the lock row) was called BEFORE
    _replay_deferred_writes, opening a window where a second process could claim
    the same deferred rows.  The fix: replay first, release lock second.

    Verification: spy on both methods to record call order, and probe
    buffer.is_full_recon_active inside the replay spy to confirm the lock is
    still held at replay time.
    """
    harness = _make_test_harness(journal, event_buffer, mock_memory_service)

    # Push enough events to trigger a cycle
    for _ in range(3):
        await event_buffer.push(_make_event())

    # Seed one deferred write so replay has something to claim
    await event_buffer.defer_write('test-project', 'some content', 'observations_and_summaries', {})

    # Build a completed ReconciliationRun for the fake cycle
    async def fake_rfc(*_a, **_k):
        return ReconciliationRun(
            id=str(uuid.uuid4()),
            project_id='test-project',
            run_type=RunType.full,
            trigger_reason='buffer_size:3',
            started_at=datetime.now(UTC),
            events_processed=3,
            status=RunStatus.completed,
        )

    call_order: list[str] = []
    lock_held_during_replay: list[bool] = []

    original_replay = harness._replay_deferred_writes
    original_complete = harness.buffer.mark_run_complete

    async def spy_replay(pid):
        call_order.append('replay')
        lock_held_during_replay.append(await event_buffer.is_full_recon_active(pid))
        return await original_replay(pid)

    async def spy_complete(pid):
        call_order.append('complete')
        return await original_complete(pid)

    with (
        patch.object(harness, 'run_full_cycle', side_effect=fake_rfc),
        patch.object(harness, '_replay_deferred_writes', side_effect=spy_replay),
        patch.object(harness.buffer, 'mark_run_complete', side_effect=spy_complete),
        contextlib.suppress(TimeoutError),
    ):
        await asyncio.wait_for(harness._project_loop('test-project'), timeout=0.5)

    assert call_order == ['replay', 'complete'], (
        f'Expected replay-then-complete, got {call_order!r}. '
        'Bug: lock released (mark_run_complete) before deferred writes are replayed.'
    )
    assert lock_held_during_replay == [True], (
        f'Expected lock held during replay, got {lock_held_during_replay!r}. '
        'Bug: mark_run_complete deleted the lock row before _replay_deferred_writes ran.'
    )


@pytest.mark.asyncio
async def test_project_loop_releases_lock_when_replay_raises(
    journal, event_buffer, mock_memory_service
):
    """_project_loop (finally path) must release the lock even when replay raises.

    A bare statement reorder (_replay_deferred_writes then mark_run_complete)
    would leak the lock if _replay_deferred_writes raises.  The correct fix
    wraps the pair as try/finally so mark_run_complete always runs.
    """
    harness = _make_test_harness(journal, event_buffer, mock_memory_service)

    # Push enough events to trigger a cycle
    for _ in range(3):
        await event_buffer.push(_make_event())

    # Seed one deferred write
    await event_buffer.defer_write('test-project', 'some content', 'observations_and_summaries', {})

    async def fake_rfc(*_a, **_k):
        return ReconciliationRun(
            id=str(uuid.uuid4()),
            project_id='test-project',
            run_type=RunType.full,
            trigger_reason='buffer_size:3',
            started_at=datetime.now(UTC),
            events_processed=3,
            status=RunStatus.completed,
        )

    complete_called: list[bool] = []
    original_complete = harness.buffer.mark_run_complete

    async def spy_complete(pid):
        complete_called.append(True)
        return await original_complete(pid)

    with (
        patch.object(harness, 'run_full_cycle', side_effect=fake_rfc),
        patch.object(harness, '_replay_deferred_writes', new=AsyncMock(side_effect=RuntimeError('boom'))),
        patch.object(harness.buffer, 'mark_run_complete', side_effect=spy_complete),
        contextlib.suppress(TimeoutError),
    ):
        await asyncio.wait_for(harness._project_loop('test-project'), timeout=0.5)

    assert complete_called, (
        'mark_run_complete was never called after replay raised. '
        'Bug: a bare reorder leaks the lock when _replay_deferred_writes raises.'
    )
    assert not await event_buffer.is_full_recon_active('test-project'), (
        'Lock was not released after replay raised. '
        'Bug: mark_run_complete must always run even if replay raises.'
    )


@pytest.mark.asyncio
async def test_project_loop_replays_before_releasing_lock_on_halt(
    journal, event_buffer, mock_memory_service
):
    """_project_loop (halt path) must replay deferred writes WHILE the lock is held.

    The halt early-return path has the same release-before-replay bug as the
    finally path: mark_run_complete was called before _replay_deferred_writes.
    The fix applies the same try/finally shape to this exit path.

    Verification: pre-halt the project, spy both methods, drive _project_loop
    directly (it returns naturally when halted), assert order and lock state.
    """
    from fused_memory.config.schema import ReconciliationConfig
    from fused_memory.reconciliation.judge import Judge

    harness = _make_test_harness(journal, event_buffer, mock_memory_service)

    # Wire a judge with the project pre-halted
    judge_config = ReconciliationConfig(
        enabled=True,
        explore_codebase_root='/tmp/test',
        agent_llm_provider='anthropic',
        agent_llm_model='claude-sonnet-4-20250514',
    )
    mock_j = AsyncMock()
    mock_j.get_run = AsyncMock(return_value=None)
    harness.judge = Judge(config=judge_config, journal=mock_j)
    harness.judge._halted_projects.add('test-project')

    # Suppress escalation side effects from _notify_judge_halt
    harness._notify_judge_halt = AsyncMock()

    # Push enough events to acquire the lock
    for _ in range(3):
        await event_buffer.push(_make_event())

    # Seed one deferred write
    await event_buffer.defer_write('test-project', 'some content', 'observations_and_summaries', {})

    call_order: list[str] = []
    lock_held_during_replay: list[bool] = []

    original_replay = harness._replay_deferred_writes
    original_complete = harness.buffer.mark_run_complete

    async def spy_replay(pid):
        call_order.append('replay')
        lock_held_during_replay.append(await event_buffer.is_full_recon_active(pid))
        return await original_replay(pid)

    async def spy_complete(pid):
        call_order.append('complete')
        return await original_complete(pid)

    with (
        patch.object(harness, '_replay_deferred_writes', side_effect=spy_replay),
        patch.object(harness.buffer, 'mark_run_complete', side_effect=spy_complete),
    ):
        # Halt path returns naturally — no wait_for/suppress needed
        await asyncio.wait_for(harness._project_loop('test-project'), timeout=1.0)

    assert call_order == ['replay', 'complete'], (
        f'Expected replay-then-complete on halt path, got {call_order!r}. '
        'Bug: halt path releases lock before replaying deferred writes.'
    )
    assert lock_held_during_replay == [True], (
        f'Expected lock held during replay on halt path, got {lock_held_during_replay!r}. '
        'Bug: halt path calls mark_run_complete before _replay_deferred_writes.'
    )
