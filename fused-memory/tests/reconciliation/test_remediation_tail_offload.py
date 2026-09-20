"""The remediation tail must not wedge the event loop (task 5550).

``ReconciliationHarness._run_remediation_pass`` ends in a tail that does real
blocking I/O: an unbounded scan of the escalation queue + dated archive to build
``resolved_fps``, and up to three 10-second ``git`` subprocesses per cited task
in the live-workflow gate.  Run inline on the loop, that tail stops every other
coroutine in the fused-memory process — including the ``/alive`` route the
orchestrator watchdog probes, which is how a slow scan gets read as a dead
process and the unit gets restarted mid-cycle.

THESE TESTS ASSERT LOOP RESPONSIVENESS DIRECTLY rather than spying on
``asyncio.to_thread``.  A spy proves a particular call took a particular hop; a
ticker proves the property the wedge is actually about, and stays meaningful
under any future refactor that keeps the loop free.  It also discriminates
sharply: 250ms of blocking against a 5ms tick is ~50 ticks when offloaded and
exactly 0 when inline, so a threshold of 3 has two orders of magnitude of margin
against CI jitter in both directions.

Harness scaffolding is re-created locally rather than imported from
tests/test_harness.py — the convention set by tests/reconciliation/test_active_runs.py,
since ``fused-memory/tests`` is not an importable package.
"""
from __future__ import annotations

import asyncio
import contextlib
import logging
import time
import uuid
from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock

import pytest
import pytest_asyncio

import fused_memory.reconciliation.harness as harness_module
from fused_memory.models.reconciliation import (
    ReconciliationRun,
    RunStatus,
    RunType,
    StageReport,
)
from fused_memory.models.scope import ProjectId, ProjectRoot, ProjectScope
from fused_memory.reconciliation.event_buffer import EventBuffer
from fused_memory.reconciliation.harness import (
    _INTEGRITY_FINDING_RECURRENCE_THRESHOLD,
    TierConfig,
)
from fused_memory.reconciliation.journal import ReconciliationJournal
from fused_memory.reconciliation.task_filter import FilteredTaskTree

pytest.importorskip('escalation.queue')
from escalation.queue import EscalationQueue  # noqa: E402

PROJECT_ID = 'test-project'
PROJECT_ROOT = '/tmp/test-project'

#: Ticker period and the floor a pass must clear.  See the module docstring for
#: why 3 is both far below the offloaded floor (~50) and unreachable inline (0).
TICK_SECONDS = 0.005
MIN_TICKS = 3

#: How long each stubbed blocking primitive sleeps, in the thread it should be
#: running in.  Sized so one call alone is ~50 ticks.
SCAN_BLOCK_SECONDS = 0.25
PROBE_BLOCK_SECONDS = 0.15


# ── Local harness scaffolding (mirrors tests/test_harness.py) ────────────────


def _scope(project_id: str, project_root: str) -> ProjectScope:
    return ProjectScope(ProjectId(project_id), ProjectRoot(project_root))


def _rescope(stages: list, scope: ProjectScope) -> list:
    """Re-scope pinned stage instances in place so the ``_make_stages`` shim
    honors the scope production passes while keeping the mocked ``.run``."""
    for s in stages:
        s.scope = scope
    return stages


@pytest_asyncio.fixture
async def journal(tmp_path):
    j = ReconciliationJournal(tmp_path / 'tail_offload_journal')
    await j.initialize()
    yield j
    await j.close()


@pytest_asyncio.fixture
async def event_buffer(tmp_path):
    buf = EventBuffer(
        db_path=tmp_path / 'tail_offload_eb.db',
        buffer_size_threshold=2,
        max_staleness_seconds=3600,
    )
    await buf.initialize()
    yield buf
    await buf.close()


@pytest.fixture
def memory_service():
    svc = AsyncMock()
    svc.search = AsyncMock(return_value=[])
    svc.get_episodes = AsyncMock(return_value=[])
    svc.get_status = AsyncMock(
        return_value={
            'graphiti': {'connected': True},
            'mem0': {'connected': True},
            'projects': {},
        }
    )
    svc.get_entity = AsyncMock(return_value={'nodes': [], 'edges': []})
    svc.get_memories_by_metadata = AsyncMock(return_value=[])
    svc.mem0 = AsyncMock()
    svc.mem0.get_all = AsyncMock(return_value={'results': []})
    return svc


def _make_harness(journal, event_buffer, memory_service):
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
        memory_service=memory_service,
        taskmaster=AsyncMock(),
        journal=journal,
        event_buffer=event_buffer,
        config=config,
    )
    harness.judge = None
    harness._known_projects = {PROJECT_ID: PROJECT_ROOT}
    harness.stages = harness._make_stages(_scope(PROJECT_ID, PROJECT_ROOT))
    harness._make_stages = lambda scope, **k: _rescope(harness.stages, scope)
    return harness


def _mock_stage_run(stage, items_flagged=None):
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


def _finding_citing(*task_ids: str, description: str | None = None) -> dict:
    """An actionable stranded-work finding citing each of *task_ids*."""
    cited = ','.join(task_ids)
    return {
        'description': description or f'Implementation-complete but not merged: {cited}',
        'severity': 'urgent',
        'actionable': True,
        'category': 'stranded_work',
        'cited_tasks': [
            {'project_id': PROJECT_ID, 'task_id': tid} for tid in task_ids
        ],
        'suggested_action': 'Manual merge required',
    }


def _in_progress_task(task_id: str) -> dict:
    return {
        'id': int(task_id),
        'title': f'In-progress task {task_id}',
        'status': 'in-progress',
        'claimant_run_id': f'run-{task_id}',
        'heartbeat_at': datetime.now(UTC).isoformat(),
        'metadata': {'task_kind': 'normal'},
        'dependencies': [],
    }


# ── The loop-responsiveness probe ────────────────────────────────────────────


async def _ticks_while(coro) -> int:
    """Await *coro* while a 5ms ticker counts how often the loop got control.

    Returns the tick count.  Zero means the loop was held by synchronous work
    for the whole duration — the wedge this task removes.
    """
    ticks = 0

    async def _tick() -> None:
        nonlocal ticks
        while True:
            await asyncio.sleep(TICK_SECONDS)
            ticks += 1

    ticker = asyncio.create_task(_tick())
    try:
        await coro
    finally:
        ticker.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await ticker
    return ticks


# ── Driving the pass ─────────────────────────────────────────────────────────


async def _seed_persistence(journal, findings: list[dict]) -> None:
    """Seed threshold-1 prior completed runs carrying *findings*.

    Together with this pass's own persisted S3 report that reaches
    ``_INTEGRITY_FINDING_RECURRENCE_THRESHOLD`` so the live-workflow gate fires.
    Uses the journal's PUBLIC async API (start_run / update_run_stage_reports /
    complete_run), never its storage internals.
    """
    n_seed = _INTEGRITY_FINDING_RECURRENCE_THRESHOLD - 1
    base_time = datetime.now(UTC) - timedelta(minutes=n_seed + 1)
    for i in range(n_seed):
        rid = str(uuid.uuid4())
        await journal.start_run(
            ReconciliationRun(
                id=rid,
                project_id=PROJECT_ID,
                run_type=RunType.full,
                trigger_reason='buffer_size:1',
                started_at=base_time + timedelta(minutes=i),
                events_processed=1,
                status=RunStatus.running,
            )
        )
        await journal.update_run_stage_reports(
            rid, {'integrity_check': {'items_flagged': findings}}
        )
        await journal.complete_run(rid, 'completed')


async def _prepare_pass(
    *, journal, event_buffer, memory_service, tmp_path, monkeypatch,
    findings: list[dict], cited_tasks: list[dict] | None = None,
):
    """Wire a harness to drive ``_run_remediation_pass`` over *findings*.

    Returns ``(harness, esc_queue, run_pass)`` where ``run_pass()`` returns a
    fresh coroutine for the pass — so a caller can hand it to ``_ticks_while``.
    """
    harness = _make_harness(journal, event_buffer, memory_service)
    esc_queue = EscalationQueue(tmp_path / 'esc')
    harness._escalation_queue = esc_queue

    # Pin the two on-disk corroboration inputs to their empty forms so neither
    # of the other corroboration signals can fire against the shared,
    # pytest-unmanaged /tmp/test-project root.
    monkeypatch.setattr(
        harness_module, 'read_scheduler_state',
        lambda _root: {
            'queue': [], 'parks': {}, 'park_stacks': {},
            'effective_priorities': {}, 'pin_queue': [], 'overrides': {},
            'current_holders': {}, 'is_paused': False, 'pause_reason': None,
            'snapshot_at': None,
        },
    )
    monkeypatch.setattr(harness_module, 'orchestrator_started_at', lambda _root: None)

    await _seed_persistence(journal, findings)

    _mock_stage_run(harness.stages[0])
    _mock_stage_run(harness.stages[1])
    _mock_stage_run(harness.stages[2], items_flagged=findings)

    tree = FilteredTaskTree(
        active_tasks=list(cited_tasks or []), total_count=len(cited_tasks or []),
    )
    tier = TierConfig(model='sonnet', episode_limit=100, memory_limit=200)

    def run_pass():
        return harness._run_remediation_pass(
            PROJECT_ID, 'parent-run-id', findings, tier,
            scope=_scope(PROJECT_ID, PROJECT_ROOT),
            filtered_task_tree=tree,
            filtered_task_tree_fetched_at=datetime.now(UTC),
        )

    return harness, esc_queue, run_pass


def _stub_scan(monkeypatch, calls: list, *, block: float = SCAN_BLOCK_SECONDS):
    """Replace the archive scan with a blocking stub that records its calls."""

    def _scan(queue_dir, *, now, window, categories):
        calls.append(queue_dir)
        time.sleep(block)
        return frozenset()

    monkeypatch.setattr(harness_module, 'scan_recently_resolved_fingerprints', _scan)


# ── SITE 1: the per-pass archive scan ────────────────────────────────────────


class TestArchiveScanLeavesTheLoop:
    """The ``resolved_fps`` build must run in a worker thread, exactly once."""

    @pytest.mark.asyncio
    async def test_loop_keeps_ticking_while_the_archive_is_scanned(
        self, journal, event_buffer, memory_service, tmp_path, monkeypatch,
    ):
        """250ms of scan must not cost the loop 250ms of silence."""
        calls: list = []
        _stub_scan(monkeypatch, calls)
        _, _, run_pass = await _prepare_pass(
            journal=journal, event_buffer=event_buffer, memory_service=memory_service,
            tmp_path=tmp_path, monkeypatch=monkeypatch,
            findings=[_finding_citing('901')],
            cited_tasks=[_in_progress_task('901')],
        )

        ticks = await _ticks_while(run_pass())

        assert calls, 'the archive scan never ran — the test is not exercising site 1'
        assert ticks >= MIN_TICKS, (
            f'event loop got control only {ticks} times while the pass ran; '
            f'{SCAN_BLOCK_SECONDS}s of archive scan is still inline on the loop'
        )

    @pytest.mark.asyncio
    async def test_archive_is_scanned_exactly_once_per_pass(
        self, journal, event_buffer, memory_service, tmp_path, monkeypatch,
    ):
        """One scan per pass — the task-1669 snapshot semantics, not per-finding.

        Offloading must not quietly turn a single pre-pass scan into one scan
        per finding: that would move the blocking work off the loop while
        multiplying it.
        """
        calls: list = []
        _stub_scan(monkeypatch, calls, block=0.0)
        findings = [_finding_citing('901'), _finding_citing('902'), _finding_citing('903')]
        _, _, run_pass = await _prepare_pass(
            journal=journal, event_buffer=event_buffer, memory_service=memory_service,
            tmp_path=tmp_path, monkeypatch=monkeypatch,
            findings=findings,
            cited_tasks=[_in_progress_task('901'), _in_progress_task('902'),
                         _in_progress_task('903')],
        )

        with caplog_silent():
            await run_pass()

        assert len(calls) == 1, (
            f'expected exactly 1 archive scan for a 3-finding pass, got {len(calls)}'
        )


@contextlib.contextmanager
def caplog_silent():
    """Keep harness INFO chatter out of the captured log for count-only tests."""
    logger = logging.getLogger('fused_memory.reconciliation.harness')
    previous = logger.level
    logger.setLevel(logging.WARNING)
    try:
        yield
    finally:
        logger.setLevel(previous)
