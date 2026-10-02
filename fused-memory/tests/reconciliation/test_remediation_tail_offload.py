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
under any future refactor that keeps the loop free — including one that still
calls ``to_thread`` but blocks either side of it.  The spy idiom is kept for
claims that really are about a call COUNT.

What the ticker reports is the WORST GAP between its turns, not how many turns
it got: a wedge puts a hard floor under that gap, while a tick count also
tracks machine speed and would red on a busy CI box.

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
from fused_memory.reconciliation import escalation_archive
from fused_memory.reconciliation.event_buffer import EventBuffer
from fused_memory.reconciliation.harness import (
    _INTEGRITY_FINDING_RECURRENCE_THRESHOLD,
    TierConfig,
    _derive_affected_ids,
)
from fused_memory.reconciliation.journal import ReconciliationJournal
from fused_memory.reconciliation.task_filter import FilteredTaskTree

pytest.importorskip('escalation.queue')
from escalation.dedupe import compute_content_fingerprint  # noqa: E402
from escalation.models import Escalation  # noqa: E402
from escalation.queue import EscalationQueue  # noqa: E402

PROJECT_ID = 'test-project'
PROJECT_ROOT = '/tmp/test-project'

TICK_SECONDS = 0.005

#: How long each stubbed blocking primitive sleeps, in the thread it should be
#: running in — a margin over scheduler jitter, not a timing the code relies on.
SCAN_BLOCK_SECONDS = 1.0
PROBE_BLOCK_SECONDS = 1.0


def _max_gap_ceiling(blocked_seconds: float) -> float:
    """The longest loop stall tolerated while *blocked_seconds* of work runs.

    A call held on the loop thread stalls it for at least *blocked_seconds*;
    offloaded, the worst gap is only scheduler jitter.  Half the block splits
    the two.
    """
    return blocked_seconds / 2


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


def _finding_citing(
    *task_ids: str, description: str | None = None, routed_to: str | None = None,
) -> dict:
    """An actionable stranded-work finding citing each of *task_ids*.

    *routed_to* sets the bare ``task_id`` field, which
    ``resolve_finding_task_target`` prefers over ``cited_tasks`` — the way to
    build a finding whose ROUTED target is not one of its cited ids, and so the
    only way to reach the routed-target liveness gate.
    """
    cited = ','.join(task_ids)
    finding = {
        'description': description or f'Implementation-complete but not merged: {cited}',
        'severity': 'urgent',
        'actionable': True,
        'category': 'stranded_work',
        'cited_tasks': [
            {'project_id': PROJECT_ID, 'task_id': tid} for tid in task_ids
        ],
        'suggested_action': 'Manual merge required',
    }
    if routed_to is not None:
        finding['task_id'] = routed_to
        finding['description'] = description or (
            f'Implementation-complete but not merged: {cited} (routed to {routed_to})'
        )
    return finding


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


async def _worst_loop_stall(coro) -> tuple[int, float]:
    """Await *coro* while a 5ms ticker watches the loop; return (ticks, worst gap).

    The worst gap is the longest the event loop went without giving the ticker
    control — the signal, per the module docstring.  The tick count comes back
    only so a caller can reject a measurement in which the ticker never ran.

    The thread-pool executor is warmed first so its one-off cold-start stall is
    not measured as if the code under test had caused it.
    """
    await asyncio.to_thread(lambda: None)

    ticks = 0
    worst = 0.0
    last = time.monotonic()

    async def _tick() -> None:
        nonlocal ticks, worst, last
        while True:
            await asyncio.sleep(TICK_SECONDS)
            now = time.monotonic()
            worst = max(worst, now - last)
            last = now
            ticks += 1

    ticker = asyncio.create_task(_tick())
    last = time.monotonic()
    try:
        await coro
    finally:
        ticker.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await ticker
    # A stall still open when the coroutine returned counts too: a wedge that
    # runs right up to the end would otherwise never be closed by a tick.
    worst = max(worst, time.monotonic() - last)
    return ticks, worst


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
    fresh coroutine for the pass — so a caller can hand it to
    ``_worst_loop_stall``.
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
        """A second of scan must not cost the loop a second of silence.

        Task 901 reads as live so nothing escalates: `_escalate` still fsyncs
        on the loop (task 5270), and under load that stall would be measured
        instead of the scan's.
        """
        calls: list = []
        _stub_scan(monkeypatch, calls)
        _stub_probe(monkeypatch, [], live_ids=frozenset({'901'}), block=0.0)
        _, _, run_pass = await _prepare_pass(
            journal=journal, event_buffer=event_buffer, memory_service=memory_service,
            tmp_path=tmp_path, monkeypatch=monkeypatch,
            findings=[_finding_citing('901')],
            cited_tasks=[_in_progress_task('901')],
        )

        ticks, worst = await _worst_loop_stall(run_pass())

        assert calls, 'the archive scan never ran — the test is not exercising site 1'
        assert ticks, 'the ticker never ran — the measurement is vacuous'
        assert worst < _max_gap_ceiling(SCAN_BLOCK_SECONDS), (
            f'the event loop stalled for {worst * 1000:.0f}ms during the pass; '
            f'{SCAN_BLOCK_SECONDS}s of archive scan is still inline on the loop '
            f'(a stall at or above {SCAN_BLOCK_SECONDS * 1000:.0f}ms is the scan '
            f'itself holding the loop thread)'
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

    @pytest.mark.asyncio
    async def test_a_failed_scan_suppresses_nothing(
        self, journal, event_buffer, memory_service, tmp_path, monkeypatch, caplog,
    ):
        """Fail-open: a scan that raises costs the pass its suppressions, not its escalations."""
        calls: list = []

        def _raising_scan(queue_dir, **_kwargs):
            calls.append(queue_dir)
            raise OSError('archive unreadable')

        monkeypatch.setattr(
            harness_module, 'scan_recently_resolved_fingerprints', _raising_scan,
        )
        _stub_probe(monkeypatch, [], block=0.0)
        findings = [_finding_citing('901'), _finding_citing('902')]
        _, esc_queue, run_pass = await _prepare_pass(
            journal=journal, event_buffer=event_buffer, memory_service=memory_service,
            tmp_path=tmp_path, monkeypatch=monkeypatch,
            findings=findings,
            cited_tasks=[_in_progress_task('901'), _in_progress_task('902')],
        )

        with caplog.at_level(logging.WARNING, logger='fused_memory.reconciliation.harness'):
            await run_pass()

        assert calls, 'the archive scan never ran — the failure branch is not exercised'
        assert [
            r for r in caplog.records
            if r.getMessage() == 'reconciliation.recently_resolved_check_failed'
        ], 'the scan failure must be logged'
        stranded = [
            e for e in esc_queue.get_pending() if 'Persistently unresolved' in e.summary
        ]
        assert len(stranded) == len(findings), (
            f'expected all {len(findings)} findings to escalate after a failed scan, '
            f'got {len(stranded)}'
        )


# ── SITE 2: the live-workflow git probes ─────────────────────────────────────


def _stub_probe(monkeypatch, probed: list, *, live_ids: frozenset[str] = frozenset(),
                block: float = PROBE_BLOCK_SECONDS):
    """Replace the live-workflow detector with a slow stub that records ids.

    Stands in for ``is_workflow_live_for_task``, a coroutine since task 3778
    whose up-to-three git probes per task each take up to 10s.  The stub
    awaits for *block* seconds, so a stall of that size means the pass is
    holding the loop around the probe rather than awaiting it.
    """

    async def _is_live(tid, _project_root, **_kw):
        probed.append(str(tid))
        await asyncio.sleep(block)
        return str(tid) in live_ids

    monkeypatch.setattr(harness_module, 'is_workflow_live_for_task', _is_live)


class TestLiveWorkflowProbesLeaveTheLoopAndAreMemoised:
    """The git probes must run off-loop, and at most once per DISTINCT task id.

    Without the memo, a pass over three findings citing two tasks between them
    would pay six probes of up to 30s each.
    """

    @pytest.mark.asyncio
    async def test_loop_keeps_ticking_while_the_git_probes_run(
        self, journal, event_buffer, memory_service, tmp_path, monkeypatch,
    ):
        """Both cited tasks read as live so nothing escalates: `_escalate` still
        fsyncs on the loop (task 5270), and under load that stall would be
        measured instead of the probes'."""
        probed: list[str] = []
        _stub_probe(monkeypatch, probed, live_ids=frozenset({'901', '902'}))
        _, _, run_pass = await _prepare_pass(
            journal=journal, event_buffer=event_buffer, memory_service=memory_service,
            tmp_path=tmp_path, monkeypatch=monkeypatch,
            findings=[_finding_citing('901', '902'), _finding_citing('901'),
                      _finding_citing('902')],
            cited_tasks=[_in_progress_task('901'), _in_progress_task('902')],
        )

        ticks, worst = await _worst_loop_stall(run_pass())

        assert probed, 'the detector was never consulted — site 2 is not exercised'
        assert ticks, 'the ticker never ran — the measurement is vacuous'
        assert worst < _max_gap_ceiling(PROBE_BLOCK_SECONDS), (
            f'the event loop stalled for {worst * 1000:.0f}ms during the pass; '
            f'the live-workflow git probes are still inline on the loop '
            f'(each probe blocks {PROBE_BLOCK_SECONDS * 1000:.0f}ms, so a stall '
            f'at or above that is a probe holding the loop thread)'
        )

    @pytest.mark.asyncio
    async def test_each_distinct_cited_task_is_probed_exactly_once(
        self, journal, event_buffer, memory_service, tmp_path, monkeypatch,
    ):
        """Three findings, two distinct cited ids → two probes, not six.

        Liveness is treated as constant for a task id across the gate loop —
        the same loop-constant assumption the pass already makes for the
        scheduler state and orchestrator start time hoisted just above it.
        """
        probed: list[str] = []
        _stub_probe(monkeypatch, probed, block=0.0)
        _, _, run_pass = await _prepare_pass(
            journal=journal, event_buffer=event_buffer, memory_service=memory_service,
            tmp_path=tmp_path, monkeypatch=monkeypatch,
            findings=[_finding_citing('901', '902'), _finding_citing('901'),
                      _finding_citing('902')],
            cited_tasks=[_in_progress_task('901'), _in_progress_task('902')],
        )

        with caplog_silent():
            await run_pass()

        assert probed == ['901', '902'], (
            f'expected one probe per DISTINCT cited task id, got {probed}'
        )

    @pytest.mark.asyncio
    async def test_all_findings_still_escalate_when_nothing_is_live(
        self, journal, event_buffer, memory_service, tmp_path, monkeypatch,
    ):
        """VERDICT PARITY — memoising False must not silence an escalation."""
        probed: list[str] = []
        _stub_probe(monkeypatch, probed, block=0.0)
        findings = [_finding_citing('901', '902'), _finding_citing('901'),
                    _finding_citing('902')]
        _, esc_queue, run_pass = await _prepare_pass(
            journal=journal, event_buffer=event_buffer, memory_service=memory_service,
            tmp_path=tmp_path, monkeypatch=monkeypatch,
            findings=findings,
            cited_tasks=[_in_progress_task('901'), _in_progress_task('902')],
        )

        with caplog_silent():
            await run_pass()

        stranded = [
            e for e in esc_queue.get_pending()
            if e.category == 'recon_integrity_issue'
            and 'Persistently unresolved' in e.summary
        ]
        assert len(stranded) == len(findings), (
            f'expected all {len(findings)} findings to escalate with nothing live, '
            f'got {len(stranded)}'
        )

    @pytest.mark.asyncio
    async def test_findings_citing_a_live_task_are_suppressed(
        self, journal, event_buffer, memory_service, tmp_path, monkeypatch, caplog,
    ):
        """VERDICT PARITY — a live cited task still suppresses, and says so."""
        probed: list[str] = []
        _stub_probe(monkeypatch, probed, live_ids=frozenset({'901'}), block=0.0)
        _, esc_queue, run_pass = await _prepare_pass(
            journal=journal, event_buffer=event_buffer, memory_service=memory_service,
            tmp_path=tmp_path, monkeypatch=monkeypatch,
            findings=[_finding_citing('901', '902'), _finding_citing('901'),
                      _finding_citing('902')],
            cited_tasks=[_in_progress_task('901'), _in_progress_task('902')],
        )

        with caplog.at_level(logging.INFO, logger='fused_memory.reconciliation.harness'):
            await run_pass()

        suppressed = [
            r for r in caplog.records
            if r.getMessage()
            == 'reconciliation.integrity_escalation_suppressed_live_workflow'
        ]
        stranded = [
            e for e in esc_queue.get_pending()
            if e.category == 'recon_integrity_issue'
            and 'Persistently unresolved' in e.summary
        ]
        assert len(suppressed) == 2, (
            f'expected the two findings citing live task 901 to be suppressed, '
            f'got {len(suppressed)} suppression records'
        )
        assert len(stranded) == 1, (
            f'expected only the 902-citing finding to escalate, got {len(stranded)}'
        )


class TestRoutedTargetGateUsesTheSameMemo:
    """The routed-target gate is the SECOND consumer of the probe and must
    share the memo rather than re-probing a task the cited gate already asked
    about."""

    @pytest.mark.asyncio
    async def test_routed_target_not_among_cited_ids_is_still_probed(
        self, journal, event_buffer, memory_service, tmp_path, monkeypatch,
    ):
        """A routed id the cited gate never saw must still consult the detector."""
        probed: list[str] = []
        _stub_probe(monkeypatch, probed, block=0.0)
        _, _, run_pass = await _prepare_pass(
            journal=journal, event_buffer=event_buffer, memory_service=memory_service,
            tmp_path=tmp_path, monkeypatch=monkeypatch,
            findings=[_finding_citing('901', routed_to='903')],
            cited_tasks=[_in_progress_task('901'), _in_progress_task('903')],
        )

        with caplog_silent():
            await run_pass()

        assert probed == ['901', '903'], (
            f'expected the cited id then the routed id to be probed, got {probed}'
        )

    @pytest.mark.asyncio
    async def test_routed_target_already_probed_as_a_cited_id_hits_the_memo(
        self, journal, event_buffer, memory_service, tmp_path, monkeypatch,
    ):
        """Cross-gate memo: 902 is probed once, as a cited id, then served."""
        probed: list[str] = []
        _stub_probe(monkeypatch, probed, block=0.0)
        _, _, run_pass = await _prepare_pass(
            journal=journal, event_buffer=event_buffer, memory_service=memory_service,
            tmp_path=tmp_path, monkeypatch=monkeypatch,
            findings=[_finding_citing('901', '902'),
                      _finding_citing('901', routed_to='902')],
            cited_tasks=[_in_progress_task('901'), _in_progress_task('902')],
        )

        with caplog_silent():
            await run_pass()

        assert probed == ['901', '902'], (
            f'expected 902 to be probed once (as a cited id) and then served from '
            f'the memo to the routed-target gate, got {probed}'
        )


# ── SITE 3: _finding_recently_resolved's fallback arm ────────────────────────


def _count_scans(monkeypatch, calls: list):
    """Count archive scans while still performing them.

    Delegates to the REAL helper so every verdict below is produced by the
    production scan — only the call COUNT is observed, which is what the bound
    is a claim about.
    """
    real = escalation_archive.scan_recently_resolved_fingerprints

    def _wrapped(queue_dir, **kwargs):
        calls.append(queue_dir)
        return real(queue_dir, **kwargs)

    monkeypatch.setattr(harness_module, 'scan_recently_resolved_fingerprints', _wrapped)


def _seed_resolved(esc_queue, esc_id: str, fingerprint: str, *, age: timedelta) -> None:
    """Seed one resolved escalation carrying *fingerprint*, resolved *age* ago."""
    esc_queue.submit(Escalation(
        id=esc_id,
        task_id='recon-seed',
        agent_role='reconciliation-harness',
        severity='info',
        category='recon_integrity_issue',
        summary=f'prior resolved finding {esc_id}',
        status='resolved',
        resolved_at=(datetime.now(UTC) - age).isoformat(),
        dedupe_fingerprint=fingerprint,
    ))


def _fingerprint_of(finding: dict) -> str:
    """The fingerprint `_escalate` will compute for *finding*."""
    return compute_content_fingerprint(
        'recon_integrity_issue',
        finding.get('category') or '',
        _derive_affected_ids(finding),
        finding.get('description') or '',
    )


@pytest.fixture
def escalating_harness(journal, event_buffer, memory_service, tmp_path):
    """A harness with a real EscalationQueue, for driving `_escalate` directly."""
    harness = _make_harness(journal, event_buffer, memory_service)
    harness._escalation_queue = EscalationQueue(tmp_path / 'esc')
    return harness


class TestFallbackArmIsBoundedToOneScanPerRun:
    """`_escalate` stays synchronous, so its archive scan is BOUNDED, not gone.

    Task 5270 owns the full offload of `_escalate`; this task routes the
    fallback arm through a run-scoped memo, so consecutive calls within one run
    share one archive walk instead of paying one each.
    """

    def test_three_escalates_sharing_a_run_id_scan_once(
        self, escalating_harness, monkeypatch,
    ):
        """(a) BOUND — same run_id, no prebuilt set → one scan, not three."""
        calls: list = []
        _count_scans(monkeypatch, calls)

        for i in range(3):
            escalating_harness._escalate(
                'recon_integrity_issue', 'run-aaaa1111',
                f'Persistently unresolved: finding {i}',
                finding=_finding_citing('901', description=f'finding {i}'),
            )

        assert len(calls) == 1, (
            f'expected 1 archive scan for 3 escalations in one run, got {len(calls)}'
        )

    def test_a_different_run_id_rescans(self, escalating_harness, monkeypatch):
        """(b) SCOPE — the memo must never serve a later run a stale snapshot."""
        calls: list = []
        _count_scans(monkeypatch, calls)

        for run_id in ('run-aaaa1111', 'run-aaaa1111', 'run-bbbb2222'):
            escalating_harness._escalate(
                'recon_integrity_issue', run_id, f'Persistently unresolved: {run_id}',
                finding=_finding_citing('901', description=f'finding for {run_id}'),
            )

        assert len(calls) == 2, (
            f'expected a fresh scan for the second run_id, got {len(calls)} scans'
        )

    def test_the_direct_path_is_never_memoised(self, escalating_harness, monkeypatch):
        """(c) `_finding_recently_resolved` with no scan key keeps today's behaviour.

        Its docstring promises this fallback for "direct / unit-test use", and
        the four task-1669 tests depend on it: every direct call scans, and none
        disturbs a run's memo, so that run's next `_escalate` is still served.
        """
        calls: list = []
        _count_scans(monkeypatch, calls)
        now = datetime.now(UTC)

        def _escalate_in_run() -> None:
            escalating_harness._escalate(
                'recon_integrity_issue', 'run-dddd4444', 'Persistently unresolved: x',
                finding=_finding_citing('901', description='x'),
            )

        _escalate_in_run()
        scans_before_direct = len(calls)
        for _ in range(3):
            escalating_harness._finding_recently_resolved(
                'recon_integrity_issue', 'some-fingerprint', now=now,
            )
        direct_scans = len(calls) - scans_before_direct
        _escalate_in_run()

        assert direct_scans == 3, (
            f'expected the direct path to scan every call, got {direct_scans}'
        )
        assert len(calls) == scans_before_direct + direct_scans, (
            "the direct path must leave the run's memo in place, but the run's "
            'next _escalate walked the archive again'
        )

    def test_verdicts_are_unchanged_across_the_memo(
        self, escalating_harness, monkeypatch, caplog,
    ):
        """(d) An in-window resolution still suppresses; an 8-day-old one re-fires.

        Same corpus as tests/test_harness.py::test_finding_recently_resolved_respects_window,
        asserted through `_escalate` so the memo is in the path.
        """
        calls: list = []
        _count_scans(monkeypatch, calls)
        esc_queue = escalating_harness._escalation_queue

        suppressed_finding = _finding_citing('901', description='recently resolved')
        refiring_finding = _finding_citing('902', description='resolved long ago')
        fp_in = _fingerprint_of(suppressed_finding)
        fp_out = _fingerprint_of(refiring_finding)
        _seed_resolved(esc_queue, 'esc-in-window', fp_in, age=timedelta(seconds=60))
        _seed_resolved(esc_queue, 'esc-out-of-window', fp_out, age=timedelta(days=8))

        with caplog.at_level(logging.INFO, logger='fused_memory.reconciliation.harness'):
            for finding in (suppressed_finding, refiring_finding):
                escalating_harness._escalate(
                    'recon_integrity_issue', 'run-cccc3333',
                    f'Persistently unresolved: {finding["description"]}',
                    finding=finding,
                )

        pending_fps = {e.dedupe_fingerprint for e in esc_queue.get_pending()}
        assert fp_in not in pending_fps, (
            'a finding resolved 60s ago must still be suppressed through the memo'
        )
        assert fp_out in pending_fps, (
            'a finding resolved 8 days ago is outside the window and must re-fire'
        )
        assert [
            r for r in caplog.records
            if r.getMessage() == 'reconciliation.escalation_suppressed_recently_resolved'
        ], 'the suppression must still be logged'
        assert len(calls) == 1, (
            f'both verdicts must come from ONE scan of the run, got {len(calls)}'
        )

    @pytest.mark.asyncio
    async def test_a_pass_filing_every_finding_walks_the_archive_once(
        self, journal, event_buffer, memory_service, tmp_path, monkeypatch,
    ):
        """(e) The pass's per-finding escalations never reach the fallback arm.

        Each takes the pass's prebuilt set (task 1669), so filing N findings
        costs the pass's one walk and no more.
        """
        calls: list = []
        _count_scans(monkeypatch, calls)
        _stub_probe(monkeypatch, [], block=0.0)
        findings = [_finding_citing('901'), _finding_citing('902'),
                    _finding_citing('903')]
        _, esc_queue, run_pass = await _prepare_pass(
            journal=journal, event_buffer=event_buffer, memory_service=memory_service,
            tmp_path=tmp_path, monkeypatch=monkeypatch,
            findings=findings,
            cited_tasks=[_in_progress_task('901'), _in_progress_task('902'),
                         _in_progress_task('903')],
        )

        with caplog_silent():
            await run_pass()

        stranded = [
            e for e in esc_queue.get_pending() if 'Persistently unresolved' in e.summary
        ]
        assert len(stranded) == len(findings), (
            'the pass must file every finding, or this bound is vacuous'
        )
        assert len(calls) == 1, (
            f'a {len(findings)}-finding pass must cost exactly one archive walk, '
            f'got {len(calls)}'
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
