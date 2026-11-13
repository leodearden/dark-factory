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

The pass driver comes from ``reconciliation.remediation_pass_fixtures`` and the
stores from ``reconciliation.harness_fixtures``.
"""
from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import shlex
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
import pytest_asyncio

import fused_memory.reconciliation.harness as harness_module
from fused_memory.reconciliation import escalation_archive
from fused_memory.reconciliation.harness import _derive_affected_ids
from reconciliation.harness_fixtures import (
    make_memory_service,
    open_event_buffer,
    open_journal,
)
from reconciliation.remediation_pass_fixtures import (
    finding_citing,
    in_progress_task,
    make_remediation_harness,
    prepare_remediation_pass,
)

pytest.importorskip('escalation.queue')
from escalation.dedupe import compute_content_fingerprint  # noqa: E402
from escalation.models import Escalation  # noqa: E402
from escalation.queue import EscalationQueue  # noqa: E402
from shared.whole_tree_guard_family import WHOLE_TREE_SCAN_TEST_TIMEOUT

pytestmark = pytest.mark.timeout(WHOLE_TREE_SCAN_TEST_TIMEOUT)

TICK_SECONDS = 0.005

#: How long each stubbed slow primitive takes, off the loop where it belongs —
#: a margin over scheduler jitter, not a timing the code relies on.
SCAN_BLOCK_SECONDS = 2.0
PROBE_BLOCK_SECONDS = 2.0
GIT_BLOCK_SECONDS = 2.0


def _max_gap_ceiling(blocked_seconds: float) -> float:
    """The longest loop stall tolerated while *blocked_seconds* of work runs.

    A call held on the loop thread stalls it for at least *blocked_seconds*;
    offloaded, the worst gap is only scheduler jitter.  Three quarters of the
    block sits between the two.
    """
    return blocked_seconds * 0.75


# ── Harness wiring ───────────────────────────────────────────────────────────


@pytest_asyncio.fixture
async def journal(tmp_path):
    async with open_journal(tmp_path / 'tail_offload_journal') as j:
        yield j


@pytest_asyncio.fixture
async def event_buffer(tmp_path):
    async with open_event_buffer(tmp_path / 'tail_offload_eb.db') as b:
        yield b


@pytest.fixture
def memory_service():
    return make_memory_service()


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
        """Seconds of scan must not cost the loop seconds of silence.

        Task 901 reads as live so nothing escalates: `_escalate` still fsyncs
        on the loop (task 5270), and under load that stall would be measured
        instead of the scan's.
        """
        calls: list = []
        _stub_scan(monkeypatch, calls)
        _stub_probe(monkeypatch, [], live_ids=frozenset({'901'}), block=0.0)
        _, _, run_pass = await prepare_remediation_pass(
            journal=journal, event_buffer=event_buffer, memory_service=memory_service,
            tmp_path=tmp_path, monkeypatch=monkeypatch,
            findings=[finding_citing('901')],
            cited_tasks=[in_progress_task('901')],
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
        findings = [finding_citing('901'), finding_citing('902'), finding_citing('903')]
        _, _, run_pass = await prepare_remediation_pass(
            journal=journal, event_buffer=event_buffer, memory_service=memory_service,
            tmp_path=tmp_path, monkeypatch=monkeypatch,
            findings=findings,
            cited_tasks=[in_progress_task('901'), in_progress_task('902'),
                         in_progress_task('903')],
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
        findings = [finding_citing('901'), finding_citing('902')]
        _, esc_queue, run_pass = await prepare_remediation_pass(
            journal=journal, event_buffer=event_buffer, memory_service=memory_service,
            tmp_path=tmp_path, monkeypatch=monkeypatch,
            findings=findings,
            cited_tasks=[in_progress_task('901'), in_progress_task('902')],
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


def _stub_detector(monkeypatch, probed: list, *, live_ids: frozenset[str], block: float):
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


def _stub_probe(monkeypatch, probed: list, *, live_ids: frozenset[str] = frozenset(),
                block: float = PROBE_BLOCK_SECONDS):
    """``_stub_detector``, with the worktree index the pass hoists stubbed out too.

    The real index spawns a ``git worktree list`` inside the measured window.
    ``TestHoistedWorktreeIndexLeavesTheLoop`` is the one test that keeps it.
    """

    async def _no_worktree_index(_project_root):
        return {}

    _stub_detector(monkeypatch, probed, live_ids=live_ids, block=block)
    monkeypatch.setattr(harness_module, 'worktree_index_kwargs', _no_worktree_index)


def _slow_git_first_on_path(monkeypatch, tmp_path: Path, *, block: float) -> Path:
    """Put a ``git`` that takes *block* seconds first on PATH; return its call log.

    Every invocation appends its arguments to the log, sleeps, then answers
    with an empty successful listing, which reads as a known-empty worktree
    index.
    """
    bin_dir = tmp_path / 'slow-git-bin'
    bin_dir.mkdir()
    call_log = tmp_path / 'slow-git-calls'
    git = bin_dir / 'git'
    git.write_text(
        '#!/bin/sh\n'
        f'echo "$*" >> {shlex.quote(str(call_log))}\n'
        f'sleep {block}\n'
    )
    git.chmod(0o755)
    monkeypatch.setenv('PATH', f'{bin_dir}{os.pathsep}{os.environ["PATH"]}')
    return call_log


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
        _, _, run_pass = await prepare_remediation_pass(
            journal=journal, event_buffer=event_buffer, memory_service=memory_service,
            tmp_path=tmp_path, monkeypatch=monkeypatch,
            findings=[finding_citing('901', '902'), finding_citing('901'),
                      finding_citing('902')],
            cited_tasks=[in_progress_task('901'), in_progress_task('902')],
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
        _, _, run_pass = await prepare_remediation_pass(
            journal=journal, event_buffer=event_buffer, memory_service=memory_service,
            tmp_path=tmp_path, monkeypatch=monkeypatch,
            findings=[finding_citing('901', '902'), finding_citing('901'),
                      finding_citing('902')],
            cited_tasks=[in_progress_task('901'), in_progress_task('902')],
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
        findings = [finding_citing('901', '902'), finding_citing('901'),
                    finding_citing('902')]
        _, esc_queue, run_pass = await prepare_remediation_pass(
            journal=journal, event_buffer=event_buffer, memory_service=memory_service,
            tmp_path=tmp_path, monkeypatch=monkeypatch,
            findings=findings,
            cited_tasks=[in_progress_task('901'), in_progress_task('902')],
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
        _, esc_queue, run_pass = await prepare_remediation_pass(
            journal=journal, event_buffer=event_buffer, memory_service=memory_service,
            tmp_path=tmp_path, monkeypatch=monkeypatch,
            findings=[finding_citing('901', '902'), finding_citing('901'),
                      finding_citing('902')],
            cited_tasks=[in_progress_task('901'), in_progress_task('902')],
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
        _, _, run_pass = await prepare_remediation_pass(
            journal=journal, event_buffer=event_buffer, memory_service=memory_service,
            tmp_path=tmp_path, monkeypatch=monkeypatch,
            findings=[finding_citing('901', routed_to='903')],
            cited_tasks=[in_progress_task('901'), in_progress_task('903')],
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
        _, _, run_pass = await prepare_remediation_pass(
            journal=journal, event_buffer=event_buffer, memory_service=memory_service,
            tmp_path=tmp_path, monkeypatch=monkeypatch,
            findings=[finding_citing('901', '902'),
                      finding_citing('901', routed_to='902')],
            cited_tasks=[in_progress_task('901'), in_progress_task('902')],
        )

        with caplog_silent():
            await run_pass()

        assert probed == ['901', '902'], (
            f'expected 902 to be probed once (as a cited id) and then served from '
            f'the memo to the routed-target gate, got {probed}'
        )


class TestHoistedWorktreeIndexLeavesTheLoop:
    """The one ``git worktree list`` the pass hoists must not hold the loop.

    This test keeps the real ``worktree_index_kwargs`` and makes ``git``
    itself slow, so a hoist that spawned git blockingly, through any spawn
    API, would stall the loop for the whole of git's run.
    """

    @pytest.mark.asyncio
    async def test_loop_keeps_ticking_while_the_worktree_index_is_listed(
        self, journal, event_buffer, memory_service, tmp_path, monkeypatch,
    ):
        """Task 901 reads as live so nothing escalates: `_escalate` still
        fsyncs on the loop (task 5270), and under load that stall would be
        measured instead of git's."""
        git_calls = _slow_git_first_on_path(monkeypatch, tmp_path, block=GIT_BLOCK_SECONDS)
        _stub_scan(monkeypatch, [], block=0.0)
        _stub_detector(monkeypatch, [], live_ids=frozenset({'901'}), block=0.0)
        _, _, run_pass = await prepare_remediation_pass(
            journal=journal, event_buffer=event_buffer, memory_service=memory_service,
            tmp_path=tmp_path, monkeypatch=monkeypatch,
            findings=[finding_citing('901')],
            cited_tasks=[in_progress_task('901')],
        )

        ticks, worst = await _worst_loop_stall(run_pass())

        assert git_calls.exists() and 'worktree list' in git_calls.read_text(), (
            'the hoisted worktree list never reached git — the index is not exercised'
        )
        assert ticks, 'the ticker never ran — the measurement is vacuous'
        assert worst < _max_gap_ceiling(GIT_BLOCK_SECONDS), (
            f'the event loop stalled for {worst * 1000:.0f}ms during the pass; '
            f'the hoisted worktree list is spawning git inline on the loop '
            f'(git takes {GIT_BLOCK_SECONDS * 1000:.0f}ms, so a stall at or above '
            f'that is git holding the loop thread)'
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
def esc_queue(tmp_path):
    return EscalationQueue(tmp_path / 'esc')


@pytest.fixture
def escalating_harness(journal, event_buffer, memory_service, esc_queue):
    """A harness with a real EscalationQueue, for driving `_escalate` directly."""
    return make_remediation_harness(journal, event_buffer, memory_service, escalation_queue=esc_queue)


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
                finding=finding_citing('901', description=f'finding {i}'),
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
                finding=finding_citing('901', description=f'finding for {run_id}'),
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
                finding=finding_citing('901', description='x'),
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
        self, escalating_harness, esc_queue, monkeypatch, caplog,
    ):
        """(d) An in-window resolution still suppresses; an 8-day-old one re-fires.

        Same corpus as tests/test_harness.py::test_finding_recently_resolved_respects_window,
        asserted through `_escalate` so the memo is in the path.
        """
        calls: list = []
        _count_scans(monkeypatch, calls)

        suppressed_finding = finding_citing('901', description='recently resolved')
        refiring_finding = finding_citing('902', description='resolved long ago')
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
        findings = [finding_citing('901'), finding_citing('902'),
                    finding_citing('903')]
        _, esc_queue, run_pass = await prepare_remediation_pass(
            journal=journal, event_buffer=event_buffer, memory_service=memory_service,
            tmp_path=tmp_path, monkeypatch=monkeypatch,
            findings=findings,
            cited_tasks=[in_progress_task('901'), in_progress_task('902'),
                         in_progress_task('903')],
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
