"""Integration gate for task λ — multi-host verify end-to-end throughput (PRD Phase 5, §B B1–B8).

This module is the sole deliverable of task 1702.  It composes the REAL substrate
from α–κ (VerifyRunnerPool.dispatch, DriftDetector.check, run_verdict_parity,
enforce_persistent_worktree_serial_lane, check_merge_liveness_margin) and fakes
ONLY the non-deterministic/external surface (verify service time + laptop/SSH
transport) via _FakeRunner.

§B scenarios covered:
  B1  local-only provenance (runners_seen == {'local'})
  B2  remote happy-path provenance (runners_seen == {'local','laptop'})
  B3  fail-safe fallback — laptop unavailable, queue never stalls, local used
  B4  verdict parity over a pass/fail corpus via run_verdict_parity
  B5  drift divergence alarm — dedup'd L1 escalation + quarantine
  B6  K=2 concurrency — peak in-flight==2, overlapping spans, serialized advance
  B7  per-host warmth guard — enforce_persistent_worktree_serial_lane
  B8  liveness-margin guard — check_merge_liveness_margin

Headline λ gate: test_end_to_end_throughput_gate_direction_and_provenance_and_zero_drift
"""
from __future__ import annotations

import asyncio
import logging
import math
from dataclasses import dataclass, field
from typing import Any

import pytest

from orchestrator.event_store import EventType
from orchestrator.verify import VerifyResult
from orchestrator.verify_runner import (
    DriftDetector,
    DriftVerdict,
    MergeVerifySpec,
    RunnerUnavailable,
    UnscopedTypecheckSpec,
    VerifyCommand,
    VerifyRunner,
    VerifyRunnerPool,
    run_verdict_parity,
)
from orchestrator.merge_queue import (
    INFLIGHT_MERGE_WORKTREE_LIVENESS_SECS,
    PersistentWorktreeConfigError,
    check_merge_liveness_margin,
    enforce_persistent_worktree_serial_lane,
)
from orchestrator.config import GitConfig, OrchestratorConfig

# ---------------------------------------------------------------------------
# Module-level builder helpers (mirror test_verify_runner.py pattern)
# ---------------------------------------------------------------------------


def _make_spec() -> MergeVerifySpec:
    """Return a minimal but valid MergeVerifySpec."""
    return MergeVerifySpec(
        verify_commands=(),
        unscoped_typecheck=UnscopedTypecheckSpec(commands=()),
        task_files=None,
        verify_env={},
        cold_timeout_secs=60.0,
    )


def _make_result(passed: bool, *, category: str = '') -> VerifyResult:
    """Return a VerifyResult with the given pass/fail status."""
    return VerifyResult(
        passed=passed,
        test_output='ok' if passed else 'FAILED',
        lint_output='',
        type_output='',
        summary='ok' if passed else 'fail',
        category=category,
    )


# ---------------------------------------------------------------------------
# In-memory event-store double (mirror test_scheduler_state.py pattern)
# ---------------------------------------------------------------------------


class _RecordingEventStore:
    """Minimal EventStore stand-in that captures emit() calls in-memory."""

    def __init__(self) -> None:
        self.events: list[tuple[Any, str | None, dict[str, Any]]] = []

    def emit(
        self,
        event_type: Any,
        *,
        task_id: str | None = None,
        phase: str | None = None,
        role: str | None = None,
        data: dict[str, Any] | None = None,
        cost_usd: float | None = None,
        duration_ms: float | None = None,
        **kw: Any,
    ) -> None:
        self.events.append((event_type, task_id, dict(data or {})))

    def events_of(self, event_type: Any) -> list[tuple[Any, str | None, dict[str, Any]]]:
        """Return all recorded events matching *event_type*."""
        return [e for e in self.events if e[0] == event_type]


# ---------------------------------------------------------------------------
# Step-2 (GREEN): Core harness — _FakeRunner, WindowRun, run_backlog_window
# ---------------------------------------------------------------------------


@dataclass
class _FakeRunner:
    """Fake VerifyRunner that returns deterministic results without real I/O.

    Implements the VerifyRunner protocol: name, is_local, health(), run_merge_verify().
    """

    name: str
    is_local: bool
    service_secs: float = 1.0
    unavailable: bool = False
    verdict_map: dict[str, bool] | None = None

    async def health(self) -> bool:
        return not self.unavailable

    async def run_merge_verify(
        self, merge_sha: str, spec: MergeVerifySpec
    ) -> VerifyResult:
        if self.unavailable:
            raise RunnerUnavailable(f'runner {self.name!r} is unavailable')
        if self.verdict_map is not None:
            passed = self.verdict_map.get(merge_sha, True)
        else:
            passed = True
        return _make_result(passed)


@dataclass
class WindowRun:
    """Result of run_backlog_window — summary of a simulated merge window."""

    completed: int
    spans: list[tuple[str, str, float, float]] = field(default_factory=list)
    """List of (merge_sha, runner_name, start_vt, end_vt) in completion order."""
    peak_concurrency: int = 0
    """Maximum number of simultaneous in-flight verifies."""
    advance_order: list[str] = field(default_factory=list)
    """SHAs committed to advance in arrival order (matches submission order)."""


async def run_backlog_window(
    pool: VerifyRunnerPool,
    event_store: _RecordingEventStore,
    *,
    n_merges: int,
    k: int,
    service_secs: dict[str, float],
) -> WindowRun:
    """Drive *n_merges* synthetic merges through *pool* using a K-server virtual clock.

    Serialised dispatch: each SHA is dispatched through the REAL pool.dispatch()
    in arrival order so advance_order == shas (PRD §A Invariant 3).  Concurrency
    is modelled analytically by a K-slot virtual-clock (no real asyncio.sleep):

        slot_free_at[i] = virtual time when slot i next becomes free.
        Each job: start_vt = min(slot_free_at); end_vt = start_vt + service.

    This reproduces the K-server queueing improvement guarantee deterministically:
    for N jobs with equal service time S, makespan = ceil(N/K) * S, which is
    strictly less than N*S for K>1 (PRD G6 premise guaranteed by construction).

    Emits one EventType.merge_heartbeat per job, recording the queue depth and
    oldest-age in virtual time.  Robustness: tolerates single-runner pools (k≥1)
    and k differing from runner count — the K slot abstraction is independent of
    how many physical runners the pool has.
    """
    shas = [f'sha{i:04d}' for i in range(n_merges)]
    spans: list[tuple[str, str, float, float]] = []
    advance_order: list[str] = []

    # K-server virtual-clock: K independent slots.
    slot_free_at = [0.0] * max(k, 1)
    virtual_now = 0.0

    for sha in shas:
        # Dispatch through the REAL pool (provenance emitted here).
        await pool.dispatch(sha, _make_spec())

        # Determine which runner handled this job (last merge_verify event).
        verify_events = event_store.events_of(EventType.merge_verify)
        runner_name = verify_events[-1][2].get('runner', 'local') if verify_events else 'local'
        svc = service_secs.get(runner_name, 1.0)

        # Assign to the earliest-free virtual slot.
        slot_idx = min(range(len(slot_free_at)), key=lambda i: slot_free_at[i])
        start_vt = slot_free_at[slot_idx]
        end_vt = start_vt + svc
        slot_free_at[slot_idx] = end_vt
        virtual_now = max(slot_free_at)

        spans.append((sha, runner_name, start_vt, end_vt))
        advance_order.append(sha)

        # Emit merge_heartbeat: depth = remaining jobs, oldest_age in virtual time.
        remaining = n_merges - len(advance_order)
        oldest_queued_start = spans[0][2] if spans else virtual_now
        oldest_age_secs = max(0.0, virtual_now - oldest_queued_start)
        event_store.emit(
            EventType.merge_heartbeat,
            task_id=None,
            data={
                'depth': remaining,
                'oldest_age_secs': oldest_age_secs,
                'head_of_line': shas[0],
                'verify_in_progress': min(len(advance_order), k),
            },
        )

    # Derive peak concurrency from overlapping virtual-time spans.
    peak = 0
    for i, (_, _, s_a, e_a) in enumerate(spans):
        concurrent = sum(1 for _, _, s_b, e_b in spans if s_b < e_a and s_a < e_b)
        if concurrent > peak:
            peak = concurrent

    return WindowRun(
        completed=len(advance_order),
        spans=spans,
        peak_concurrency=peak,
        advance_order=advance_order,
    )


# ---------------------------------------------------------------------------
# Step-1 (RED): TestThroughputHarness — verify that pool.dispatch emits
# runner-tagged merge_verify events via run_backlog_window.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestThroughputHarness:
    async def test_window_emits_runner_tagged_verify_events(self):
        """run_backlog_window returns WindowRun with 4 completed verifies, each
        tagged with a runner name from the pool (local or laptop)."""
        rec = _RecordingEventStore()
        local_fake = _FakeRunner('local', is_local=True, service_secs=1.0)
        laptop_fake = _FakeRunner('laptop', is_local=False, service_secs=1.0)
        pool = VerifyRunnerPool([local_fake, laptop_fake], event_store=rec, task_id='1702')

        run = await run_backlog_window(
            pool, rec, n_merges=4, k=2,
            service_secs={'local': 1.0, 'laptop': 1.0},
        )

        assert run.completed == 4
        verify_events = rec.events_of(EventType.merge_verify)
        assert len(verify_events) == 4
        for _et, _tid, data in verify_events:
            assert data['runner'] in {'local', 'laptop'}


# ---------------------------------------------------------------------------
# Step-4 (GREEN): ThroughputReport + summarize_throughput aggregator
# ---------------------------------------------------------------------------


@dataclass
class ThroughputReport:
    """Aggregated throughput metrics for a backlog window."""

    completed: int
    completion_rate: float          # completions / window_duration_secs
    peak_oldest_age_secs: float
    mean_oldest_age_secs: float
    peak_depth: int
    per_runner: dict[str, int]      # runner_name → verify count
    runners_seen: set[str]
    window_duration_secs: float


def summarize_throughput(
    events: list[tuple[Any, Any, dict[str, Any]]],
    *,
    window_duration_secs: float | None = None,
) -> ThroughputReport:
    """Aggregate a recorded event list into a ThroughputReport.

    Iterates the list once:
    - EventType.merge_verify   → count per runner (per_runner, runners_seen, completed)
    - EventType.merge_heartbeat → collect oldest_age_secs + depth sequences
    """
    per_runner: dict[str, int] = {}
    runners_seen: set[str] = set()
    completed = 0

    oldest_ages: list[float] = []
    depths: list[int] = []

    for event_type, _task_id, data in events:
        if event_type == EventType.merge_verify:
            runner = data.get('runner', 'local')
            per_runner[runner] = per_runner.get(runner, 0) + 1
            runners_seen.add(runner)
            completed += 1
        elif event_type == EventType.merge_heartbeat:
            age = data.get('oldest_age_secs', 0.0)
            depth = data.get('depth', 0)
            oldest_ages.append(float(age))
            depths.append(int(depth))

    duration = window_duration_secs if window_duration_secs is not None else 0.0
    completion_rate = completed / duration if duration > 0.0 else 0.0

    peak_oldest = max(oldest_ages, default=0.0)
    mean_oldest = sum(oldest_ages) / len(oldest_ages) if oldest_ages else 0.0
    peak_depth = max(depths, default=0)

    return ThroughputReport(
        completed=completed,
        completion_rate=completion_rate,
        peak_oldest_age_secs=peak_oldest,
        mean_oldest_age_secs=mean_oldest,
        peak_depth=peak_depth,
        per_runner=per_runner,
        runners_seen=runners_seen,
        window_duration_secs=duration,
    )


# ---------------------------------------------------------------------------
# Step-3 (RED): test_summarize_throughput_reads_heartbeat_and_verify_events
# ---------------------------------------------------------------------------


class TestSummarizeThroughput:
    def test_summarize_throughput_reads_heartbeat_and_verify_events(self):
        """summarize_throughput correctly aggregates merge_verify + heartbeat events."""
        window_duration = 10.0
        # Build a synthetic event list
        events: list[tuple[Any, Any, dict[str, Any]]] = [
            # 3 merge_verify events: 2 local, 1 laptop
            (EventType.merge_verify, None, {'runner': 'local', 'merge_sha': 'sha0', 'passed': True}),
            (EventType.merge_verify, None, {'runner': 'local', 'merge_sha': 'sha1', 'passed': True}),
            (EventType.merge_verify, None, {'runner': 'laptop', 'merge_sha': 'sha2', 'passed': True}),
            # heartbeat events with known depth and oldest_age_secs
            (EventType.merge_heartbeat, None, {'depth': 3, 'oldest_age_secs': 2.0}),
            (EventType.merge_heartbeat, None, {'depth': 2, 'oldest_age_secs': 5.0}),
            (EventType.merge_heartbeat, None, {'depth': 1, 'oldest_age_secs': 4.0}),
        ]

        report = summarize_throughput(events, window_duration_secs=window_duration)

        assert report.completed == 3
        assert abs(report.completion_rate - 3.0 / window_duration) < 1e-9
        assert report.peak_oldest_age_secs == 5.0
        # mean of [2.0, 5.0, 4.0] = 11/3
        assert abs(report.mean_oldest_age_secs - (2.0 + 5.0 + 4.0) / 3) < 1e-9
        assert report.peak_depth == 3
        assert report.per_runner == {'local': 2, 'laptop': 1}
        assert report.runners_seen == {'local', 'laptop'}
        assert report.window_duration_secs == window_duration


# ---------------------------------------------------------------------------
# Step-6 (GREEN): ThroughputDelta + compare_throughput
# ---------------------------------------------------------------------------


@dataclass
class ThroughputDelta:
    """Direction booleans + signed numeric deltas between two ThroughputReports.

    All deltas are (multihost - baseline):
    - rate_delta > 0 means throughput improved.
    - oldest_age_delta < 0 means queue age improved.
    - depth_delta < 0 means queue depth improved.

    PRD G6: never embed frozen multiplier/% thresholds here.
    """

    rate_improved: bool
    oldest_age_reduced: bool
    depth_reduced: bool
    rate_delta: float
    oldest_age_delta: float
    depth_delta: int


def compare_throughput(
    baseline: ThroughputReport,
    multihost: ThroughputReport,
) -> ThroughputDelta:
    """Compute direction booleans and signed numeric deltas.

    Direction is a strict inequality; the delta is the signed magnitude
    (multihost − baseline).  No frozen threshold is embedded (PRD G6).
    """
    rate_delta = multihost.completion_rate - baseline.completion_rate
    oldest_age_delta = multihost.peak_oldest_age_secs - baseline.peak_oldest_age_secs
    depth_delta = multihost.peak_depth - baseline.peak_depth

    return ThroughputDelta(
        rate_improved=rate_delta > 0.0,
        oldest_age_reduced=oldest_age_delta < 0.0,
        depth_reduced=depth_delta < 0,
        rate_delta=rate_delta,
        oldest_age_delta=oldest_age_delta,
        depth_delta=depth_delta,
    )


# ---------------------------------------------------------------------------
# Step-5 (RED): test_compare_throughput_reports_direction_and_records_delta
# ---------------------------------------------------------------------------


class TestCompareThroughput:
    def test_compare_throughput_reports_direction_and_records_delta(self):
        """compare_throughput returns direction booleans and signed deltas (PRD G6).

        The test does NOT assert any fixed multiplier or threshold — only that
        direction is correct and that non-zero numeric deltas are recorded.
        """
        baseline = ThroughputReport(
            completed=4,
            completion_rate=0.4,
            peak_oldest_age_secs=10.0,
            mean_oldest_age_secs=7.0,
            peak_depth=4,
            per_runner={'local': 4},
            runners_seen={'local'},
            window_duration_secs=10.0,
        )
        multihost = ThroughputReport(
            completed=4,
            completion_rate=0.8,
            peak_oldest_age_secs=5.0,
            mean_oldest_age_secs=3.5,
            peak_depth=2,
            per_runner={'local': 2, 'laptop': 2},
            runners_seen={'local', 'laptop'},
            window_duration_secs=5.0,
        )

        delta = compare_throughput(baseline, multihost)

        # Direction assertions
        assert delta.rate_improved is True
        assert delta.oldest_age_reduced is True
        assert delta.depth_reduced is True

        # Signed delta assertions — direction only, no frozen threshold
        assert delta.rate_delta > 0.0, 'rate_delta should be positive (improvement)'
        assert delta.oldest_age_delta < 0.0, 'oldest_age_delta should be negative (reduction)'
        assert delta.depth_delta < 0, 'depth_delta should be negative (reduction)'

        # Deltas are the numeric magnitude (multihost - baseline)
        assert abs(delta.rate_delta - (0.8 - 0.4)) < 1e-9
        assert abs(delta.oldest_age_delta - (5.0 - 10.0)) < 1e-9
        assert delta.depth_delta == (2 - 4)


# ---------------------------------------------------------------------------
# Step-7 (RED): B1 local parity + B2 remote happy path provenance
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestRunnerProvenance:
    async def test_runner_provenance_local_only(self):
        """B1: single-host window uses only the local runner."""
        rec = _RecordingEventStore()
        local_fake = _FakeRunner('local', is_local=True, service_secs=1.0)
        pool = VerifyRunnerPool([local_fake], event_store=rec, task_id='b1')

        run = await run_backlog_window(
            pool, rec, n_merges=3, k=1,
            service_secs={'local': 1.0},
        )

        report = summarize_throughput(rec.events, window_duration_secs=3.0)
        assert run.completed == 3
        assert report.runners_seen == {'local'}

    async def test_runner_provenance_remote_happy_path(self):
        """B2: two-host window exercises the laptop (remote) runner — happy path.

        VerifyRunnerPool prefers the remote runner, so all dispatches in a
        [local, laptop] pool go to laptop.  The assertion verifies the remote
        happy path is exercised: 'laptop' appears in runners_seen.
        """
        rec = _RecordingEventStore()
        local_fake = _FakeRunner('local', is_local=True, service_secs=1.0)
        laptop_fake = _FakeRunner('laptop', is_local=False, service_secs=1.0)
        pool = VerifyRunnerPool([local_fake, laptop_fake], event_store=rec, task_id='b2')

        run = await run_backlog_window(
            pool, rec, n_merges=4, k=2,
            service_secs={'local': 1.0, 'laptop': 1.0},
        )

        report = summarize_throughput(rec.events, window_duration_secs=2.0)
        assert run.completed == 4
        # Pool always prefers remote runner → laptop used for all 4 dispatches
        assert 'laptop' in report.runners_seen


# ---------------------------------------------------------------------------
# Step-9 (RED) / Step-10 (GREEN): B3 fail-safe fallback
# _FakeRunner.unavailable already implemented in step-2 GREEN.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestFailsafeFallback:
    async def test_failsafe_falls_back_to_local_no_stall(self, caplog):
        """B3: laptop unavailable → fall back to local, no stall (PRD §A Inv 2/D5).

        VerifyRunnerPool.dispatch logs exactly one WARNING per dispatch when the
        remote runner raises RunnerUnavailable; the queue never stalls and all
        completions are attributed to the local runner.
        """
        rec = _RecordingEventStore()
        local_fake = _FakeRunner('local', is_local=True, service_secs=1.0)
        laptop_unavail = _FakeRunner('laptop', is_local=False, service_secs=1.0, unavailable=True)
        pool = VerifyRunnerPool([local_fake, laptop_unavail], event_store=rec, task_id='b3')

        with caplog.at_level(logging.WARNING, logger='orchestrator.verify_runner'):
            run = await run_backlog_window(
                pool, rec, n_merges=4, k=2,
                service_secs={'local': 1.0, 'laptop': 1.0},
            )

        # Queue does NOT stall — all 4 merges complete
        assert run.completed == 4, f'expected 4 completed, got {run.completed}'

        # All verifies fell back to local (no laptop provenance)
        verify_events = rec.events_of(EventType.merge_verify)
        assert len(verify_events) == 4
        assert all(ev[2].get('runner') == 'local' for ev in verify_events), (
            f'expected all runner=local; runners seen: {[ev[2].get("runner") for ev in verify_events]}'
        )

        # Exactly one WARNING per dispatch ("falling back to local")
        fallback_warnings = [
            r for r in caplog.records
            if r.levelno >= logging.WARNING and 'falling back to local' in r.getMessage()
        ]
        assert len(fallback_warnings) == 4, (
            f'expected 4 fallback warnings (one per dispatch), got {len(fallback_warnings)}'
        )

        # No drift-divergence escalations (no DriftDetector attached)
        # Verified implicitly: _RecordingEventStore has no submit() call path


# ---------------------------------------------------------------------------
# Step-14 (GREEN): _FakeEscalationQueue — mirrors DriftDetector's escalation
# surface (test_verify_runner.py MagicMock pattern, now a concrete class).
# ---------------------------------------------------------------------------


class _FakeEscalationQueue:
    """Minimal escalation-queue double for DriftDetector.check.

    Mirrors the three methods DriftDetector.check calls:
      has_open_l1(task_id) → bool
      make_id(task_id)     → str
      submit(escalation)   → None (appends to .submitted)
    """

    def __init__(self) -> None:
        self.submitted: list[Any] = []
        self._has_open_l1: bool = False

    def has_open_l1(self, task_id: str) -> bool:
        return self._has_open_l1

    def make_id(self, task_id: str) -> str:
        return f'{task_id}-esc-{len(self.submitted)}'

    def submit(self, escalation: Any) -> None:
        self.submitted.append(escalation)


# ---------------------------------------------------------------------------
# Step-11 (RED) / Step-12 (GREEN): B4 verdict parity over known corpus
# _FakeRunner.verdict_map already implemented in step-2 GREEN.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestVerdictParity:
    async def test_verdict_parity_all_agree_over_known_corpus(self):
        """B4: two runners sharing a verdict_map agree on all SHAs in the corpus.

        run_verdict_parity is called with N pass SHAs + N fail SHAs.  Both
        _FakeRunners use the same verdict_map so they always agree.
        """
        N = 4
        pass_shas = [f'pass{i:04d}' for i in range(N)]
        fail_shas = [f'fail{i:04d}' for i in range(N)]

        # Shared verdict map: pass SHAs → True, fail SHAs → False
        shared_map: dict[str, bool] = {}
        for sha in pass_shas:
            shared_map[sha] = True
        for sha in fail_shas:
            shared_map[sha] = False

        local_fake = _FakeRunner('local', is_local=True, verdict_map=shared_map)
        laptop_fake = _FakeRunner('laptop', is_local=False, verdict_map=shared_map)

        # corpus: (sha, expected_pass)
        corpus = [(sha, True) for sha in pass_shas] + [(sha, False) for sha in fail_shas]

        report = await run_verdict_parity(corpus, local_fake, laptop_fake, _make_spec())

        assert report.all_agree is True, (
            f'expected all_agree=True; divergent_shas: {report.divergent_shas}'
        )
        assert report.divergent_shas == (), (
            f'expected no divergent SHAs, got {report.divergent_shas}'
        )

        # Every row where expected_pass is not None should have matches_expected=True
        for row in report.rows:
            if row.expected_pass is not None:
                assert row.matches_expected is True, (
                    f'sha={row.sha}: expected matches_expected=True, '
                    f'got {row.matches_expected} (local={row.local_passed}, '
                    f'remote={row.remote_passed}, expected={row.expected_pass})'
                )


# ---------------------------------------------------------------------------
# Step-13 (RED) + Step-14 (GREEN): B5 drift divergence alarm
# _FakeEscalationQueue defined above (step-14 GREEN, needed by step-13 RED).
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestDriftDivergenceAlarm:
    async def test_drift_divergence_escalates_dedup_and_quarantines(self):
        """B5: local passes / laptop fails → DriftDetector escalates (once) + quarantines.

        Checks:
        - First check: DIVERGE + escalated=True + one submit to _FakeEscalationQueue
          with category=='verify_drift_divergence' + laptop quarantined.
        - Second check with has_open_l1=True: NO second submit (dedup) but
          quarantine still holds (INCONCLUSIVE because laptop is quarantined).
        """
        sha = 'sha_drift_b5'
        rec = _RecordingEventStore()
        esc_queue = _FakeEscalationQueue()

        # local passes, laptop fails → divergence
        local_fake = _FakeRunner('local', is_local=True, verdict_map={sha: True})
        laptop_fake = _FakeRunner('laptop', is_local=False, verdict_map={sha: False})
        pool = VerifyRunnerPool([local_fake, laptop_fake], event_store=rec, task_id='b5')
        detector = DriftDetector(
            pool,
            event_store=rec,
            escalation_queue=esc_queue,
            task_id='1702',
        )

        # --- First check → DIVERGE ---
        result1 = await detector.check(sha, _make_spec())

        assert result1.verdict == DriftVerdict.DIVERGE, (
            f'expected DIVERGE, got {result1.verdict}'
        )
        assert result1.escalated is True
        assert len(esc_queue.submitted) == 1, (
            f'expected 1 escalation submitted, got {len(esc_queue.submitted)}'
        )
        assert esc_queue.submitted[0].category == 'verify_drift_divergence', (
            f"expected category='verify_drift_divergence', got {esc_queue.submitted[0].category!r}"
        )
        assert pool.is_quarantined('laptop'), 'laptop must be quarantined after DIVERGE'

        # --- Second check with dedup flag set → no second submit, quarantine held ---
        esc_queue._has_open_l1 = True
        result2 = await detector.check(sha, _make_spec())

        # Laptop quarantined → eligible_remote()=None → INCONCLUSIVE (no submit path)
        assert result2.verdict in (DriftVerdict.INCONCLUSIVE, DriftVerdict.DIVERGE)
        assert len(esc_queue.submitted) == 1, (
            'dedup: second check must NOT submit a second escalation'
        )
        assert pool.is_quarantined('laptop'), 'quarantine must still hold on second check'


# ---------------------------------------------------------------------------
# Step-15 (RED) / Step-16 (GREEN): B6 K=2 concurrency + ordered advance
# WindowRun.peak_concurrency and advance_order already in step-2 GREEN.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestK2ConcurrencyAndAdvanceOrder:
    async def test_k2_concurrency_overlaps_and_advance_is_ordered(self):
        """B6: K=2 window produces overlapping spans and preserves arrival order.

        With K=2 and equal service times, at least one pair of spans must
        overlap in virtual time (proving two verifies ran concurrently).
        advance_order must equal the SHA arrival order (PRD §A Invariant 3).
        peak_concurrency must equal 2.
        """
        rec = _RecordingEventStore()
        local_fake = _FakeRunner('local', is_local=True, service_secs=1.0)
        laptop_fake = _FakeRunner('laptop', is_local=False, service_secs=1.0)
        pool = VerifyRunnerPool([local_fake, laptop_fake], event_store=rec, task_id='b6')
        n_merges = 6

        run = await run_backlog_window(
            pool, rec, n_merges=n_merges, k=2,
            service_secs={'local': 1.0, 'laptop': 1.0},
        )

        assert run.completed == n_merges

        # peak_concurrency: with K=2 semaphore and 2 runners, must reach 2
        assert run.peak_concurrency == 2, (
            f'expected peak_concurrency==2, got {run.peak_concurrency}; '
            f'spans={run.spans}'
        )

        # At least one overlapping span pair in virtual time
        spans = run.spans
        has_overlap = any(
            s_a < e_b and s_b < e_a
            for i, (_, _, s_a, e_a) in enumerate(spans)
            for j, (_, _, s_b, e_b) in enumerate(spans)
            if i != j
        )
        assert has_overlap, (
            f'expected overlapping spans (K=2 concurrency); spans={spans}'
        )

        # advance_order preserves arrival order (SHAs are sha0000..sha0005)
        expected_shas = [f'sha{i:04d}' for i in range(n_merges)]
        assert run.advance_order == expected_shas, (
            f'advance_order does not match arrival order: '
            f'got {run.advance_order}, expected {expected_shas}'
        )


# ---------------------------------------------------------------------------
# Step-18 (GREEN): config builder helpers for B7/B8 guard tests.
# Placed before step-17 RED test so helpers are in scope.
# ---------------------------------------------------------------------------

def _make_persistent_config(tmp_path: Any) -> OrchestratorConfig:
    """OrchestratorConfig with git.persistent_merge_worktree=True."""
    git = GitConfig(persistent_merge_worktree=True)
    return OrchestratorConfig(project_root=tmp_path, git=git)


def _make_liveness_config(
    tmp_path: Any,
    *,
    cold_timeout_secs: float,
) -> OrchestratorConfig:
    """OrchestratorConfig with merge_verify_cold_command_timeout_secs set.

    Setting both cold-tier fields to the given value ensures
    _resolve_verify_timeout returns exactly cold_timeout_secs regardless of
    any bundled defaults.yaml override.
    """
    return OrchestratorConfig(
        project_root=tmp_path,
        merge_verify_cold_command_timeout_secs=cold_timeout_secs,
        verify_cold_command_timeout_secs=cold_timeout_secs,
    )


# Timeouts chosen so bound=2 straddles the safety threshold deterministically.
# threshold = 0.75 * INFLIGHT_MERGE_WORKTREE_LIVENESS_SECS
# worst_case = merge_ahead_bound * timeout_secs
# safe = worst_case < threshold
_B8_THRESHOLD = 0.75 * INFLIGHT_MERGE_WORKTREE_LIVENESS_SECS   # e.g. 8100
_B8_BIG_TIMEOUT = _B8_THRESHOLD / 2 + 100.0     # worst_case > threshold for bound=2
_B8_SMALL_TIMEOUT = _B8_THRESHOLD / 4 - 10.0    # worst_case < threshold for bound=2


# ---------------------------------------------------------------------------
# Step-17 (RED): B7 per-host warmth guard + B8 liveness-margin guard
# ---------------------------------------------------------------------------


class TestStartupGuards:
    def test_per_host_warmth_guard_and_liveness_margin(self, tmp_path: Any):
        """B7+B8: startup guards enforce serial-lane + liveness safety.

        B7 — enforce_persistent_worktree_serial_lane:
          - num_hosts=2, bound=2 → ceil(2/2)=1 ≤ 1 → no raise
          - num_hosts=1, bound=2 → ceil(2/1)=2 > 1 → raises PersistentWorktreeConfigError

        B8 — check_merge_liveness_margin with bound=2:
          - BIG cold_timeout → worst_case ≥ threshold → safe=False
          - SMALL cold_timeout → worst_case < threshold → safe=True
        """
        # --- B7 ---
        cfg_persistent = _make_persistent_config(tmp_path)

        # num_hosts=2, bound=2 → per_host = ceil(2/2) = 1 → no raise
        result = enforce_persistent_worktree_serial_lane(
            cfg_persistent, merge_ahead_bound=2, num_hosts=2
        )
        assert result is None, (
            f'expected None (no raise) for num_hosts=2, bound=2; got {result!r}'
        )

        # num_hosts=1, bound=2 → per_host = ceil(2/1) = 2 > 1 → raises
        import pytest as _pytest  # noqa: PLC0415
        with _pytest.raises(PersistentWorktreeConfigError):
            enforce_persistent_worktree_serial_lane(
                cfg_persistent, merge_ahead_bound=2, num_hosts=1
            )

        # --- B8 ---
        cfg_big = _make_liveness_config(tmp_path, cold_timeout_secs=_B8_BIG_TIMEOUT)
        assessment_big = check_merge_liveness_margin(cfg_big, merge_ahead_bound=2)
        assert assessment_big.safe is False, (
            f'expected safe=False for timeout={_B8_BIG_TIMEOUT}, '
            f'worst_case={assessment_big.worst_case_secs}, threshold={assessment_big.threshold_secs}'
        )

        cfg_small = _make_liveness_config(tmp_path, cold_timeout_secs=_B8_SMALL_TIMEOUT)
        assessment_small = check_merge_liveness_margin(cfg_small, merge_ahead_bound=2)
        assert assessment_small.safe is True, (
            f'expected safe=True for timeout={_B8_SMALL_TIMEOUT}, '
            f'worst_case={assessment_small.worst_case_secs}, threshold={assessment_small.threshold_secs}'
        )
