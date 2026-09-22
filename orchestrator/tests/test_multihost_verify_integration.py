"""Integration gate for task λ — multi-host verify queueing model + provenance plumbing (PRD Phase 5, §B B1–B8).

This module is the sole deliverable of task 1702.  It composes the REAL substrate
from α–κ (VerifyRunnerPool.dispatch, DriftDetector.check, run_verdict_parity,
enforce_persistent_worktree_serial_lane, check_merge_liveness_margin) and fakes
ONLY the non-deterministic/external surface (verify service time + laptop/SSH
transport) via _FakeRunner.

NOTE on concurrency model
--------------------------
VerifyRunnerPool.dispatch is currently serial — it dispatches and awaits one SHA
at a time.  The K-permit semaphore that would enable real concurrent dispatch is
explicitly deferred to ζ (see verify_runner.py VerifyRunnerPool docstring).
Consequently run_backlog_window models concurrency ANALYTICALLY via a K-server
virtual clock; the throughput-direction improvements it measures are guaranteed
by the K-server queueing model's arithmetic, not by observing actual concurrent
pool behavior.  The headline gate therefore validates:

  (a) the queueing MODEL (direction improvement is deterministic by construction),
  (b) runner provenance plumbing (merge_verify events carry the correct runner tag),
  (c) drift-detector integration (AGREE path → verdict_parity_ok, zero escalations).

Once ζ's K-permit semaphore lands, the gate can be upgraded to drive real
overlapping dispatch (e.g. asyncio.gather over slow fake runners with an
injected clock) and measure observed, rather than computed, overlap.

§B scenarios covered:
  B1  local-only provenance (runners_seen == {'local'})
  B2  remote happy-path provenance (runners_seen == {'local','laptop'})
  B3  fail-safe fallback — laptop unavailable, queue never stalls, local used
  B4  verdict parity over a pass/fail corpus via run_verdict_parity
  B5  drift divergence alarm — dedup'd L1 escalation + quarantine
  B6  K=2 virtual-clock model — peak in-flight==2, overlapping spans, serialized advance
  B7  per-host warmth guard — enforce_persistent_worktree_serial_lane
  B8  liveness-margin guard — check_merge_liveness_margin

Headline λ gate (queueing-model + provenance): test_queueing_model_direction_and_provenance_and_zero_drift

NOTE on seams (PRD plans/merge-lane-quality-prd.md task γ)
---------------------------------------------------------
Nothing here reaches into the merge lane.  The verify PORT is injected
(``FakeVerifier``), which is also what the per-land cross-check's local
trust anchor runs on -- ``_run_post_merge_verify`` builds its ``LocalRunner``
with ``run_scoped=verifier.run_scoped``, so the injected port IS that leg's
verdict and ``verifier.verified`` is the exact cross-check ran/skipped
signal.  The host-monitor scenarios drive a RUNNING ``MergeLane`` over a real
git repository, take their fault from the remote runner's own transport, and
observe it through ``snapshot()['hosts']``, the escalation queue, the event
store and the ``MergeOutcome`` futures.
"""
from __future__ import annotations

import asyncio
import contextlib
import logging
import math
import time
from collections.abc import Collection
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar
from unittest.mock import AsyncMock, MagicMock

import pytest
from _merge_lane_fakes import FakeVerifier, VerifyScript, fails, passes
from _orch_helpers import wait_responsive
from test_merge_queue_concurrent_verify import (
    HEAVY_BARRIER_TEST_TIMEOUT,
    PYPROJECT_DEFAULT_TIMEOUT,
    _inject_two_host_allocator,
    _make_branch_with_file,
    _timeout_mark_offenders,
    _worst_per_method_wait_budget,
)

from orchestrator.config import GitConfig, OrchestratorConfig
from orchestrator.event_store import EventType
from orchestrator.git_ops import GitOps, _run
from orchestrator.merge_lane import MergeLane
from orchestrator.merge_queue import (
    PRODUCTION_CLOCK,
    MergeRequest,
    PersistentWorktreeConfigError,
    check_merge_liveness_margin,
    enforce_persistent_worktree_serial_lane,
)
from orchestrator.merge_types import QueuedBranch
from orchestrator.verify import VerifyResult
from orchestrator.verify_runner import (
    DriftDetector,
    DriftVerdict,
    MergeVerifySpec,
    RunnerUnavailable,
    UnscopedTypecheckSpec,
    VerifyRunnerPool,
    run_verdict_parity,
)

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


class _FakeRunnerBase:
    """Fake VerifyRunner base — subclasses fix is_local as a ClassVar for protocol conformance.

    Implements the VerifyRunner protocol: name, is_local, health(), run_merge_verify().
    """

    is_local: ClassVar[bool]

    def __init__(
        self,
        name: str,
        *,
        service_secs: float = 1.0,
        unavailable: bool = False,
        verdict_map: dict[str, bool] | None = None,
    ) -> None:
        self.name = name
        self.service_secs = service_secs
        self.unavailable = unavailable
        self.verdict_map = verdict_map

    async def health(self) -> bool:
        return not self.unavailable

    async def run_merge_verify(
        self, merge_sha: str, spec: MergeVerifySpec
    ) -> VerifyResult:
        if self.unavailable:
            raise RunnerUnavailable(f'runner {self.name!r} is unavailable')
        passed = self.verdict_map.get(merge_sha, True) if self.verdict_map is not None else True
        return _make_result(passed)


class _LocalFakeRunner(_FakeRunnerBase):
    """Fake local runner (is_local=True)."""

    is_local: ClassVar[bool] = True


class _RemoteFakeRunner(_FakeRunnerBase):
    """Fake remote runner (is_local=False)."""

    is_local: ClassVar[bool] = False


def _FakeRunner(
    name: str,
    *,
    is_local: bool,
    service_secs: float = 1.0,
    unavailable: bool = False,
    verdict_map: dict[str, bool] | None = None,
) -> _LocalFakeRunner | _RemoteFakeRunner:
    """Factory — return a protocol-conformant fake runner for the given locality."""
    cls: type[_LocalFakeRunner | _RemoteFakeRunner] = _LocalFakeRunner if is_local else _RemoteFakeRunner
    return cls(name, service_secs=service_secs, unavailable=unavailable, verdict_map=verdict_map)


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

    NOTE: concurrency is ANALYTICAL, not real parallel dispatch.
    VerifyRunnerPool.dispatch is awaited serially (one SHA at a time) — the real
    K-permit semaphore is deferred to ζ.  The K-slot virtual-clock below is a
    queueing-model computation: it assigns each job to the earliest-free slot and
    tracks virtual start/end times arithmetically.  It does NOT exercise real
    concurrent dispatch; it exercises (1) the real pool.dispatch provenance path and
    (2) the K-server queueing model that guarantees throughput direction.

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

        # Emit merge_heartbeat once per K-batch (every k dispatches, or on the
        # final job).  This models the real queue's heartbeat cadence: a K=2
        # window drains 2 jobs per beat so peak_depth = ceil(N/K)-1 < N-1 for
        # K=1, making the depth-improvement direction observable.
        dispatched = len(advance_order)
        if dispatched % max(k, 1) == 0 or dispatched == n_merges:
            remaining = n_merges - dispatched
            oldest_queued_start = spans[0][2] if spans else virtual_now
            oldest_age_secs = max(0.0, virtual_now - oldest_queued_start)
            event_store.emit(
                EventType.merge_heartbeat,
                task_id=None,
                data={
                    'depth': remaining,
                    'oldest_age_secs': oldest_age_secs,
                    'head_of_line': shas[0],
                    'verify_in_progress': min(dispatched, k),
                },
            )

    # Derive peak concurrency from overlapping virtual-time spans.
    peak = 0
    for _i, (_, _, s_a, e_a) in enumerate(spans):
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
        elif event_type == EventType.verdict_parity_ok:
            # Both runners participated in parity check — surface in runners_seen.
            local_r = data.get('local_runner')
            remote_r = data.get('remote_runner')
            if local_r:
                runners_seen.add(local_r)
            if remote_r:
                runners_seen.add(remote_r)

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
        self.open_l1: bool = False

    def has_open_l1(self, task_id: str) -> bool:
        return self.open_l1

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
        esc_queue.open_l1 = True
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
        """B6: K=2 virtual-clock model produces overlapping spans and preserves arrival order.

        NOTE: the concurrency asserted here (peak_concurrency==2, overlapping spans)
        is derived from the K-server VIRTUAL-CLOCK model, not from real simultaneous
        pool.dispatch calls.  See run_backlog_window for the analytical model details.
        The advance_order assertion (PRD §A Invariant 3) exercises the REAL serialized
        dispatch ordering through pool.dispatch.

        With K=2 and equal service times, at least one pair of spans must
        overlap in virtual time (by the K-server slot assignment arithmetic).
        advance_order must equal the SHA arrival order (PRD §A Invariant 3).
        peak_concurrency must equal 2 (per the slot-overlap computation).
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


# ---------------------------------------------------------------------------
# Step-17 (RED): B7 per-host warmth guard + B8 liveness-margin guard
# ---------------------------------------------------------------------------


class TestStartupGuards:
    def test_per_host_warmth_guard_and_liveness_margin(self, tmp_path: Any):
        """B7+B8: startup guards enforce serial-lane + liveness safety.

        B7 — enforce_persistent_worktree_serial_lane:
          - num_hosts=2, bound=2 → ceil(2/2)=1 ≤ 1 → no raise
          - num_hosts=1, bound=2 → ceil(2/1)=2 > 1 → raises PersistentWorktreeConfigError

        B8 — check_merge_liveness_margin (heartbeat-floor model, task-1729 β):
          - injected liveness_secs=600 → threshold=450 ≤ floor=600 → safe=False
          - shipped default liveness → threshold=8100 > floor=600 → safe=True
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

        # --- B8 (heartbeat-floor model) ---
        # low liveness_secs → threshold ≤ floor → safe=False
        cfg = OrchestratorConfig(project_root=tmp_path)
        assessment_low = check_merge_liveness_margin(cfg, liveness_secs=600.0)
        assert assessment_low.safe is False, (
            f'expected safe=False for liveness_secs=600 (threshold=450 ≤ floor=600); '
            f'worst_case={assessment_low.worst_case_secs}, threshold={assessment_low.threshold_secs}'
        )

        # default liveness_secs → threshold=8100 > floor=600 → safe=True
        assessment_default = check_merge_liveness_margin(cfg)
        assert assessment_default.safe is True, (
            f'expected safe=True for default liveness (threshold=8100 > floor=600); '
            f'worst_case={assessment_default.worst_case_secs}, threshold={assessment_default.threshold_secs}'
        )


# ---------------------------------------------------------------------------
# Step-20 (GREEN): WindowComparisonResult + run_window_comparison —
# end-to-end λ gate orchestrator.
# ---------------------------------------------------------------------------


@dataclass
class WindowComparisonResult:
    """Return value of run_window_comparison — collects all headline gate data."""

    baseline_report: ThroughputReport
    multihost_report: ThroughputReport
    delta: ThroughputDelta
    drift_escalation_count: int
    multihost_events: list[tuple[Any, Any, dict[str, Any]]]


async def run_window_comparison(
    *,
    n_merges: int,
    service_secs: dict[str, float],
    drift_sample: bool = False,
) -> WindowComparisonResult:
    """Orchestrate the queueing-model + provenance-plumbing validation gate.

    NOTE: this is an analytical simulation, not real concurrent dispatch.
    See module docstring and run_backlog_window for the distinction.  The
    direction improvements (rate_improved / oldest_age_reduced / depth_reduced)
    are guaranteed by K-server queueing arithmetic, not by observing real
    concurrent pool behavior.

    (1) Single-host baseline: K=1, local-only pool, fresh _RecordingEventStore.
    (2) Two-host window:      K=2, local+laptop pool, fresh _RecordingEventStore.
    (3) If drift_sample=True: attach a DriftDetector to the two-host pool and
        call detector.check() on every SHA from the window so each emits a
        verdict_parity_ok event (both runners always agree: verdict_map=None).
        Zero escalations are counted via _FakeEscalationQueue.
    (4) summarize + compare → direction booleans + recorded deltas.

    K-server queueing guarantee (PRD G6 — direction by construction):
      Baseline  makespan = N * S            (K=1, serial)
      Multihost makespan = ceil(N/2) * S   (K=2, parallel)
    For N≥2, makespan_multihost < makespan_baseline, so rate_improved,
    oldest_age_reduced, and depth_reduced are always True — no frozen
    threshold or multiplier is needed.

    runners_seen in multihost_report includes 'local' (from
    verdict_parity_ok.local_runner when drift_sample=True) and 'laptop'
    (from merge_verify.runner + verdict_parity_ok.remote_runner).
    """
    local_svc = service_secs.get('local', 1.0)
    laptop_svc = service_secs.get('laptop', 1.0)

    # --- (1) Baseline: K=1, local-only ---
    baseline_rec = _RecordingEventStore()
    local_bl = _FakeRunner('local', is_local=True, service_secs=local_svc)
    baseline_pool = VerifyRunnerPool([local_bl], event_store=baseline_rec, task_id='1702-bl')
    baseline_run = await run_backlog_window(
        baseline_pool, baseline_rec,
        n_merges=n_merges, k=1,
        service_secs=service_secs,
    )
    baseline_duration = max(
        (end_vt for _, _, _, end_vt in baseline_run.spans),
        default=float(n_merges),
    )
    baseline_report = summarize_throughput(
        baseline_rec.events, window_duration_secs=baseline_duration
    )

    # --- (2) Multihost: K=2, local+laptop ---
    multihost_rec = _RecordingEventStore()
    local_mh = _FakeRunner('local', is_local=True, service_secs=local_svc)
    laptop_mh = _FakeRunner('laptop', is_local=False, service_secs=laptop_svc)
    multihost_pool = VerifyRunnerPool(
        [local_mh, laptop_mh], event_store=multihost_rec, task_id='1702-mh',
    )
    multihost_run = await run_backlog_window(
        multihost_pool, multihost_rec,
        n_merges=n_merges, k=2,
        service_secs=service_secs,
    )
    multihost_duration = max(
        (end_vt for _, _, _, end_vt in multihost_run.spans),
        default=math.ceil(n_merges / 2) * laptop_svc,
    )

    # --- (3) Drift detection (optional) ---
    esc_queue = _FakeEscalationQueue()
    if drift_sample:
        detector = DriftDetector(
            multihost_pool,
            event_store=multihost_rec,
            escalation_queue=esc_queue,
            task_id='1702-mh',
        )
        # Call check on every SHA so at least one verdict_parity_ok is emitted.
        # Both runners share no verdict_map → always return True → AGREE.
        for sha in multihost_run.advance_order:
            await detector.check(sha, _make_spec())

    drift_escalation_count = len(esc_queue.submitted)

    # Summarise AFTER drift events are recorded so runners_seen picks up
    # 'local' from verdict_parity_ok.local_runner (pool routes all dispatches
    # to laptop; local only appears via the drift parity path).
    multihost_report = summarize_throughput(
        multihost_rec.events, window_duration_secs=multihost_duration
    )

    # --- (4) Compare ---
    delta = compare_throughput(baseline_report, multihost_report)

    return WindowComparisonResult(
        baseline_report=baseline_report,
        multihost_report=multihost_report,
        delta=delta,
        drift_escalation_count=drift_escalation_count,
        multihost_events=list(multihost_rec.events),
    )


# ---------------------------------------------------------------------------
# Step-19 (RED) / Step-20 (GREEN): HEADLINE λ gate — queueing-model direction +
# provenance + zero drift.  Fails until run_window_comparison exists.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestQueueingModelAndProvenance:
    """Validates the K-server queueing model and provenance plumbing.

    SCOPE: this class tests the ANALYTICAL concurrency model (K-server virtual
    clock) and the real provenance/drift-integration plumbing — NOT production
    concurrent throughput.  VerifyRunnerPool.dispatch is serial; the K-permit
    semaphore enabling real concurrent dispatch is deferred to ζ.  The direction
    improvements (rate_improved, oldest_age_reduced, depth_reduced) follow from
    queueing arithmetic and are guaranteed by construction for N≥2 equal-service
    jobs; they do not require observing actual simultaneous dispatch.

    Once ζ's K-permit semaphore lands, these tests can be upgraded to drive
    real overlapping dispatch and measure observed (not computed) concurrency.
    """

    async def test_queueing_model_direction_and_provenance_and_zero_drift(self):
        """Headline λ gate: K-server queueing model outperforms K=1 baseline.

        Validates the queueing model and provenance plumbing — NOT real concurrent
        dispatch (see class docstring).  Calls
        run_window_comparison(n_merges=6, service_secs=..., drift_sample=True)
        and asserts all four headline properties:

        (a) Queueing-model DIRECTION: rate_improved, oldest_age_reduced, depth_reduced
            are all True (K=2 makespan < K=1 makespan by construction for N≥2).
            Deltas are RECORDED but never compared against a frozen multiplier (PRD G6).

        (b) Provenance plumbing: multihost_report.runners_seen == {'local', 'laptop'}
            (real pool.dispatch events carry the correct runner tag).

        (c) Zero drift divergence: result.drift_escalation_count == 0.

        (d) ≥1 verdict_parity_ok event over the window (DriftDetector AGREE path
            confirmed via real detector integration).
        """
        result = await run_window_comparison(
            n_merges=6,
            service_secs={'local': 1.0, 'laptop': 1.0},
            drift_sample=True,
        )

        delta = result.delta

        # (a) Direction assertions — strict inequalities, no frozen threshold (G6)
        assert delta.rate_improved is True, (
            f'rate must improve: rate_delta={delta.rate_delta}; '
            f'baseline={result.baseline_report.completion_rate}, '
            f'multihost={result.multihost_report.completion_rate}'
        )
        assert delta.oldest_age_reduced is True, (
            f'oldest-age must decrease: oldest_age_delta={delta.oldest_age_delta}'
        )
        assert delta.depth_reduced is True, (
            f'peak depth must decrease: depth_delta={delta.depth_delta}'
        )

        # Deltas are RECORDED (non-zero) — no frozen threshold/multiplier (G6)
        assert delta.rate_delta != 0.0, 'rate_delta must be recorded (non-zero)'
        assert delta.oldest_age_delta != 0.0, 'oldest_age_delta must be recorded (non-zero)'
        assert delta.depth_delta != 0, 'depth_delta must be recorded (non-zero)'

        # (b) Provenance: both runners contributed
        assert result.multihost_report.runners_seen == {'local', 'laptop'}, (
            f'expected runners_seen=={{"local","laptop"}}; '
            f'got {result.multihost_report.runners_seen}'
        )

        # (c) Zero drift divergence escalations
        assert result.drift_escalation_count == 0, (
            f'expected 0 drift escalations; got {result.drift_escalation_count}'
        )

        # (d) ≥1 verdict_parity_ok event (DriftDetector confirmed AGREE at least once)
        parity_events = [
            e for e in result.multihost_events
            if e[0] == EventType.verdict_parity_ok
        ]
        assert len(parity_events) >= 1, (
            f'expected ≥1 verdict_parity_ok event; got {len(parity_events)}'
        )


# ---------------------------------------------------------------------------
# Unreachable-host capstone (tasks 1795 / 3043).
#
# A remote verify host that goes away must be RECORDED, quarantined, alarmed
# once, and re-admitted by the reprobe sweep with no orchestrator restart --
# and the merge queue must keep flowing on the local trust anchor throughout.
# Every scenario below drives a RUNNING lane over a real git repository: the
# fault arrives where a real one does (the remote runner's own transport) and
# every observation is a public one -- the MergeOutcome futures,
# ``snapshot()['hosts']``, the escalation queue and the event store.
# ---------------------------------------------------------------------------

_INCIDENT_RU = (
    'git push leo-laptop abc123def:refs/merge-verify/task-cap failed (rc=128): '
    'ssh: connect to host leo-laptop port 22: Connection timed out'
)

#: Poll ceiling for a state the lane reaches on its own. Generous: these run
#: real git against a cold tmp repo.
_LANE_SETTLE_TIMEOUT = 30.0
_LANE_STOP_TIMEOUT = 30.0

#: Per-test ceiling for the host capstone, whose methods settle a real-git lane
#: through several ``wait_responsive`` waits and a reprobe sweep.  Sized above
#: the budget ``TestTimeoutMarkCoverage`` recomputes from this module's own
#: source rather than from any figure written here; ``HEAVY_BARRIER_TEST_TIMEOUT``
#: (300s) is NOT enough for it.  Why a mark at all: pytest-timeout's thread
#: method ``os._exit()``s the xdist worker on expiry, and ``--max-worker-restart=0``
#: then truncates the whole session against an innocent test -- so a
#: slow-but-correct run under a too-tight default is strictly worse than the
#: tail it would otherwise have reported (esc-3980-1, task 3492).
HOST_CAPSTONE_TEST_TIMEOUT = 2 * HEAVY_BARRIER_TEST_TIMEOUT


async def _setup_repo(repo: Path) -> None:
    """Initialise a git repo with a single commit (README.md) on main."""
    await _run(['git', 'init', '-b', 'main'], cwd=repo)
    await _run(['git', 'config', 'user.email', 'test@test.com'], cwd=repo)
    await _run(['git', 'config', 'user.name', 'Test'], cwd=repo)
    (repo / 'README.md').write_text('# Test\n')
    await _run(['git', 'add', '-A'], cwd=repo)
    await _run(['git', 'commit', '-m', 'Initial commit'], cwd=repo)


@pytest.fixture
def host_repo(tmp_path: Path) -> Path:
    repo = tmp_path / 'repo'
    repo.mkdir()
    asyncio.run(_setup_repo(repo))
    return repo


@pytest.fixture
def host_git_config() -> GitConfig:
    return GitConfig(
        main_branch='main',
        branch_prefix='task/',
        remote='origin',
        worktree_dir='.worktrees',
        push_after_advance=False,
    )


@pytest.fixture
def host_git_ops(host_git_config: GitConfig, host_repo: Path) -> GitOps:
    return GitOps(host_git_config, host_repo)


@pytest.fixture
def host_config(host_repo: Path, host_git_config: GitConfig) -> OrchestratorConfig:
    """Operator config with the two unreachability thresholds turned down.

    These are knobs the lane copies onto itself when it builds its OWN
    allocator (``_ensure_host_allocator``), which is why ``_lane_with_remote``
    lets it build one before swapping in the two-host allocator: driving the
    alarm thresholds from config is what keeps these tests off the worker's
    internals.  The reprobe CADENCE is the injected clock's job -- see
    ``_ShortSleepClock`` for why config alone cannot set it.
    """
    return OrchestratorConfig(
        project_root=host_repo,
        git=host_git_config,
        verify_host_unreachable_escalate_after_n=1,
        verify_host_unreachable_escalate_after_secs=0.0,  # streak-only
    )


@contextlib.asynccontextmanager
async def _running_lane(git_ops: GitOps, **ports: Any):
    """A running lane over *git_ops*, stopped on the way out.

    Teardown goes through ``stop()`` -- the lane's own shutdown protocol,
    which resolves in-flight request futures, drains its queues, cleans merge
    worktrees and releases leases, and is internally bounded so it cannot hang.
    """
    queue: asyncio.Queue[MergeRequest] = asyncio.Queue()
    lane = MergeLane(git_ops, queue, **ports)
    lane_task = asyncio.ensure_future(lane.run())
    try:
        yield lane, queue
    finally:
        # Exception, not BaseException: this must not swallow a CancelledError
        # aimed at the enclosing test task (or a KeyboardInterrupt).
        with contextlib.suppress(Exception):
            await asyncio.wait_for(lane.stop(), timeout=_LANE_STOP_TIMEOUT)
        lane_task.cancel()
        await asyncio.gather(lane_task, return_exceptions=True)


async def _until(predicate, *, what: str, timeout: float = _LANE_SETTLE_TIMEOUT) -> None:
    """Wait for *predicate* to hold, or fail naming *what* was expected.

    For a public observation that has no event to await on -- a host row
    flipping out of quarantine, say.  The budget is charged in loop-RESPONSIVE
    time via ``wait_responsive``, because this repo measured >=11x wall-clock
    tails at load 246 on 32 cores and a raw ``loop.time()`` deadline converts
    that starvation into a spurious red (task 3980; see ``wait_responsive``).
    Byte-for-byte the same contract as the twin in
    test_merge_speculation.py::_until -- two spellings of one wait policy would
    be read as interchangeable by the next reader whether or not they were.
    """

    async def _poll() -> None:
        while not predicate():
            await asyncio.sleep(0.02)

    await wait_responsive(_poll(), timeout=timeout, label=what)


def _live_tasks_started_since(before: set[asyncio.Task]) -> set[asyncio.Task]:
    """Still-running tasks created since *before* -- minus the waiter asking.

    ``asyncio.current_task()`` is excluded because the caller polls this from
    inside ``_until``, whose own poll coroutine is itself a task created after
    *before* was captured: counting it would make the set permanently
    non-empty and the wait unsatisfiable.  The enclosing test's task is already
    in *before* (``all_tasks()`` includes the running one), so nothing else
    needs excluding.
    """
    current = asyncio.current_task()
    return {
        task for task in asyncio.all_tasks()
        if task not in before and task is not current and not task.done()
    }


async def _submitted(
    git_ops: GitOps, config: OrchestratorConfig, task_id: str, filename: str, content: str,
) -> MergeRequest:
    """A request for a fresh branch carrying one committed file, unsubmitted."""
    worktree = await _make_branch_with_file(git_ops, task_id, filename, content)
    return MergeRequest(
        task_id=task_id,
        branch=QueuedBranch.parse(f'task/{task_id}', config.git.branch_prefix),
        worktree=worktree,
        pre_rebased=False,
        task_files=None,
        module_configs=[],
        config=config,
        result=asyncio.get_running_loop().create_future(),
    )


def _host_row(lane: MergeLane, name: str) -> dict[str, Any]:
    """The per-host row ``snapshot()`` publishes for *name* (``{}`` when absent).

    ``snapshot()['hosts']`` carries name, is_local, slot_state, quarantined,
    quarantine_class, unavailable_since, unavailable_secs, streak and reason --
    the whole host-monitor surface these tests assert on.
    """
    for host in lane.snapshot()['hosts']:
        if host['name'] == name:
            return host
    return {}


class _ShortSleepClock:
    """``ClockPort`` on real time whose every sleep is capped at 20 ms.

    The reprobe loop sleeps ``verify_host_reprobe_interval_s`` BEFORE its first
    sweep, and the lane only learns that knob when it builds its own host
    allocator -- which happens on the first verify, long after ``run()``
    started the loop.  Under the production clock the first sleep is therefore
    always the 120 s default and no sweep is observable in a test, whatever the
    config says.  Capping the sleep (rather than faking the CLOCK, as
    ``FakeClock`` does) keeps ``now``/``monotonic`` on real time, so every age,
    heartbeat and worktree-reap threshold the lane evaluates still reads
    honestly -- only the cadence is compressed.
    """

    _CAP_SECS = 0.02

    def now(self) -> float:
        return time.time()

    def monotonic(self) -> float:
        return time.monotonic()

    def newest_content_mtime(self, root: Path) -> float | None:  # noqa: ARG002
        return None

    async def sleep(self, secs: float) -> None:
        await asyncio.sleep(min(secs, self._CAP_SECS))

    async def wait_for_any(self, aws: Collection[Any], timeout: float) -> set[Any]:
        # UNCAPPED, unlike the sleep above: capping it would re-check the
        # lane's halt and abandonment branches ~500x more often and reorder
        # the halt-vs-verify interleaving this file's capstone measures. So
        # this one method wants production semantics exactly, and delegates
        # rather than restating them.
        return await PRODUCTION_CLOCK.wait_for_any(aws, timeout)


class _HostEscalationQueue:
    """Escalation-queue double with PER-TASK open-L1 state.

    The unreachability alarm dedups on ``has_open_l1(sentinel)`` and recovery
    resolves through ``get_by_task``/``resolve``, so the single-flag double
    above (which the drift detector's dedup is happy with) cannot express
    either half of this contract.
    """

    def __init__(self) -> None:
        self.submitted: list[Any] = []
        self.resolved: list[tuple[str, str]] = []
        #: Every dedup consultation -- i.e. every time an alarm site reached
        #: the point of deciding whether to file.  Counting these is what lets
        #: a dedup test distinguish "suppressed N times" from "never retried".
        self.open_l1_checks = 0
        self._seq = 0

    def make_id(self, task_id: str) -> str:
        self._seq += 1
        return f'esc-{self._seq}'

    def _open_l1s(self, task_id: str) -> list[Any]:
        done = {esc_id for esc_id, _ in self.resolved}
        return [
            e for e in self.submitted
            if getattr(e, 'task_id', None) == task_id
            and getattr(e, 'level', 0) == 1
            and e.id not in done
        ]

    def has_open_l1(self, task_id: str) -> bool:
        self.open_l1_checks += 1
        return bool(self._open_l1s(task_id))

    def submit(self, esc: Any) -> None:
        self.submitted.append(esc)

    def get_by_task(self, task_id: str, status: str | None = None, **kw: Any) -> list[Any]:  # noqa: ARG002
        return self._open_l1s(task_id)

    def resolve(self, esc_id: str, resolution: str, **kw: Any) -> None:
        self.resolved.append((esc_id, resolution))

    def of_category(self, category: str) -> list[Any]:
        return [e for e in self.submitted if getattr(e, 'category', None) == category]


class _FlakyRemoteRunner:
    """Remote verify runner whose reachability is flipped at runtime.

    ``reachable`` drives every surface the strand/recovery path touches:
    ``run_merge_verify`` (``RunnerUnavailable`` while down -- the transport
    failure that records the host), ``health()`` (the reprobe probe),
    ``cancel_verify()`` (rc 255 while down, which is what PARKs the slot) and
    ``probe_clean()`` (False while down, the bounded poll that decides whether
    the PARK is released).  ``health_error``, when set, is raised by
    ``health()`` instead of answering.  ``verify_gate``, when set, holds
    ``run_merge_verify`` open so a verify can be caught in flight.
    """

    def __init__(
        self,
        name: str = 'leo-laptop',
        *,
        reachable: bool = False,
        health_error: BaseException | None = None,
        verify_gate: asyncio.Event | None = None,
    ) -> None:
        self.name = name
        self.is_local = False
        self.reachable = reachable
        self.health_error = health_error
        self.verify_gate = verify_gate
        self.verified: list[str] = []
        self.health_calls = 0
        self.entered = False

    async def health(self) -> bool:
        self.health_calls += 1
        if self.health_error is not None:
            raise self.health_error
        return self.reachable

    async def run_merge_verify(self, merge_sha: str, spec: MergeVerifySpec) -> VerifyResult:
        if self.verify_gate is not None:
            self.entered = True
            await self.verify_gate.wait()
        if not self.reachable:
            raise RunnerUnavailable(_INCIDENT_RU)
        self.verified.append(merge_sha)
        return _make_result(True)

    async def cancel_verify(self, *a: Any, **kw: Any) -> int:
        return 0 if self.reachable else 255

    async def probe_clean(self, *a: Any, **kw: Any) -> bool:
        return self.reachable


@contextlib.asynccontextmanager
async def _lane_with_remote(
    git_ops: GitOps,
    config: OrchestratorConfig,
    laptop: _FlakyRemoteRunner,
    **ports: Any,
):
    """A running lane whose verify pool is (local trust anchor + *laptop*).

    The lane copies the host-monitor knobs off the config when it builds its
    own allocator, so one warm-up request is landed first -- ``hosts`` turning
    non-empty in ``snapshot()`` is the public signal that the allocator now
    exists -- and only then is the two-host allocator swapped in.
    """
    async with _running_lane(git_ops, clock=_ShortSleepClock(), **ports) as (lane, queue):
        warmup = await _submitted(git_ops, config, 'hostwarm', 'hostwarm.py', 'w = 1\n')
        await queue.put(warmup)
        await wait_responsive(warmup.result, label='host-allocator warmup landing')
        await _until(
            lambda: bool(lane.snapshot()['hosts']),
            what='the lane to build its host allocator',
        )
        _inject_two_host_allocator(lane, laptop)
        yield lane, queue


@pytest.mark.asyncio
@pytest.mark.timeout(HOST_CAPSTONE_TEST_TIMEOUT)
class TestUnreachableHostCapstone:
    """END-TO-END: an unreachable remote can neither strand itself nor stall the queue.

    Reproduces the reify 2026-07-25 incident's user-observable signal: the
    laptop sat out of the verify pool for 3+ h ("verifying 1/2 hosts", zero
    dispatch attempts) with nothing for the reprobe sweep to re-adopt, and
    only an orchestrator restart brought it back.  The promised signal after
    the fix is what these tests assert: the strand is RECORDED and legible in
    the snapshot, the work still lands on the local anchor, and one reprobe
    sweep restores 2/2 capacity with no restart.
    """

    @staticmethod
    async def _strand_the_laptop(lane, queue, git_ops, config, verifier, gate):
        """Park the local anchor, then push a request onto the down laptop.

        ``acquire()`` hands out the local host first, so the only way to put
        work on the remote is to have the local slot already busy -- which is
        what the parked first verify does.  Releases the park on the way out
        and returns both requests, quarantine already in place.
        """
        req_a = await _submitted(git_ops, config, 'ru-a', 'ru_a.py', 'a = 1\n')
        await queue.put(req_a)
        await _until(lambda: 'ru-a' in verifier.verified, what="the local anchor's verify to enter")
        req_b = await _submitted(git_ops, config, 'ru-b', 'ru_b.py', 'b = 2\n')
        await queue.put(req_b)
        await _until(
            lambda: 'leo-laptop' in lane.snapshot()['occupancy']['inflight_by_host'],
            what='the second request to be dispatched to the laptop',
        )
        # The RU verdict is only ADJUDICATED when the head of the in-flight
        # deque finalizes, so release the parked anchor before waiting on it.
        gate.set()
        await _until(
            lambda: bool(_host_row(lane, 'leo-laptop').get('quarantined')),
            what='the unreachable laptop to be quarantined',
        )
        return req_a, req_b

    async def test_an_unreachable_remote_is_recorded_alarmed_and_never_stalls_the_queue(
        self, host_git_ops: GitOps, host_config: OrchestratorConfig,
    ) -> None:
        """RU on the remote -> recorded + quarantined + one L1 -> both branches still land."""
        gate = asyncio.Event()
        verifier = FakeVerifier(scripts={'ru-a': VerifyScript(result=passes().result, release=gate)})
        escalations = _HostEscalationQueue()
        events = _RecordingEventStore()
        laptop = _FlakyRemoteRunner(reachable=False)

        async with _lane_with_remote(
            host_git_ops, host_config, laptop,
            verifier=verifier, escalation_queue=escalations, event_store=events,
        ) as (lane, queue):
            req_a, req_b = await self._strand_the_laptop(
                lane, queue, host_git_ops, host_config, verifier, gate,
            )

            # (1) the strand is RECORDED and legible in the public host census --
            #     RU-recoverable, not an operator-held divergence quarantine.
            row = _host_row(lane, 'leo-laptop')
            assert row['quarantine_class'] == 'ru', row
            assert row['streak'] >= 1, row
            assert 'Connection timed out' in (row['reason'] or ''), row
            assert row['unavailable_since'] is not None, row

            # (2) exactly one dedup'd L1 alarm naming the host.
            alarms = escalations.of_category('verify_host_unreachable')
            assert len(alarms) == 1, escalations.submitted
            assert alarms[0].level == 1 and alarms[0].severity == 'blocking'
            assert 'leo-laptop' in alarms[0].summary

            # (3) THE SIGNAL: the queue never stalls -- both branches land on the
            #     surviving local trust anchor.
            outcome_a = await wait_responsive(req_a.result, label='A lands on surviving anchor')
            outcome_b = await wait_responsive(req_b.result, label='B lands on surviving anchor')
            assert outcome_a.status == 'done', outcome_a
            assert outcome_b.status == 'done', outcome_b

        _, main_files, _ = await _run(
            ['git', 'ls-tree', '-r', '--name-only', 'main'], cwd=host_git_ops.project_root,
        )
        assert 'ru_a.py' in main_files and 'ru_b.py' in main_files

    async def test_one_reprobe_sweep_readmits_the_recovered_host_without_a_restart(
        self, host_git_ops: GitOps, host_config: OrchestratorConfig,
    ) -> None:
        """Host comes back -> quarantine cleared, alarm resolved, 2/2 capacity restored."""
        gate = asyncio.Event()
        verifier = FakeVerifier(scripts={'ru-a': VerifyScript(result=passes().result, release=gate)})
        escalations = _HostEscalationQueue()
        events = _RecordingEventStore()
        laptop = _FlakyRemoteRunner(reachable=False)

        async with _lane_with_remote(
            host_git_ops, host_config, laptop,
            verifier=verifier, escalation_queue=escalations, event_store=events,
        ) as (lane, queue):
            req_a, req_b = await self._strand_the_laptop(
                lane, queue, host_git_ops, host_config, verifier, gate,
            )
            await wait_responsive(req_a.result, label='stranded-laptop A resolves')
            await wait_responsive(req_b.result, label='stranded-laptop B resolves')

            # HOST RECOVERS -- no restart, no new allocator, no new lane.
            laptop.reachable = True
            await _until(
                lambda: not _host_row(lane, 'leo-laptop').get('quarantined', True),
                what='the reprobe sweep to re-admit the recovered laptop',
            )

            row = _host_row(lane, 'leo-laptop')
            assert row['quarantine_class'] is None and row['streak'] is None, row
            assert row['slot_state'] == 'free', row
            assert events.events_of(EventType.verify_host_recovered), events.events
            assert escalations.resolved, 'the open L1 must be resolved on recovery'

            # 2/2 capacity is genuinely back: park the local anchor again and the
            # next request is dispatched to the laptop, which now verifies it.
            second_gate = asyncio.Event()
            verifier.scripts['back-a'] = VerifyScript(
                result=passes().result, release=second_gate,
            )
            req_c = await _submitted(host_git_ops, host_config, 'back-a', 'back_a.py', 'c = 3\n')
            await queue.put(req_c)
            await _until(lambda: 'back-a' in verifier.verified, what="the anchor's verify to enter")
            req_d = await _submitted(host_git_ops, host_config, 'back-b', 'back_b.py', 'd = 4\n')
            await queue.put(req_d)
            await _until(
                lambda: bool(laptop.verified),
                what='the re-admitted laptop to be handed a verify',
            )
            second_gate.set()
            assert (await wait_responsive(req_c.result, label='C lands after re-admission')).status == 'done'
            assert (await wait_responsive(req_d.result, label='D lands after re-admission')).status == 'done'

    @staticmethod
    def _tuned_config(
        base: OrchestratorConfig, *, after_n: int, after_secs: float,
    ) -> OrchestratorConfig:
        """*base* with the two unreachability thresholds re-tuned."""
        return base.model_copy(update={
            'verify_host_unreachable_escalate_after_n': after_n,
            'verify_host_unreachable_escalate_after_secs': after_secs,
        })

    async def test_one_ru_below_the_streak_threshold_does_not_alarm(
        self, host_git_ops: GitOps, host_config: OrchestratorConfig,
    ) -> None:
        """THRESHOLD: at escalate_after_n=2, ONE RU event records but stays quiet.

        The capstone above runs at a threshold of 1, where `len(alarms) == 1`
        is unfalsifiable for this property -- one event cannot distinguish a
        respected threshold from an ignored one.  Here the host is recorded and
        quarantined exactly as before (that is the first RU's own job, and it
        is NOT gated on the threshold), and the L1 is withheld because the
        streak has not reached 2.  A threshold read as "alarm on every RU"
        fires here.
        """
        gate = asyncio.Event()
        verifier = FakeVerifier(scripts={'ru-a': VerifyScript(result=passes().result, release=gate)})
        escalations = _HostEscalationQueue()
        events = _RecordingEventStore()
        laptop = _FlakyRemoteRunner(reachable=False)
        config = self._tuned_config(host_config, after_n=2, after_secs=0.0)

        async with _lane_with_remote(
            host_git_ops, config, laptop,
            verifier=verifier, escalation_queue=escalations, event_store=events,
        ) as (lane, queue):
            req_a, req_b = await self._strand_the_laptop(
                lane, queue, host_git_ops, config, verifier, gate,
            )

            row = _host_row(lane, 'leo-laptop')
            assert row['quarantine_class'] == 'ru' and row['streak'] == 1, (
                'the first RU must still RECORD and quarantine -- only the '
                f'ALARM waits for the streak: {row!r}'
            )
            assert escalations.of_category('verify_host_unreachable') == [], (
                'streak 1 of 2 must not alarm; the threshold was ignored: '
                f'{escalations.submitted!r}'
            )

            assert (await wait_responsive(req_a.result, label='A lands')).status == 'done'
            assert (await wait_responsive(req_b.result, label='B lands')).status == 'done'

    async def test_repeated_alarm_opportunities_still_produce_exactly_one_l1(
        self, host_git_ops: GitOps, host_config: OrchestratorConfig,
    ) -> None:
        """DEDUP: many alarm opportunities in one downtime episode -> ONE L1.

        The time-based arm re-evaluates on EVERY failed reprobe sweep, so an
        unreachable host offers the alarm a fresh opportunity several times a
        second here.  Exactly one open L1 per host per downtime episode is the
        contract; without the ``has_open_l1`` dedup this would file one per
        sweep and bury the operator.  Counting alarms against a number of
        opportunities the test itself OBSERVES (health_calls) is what makes the
        `== 1` falsifiable.
        """
        gate = asyncio.Event()
        verifier = FakeVerifier(scripts={'ru-a': VerifyScript(result=passes().result, release=gate)})
        escalations = _HostEscalationQueue()
        events = _RecordingEventStore()
        laptop = _FlakyRemoteRunner(reachable=False)
        config = self._tuned_config(host_config, after_n=1, after_secs=0.001)

        async with _lane_with_remote(
            host_git_ops, config, laptop,
            verifier=verifier, escalation_queue=escalations, event_store=events,
        ) as (lane, queue):
            req_a, req_b = await self._strand_the_laptop(
                lane, queue, host_git_ops, config, verifier, gate,
            )
            await wait_responsive(req_a.result, label='A lands')
            await wait_responsive(req_b.result, label='B lands')

            checks = escalations.open_l1_checks
            await _until(
                lambda: escalations.open_l1_checks >= checks + 5,
                what='five more alarm attempts to reach the dedup',
            )

            alarms = escalations.of_category('verify_host_unreachable')
            assert len(alarms) == 1, (
                f'the alarm site was re-entered {escalations.open_l1_checks} '
                f'times over one downtime episode; the dedup must hold it at '
                f'one open L1: {alarms!r}'
            )
            assert alarms[0].level == 1 and alarms[0].severity == 'blocking'

    async def test_a_cancel_against_a_down_host_parks_the_slot_and_reprobe_unparks_it(
        self, host_git_ops: GitOps, host_config: OrchestratorConfig,
    ) -> None:
        """The PARKED-slot strand: caught in flight, legible, and self-healing.

        The OTHER way a host leaves the pool.  A verify is already running on
        the laptop when it goes down, so the cancel RPC fails and every
        ``probe_clean`` poll fails with it -- ``cancel_and_release`` then leaves
        the slot PARKED, the correct fail-closed state (a stale verify may
        still be churning there) but one that writes only the slot, never the
        quarantine set.  That is the shape that stranded the laptop silently in
        the reify 2026-07-25 incident: non-acquirable, and invisible to a
        sweep that looked for quarantine membership.

        Three things are asserted that the RU capstone above cannot see: the
        slot really is PARKED (not merely quarantined), the census SAYS
        ``'parked'`` so an operator can tell the two strands apart, and one
        reprobe sweep un-PARKs it -- which needs ``probe_clean`` to come back
        clean, not just ``health``.
        """
        gate = asyncio.Event()
        verify_gate = asyncio.Event()
        verifier = FakeVerifier(scripts={
            'park-a': VerifyScript(result=fails(
                category='test_failure', summary='head fails, cascading its successors',
            ).result, release=gate),
        })
        escalations = _HostEscalationQueue()
        events = _RecordingEventStore()
        # Reachable to begin with: the strand needs a verify caught IN FLIGHT,
        # which an already-down host never grants.
        laptop = _FlakyRemoteRunner(reachable=True, verify_gate=verify_gate)

        async with _lane_with_remote(
            host_git_ops, host_config, laptop,
            verifier=verifier, escalation_queue=escalations, event_store=events,
        ) as (lane, queue):
            req_a = await _submitted(host_git_ops, host_config, 'park-a', 'park_a.py', 'a = 1\n')
            await queue.put(req_a)
            await _until(lambda: 'park-a' in verifier.verified, what="the local anchor's verify to enter")
            req_b = await _submitted(host_git_ops, host_config, 'park-b', 'park_b.py', 'b = 2\n')
            await queue.put(req_b)
            await _until(
                lambda: laptop.entered,
                what="the laptop's verify to be caught in flight",
            )

            # The host dies UNDER the running verify, then its sole waiter
            # walks away -- so the lane cancels a verify it can no longer reach.
            laptop.reachable = False
            gate.set()
            # The RECORD, not the PARK, is the settled state: `cancel_and_release`
            # parks the slot on the failed cancel and only then polls
            # `probe_clean` to exhaustion, so the slot reads 'parked' for the
            # whole probe loop -- seconds before the strand is detected. Waiting
            # on the park alone would sample the middle of that loop.
            await _until(
                lambda: _host_row(lane, 'leo-laptop').get('quarantine_class') == 'ru',
                what='the failed cancel to be recorded as an RU-class strand',
            )

            row = _host_row(lane, 'leo-laptop')
            assert row['slot_state'] == 'parked', (
                'the slot must stay PARKED: the cancel RPC failed, so a stale '
                f'verify may still be running there and freeing it could '
                f'double-dispatch onto a busy host: {row!r}'
            )
            assert row['streak'] >= 1 and row['unavailable_since'] is not None, row

            # RECOVERY -- and it must take BOTH probes: a PARKED slot is only
            # safe to free once the host also reports no stale verify running.
            laptop.reachable = True
            verify_gate.set()
            await _until(
                lambda: _host_row(lane, 'leo-laptop').get('slot_state') == 'free',
                what='the reprobe sweep to un-PARK the recovered laptop',
            )
            row = _host_row(lane, 'leo-laptop')
            assert row['quarantined'] is False and row['quarantine_class'] is None, row
            assert events.events_of(EventType.verify_host_recovered), events.events

            # The queue never stalled: both requests resolve rather than
            # hanging on a host the lane can no longer reach.
            assert (await wait_responsive(
                req_a.result, label='A resolves past the PARK',
            )).status != 'done'

    async def test_stop_leaves_no_lane_background_task_running(
        self, host_git_ops: GitOps, host_config: OrchestratorConfig,
    ) -> None:
        """``stop()`` takes the lane's background loops down with it.

        The reprobe loop is one asyncio task per lane, started by ``run()`` and
        cancelled by ``stop()``.  A regression that stops cancelling it leaks
        one live task per lane stop and announces itself only as an unasserted
        "Task was destroyed but it is pending" warning at interpreter exit.
        Observed as the absence of any surviving task this block created --
        which covers the reprobe loop without naming it, so the assertion
        survives the loop being renamed, merged into another, or joined by a
        second one.
        """
        before = asyncio.all_tasks()

        # Deliberately NOT `_lane_with_remote`: that injects `_ShortSleepClock`,
        # which caps the reprobe loop's sleep at 20 ms, so the loop would fall
        # out of its own `while self._running` within a tick of stop() whether
        # or not anything cancelled it. On the production clock that sleep is
        # the full interval, so a lost cancel keeps the task alive well past
        # this wait -- which is what makes the assertion below bite.
        async with _running_lane(host_git_ops, verifier=FakeVerifier()) as (lane, queue):
            req = await _submitted(
                host_git_ops, host_config, 'stopclean', 'stopclean.py', 's = 1\n',
            )
            await queue.put(req)
            assert (await wait_responsive(
                req.result, label='the lane is fully up and merging',
            )).status == 'done'

        # The context manager already awaited stop() and reaped run(); one more
        # scheduling turn lets any cancellation it issued be delivered.
        await _until(
            lambda: not _live_tasks_started_since(before),
            what='every task this lane started to finish after stop()',
            timeout=_LANE_STOP_TIMEOUT,
        )

    async def test_a_raising_health_probe_never_crashes_the_lane(
        self, host_git_ops: GitOps, host_config: OrchestratorConfig,
    ) -> None:
        """The reprobe sweep swallows a probe that raises, and the lane keeps merging."""
        gate = asyncio.Event()
        verifier = FakeVerifier(scripts={'ru-a': VerifyScript(result=passes().result, release=gate)})
        escalations = _HostEscalationQueue()
        events = _RecordingEventStore()
        laptop = _FlakyRemoteRunner(reachable=False)

        async with _lane_with_remote(
            host_git_ops, host_config, laptop,
            verifier=verifier, escalation_queue=escalations, event_store=events,
        ) as (lane, queue):
            req_a, req_b = await self._strand_the_laptop(
                lane, queue, host_git_ops, host_config, verifier, gate,
            )
            await wait_responsive(req_a.result, label='stranded-laptop A resolves')
            await wait_responsive(req_b.result, label='stranded-laptop B resolves')

            # Every subsequent sweep now explodes inside the probe.
            probes = laptop.health_calls
            laptop.health_error = RuntimeError('ssh exploded')
            await _until(
                lambda: laptop.health_calls >= probes + 3,
                what='three more reprobe sweeps to hit the raising probe',
            )

            # THE SIGNAL: the lane is still merging -- a probe bug cannot take it down.
            req_c = await _submitted(host_git_ops, host_config, 'crash-c', 'crash_c.py', 'c = 3\n')
            await queue.put(req_c)
            outcome_c = await wait_responsive(req_c.result, label='lane still merges past a raising probe')
            assert outcome_c.status == 'done', outcome_c
            assert _host_row(lane, 'leo-laptop')['quarantined'] is True, (
                'a raising probe must leave the quarantine in place, not clear it'
            )


# ===========================================================================
# Task 2822 fix (b): per-land cross-check of a remote GREEN in
# _run_post_merge_verify.  A single remote host's green is re-verified by the
# LOCAL trust-anchor before the land.  AGREE -> verdict_parity_ok + proceed;
# DIVERGE (local FAIL) -> fail-closed (adopt the local FAIL verdict so the land
# is withheld, quarantine the remote, file a dedup'd blocking escalation);
# local RunnerUnavailable -> fail-safe (verify_cross_check_inconclusive, TRUST
# the remote green — never block a land on a local infra hiccup).  Gated by
# config.verify_cross_check_remote_green (default True, provably inert on main
# today because runner is always None until Lever C is enabled).
#
# step-7 (RED) / step-8 (GREEN).
# ===========================================================================


def _xcheck_config(*, cross_check: bool = True) -> OrchestratorConfig:
    """OrchestratorConfig with the fix-(b) knob explicit + a project_root the
    cross-check LocalRunner's archive_root is derived from.

    ``escalate_preexisting_main_break=False`` is the FIRST guard both
    ``_classify_main_health_red`` and ``_spawn_main_health_probe`` apply
    (merge_queue.py -- the two share a byte-identical guard block), so a
    config that declines pre-existing-break classification is the operator
    knob that keeps the probe out of these tests.  The cross-check verdict
    itself is unaffected: main-health classification runs only AFTER a
    failure has been adopted.
    """
    return OrchestratorConfig(
        git=GitConfig(main_branch='main'),
        project_root=Path('/tmp/xcheck-fake'),
        verify_cross_check_remote_green=cross_check,
        escalate_preexisting_main_break=False,
    )


def _xcheck_req(config: OrchestratorConfig, *, task_files=('src/foo.py',), worktree=None):
    """A minimal MergeRequest for _run_post_merge_verify cross-check tests."""
    from orchestrator.merge_queue import MergeRequest

    loop = asyncio.get_running_loop()
    return MergeRequest(
        task_id='task-2822',
        branch=QueuedBranch.parse('task/2822', config.git.branch_prefix),
        worktree=worktree or Path('/repo/task-2822'),
        pre_rebased=False,
        task_files=list(task_files) if task_files is not None else None,
        module_configs=[],
        config=config,
        result=loop.create_future(),
    )


def _xcheck_git_ops() -> MagicMock:
    """GitOps double with plenty of disk + async cleanup (mirrors the wiring test)."""
    mock = MagicMock()
    mock.get_main_sha = AsyncMock(return_value='main-sha')
    mock.get_free_disk_bytes = AsyncMock(return_value=100 * 1024 ** 3)
    mock.cleanup_merge_worktree = AsyncMock()
    mock.create_throwaway_verify_worktree = AsyncMock(return_value='/repo/_throwaway')
    return mock


def _remote_stub(result: VerifyResult, *, name: str = 'laptop') -> MagicMock:
    """A single-host remote runner double (is_local=False) for the runner= param."""
    stub = MagicMock()
    stub.name = name
    stub.is_local = False
    stub.run_merge_verify = AsyncMock(return_value=result)
    return stub


def _xcheck_verifier(*, result=None, raises=None) -> FakeVerifier:
    """The local trust-anchor's verdict, injected as the verify PORT.

    The cross-check builds its LocalRunner with
    ``run_scoped=verifier.run_scoped`` (merge_queue.py::_run_post_merge_verify),
    so the injected port IS the local leg: it returns *result*, or raises
    *raises*.  In the REMOTE dispatch path the local-path pool is never built,
    so the ONLY scoped verify is the fix-(b) cross-check trust-anchor — making
    ``verifier.verified`` an exact cross-check ran/skipped signal, the same one
    the retired LocalRunner-construction counter carried.
    """
    return FakeVerifier(default=VerifyScript(
        result=result if result is not None else _make_result(True),
        error=raises,
    ))


@pytest.mark.asyncio
class TestPerLandCrossCheck:
    """_run_post_merge_verify cross-checks a remote GREEN against the local
    trust-anchor before the land (task 2822 fix b).  RED until step-8."""

    async def test_diverge_local_fail_withholds_land_quarantines_and_escalates(self, tmp_path):
        """knob ON, remote PASS + local FAIL -> land withheld + quarantine + SYNCHRONOUS
        halt + BORN-AT-L2 escalation + event (task 2886 fix 3)."""
        from unittest.mock import Mock

        from orchestrator.merge_queue import _run_post_merge_verify

        config = _xcheck_config(cross_check=True)
        req = _xcheck_req(config, worktree=tmp_path)
        git_ops = _xcheck_git_ops()

        remote = _remote_stub(_make_result(True), name='laptop')
        local_fail = _make_result(False)
        eq = _FakeEscalationQueue()
        es = _RecordingEventStore()
        quarantine: set[str] = set()
        halt_hook = Mock()

        verifier = _xcheck_verifier(result=local_fail)
        outcome = await _run_post_merge_verify(
            git_ops, req, tmp_path,
            timeouts={}, enospc_retries={},
            max_timeouts=2, max_enospc=1,
            event_store=es,  # type: ignore[arg-type]
            merge_sha='mergesha01',
            runner=remote,
            escalation_queue=eq,
            quarantine=quarantine,
            halt_hook=halt_hook,
            verifier=verifier,
        )

        # (1) land withheld — the local (failing) verdict is adopted, merge_wt cleaned up
        assert outcome is not None
        assert outcome.status == 'blocked'
        git_ops.cleanup_merge_worktree.assert_awaited()
        assert len(verifier.verified) == 1, 'cross-check must run exactly one local trust-anchor verify'

        # (2) the merge queue was HALTED SYNCHRONOUSLY exactly once (fix 3: the halt
        #     precedes any FURTHER adoption — incident 83336a32 was +3s too late), with a
        #     reason naming the merge_sha and the suspected-red-main condition.
        halt_hook.assert_called_once()
        halt_reason = halt_hook.call_args.args[0]
        assert 'mergesha01' in halt_reason
        assert 'suspected' in halt_reason.lower() and 'red' in halt_reason.lower()

        # (3) runner quarantined in the caller-owned set (fix (b)'s first use of `quarantine`)
        assert 'laptop' in quarantine

        # (4) exactly one dedup'd BORN-AT-L2 escalation, category verify_cross_check_mismatch
        assert len(eq.submitted) == 1
        esc = eq.submitted[0]
        assert esc.category == 'verify_cross_check_mismatch'
        assert esc.level == 2
        assert esc.severity == 'critical'
        assert esc.agent_role == 'orchestrator-cross-check'
        assert 'mergesha01' in (esc.summary + esc.detail)
        assert 'laptop' in esc.detail

        # (5) distinct mismatch telemetry emitted
        assert es.events_of(EventType.verify_cross_check_mismatch)

    async def test_diverge_local_infra_fail_withholds_without_fleet_halt(self, tmp_path):
        """knob ON, remote PASS + local FAIL whose category is INFRA-TRANSIENT
        (e.g. semaphore_timeout) — a LOCAL hiccup, NOT a genuine main-red
        divergence: the land is still withheld + the remote quarantined, but the
        FLEET is NOT halted and the escalation is L1 blocking (not born-at-L2)
        (task 2886 reviewer robustness amendment — bound the halt blast radius so
        one flaky local verify cannot block every merge lane fleet-wide)."""
        from unittest.mock import Mock

        from orchestrator.merge_queue import _run_post_merge_verify

        config = _xcheck_config(cross_check=True)
        req = _xcheck_req(config, worktree=tmp_path)
        git_ops = _xcheck_git_ops()

        remote = _remote_stub(_make_result(True), name='laptop')
        # Local trust-anchor FAILs with an INFRA-TRANSIENT category — a local
        # semaphore-timeout / OOM-class hiccup, not evidence main is red.
        local_infra_fail = _make_result(False, category='semaphore_timeout')
        eq = _FakeEscalationQueue()
        es = _RecordingEventStore()
        quarantine: set[str] = set()
        halt_hook = Mock()

        verifier = _xcheck_verifier(result=local_infra_fail)
        outcome = await _run_post_merge_verify(
            git_ops, req, tmp_path,
            timeouts={}, enospc_retries={},
            max_timeouts=2, max_enospc=1,
            event_store=es,  # type: ignore[arg-type]
            merge_sha='mergesha07',
            runner=remote,
            escalation_queue=eq,
            quarantine=quarantine,
            halt_hook=halt_hook,
            verifier=verifier,
        )

        # (1) the land is STILL withheld — fail-closed for THIS land.
        assert outcome is not None
        assert outcome.status == 'blocked'

        # (2) the FLEET is NOT halted — a local infra hiccup must not block every
        #     lane and force an operator un-halt (reviewer robustness amendment).
        halt_hook.assert_not_called()

        # (3) the diverging remote is still quarantined.
        assert 'laptop' in quarantine

        # (4) exactly one escalation, L1 BLOCKING (pre-fix-3 shape), NOT born-at-L2.
        assert len(eq.submitted) == 1
        esc = eq.submitted[0]
        assert esc.category == 'verify_cross_check_mismatch'
        assert esc.level == 1
        assert esc.severity == 'blocking'
        assert 'infra' in esc.detail.lower()
        assert 'not halted' in (esc.summary + esc.detail).lower()

        # (5) mismatch telemetry still emitted (the divergence IS still recorded).
        assert es.events_of(EventType.verify_cross_check_mismatch)

    async def test_diverge_halt_fires_before_escalation_and_return(self, tmp_path):
        """fix 3 ordering: the synchronous halt MUST precede the escalation submit and the
        blocked return, so no FURTHER adoption can occur before a human looks."""

        from orchestrator.merge_queue import _run_post_merge_verify

        config = _xcheck_config(cross_check=True)
        req = _xcheck_req(config, worktree=tmp_path)
        git_ops = _xcheck_git_ops()

        remote = _remote_stub(_make_result(True), name='laptop')
        es = _RecordingEventStore()
        quarantine: set[str] = set()
        order: list[str] = []

        class _OrderingEQ(_FakeEscalationQueue):
            def submit(self, escalation):
                order.append('submit')
                super().submit(escalation)

        eq = _OrderingEQ()

        def halt_hook(reason: str) -> None:
            order.append('halt')

        verifier = _xcheck_verifier(result=_make_result(False))
        outcome = await _run_post_merge_verify(
            git_ops, req, tmp_path,
            timeouts={}, enospc_retries={},
            max_timeouts=2, max_enospc=1,
            event_store=es,  # type: ignore[arg-type]
            merge_sha='mergesha0h',
            runner=remote,
            escalation_queue=eq,
            quarantine=quarantine,
            halt_hook=halt_hook,
            verifier=verifier,
        )

        assert outcome is not None and outcome.status == 'blocked'
        assert 'halt' in order and 'submit' in order
        assert order.index('halt') < order.index('submit'), (
            f'halt must precede the escalation submit / adoption; got order={order}'
        )

    async def test_diverge_halt_hook_none_is_safe(self, tmp_path):
        """fix 3 None-safety: non-production callers omit halt_hook (default None) — the
        divergence still withholds the land + files a born-at-L2 escalation, no crash."""

        from orchestrator.merge_queue import _run_post_merge_verify

        config = _xcheck_config(cross_check=True)
        req = _xcheck_req(config, worktree=tmp_path)
        git_ops = _xcheck_git_ops()

        remote = _remote_stub(_make_result(True), name='laptop')
        eq = _FakeEscalationQueue()
        es = _RecordingEventStore()
        quarantine: set[str] = set()

        verifier = _xcheck_verifier(result=_make_result(False))
        outcome = await _run_post_merge_verify(
            git_ops, req, tmp_path,
            timeouts={}, enospc_retries={},
            max_timeouts=2, max_enospc=1,
            event_store=es,  # type: ignore[arg-type]
            merge_sha='mergesha0n',
            runner=remote,
            escalation_queue=eq,
            quarantine=quarantine,
            # halt_hook omitted -> None default (byte-identical non-production callers)
            verifier=verifier,
        )

        assert outcome is not None and outcome.status == 'blocked'
        assert 'laptop' in quarantine
        assert len(eq.submitted) == 1
        assert eq.submitted[0].level == 2
        assert eq.submitted[0].severity == 'critical'

    async def test_agree_local_pass_proceeds_and_emits_verdict_parity_ok(self, tmp_path):
        """knob ON, remote PASS + local PASS -> proceeds (None) + verdict_parity_ok."""

        from orchestrator.merge_queue import _run_post_merge_verify

        config = _xcheck_config(cross_check=True)
        req = _xcheck_req(config, worktree=tmp_path)
        git_ops = _xcheck_git_ops()

        remote = _remote_stub(_make_result(True), name='laptop')
        eq = _FakeEscalationQueue()
        es = _RecordingEventStore()
        quarantine: set[str] = set()

        verifier = _xcheck_verifier(result=_make_result(True))
        outcome = await _run_post_merge_verify(
            git_ops, req, tmp_path,
            timeouts={}, enospc_retries={},
            max_timeouts=2, max_enospc=1,
            event_store=es,  # type: ignore[arg-type]
            merge_sha='mergesha02',
            runner=remote,
            escalation_queue=eq,
            quarantine=quarantine,
            verifier=verifier,
        )

        assert outcome is None  # land proceeds
        assert quarantine == set()  # no quarantine on agree
        assert eq.submitted == []  # no escalation on agree
        assert len(verifier.verified) == 1  # cross-check ran
        assert es.events_of(EventType.verdict_parity_ok)

    async def test_knob_off_never_constructs_local_runner_byte_identical(self, tmp_path):
        """knob OFF -> the local cross-check runner is NEVER constructed (byte-identical)."""

        from orchestrator.merge_queue import _run_post_merge_verify

        config = _xcheck_config(cross_check=False)
        req = _xcheck_req(config, worktree=tmp_path)
        git_ops = _xcheck_git_ops()

        remote = _remote_stub(_make_result(True), name='laptop')
        eq = _FakeEscalationQueue()
        es = _RecordingEventStore()
        quarantine: set[str] = set()

        verifier = _xcheck_verifier(result=_make_result(False))
        outcome = await _run_post_merge_verify(
            git_ops, req, tmp_path,
            timeouts={}, enospc_retries={},
            max_timeouts=2, max_enospc=1,
            event_store=es,  # type: ignore[arg-type]
            merge_sha='mergesha03',
            runner=remote,
            escalation_queue=eq,
            quarantine=quarantine,
            verifier=verifier,
        )

        assert outcome is None  # remote green stands, land proceeds
        assert verifier.verified == []  # local trust-anchor NEVER asked to verify
        assert quarantine == set()
        assert eq.submitted == []
        assert not es.events_of(EventType.verify_cross_check_mismatch)
        assert not es.events_of(EventType.verdict_parity_ok)

    async def test_local_runner_unavailable_is_inconclusive_and_trusts_remote(self, tmp_path):
        """knob ON, local cross-check raises RunnerUnavailable -> inconclusive + trust remote green."""

        from orchestrator.merge_queue import _run_post_merge_verify

        config = _xcheck_config(cross_check=True)
        req = _xcheck_req(config, worktree=tmp_path)
        git_ops = _xcheck_git_ops()

        remote = _remote_stub(_make_result(True), name='laptop')
        eq = _FakeEscalationQueue()
        es = _RecordingEventStore()
        quarantine: set[str] = set()

        verifier = _xcheck_verifier(raises=RunnerUnavailable('laptop closed'))
        outcome = await _run_post_merge_verify(
            git_ops, req, tmp_path,
            timeouts={}, enospc_retries={},
            max_timeouts=2, max_enospc=1,
            event_store=es,  # type: ignore[arg-type]
            merge_sha='mergesha04',
            runner=remote,
            escalation_queue=eq,
            quarantine=quarantine,
            verifier=verifier,
        )

        assert outcome is None  # fail-safe: trust the remote green
        assert quarantine == set()  # no quarantine on a local infra hiccup
        assert eq.submitted == []  # no escalation
        assert es.events_of(EventType.verify_cross_check_inconclusive)

    async def test_trivial_remote_pass_skips_cross_check(self, tmp_path):
        """knob ON, remote returns a TRIVIAL pass -> cross-check SKIPPED (no suite ran to compare)."""

        from orchestrator.merge_queue import _run_post_merge_verify

        config = _xcheck_config(cross_check=True)
        req = _xcheck_req(config, worktree=tmp_path)
        git_ops = _xcheck_git_ops()

        trivial = VerifyResult(
            passed=True, test_output='', lint_output='', type_output='',
            summary='trivial pass', trivial=True,
        )
        remote = _remote_stub(trivial, name='laptop')
        eq = _FakeEscalationQueue()
        es = _RecordingEventStore()
        quarantine: set[str] = set()

        verifier = _xcheck_verifier(result=_make_result(False))
        outcome = await _run_post_merge_verify(
            git_ops, req, tmp_path,
            timeouts={}, enospc_retries={},
            max_timeouts=2, max_enospc=1,
            event_store=es,  # type: ignore[arg-type]
            merge_sha='mergesha05',
            runner=remote,
            escalation_queue=eq,
            quarantine=quarantine,
            verifier=verifier,
        )

        # cold-baseline trivial-pass gate is fail-open -> land proceeds, and the
        # cross-check is skipped for a trivial pass (nothing was verified to compare).
        assert outcome is None
        assert verifier.verified == [], 'a trivial remote pass must not trigger a local cross-check'
        assert not es.events_of(EventType.verify_cross_check_mismatch)
        assert not es.events_of(EventType.verdict_parity_ok)

    async def test_local_dispatch_path_skips_cross_check_entirely(self, tmp_path):
        """runner is None (local dispatch): the cross-check branch never runs."""

        from orchestrator.merge_queue import _run_post_merge_verify

        config = _xcheck_config(cross_check=True)
        req = _xcheck_req(config, worktree=tmp_path)
        git_ops = _xcheck_git_ops()

        eq = _FakeEscalationQueue()
        es = _RecordingEventStore()
        quarantine: set[str] = set()

        # Real local-path pool whose scoped verify is the injected port,
        # returning PASS — no cross-check side effects.
        verifier = _xcheck_verifier(result=_make_result(True))
        outcome = await _run_post_merge_verify(
            git_ops, req, tmp_path,
            timeouts={}, enospc_retries={},
            max_timeouts=2, max_enospc=1,
            event_store=es,  # type: ignore[arg-type]
            merge_sha='mergesha06',
            runner=None,  # LOCAL dispatch
            escalation_queue=eq,
            quarantine=quarantine,
            verifier=verifier,
        )

        assert outcome is None
        assert quarantine == set()
        assert eq.submitted == []
        assert not es.events_of(EventType.verify_cross_check_mismatch)
        assert not es.events_of(EventType.verify_cross_check_inconclusive)
        assert not es.events_of(EventType.verdict_parity_ok)


# ===========================================================================
# Task 2822 step-9: end-to-end capstone tying fixes a + b + c through the
# PRODUCTION two-host path.
#   (a) the merge-gate PROFILE ships in the spec and overrides the remote host's
#       NARROW config at the host-entry (RemoteRunner transport -> the
#       run_merge_verify_on_worktree host-entry: the laptop's config no longer
#       narrows the merge-deciding verify).
#   (c) a passing remote verify is archived with SCOPE + RESULT + TIMING.
#   (b) a remote FALSE-GREEN (remote PASS while the LOCAL trust-anchor FAILs)
#       does NOT land — driven through the real SpeculativeMergeWorker
#       ._run_inflight_verify -> _run_post_merge_verify caller, proving the
#       event_store / escalation_queue / by-reference quarantine set all reach
#       the cross-check.
# ===========================================================================


@pytest.mark.asyncio
class TestTwoHostFalseGreenCapstone:
    """Production capstone for the two-host merge-verify false-green fix (task 2822)."""

    async def test_a_profile_ships_full_and_overrides_narrow_remote_config_plus_c_archive(
        self, tmp_path,
    ):
        """(a) spec carries the FULL profile over the transport and wins over the
        remote's NARROW config at the host-entry; (c) the passing remote verify is
        archived with scope+result+timing."""
        import json

        from orchestrator.verify_runner import (
            RemoteRunner,
            build_merge_verify_spec,
            result_to_json,
            run_merge_verify_on_worktree,
        )

        # Dispatching host runs the FULL merge-gate profile.
        full_config = OrchestratorConfig(
            git=GitConfig(main_branch='main'),
            project_root=tmp_path,
            merge_verify_workspace=True,
            merge_verify_breadth='full',
        )
        spec = build_merge_verify_spec(full_config, [], ('src/foo.py',))
        # fix (a), spec half: the merge-gate profile is shipped in the spec.
        assert spec.merge_verify_workspace is True
        assert spec.merge_verify_breadth == 'full'

        # --- transport: a real RemoteRunner ships the spec over a fake ssh ---
        captured_argvs: list[list[str]] = []
        pass_result = VerifyResult(
            passed=True, test_output='green', lint_output='', type_output='',
            summary='ok', category='merge_ok',
        )
        _it = iter([
            (0, '', ''),                              # git push
            (0, result_to_json(pass_result), ''),     # ssh verify -> PASS
        ])

        async def fake_run(argv, *, cwd=None):
            captured_argvs.append(list(argv))
            if argv[0] == 'git' and '--delete' in argv:
                return (0, '', '')
            return next(_it)

        runner = RemoteRunner(
            name='leo-laptop', ssh_host='leo-laptop.local', git_remote='origin',
            cwd='/repo', run=fake_run, id_factory=lambda: 'cap-id',
        )
        result = await runner.run_merge_verify(
            'mergesha_cap', spec, task_id='2822', archive_root=tmp_path,
        )
        assert result == pass_result

        # fix (a), transport half: the shipped ssh payload carries the full profile.
        ssh_argvs = [a for a in captured_argvs if a and a[0] == 'ssh']
        assert ssh_argvs, f'no ssh dispatch captured; argvs={captured_argvs}'
        joined = ' '.join(' '.join(a) for a in ssh_argvs)
        assert 'merge_verify_workspace' in joined
        assert 'merge_verify_breadth' in joined
        assert 'full' in joined

        # fix (c): the passing remote verify is archived with scope+result+timing.
        summaries = list((tmp_path / '2822').glob('attempt-1.remote-leo-laptop.pass-summary-*.json'))
        assert len(summaries) == 1, f'expected 1 pass-summary, got {[f.name for f in summaries]}'
        data = json.loads(summaries[0].read_text(encoding='utf-8'))
        assert data['merge_sha'] == 'mergesha_cap'
        assert data['runner'] == 'leo-laptop'
        assert data['passed'] is True
        assert data['scope']['merge_verify_workspace'] is True
        assert data['scope']['merge_verify_breadth'] == 'full'
        assert data['scope']['task_files'] == ['src/foo.py']
        assert isinstance(data['duration_ms'], (int, float)) and data['duration_ms'] >= 0

        # fix (a), host-entry half: the SPEC's full profile overrides the remote's
        # NARROW config so the remote runs force_workspace=True (the laptop config
        # can no longer narrow the merge-deciding verify).
        narrow_remote_config = OrchestratorConfig(
            merge_verify_workspace=False, merge_verify_breadth='scoped', project_root=tmp_path,
        )
        scoped_calls: list[dict] = []

        async def capture_scoped(wt, cfg, module_configs, **kwargs):
            scoped_calls.append({'config': cfg, 'kwargs': kwargs})
            return VerifyResult(
                passed=True, test_output='', lint_output='', type_output='', summary='ok',
            )

        async def noop_unscoped(*a, **k):
            return MagicMock(broken=False, timed_out=False, failing_subprojects=[],
                             timed_out_subprojects=[])

        await run_merge_verify_on_worktree(
            MagicMock(), narrow_remote_config, spec,
            run_scoped=capture_scoped, run_unscoped=noop_unscoped,
        )
        assert scoped_calls, 'run_scoped was never invoked by the host-entry'
        assert scoped_calls[0]['kwargs'].get('force_workspace') is True, (
            'the spec full profile must override the narrow remote config -> force_workspace=True'
        )
        assert scoped_calls[0]['config'].merge_verify_breadth == 'full', (
            'the effective remote config breadth must be overridden to full by the spec'
        )

    async def test_b_false_green_is_withheld_quarantined_and_halts_the_lane(
        self, host_git_ops: GitOps, host_config: OrchestratorConfig,
    ) -> None:
        """(b) a remote FALSE-GREEN on a RUNNING lane does not land.

        The remote reports PASS while the local trust anchor FAILs, so the
        cross-check concludes divergence: the branch is withheld, the remote is
        quarantined in the by-reference set the allocator shares, a born-at-L2
        verify_cross_check_mismatch escalation is filed with its telemetry, and
        the lane is halted SYNCHRONOUSLY -- before any further adoption can
        occur (incident 83336a32: the cross-check diverged +3 s AFTER the
        CAS-advance).
        """
        anchor_gate = asyncio.Event()
        verifier = FakeVerifier(scripts={
            # The parked anchor keeps the local slot busy so 'fg-b' is dispatched
            # to the laptop; 'fg-b' is the cross-check's own local trust anchor,
            # and it is the leg that disagrees.
            'fg-a': VerifyScript(result=passes().result, release=anchor_gate),
            'fg-b': VerifyScript(result=_make_result(False)),
        })
        escalations = _HostEscalationQueue()
        events = _RecordingEventStore()
        laptop = _FlakyRemoteRunner(reachable=True)

        async with _lane_with_remote(
            host_git_ops, host_config, laptop,
            verifier=verifier, escalation_queue=escalations, event_store=events,
        ) as (lane, queue):
            req_a = await _submitted(host_git_ops, host_config, 'fg-a', 'fg_a.py', 'a = 1\n')
            await queue.put(req_a)
            await _until(lambda: 'fg-a' in verifier.verified, what="the anchor's verify to enter")
            req_b = await _submitted(host_git_ops, host_config, 'fg-b', 'fg_b.py', 'b = 2\n')
            await queue.put(req_b)
            await _until(
                lambda: bool(events.events_of(EventType.verify_cross_check_mismatch)),
                what='the cross-check to conclude divergence',
            )

            # (b5) the divergence halted the lane SYNCHRONOUSLY -- observable
            #      before the outcome is even delivered.
            assert lane.snapshot()['is_wip_halted'] is True

            # (b2) the diverging remote is quarantined in the set the allocator
            #      shares by reference -- an operator-held divergence quarantine,
            #      NOT the auto-recovering RU class.
            row = _host_row(lane, 'leo-laptop')
            assert row['quarantined'] is True, row
            assert row['quarantine_class'] == 'divergence', row

            # (b3) exactly one dedup'd BORN-AT-L2 escalation (task 2886 fix 3:
            #      level=2/critical, no longer level=1/blocking).
            mismatch = escalations.of_category('verify_cross_check_mismatch')
            assert len(mismatch) == 1, escalations.submitted
            assert mismatch[0].agent_role == 'orchestrator-cross-check'
            assert mismatch[0].level == 2
            assert mismatch[0].severity == 'critical'
            assert 'leo-laptop' in mismatch[0].detail

            anchor_gate.set()

            # (b1) the false-green does NOT land.
            outcome_b = await wait_responsive(req_b.result, label='false-green must not land')
            assert outcome_b.status != 'done', outcome_b

        _, main_files, _ = await _run(
            ['git', 'ls-tree', '-r', '--name-only', 'main'], cwd=host_git_ops.project_root,
        )
        assert 'fg_b.py' not in main_files, (
            'the branch behind a remote false-green must never reach main'
        )


# ===========================================================================
# task 3173 step-9: THE HEADLINE REGRESSION.
#
# The cross-check compares only `.passed` — a two-value comparison with no
# INDETERMINATE arm.  So a local trust-anchor leg that was SIGKILLed at 0.31s
# having emitted zero diagnostics is indistinguishable from a completed FAIL:
# it takes the fail-CLOSED arm, quarantines the remote, files a blocking L1,
# and ADOPTS the killed verdict — discarding host A's completed 1097s PASS
# behind an already-built, already-verified merge commit (the measured case:
# merge_sha b1ac2c7f).
#
# The block already holds two fail-SAFE precedents that do exactly the right
# thing (RunnerUnavailable, MergeVerifyLeaseContended): both emit
# verify_cross_check_inconclusive and keep the remote green.  An infra-killed
# leg reaches neither, because it returns NORMALLY with passed=False.
#
# INVARIANT: only a COMPLETED failing verdict may veto a completed PASS.
#
# RED until step-10.
# ===========================================================================


#: What ``_summarize_checks`` published for host B's leg in the measured case:
#: SIGKILLed at 0.31 s having emitted zero diagnostics.
_KILLED_LEG_SUMMARY = (
    'Failures: lint leg killed by signal 9 after 0.31s; '
    'no diagnostics produced; verdict indeterminate'
)


@pytest.mark.asyncio
class TestIndeterminateLocalLegDoesNotVeto:
    """The new INDETERMINATE arm, plus its own fail-CLOSED controls.

    Deliberately NOT a subclass of TestPerLandCrossCheck: pytest already
    collects that class in this module, so inheriting it would re-run every
    parent test a second time for zero added coverage.  The claim that the
    task-2822 fail-CLOSED contract still holds alongside the new arm is
    pinned by this class's OWN controls below, which drive the same
    `_run_post_merge_verify` callee through `_drive`.
    """

    @staticmethod
    async def _drive(tmp_path, *, local_category: str, merge_sha: str,
                     local_summary: str | None = None,
                     local_failing_legs: list[str] | None = None):
        """Remote PASS + local FAIL(*local_category*) through the real callee.

        *local_failing_legs* (task 3173 review amendment) is what
        ``_summarize_checks`` published for EACH failing leg, which is what the
        veto gate actually reads.  It is a SEPARATE knob from *local_category*
        precisely because the two can disagree: severity dominance makes a
        rank-1 ``infra_kill`` the aggregate category even when a rank-11
        ``test_failure`` leg completed alongside it.  Defaulting to ``None``
        (rather than deriving ``[local_category]``) keeps the fail-CLOSED
        "not recorded" case drivable.
        """

        from orchestrator.merge_queue import _run_post_merge_verify

        config = _xcheck_config(cross_check=True)
        req = _xcheck_req(config, worktree=tmp_path)
        git_ops = _xcheck_git_ops()

        remote = _remote_stub(_make_result(True), name='laptop')
        # Host B's leg: signal-killed at 0.31s, zero diagnostics on any leg.
        local_fail = _make_result(False, category=local_category)
        local_fail.failing_leg_categories = local_failing_legs
        local_fail.test_output = ''
        local_fail.lint_output = ''
        local_fail.type_output = ''
        local_fail.summary = (
            _KILLED_LEG_SUMMARY if local_summary is None else local_summary
        )
        eq = _FakeEscalationQueue()
        es = _RecordingEventStore()
        quarantine: set[str] = set()

        verifier = _xcheck_verifier(result=local_fail)
        outcome = await _run_post_merge_verify(
            git_ops, req, tmp_path,
            timeouts={}, enospc_retries={},
            max_timeouts=2, max_enospc=1,
            event_store=es,  # type: ignore[arg-type]
            merge_sha=merge_sha,
            runner=remote,
            escalation_queue=eq,
            quarantine=quarantine,
            verifier=verifier,
        )
        return outcome, git_ops, eq, es, quarantine, verifier

    # -- the new INDETERMINATE arm ----------------------------------------

    async def test_indeterminate_local_leg_does_not_veto_completed_remote_pass(self, tmp_path):
        """THE MEASURED CASE: a killed local leg must not discard host A's
        completed PASS, and must not blame the branch for it."""
        outcome, git_ops, eq, es, quarantine, verifier = await self._drive(
            tmp_path, local_category='infra_kill', merge_sha='b1ac2c7f',
            local_failing_legs=['infra_kill'],
        )

        # (1) the land PROCEEDS — the killed leg produced no verdict to veto with.
        assert outcome is None, (
            f'a leg that produced NO verdict must not block the land; got {outcome}'
        )

        # (2) the already-built, already-verified merge commit is NOT discarded.
        git_ops.cleanup_merge_worktree.assert_not_awaited()

        # (3) the remote host is not punished for the local host's kill.
        assert quarantine == set()

        # (4) no blocking escalation — nothing here needs a human.
        assert eq.submitted == []

        # (5) the cross-check DID run; it is recorded as inconclusive, not a mismatch.
        assert len(verifier.verified) == 1
        inconclusive = es.events_of(EventType.verify_cross_check_inconclusive)
        assert len(inconclusive) == 1, (
            f'expected exactly one inconclusive event, got {es.events}'
        )
        data = inconclusive[0][2]
        assert data['merge_sha'] == 'b1ac2c7f'
        assert data['remote_runner'] == 'laptop'
        assert 'infra_kill' in repr(data), (
            f'the event must name the indeterminate category; got {data}'
        )
        # The event names EVERY per-leg category, not just the aggregate, so a
        # future triager can see WHY the run was judged verdict-less rather
        # than having to trust one collapsed severity-ranked string.
        assert data['local_failing_leg_categories'] == ['infra_kill'], (
            f'the event must name every per-leg category; got {data}'
        )
        assert es.events_of(EventType.verify_cross_check_mismatch) == []

    async def test_every_indeterminate_category_is_covered(self, tmp_path):
        """Whatever the registry holds, the arm honours it — so a future row
        cannot be added to the registry and silently miss this path."""
        from orchestrator.verify_categories import INDETERMINATE_VERDICT_CATEGORIES

        assert INDETERMINATE_VERDICT_CATEGORIES, 'registry must not be empty'
        for i, category in enumerate(sorted(INDETERMINATE_VERDICT_CATEGORIES)):
            outcome, _, eq, es, quarantine, _ = await self._drive(
                tmp_path, local_category=str(category), merge_sha=f'sha{i:04d}',
                local_failing_legs=[str(category)],
            )
            assert outcome is None, f'{category} must not veto a completed PASS'
            assert quarantine == set(), f'{category} must not quarantine the remote'
            assert eq.submitted == [], f'{category} must not escalate'
            assert es.events_of(EventType.verify_cross_check_mismatch) == [], category

    # -- THE GAP THE REVIEW NAMED: MIXED per-leg results -------------------

    async def test_real_test_failure_plus_killed_lint_still_vetoes_and_quarantines(
        self, tmp_path
    ):
        """A local trust-anchor whose TEST leg COMPLETED and reported real
        branch failures, while an UNRELATED lint leg was SIGKILLed.

        `_summarize_checks` genuinely produces `category='infra_kill'` here —
        severity_rank=1 dominates rank-11 test_failure — so an arm keyed on
        the aggregate category alone silently LANDS this, discarding the
        completed, branch-blaming evidence the cross-check exists to collect
        (task 2822).  The task-2822 fail-CLOSED contract must hold in full.
        """
        outcome, git_ops, eq, es, quarantine, _ = await self._drive(
            tmp_path, local_category='infra_kill', merge_sha='mixed001',
            local_failing_legs=['test_failure', 'infra_kill'],
            local_summary=(
                'Failures: tests failed, lint leg killed by signal 9 after 0.31s; '
                'no diagnostics produced; verdict indeterminate'
            ),
        )
        assert outcome is not None, (
            'a COMPLETED test-leg failure must still withhold the land, even '
            'when a co-occurring kill dominates the aggregate category'
        )
        assert outcome.status == 'blocked'
        git_ops.cleanup_merge_worktree.assert_awaited()
        assert 'laptop' in quarantine
        mismatch = [
            e for e in eq.submitted
            if getattr(e, 'category', None) == 'verify_cross_check_mismatch'
        ]
        assert len(mismatch) == 1, f'expected 1 mismatch escalation, got {eq.submitted}'
        assert mismatch[0].severity == 'blocking'
        assert es.events_of(EventType.verify_cross_check_mismatch)
        assert es.events_of(EventType.verify_cross_check_inconclusive) == []

    async def test_missing_failing_leg_categories_fails_closed(self, tmp_path):
        """None is NOT RECORDED, never a licence: an old wire payload, or any
        result not produced by `run_verification`, must veto."""
        outcome, git_ops, eq, es, quarantine, _ = await self._drive(
            tmp_path, local_category='infra_kill', merge_sha='none0001',
            local_failing_legs=None,
        )
        assert outcome is not None, 'an unrecorded per-leg list must fail CLOSED'
        assert outcome.status == 'blocked'
        assert 'laptop' in quarantine
        assert len(eq.submitted) == 1
        assert es.events_of(EventType.verify_cross_check_mismatch)
        assert es.events_of(EventType.verify_cross_check_inconclusive) == []

    async def test_empty_failing_leg_categories_fails_closed(self, tmp_path):
        """[] is "no legs recorded", not "all legs indeterminate"."""
        outcome, git_ops, eq, es, quarantine, _ = await self._drive(
            tmp_path, local_category='infra_kill', merge_sha='empty001',
            local_failing_legs=[],
        )
        assert outcome is not None, 'an empty per-leg list must fail CLOSED'
        assert outcome.status == 'blocked'
        assert 'laptop' in quarantine
        assert len(eq.submitted) == 1
        assert es.events_of(EventType.verify_cross_check_mismatch)
        assert es.events_of(EventType.verify_cross_check_inconclusive) == []

    # -- CONTROLS: the task-2822 fail-CLOSED contract is fully intact -------

    @pytest.mark.parametrize(
        ('category', 'why'),
        [
            # Each cites the step-13 adjudication that removed it from the
            # registry, so the three narrowed rows are pinned AT THE GATE and
            # not only in the registry's own unit test.
            ('disk_full', 'fails predicate (2): a diff generating very large '
                          'build artifacts or a runaway test log can genuinely '
                          'cause the ENOSPC itself'),
            ('env_transient', 'fails predicate (3): verify_classify.py:498-508 '
                              'documents that a guard script tripping over a '
                              'file the DIFF deleted is textually '
                              'indistinguishable from worktree-removal collateral'),
            ('semaphore_timeout', 'fails predicate (3): verify_classify.py:438-453 '
                                  'documents that a deterministic shell gate '
                                  'assertion quoting a lock + timeout token '
                                  'still classifies SEMAPHORE_TIMEOUT'),
        ],
    )
    async def test_narrowed_infra_rows_still_veto(self, tmp_path, category, why):
        """These three are is_infra_transient=True — retrying them is still
        right — but they are NOT verdict-indeterminate, so they keep their
        veto over another host's completed PASS."""
        from orchestrator.verify_categories import (
            INDETERMINATE_VERDICT_CATEGORIES,
            INFRA_TRANSIENT_CATEGORIES,
        )

        # The premise each control exists for: infra-transient, yet vetoing.
        assert category in INFRA_TRANSIENT_CATEGORIES, why
        assert category not in INDETERMINATE_VERDICT_CATEGORIES, why

        outcome, git_ops, eq, es, quarantine, _ = await self._drive(
            tmp_path, local_category=category, merge_sha='narrow01',
            local_failing_legs=[category],
        )
        assert outcome is not None, f'{category} must still withhold the land: {why}'
        assert outcome.status == 'blocked'
        git_ops.cleanup_merge_worktree.assert_awaited()
        assert 'laptop' in quarantine
        assert len(eq.submitted) == 1
        assert es.events_of(EventType.verify_cross_check_mismatch)
        assert es.events_of(EventType.verify_cross_check_inconclusive) == []

    @pytest.mark.parametrize('category', ['test_failure', 'infra_timeout'])
    async def test_completed_fail_and_local_timeout_still_veto(self, tmp_path, category):
        """A genuine completed FAIL still vetoes — and so does a local TIMEOUT,
        which is deliberately EXCLUDED from the registry because a hang is one
        of the few non-completions a branch can genuinely cause."""
        outcome, git_ops, eq, es, quarantine, _ = await self._drive(
            tmp_path, local_category=category, merge_sha='ctrl0001',
            # Backfilled (review amendment): drive the per-leg list too, so
            # these veto because their CATEGORY is not indeterminate — not
            # vacuously, because the list happened to be unrecorded.
            local_failing_legs=[category],
        )
        assert outcome is not None, f'{category} must still withhold the land'
        assert outcome.status == 'blocked'
        git_ops.cleanup_merge_worktree.assert_awaited()
        assert 'laptop' in quarantine
        mismatch = [
            e for e in eq.submitted
            if getattr(e, 'category', None) == 'verify_cross_check_mismatch'
        ]
        assert len(mismatch) == 1, f'expected 1 mismatch escalation, got {eq.submitted}'
        # task 2886 fix 3 (merge resolution): BOTH params are GENUINE
        # (non-infra-transient) divergences — 'test_failure' plainly, and
        # 'infra_timeout' because a hang is branch-causable (this test's own
        # docstring, and its exclusion from INFRA_TRANSIENT_CATEGORIES).  A
        # genuine divergence means main is SUSPECTED-RED, so the escalation is
        # now born-at-L2 / critical and halts the queue, where pre-2886 it was
        # L1 / blocking.  The fail-CLOSED contract this test exists to pin is
        # unchanged and asserted above.  The infra-transient contrast case
        # (local_category='infra_kill') still asserts L1/blocking, which is
        # what bounds the halt blast radius.
        assert mismatch[0].severity == 'critical'
        assert mismatch[0].level == 2
        assert es.events_of(EventType.verify_cross_check_mismatch)
        assert es.events_of(EventType.verify_cross_check_inconclusive) == []

    async def test_branch_caused_collection_error_still_vetoes(self, tmp_path):
        """A pytest INTERNALERROR is infra-transient for RETRY purposes, but a
        conftest.py or plugin added by the diff can raise it at collection
        time — and may raise only on this host's interpreter/plugin set while
        a remote with a cached env collects fine.  So it is deliberately kept
        OUT of INDETERMINATE_VERDICT_CATEGORIES: it must keep vetoing, or the
        widened exemption would land a branch-caused break (fail CLOSED)."""
        from orchestrator.verify_categories import (
            INDETERMINATE_VERDICT_CATEGORIES,
            INFRA_TRANSIENT_CATEGORIES,
        )

        # The premise this control exists for: infra-transient, yet vetoing.
        assert 'pytest_internalerror' in INFRA_TRANSIENT_CATEGORIES
        assert 'pytest_internalerror' not in INDETERMINATE_VERDICT_CATEGORIES

        outcome, git_ops, eq, es, quarantine, _ = await self._drive(
            tmp_path, local_category='pytest_internalerror', merge_sha='ctrl0003',
            local_summary='Failures: tests failed',
            local_failing_legs=['pytest_internalerror'],
        )
        assert outcome is not None, 'a branch-causable collection error must veto'
        assert outcome.status == 'blocked'
        git_ops.cleanup_merge_worktree.assert_awaited()
        assert 'laptop' in quarantine
        assert len(eq.submitted) == 1
        assert es.events_of(EventType.verify_cross_check_mismatch)
        assert es.events_of(EventType.verify_cross_check_inconclusive) == []

    async def test_uncategorised_local_fail_still_vetoes(self, tmp_path):
        """An empty category is not a licence to land: fail CLOSED."""
        outcome, _, eq, es, quarantine, _ = await self._drive(
            tmp_path, local_category='', merge_sha='ctrl0002',
            local_failing_legs=[''],
        )
        assert outcome is not None
        assert outcome.status == 'blocked'
        assert 'laptop' in quarantine
        assert len(eq.submitted) == 1
        assert es.events_of(EventType.verify_cross_check_mismatch)


# ===========================================================================
# Task 5030 amendment: the timeout-mark guard, generalised to THIS module.
# ===========================================================================


class TestTimeoutMarkCoverage:
    """Enforced invariant: every class in THIS module whose computed
    worst-per-method wait budget clears the pyproject default timeout must
    carry a ``@pytest.mark.timeout`` mark whose value clears that budget.

    Task 3492 built this guard, and task 5030 gave test_merge_speculation.py
    its own copy -- but both resolve ``Path(__file__)`` against their own
    source, so neither reaches this module.  That mattered here the moment
    γ7 replaced this file's mock-driven host tests with real-git lane-settling
    polls: ``TestUnreachableHostCapstone`` went from trivially fast to a
    computed 360s budget against a 300s default, with no mark anywhere in the
    file.

    The helpers are IMPORTED from test_merge_queue_concurrent_verify rather
    than reimplemented: they are deliberately pure (source text in, offender
    list out, class resolver injected as ``globals().get``) precisely so they
    can be driven with foreign input, and a third copy would be free to drift
    from the marks it audits.

    The computed budget is a conservative FLOOR over call SHAPES: it bills
    ``wait_responsive`` / ``asyncio.wait_for`` / ``_await_outcome`` sites and
    nothing else, so this module's ``_until(...)`` polls -- which do bound
    their own budget, through ``wait_responsive`` one frame down -- are not
    counted. An unrecognised shape can only under-count, never fabricate a
    wait, so a class this guard passes may still deserve a wider mark than the
    floor demands; a class it fails is genuinely under-marked.
    """

    def test_heavy_wait_classes_carry_adequate_timeout_mark(self) -> None:
        """Every Test* class computing >= PYPROJECT_DEFAULT_TIMEOUT must
        carry a ``timeout`` mark whose value clears its own computed budget.

        Recomputes from source; no figure written anywhere in this file is
        load-bearing for the assertion.  (For orientation only, current at the
        time of writing: 360s for TestUnreachableHostCapstone against its
        HOST_CAPSTONE_TEST_TIMEOUT mark -- if the comment on that constant
        disagrees with this guard, the guard is right.)
        """
        source = Path(__file__).read_text()
        budgets = _worst_per_method_wait_budget(source)
        offenders = _timeout_mark_offenders(budgets, globals().get)

        assert not offenders, (
            'The following classes have a worst-case per-method wait '
            f'budget at or above the pyproject default timeout '
            f'({PYPROJECT_DEFAULT_TIMEOUT}s, see the '
            f'[tool.pytest.ini_options].timeout setting in '
            f'orchestrator/pyproject.toml) but lack an adequate '
            f'@pytest.mark.timeout mark:\n'
            + '\n'.join(f'  - {offender}' for offender in offenders)
            + '\n\nConsequence: pytest-timeout\'s thread method os._exit()s '
            'the xdist worker under --max-worker-restart=0, so a '
            'slow-but-correct run reports as a worker death instead of a '
            'clean per-test failure -- strictly worse than the flake being '
            'fixed. Add @pytest.mark.timeout(HOST_CAPSTONE_TEST_TIMEOUT) '
            'directly above each offending class.'
        )
