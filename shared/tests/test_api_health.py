"""Tests for ApiHealthGate — the C4 provider-degradation circuit breaker.

Contract: plans/server-side-api-error-handling-prd.md C4 (task 3325 / mu).
"""

from __future__ import annotations

import asyncio
import dataclasses
import json
import logging
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from shared.api_health import ApiHealthGate, Closed, GateState, GateStats, Open, Probing
from shared.cost_store import CostStore
from shared.invocation_outcome import (
    OK,
    AuthFailed,
    CapHit,
    CliLocalError,
    Failure,
    ModelNotFound,
    NearCap,
    ServerError,
    ZeroOutputWedge,
)

_WALL = datetime(2026, 7, 31, 12, 0, 0, tzinfo=UTC)


def _stats(
    *,
    failures: int = 8,
    completions: int = 8,
    distinct_tasks: int = 3,
    failure_rate: float = 1.0,
    window_secs: float = 600.0,
) -> GateStats:
    return GateStats(
        failures=failures,
        completions=completions,
        distinct_tasks=distinct_tasks,
        failure_rate=failure_rate,
        window_secs=window_secs,
    )


class TestGateSurface:
    """step-5: the constructed gate exposes PRD C4's defaults and starts Closed."""

    def test_defaults_match_prd_c4_config_block(self) -> None:
        """Defaults mirror `api_health.*` so config maps 1:1 with no translation layer."""
        gate = ApiHealthGate()
        assert gate.window_secs == 600.0
        assert gate.min_failures == 8
        assert gate.min_distinct_tasks == 3
        assert gate.failure_rate_threshold == 0.5
        assert gate.close_after_successes == 2

    def test_fresh_gate_is_closed_and_not_degraded(self) -> None:
        gate = ApiHealthGate()
        assert isinstance(gate.state(), Closed)
        assert gate.is_degraded is False

    @pytest.mark.parametrize('variant', [Closed, Open, Probing])
    def test_variants_are_gate_state_subclasses(self, variant: type) -> None:
        """isinstance(state, GateState) is the discriminator consumers hold onto."""
        assert issubclass(variant, GateState)

    def test_states_are_frozen(self) -> None:
        """state() hands out an immutable snapshot — a consumer cannot mutate gate state."""
        for state in (
            Closed(),
            Open(since=_WALL, stats=_stats()),
            Probing(since=_WALL, stats=_stats(), consecutive_successes=1),
        ):
            with pytest.raises(dataclasses.FrozenInstanceError):
                state.wat = 1  # type: ignore[attr-defined]

    def test_gate_stats_is_frozen(self) -> None:
        stats = _stats()
        with pytest.raises(dataclasses.FrozenInstanceError):
            stats.failures = 99  # type: ignore[misc]


class TestConstructorValidation:
    """step-7: threshold misconfiguration fails loud at construction.

    A gate built with `min_failures=0` or `failure_rate_threshold=0` would be
    permanently tripped and would throttle the entire fleet with no signal —
    the exact silent degradation the loud-over-silent norm forbids.
    """

    @pytest.mark.parametrize(
        ('kwargs', 'param'),
        [
            ({'window_secs': 0.0}, 'window_secs'),
            ({'window_secs': -1.0}, 'window_secs'),
            ({'min_failures': 0}, 'min_failures'),
            ({'min_failures': -1}, 'min_failures'),
            ({'min_distinct_tasks': 0}, 'min_distinct_tasks'),
            ({'min_distinct_tasks': -1}, 'min_distinct_tasks'),
            ({'failure_rate_threshold': 0.0}, 'failure_rate_threshold'),
            ({'failure_rate_threshold': -0.1}, 'failure_rate_threshold'),
            ({'failure_rate_threshold': 1.01}, 'failure_rate_threshold'),
            ({'close_after_successes': 0}, 'close_after_successes'),
            ({'close_after_successes': -1}, 'close_after_successes'),
        ],
    )
    def test_rejects_out_of_range_threshold(self, kwargs: dict, param: str) -> None:
        """The message names the offending parameter — an operator must not have to guess."""
        with pytest.raises(ValueError, match=param):
            ApiHealthGate(**kwargs)

    def test_error_message_reports_received_value(self) -> None:
        with pytest.raises(ValueError, match='-1'):
            ApiHealthGate(min_failures=-1)

    @pytest.mark.parametrize(
        'kwargs',
        [
            {'failure_rate_threshold': 1.0},
            {'min_failures': 1},
            {'min_distinct_tasks': 1},
            {'close_after_successes': 1},
            {'window_secs': 0.001},
        ],
    )
    def test_boundary_valid_values_construct(self, kwargs: dict) -> None:
        """A 100%-failure-rate or single-failure gate is strict, not invalid."""
        gate = ApiHealthGate(**kwargs)
        assert isinstance(gate.state(), Closed)


# ---------------------------------------------------------------------------
# Fake clocks — window/hysteresis math is driven deterministically, no sleeps.
# ---------------------------------------------------------------------------


class _FakeMonotonic:
    """A monotonic clock the test advances by hand."""

    def __init__(self, start: float = 1000.0) -> None:
        self.now = start

    def __call__(self) -> float:
        return self.now

    def advance(self, secs: float) -> None:
        self.now += secs


class _FakeWall:
    """A wall clock the test advances by hand, independent of the monotonic one."""

    def __init__(self, start: datetime = _WALL) -> None:
        self.now = start

    def __call__(self) -> datetime:
        return self.now

    def advance(self, secs: float) -> None:
        self.now += timedelta(seconds=secs)


def _gate(**kwargs) -> tuple[ApiHealthGate, _FakeMonotonic, _FakeWall]:
    """Build a gate on injected clocks, returning both so tests can advance time."""
    mono, wall = _FakeMonotonic(), _FakeWall()
    gate = ApiHealthGate(monotonic_clock=mono, wall_clock=wall, **kwargs)
    return gate, mono, wall


async def _report_failures(
    gate: ApiHealthGate,
    n: int,
    *,
    task_ids: list[str | None] | None = None,
    status: int = 529,
) -> GateState:
    """Report ``n`` ServerErrors, cycling through ``task_ids`` (default t1/t2/t3)."""
    ids: list[str | None] = ['t1', 't2', 't3'] if task_ids is None else task_ids
    state: GateState = gate.state()
    for i in range(n):
        state = await gate.report(
            ServerError(status=status),
            task_id=ids[i % len(ids)],
            account='max-d',
            role='agent',
        )
    return state


async def _report_ok(gate: ApiHealthGate, n: int = 1, *, task_id: str | None = 't1') -> GateState:
    state: GateState = gate.state()
    for _ in range(n):
        state = await gate.report(OK(), task_id=task_id, account='max-d', role='agent')
    return state


class TestTrip:
    """step-9: Closed -> Open when count, breadth and rate are all breached."""

    async def test_trips_on_the_nth_failure_not_before(self) -> None:
        gate, _mono, _wall = _gate()
        for i in range(7):
            state = await _report_failures(gate, 1, task_ids=[f't{i % 3 + 1}'])
            assert isinstance(state, Closed), f'tripped early at failure {i + 1}'
        state = await _report_failures(gate, 1, task_ids=['t2'])
        assert isinstance(state, Open)

    async def test_report_returns_the_post_report_state(self) -> None:
        """The return value IS the new state — xi needs no second state() call."""
        gate, _mono, _wall = _gate()
        returned = await _report_failures(gate, 8)
        assert isinstance(returned, Open)
        assert returned is gate.state()

    async def test_open_since_is_the_injected_wall_clock(self) -> None:
        gate, _mono, wall = _gate()
        wall.advance(3600)
        state = await _report_failures(gate, 8)
        assert isinstance(state, Open)
        assert state.since == _WALL + timedelta(seconds=3600)

    async def test_open_carries_the_trip_evidence(self) -> None:
        """stats is the narrative omicron's escalation and the dashboard render."""
        gate, _mono, _wall = _gate()
        state = await _report_failures(gate, 8)
        assert isinstance(state, Open)
        assert state.stats == GateStats(
            failures=8,
            completions=8,
            distinct_tasks=3,
            failure_rate=1.0,
            window_secs=600.0,
        )

    async def test_is_degraded_true_once_open(self) -> None:
        gate, _mono, _wall = _gate()
        await _report_failures(gate, 8)
        assert gate.is_degraded is True


class TestPartialDegradationStaysClosed:
    """step-11: decision 10 — throttle, not halt; never strand the healthy fraction.

    Each case breaches two of the three trip conditions and misses the third.
    The gate must stay Closed: a partial degradation is not a provider outage,
    and tripping on one would halt a fleet that is still mostly succeeding.
    """

    async def test_count_below_min_failures_does_not_trip(self) -> None:
        """7 failures across 3 tasks at 100% — breadth and rate met, count short."""
        gate, _mono, _wall = _gate()
        state = await _report_failures(gate, 7)
        assert isinstance(state, Closed)
        assert gate.is_degraded is False

    async def test_breadth_below_min_distinct_tasks_does_not_trip(self) -> None:
        """8 failures all on t1 — one pathological task retrying is not an outage."""
        gate, _mono, _wall = _gate()
        state = await _report_failures(gate, 8, task_ids=['t1'])
        assert isinstance(state, Closed)
        assert gate.is_degraded is False

    async def test_rate_below_threshold_does_not_trip(self) -> None:
        """8 failures / 20 completions = 40% — the healthy 60% must not be stranded.

        Count (8) and breadth (3) are both met; only the rate is short.  The
        12 successes land first so the running rate never crosses 0.5 at any
        intermediate report — see
        :meth:`test_transiently_crossing_the_rate_does_trip` for why that
        ordering is load-bearing rather than a rigged setup.
        """
        gate, _mono, _wall = _gate()
        await _report_ok(gate, 12)
        state = await _report_failures(gate, 8)
        assert isinstance(state, Closed)
        assert gate.is_degraded is False
        assert state is gate.state()

    async def test_rate_case_trips_once_the_rate_crosses(self) -> None:
        """Positive control for the case above: same window, more failures, does trip.

        Proves the 40% case stayed Closed because of the RATE specifically, not
        because the gate is inert or the harness never reaches a trip.
        """
        gate, _mono, _wall = _gate()
        await _report_ok(gate, 12)
        await _report_failures(gate, 8)
        assert isinstance(gate.state(), Closed)
        # 8F/20 = 40%.  (8+k)/(20+k) >= 0.5 first holds at k=4 -> 12/24 = 50%.
        state = await _report_failures(gate, 3)
        assert isinstance(state, Closed), '11/23 = 48% is still below threshold'
        state = await _report_failures(gate, 1)
        assert isinstance(state, Open)
        assert state.stats == GateStats(
            failures=12,
            completions=24,
            distinct_tasks=3,
            failure_rate=0.5,
            window_secs=600.0,
        )

    async def test_transiently_crossing_the_rate_does_trip(self) -> None:
        """The predicate is evaluated on EVERY report, not on a settled window.

        The same 8 failures and 12 successes, interleaved failure-first, put
        the window at 8/15 = 53% before the trailing successes arrive — and
        the gate trips there.  That is correct: at that instant the provider
        really had failed 8 of the last 15 calls across 3 tasks.  Pinning it
        stops a later refactor from "fixing" the ordering sensitivity by
        deferring evaluation, which would delay every real trip.
        """
        gate, _mono, _wall = _gate()
        for i in range(7):
            state = await _report_failures(gate, 1, task_ids=[f't{i % 3 + 1}'])
            assert isinstance(state, Closed), f'tripped early at failure {i + 1}'
            await _report_ok(gate, 1)
        # 7F/14 = 50%, but count is short.  The 8th failure meets all three.
        state = await _report_failures(gate, 1, task_ids=['t2'])
        assert isinstance(state, Open)
        assert state.stats.failures == 8
        assert state.stats.completions == 15
        assert state.stats.failure_rate == pytest.approx(8 / 15)

    async def test_none_task_ids_cannot_attest_breadth(self) -> None:
        """8 task-less failures — no breadth attestable, so no trip."""
        gate, _mono, _wall = _gate()
        state = await _report_failures(gate, 8, task_ids=[None])
        assert isinstance(state, Closed)
        assert gate.is_degraded is False

    async def test_none_task_id_failures_still_count_toward_the_numerator(self) -> None:
        """They cannot attest breadth, but dropping them would under-report an outage.

        5 task-less failures + 3 task-bearing ones across 3 distinct tasks
        would be only 3 failures if Nones were discarded — short of
        min_failures=8, so no trip.  It trips because all 8 are counted.
        """
        gate, _mono, _wall = _gate()
        await _report_failures(gate, 5, task_ids=[None])
        state = await _report_failures(gate, 3, task_ids=['t1', 't2', 't3'])
        assert isinstance(state, Open)
        assert state.stats.failures == 8
        assert state.stats.distinct_tasks == 3


class TestWindowEviction:
    """step-13: the window is aged out by monotonic timestamp, not reset wholesale."""

    async def test_stale_failures_do_not_accumulate_into_a_trip(self) -> None:
        """7 failures, then a gap longer than the window, then 1 more.

        The 8th failure is a lone failure in a fresh window, not the 8th of a
        burst — an outage an hour ago must not add itself to one failure now.
        """
        gate, mono, _wall = _gate()
        await _report_failures(gate, 7)
        mono.advance(601.0)
        state = await _report_failures(gate, 1)
        assert isinstance(state, Closed)
        assert gate.is_degraded is False

    async def test_eviction_is_by_age_not_a_blanket_reset(self) -> None:
        """4 failures, half a window later 4 more — all 8 are still in window, so it trips.

        Guards against "any gap clears everything", which would make a slow
        provider bleed forever without ever tripping.
        """
        gate, mono, _wall = _gate()
        await _report_failures(gate, 4, task_ids=['t1', 't2'])
        mono.advance(300.0)
        state = await _report_failures(gate, 4, task_ids=['t3', 't4'])
        assert isinstance(state, Open)
        assert state.stats.failures == 8

    async def test_reported_stats_count_only_in_window_samples(self) -> None:
        """The evidence on the Open state must describe the window, not all history."""
        gate, mono, _wall = _gate()
        await _report_ok(gate, 5)
        await _report_failures(gate, 3)
        mono.advance(601.0)
        state = await _report_failures(gate, 8)
        assert isinstance(state, Open)
        assert state.stats == GateStats(
            failures=8,
            completions=8,
            distinct_tasks=3,
            failure_rate=1.0,
            window_secs=600.0,
        )

    async def test_sample_exactly_at_the_window_edge_is_retained(self) -> None:
        """Eviction is strict (`< cutoff`), so a sample exactly window_secs old still counts.

        Pinned because an off-by-one here silently shortens every window.
        """
        gate, mono, _wall = _gate()
        await _report_failures(gate, 7)
        mono.advance(600.0)
        state = await _report_failures(gate, 1)
        assert isinstance(state, Open)
        assert state.stats.failures == 8


class TestProbeClose:
    """step-15: Open --OK--> Probing --OK--> Closed, with hysteresis."""

    async def test_first_success_moves_open_to_probing(self) -> None:
        gate, _mono, _wall = _gate()
        opened = await _report_failures(gate, 8)
        assert isinstance(opened, Open)
        state = await _report_ok(gate, 1)
        assert isinstance(state, Probing)
        assert state.consecutive_successes == 1

    async def test_probing_carries_since_and_stats_from_open(self) -> None:
        gate, _mono, wall = _gate()
        opened = await _report_failures(gate, 8)
        assert isinstance(opened, Open)
        wall.advance(120)
        state = await _report_ok(gate, 1)
        assert isinstance(state, Probing)
        assert state.since == opened.since
        assert state.stats == opened.stats

    async def test_probing_is_still_degraded(self) -> None:
        """One success is not recovery — consumers must keep throttling."""
        gate, _mono, _wall = _gate()
        await _report_failures(gate, 8)
        await _report_ok(gate, 1)
        assert gate.is_degraded is True

    async def test_second_success_closes_the_gate(self) -> None:
        gate, _mono, _wall = _gate()
        await _report_failures(gate, 8)
        await _report_ok(gate, 1)
        state = await _report_ok(gate, 1)
        assert isinstance(state, Closed)
        assert state is gate.state()
        assert gate.is_degraded is False

    async def test_close_clears_the_window_so_one_failure_cannot_re_trip(self) -> None:
        """The outage's own failures are still inside window_secs at close time.

        Without clearing, the very next 5xx would instantly re-trip and the
        breaker would oscillate.  A re-trip must mean "the provider is down
        again", not "the provider was down a minute ago".
        """
        gate, _mono, _wall = _gate()
        await _report_failures(gate, 8)
        await _report_ok(gate, 2)
        assert isinstance(gate.state(), Closed)
        state = await _report_failures(gate, 1)
        assert isinstance(state, Closed)
        assert state.__class__ is Closed

    async def test_re_trip_requires_a_fresh_full_burst(self) -> None:
        gate, _mono, _wall = _gate()
        await _report_failures(gate, 8)
        await _report_ok(gate, 2)
        state = await _report_failures(gate, 7)
        assert isinstance(state, Closed), 'a partial post-close burst must not trip'
        state = await _report_failures(gate, 1)
        assert isinstance(state, Open)

    async def test_a_still_degraded_provider_can_flap_closed_on_two_lucky_probes(self) -> None:
        """amend: pin the cost of `_close`'s clear, so the trade-off stays visible.

        A provider that is 60% degraded — not recovered — hands out two
        consecutive successes soon enough.  The gate closes, the fleet
        un-throttles, AND the accumulated failure evidence is discarded, so
        re-tripping needs a fresh full burst (8 failures / 3 tasks / >=50%)
        rather than the one further failure the live window would justify.

        Accepted, not overlooked: oscillation after a REAL recovery is
        guaranteed and self-sustaining (the trip's own failures are always
        still in window at close time), while this flap costs one burst of
        throughput and then re-trips.  See `ApiHealthGate._close`.  If this
        assertion ever has to change, that is a deliberate policy change.
        """
        gate, _mono, _wall = _gate()
        assert isinstance(await _report_failures(gate, 8), Open)

        # Two lucky probes out of a still-failing provider.
        assert isinstance(await _report_ok(gate, 2), Closed)
        assert gate.is_degraded is False, 'the whole fleet is now un-throttled'

        # The evidence went with it: a 60%-failing provider needs a full fresh
        # burst to trip again, not the single failure the true window implies.
        state = await _report_failures(gate, 7)
        assert isinstance(state, Closed)
        assert state is gate.state()
        assert isinstance(await _report_failures(gate, 1), Open), 're-trip needs the 8th'

    async def test_hysteresis_is_configurable(self) -> None:
        """close_after_successes=1 closes on the first success; =3 needs three."""
        gate, _mono, _wall = _gate(close_after_successes=1)
        await _report_failures(gate, 8)
        assert isinstance(await _report_ok(gate, 1), Closed)

        strict, _mono2, _wall2 = _gate(close_after_successes=3)
        await _report_failures(strict, 8)
        assert isinstance(await _report_ok(strict, 1), Probing)
        assert isinstance(await _report_ok(strict, 1), Probing)
        assert isinstance(await _report_ok(strict, 1), Closed)


class TestProbeRegression:
    """step-17: a failed probe returns to Open without restarting the outage clock."""

    async def test_failure_while_probing_returns_to_open(self) -> None:
        gate, _mono, _wall = _gate()
        await _report_failures(gate, 8)
        assert isinstance(await _report_ok(gate, 1), Probing)
        state = await _report_failures(gate, 1)
        assert isinstance(state, Open)
        assert gate.is_degraded is True

    async def test_since_is_the_original_trip_not_restamped(self) -> None:
        """The L2 promotion clock must measure the TRUE continuous outage.

        Restamping on every probe flap would let a provider that fails one
        probe every 90 minutes stay Open indefinitely and never promote to
        L2 — precisely the storm-escape decision 12 exists to prevent.
        """
        gate, _mono, wall = _gate()
        opened = await _report_failures(gate, 8)
        assert isinstance(opened, Open)
        wall.advance(5400)
        await _report_ok(gate, 1)
        wall.advance(5400)
        state = await _report_failures(gate, 1)
        assert isinstance(state, Open)
        assert state.since == opened.since
        assert state.since == _WALL

    async def test_stats_are_refreshed_from_the_current_window(self) -> None:
        """since is preserved, but the evidence is not frozen at the first trip."""
        gate, _mono, _wall = _gate()
        opened = await _report_failures(gate, 8)
        assert isinstance(opened, Open)
        await _report_ok(gate, 1)
        state = await _report_failures(gate, 1)
        assert isinstance(state, Open)
        assert state.stats.completions == 10
        assert state.stats.failures == 9
        assert state.stats != opened.stats

    async def test_streak_resets_to_zero_not_resumed(self) -> None:
        """After a failed probe it takes a FULL fresh streak to close."""
        gate, _mono, _wall = _gate()
        await _report_failures(gate, 8)
        await _report_ok(gate, 1)
        await _report_failures(gate, 1)
        state = await _report_ok(gate, 1)
        assert isinstance(state, Probing), 'streak resumed at 1 and closed too early'
        assert state.consecutive_successes == 1
        state = await _report_ok(gate, 1)
        assert isinstance(state, Closed)

    async def test_failure_while_open_keeps_since_and_refreshes_stats(self) -> None:
        gate, _mono, wall = _gate()
        opened = await _report_failures(gate, 8)
        assert isinstance(opened, Open)
        wall.advance(600)
        state = await _report_failures(gate, 1)
        assert isinstance(state, Open)
        assert state.since == opened.since
        assert state.stats.failures == 9


_NEUTRAL_OUTCOMES = [
    CapHit(resets_at=None, reason='limit reached'),
    NearCap(reason='approaching limit'),
    AuthFailed(status=401),
    CliLocalError(marker='ENOENT'),
    ModelNotFound(reason='no such model'),
    ZeroOutputWedge(),
    Failure(kind='unknown'),
]


async def _report_outcome(
    gate: ApiHealthGate, outcome, n: int = 1, *, task_id: str | None = 't1'
) -> GateState:
    state: GateState = gate.state()
    for _ in range(n):
        state = await gate.report(outcome, task_id=task_id, account='max-d', role='agent')
    return state


class TestNeutralOutcomes:
    """step-19: everything that is neither OK nor ServerError is denominator-only."""

    @pytest.mark.parametrize('outcome', _NEUTRAL_OUTCOMES, ids=lambda o: type(o).__name__)
    async def test_counted_in_completions_never_in_failures(self, outcome) -> None:
        gate, _mono, _wall = _gate()
        await _report_outcome(gate, outcome, 5)
        await _report_failures(gate, 3)
        state = gate.state()
        assert isinstance(state, Closed)
        # Force a trip to read the stats off a state that carries them.
        opened = await _report_failures(gate, 5)
        assert isinstance(opened, Open)
        assert opened.stats.completions == 13
        assert opened.stats.failures == 8

    @pytest.mark.parametrize('outcome', _NEUTRAL_OUTCOMES, ids=lambda o: type(o).__name__)
    async def test_closed_gate_saturated_with_neutrals_never_trips(self, outcome) -> None:
        """A locally-sick fleet must not trip the PROVIDER breaker."""
        gate, _mono, _wall = _gate()
        state = await _report_outcome(gate, outcome, 50)
        assert isinstance(state, Closed)
        assert gate.is_degraded is False

    @pytest.mark.parametrize('outcome', _NEUTRAL_OUTCOMES, ids=lambda o: type(o).__name__)
    async def test_neutral_neither_advances_nor_resets_the_probe_streak(self, outcome) -> None:
        """From Probing(1): a neutral leaves it Probing(1) — no close, no reopen.

        A CapHit is no evidence the provider recovered (so it must not close
        the gate) and none that it is still down (so it must not reopen it).
        """
        gate, _mono, _wall = _gate()
        await _report_failures(gate, 8)
        probing = await _report_ok(gate, 1)
        assert isinstance(probing, Probing)

        state = await _report_outcome(gate, outcome, 3)
        assert isinstance(state, Probing)
        assert state.consecutive_successes == 1
        assert gate.is_degraded is True

        state = await _report_ok(gate, 1)
        assert isinstance(state, Closed), 'the streak was not preserved across neutrals'

    async def test_neutrals_dilute_the_rate_rather_than_inflating_it(self) -> None:
        """8 failures across 3 tasks + 12 neutrals = 40%, so no trip.

        Excluding neutrals from the denominator would read 100% and halt a
        fleet whose problem is entirely local.
        """
        gate, _mono, _wall = _gate()
        await _report_outcome(gate, CapHit(resets_at=None, reason='limit'), 12)
        state = await _report_failures(gate, 8)
        assert isinstance(state, Closed)
        assert gate.is_degraded is False


class TestForensicRows:
    """step-21: one api_error row per ServerError, readable via the cost-store read path.

    This is the task's VERIFY signal end to end: gate -> save_api_error_event
    -> account_events -> account_events_in_window.
    """

    async def _gate_with_store(self, tmp_path: Path) -> tuple[ApiHealthGate, CostStore, _FakeWall]:
        store = CostStore(tmp_path / 'costs.db')
        await store.open()
        mono, wall = _FakeMonotonic(), _FakeWall()
        gate = ApiHealthGate(
            cost_store=store,
            project_id='dark_factory',
            run_id='run-xyz',
            monotonic_clock=mono,
            wall_clock=wall,
        )
        return gate, store, wall

    async def _rows(self, store: CostStore) -> list[dict]:
        return await store.account_events_in_window(
            '2000-01-01T00:00:00+00:00',
            '2099-01-01T00:00:00+00:00',
            event_type='api_error',
        )

    async def test_one_row_per_server_error_and_none_for_other_outcomes(
        self, tmp_path: Path
    ) -> None:
        gate, store, _wall = await self._gate_with_store(tmp_path)
        try:
            await _report_failures(gate, 3)
            await _report_ok(gate, 4)
            await _report_outcome(gate, CapHit(resets_at=None, reason='limit'), 2)
            await _report_outcome(gate, ZeroOutputWedge(), 2)
            rows = await self._rows(store)
        finally:
            await store.close()
        assert len(rows) == 3

    async def test_row_carries_account_project_run_and_wall_clock(self, tmp_path: Path) -> None:
        gate, store, wall = await self._gate_with_store(tmp_path)
        wall.advance(60)
        try:
            await gate.report(
                ServerError(status=503), task_id='t9', account='max-c', role='watcher'
            )
            rows = await self._rows(store)
        finally:
            await store.close()
        assert len(rows) == 1
        assert rows[0]['account_name'] == 'max-c'
        assert rows[0]['project_id'] == 'dark_factory'
        assert rows[0]['run_id'] == 'run-xyz'
        assert rows[0]['created_at'] == (_WALL + timedelta(seconds=60)).isoformat()

    async def test_details_json_reconstructs_the_moment(self, tmp_path: Path) -> None:
        """A trip must be reconstructable from the row stream alone."""
        gate, store, _wall = await self._gate_with_store(tmp_path)
        try:
            await _report_failures(gate, 8)
            rows = await self._rows(store)
        finally:
            await store.close()
        assert len(rows) == 8
        first = json.loads(rows[0]['details'])
        assert first['status'] == 529
        assert first['task_id'] == 't1'
        assert first['role'] == 'agent'
        assert first['state_after'] == 'Closed'
        assert first['failures'] == 1

        last = json.loads(rows[-1]['details'])
        assert last['state_after'] == 'Open', 'the row must record the POST-report state'
        assert last['failures'] == 8
        assert last['completions'] == 8
        assert last['distinct_tasks'] == 3
        assert last['failure_rate'] == 1.0

    async def test_status_is_recorded_verbatim(self, tmp_path: Path) -> None:
        gate, store, _wall = await self._gate_with_store(tmp_path)
        try:
            for status in (500, 502, 503, 529):
                await gate.report(
                    ServerError(status=status), task_id='t1', account='max-d', role='agent'
                )
            rows = await self._rows(store)
        finally:
            await store.close()
        assert [json.loads(r['details'])['status'] for r in rows] == [500, 502, 503, 529]


class _ExplodingStore:
    """A cost store whose forensics write always fails (full disk, locked DB, ...)."""

    def __init__(self) -> None:
        self.calls = 0

    async def save_api_error_event(self, **kwargs) -> None:
        self.calls += 1
        raise OSError('disk full')


class TestPersistenceIsTelemetryOnly:
    """step-23: forensics are best-effort; the in-memory state machine is authoritative."""

    async def test_write_failure_does_not_propagate(self) -> None:
        """A full disk must never fail the invocation that reported the 5xx."""
        store = _ExplodingStore()
        gate, _mono, _wall = _gate(cost_store=store)
        state = await gate.report(
            ServerError(status=529), task_id='t1', account='max-d', role='agent'
        )
        assert isinstance(state, Closed)
        assert store.calls == 1

    async def test_trip_still_happens_through_a_failing_store(self) -> None:
        """A write failure must not suppress the trip — that would hide an outage."""
        store = _ExplodingStore()
        gate, _mono, _wall = _gate(cost_store=store)
        state = await _report_failures(gate, 8)
        assert isinstance(state, Open)
        assert gate.is_degraded is True
        assert store.calls == 8

    async def test_close_still_happens_through_a_failing_store(self) -> None:
        store = _ExplodingStore()
        gate, _mono, _wall = _gate(cost_store=store)
        await _report_failures(gate, 8)
        await _report_ok(gate, 2)
        assert isinstance(gate.state(), Closed)

    async def test_write_failure_is_logged_at_warning(self, caplog) -> None:
        """Loud, not silent: the operator must be able to see forensics were lost."""
        store = _ExplodingStore()
        gate, _mono, _wall = _gate(cost_store=store)
        with caplog.at_level(logging.WARNING, logger='shared.api_health'):
            await gate.report(ServerError(status=529), task_id='t1', account='max-d', role='agent')
        warnings = [r for r in caplog.records if r.levelno >= logging.WARNING]
        assert warnings, 'the dropped forensics row was swallowed silently'
        assert 'disk full' in caplog.text

    async def test_gate_without_a_cost_store_trips_normally(self) -> None:
        """Usable in xi before the store is wired."""
        gate, _mono, _wall = _gate(cost_store=None)
        state = await _report_failures(gate, 8)
        assert isinstance(state, Open)
        state = await _report_ok(gate, 2)
        assert isinstance(state, Closed)


class _SuspendingStore:
    """A cost store whose forensics write parks until the test releases it.

    Stands in for a slow disk / contended DB: it holds every reporter inside
    `report()` at the one and only await, which is what lets a test observe
    the state machine mid-flight.
    """

    def __init__(self) -> None:
        self.entered = 0
        self.release = asyncio.Event()

    async def save_api_error_event(self, **kwargs) -> None:
        self.entered += 1
        await self.release.wait()


async def _yield_until(predicate, *, ticks: int = 100) -> None:
    """Yield to the loop until ``predicate()`` holds, or ``ticks`` are exhausted.

    Deterministic (no wall-clock sleeps): the caller asserts on the condition
    afterwards, so a genuine failure surfaces as that assertion, not a hang.
    """
    for _ in range(ticks):
        if predicate():
            return
        await asyncio.sleep(0)


class TestConcurrentReports:
    """amend: `report()` mutates state before its only await — no lock needed.

    The gate sits at the invocation choke point, so concurrent reports are the
    normal case, not an edge.  The invariant is load-bearing and fragile to an
    innocuous edit (awaiting anything before `self._state = ...`, or moving the
    forensics write above the transition); the failure it would cause — a
    sample or a trip silently lost during an outage — is invisible to every
    other test here, all of which report sequentially.
    """

    async def _gather_failures(
        self, gate: ApiHealthGate, store: _SuspendingStore, n: int
    ) -> asyncio.Future:
        """Start ``n`` concurrent ServerError reports and park them in the write."""
        pending = asyncio.gather(
            *[
                gate.report(
                    ServerError(status=529),
                    task_id=f't{i % 3 + 1}',
                    account='max-d',
                    role='agent',
                )
                for i in range(n)
            ]
        )
        await _yield_until(lambda: store.entered == n)
        return pending

    async def test_no_sample_is_lost_to_interleaving(self) -> None:
        store = _SuspendingStore()
        gate, _mono, _wall = _gate(cost_store=store)

        pending = await self._gather_failures(gate, store, 12)
        assert store.entered == 12, 'a report never reached the forensics write'
        store.release.set()
        states = await pending

        final = gate.state()
        assert isinstance(final, Open)
        assert final.stats.completions == 12, 'a concurrent report was dropped from the window'
        assert final.stats.failures == 12
        assert final.stats.distinct_tasks == 3
        assert states[-1] is final

    async def test_the_transition_happens_exactly_once(self, caplog) -> None:
        """12 concurrent failures trip the gate once, not once per over-threshold report."""
        store = _SuspendingStore()
        gate, _mono, _wall = _gate(cost_store=store)

        with caplog.at_level(logging.WARNING, logger='shared.api_health'):
            pending = await self._gather_failures(gate, store, 12)
            store.release.set()
            await pending

        trips = [r for r in caplog.records if 'TRIPPED' in r.getMessage()]
        assert len(trips) == 1, f'expected one Closed->Open transition, got {len(trips)}'

    async def test_a_parked_reporter_returns_the_current_state_not_a_stale_snapshot(
        self,
    ) -> None:
        """A reporter held in a slow write during someone else's trip must not un-throttle.

        `report()` reads `self._state` at RETURN time, after its forensics
        write, so every concurrent reporter observes the trip that landed while
        it was parked.  Returning each reporter's own pre-trip snapshot instead
        would hand a caller `Closed` during a known outage — the silent
        un-throttle the module docstring warns about — so this ordering is the
        contract, not an accident of where the await sits.
        """
        store = _SuspendingStore()
        gate, _mono, _wall = _gate(cost_store=store)

        pending = await self._gather_failures(gate, store, 12)
        store.release.set()
        states = await pending

        final = gate.state()
        assert isinstance(final, Open)
        assert all(s is final for s in states), 'a parked reporter returned a superseded state'

    async def test_state_is_visible_before_the_forensics_write_completes(self) -> None:
        """The trip must not be gated on I/O — a reader mid-write already sees Open.

        This is why the write is awaited AFTER the mutation: eight 5xx have
        landed, the disk is stuck, and the scheduler reading the gate right now
        must still throttle.
        """
        store = _SuspendingStore()
        gate, _mono, _wall = _gate(cost_store=store)

        pending = await self._gather_failures(gate, store, 8)
        assert isinstance(gate.state(), Open), 'the trip waited on the forensics write'
        assert gate.is_degraded is True
        store.release.set()
        await pending


class TestStaleEvidence:
    """amend: a degraded gate can outlive its evidence — expose that, don't hide it.

    Nothing ages out and nothing closes while the fleet is silent, and a trip
    CAUSES silence (dispatch throttles to a single probe).  Consumers whose
    behaviour escalates with time-spent-open must be able to tell "failing us
    right now" from "failed us hours ago and nothing has asked since".
    """

    async def test_fresh_gate_has_no_report_age(self) -> None:
        gate, _mono, _wall = _gate()
        assert gate.seconds_since_last_report() is None
        assert gate.evidence_is_stale is True, 'no evidence at all is not fresh evidence'

    async def test_report_age_tracks_the_monotonic_clock(self) -> None:
        gate, mono, _wall = _gate()
        await _report_failures(gate, 1)
        assert gate.seconds_since_last_report() == 0.0
        mono.advance(42.0)
        assert gate.seconds_since_last_report() == 42.0

    async def test_open_gate_goes_stale_after_a_silent_window(self) -> None:
        """The L2 promotion clock reads Open.since; this is what qualifies it."""
        gate, mono, _wall = _gate()
        assert isinstance(await _report_failures(gate, 8), Open)
        assert gate.evidence_is_stale is False

        mono.advance(599.0)
        assert gate.evidence_is_stale is False, 'still inside the window'
        mono.advance(2.0)
        assert gate.evidence_is_stale is True
        assert gate.is_degraded is True, 'staleness must not silently un-throttle'
        assert isinstance(gate.state(), Open), 'silence is not evidence of recovery'

    async def test_a_probe_refreshes_the_evidence_without_closing(self) -> None:
        gate, mono, _wall = _gate()
        await _report_failures(gate, 8)
        mono.advance(601.0)
        assert gate.evidence_is_stale is True
        await _report_ok(gate, 1)
        assert gate.evidence_is_stale is False
        assert gate.is_degraded is True, 'one probe is not recovery'

    async def test_age_survives_the_close_time_window_clear(self) -> None:
        """`_close` clears the samples; the report age is kept outside them on purpose."""
        gate, mono, _wall = _gate()
        await _report_failures(gate, 8)
        await _report_ok(gate, 2)
        assert isinstance(gate.state(), Closed)
        mono.advance(5.0)
        assert gate.seconds_since_last_report() == 5.0

    async def test_state_is_not_mutated_by_reading_it(self) -> None:
        """Reads never advance the machine — only a successful probe closes the gate."""
        gate, mono, _wall = _gate()
        opened = await _report_failures(gate, 8)
        mono.advance(86_400.0)
        for _ in range(3):
            assert gate.state() is opened
            assert gate.is_degraded is True
        assert isinstance(opened, Open)
        assert opened.stats.failures == 8, 'the carried evidence was rewritten by a read'
