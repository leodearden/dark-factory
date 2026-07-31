"""Tests for ApiHealthGate — the C4 provider-degradation circuit breaker.

Contract: plans/server-side-api-error-handling-prd.md C4 (task 3325 / mu).
"""

from __future__ import annotations

import dataclasses
from datetime import UTC, datetime, timedelta

import pytest

from shared.api_health import ApiHealthGate, Closed, GateState, GateStats, Open, Probing
from shared.invocation_outcome import OK, ServerError

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
