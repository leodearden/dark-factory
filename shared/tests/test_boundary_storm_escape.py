"""The burst escape's own contract, tested away from any FastMCP server.

This collaborator exists because ``uuid_prefix_guard`` shipped ~120 lines that
were near-verbatim copies of ``mcp_markup_middleware``'s — including one
34-line contiguous identical block — and lock-step copies drift (INV-5). Its
tests belong HERE for the same reason: pinned once, they hold for every guard
that composes it, where a copy per guard would drift exactly as the code did.

Everything below drives the public surface — ``record``, ``call_sink``,
``tracked_keys`` and the two public tuning attributes. No test reaches a
private member; the one property that needed a window into this object's state
(that dormant counters are EVICTED, so a caller-supplied project key cannot
grow it without bound) is served by ``tracked_keys`` being part of the
interface rather than by poking at the dict behind it.
"""

from __future__ import annotations

import logging
from typing import Any

import pytest

from shared.boundary_storm_escape import BoundaryStormEscape, call_sink


class _Clock:
    """A hand-advanced clock, so a window is a decision and not a race."""

    def __init__(self, now: float = 0.0) -> None:
        self.now = now

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


OWNER = 'test guard'
ERROR_TYPE = 'test_boundary_storm'


def build_escape(
    escalation_sink: Any = None,
    *,
    clock: _Clock | None = None,
    **kwargs: Any,
) -> tuple[BoundaryStormEscape, _Clock]:
    clock = clock if clock is not None else _Clock()
    escape = BoundaryStormEscape(
        owner=OWNER,
        error_type=ERROR_TYPE,
        log_event='test_guard_storm',
        escalation_sink=escalation_sink,
        time_provider=clock,
        **kwargs,
    )
    return escape, clock


# ---------------------------------------------------------------------------
# Counting.
# ---------------------------------------------------------------------------


class TestABurstFiresOnceAtTheThreshold:
    async def test_below_the_threshold_nothing_is_returned(self) -> None:
        """``None`` is the overwhelmingly common answer, so a caller folds the
        result in with one ``is not None`` and never has to know what a quiet
        window looks like."""
        escape, _ = build_escape()
        assert [await escape.record('absorbed', 'alpha') for _ in range(2)] == [None, None]

    async def test_the_third_fires_with_the_declared_summary(self) -> None:
        escape, _ = build_escape()
        for _ in range(2):
            await escape.record('absorbed', 'alpha')
        assert await escape.record('absorbed', 'alpha') == {
            'count': 3,
            'threshold': 3,
            'window_seconds': 3600.0,
            'outcome': 'absorbed',
            'project': 'alpha',
        }

    async def test_the_burst_is_logged_at_error_under_the_declared_event_name(
        self, caplog: Any
    ) -> None:
        """Greppable and operator-facing: the summary a guard folds into its
        response reaches only the caller, which already knows."""
        escape, _ = build_escape()
        with caplog.at_level(logging.ERROR, logger='shared.boundary_storm_escape'):
            for _ in range(3):
                await escape.record('absorbed', 'alpha')
        assert [record.levelno for record in caplog.records] == [logging.ERROR]
        assert caplog.records[0].getMessage().startswith('test_guard_storm: 3 absorbed')

    async def test_a_project_with_no_name_still_counts(self) -> None:
        """``project`` is Optional on the wire — a guard that could not scope a
        call still owes its operator the burst."""
        escape, _ = build_escape()
        for _ in range(2):
            await escape.record('absorbed', None)
        summary = await escape.record('absorbed', None)
        assert summary is not None and summary['project'] is None


class TestTheKeyIsProjectAndOutcomeTogether:
    async def test_two_projects_do_not_pool_into_a_premature_fire(self) -> None:
        """One counter whose window spans every event regardless of label would
        fire on the fourth event here and name a project that saw only two."""
        escape, _ = build_escape()
        fired = [
            await escape.record('absorbed', project)
            for project in ('alpha', 'alpha', 'beta', 'beta')
        ]
        assert fired == [None, None, None, None]

    async def test_two_outcomes_do_not_pool_either(self) -> None:
        escape, _ = build_escape()
        fired = [
            await escape.record(outcome, 'alpha')
            for outcome in ('absorbed', 'absorbed', 'degraded', 'degraded')
        ]
        assert fired == [None, None, None, None]

    async def test_each_key_fires_on_its_own_third(self) -> None:
        escape, _ = build_escape()
        for _ in range(3):
            await escape.record('absorbed', 'alpha')
        assert await escape.record('absorbed', 'beta') is None


class TestTheTuningIsReadLive:
    """``StormCounter``'s reload-safety contract, kept at this layer.

    Threshold and window are passed PER record rather than captured into the
    counters, so a registration site whose numbers come from a green-tier
    config leaf can rebind them and have the next call honour them.
    """

    async def test_a_lowered_threshold_takes_effect_on_the_next_record(self) -> None:
        escape, _ = build_escape()
        assert await escape.record('absorbed', 'alpha') is None
        escape.threshold = 2
        summary = await escape.record('absorbed', 'alpha')
        assert summary is not None and (summary['count'], summary['threshold']) == (2, 2)

    async def test_the_window_is_read_live_too(self) -> None:
        escape, clock = build_escape(threshold=2)
        await escape.record('absorbed', 'alpha')
        clock.advance(50.0)
        escape.window_seconds = 10.0
        assert await escape.record('absorbed', 'alpha') is None, (
            'the earlier event is outside the narrowed window, so the burst has '
            'a count of one'
        )


class TestDormantCountersAreEvicted:
    """``project`` is CALLER-SUPPLIED, so one counter per key ever seen is a
    memory bound this object owes rather than an implementation detail."""

    async def test_an_idle_key_is_swept_when_another_records(self) -> None:
        escape, clock = build_escape(window_seconds=100.0)
        await escape.record('absorbed', 'alpha')
        clock.advance(200.0)
        await escape.record('absorbed', 'beta')
        assert escape.tracked_keys == {'beta\x1fabsorbed'}

    async def test_a_live_key_survives_the_sweep(self) -> None:
        """Eviction is age, never "everything but the key being recorded"."""
        escape, clock = build_escape(window_seconds=100.0)
        await escape.record('absorbed', 'alpha')
        clock.advance(10.0)
        await escape.record('absorbed', 'beta')
        assert escape.tracked_keys == {'alpha\x1fabsorbed', 'beta\x1fabsorbed'}

    async def test_tracked_keys_cannot_be_edited_through(self) -> None:
        escape, _ = build_escape()
        await escape.record('absorbed', 'alpha')
        assert isinstance(escape.tracked_keys, frozenset)


# ---------------------------------------------------------------------------
# The injected escalation sink.
# ---------------------------------------------------------------------------


class TestTheFiredBurstReachesTheSink:
    async def test_the_record_carries_the_error_type_and_the_summary(self) -> None:
        filed: list[dict[str, Any]] = []
        escape, _ = build_escape(filed.append)
        for _ in range(3):
            await escape.record('absorbed', 'alpha')
        assert filed == [
            {
                'error_type': ERROR_TYPE,
                'count': 3,
                'threshold': 3,
                'window_seconds': 3600.0,
                'outcome': 'absorbed',
                'project': 'alpha',
            }
        ]

    async def test_nothing_is_filed_below_the_threshold(self) -> None:
        filed: list[dict[str, Any]] = []
        escape, _ = build_escape(filed.append)
        for _ in range(2):
            await escape.record('absorbed', 'alpha')
        assert filed == []

    async def test_no_sink_is_not_an_error(self) -> None:
        escape, _ = build_escape(None)
        for _ in range(2):
            await escape.record('absorbed', 'alpha')
        assert await escape.record('absorbed', 'alpha') is not None

    async def test_an_async_sink_is_awaited(self) -> None:
        filed: list[dict[str, Any]] = []

        async def sink(record: dict[str, Any]) -> None:
            filed.append(record)

        escape, _ = build_escape(sink)
        for _ in range(3):
            await escape.record('absorbed', 'alpha')
        assert [record['count'] for record in filed] == [3]

    async def test_a_raising_sink_does_not_lose_the_summary(self) -> None:
        """The caller's outcome is already decided: a sink outage costs an
        operator visibility, never the burst the guard folds into its response."""

        def boom(_record: dict[str, Any]) -> None:
            raise RuntimeError('the queue is unavailable')

        escape, _ = build_escape(boom)
        for _ in range(2):
            await escape.record('absorbed', 'alpha')
        summary = await escape.record('absorbed', 'alpha')
        assert summary is not None and summary['count'] == 3


# ---------------------------------------------------------------------------
# call_sink, on its own.
# ---------------------------------------------------------------------------


class TestCallSink:
    async def test_a_sync_sink_result_is_returned(self) -> None:
        """The markup guard puts a returned escalation id in front of a caller,
        so the value is part of the contract and not just the call."""
        assert await call_sink(lambda record: record['id'], {'id': 'esc_1'}, 'x', owner=OWNER)

    async def test_an_awaitable_result_is_awaited_not_returned_as_a_coroutine(self) -> None:
        """A coroutine handed back unawaited LOOKS like a result while having
        queued nothing — the silent fail-soft these guards exist to end."""

        async def sink(record: dict[str, Any]) -> str:
            return record['id']

        assert await call_sink(sink, {'id': 'esc_1'}, 'x', owner=OWNER) == 'esc_1'

    @pytest.mark.parametrize('failing', ['sync', 'async'])
    async def test_a_raising_sink_yields_none_and_is_logged(
        self, failing: str, caplog: Any
    ) -> None:
        """Never raises — the never-raises contract has to survive the AWAIT as
        well as the call."""

        def sync_boom(_record: dict[str, Any]) -> None:
            raise RuntimeError('unavailable')

        async def async_boom(_record: dict[str, Any]) -> None:
            raise RuntimeError('unavailable')

        sink = sync_boom if failing == 'sync' else async_boom
        with caplog.at_level(logging.ERROR, logger='shared.boundary_storm_escape'):
            assert await call_sink(sink, {'fact': 'f'}, 'fact', owner=OWNER) is None
        assert 'test guard: the fact sink failed' in caplog.text

    async def test_the_log_names_the_owner_so_two_guards_are_distinguishable(
        self, caplog: Any
    ) -> None:
        def boom(_record: dict[str, Any]) -> None:
            raise RuntimeError('unavailable')

        with caplog.at_level(logging.ERROR, logger='shared.boundary_storm_escape'):
            await call_sink(boom, {'error_type': 'e'}, 'escalation', owner='other guard')
        assert 'other guard: the escalation sink failed' in caplog.text
