"""Tests for ``shared.storm_counter.StormCounter`` — the rolling-window burst
detector, at its single home in the base layer (task 3689).

WHY THE CLASS MOVED HERE. Task 3088 extracted this body out of its three
copies (``reconciliation/harness.py::_record_placeholder_finding_drop``,
``harness._dead_owner_suppressions``, ``server/markup_tripwire.MarkupStormCounter``)
into ``fused_memory.server.storm_counter`` "so a fourth consumer reuses rather
than re-copies it (INV-5)". ``shared.mcp_markup_middleware`` IS that fourth
consumer, and it lives in the base layer — ``shared`` may not import
``fused_memory``, so the class had to come to it. The old module stays as a
re-export shim, mirroring exactly what task 3688 did to markup_tripwire's
``MCP_MARKUP_PATTERNS``; ``fused-memory/tests/server/test_storm_counter.py``
keeps running unedited against that shim, which is what pins the shim honest.

These assertions are PORTED from that existing coverage rather than invented,
because the contract is not supposed to change in the move — only the import
path is.
"""

from __future__ import annotations

import pytest

from shared.storm_counter import StormCounter


class _FakeClock:
    """Injectable clock, so a 3600s window is exercised without sleeping."""

    def __init__(self, start: float = 1000.0) -> None:
        self.now = start

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


@pytest.fixture
def clock():
    return _FakeClock()


@pytest.fixture
def counter(clock):
    return StormCounter(time_provider=clock)


class TestReExportShim:
    """The old home is the SAME class object, not a second implementation.

    The re-export contract, in the shape task 3688 established for
    ``MCP_MARKUP_PATTERNS``: promote to ``shared``, re-export from the old home,
    public names unchanged, so fused-memory's importers
    (``server/markup_tripwire.py``, ``services/memory_service.py``) need no edit.

    This identity is the one thing the OLD suite is uniquely qualified to pin,
    and it is what makes every contract assertion below cover both import paths
    at once — ``fused-memory/tests/server/test_storm_counter.py`` re-tests the
    same contract against the same object today, which is duplication that
    belongs at the shim, not at the behaviour. Collapsing that suite is out of
    this task's lock scope.

    Guarded by importorskip so shared's suite stays independent of fused-memory
    being installed — shared is the base layer and may not require it.
    """

    def test_the_old_home_re_exports_this_very_class(self):
        shim = pytest.importorskip('fused_memory.server.storm_counter')

        assert shim.StormCounter is StormCounter


class TestThreshold:
    """(a) below threshold returns None; (b) at threshold returns the summary."""

    def test_returns_none_below_threshold(self, counter):
        assert counter.record(threshold=3, window_seconds=3600.0, label='p') is None
        assert counter.record(threshold=3, window_seconds=3600.0, label='p') is None

    def test_fires_exactly_at_threshold_with_a_complete_summary(self, counter):
        counter.record(threshold=3, window_seconds=3600.0, label='p')
        counter.record(threshold=3, window_seconds=3600.0, label='p')
        summary = counter.record(threshold=3, window_seconds=3600.0, label='p')

        assert summary is not None, 'the threshold-th event must fire'
        # The whole summary shape, not just the count: this is what the
        # middleware folds into its rejection payload under `storm`, so a
        # missing key would surface as a silently thinner operator signal.
        assert summary == {
            'count': 3,
            'threshold': 3,
            'window_seconds': 3600.0,
            'labels': ['p'],
        }


class TestRateLimit:
    """(c) a second crossing while a previous fire is still in the window."""

    def test_burst_of_hundreds_yields_exactly_one_fire(self, counter):
        fires = [
            counter.record(threshold=5, window_seconds=3600.0, label='p')
            for _ in range(500)
        ]
        assert sum(1 for f in fires if f is not None) == 1, (
            'a burst must escalate ONCE per window, not once per event'
        )

    def test_fires_again_in_the_next_window(self, counter, clock):
        for _ in range(3):
            counter.record(threshold=3, window_seconds=100.0, label='p')
        clock.advance(101.0)
        for _ in range(2):
            assert counter.record(threshold=3, window_seconds=100.0, label='p') is None
        assert counter.record(threshold=3, window_seconds=100.0, label='p') is not None


class TestWindowPruning:
    """(d) events older than window_seconds are pruned before the count."""

    def test_window_is_half_open(self, counter, clock):
        """An event aged EXACTLY window_seconds is already out of the window."""
        counter.record(threshold=2, window_seconds=100.0, label='p')
        clock.advance(100.0)
        assert counter.record(threshold=2, window_seconds=100.0, label='p') is None, (
            'the first event aged exactly window_seconds must have been pruned, '
            'leaving a count of 1 below the threshold of 2'
        )

    def test_event_just_inside_window_still_counts(self, counter, clock):
        counter.record(threshold=2, window_seconds=100.0, label='p')
        clock.advance(99.0)
        assert counter.record(threshold=2, window_seconds=100.0, label='p') is not None

    def test_a_burst_spread_across_two_windows_does_not_fire(self, counter, clock):
        """Calls spaced wider than the window are always evicted first."""
        for _ in range(20):
            assert counter.record(threshold=3, window_seconds=100.0, label='p') is None
            clock.advance(101.0)


class TestLabels:
    """(e) labels are the sorted DISTINCT non-None labels seen in the window."""

    def test_reports_sorted_distinct_labels(self, counter):
        counter.record(threshold=3, window_seconds=3600.0, label='b')
        counter.record(threshold=3, window_seconds=3600.0, label='a')
        summary = counter.record(threshold=3, window_seconds=3600.0, label='b')

        assert summary is not None
        assert summary['labels'] == ['a', 'b'], (
            f'expected sorted distinct labels, got {summary["labels"]!r}'
        )

    def test_unlabelled_events_count_but_are_not_named(self, counter):
        counter.record(threshold=3, window_seconds=3600.0, label=None)
        counter.record(threshold=3, window_seconds=3600.0, label=None)
        summary = counter.record(threshold=3, window_seconds=3600.0, label='a')

        assert summary is not None
        assert summary['count'] == 3, 'unlabelled events must count toward the burst'
        assert summary['labels'] == ['a']

    def test_labels_are_opaque_strings_carrying_no_schema(self, counter):
        """What lets the middleware key on a COMPOSED (project, outcome) label.

        The same class already serves per-``project_root`` keying
        (``MarkupStormCounter``) and per-``agent_id`` keying
        (``MemoryService.update_memory``). ``shared.mcp_markup_middleware``
        composes both halves into one opaque string rather than growing this
        class a second key dimension.
        """
        counter.record(threshold=2, window_seconds=3600.0, label='/home/leo/x|repaired')
        summary = counter.record(threshold=2, window_seconds=3600.0, label='recon-stage-1')

        assert summary is not None
        assert summary['labels'] == ['/home/leo/x|repaired', 'recon-stage-1']


class TestLiveReadContract:
    """(f) threshold and window_seconds are supplied PER record() call.

    The class docstring's RELOAD SAFETY paragraph: a config value captured by
    value at construction cannot observe an in-place reload and would have to
    stay restart-only, so a constructor-captured threshold would make
    ``mem0_update.storm_threshold`` a restart-only leaf masquerading as
    green-tier in ``RELOADABLE_FIELDS``.
    """

    def test_threshold_lowered_between_calls_takes_effect_on_the_second(self, counter):
        assert counter.record(threshold=10, window_seconds=3600.0, label='p') is None
        summary = counter.record(threshold=2, window_seconds=3600.0, label='p')

        assert summary is not None, (
            'a threshold lowered between two record() calls must take effect on '
            'the second — this is what a constructor-captured threshold cannot do'
        )
        assert summary['threshold'] == 2, 'the summary must report the LIVE threshold'

    def test_window_narrowed_between_calls_prunes_on_the_next_call(self, counter, clock):
        counter.record(threshold=2, window_seconds=3600.0, label='p')
        clock.advance(200.0)
        assert counter.record(threshold=2, window_seconds=100.0, label='p') is None, (
            'a narrowed window must prune on the very next call'
        )

    def test_raised_threshold_stops_firing(self, counter):
        counter.record(threshold=2, window_seconds=3600.0, label='p')
        assert counter.record(threshold=2, window_seconds=3600.0, label='p') is not None
        assert counter.record(threshold=100, window_seconds=3600.0, label='p') is None


class TestPerInstanceState:
    """Two counters share nothing — the property the middleware relies on.

    ``MarkupGuardMiddleware`` holds ONE counter per middleware instance so no
    burst state bleeds between servers, or between tests in one process. That
    is only true if the deque and the rate-limit stamp are instance state.
    """

    def test_two_instances_do_not_pool_events(self, clock):
        a = StormCounter(time_provider=clock)
        b = StormCounter(time_provider=clock)

        for _ in range(2):
            assert a.record(threshold=3, window_seconds=3600.0, label='p') is None
            assert b.record(threshold=3, window_seconds=3600.0, label='p') is None

        assert a.record(threshold=3, window_seconds=3600.0, label='p') is not None
        assert b.record(threshold=3, window_seconds=3600.0, label='p') is not None, (
            'b must reach its own threshold on its own third event — if the two '
            'instances pooled, b would have fired on its second'
        )

    def test_one_instances_rate_limit_does_not_silence_another(self, clock):
        a = StormCounter(time_provider=clock)
        b = StormCounter(time_provider=clock)

        for _ in range(3):
            a.record(threshold=3, window_seconds=3600.0, label='p')
        for _ in range(2):
            b.record(threshold=3, window_seconds=3600.0, label='p')

        assert b.record(threshold=3, window_seconds=3600.0, label='p') is not None, (
            "a's fire must not consume b's one-per-window budget"
        )


class TestPruneSweepHook:
    """``prune()`` — ages events out without recording one.

    Ported because the promoted module must carry the whole public surface, not
    just the part the new consumer happens to call: ``MemoryService`` keys one
    counter per caller-supplied ``agent_id`` and needs this to evict quiet ones.
    """

    def test_reports_the_live_count_without_recording_an_event(self, counter):
        counter.record(threshold=99, window_seconds=100.0, label='p')
        counter.record(threshold=99, window_seconds=100.0, label='p')

        assert counter.prune(100.0) == 2
        assert counter.prune(100.0) == 2, 'prune must not itself count as an event'

    def test_returns_zero_once_the_window_has_emptied(self, counter, clock):
        counter.record(threshold=99, window_seconds=100.0, label='p')
        clock.advance(101.0)

        assert counter.prune(100.0) == 0

    def test_a_pruned_counter_decides_like_a_fresh_one(self, counter, clock):
        """Why dropping an empty counter is behaviour-preserving, not just cheap.

        The only state besides the deque is ``_last_fire_ts``, and it is stamped
        while its own event is still inside the window — so an empty window
        implies that fire has already aged past the rate limit. A caller that
        evicts on ``prune() == 0`` and reconstructs later must therefore get the
        same answer it would have got by keeping the object.

        This is what licenses ``MarkupGuardMiddleware``'s dormant-key sweep:
        it holds one counter per ``(project, outcome)`` key and deletes the ones
        whose ``prune()`` reports zero, which is only sound if a reconstructed
        counter cannot decide differently from the one it replaced.
        """
        for _ in range(3):
            counter.record(threshold=3, window_seconds=100.0, label='p')
        clock.advance(101.0)
        assert counter.prune(100.0) == 0

        kept = [
            counter.record(threshold=3, window_seconds=100.0, label='p')
            for _ in range(3)
        ]
        fresh_counter = StormCounter(time_provider=clock)
        fresh = [
            fresh_counter.record(threshold=3, window_seconds=100.0, label='p')
            for _ in range(3)
        ]

        assert [f is not None for f in kept] == [f is not None for f in fresh]
        assert kept[-1] == fresh[-1]
