"""Tests for the shared rolling-window StormCounter (task 3088).

The body — append, prune to the window, count, compare to threshold,
rate-limit to one fire per window, report the distinct labels seen — had THREE
copies on main (``reconciliation/harness.py::_record_placeholder_finding_drop``,
``harness._dead_owner_suppressions``, ``server/markup_tripwire.MarkupStormCounter``),
each carrying a "reproduces the established body from X" comment. This is its
single home (INV-5); ``MarkupStormCounter`` becomes a thin adapter over it and
``MemoryService`` is the second consumer.

ONE contract change versus MarkupStormCounter, and it is load-bearing rather
than cosmetic: ``threshold`` and ``window_seconds`` are supplied PER
``record()`` CALL, not captured in ``__init__``. ``config/reload.py``'s
reload-safety rule states that a knob captured by value at construction cannot
observe an in-place reload and must stay restart-only — so a
constructor-captured threshold would make ``mem0_update.storm_threshold`` a
restart-only leaf masquerading as green-tier in ``RELOADABLE_FIELDS``.
"""

from __future__ import annotations

import pytest

from fused_memory.server.storm_counter import StormCounter


class _FakeClock:
    """Injectable monotonic-ish clock, so a 3600s window needs no sleeping."""

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


class TestBelowThreshold:
    def test_returns_none_below_threshold(self, counter):
        assert counter.record(threshold=3, window_seconds=3600.0, label='p') is None
        assert counter.record(threshold=3, window_seconds=3600.0, label='p') is None

    def test_fires_exactly_at_threshold(self, counter):
        counter.record(threshold=3, window_seconds=3600.0, label='p')
        counter.record(threshold=3, window_seconds=3600.0, label='p')
        summary = counter.record(threshold=3, window_seconds=3600.0, label='p')
        assert summary is not None, 'the threshold-th event must fire'
        assert summary['count'] == 3
        assert summary['threshold'] == 3
        assert summary['window_seconds'] == 3600.0


class TestWindowPruning:
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

    def test_slow_steady_rate_never_trips(self, counter, clock):
        """Calls spaced wider than the window are always evicted first."""
        for _ in range(20):
            assert counter.record(threshold=3, window_seconds=100.0, label='p') is None
            clock.advance(101.0)


class TestPruneSweepHook:
    """``prune()`` — the hook a caller with UNBOUNDED label cardinality needs.

    ``MemoryService`` keys one counter per caller-supplied ``agent_id``. Each
    counter self-prunes its own deque on ``record()``, but nothing would ever
    drop the counter OBJECT, so a server that runs for weeks accumulates one
    per label it ever saw. ``prune()`` lets the caller ask "has this one gone
    quiet?" without recording a phantom event that would itself keep it alive.
    """

    def test_reports_the_live_count_without_recording_an_event(self, counter, clock):
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


class TestRateLimit:
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


class TestLabels:
    def test_reports_sorted_distinct_labels(self, counter):
        counter.record(threshold=3, window_seconds=3600.0, label='b')
        counter.record(threshold=3, window_seconds=3600.0, label='a')
        summary = counter.record(threshold=3, window_seconds=3600.0, label='b')
        assert summary is not None
        assert summary['labels'] == ['a', 'b'], (
            f'expected sorted distinct labels, got {summary["labels"]!r}'
        )

    def test_unlabelled_events_count_but_are_not_named(self, counter):
        """An unresolvable label still contributes to the burst, but there is
        nothing to attribute it to, so it is simply not named."""
        counter.record(threshold=3, window_seconds=3600.0, label=None)
        counter.record(threshold=3, window_seconds=3600.0, label=None)
        summary = counter.record(threshold=3, window_seconds=3600.0, label='a')
        assert summary is not None
        assert summary['count'] == 3, 'unlabelled events must count toward the burst'
        assert summary['labels'] == ['a']

    def test_labels_are_opaque_strings(self, counter):
        """The same class serves per-project_root (markup) and per-agent_id
        (mem0 update) keying — the label carries no schema."""
        counter.record(threshold=2, window_seconds=3600.0, label='recon-stage-1')
        summary = counter.record(threshold=2, window_seconds=3600.0, label='/home/leo/x')
        assert summary is not None
        assert summary['labels'] == ['/home/leo/x', 'recon-stage-1']


class TestLiveReadContract:
    """The assertion that makes the green-tier registration honest."""

    def test_threshold_changed_between_calls_takes_effect_on_the_second(self, counter):
        assert counter.record(threshold=10, window_seconds=3600.0, label='p') is None
        summary = counter.record(threshold=2, window_seconds=3600.0, label='p')
        assert summary is not None, (
            'a threshold lowered between two record() calls must take effect on '
            'the second — this is what a constructor-captured threshold cannot do, '
            'and why mem0_update.storm_threshold can be registered as green-tier'
        )
        assert summary['threshold'] == 2, 'the summary must report the LIVE threshold'

    def test_window_changed_between_calls_takes_effect_on_the_second(self, counter, clock):
        counter.record(threshold=2, window_seconds=3600.0, label='p')
        clock.advance(200.0)
        # A window narrowed to 100s now excludes the first event.
        assert counter.record(threshold=2, window_seconds=100.0, label='p') is None, (
            'a narrowed window must prune on the very next call'
        )

    def test_raised_threshold_stops_firing(self, counter):
        counter.record(threshold=2, window_seconds=3600.0, label='p')
        assert counter.record(threshold=2, window_seconds=3600.0, label='p') is not None
        # Raised beyond the standing count and past the rate limit window.
        assert counter.record(threshold=100, window_seconds=3600.0, label='p') is None
