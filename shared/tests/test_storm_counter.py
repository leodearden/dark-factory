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


class TestPerCallClockOverride:
    """``now=`` on :meth:`record` / :meth:`prune` — a PER-CALL clock override.

    Why the class needs a second time-injection door when ``time_provider``
    already exists. ``time_provider`` binds the clock at CONSTRUCTION, which
    fits a consumer that owns its counter for the process lifetime
    (``MarkupStormCounter``, ``MemoryService``). But
    ``reconciliation/harness.py``'s three storm counters inject time PER CALL —
    ``now: datetime | None = None``, the ``_finding_recently_resolved``
    convention at harness.py:1592 — and their callers pass an explicit instant
    that the harness itself resolved. Without a per-call override those
    counters could only delegate here via a mutable clock-holder shim mutated
    around every call, which is exactly the hand-rolled state INV-5 is trying
    to delete.

    The override is optional and defaults to ``None``, so all five existing
    consumers — every one of which relies on constructor-injected clocks —
    keep the behaviour pinned by the classes above.
    """

    def test_injected_now_drives_the_window_not_the_time_provider(self):
        """A threshold crossing driven entirely by injected instants.

        The counter is built with the DEFAULT ``time.time`` clock, so if
        ``now=`` were ignored the window would be built from wall-clock
        instants instead — and these three events, injected 10 000s apart,
        would all land inside the 100s window and fire.
        """
        counter = StormCounter()

        assert counter.record(
            threshold=3, window_seconds=3600.0, label='p', now=5_000.0
        ) is None
        assert counter.record(
            threshold=3, window_seconds=3600.0, label='p', now=5_001.0
        ) is None
        summary = counter.record(
            threshold=3, window_seconds=3600.0, label='p', now=5_002.0
        )

        assert summary is not None, 'the threshold-th injected event must fire'
        assert summary == {
            'count': 3,
            'threshold': 3,
            'window_seconds': 3600.0,
            'labels': ['p'],
        }

    def test_injected_now_prunes_so_spread_events_never_accumulate(self):
        """``now=`` ages events out, it does not merely stamp them.

        Wall clock barely moves across these calls, so only the injected
        instants can push each prior event out of the window.
        """
        counter = StormCounter()

        for i in range(20):
            assert counter.record(
                threshold=3,
                window_seconds=100.0,
                label='p',
                now=1_000.0 + i * 101.0,
            ) is None

    def test_injected_now_re_arms_the_rate_limit(self):
        """The per-window rate limit is measured against injected time too."""
        counter = StormCounter()

        fires = [
            counter.record(threshold=3, window_seconds=100.0, label='p', now=1_000.0)
            for _ in range(5)
        ]
        assert sum(1 for f in fires if f is not None) == 1

        # A full window later — prior events pruned, rate limit re-armed.
        assert counter.record(
            threshold=3, window_seconds=100.0, label='p', now=1_200.0
        ) is None
        assert counter.record(
            threshold=3, window_seconds=100.0, label='p', now=1_201.0
        ) is None
        assert counter.record(
            threshold=3, window_seconds=100.0, label='p', now=1_202.0
        ) is not None

    def test_prune_ages_out_against_the_injected_instant(self):
        counter = StormCounter()
        counter.record(threshold=99, window_seconds=100.0, label='p', now=1_000.0)
        counter.record(threshold=99, window_seconds=100.0, label='p', now=1_001.0)

        assert counter.prune(100.0, now=1_050.0) == 2
        assert counter.prune(100.0, now=1_200.0) == 0

    def test_omitting_now_still_uses_the_constructor_clock(self, counter, clock):
        """The default path is unchanged — what keeps the five consumers safe."""
        counter.record(threshold=2, window_seconds=100.0, label='p')
        clock.advance(101.0)

        assert counter.prune(100.0) == 0, 'prune() with no now= must read self._now()'
        assert counter.record(threshold=2, window_seconds=100.0, label='p') is None, (
            'record() with no now= must read self._now(), so the first event is '
            'already pruned and the count is 1'
        )

    def test_now_and_time_provider_can_be_mixed_on_one_counter(self, counter, clock):
        """An injected instant does not disturb the constructor clock's state.

        The fake clock sits at 1000.0; an event injected at 1000.0 via ``now=``
        must be indistinguishable from one stamped by the provider.
        """
        counter.record(threshold=2, window_seconds=100.0, label='a', now=clock.now)
        summary = counter.record(threshold=2, window_seconds=100.0, label='b')

        assert summary is not None
        assert summary['count'] == 2
        assert summary['labels'] == ['a', 'b']


class TestDistinctKeyCounting:
    """``count_distinct=True`` + per-call ``key=`` — threshold on DISTINCT keys.

    Why a SECOND dimension rather than reusing ``label``.
    ``reconciliation/harness.py::_record_dead_owner_suppression`` needs two
    ORTHOGONAL axes: it thresholds on the number of distinct non-``None``
    ``instance_id`` values (task 2039 — all orphans recovered by ONE restart
    share that one dead owner's instance_id, so a single multi-project restart
    must contribute 1, not N), while still attributing the burst to the
    distinct ``project_id`` values it touched.

    Neither existing mechanism can express that. Labelling by ``instance_id``
    fails because the threshold is compared to the RAW event count, so ten
    suppressions sharing one instance_id would fire — precisely what
    ``test_record_dead_owner_suppression_single_dead_owner_multi_project_no_storm``
    (the esc-recon-50da2482-1 regression) forbids. The middleware's
    one-counter-per-key convention also fails, because the threshold is on the
    NUMBER of distinct keys — i.e. the number of counter objects — which no
    individual counter can observe.

    ``count_distinct`` is deliberately a CONSTRUCTOR flag while
    threshold/window stay per-call: it is a structural mode fixed by the call
    site, not a config leaf, so capturing it at construction cannot go stale.
    ``config/reload.py``'s reload-safety rule constrains config VALUES only.
    """

    def test_one_key_repeated_never_fires_however_large_the_burst(self):
        """The task-2039 benign-restart case: N events, ONE dead owner."""
        counter = StormCounter(count_distinct=True)

        fires = [
            counter.record(threshold=3, window_seconds=3600.0, label=f'p{i}', key='owner-1')
            for i in range(100)
        ]

        assert not any(f is not None for f in fires), (
            '100 events sharing ONE key are ONE incident — a distinct-key '
            'counter must never fire on them'
        )

    def test_fires_on_the_threshold_th_distinct_key(self):
        counter = StormCounter(count_distinct=True)

        assert counter.record(
            threshold=3, window_seconds=3600.0, label='p', key='owner-1'
        ) is None
        assert counter.record(
            threshold=3, window_seconds=3600.0, label='p', key='owner-2'
        ) is None
        summary = counter.record(
            threshold=3, window_seconds=3600.0, label='p', key='owner-3'
        )

        assert summary is not None, 'the threshold-th DISTINCT key must fire'

    def test_summary_count_reports_distinct_keys_not_raw_events(self):
        counter = StormCounter(count_distinct=True)

        # 7 events, 3 distinct keys.
        for key in ('owner-1', 'owner-1', 'owner-2', 'owner-2', 'owner-2', 'owner-3'):
            counter.record(threshold=3, window_seconds=3600.0, label='p', key=key)
        summary = counter.record(threshold=3, window_seconds=3600.0, label='p', key='owner-3')

        assert summary is not None
        assert summary['count'] == 3, (
            f'count must be the DISTINCT key count, got {summary["count"]!r} — '
            'reporting the raw event count would misstate the incident size to '
            'the operator reading the escalation'
        )

    def test_none_keys_neither_count_nor_block(self):
        """``key=None`` is excluded from the distinct set entirely.

        Defensive: ``dead_owner_shielded`` requires a matching non-None
        instance_id, so a None should not occur — but if one does it must not
        inflate the count toward the threshold, and it must not wedge the
        counter either.
        """
        counter = StormCounter(count_distinct=True)

        fires = [
            counter.record(threshold=2, window_seconds=3600.0, label='p', key=None)
            for _ in range(50)
        ]
        assert not any(f is not None for f in fires), 'a window of only-None keys never fires'

        # Real keys still accumulate normally alongside the None entries.
        assert counter.record(
            threshold=2, window_seconds=3600.0, label='p', key='owner-1'
        ) is None
        assert counter.record(
            threshold=2, window_seconds=3600.0, label='p', key='owner-2'
        ) is not None

    def test_label_stays_an_independent_attribution_dimension(self):
        """A burst keyed on one dimension is NAMED by the other.

        The dead-owner case exactly: threshold on distinct instance_ids, while
        ``labels`` reports the distinct project_ids those suppressions touched.
        The two counts differ, which is the whole point.
        """
        counter = StormCounter(count_distinct=True)

        # 2 distinct keys spread across 4 distinct labels.
        counter.record(threshold=2, window_seconds=3600.0, label='proj-b', key='owner-1')
        counter.record(threshold=2, window_seconds=3600.0, label='proj-a', key='owner-1')
        counter.record(threshold=2, window_seconds=3600.0, label='proj-d', key='owner-1')
        summary = counter.record(
            threshold=2, window_seconds=3600.0, label='proj-c', key='owner-2'
        )

        assert summary is not None
        assert summary['count'] == 2, 'count follows the KEY dimension'
        assert summary['labels'] == ['proj-a', 'proj-b', 'proj-c', 'proj-d'], (
            'labels follow the LABEL dimension, independently of the keys'
        )

    def test_unlabelled_events_are_still_not_named(self):
        """``label=None`` behaves as it always has, orthogonally to keys."""
        counter = StormCounter(count_distinct=True)

        counter.record(threshold=2, window_seconds=3600.0, label=None, key='owner-1')
        summary = counter.record(threshold=2, window_seconds=3600.0, label='p', key='owner-2')

        assert summary is not None
        assert summary['count'] == 2
        assert summary['labels'] == ['p']

    def test_the_rate_limit_still_allows_exactly_one_fire(self):
        counter = StormCounter(count_distinct=True)

        fires = [
            counter.record(threshold=3, window_seconds=3600.0, label='p', key=f'owner-{i}')
            for i in range(200)
        ]
        assert sum(1 for f in fires if f is not None) == 1, (
            'a distinct-key burst must escalate ONCE per window too'
        )

    def test_keys_are_pruned_with_their_events(self, clock):
        """A key that has aged out of the window no longer counts."""
        counter = StormCounter(time_provider=clock, count_distinct=True)

        counter.record(threshold=3, window_seconds=100.0, label='p', key='owner-1')
        counter.record(threshold=3, window_seconds=100.0, label='p', key='owner-2')
        clock.advance(101.0)
        assert counter.record(
            threshold=3, window_seconds=100.0, label='p', key='owner-3'
        ) is None, 'the two aged-out keys must not still count toward the threshold'


class TestDistinctKeyOffByDefault:
    """``count_distinct`` defaults to False — the five existing consumers.

    None of markup_tripwire, memory_service, mcp_markup_middleware,
    escalation/server.py or orchestrator/harness.py passes a key, and all of
    them threshold on the raw event count. Passing a key to a default counter
    must therefore change nothing at all.
    """

    def test_default_counter_thresholds_on_raw_events_even_when_keyed(self, counter):
        assert counter.record(
            threshold=3, window_seconds=3600.0, label='p', key='same'
        ) is None
        assert counter.record(
            threshold=3, window_seconds=3600.0, label='p', key='same'
        ) is None
        summary = counter.record(threshold=3, window_seconds=3600.0, label='p', key='same')

        assert summary is not None, (
            'with count_distinct off, three events sharing one key are still '
            'three events — this is the pre-existing contract'
        )
        assert summary['count'] == 3, 'count stays the RAW event count by default'

    def test_default_counter_summary_shape_is_unchanged(self, counter):
        counter.record(threshold=2, window_seconds=3600.0, label='p')
        summary = counter.record(threshold=2, window_seconds=3600.0, label='p')

        assert summary == {
            'count': 2,
            'threshold': 2,
            'window_seconds': 3600.0,
            'labels': ['p'],
        }


class TestPruneReturnsRawCountInDistinctMode:
    """``prune()`` keeps reporting REMAINING EVENTS, never distinct keys.

    ``prune()`` is the emptiness probe ``MemoryService`` and
    ``MarkupGuardMiddleware`` sweep dormant counters with — they drop whatever
    returns ``0``. Its contract is "how much state is left", so it must not
    start answering a different question in distinct mode: a counter holding
    only ``key=None`` events has zero distinct keys but is NOT empty, and
    evicting it would silently discard live window state.
    """

    def test_prune_counts_events_not_keys(self):
        counter = StormCounter(count_distinct=True)
        for _ in range(3):
            counter.record(
                threshold=99, window_seconds=100.0, label='p', key='owner-1', now=1_000.0
            )

        assert counter.prune(100.0, now=1_000.0) == 3, (
            'three events sharing one key are three events to prune() — its '
            'contract is remaining STATE, not distinct keys'
        )

    def test_a_window_of_only_none_keys_is_not_reported_empty(self):
        counter = StormCounter(count_distinct=True)
        counter.record(threshold=99, window_seconds=100.0, label='p', key=None, now=1_000.0)

        assert counter.prune(100.0, now=1_000.0) == 1, (
            'zero distinct keys but one live event — a sweeper that evicted '
            'this counter would discard live window state'
        )
