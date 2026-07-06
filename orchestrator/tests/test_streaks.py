"""Tests for orchestrator.streaks — StreakCounter/StreakRegistry.

Consolidates the five hand-rolled "N consecutive ticks then fire, reset on
clean tick, GC on terminal" counters in scheduler.py behind a uniform
counting + GC abstraction.  Task 2124 (PRD task epsilon,
plans/supervision-quick-fixes-prd.md).

Registry owns COUNTING + GC ONLY — fire/escalate/block/resolve decisions
stay at scheduler.py call sites (`counter.value(key) >= threshold`).
"""

from __future__ import annotations

import pytest

from orchestrator.streaks import StreakCounter, StreakRegistry


class TestStreakCounterCountVariant:
    """Unit tests for the plain count variant: bump/value/clear/gc.

    This variant backs both ``_external_unresolved_counts`` (tuple
    ``(task_id, dep)`` keys) and ``_external_resolver_degraded_counts`` /
    ``_local_backfill_unresolved_counts`` (str or tuple keys) — the
    ``key_fn`` parameter is what lets one implementation serve both shapes.
    """

    def test_bump_increments_and_returns_new_count(self) -> None:
        counter = StreakCounter(threshold=3)
        assert counter.bump('a') == 1
        assert counter.bump('a') == 2
        assert counter.bump('a') == 3

    def test_value_defaults_to_zero(self) -> None:
        counter = StreakCounter(threshold=3)
        assert counter.value('missing') == 0

    def test_value_reads_current_count_without_mutating(self) -> None:
        counter = StreakCounter(threshold=3)
        counter.bump('a')
        counter.bump('a')
        assert counter.value('a') == 2
        assert counter.value('a') == 2  # reading twice must not bump

    def test_clear_pops_the_key(self) -> None:
        counter = StreakCounter(threshold=3)
        counter.bump('a')
        counter.clear('a')
        assert counter.value('a') == 0
        assert 'a' not in counter.counts

    def test_clear_missing_key_is_a_noop(self) -> None:
        counter = StreakCounter(threshold=3)
        counter.clear('missing')  # must not raise

    def test_supports_str_keys(self) -> None:
        counter = StreakCounter(threshold=3)
        assert counter.bump('task-1') == 1
        assert counter.bump('task-1') == 2

    def test_supports_tuple_keys(self) -> None:
        counter = StreakCounter(threshold=3, key_fn=lambda k: k[0])
        assert counter.bump(('task-1', 'dark_factory:5')) == 1
        assert counter.bump(('task-1', 'dark_factory:5')) == 2
        assert counter.value(('task-1', 'other-dep')) == 0

    def test_counts_is_identity_stable_across_calls(self) -> None:
        """The backing dict is the SAME object across calls so a caller
        (Scheduler.__init__) can alias a legacy attribute to it once and
        observe all future mutations in place — no rebind allowed.
        """
        counter = StreakCounter(threshold=3)
        backing = counter.counts
        counter.bump('a')
        counter.bump('b')
        assert counter.counts is backing
        assert backing == {'a': 1, 'b': 1}

    def test_gc_drops_keys_whose_task_id_is_stale_str_keys(self) -> None:
        counter = StreakCounter(threshold=3)
        counter.bump('task-1')
        counter.bump('task-2')
        counter.gc({'task-1'})
        assert counter.value('task-1') == 0
        assert counter.value('task-2') == 1

    def test_gc_drops_keys_whose_task_id_is_stale_tuple_keys(self) -> None:
        counter = StreakCounter(threshold=3, key_fn=lambda k: k[0])
        counter.bump(('task-1', 'dep-a'))
        counter.bump(('task-1', 'dep-b'))
        counter.bump(('task-2', 'dep-a'))
        counter.gc({'task-1'})
        assert counter.value(('task-1', 'dep-a')) == 0
        assert counter.value(('task-1', 'dep-b')) == 0
        assert counter.value(('task-2', 'dep-a')) == 1

    def test_gc_mutates_counts_in_place(self) -> None:
        counter = StreakCounter(threshold=3)
        backing = counter.counts
        counter.bump('task-1')
        counter.gc({'task-1'})
        assert counter.counts is backing
        assert backing == {}

    def test_gc_leaves_unrelated_keys_untouched(self) -> None:
        counter = StreakCounter(threshold=3)
        counter.bump('task-1')
        counter.bump('task-2')
        counter.bump('task-3')
        counter.gc({'task-2'})
        assert counter.value('task-1') == 1
        assert counter.value('task-2') == 0
        assert counter.value('task-3') == 1


class TestStreakCounterCauseChangeVariant:
    """Unit tests for the cause-change-reset variant: touch(key, cause=...).

    Collapses the ``_external_hold_streak`` / ``_external_hold_cause`` pair
    into one counter.  Mirrors ``Scheduler._note_external_hold``: the streak
    resets to 0 (then bumps to 1) whenever the cause differs from the
    last-seen cause for that key; it keeps incrementing across ticks that
    repeat the same cause.
    """

    def test_touch_first_call_returns_one(self) -> None:
        counter = StreakCounter(threshold=3)
        assert counter.touch('task-1', cause='deps_live') == 1

    def test_touch_same_cause_increments(self) -> None:
        counter = StreakCounter(threshold=3)
        assert counter.touch('task-1', cause='deps_live') == 1
        assert counter.touch('task-1', cause='deps_live') == 2
        assert counter.touch('task-1', cause='deps_live') == 3

    def test_touch_cause_change_resets_to_one(self) -> None:
        counter = StreakCounter(threshold=3)
        counter.touch('task-1', cause='deps_live')
        counter.touch('task-1', cause='deps_live')
        assert counter.touch('task-1', cause='deps_live') == 3
        # Cause changes tick-over-tick — streak drops back to 1, not 4.
        assert counter.touch('task-1', cause='resolver_degraded') == 1

    def test_touch_records_last_seen_cause(self) -> None:
        counter = StreakCounter(threshold=3)
        counter.touch('task-1', cause='deps_live')
        assert counter.causes['task-1'] == 'deps_live'
        counter.touch('task-1', cause='resolver_degraded')
        assert counter.causes['task-1'] == 'resolver_degraded'

    def test_value_reads_the_streak_after_touch(self) -> None:
        counter = StreakCounter(threshold=3)
        counter.touch('task-1', cause='deps_live')
        counter.touch('task-1', cause='deps_live')
        assert counter.value('task-1') == 2

    def test_counts_and_causes_are_identity_stable_across_calls(self) -> None:
        counter = StreakCounter(threshold=3)
        counts_backing = counter.counts
        causes_backing = counter.causes
        counter.touch('task-1', cause='deps_live')
        assert counter.counts is counts_backing
        assert counter.causes is causes_backing
        assert causes_backing == {'task-1': 'deps_live'}

    def test_clear_pops_both_counts_and_causes(self) -> None:
        counter = StreakCounter(threshold=3)
        counter.touch('task-1', cause='deps_live')
        counter.clear('task-1')
        assert 'task-1' not in counter.counts
        assert 'task-1' not in counter.causes
        assert counter.value('task-1') == 0

    def test_clear_missing_key_is_a_noop_for_cause_variant(self) -> None:
        counter = StreakCounter(threshold=3)
        counter.clear('missing')  # must not raise

    def test_gc_drops_key_from_both_containers_in_place(self) -> None:
        counter = StreakCounter(threshold=3)
        counts_backing = counter.counts
        causes_backing = counter.causes
        counter.touch('task-1', cause='deps_live')
        counter.touch('task-2', cause='deps_live')
        counter.gc({'task-1'})
        assert counter.counts is counts_backing
        assert counter.causes is causes_backing
        assert 'task-1' not in counter.counts
        assert 'task-1' not in counter.causes
        assert counter.value('task-2') == 1
        assert counter.causes['task-2'] == 'deps_live'


class TestStreakCounterFirstSeenAgeVariant:
    """Unit tests for the first-seen-age variant: touch(key, now=)/age().

    Backs ``_starvation_first_seen`` / ``_starvation_escalated``: the
    starvation watchdog stamps an anchor the first tick a task appears as a
    dispatch-eligible candidate, then measures elapsed idle time against
    that anchor on every subsequent tick until the task dispatches or is
    GC'd.
    """

    def test_touch_stamps_first_seen_on_first_sight(self) -> None:
        counter = StreakCounter(threshold=3)
        assert counter.touch('task-1', now=100.0) == 100.0
        assert counter.first_seen['task-1'] == 100.0

    def test_touch_is_idempotent_after_first_sight(self) -> None:
        counter = StreakCounter(threshold=3)
        counter.touch('task-1', now=100.0)
        # A later tick's `now` must NOT overwrite the anchor.
        assert counter.touch('task-1', now=200.0) == 100.0
        assert counter.first_seen['task-1'] == 100.0

    def test_absence_of_touch_does_not_auto_stamp(self) -> None:
        counter = StreakCounter(threshold=3)
        assert 'task-1' not in counter.first_seen

    def test_age_returns_elapsed_since_first_seen(self) -> None:
        counter = StreakCounter(threshold=3)
        counter.touch('task-1', now=100.0)
        assert counter.age('task-1', now=145.0) == 45.0

    def test_first_seen_is_identity_stable_across_calls(self) -> None:
        counter = StreakCounter(threshold=3)
        backing = counter.first_seen
        counter.touch('task-1', now=100.0)
        assert counter.first_seen is backing
        assert backing == {'task-1': 100.0}

    def test_mark_escalated_and_is_escalated(self) -> None:
        counter = StreakCounter(threshold=3)
        assert counter.is_escalated('task-1') is False
        counter.mark_escalated('task-1')
        assert counter.is_escalated('task-1') is True

    def test_escalated_is_identity_stable_across_calls(self) -> None:
        counter = StreakCounter(threshold=3)
        backing = counter.escalated
        counter.mark_escalated('task-1')
        assert counter.escalated is backing
        assert backing == {'task-1'}

    def test_clear_pops_first_seen_and_discards_escalated(self) -> None:
        counter = StreakCounter(threshold=3)
        counter.touch('task-1', now=100.0)
        counter.mark_escalated('task-1')
        counter.clear('task-1')
        assert 'task-1' not in counter.first_seen
        assert not counter.is_escalated('task-1')

    def test_clear_missing_key_is_a_noop_for_age_variant(self) -> None:
        counter = StreakCounter(threshold=3)
        counter.clear('missing')  # must not raise, incl. escalated.discard

    def test_gc_sweeps_both_first_seen_and_escalated_in_place(self) -> None:
        counter = StreakCounter(threshold=3)
        first_seen_backing = counter.first_seen
        escalated_backing = counter.escalated
        counter.touch('task-1', now=100.0)
        counter.mark_escalated('task-1')
        counter.touch('task-2', now=100.0)
        counter.mark_escalated('task-2')
        counter.gc({'task-1'})
        assert counter.first_seen is first_seen_backing
        assert counter.escalated is escalated_backing
        assert 'task-1' not in counter.first_seen
        assert not counter.is_escalated('task-1')
        assert 'task-2' in counter.first_seen
        assert counter.is_escalated('task-2')


class TestStreakRegistry:
    """Unit tests for StreakRegistry: register(name, counter, on_gc=None) +
    ``await gc(stale_ids, extra=None)``.

    The registry owns GC ONLY — it replaces the manual per-dict enumeration
    in ``acquire_next`` with one consolidated sweep over every registered
    counter, including the async starvation on_gc/resolve callback.
    """

    @pytest.mark.asyncio
    async def test_gc_sweeps_the_registered_counter(self) -> None:
        registry = StreakRegistry()
        counter = StreakCounter(threshold=3)
        registry.register('unresolved', counter)
        counter.bump('task-1')
        counter.bump('task-2')

        await registry.gc({'task-1'})

        assert counter.value('task-1') == 0
        assert counter.value('task-2') == 1

    @pytest.mark.asyncio
    async def test_gc_sweeps_every_registered_counter(self) -> None:
        registry = StreakRegistry()
        counter_a = StreakCounter(threshold=3)
        counter_b = StreakCounter(threshold=3)
        registry.register('a', counter_a)
        registry.register('b', counter_b)
        counter_a.bump('task-1')
        counter_b.bump('task-1')

        await registry.gc({'task-1'})

        assert counter_a.value('task-1') == 0
        assert counter_b.value('task-1') == 0

    @pytest.mark.asyncio
    async def test_registered_but_forgotten_counter_is_still_swept(self) -> None:
        """A counter no call site references again after registration is
        still GC'd — the registry, not manual per-dict enumeration at the
        call site, owns the sweep."""
        registry = StreakRegistry()
        forgotten = StreakCounter(threshold=3)
        registry.register('forgotten', forgotten)
        forgotten.bump('task-1')

        await registry.gc({'task-1'})

        assert forgotten.value('task-1') == 0

    @pytest.mark.asyncio
    async def test_extra_unions_additional_ids_for_the_named_counter_only(self) -> None:
        registry = StreakRegistry()
        starvation = StreakCounter(threshold=3)
        other = StreakCounter(threshold=3)
        registry.register('starvation', starvation)
        registry.register('other', other)
        starvation.touch('escalated-task', now=100.0)
        starvation.mark_escalated('escalated-task')
        other.bump('escalated-task')

        # 'escalated-task' is not in stale_ids (not terminal) but IS passed
        # via extra['starvation'] (the non-eligible-status sweep) — only the
        # 'starvation' counter is swept for it; 'other' is untouched.
        await registry.gc(set(), extra={'starvation': {'escalated-task'}})

        assert 'escalated-task' not in starvation.first_seen
        assert other.value('escalated-task') == 1

    @pytest.mark.asyncio
    async def test_on_gc_awaited_once_per_cleared_key_present_in_escalated(self) -> None:
        registry = StreakRegistry()
        starvation = StreakCounter(threshold=3)
        resolved: list[str] = []

        async def on_gc(key: str) -> None:
            resolved.append(key)

        registry.register('starvation', starvation, on_gc=on_gc)
        starvation.touch('task-1', now=100.0)
        starvation.mark_escalated('task-1')
        starvation.touch('task-2', now=100.0)  # terminal but never escalated

        await registry.gc({'task-1', 'task-2'})

        # task-2 was never escalated — no resolve callback for it, but it is
        # still cleared (mirrors the unconditional _starvation_first_seen.pop
        # alongside the conditional resolve-on-escalated in scheduler.py).
        assert resolved == ['task-1']
        assert 'task-1' not in starvation.first_seen
        assert not starvation.is_escalated('task-1')
        assert 'task-2' not in starvation.first_seen

    @pytest.mark.asyncio
    async def test_on_gc_not_invoked_for_ids_outside_the_sweep(self) -> None:
        registry = StreakRegistry()
        starvation = StreakCounter(threshold=3)
        resolved: list[str] = []

        async def on_gc(key: str) -> None:
            resolved.append(key)

        registry.register('starvation', starvation, on_gc=on_gc)
        starvation.touch('task-1', now=100.0)
        starvation.mark_escalated('task-1')

        await registry.gc({'task-2'})  # task-1 is not stale

        assert resolved == []
        assert starvation.is_escalated('task-1')

    @pytest.mark.asyncio
    async def test_zero_entries_in_any_registered_counter_after_gc(self) -> None:
        """The 'zero entries in ANY registered counter' invariant."""
        registry = StreakRegistry()
        unresolved = StreakCounter(threshold=3, key_fn=lambda k: k[0])
        hold = StreakCounter(threshold=3)
        starvation = StreakCounter(threshold=3)
        registry.register('unresolved', unresolved)
        registry.register('hold', hold)
        registry.register('starvation', starvation)

        unresolved.bump(('task-1', 'dep-a'))
        hold.touch('task-1', cause='deps_live')
        starvation.touch('task-1', now=100.0)
        starvation.mark_escalated('task-1')

        await registry.gc({'task-1'})

        assert unresolved.value(('task-1', 'dep-a')) == 0
        assert hold.value('task-1') == 0
        assert 'task-1' not in hold.causes
        assert 'task-1' not in starvation.first_seen
        assert not starvation.is_escalated('task-1')
