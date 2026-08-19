"""Tests for orchestrator.log_coalesce.ExponentialLogCoalescer (task 3203).

step-1  RED   — unit tests for the not-yet-existing ExponentialLogCoalescer
step-2  GREEN — implement orchestrator/log_coalesce.py

The coalescer is PURE: no logging, no clock of its own, no I/O.  Every test
here drives it by advancing an injected ``now`` float, so the whole schedule
is exercised deterministically with zero sleeping.

``orchestrator.log_coalesce`` is imported LOCALLY inside each test (mirrors
test_merge_queue_resource_audit.py's convention — see that module's
docstring) so a not-yet-implemented symbol never breaks collection of the
rest of the file during the RED step.
"""

from __future__ import annotations

# Poll cadence used by every test that advances the clock. Matches
# merge_queue._HEARTBEAT_POLL_S so the derived `unchanged_secs` assertions
# read the way the production call site does.
_POLL_S = 30.0
_NOW = 1_000_000.0

# High enough that the cap never binds within any schedule exercised below —
# lets the pure doubling schedule be asserted in isolation from the cap.
_CAP_HIGH = 10_000


def _due_indices(coalescer, fingerprint, n: int, *, start: float = _NOW) -> list[int]:
    """Observe *fingerprint* *n* times and return the 1-based poll indices
    whose decision reported ``should_log``."""
    due: list[int] = []
    for i in range(n):
        decision = coalescer.observe(fingerprint, start + i * _POLL_S)
        if decision.should_log:
            due.append(i + 1)
    return due


class TestExponentialLogCoalescerSchedule:
    """(a)/(b)/(c) — first observation, the doubling schedule, and the cap."""

    def test_first_observation_of_any_fingerprint_logs_immediately(self) -> None:
        """(a) A fact nobody has seen yet is never withheld."""
        from orchestrator.log_coalesce import ExponentialLogCoalescer

        coalescer = ExponentialLogCoalescer(cap_polls=_CAP_HIGH)
        decision = coalescer.observe(('x',), _NOW)

        assert decision.should_log is True
        assert decision.changed is True
        assert decision.unchanged_polls == 1
        assert decision.unchanged_secs == 0.0

    def test_schedule_is_exponential_not_linear(self) -> None:
        """(b) 512 unchanged polls produce exactly 10 emissions, at the
        powers of two.  This is the property the whole task exists to buy."""
        from orchestrator.log_coalesce import ExponentialLogCoalescer

        coalescer = ExponentialLogCoalescer(cap_polls=_CAP_HIGH)
        due = _due_indices(coalescer, ('leak',), 512)

        assert due == [1, 2, 4, 8, 16, 32, 64, 128, 256, 512], (
            f'expected the powers of two, got {due}'
        )
        assert len(due) == 10
        # Stated explicitly: sub-linear, by more than an order of magnitude.
        assert len(due) < 512 / 10

    def test_cap_pins_the_interval_and_degrades_to_linear(self) -> None:
        """(c) With cap_polls=4 the interval doubles 1,2,4 then pins at 4."""
        from orchestrator.log_coalesce import ExponentialLogCoalescer

        coalescer = ExponentialLogCoalescer(cap_polls=4)
        due = _due_indices(coalescer, ('leak',), 40)

        assert due == [1, 2, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40], (
            f'expected doubling-then-capped-at-4, got {due}'
        )
        assert len(due) == 12


class TestExponentialLogCoalescerChange:
    """(d) — a fingerprint change restarts the schedule, never resumes it."""

    def test_change_mid_backoff_logs_immediately_and_restarts_the_schedule(self) -> None:
        from orchestrator.log_coalesce import ExponentialLogCoalescer

        coalescer = ExponentialLogCoalescer(cap_polls=_CAP_HIGH)
        # Drive A deep into the backoff: after 100 polls the next due poll is
        # 128, so a *resumed* schedule would withhold B for another 28 polls.
        for i in range(100):
            coalescer.observe(('a',), _NOW + i * _POLL_S)
        deep = coalescer.observe(('a',), _NOW + 100 * _POLL_S)
        assert deep.should_log is False, 'precondition: A must be mid-backoff'
        assert deep.next_report_in_polls > 1

        changed = coalescer.observe(('b',), _NOW + 101 * _POLL_S)
        assert changed.should_log is True
        assert changed.changed is True
        assert changed.unchanged_polls == 1
        assert changed.unchanged_secs == 0.0
        assert changed.next_report_in_polls == 1

        # The very next poll of B is due again — proving interval was reset to
        # 1 rather than resuming A's 32-poll interval.
        nxt = coalescer.observe(('b',), _NOW + 102 * _POLL_S)
        assert nxt.should_log is True
        assert nxt.changed is False
        assert nxt.unchanged_polls == 2

    def test_fingerprint_equality_is_by_value_not_identity(self) -> None:
        """(g) Two separately-built equal tuples are the SAME fingerprint —
        the production fingerprint is rebuilt from scratch every poll."""
        from orchestrator.log_coalesce import ExponentialLogCoalescer

        coalescer = ExponentialLogCoalescer(cap_polls=_CAP_HIGH)
        first = coalescer.observe(tuple(['v1', 'v2']), _NOW)
        second = coalescer.observe(tuple(['v1', 'v2']), _NOW + _POLL_S)

        assert first.changed is True
        assert second.changed is False, 'an equal-by-value fingerprint must not read as a change'
        assert second.unchanged_polls == 2


class TestExponentialLogCoalescerReportedContext:
    """(e) — the numbers a caller puts in the log line."""

    def test_unchanged_secs_is_measured_from_the_first_observation(self) -> None:
        from orchestrator.log_coalesce import ExponentialLogCoalescer

        coalescer = ExponentialLogCoalescer(cap_polls=_CAP_HIGH)
        for i in range(20):
            decision = coalescer.observe(('leak',), _NOW + i * _POLL_S)
            assert decision.unchanged_secs == i * _POLL_S
            assert decision.unchanged_polls == i + 1

    def test_next_report_in_polls_counts_down_to_the_next_due_poll(self) -> None:
        from orchestrator.log_coalesce import ExponentialLogCoalescer

        coalescer = ExponentialLogCoalescer(cap_polls=_CAP_HIGH)
        pending: list[int] = []
        for i in range(64):
            decision = coalescer.observe(('leak',), _NOW + i * _POLL_S)
            pending.append(decision.next_report_in_polls)
            # A due poll's countdown restarts from the new interval; every
            # countdown is strictly positive.
            assert decision.next_report_in_polls >= 1

        # Immediately after the change (poll 1) the next report is 1 poll away.
        assert pending[0] == 1
        # And each non-due poll decrements it by exactly one.
        for i in range(1, 64):
            if pending[i] != 1 and pending[i - 1] > 1:
                assert pending[i] == pending[i - 1] - 1, f'countdown broke at poll {i + 1}'

    def test_reported_countdown_predicts_the_actual_next_due_poll(self) -> None:
        from orchestrator.log_coalesce import ExponentialLogCoalescer

        coalescer = ExponentialLogCoalescer(cap_polls=_CAP_HIGH)
        promised: dict[int, int] = {}
        due: list[int] = []
        for i in range(128):
            decision = coalescer.observe(('leak',), _NOW + i * _POLL_S)
            promised[i + 1] = (i + 1) + decision.next_report_in_polls
            if decision.should_log:
                due.append(i + 1)

        # Every poll's promise names a poll that really is due.
        for poll, predicted in promised.items():
            if predicted <= 128:
                assert predicted in due, (
                    f'poll {poll} promised a report at poll {predicted}, which was not due'
                )


class TestExponentialLogCoalescerClear:
    """(f) — clear() is single-shot, which is what makes the caller's
    'exactly one clear line' property fall out of the gate."""

    def test_clear_reports_the_run_that_just_ended(self) -> None:
        from orchestrator.log_coalesce import ExponentialLogCoalescer

        coalescer = ExponentialLogCoalescer(cap_polls=_CAP_HIGH)
        for i in range(50):
            coalescer.observe(('leak',), _NOW + i * _POLL_S)

        cleared = coalescer.clear(_NOW + 50 * _POLL_S)
        assert cleared is not None
        assert cleared.polls == 50
        assert cleared.duration_secs == 50 * _POLL_S

    def test_clear_on_a_never_observed_coalescer_returns_none(self) -> None:
        from orchestrator.log_coalesce import ExponentialLogCoalescer

        coalescer = ExponentialLogCoalescer(cap_polls=_CAP_HIGH)
        assert coalescer.clear(_NOW) is None

    def test_clear_is_single_shot(self) -> None:
        """A clean poll following a clean poll cannot emit twice."""
        from orchestrator.log_coalesce import ExponentialLogCoalescer

        coalescer = ExponentialLogCoalescer(cap_polls=_CAP_HIGH)
        coalescer.observe(('leak',), _NOW)

        assert coalescer.clear(_NOW + _POLL_S) is not None
        for i in range(2, 12):
            assert coalescer.clear(_NOW + i * _POLL_S) is None

    def test_a_fresh_episode_after_a_clear_starts_from_scratch(self) -> None:
        from orchestrator.log_coalesce import ExponentialLogCoalescer

        coalescer = ExponentialLogCoalescer(cap_polls=_CAP_HIGH)
        for i in range(50):
            coalescer.observe(('leak',), _NOW + i * _POLL_S)
        coalescer.clear(_NOW + 50 * _POLL_S)

        # The SAME fingerprint returning is a new episode, not a continuation.
        again = coalescer.observe(('leak',), _NOW + 51 * _POLL_S)
        assert again.should_log is True
        assert again.changed is True
        assert again.unchanged_polls == 1
        assert again.unchanged_secs == 0.0
