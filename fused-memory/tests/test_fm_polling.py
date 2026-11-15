"""Tests for the bounded asyncio wait barriers in _fm_polling.py."""

import pytest

# ---------------------------------------------------------------------------
# Tests for the shared poll_until() helper (task 2377)
# ---------------------------------------------------------------------------
# Converts fixed "sleep(N) then assert background work is done" sites (fragile
# under host CPU oversubscription) into a bounded poll on the same
# post-condition. These tests pin the contract using ONLY predicate-call-count
# assertions and raise assertions — never wall-clock-elapsed assertions — so
# this suite stays load-independent and does not reintroduce the very
# fragility class poll_until exists to remove.

class TestPollUntil:
    """Unit tests for the shared poll_until(predicate, *, timeout, interval, message) helper."""

    @pytest.mark.asyncio
    async def test_already_true_sync_predicate_returns_after_one_call(self):
        """An already-true sync predicate returns immediately after exactly 1 call (check-before-sleep)."""
        from _fm_polling import poll_until

        calls = 0

        def predicate():
            nonlocal calls
            calls += 1
            return True

        result = await poll_until(predicate, timeout=5.0)

        assert result is True
        assert calls == 1, f'expected exactly 1 call, got {calls}'

    @pytest.mark.asyncio
    async def test_sync_predicate_flips_true_on_nth_call(self):
        """A sync predicate that flips true on the Nth call returns after exactly N calls.

        Proves check-before-sleep and that polling stops as soon as the
        post-condition holds, rather than over-polling or racing ahead.
        """
        from _fm_polling import poll_until

        calls = 0
        n = 4

        def predicate():
            nonlocal calls
            calls += 1
            return calls >= n

        result = await poll_until(predicate, timeout=5.0, interval=0.001)

        assert result is True
        assert calls == n, f'expected exactly {n} calls, got {calls}'

    @pytest.mark.asyncio
    async def test_async_predicate_is_awaited_and_flips_true_on_nth_call(self):
        """An async/coroutine predicate is awaited and behaves identically to the sync case.

        If poll_until failed to await the coroutine, the (always-truthy)
        coroutine object itself would satisfy the truthiness check on the
        first call and `calls` would remain 0 — so this also proves the
        coroutine body actually ran.
        """
        from _fm_polling import poll_until

        calls = 0
        n = 3

        async def predicate():
            nonlocal calls
            calls += 1
            return calls >= n

        result = await poll_until(predicate, timeout=5.0, interval=0.001)

        assert result is True
        assert calls == n, f'expected exactly {n} calls, got {calls}'

    @pytest.mark.asyncio
    async def test_return_value_is_truthy_predicate_result_verbatim(self):
        """poll_until returns the predicate's truthy result verbatim, not coerced to True.

        Lets callers chain on the returned value (e.g. a stats dict) instead
        of re-fetching state after the poll succeeds.
        """
        from _fm_polling import poll_until

        payload = {'done': True, 'value': 42}

        def predicate():
            return payload

        result = await poll_until(predicate, timeout=5.0)

        assert result is payload, f'expected the exact object {payload!r}, got {result!r}'

    @pytest.mark.asyncio
    async def test_never_true_predicate_raises_assertion_error_at_deadline(self):
        """A never-true predicate raises AssertionError once the deadline passes.

        Uses a tiny timeout and asserts only the raise (never elapsed time) —
        a slow host makes this test slower, never falsely passing or failing.
        """
        from _fm_polling import poll_until

        with pytest.raises(AssertionError):
            await poll_until(lambda: False, timeout=0.05, interval=0.01)

    @pytest.mark.asyncio
    async def test_caller_message_appears_in_raised_assertion_error(self):
        """A caller-supplied `message` appears verbatim in the raised AssertionError."""
        from _fm_polling import poll_until

        with pytest.raises(AssertionError, match='custom diagnostic xyz'):
            await poll_until(
                lambda: False, timeout=0.05, interval=0.01, message='custom diagnostic xyz',
            )

    @pytest.mark.asyncio
    async def test_default_message_names_the_timeout(self):
        """When message=None, the default AssertionError message names the timeout value."""
        from _fm_polling import poll_until

        with pytest.raises(AssertionError, match='0.05'):
            await poll_until(lambda: False, timeout=0.05, interval=0.01)


# ---------------------------------------------------------------------------
# Tests for the shared poll_until_dead() dead-letter waiter (task 4847)
# ---------------------------------------------------------------------------
# One waiter shared by the durable-queue suites. A scripted get_stats stands in
# for the queue, so these pin call counts and the timeout message only.

class _ScriptedStatsQueue:
    """Answers get_stats with the next scripted counts; the last entry repeats."""

    def __init__(self, counts_script: list[dict]):
        self._script = counts_script
        self.group_ids: list[str | None] = []

    async def get_stats(self, group_id=None):
        index = min(len(self.group_ids), len(self._script) - 1)
        self.group_ids.append(group_id)
        return {'counts': self._script[index]}


class TestPollUntilDead:
    """Unit tests for poll_until_dead(q, *, group_id, expected_dead, timeout)."""

    @pytest.mark.asyncio
    async def test_returns_once_dead_count_reaches_expected(self):
        from _fm_polling import poll_until_dead

        q = _ScriptedStatsQueue([{'pending': 2}, {'dead': 1, 'pending': 1}, {'dead': 2}])

        await poll_until_dead(q, group_id='g1', expected_dead=2, timeout=5.0)

        assert q.group_ids == ['g1', 'g1', 'g1']

    @pytest.mark.asyncio
    async def test_timeout_names_expected_count_group_and_last_counts(self):
        from _fm_polling import poll_until_dead

        q = _ScriptedStatsQueue([{'dead': 1, 'retry': 3}])

        with pytest.raises(AssertionError) as exc_info:
            await poll_until_dead(q, group_id='g1', expected_dead=2, timeout=0.05)

        message = str(exc_info.value)
        assert 'Timed out waiting for 2 dead item(s)' in message
        assert "group_id='g1'" in message
        assert "'retry': 3" in message

    @pytest.mark.asyncio
    async def test_timeout_reports_latest_snapshot_not_a_merge_of_past_polls(self):
        """get_stats GROUPs BY status, so a drained status vanishes from counts."""
        from _fm_polling import poll_until_dead

        q = _ScriptedStatsQueue([{'pending': 1}, {'dead': 1}])

        with pytest.raises(AssertionError) as exc_info:
            await poll_until_dead(q, group_id='g1', expected_dead=2, timeout=0.2)

        assert "last counts={'dead': 1}" in str(exc_info.value)


# ---------------------------------------------------------------------------
# Tests for the shared poll_until_stable() settle barrier (task 3697)
# ---------------------------------------------------------------------------
# poll_until is a LIVENESS barrier: it returns at the FIRST truthy sample. That
# makes it structurally unable to back an EXACT-COUNT assertion — a second event
# arriving milliseconds later is invisible, because the assertion already ran.
# poll_until_stable is the missing SETTLE barrier: it waits for the sampled
# value to STOP CHANGING before returning.
#
# Every value change below is driven off the SAMPLER'S CALL COUNT, never off
# wall clock, so this suite is load-independent and cannot itself become the
# next flake. (A settle window is safe in the same asymmetric way the helper
# documents: too short can only cause a false PASS, never a false FAIL.)

class TestPollUntilStable:
    """Unit tests for poll_until_stable(sample, *, settle, timeout, interval, message)."""

    @pytest.mark.asyncio
    async def test_poll_until_returns_first_value_while_stable_waits_for_the_settled_one(self):
        """THE FLAW, PINNED: a liveness poll observes only the first occurrence.

        This is the executable statement of why poll_until_stable exists. The
        motivating call site is
        fused-memory/tests/test_ticket_worker.py::test_worker_created_path_emits_journal_event,
        whose `assert len(task_created_events) == 1` was being gated on a
        `poll_until(lambda: any(...))` — so the assertion ran the instant the
        first event landed and a duplicate arriving milliseconds later was
        structurally invisible.

        Both samplers here are equivalent: the value goes 1 -> 2 on the SECOND
        call. poll_until returns 1 (it stopped looking); poll_until_stable
        returns 2 (it kept looking until the value stopped changing).
        """
        from _fm_polling import poll_until, poll_until_stable

        def make_sampler():
            calls = 0

            def sample():
                nonlocal calls
                calls += 1
                return 1 if calls == 1 else 2

            return sample

        liveness_result = await poll_until(make_sampler(), timeout=5.0, interval=0.001)
        settled_result = await poll_until_stable(
            make_sampler(), settle=0.05, timeout=5.0, interval=0.001,
        )

        assert liveness_result == 1, (
            f'poll_until should return the FIRST truthy value; got {liveness_result!r}'
        )
        assert settled_result == 2, (
            f'poll_until_stable should return the SETTLED value; got {settled_result!r}'
        )

    @pytest.mark.asyncio
    async def test_settle_window_restarts_when_the_value_changes(self):
        """A 1 -> 2 -> 2 -> 2... sampler returns 2, and polling continued past the first truthy sample.

        The call-count assertion is what proves the settle window is real: the
        value was already truthy on call 1, so a liveness poll would have
        stopped there. Reaching call 3 or beyond means the helper re-sampled,
        saw the change, and restarted its window.
        """
        from _fm_polling import poll_until_stable

        calls = 0

        def sample():
            nonlocal calls
            calls += 1
            return 1 if calls == 1 else 2

        result = await poll_until_stable(sample, settle=0.05, timeout=5.0, interval=0.001)

        assert result == 2, f'expected the settled value 2, got {result!r}'
        # First truthy sample was call 1; a settle barrier must sample again
        # (call 2, seeing the change) and once more to confirm (call 3+).
        assert calls >= 3, (
            f'expected the helper to keep polling past the first truthy sample; got {calls} calls'
        )

    @pytest.mark.asyncio
    async def test_already_stable_value_is_returned_verbatim(self):
        """An already-stable value settles promptly and is returned verbatim, not coerced to True.

        Matches poll_until's documented return contract so callers can chain on
        the returned object (e.g. a stats dict) instead of re-fetching state.
        """
        from _fm_polling import poll_until_stable

        payload = {'done': True, 'value': 42}

        result = await poll_until_stable(
            lambda: payload, settle=0.02, timeout=5.0, interval=0.001,
        )

        assert result is payload, f'expected the exact object {payload!r}, got {result!r}'

    @pytest.mark.asyncio
    async def test_never_ready_sampler_raises_with_the_caller_message(self):
        """A sampler that is always falsy raises AssertionError carrying the caller's message."""
        from _fm_polling import poll_until_stable

        with pytest.raises(AssertionError, match='worker never journalled anything'):
            await poll_until_stable(
                lambda: 0,
                settle=0.02,
                timeout=0.05,
                interval=0.01,
                message='worker never journalled anything',
            )

    @pytest.mark.asyncio
    async def test_never_stable_sampler_raises_a_distinguishable_message_naming_the_values(self):
        """A forever-changing value times out with a message distinct from the never-ready one.

        'It never happened' and 'it kept changing' demand different debugging.
        Collapsing them into one 'timed out' string is exactly the legibility
        failure this case exists to prevent, so assert on the distinguishing
        substring AND on an actually-observed value appearing in the trail —
        not merely on the exception type.
        """
        from _fm_polling import poll_until_stable

        observed = []

        def sample():
            observed.append(len(observed) + 1)
            return observed[-1]

        with pytest.raises(AssertionError) as excinfo:
            await poll_until_stable(sample, settle=0.05, timeout=0.2, interval=0.001)

        text = str(excinfo.value)
        assert 'never settled' in text, f'expected a never-settled diagnostic, got: {text!r}'
        assert str(observed[-1]) in text, (
            f'expected the observed-value trail to name {observed[-1]!r}, got: {text!r}'
        )

    @pytest.mark.asyncio
    async def test_async_sampler_is_awaited(self):
        """A coroutine-function sampler is awaited, mirroring poll_until's isawaitable contract.

        Pinned the same way the existing async poll_until test pins it: an
        un-awaited coroutine object is always truthy AND never equal to the
        previous one, so a helper that failed to await would never settle and
        would raise instead of returning 2.
        """
        from _fm_polling import poll_until_stable

        calls = 0

        async def sample():
            nonlocal calls
            calls += 1
            return 1 if calls == 1 else 2

        result = await poll_until_stable(sample, settle=0.05, timeout=5.0, interval=0.001)

        assert result == 2, f'expected the settled value 2, got {result!r}'
        assert calls >= 3, f'expected the coroutine body to run repeatedly; got {calls} calls'


# ---------------------------------------------------------------------------
# Tests for the shared retry_until_observed() observation barrier (task 4972)
# ---------------------------------------------------------------------------
# The third member of the barrier family, and the reason it exists: poll_until
# and poll_until_stable both assume the condition, once it arrives, stays
# observable long enough for a client read to catch it. A racy TRANSIENT breaks
# that assumption -- it can close before any client read lands, and then no
# interval and no deadline on a single attempt can help, because there is
# nothing left to observe. The only lever that acts on an absent window is to
# RE-OPEN it and look again.
#
# The motivating call site is
# tests/test_drop_vector_indices_integration.py::TestDropRebuildWindow, whose
# post-DROP rebuild window measured only 28/30 first-attempt observations under
# the offline lane's own `nice -n 19 ionice -c3` scheduling: on a miss FalkorDB
# had finished the rebuild INSIDE the DROP round-trip, so the very first
# post-DROP read (6.8 ms after DROP returned) already showed one settled
# OPERATIONAL row.
#
# Every assertion below is on CALL COUNTS and CALL ORDER against a shared
# append-only log -- never on wall clock -- so this suite is load-independent
# and cannot itself become the next flake, the same discipline TestPollUntil
# and TestPollUntilStable above already keep.

class TestRetryUntilObserved:
    """Unit tests for retry_until_observed(observe, *, reopen, attempts, message)."""

    @pytest.mark.asyncio
    async def test_sync_observation_on_the_first_attempt_returns_verbatim_and_never_reopens(self):
        """A first-attempt observation returns the value verbatim, at zero reopen cost.

        The no-miss path costing nothing is what makes this barrier safe to
        adopt in a fixture-backed live test: when the window IS open (the
        measured 28/30 case) the helper is indistinguishable from the single
        un-retried observation it replaces.

        Returning the observed value verbatim rather than coerced to True
        mirrors poll_until's documented return contract, so all three
        primitives in the family chain the same way.
        """
        from _fm_polling import retry_until_observed

        log: list[str] = []
        payload = {'rows': 2, 'phantom': True}

        def observe():
            log.append('observe')
            return payload

        def reopen():
            log.append('reopen')

        result = await retry_until_observed(observe, reopen=reopen, attempts=4)

        assert result is payload, f'expected the exact object {payload!r}, got {result!r}'
        assert log == ['observe'], f'expected exactly one observation and no reopen, got {log!r}'

    @pytest.mark.asyncio
    async def test_async_observation_on_the_first_attempt_is_awaited(self):
        """A coroutine-function `observe` is awaited, matching poll_until's isawaitable contract.

        Pinned the way the sibling async tests pin it: an un-awaited coroutine
        object is always truthy, so a helper that failed to await would still
        "succeed" on the first attempt -- but it would return the coroutine
        rather than the payload, and the log would stay empty because the
        coroutine body never ran.
        """
        from _fm_polling import retry_until_observed

        log: list[str] = []
        payload = {'rows': 2, 'phantom': True}

        async def observe():
            log.append('observe')
            return payload

        def reopen():
            log.append('reopen')

        result = await retry_until_observed(observe, reopen=reopen, attempts=4)

        assert result is payload, f'expected the exact object {payload!r}, got {result!r}'
        assert log == ['observe'], f'expected exactly one observation and no reopen, got {log!r}'

    @pytest.mark.asyncio
    async def test_two_misses_then_an_observation_reopens_exactly_between_attempts(self):
        """THE BEHAVIOUR THIS BARRIER EXISTS FOR: a missed window is RE-OPENED and observed again.

        `observe` misses on attempts 1 and 2, then observes on attempt 3. The
        ORDER assertion is the load-bearing one and is what call counts alone
        cannot express:

        * `reopen` is never called BEFORE the first attempt -- the caller has
          already opened the window once, and re-opening first would silently
          discard that opening and double the cost of the common case;
        * `reopen` is called exactly ONCE BETWEEN attempts -- an extra
          re-opening would leave the graph in a state the caller did not ask
          for, and a missing one would re-observe a window that already closed.
        """
        from _fm_polling import retry_until_observed

        log: list[str] = []
        sentinel = object()
        observations = 0

        def observe():
            nonlocal observations
            observations += 1
            log.append('observe')
            return sentinel if observations >= 3 else None

        def reopen():
            log.append('reopen')

        result = await retry_until_observed(observe, reopen=reopen, attempts=4)

        assert result is sentinel, f'expected the observation verbatim, got {result!r}'
        assert log == ['observe', 'reopen', 'observe', 'reopen', 'observe'], (
            f'expected reopen to fire exactly once BETWEEN attempts and never '
            f'before the first one, got {log!r}'
        )
        assert log.count('observe') == 3, f'expected exactly 3 observations, got {log!r}'
        assert log.count('reopen') == 2, f'expected exactly 2 reopens, got {log!r}'

    @pytest.mark.asyncio
    async def test_async_reopen_is_awaited_between_attempts(self):
        """A coroutine-function `reopen` is awaited, matching the sync/async symmetry of `observe`.

        Pinned the way the sibling async tests pin it: an un-awaited coroutine
        object is silently discarded, so a helper that failed to await would
        still produce the right CALL ORDER while never actually re-opening the
        window -- the log entry proves the coroutine BODY ran, not merely that
        the callable was invoked.
        """
        from _fm_polling import retry_until_observed

        log: list[str] = []
        sentinel = object()
        observations = 0

        async def observe():
            nonlocal observations
            observations += 1
            log.append('observe')
            return sentinel if observations >= 3 else None

        async def reopen():
            log.append('reopen')

        result = await retry_until_observed(observe, reopen=reopen, attempts=4)

        assert result is sentinel, f'expected the observation verbatim, got {result!r}'
        assert log == ['observe', 'reopen', 'observe', 'reopen', 'observe'], (
            f'expected the awaited reopen to fire exactly once between attempts, got {log!r}'
        )

    @pytest.mark.asyncio
    async def test_exhausting_every_attempt_raises_after_spending_the_whole_budget(self):
        """An always-missing `observe` raises AssertionError, having spent its whole budget.

        This is what preserves the motivating live test's PRIMARY durable job:
        fail loudly if FalkorDB ever stops opening the rebuild window at all.
        Returning None on exhaustion would let that failure be read as an
        ordinary miss and silently swallowed by the caller, leaving every
        post-drop barrier task 4748 added as undetected dead weight.

        AssertionError, not a bespoke type, matches the family convention --
        poll_until and poll_until_stable both raise AssertionError on timeout.

        The call-count assertions prove the helper spent its budget rather
        than giving up early, and that it did not re-arm a window nobody
        would read: exactly `attempts` observations and exactly
        `attempts - 1` reopens.
        """
        from _fm_polling import retry_until_observed

        log: list[str] = []

        def observe():
            log.append('observe')
            return None

        def reopen():
            log.append('reopen')

        with pytest.raises(AssertionError):
            await retry_until_observed(observe, reopen=reopen, attempts=5)

        assert log.count('observe') == 5, f'expected exactly 5 observations, got {log!r}'
        assert log.count('reopen') == 4, f'expected exactly 4 reopens, got {log!r}'
        assert log[-1] == 'observe', (
            f'expected the final act to be an observation, not a reopen nobody reads: {log!r}'
        )

    @pytest.mark.asyncio
    async def test_exhaustion_message_names_the_attempt_count(self):
        """The raised message names how many attempts were made.

        Without the count an operator cannot tell a one-shot miss from a
        budget genuinely spent, which is the whole distinction that makes
        exhaustion evidence the transient is GONE rather than hard to catch.
        """
        from _fm_polling import retry_until_observed

        # match=r'in 7 attempt\(s\)', not a bare '7': a lone digit matches
        # anywhere in the message -- a timeout value, a count of something
        # else, a future prefix -- so it would pass on a message that had
        # stopped naming the ATTEMPT COUNT, which is the only thing this test
        # exists to pin.
        with pytest.raises(AssertionError, match=r'in 7 attempt\(s\)'):
            await retry_until_observed(lambda: None, attempts=7)

    @pytest.mark.asyncio
    async def test_caller_message_survives_verbatim_into_the_exhaustion_error(self):
        """A caller-supplied `message` appears verbatim in the raised AssertionError.

        Matches poll_until's caller-message contract, and matters here for a
        concrete reason: the live call site's message is the "see this class
        docstring before deleting these barriers" guidance, and it must
        survive into the failure an operator actually reads.
        """
        from _fm_polling import retry_until_observed

        with pytest.raises(AssertionError, match='see this class docstring before deleting'):
            await retry_until_observed(
                lambda: None,
                attempts=2,
                message='see this class docstring before deleting them',
            )

    @pytest.mark.asyncio
    async def test_reopen_none_is_a_supported_no_op_re_arm(self):
        """`reopen=None` still makes exactly `attempts` observations and still raises.

        Supports a transient that re-arms ITSELF (an external producer, a
        periodic emitter): there is nothing for the caller to do between
        attempts, but the budget and the loud exhaustion must be identical.
        """
        from _fm_polling import retry_until_observed

        log: list[str] = []

        def observe():
            log.append('observe')
            return None

        with pytest.raises(AssertionError):
            await retry_until_observed(observe, attempts=3)

        assert log == ['observe', 'observe', 'observe'], (
            f'expected exactly 3 observations with no reopen, got {log!r}'
        )

    @pytest.mark.asyncio
    async def test_the_miss_path_always_yields_and_never_sleeps_after_the_final_miss(self, monkeypatch):
        """`delay` defaults to 0.0, but the miss path still AWAITS it -- so the loop always yields.

        Without this, a synchronous `observe` with `reopen=None` burns the
        whole attempt budget inside one event-loop tick: no wall-clock time
        passes and no other coroutine runs between attempts, so a barrier
        that gates nothing wears a plausible-looking exhaustion message. That
        directly contradicts the documented `reopen=None` use case -- a
        transient that re-arms ITSELF (an external producer, a periodic
        emitter) -- since nothing can possibly re-arm it in zero time.

        `await asyncio.sleep(0)` is the cheapest thing that fixes it: it costs
        one event-loop tick and hands control back, which is exactly what a
        self-re-arming producer needs to make progress.

        The sleep is inside the between-attempts guard, alongside `reopen`,
        for the same reason: sleeping after the FINAL miss is pure latency
        paid on the failure path for an attempt nobody will make.
        """
        import asyncio

        from _fm_polling import retry_until_observed

        log: list[str] = []
        real_sleep = asyncio.sleep

        async def recording_sleep(seconds):
            log.append(f'sleep:{seconds}')
            await real_sleep(0)

        monkeypatch.setattr(asyncio, 'sleep', recording_sleep)

        def observe():
            log.append('observe')
            return None

        with pytest.raises(AssertionError):
            await retry_until_observed(observe, attempts=3)

        assert log == ['observe', 'sleep:0.0', 'observe', 'sleep:0.0', 'observe'], (
            f'expected a yield between every pair of attempts and none after the last, got {log!r}'
        )

    @pytest.mark.asyncio
    async def test_an_explicit_delay_is_slept_between_attempts_after_reopen(self, monkeypatch):
        """An explicit `delay` is slept AFTER `reopen` and BEFORE the next observation.

        The order is the assertion: `reopen` opens the window, then the delay
        gives it time to develop, then it is observed. Sleeping BEFORE the
        re-open would spend the wait on a window that does not exist yet.

        `delay` is the knob the two sibling primitives already expose as
        `interval`; naming it differently here is deliberate, because it is
        not a poll cadence -- it is the gap between two independent openings.
        """
        import asyncio

        from _fm_polling import retry_until_observed

        log: list[str] = []
        real_sleep = asyncio.sleep
        payload = {'rows': 2}

        async def recording_sleep(seconds):
            log.append(f'sleep:{seconds}')
            await real_sleep(0)

        monkeypatch.setattr(asyncio, 'sleep', recording_sleep)

        results = [None, payload]

        def observe():
            log.append('observe')
            return results.pop(0)

        def reopen():
            log.append('reopen')

        result = await retry_until_observed(observe, reopen=reopen, attempts=4, delay=0.25)

        assert result is payload, f'expected the exact object {payload!r}, got {result!r}'
        assert log == ['observe', 'reopen', 'sleep:0.25', 'observe'], (
            f'expected the delay to fall between reopen and the next observation, got {log!r}'
        )

    @pytest.mark.asyncio
    async def test_an_exception_from_observe_propagates_instead_of_counting_as_a_miss(self):
        """A raising `observe` propagates on the spot; it is NOT absorbed as a falsy observation.

        This is the contract the motivating live call site's SECOND durable
        job rests on. There, `observe` reads `CALL db.indexes()` and raises
        IndexHeaderError when FalkorDB no longer exposes the label/types/status
        columns the barriers depend on. Absorbing that as an ordinary miss
        would spend the whole attempt budget re-opening a window nobody can
        read, and then report the FIRST durable job's diagnosis ("FalkorDB
        stopped rebuilding the merged index in place") for a completely
        different defect -- the two-modes-one-string collapse IndexHeaderError
        exists to prevent.

        Only a FALSY observation means "missed". Every other outcome is the
        caller's to see. Pinned here because a future edit wrapping the loop
        body in a broad try/except would leave every other test in this class
        green while silently deleting that guarantee.
        """
        from _fm_polling import retry_until_observed

        log: list[str] = []
        boom = RuntimeError('CALL db.indexes() changed shape')

        def observe():
            log.append('observe')
            raise boom

        def reopen():
            log.append('reopen')

        with pytest.raises(RuntimeError) as excinfo:
            await retry_until_observed(observe, reopen=reopen, attempts=5)

        assert excinfo.value is boom, f'expected the original exception, got {excinfo.value!r}'
        assert log == ['observe'], (
            f'expected the raise to abort on the FIRST attempt with no reopen, got {log!r}'
        )

    @pytest.mark.asyncio
    async def test_an_exception_from_reopen_aborts_the_retry_rather_than_being_retried_around(self):
        """A raising `reopen` propagates after the first miss; the retry does not continue past it.

        The live call site's `reopen` re-creates and re-drops a real index and
        holds two index-readiness barriers, any of which can raise. If those
        failures were swallowed and retried around, the helper would keep
        observing a window that was never successfully re-opened and then
        report exhaustion -- again the wrong diagnosis for the actual defect.
        `_reopen_rebuild_window`'s comment asserts this in prose; this test is
        what makes it checkable.
        """
        from _fm_polling import retry_until_observed

        log: list[str] = []
        boom = RuntimeError('could not re-create the vector index')

        def observe():
            log.append('observe')
            return None

        def reopen():
            log.append('reopen')
            raise boom

        with pytest.raises(RuntimeError) as excinfo:
            await retry_until_observed(observe, reopen=reopen, attempts=5)

        assert excinfo.value is boom, f'expected the original exception, got {excinfo.value!r}'
        assert log == ['observe', 'reopen'], (
            f'expected exactly one observation and one failed reopen, got {log!r}'
        )

    @pytest.mark.asyncio
    async def test_attempts_one_degenerates_to_a_single_un_retried_observation(self):
        """`attempts=1` observes exactly once, never reopens, and still raises on a miss.

        That is EXACTLY the semantics of the un-retried single observation
        this helper replaces at the live call site, which makes
        retry_until_observed a strict generalisation of that code rather than
        a behaviour change wearing a new name.
        """
        from _fm_polling import retry_until_observed

        log: list[str] = []

        def observe():
            log.append('observe')
            return None

        def reopen():
            log.append('reopen')

        with pytest.raises(AssertionError):
            await retry_until_observed(observe, reopen=reopen, attempts=1)

        assert log == ['observe'], (
            f'expected exactly one observation and no reopen, got {log!r}'
        )

    @pytest.mark.parametrize('bad_attempts', [0, -1])
    @pytest.mark.asyncio
    async def test_non_positive_attempts_raises_value_error_without_observing(self, bad_attempts):
        """THE LOUD-OVER-SILENT GUARD: `attempts` < 1 is a ValueError, raised before any observation.

        With a bare loop, `attempts=0` observes ZERO times and then raises the
        ordinary exhaustion AssertionError -- a barrier that gated nothing
        while reporting a plausible-looking failure. That is precisely the
        silent-no-op class await_index_operational's empty-`result_set` note
        already guards against in this same module.

        ValueError, deliberately NOT AssertionError: a caller that catches the
        exhaustion error to handle a genuine missed observation must not
        swallow a mis-configuration with it. The zero-call assertion is what
        proves the guard runs BEFORE the loop rather than being an
        after-the-fact relabelling of the exhaustion path.
        """
        from _fm_polling import retry_until_observed

        log: list[str] = []

        def observe():
            log.append('observe')
            return None

        with pytest.raises(ValueError, match='attempts'):
            await retry_until_observed(observe, attempts=bad_attempts)

        assert log == [], f'expected no observation at all, got {log!r}'
