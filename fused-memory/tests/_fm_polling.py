"""Bounded asyncio wait barriers that replace fixed sleeps in tests.

Three barrier primitives -- poll_until (liveness), poll_until_stable (settle)
and retry_until_observed (observation) -- plus poll_until_dead, the
durable-queue dead-count waiter built on poll_until.

Lives outside conftest.py, imported by name, under the convention stated in
_fm_helpers' module docstring.
"""

import asyncio
import inspect
from collections.abc import Callable
from typing import Any, Protocol

# ---------------------------------------------------------------------------
# Shared poll_until() helper (task 2377)
# ---------------------------------------------------------------------------
#
# Generalizes the local poll-then-assert precedents in test_durable_queue.py
# (the dead-letter waiter, now poll_until_dead below, and the inline deadline
# loop) into a single shared helper. Converts fixed "sleep(N) then assert
# background work is done" sites — fragile under host CPU oversubscription,
# since the hardcoded window may elapse before the background
# drainer/worker/watchdog is even scheduled — into a bounded poll on the same
# post-condition. A genuinely stuck background task still surfaces as a clear
# AssertionError once the (generous) deadline passes.
# ---------------------------------------------------------------------------

async def poll_until(
    predicate: Callable[[], Any],
    *,
    timeout: float = 10.0,
    interval: float = 0.02,
    message: str | None = None,
) -> Any:
    """Poll *predicate* until it is truthy, returning its truthy result verbatim.

    *predicate* may be a plain sync callable or an async/coroutine function —
    either way it is called with no arguments on each iteration; if the
    result is awaitable (``inspect.isawaitable``), it is awaited before the
    truthiness check. This lets callers poll on state that itself requires an
    await (e.g. ``lambda: queue.get_stats()``).

    Checks *before* sleeping, so an already-true predicate returns on the
    first call with no sleep at all. Uses the asyncio event-loop monotonic
    clock rather than wall-clock/perf_counter, to avoid the same class of
    timing fragility this helper exists to remove.

    Args:
        predicate: Zero-arg sync or async callable evaluated each iteration.
        timeout: Seconds to keep polling before giving up. Defaults to 10s —
            generous enough to absorb host CPU oversubscription while still
            catching a genuinely-stuck post-condition.
        interval: Seconds to sleep between polls. Defaults to 0.02s.
        message: Custom AssertionError message on timeout. When None, a
            default message naming *timeout* is used.

    Returns:
        The predicate's truthy result, returned verbatim (not coerced to
        ``True``) so callers can chain on it (e.g. a stats dict).

    Raises:
        AssertionError: *predicate* never became truthy within *timeout*
            seconds — signals a genuinely-stuck background task, the same
            failure class the fixed-sleep asserts this helper replaces used
            to (accidentally) surface.
    """
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while True:
        result = predicate()
        if inspect.isawaitable(result):
            result = await result
        if result:
            return result
        if loop.time() >= deadline:
            raise AssertionError(message or f'poll_until: condition not met within {timeout}s')
        await asyncio.sleep(interval)


# ---------------------------------------------------------------------------
# Shared durable-queue dead-letter waiter (task 4847)
# ---------------------------------------------------------------------------

class SupportsQueueStats(Protocol):
    """The one DurableWriteQueue member poll_until_dead reads."""

    async def get_stats(self, group_id: str | None = None) -> dict[str, Any]: ...


async def poll_until_dead(
    q: SupportsQueueStats,
    *,
    group_id: str | None = None,
    expected_dead: int,
    timeout: float = 20.0,
) -> None:
    """Poll queue stats until at least *expected_dead* items have status='dead'.

    Poll-based, never a fixed sleep: the suites run under ``-n auto`` with a
    60s per-test timeout that kills the whole worker, so a fixed window either
    wastes time on a fast host or races on a loaded one. The 20s default
    absorbs that load while staying well under the per-test timeout.
    last_counts is a side-channel capture so the timeout message can still
    report the latest observed counts, since poll_until's message is static.
    """
    last_counts: dict = {}

    async def _dead_count_reached():
        nonlocal last_counts
        last_counts = (await q.get_stats(group_id=group_id))['counts']
        return last_counts.get('dead', 0) >= expected_dead

    try:
        await poll_until(_dead_count_reached, timeout=timeout, interval=0.05)
    except AssertionError as exc:
        raise AssertionError(
            f'Timed out waiting for {expected_dead} dead item(s) '
            f'(group_id={group_id!r}); last counts={last_counts}'
        ) from exc


# ---------------------------------------------------------------------------
# Shared poll_until_stable() settle barrier (task 3697)
# ---------------------------------------------------------------------------
#
# The companion to poll_until above, and deliberately placed beside it so the
# two read as a matched PAIR of barrier primitives rather than two unrelated
# utilities. poll_until is a LIVENESS barrier; this is a SETTLE barrier. Picking
# the wrong one is the defect this exists to prevent -- see the docstring's
# decision rule. Public (no leading underscore) for the reason this module
# already documents for poll_until / ensure_fresh_collection: it is imported
# cross-module.
# ---------------------------------------------------------------------------

#: Cap on the observed-value trail carried in the never-settled AssertionError.
#: A pathological sampler must not be able to produce a multi-megabyte message.
_STABLE_TRAIL_LIMIT = 10


async def poll_until_stable(
    sample: Callable[[], Any],
    *,
    settle: float = 0.2,
    timeout: float = 10.0,
    interval: float = 0.02,
    message: str | None = None,
) -> Any:
    """Poll *sample* until its truthy value STOPS CHANGING, then return it verbatim.

    Choosing between this and :func:`poll_until` — the whole point of having
    both:

    * ``poll_until`` is a **liveness** barrier. Correct when the assertion that
      follows is "X happened".
    * ``poll_until_stable`` is a **settle** barrier. **Required** when the
      assertion that follows is an exact count or an "exactly once" claim,
      because a liveness poll returns at the *first* occurrence and therefore
      structurally cannot observe a duplicate that arrives after it.

    The motivating call site is
    ``test_ticket_worker.py::test_worker_created_path_emits_journal_event``,
    whose ``assert len(task_created_events) == 1`` was gated on
    ``poll_until(lambda: any(...))`` — so the assertion ran the instant the
    first event landed. The duplicate that assertion exists to catch is
    concrete, not hypothetical: ``task_created`` is emitted from two distinct
    code paths (``task_interceptor.py`` lines 3786-3795 and 4164-4173), each
    persisting the terminal ticket row *before* emitting — so no ticket-row
    predicate closes the emission window either, and only a settle window does.

    **The flake asymmetry**, which is what makes adding a settle window safe: a
    too-short *settle* can only cause a false PASS (a missed duplicate), never
    a false FAIL, because a stable value stays stable no matter how starved the
    host is. The only false-FAIL surface is the outer *timeout*, which keeps
    ``poll_until``'s already-proven 10s default. Raising *settle* is therefore
    always safe, and is the correct response to a suspected missed duplicate.

    *sample* may be a plain sync callable or an async/coroutine function —
    either way it is called with no arguments each iteration and, if the result
    is awaitable (``inspect.isawaitable``), awaited before use. A falsy result
    means "not ready yet": polling continues and the settle window does not
    start. Change detection uses ``!=``, and the deadline uses the asyncio
    event-loop monotonic clock (same rationale as ``poll_until``), bounding
    total time *including* settle restarts so a forever-changing value cannot
    spin past *timeout*.

    Args:
        sample: Zero-arg sync or async callable evaluated each iteration.
        settle: Seconds the value must be observed unchanged before it is
            returned. Any change restarts this window. Defaults to 0.2s.
        timeout: Total seconds to keep polling before giving up. Defaults to
            10s, matching ``poll_until``.
        interval: Seconds to sleep between polls. Defaults to 0.02s.
        message: Custom AssertionError message used when the value never
            became truthy at all.

    Returns:
        The settled truthy value, returned verbatim (not coerced to ``True``),
        matching ``poll_until``.

    Raises:
        AssertionError: with two deliberately DISTINGUISHABLE messages, because
            "it never happened" and "it kept changing" demand different
            debugging and must not collapse into one string —

            * never became truthy → the caller's *message* (or a default
              naming *timeout*), matching ``poll_until``'s failure text so the
              common case reads identically;
            * became truthy but never settled → a message saying so and
              naming the observed values (capped at the last
              ``_STABLE_TRAIL_LIMIT`` distinct ones).
    """
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    candidate: Any = None
    have_candidate = False
    stable_since = 0.0
    trail: list[Any] = []
    while True:
        result = sample()
        if inspect.isawaitable(result):
            result = await result
        if result:
            now = loop.time()
            if not have_candidate or result != candidate:
                candidate = result
                have_candidate = True
                stable_since = now
                trail.append(result)
                del trail[:-_STABLE_TRAIL_LIMIT]
            elif now - stable_since >= settle:
                return candidate
        else:
            # Not ready yet — a falsy sample does not start the settle window.
            have_candidate = False
        if loop.time() >= deadline:
            if trail:
                raise AssertionError(
                    f'poll_until_stable: value never settled for {settle}s within '
                    f'{timeout}s; observed (last {len(trail)}): {trail!r}'
                )
            raise AssertionError(
                message or f'poll_until_stable: condition not met within {timeout}s'
            )
        await asyncio.sleep(interval)


# ---------------------------------------------------------------------------
# Shared retry_until_observed() observation barrier (task 4972)
# ---------------------------------------------------------------------------
#
# The THIRD member of the barrier family above, placed beside its two siblings
# for the same reason poll_until_stable was placed beside poll_until: picking
# the wrong one is the defect this exists to prevent -- see the docstring's
# decision rule. Public (no leading underscore) for the reason this module
# already documents for poll_until / poll_until_stable: it is imported
# cross-module.
# ---------------------------------------------------------------------------

async def retry_until_observed(
    observe: Callable[[], Any],
    *,
    reopen: Callable[[], Any] | None = None,
    attempts: int = 4,
    delay: float = 0.0,
    message: str | None = None,
) -> Any:
    """Observe a RACY TRANSIENT, RE-OPENING it between attempts, and return what was seen.

    Choosing between this and its two siblings — the whole point of having
    three:

    * :func:`poll_until` is a **liveness** barrier. Correct when the assertion
      that follows is "X happened".
    * :func:`poll_until_stable` is a **settle** barrier. Required when the
      assertion that follows is an exact count or an "exactly once" claim.
    * ``retry_until_observed`` is an **observation** barrier. **Required** when
      the condition is a racy TRANSIENT that may close before any client read
      can see it — the case both siblings structurally cannot handle, because
      each only ever looks at ONE opening of the window.

    The discriminator, stated plainly: reach for this only when a miss means
    the window never opened *for this attempt*, so no interval and no deadline
    on a single attempt can help — there is nothing left to observe. The only
    lever that acts on an absent window is to open a new one, which is what
    *reopen* does.

    The motivating call site is
    ``tests/test_drop_vector_indices_integration.py::TestDropRebuildWindow``,
    which observes the rebuild window FalkorDB opens when ``DROP VECTOR INDEX``
    leaves a merged index's other fields behind. Measured under the offline
    lane's own scheduling (``nice -n 19 ionice -c3``, serial): the window was
    observable on 28 of 30 first attempts. On a miss it was not merely hard to
    catch — the first post-DROP read, taken 6.8 ms after ``DROP`` returned,
    already showed ONE Entity row, already ``OPERATIONAL``, and it stayed that
    way for the whole deadline. FalkorDB had completed the rebuild inside the
    ``DROP`` round-trip; the window never opened at all.

    *observe* and *reopen* may each be a plain sync callable or an
    async/coroutine function — either way called with no arguments and, if the
    result is awaitable (``inspect.isawaitable``), awaited before use, matching
    ``poll_until``'s contract exactly so all three primitives accept the same
    callable shapes. A falsy *observe* result means "the window was not
    observed on this attempt". Only a FALSY result counts as a miss —
    an exception from either callable propagates untouched, so a caller
    reading a live system can still tell "the shape changed" apart from
    "the window closed too fast" (the motivating call site's second durable
    job depends on exactly that).

    WHAT MAKES AN ATTEMPT AN ATTEMPT. Nothing here re-observes faster than
    the caller lets it, so each attempt must be able to see a window that
    was not open at the instant the previous one gave up. Satisfy that in
    one of two ways: make *observe* itself a TIME-BOUNDED awaiting attempt
    (typically a bounded :func:`poll_until`, which is what the motivating
    call site does), or pass a non-zero *delay*. A synchronous *observe*
    with ``reopen=None`` and ``delay=0.0`` would otherwise spend its whole
    budget inside a single event-loop tick — a barrier that gated nothing
    while wearing a plausible exhaustion message. The miss path always
    awaits ``asyncio.sleep(delay)`` even at the 0.0 default, so the loop at
    minimum YIELDS between attempts rather than starving the event loop.

    Args:
        observe: Zero-arg sync or async callable making ONE bounded attempt to
            observe the transient. Returns a truthy observation, or falsy for a
            miss. Typically itself a bounded ``poll_until``.
        reopen: Zero-arg sync or async callable that re-opens the transient,
            called only BETWEEN attempts — never before the first (the caller
            has already opened the window) and never after the last (nobody
            would read it). ``None`` means the transient re-arms itself, and
            is a supported no-op: the attempt budget and the exhaustion raise
            are unchanged.
        attempts: Maximum number of observation attempts. Must be >= 1;
            ``attempts=1`` degenerates to a single un-retried observation.
        delay: Seconds slept BETWEEN attempts — after *reopen*, before the
            next observation, so the wait is spent on a window that exists
            rather than one not yet opened. The sibling primitives spell
            their cadence knob ``interval``; this one is deliberately named
            differently because it is not a poll cadence but the gap between
            two INDEPENDENT openings. Awaited even at the 0.0 default (see
            above) and, like *reopen*, never after the final miss.
        message: Caller diagnostic appended to the exhaustion AssertionError.

    Returns:
        The observation, returned verbatim (not coerced to ``True``), matching
        ``poll_until`` / ``poll_until_stable``.

    Raises:
        AssertionError: The transient was not observed in any of *attempts*
            independent openings — the message names the attempt count and
            carries the caller's *message*. This helper RAISES rather than
            returning ``None`` because exhausting the budget is the one signal
            it must never swallow: it is the evidence that the transient is
            genuinely GONE rather than merely hard to catch. At the motivating
            call site that distinction IS the test's primary durable job — a
            FalkorDB that stopped rebuilding the merged index in place would
            make every post-drop barrier task 4748 added dead weight, and a
            silent ``None`` would let that pass as an ordinary miss.
        ValueError: *attempts* was less than 1 — raised before *observe* is
            called even once, so a mis-configured budget can never masquerade
            as the exhaustion AssertionError above.
    """
    # Fail CLOSED on a mis-configured budget, before observing anything: with a
    # bare loop, attempts=0 would make zero observations and then raise the
    # ordinary exhaustion AssertionError below -- a barrier that gated nothing
    # while wearing a plausible-looking failure message. Same argument
    # await_index_operational makes for treating an empty result_set as NOT
    # ready. ValueError (not AssertionError) so a caller catching a genuine
    # missed observation cannot swallow a mis-configuration with it.
    if attempts < 1:
        raise ValueError(f'retry_until_observed: attempts must be >= 1, got {attempts!r}')
    for attempt in range(1, attempts + 1):
        result = observe()
        if inspect.isawaitable(result):
            result = await result
        if result:
            return result
        # Re-open, then wait, ONLY between attempts:
        #   * not before attempt 1 -- the caller has already opened the window
        #     once, and re-opening first would silently discard that opening
        #     and double the cost of the common (observed-first-try) case;
        #   * not after the final miss -- re-arming a window nobody will look
        #     at is pure waste, and would leave the caller's subject in a
        #     different state than it expects on the failure path; sleeping
        #     there is pure latency on the failure path for the same reason.
        if attempt < attempts:
            if reopen is not None:
                reopened = reopen()
                if inspect.isawaitable(reopened):
                    await reopened
            # Unconditional, even at delay=0.0: asyncio.sleep(0) costs one
            # event-loop tick and hands control back, which is the minimum
            # that makes a reopen=None caller (a transient re-armed by an
            # external producer) able to make progress at all. Placed AFTER
            # reopen so the wait is spent on an already-open window.
            await asyncio.sleep(delay)
    detail = f' {message}' if message else ''
    raise AssertionError(
        f'retry_until_observed: condition not observed in {attempts} attempt(s).{detail}'
    )
