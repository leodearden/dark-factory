"""Exponential coalescence for a poll-driven log line that repeats an UNCHANGING fact.

THE PROBLEM (task 3203).  A periodic audit that re-logs the same finding on
every poll destroys the greppability the logging exists to buy.  The merge
queue's resource-conservation audit runs every ``_HEARTBEAT_POLL_S`` (30s)
and logged one WARNING per violating poll: a single leaked worktree held for
two hours emitted 246 identical WARNINGs, and one measured incident emitted
897.  ``journalctl -p warning`` becomes useless exactly when an operator most
needs it, and the *interesting* events — the first occurrence, and every
change to the finding — are buried in the repetition.

THE SCHEDULE.  A caller reduces its finding to a hashable *fingerprint* and
calls :meth:`ExponentialLogCoalescer.observe` once per poll.

  * The FIRST observation of a fingerprint (including the first observation
    ever, and any observation whose fingerprint differs from the previous
    one) always reports ``should_log=True, changed=True``.  A change is never
    withheld — that is the whole point of keying on a fingerprint rather than
    on elapsed time.
  * While the fingerprint is unchanged the emit interval doubles: polls
    1, 2, 4, 8, 16, 32, ... are due, everything between them is not.
  * ``cap_polls`` pins the interval so the cadence never degrades past a
    floor: once ``interval == cap_polls`` the schedule is linear at one
    emission per ``cap_polls`` polls.

LINE-COUNT PROPERTY.  Over ``N`` consecutive unchanged polls the number of
due emissions is exactly ``floor(log2(N)) + 1`` until the cap binds, and
grows at ``1/cap_polls`` thereafter.  So 512 unchanged polls emit 10 lines
instead of 512; the 246-poll and 897-poll incidents above would have emitted
9 and 12.

CALLER-AGNOSTIC AND CLOCK-INJECTED.  This module knows nothing about merge
queues, logging levels, or log formats.  It does not import ``logging``, has
no clock of its own (every method takes an explicit ``now: float``, matching
the merge-queue heartbeat path's clock-injection convention), and performs no
I/O.  The caller decides what to emit and at what level; this only decides
*whether* this poll is due.  That also makes the schedule unit-testable in
isolation, with exact line-count assertions and no sleeping.

TOTAL.  Neither :meth:`observe` nor :meth:`clear` raises for any input: a
non-positive ``cap_polls`` is clamped to 1 at use time (degrading to
log-every-poll rather than wedging or raising), and fingerprints are
compared with ``!=`` only (never hashed into a dict), so any value a caller
can construct is acceptable.  A gate that could raise inside a heartbeat
sub-check would be strictly worse than the repetition it replaces.

``None`` IS A LEGAL FINGERPRINT.  "No run in progress" is a private
module-level sentinel object, not ``None`` — so an adopter whose finding
reduces to a single optional value (plausible for the sweeps named below)
can pass ``None`` and still be coalesced, instead of silently getting
``changed=True`` on every poll and a :meth:`clear` that never fires.  That
overload would have reproduced the exact pathology this module removes, in
the one caller least likely to notice.

PRECEDENT.  This generalises the ``_reprobe_last_info`` /
``REPROBE_STILL_DOWN_INFO_SWEEPS`` rate-limiter already in
``orchestrator/merge_lane/worker.py`` (see its emitter in
``SpeculativeMergeWorker._reprobe_quarantined_hosts``),
whose three shape decisions are adopted verbatim: log immediately on any
CHANGE and throttle only the unchanged repeat; emit suppressed polls at DEBUG
rather than dropping them, so nothing is ever actually lost; and select the
level with a single ``logger.log(level, fmt, ...)`` call.

ADOPTION CANDIDATES, DELIBERATELY OUT OF SCOPE FOR TASK 3203.  Two merge-queue
sweeps outside the heartbeat path have the same unbounded-repetition shape and
could adopt this by construction rather than by rewrite:
``_reprobe_quarantined_hosts`` (120s cadence) and ``reap_orphaned_merge_worktrees``
(300s cadence — whose reclaim behaviour task 3203's
scope note bars touching).  Extracting the schedule here is what makes that
adoption cheap; doing it is a separate task.
"""

from __future__ import annotations

import dataclasses

_UNSET: object = object()
"""Private "no run in progress" marker (never observed, or already cleared).

Deliberately NOT ``None``: ``None`` is a value a caller may legitimately pass
as a fingerprint, and overloading it as the sentinel would make such a caller
report ``changed=True`` on every poll while :meth:`ExponentialLogCoalescer.clear`
never fired — the pathology this module exists to remove.  Identity-compared
(``is``) so it can never collide with a caller value.
"""


@dataclasses.dataclass(frozen=True)
class CoalescedLogDecision:
    """What :meth:`ExponentialLogCoalescer.observe` decided about one poll.

    Attributes:
        should_log: True when this poll is due to emit a report line.  True on
            every change and on each scheduled repeat; False on a suppressed
            poll (which the caller should still emit at DEBUG rather than drop
            — see the module docstring's precedent note).
        changed: True when this poll's fingerprint differs from the previous
            one (including the first observation of an episode).  ``changed``
            implies ``should_log``.  Callers route this to a louder level, and
            emit full detail rather than the coalescing summary.
        unchanged_polls: 1-based count of consecutive polls carrying the
            CURRENT fingerprint, this one included.  1 on a change.
        unchanged_secs: ``now`` minus the ``now`` of the first observation of
            the current fingerprint.  0.0 on a change.
        next_report_in_polls: How many further polls until the next due one.
            Always >= 1; 1 immediately after a change.  Lets a repeat line
            tell an operator when to expect the next one.
    """

    should_log: bool
    changed: bool
    unchanged_polls: int
    unchanged_secs: float
    next_report_in_polls: int


@dataclasses.dataclass(frozen=True)
class ClearedRun:
    """The EPISODE of consecutive observations that a :meth:`clear` ended.

    Both fields are measured over the WHOLE episode — every observation since
    the coalescer last left the cleared state — and not over the final
    fingerprint's segment alone.  The two differ whenever the fingerprint
    changed mid-episode, and a caller that mixed the bases would emit a
    self-contradicting line: an episode of 50 polls whose set changed at poll
    41 would have reported "50 polls / 300s" when 50 polls at a 30s cadence is
    1500s, a 5x understatement of the outage (reviewer_comprehensive
    amendment, task 3203).  Reporting both from here keeps
    ``polls * poll_interval == duration_secs`` true for a fixed-cadence
    caller, which is what makes the single clear line usable for sizing.

    Attributes:
        polls: How many polls the just-ended EPISODE lasted, across every
            fingerprint it passed through.
        duration_secs: Wall-clock span of the EPISODE, from its first
            observation to the ``now`` passed to :meth:`clear`.
    """

    polls: int
    duration_secs: float


class ExponentialLogCoalescer:
    """Decide whether a poll-driven, repeating log line is due this poll.

    Not thread-safe and not async-aware: intended for a single synchronous
    poll loop (the merge queue's heartbeat), which is the only place its
    state is touched.  One instance per logical log line.

    ``cap_polls`` is a plain public attribute rather than a constructor-frozen
    value so a caller whose bound lives on a monkeypatchable class attribute
    can re-sync it before each :meth:`observe` (this is how
    ``SpeculativeMergeWorker.RESOURCE_AUDIT_LOG_COALESCE_CAP_POLLS`` reaches
    the gate when a test overrides it per-instance after ``__init__``).
    """

    def __init__(self, *, cap_polls: int) -> None:
        self.cap_polls: int = cap_polls
        # _UNSET == no run in progress (never observed, or cleared). NOT
        # None — see that sentinel's docstring; None is a legal fingerprint.
        self._fingerprint: object = _UNSET
        # Per-SEGMENT (current fingerprint only) — drives the backoff and the
        # `unchanged for N polls / Ss` context on a repeat line.
        self._since: float = 0.0
        self._unchanged_polls: int = 0
        self._interval: int = 0
        self._next_due: int = 0
        # Per-EPISODE (spans every fingerprint since the cleared state) —
        # drives ClearedRun, so the clear line can size the whole outage.
        self._episode_since: float = 0.0
        self._episode_polls: int = 0

    def observe(self, fingerprint: object, now: float) -> CoalescedLogDecision:
        """Record one poll carrying *fingerprint* and report whether to log.

        *fingerprint* is compared to the previous one by VALUE (``!=``), so a
        caller may rebuild an equal tuple from scratch on every poll — which
        is exactly what a fingerprint derived from a freshly-computed finding
        does.

        Passing a fingerprint the caller considers "empty" is NOT a clear:
        use :meth:`clear` for that, so the end of a run is reported once and
        only once.
        """
        if self._fingerprint is _UNSET:
            # First observation since the cleared state: the episode starts
            # here. Kept distinct from the segment reset below so a mid-run
            # fingerprint change does not restart the episode clock.
            self._episode_since = now
            self._episode_polls = 0
        self._episode_polls += 1

        if self._fingerprint is _UNSET or fingerprint != self._fingerprint:
            # A change (or the first observation of an episode) always logs,
            # at full detail, and restarts the SEGMENT schedule from
            # interval=1. It never resumes the previous fingerprint's backoff
            # — a set that just grew or shrank is new information. The
            # EPISODE counters above are deliberately untouched here.
            self._fingerprint = fingerprint
            self._since = now
            self._unchanged_polls = 1
            self._interval = 1
            self._next_due = 2
            return CoalescedLogDecision(
                should_log=True,
                changed=True,
                unchanged_polls=1,
                unchanged_secs=0.0,
                next_report_in_polls=1,
            )

        self._unchanged_polls += 1
        should_log = self._unchanged_polls >= self._next_due
        if should_log:
            # Clamped at use time (not in __init__) so a caller re-syncing
            # cap_polls between polls takes effect, and so a nonsensical cap
            # degrades to log-every-poll rather than raising or wedging.
            cap = max(1, self.cap_polls)
            self._interval = min(self._interval * 2, cap)
            self._next_due = self._unchanged_polls + self._interval
        return CoalescedLogDecision(
            should_log=should_log,
            changed=False,
            unchanged_polls=self._unchanged_polls,
            unchanged_secs=now - self._since,
            next_report_in_polls=self._next_due - self._unchanged_polls,
        )

    def clear(self, now: float) -> ClearedRun | None:
        """End the current EPISODE, if any, and report what it was.

        Returns a :class:`ClearedRun` exactly once per episode: the first
        call after an episode ends describes it, and every further call
        returns None until :meth:`observe` starts a new one.  That
        single-shot property lives HERE rather than in the caller so that
        "exactly one clear line" falls out of the gate for every call site
        that adopts it, instead of each one re-deriving it from its own
        bookkeeping.

        The reported counts span the WHOLE episode, not just the final
        fingerprint's segment — see :class:`ClearedRun` for why mixing the
        two bases produced a self-contradicting clear line.
        """
        if self._fingerprint is _UNSET:
            return None
        cleared = ClearedRun(
            polls=self._episode_polls,
            duration_secs=now - self._episode_since,
        )
        self._fingerprint = _UNSET
        self._since = 0.0
        self._unchanged_polls = 0
        self._interval = 0
        self._next_due = 0
        self._episode_since = 0.0
        self._episode_polls = 0
        return cleared
