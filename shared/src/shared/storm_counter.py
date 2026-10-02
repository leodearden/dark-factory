"""Shared rolling-window burst detector (INV-4 storm counters, INV-5 single home).

The body below — append ``(now, label, key)``, prune to the window, count,
compare to the threshold, then rate-limit to one fire per window, and report the
DISTINCT labels seen in the window — is the established storm-counter pattern
first written as ``reconciliation/harness.py::_record_placeholder_finding_drop``
and since reproduced in ``harness._dead_owner_suppressions`` and
``server/markup_tripwire.MarkupStormCounter``. Task 3088 extracted it (into
``fused_memory.server.storm_counter``) so a fourth consumer reuses rather than
re-copies it (INV-5).

Task 3689 PROMOTED it to ``shared`` when that fourth consumer arrived:
``shared.mcp_markup_middleware`` keys a burst by ``(project, policy_outcome)``,
and ``shared`` is the base layer every other package imports, so it may not
import ``fused_memory``. The old module is now a re-export shim naming this one
as the single home; every existing importer (``server/markup_tripwire``,
``services/memory_service``) and fused-memory's own test suite keep working
unedited, which is what pins the shim honest.

Uses bulk_reset_guard's guard-side injectable-clock convention
(``time_provider`` stored as ``self._now``) so a 3600s window can be tested by
advancing a fake clock instead of sleeping.

FIRE POLICY — two modes, chosen at construction via ``fire_mode``. The default
``rate_limited`` answers "this burst is still going" at most once per window,
which is what a consumer wants when it logs or escalates a STANDING condition.
``latched`` instead fires exactly once on the threshold CROSSING and re-arms
only when the window drains back below it. The consumer that needs the latch is
``fused_memory/services/memory_metadata_census.py::UnknownKeyStormDetector``,
which sits on the live memory-write path: the crossing is the event, not the
state, and a counter that kept returning a summary would push its escalation
filer into an open-escalation ``queue.get_by_task`` read on every memory write
for a condition already filed. Task 4519 moved that policy in here rather than
leaving a second count-compare-and-fire body outside this module (INV-5).

RELOAD SAFETY — the one contract difference from ``MarkupStormCounter``, and
the reason this class takes them as arguments rather than storing them:
``threshold`` and ``window_seconds`` are supplied PER :meth:`record` CALL.
``config/reload.py``'s reload-safety rule states that a config value captured
by value at construction cannot observe an in-place reload and must therefore
stay restart-only. A consumer whose threshold comes from a green-tier config
leaf (``mem0_update.storm_threshold``) must read it live on every call, or the
leaf is restart-only in disguise — registered in ``RELOADABLE_FIELDS`` while
silently ignoring reloads. Callers whose thresholds are module constants
(``MarkupStormCounter``) simply pass their stored values through.
"""

from __future__ import annotations

import time
from collections import deque
from collections.abc import Callable
from typing import Any, Literal, get_args

#: The accepted ``fire_mode`` spellings as a TYPE. The mode is structural and
#: fixed by the call site (see the class docstring), which is exactly the case
#: a ``Literal`` closes statically: a misspelling is then caught by pyright —
#: which this repo gates on in pre-commit — instead of surfacing only as a
#: runtime ``ValueError`` the first time that call site constructs a counter.
#: ``count_distinct`` gets this property for free by being a ``bool``; a bare
#: ``str`` would be the one spelling of this mode that throws it away.
FireMode = Literal['rate_limited', 'latched']

#: The accepted ``fire_mode`` spellings as a VALUE, public so a consumer can
#: name the policy against the constant rather than re-typing the literal.
#: DERIVED from :data:`FireMode` rather than written out a second time, so the
#: type and the constant cannot drift apart. See the module docstring's FIRE
#: POLICY note for what each one means.
FIRE_MODES: tuple[FireMode, ...] = get_args(FireMode)


class StormCounter:
    """Rolling-window burst detector over labelled events.

    One event is routine; a BURST within the window is the condition worth
    escalating rather than merely logging (INV-4).

    The label dimension is load-bearing, not decoration: a window that mixed
    two labels into a bare count would let the caller attribute the whole burst
    to whichever event happened to cross the threshold. Labels are opaque
    strings carrying no schema, so the same class serves per-``project_root``
    keying (``MarkupStormCounter``) and per-``agent_id`` keying
    (``MemoryService.update_memory``).

    A counter built with ``count_distinct=True`` gains a SECOND, orthogonal
    dimension: the per-call ``key``. The threshold is then compared against the
    number of DISTINCT non-``None`` keys in the window rather than the raw
    event count, while ``label`` keeps naming the burst independently. The
    motivating consumer is
    ``reconciliation/harness.py::_record_dead_owner_suppression``, which must
    threshold on distinct dead-owner ``instance_id`` values (task 2039 — every
    orphan recovered by ONE restart shares that one owner's instance_id, so a
    single multi-project restart contributes 1 no matter how many projects it
    touched) while still attributing the burst to the distinct ``project_id``
    values it spanned. Neither the single ``label`` nor the middleware's
    one-counter-per-key convention can express that: the former thresholds on
    raw events, and the latter would put the threshold on the NUMBER of counter
    objects, which no individual counter can see.

    ``count_distinct`` is deliberately a CONSTRUCTOR flag while ``threshold``
    and ``window_seconds`` stay per-call. It is a STRUCTURAL mode fixed by the
    call site, not a config leaf, so capturing it at construction cannot go
    stale — the RELOAD SAFETY rule in the module docstring constrains config
    VALUES only. It is readable back off :attr:`count_distinct` so a consumer's
    tests can pin the mode without reaching into private state.

    Because the mode is structural, a ``key`` handed to a counter that is NOT in
    ``count_distinct`` mode is a WIRING BUG, not a benign extra argument, and
    :meth:`record` raises rather than ignoring it — see that method.

    ``fire_mode`` is a THIRD dimension, orthogonal to both: ``count_distinct``
    decides what is counted, ``fire_mode`` decides when a crossed threshold is
    reported (see the module docstring's FIRE POLICY note). It is structural in
    exactly the sense ``count_distinct`` is — fixed by the call site, never a
    config leaf — so capturing it at construction cannot go stale and the
    RELOAD SAFETY rule, which constrains config VALUES only, permits it. It is
    likewise readable back off :attr:`fire_mode`, and an unrecognised spelling
    raises at construction for the same reason a mismatched ``key`` does.

    Being structural is also why it is annotated :data:`FireMode` (a
    ``Literal``) rather than ``str``: a call site that names the mode as a
    literal — which is every call site, the mode not being a config leaf — has
    its typo caught by pyright instead of by the constructor on first
    construction. The ``ValueError`` stays as the backstop for the untyped
    callers a ``Literal`` cannot reach.

    State is PROCESS-LOCAL and resets on restart, like every other in-process
    storm counter in this codebase: the counter exists to catch a live burst,
    not to keep durable statistics. It is also per-instance, so no state bleeds
    between servers (or between tests).

    Not thread-safe by construction; the callers run on a single event loop and
    :meth:`record` never awaits.
    """

    def __init__(
        self,
        time_provider: Callable[[], float] = time.time,
        *,
        count_distinct: bool = False,
        fire_mode: FireMode = 'rate_limited',
    ) -> None:
        # Kept despite the :data:`FireMode` annotation, which only closes the
        # TYPED call sites: a mode arriving as a dynamically-computed string (a
        # dict-splatted kwarg, a plain-script import) is still checked here.
        if fire_mode not in FIRE_MODES:
            raise ValueError(
                f'fire_mode={fire_mode!r} is not a StormCounter fire mode; '
                f'accepted spellings are {", ".join(repr(m) for m in FIRE_MODES)}. '
                'The mode is structural and fixed by the call site, so an '
                'unrecognised spelling is a wiring bug: defaulting it would '
                'silently degrade a latched consumer to per-window rate '
                'limiting.'
            )
        self._now = time_provider
        self._count_distinct = count_distinct
        # Annotated, not inferred: pyright widens a literal to its base type
        # when inferring a mutable attribute, which would make this ``str`` and
        # silently drop the guarantee :data:`FireMode` exists to give.
        self._fire_mode: FireMode = fire_mode
        self._events: deque[tuple[float, str | None, str | None]] = deque()
        self._last_fire_ts: float | None = None
        self._latched: bool = False

    @property
    def count_distinct(self) -> bool:
        """Whether this counter thresholds on distinct ``key`` values.

        Read-only: the mode is structural and fixed at construction (see the
        class docstring). Exposed so a consumer's tests can pin the mode they
        depend on — ``reconciliation/harness.py::_record_dead_owner_suppression``
        is wrong, not merely differently-tuned, if its counter is ever built in
        the default mode — without asserting on private state this class makes
        no compatibility promise about.
        """
        return self._count_distinct

    @property
    def fire_mode(self) -> FireMode:
        """Which FIRE POLICY this counter applies once the threshold is met.

        One of :data:`FIRE_MODES`. Read-only: the mode is structural and fixed
        at construction (see the class docstring). Exposed for the reason
        :attr:`count_distinct` is — a consumer whose correctness depends on the
        policy, such as
        ``fused_memory/services/memory_metadata_census.py::UnknownKeyStormDetector``
        and its tests, must be able to pin it through a supported surface
        rather than coupling to private attributes of another package.
        """
        return self._fire_mode

    @property
    def latched(self) -> bool:
        """Whether a ``fire_mode='latched'`` counter has already reported.

        ``True`` between the call that crossed the threshold and the first
        call that finds the window back below it (the re-arm). Always ``False``
        in ``rate_limited`` mode, which has no latch — that mode suppresses on
        the elapsed window instead, via ``_last_fire_ts``.

        Read-only, and exposed for the same reason :attr:`count_distinct` and
        :attr:`fire_mode` are: a per-key consumer and its tests read this state
        through a supported surface instead of coupling to private attributes
        of another package — the coupling task 3259's amendment (3d4418c777)
        removed when it replaced ``harness._dead_owner_storm._count_distinct``
        with the public property.
        """
        return self._latched

    def _prune(self, now: float, window_seconds: float) -> int:
        """Drop events older than the window as of *now*; return how many remain.

        The window is half-open: an event aged exactly *window_seconds* is
        already out.

        Always the RAW remaining-event count, even in ``count_distinct`` mode —
        :meth:`record` derives the distinct-key count from the pruned deque
        itself, and :meth:`prune`'s public contract is remaining STATE.
        """
        cutoff = now - window_seconds
        while self._events and self._events[0][0] <= cutoff:
            self._events.popleft()
        return len(self._events)

    def prune(self, window_seconds: float, now: float | None = None) -> int:
        """Age out stale events without recording one; return the live count.

        The sweep hook for a caller that keys counters by an UNBOUNDED label
        (``MemoryService`` keys them by caller-supplied ``agent_id``): each
        counter self-prunes its own deque, but nothing evicts the counter
        OBJECT, so a long-lived server would accumulate one per label it ever
        saw. A caller can sweep with this and drop whatever returns ``0``.

        Dropping an empty counter is behaviour-preserving, not merely cheap.
        The only other state is ``_last_fire_ts``, stamped while its own event
        was still in the deque — so an empty window implies that fire has
        already aged past the rate limit, and a freshly constructed counter
        would decide identically on the next event.

        That stays true in ``fire_mode='latched'``, whose only extra state is
        :attr:`latched`. An empty window is below any threshold ``>= 1``, so
        the next event lands under the line and RE-ARMS a kept counter before
        the latch could ever suppress — it decides exactly as a fresh one
        would. This is what lets a one-counter-per-key sweeper clear a latch
        structurally, by deleting the object that holds it, rather than
        remembering to reset a parallel set (the residue INV-5 deletes at
        ``fused_memory/services/memory_metadata_census.py``). Pinned by
        ``shared/tests/test_storm_counter.py::TestLatchedState::
        test_a_latched_drained_counter_decides_like_a_fresh_one``.

        The returned count is the number of remaining EVENTS in every mode,
        never the distinct-key count: this is an emptiness probe for sweepers
        that drop whatever returns ``0``, so it must answer "how much state is
        left". A ``count_distinct`` counter holding only ``key=None`` events
        has zero distinct keys but is not empty, and evicting it would discard
        live window state.

        *now* is the optional PER-CALL clock override described on
        :meth:`record`; omitting it reads the constructor-injected
        ``time_provider``, which is the default every existing consumer uses.
        """
        return self._prune(now if now is not None else self._now(), window_seconds)

    def record(
        self,
        *,
        threshold: int,
        window_seconds: float,
        label: str | None = None,
        key: str | None = None,
        now: float | None = None,
    ) -> dict[str, Any] | None:
        """Record one event; return a storm summary iff a burst just fired.

        *threshold* and *window_seconds* are read fresh on every call so a
        consumer backed by a green-tier config leaf can pass live values (see
        the module docstring's RELOAD SAFETY note).

        *label* is what the event should be attributed to, or ``None`` when the
        caller could not resolve one. Unlabelled events still count toward the
        burst — there is simply nothing to name them against.

        *key* is the DISTINCT-COUNT dimension, and requires the counter to have
        been built with ``count_distinct=True`` (see the class docstring). It is
        orthogonal to *label*: the burst is thresholded on distinct keys and
        NAMED by distinct labels. ``key=None`` is excluded from the distinct set
        entirely — such an event neither counts toward the threshold nor blocks
        it — and is always accepted, in either mode, so a caller forwarding an
        optional identifier that happens to be missing is never a mismatch.

        Passing a non-``None`` *key* to a DEFAULT-mode counter raises
        ``ValueError``. Silently ignoring it would degrade that counter to raw-
        event thresholding while the call site reads as if it were counting
        distinct keys — precisely the pre-task-2039 regression
        (esc-recon-50da2482-1) that ``count_distinct`` exists to prevent, and
        invisible in the summary, the logs and the return value alike. A mode
        mismatch is a wiring bug in the CALL SITE, deterministic and caught on
        its first call, so this fails loudly rather than fails soft (INV
        no-silent-fail-soft / structured-facts-at-failure).

        *now* is an optional PER-CALL clock override, as an epoch float. The
        constructor-injected *time_provider* remains the default and is what
        every consumer holding its counter for the process lifetime uses
        (``MarkupStormCounter``, ``MemoryService``). ``now=`` exists for a
        caller that already carries a per-call injected timestamp of its own:
        ``reconciliation/harness.py``'s three storm counters take
        ``now: datetime | None`` on every recording method (the
        ``reconciliation/harness.py::_finding_recently_resolved`` convention)
        and resolve it against ``datetime.now(UTC)`` before calling in. Without
        this door they could only delegate through a mutable clock-holder
        mutated around each call, which is the hand-rolled state INV-5 exists
        to delete. It threads through the window, the pruning and the
        rate-limit arithmetic alike, so an injected instant behaves exactly as
        a provider-stamped one would.

        The count compared to *threshold* is the number of events in the
        window, or — in ``count_distinct`` mode — the number of distinct
        non-``None`` keys among them.

        Returns ``None`` whenever the count within the window is below
        *threshold*. Above it, WHEN a fire is reported depends on the
        constructor's ``fire_mode`` (see the module docstring's FIRE POLICY
        note); the returned SUMMARY is identical either way.

        In the default ``rate_limited`` mode it also returns ``None`` when the
        threshold is met but a previous fire is still inside the window —
        without that limit a runaway emitting hundreds of events would escalate
        hundreds of times for one incident. Otherwise it stamps the rate-limit
        timestamp and returns the summary, so a condition that stays over the
        line is re-reported once per window.

        In ``latched`` mode it returns the summary on the call that CROSSES the
        threshold and ``None`` on every call thereafter, however many windows
        the writer stays over the line — the crossing is the event, not the
        state. The latch clears as soon as the count falls back below
        *threshold*, so a writer that drifts, is fixed, and later drifts again
        is heard both times. ``_last_fire_ts`` is neither consulted nor stamped
        in this mode: the latch, not the elapsed window, is what suppresses.

        The summary itself is JSON-serializable with ``count``, ``threshold``,
        ``window_seconds`` and ``labels`` — the sorted DISTINCT non-``None``
        labels seen in the window, so the caller can attribute the burst
        instead of blaming whichever event crossed the threshold.

        :raises ValueError: if *key* is not ``None`` on a counter built without
            ``count_distinct=True``.
        """
        if key is not None and not self._count_distinct:
            raise ValueError(
                f'key={key!r} was passed to a StormCounter built without '
                'count_distinct=True; that counter thresholds on the RAW event '
                'count and would silently ignore the key. Construct it with '
                'StormCounter(count_distinct=True), or drop the key.'
            )

        effective_now = now if now is not None else self._now()

        # Append, then prune.
        self._events.append((effective_now, label, key))
        count = self._prune(effective_now, window_seconds)
        if self._count_distinct:
            count = len({k for _, _, k in self._events if k is not None})
        if count < threshold:
            # Back under the line. In latched mode that is the RE-ARM: the next
            # crossing is a fresh event and must be heard.
            self._latched = False
            return None

        # Threshold crossed — the append / prune / count above is mode-agnostic;
        # only the decision to REPORT differs.
        if self._fire_mode == 'latched':
            if self._latched:
                return None
            self._latched = True
        else:
            # Default mode: at most one fire per window.
            if (
                self._last_fire_ts is not None
                and (effective_now - self._last_fire_ts) < window_seconds
            ):
                return None
            self._last_fire_ts = effective_now

        return {
            'count': count,
            'threshold': threshold,
            'window_seconds': window_seconds,
            # Unlabelled events still count toward the burst, but there is
            # nothing to escalate them against, so they are simply not named.
            'labels': sorted({lbl for _, lbl, _ in self._events if lbl is not None}),
        }
