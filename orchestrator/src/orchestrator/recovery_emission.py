"""Structured recovery-decision emission — what held a stranded task, in facts.

Task 3535 (beta).  PRD ``plans/task-escalation-state-graph-prd.md`` D5; spec
``docs/task-escalation-state-spec.md`` S6/E12; design invariants INV-2
(structured-facts-at-failure) and INV-4 (storm escape is DEDUPE, never log spam).

WHY THIS MODULE EXISTS
----------------------
**This docstring is the single canonical explanation.**  Every other site that
touches this mechanism — ``config.RecoveryEmissionConfig``, the
``recovery_emission`` stanza in ``defaults.yaml``, the
``EventType.recovery_vetoed`` / ``recovery_left`` members, the five call sites
in ``harness.py`` / ``scheduler.py``, ``task_ground_truth.leave_reason``, and
the test modules — carries a ONE-LINE pointer here instead of a copy.  The
``zero_progress_requeue`` docstring records why: near-verbatim copies of a
causal chain go stale in as many ways as there are copies.

A task can sit ``in-progress`` with no live claimant for days while every
recovery sweep that looks at it decides, correctly, to LEAVE it alone — because
an open escalation pins it.  Before this module, that decision was invisible:
the reconcile sweep's LEAVE fall-through logged nothing, the harness's
open-escalation early-return was a bare ``return None``, the scheduler's
blocked-redispatch veto was a bare ``continue``, and one half of the
deterministic-recon pair was a completely silent ``continue``.  An operator
looking at a strand could see THAT nothing happened but never WHAT held it, and
no consumer could count holds at all.

So every veto/LEAVE site now emits a structured event naming the task, the
site, the discretized ``_shape`` it classified on, the reason, the pinning
escalation ids and their ages.

EMISSION BEFORE BEHAVIOR
------------------------
This module changes NO disposition, and neither does any of its wiring.  Every
consumer keeps its existing veto predicate byte-identical
(``bool(report.open_escalations)`` / ``bool(rows)``) and calls in here only to
DESCRIBE what it already decided.  Three specific guards make that concrete:

1. ``escalation.pins.classify_pins`` is consulted for id BUCKETING and reason
   selection ONLY — never for the veto answer.  ``PinReport.pins`` deliberately
   treats an info-severity record and a dead-L0 as non-pinning, so swapping it
   in at the reconcile sweep or the scheduler phase would change which tasks
   get reverted or redispatched.  That rewiring is task eta (3541), behind the
   operator flip; ``pins.py``'s own docstring already assigns it there.
2. ``TruthReport.escalation_store_unavailable`` is recorded and EMITTED but is
   deliberately not folded into ``task_ground_truth._shape``'s table key.
   Folding it would flip a store-outage strand from REVERT_TO_PENDING to LEAVE.
3. The streak alarm files against a SENTINEL task id, never the real one — see
   :data:`RECOVERY_VETO_STREAK_SENTINEL_PREFIX`.

EMISSION CADENCE — why signature-transition-gated, not one row per observation
-----------------------------------------------------------------------------
Two of the five sites run per dispatch TICK, not per sweep:
``Scheduler._phase_redispatch_stranded_blocked`` and
``Harness._already_landed_dispatch_gate``.  Emitting unconditionally there
would append one SQLite row per tick per pinned task, forever — the INV-4 storm
that the gate's own ``note_hold_observed`` / ``clear_hold_observed``
transition-gating already exists to avoid at that very site.

So an event is emitted when the ``(site, task_id)`` veto SIGNATURE is new or
has CHANGED, and once more exactly at the streak threshold crossing (see
:func:`should_emit_event`).  The per-sweep operational cadence operators
actually want is delivered instead by the reconcile sweep's always-logged
summary line (one aggregate line naming held/left counts and the pinning ids),
which costs one line per sweep rather than one row per task per tick.

Two consequences a consumer must know, both documented on the ``EventType``
members as well:

* Read these rows as STATE, not as a rate.  Three rows do not mean three holds.
* The tracker is in-memory and deliberately NOT durable, so a fleet restart
  re-arms every signature and the first post-deploy sweep re-emits for every
  live strand.  That is precisely the D5 user-observable signal, not a bug.

A LEAVE caused by a LIVE claimant emits nothing at all: on a healthy fleet that
is the overwhelming majority of every sweep, and emitting for it would bury the
strand signal under normal traffic.

SHAPE
-----
Following ``zero_progress_requeue.py`` member-for-member:

* pure payload primitives (:func:`render_shape`, :func:`escalation_ages_secs`,
  :func:`build_recovery_payload`) — no I/O, every collaborator injected;
* :func:`emit_recovery_event` — the fire-and-forget event wrapper;
* :class:`RecoveryVetoStreakTracker` — pure in-memory streak + span tracking;
* the fail-open escalation filer and its mandatory recovery half.
"""

from __future__ import annotations

import enum
import json
import logging
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

logger = logging.getLogger(__name__)

__all__ = [
    'RECOVERY_VETO_STREAK_SENTINEL_PREFIX',
    'AgeableRecord',
    'LeaveReason',
    'Observation',
    'RecoverySite',
    'RecoverySweepTally',
    'RecoveryVetoStreakTracker',
    'as_ageable_records',
    'build_recovery_payload',
    'emit_recovery_event',
    'emit_recovery_veto_streak_escalation',
    'escalation_ages_secs',
    'render_shape',
    'resolve_recovery_veto_streak_escalation',
    'should_emit_event',
]


class RecoverySite(enum.StrEnum):
    """The closed vocabulary of veto/LEAVE sites that emit.

    Genuine ``str`` members (mirrors ``escalation.pins.PinClass`` /
    ``orchestrator.task_ground_truth.RecoveryAction``) so equality against a
    plain string holds without an explicit ``.value`` and a member JSON-encodes
    as its spelling.

    ``deterministic_recon_sweep`` and ``deterministic_recon_deploy`` are the
    duplicated deterministic-recon predicate.  They are DELIBERATELY two labels
    rather than one: collapsing the pair is task eta's job (3541), and a shared
    label would hide exactly the duplication eta needs to measure.
    """

    #: ``Harness._reconcile_one_stranded`` — the classify/apply seam, per sweep.
    reconcile_sweep = 'reconcile_sweep'
    #: ``Scheduler._phase_redispatch_stranded_blocked`` — per dispatch TICK.
    scheduler_blocked_redispatch = 'scheduler_blocked_redispatch'
    #: ``Harness._run_deterministic_recon_sweep``'s Source-A dedup skip.
    deterministic_recon_sweep = 'deterministic_recon_sweep'
    #: ``Harness._recover_stranded_deterministic_task``'s dedup skip (its twin).
    deterministic_recon_deploy = 'deterministic_recon_deploy'
    #: ``Harness._already_landed_dispatch_gate`` — per dispatch TICK.
    already_landed_gate = 'already_landed_gate'


#: The sites whose observations CHARGE the N-consecutive-vetoes alarm.
#:
#: A property of the SITE, not of any one call, so it is stated once here
#: rather than passed as a flag a future call site could get wrong.  Only
#: sweep-frequency sites qualify: ``stranded_reconcile_interval_secs`` and
#: ``deterministic_recon_sweep_interval_secs`` are both 900s, so three
#: consecutive observations mean a task has been held for half an hour.  The
#: two sites deliberately ABSENT here — ``scheduler_blocked_redispatch`` and
#: ``already_landed_gate`` — run per dispatch TICK, so charging them would file
#: a blocking L1 seconds after a hold appeared.  ``veto_streak_min_span_secs``
#: is the belt-and-braces backstop for that; this set is the primary guard.
STREAK_CHARGING_SITES = frozenset({
    RecoverySite.reconcile_sweep,
    RecoverySite.deterministic_recon_sweep,
    RecoverySite.deterministic_recon_deploy,
})


class LeaveReason(enum.StrEnum):
    """The closed vocabulary of reasons a site held back or fell through.

    Ordered here the way :func:`task_ground_truth.leave_reason` evaluates them;
    that precedence chain is the normative one and is documented at its
    definition, not restated here.
    """

    #: An open escalation actively pinned the task (the veto).
    escalation_pinned = 'escalation_pinned'
    #: The escalation store could not be READ — never collapsed into "no
    #: records", because a false ``[]`` would route a genuinely-pinned strand
    #: into the plain revert branch (esc-3163; see
    #: ``escalation.pins.classify_pins``'s store-correctness contract).
    escalation_store_unavailable = 'escalation_store_unavailable'
    #: The ``_RECOVERY`` table has no row for this shape — the fail-safe
    #: default, not an error.
    unmapped_shape = 'unmapped_shape'
    #: A claimant is live, so there is nothing to recover.  The healthy
    #: majority; deliberately NOT emitted (see the module docstring).
    live_claimant = 'live_claimant'
    #: A deliberately-unmapped in-flight deploy phase (VERIFIED / FAILED /
    #: SCHEDULED / ESCALATED / DONE).
    deploy_phase_in_flight = 'deploy_phase_in_flight'
    #: The already-landed gate's provenance-arbitration hold.
    provenance_arbitration = 'provenance_arbitration'


#: Rendered in place of a ``None`` deploy phase.  Distinct from ``unknown``:
#: most tasks legitimately have NO deploy phase, which is a known fact, whereas
#: ``unknown`` means the site did not resolve the element at all.
_NO_VALUE = '-'
_UNKNOWN = 'unknown'


def _render_element(value: Any, *, none_as: str = _UNKNOWN) -> str:
    """Render one shape element to its stable lowercase spelling."""
    if value is None:
        return none_as
    if isinstance(value, bool):
        return 'true' if value else 'false'
    # StrEnum members stringify to their value; a plain enum to 'Cls.NAME', so
    # prefer .value when present.
    raw = getattr(value, 'value', value)
    text = str(raw).strip().lower()
    return text or none_as


def render_shape(
    status: Any,
    live_claimant: Any,
    branch_state: Any,
    has_open_escalation: Any,
    deploy_phase: Any,
) -> str:
    """Render the 5-tuple ``task_ground_truth._shape`` keys ``_RECOVERY`` on.

    The element ORDER is load-bearing: it mirrors ``_shape`` exactly, so an
    emitted ``shape`` string and the table key can be read against each other.
    ``task_ground_truth.recovery_shape_str`` is the binding that keeps the two
    from drifting; do not re-derive this ordering anywhere else.

    Elements accept either the domain enum or its plain-string spelling.  A
    ``None`` deploy phase renders ``'-'`` (a known "no deploy state"); any
    OTHER ``None`` renders ``'unknown'``, which is what a site that does not
    resolve that element — the dispatch gate performs no claimant or branch
    resolution by design — emits rather than guessing.
    """
    return '|'.join((
        _render_element(status),
        _render_element(live_claimant),
        _render_element(branch_state),
        _render_element(has_open_escalation),
        _render_element(deploy_phase, none_as=_NO_VALUE),
    ))


def _parse_timestamp(raw: Any) -> datetime | None:
    """Parse an ISO-8601 timestamp to tz-aware UTC, or ``None``.

    A naive value is ASSUMED UTC — the same legacy-value guard the harness's
    stale-lane census already applies (``harness.py``: "Records are written
    tz-aware (isoformat of a UTC datetime); guard a legacy naive value so the
    subtraction never raises TypeError").
    """
    if not isinstance(raw, str):
        return None
    text = raw.strip()
    if not text:
        return None
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed


def escalation_ages_secs(
    records: Any,
    *,
    now: Any,
) -> dict[str, float | None]:
    """Age each record in *records*, as a ``{escalation_id: seconds}`` MAPPING.

    A MAPPING rather than a parallel list: a list couples an age to a POSITION
    and would silently mis-attribute it if a consumer re-sorted the ids.

    TOTAL by construction — a record whose ``created_at`` is missing, blank,
    ``None`` or unparseable maps to ``None`` rather than being DROPPED.  A hold
    that cannot be aged is still a hold, and an absent key would read as "no
    such record" to a consumer joining this against ``escalation_ids``.  A
    FUTURE timestamp clamps to ``0.0``, so a clock skew can never put a
    negative age in the payload.

    Args:
        records: Any sequence of objects exposing ``.id`` and (ideally)
            ``.created_at`` — structural, so both
            ``task_ground_truth.EscalationRef`` and
            ``escalation.models.Escalation`` satisfy it (the latter's creation
            time is its ``timestamp`` field, normalised by the caller).
            ``None`` yields ``{}``.
        now: The measurement instant, as an ISO-8601 string or a ``datetime``.

    Never raises: this runs inside recovery sweeps, and telemetry must not be
    able to abort one.
    """
    if not records:
        return {}
    reference = now if isinstance(now, datetime) else _parse_timestamp(now)
    ages: dict[str, float | None] = {}
    for record in records:
        try:
            rec_id = str(record.id)
        except Exception:  # noqa: BLE001 — a corrupt record must not abort a sweep
            logger.debug('recovery emission: skipping an unreadable escalation record')
            continue
        created = _parse_timestamp(getattr(record, 'created_at', None))
        if created is None or reference is None:
            ages[rec_id] = None
            continue
        if reference.tzinfo is None:
            reference = reference.replace(tzinfo=UTC)
        ages[rec_id] = round(max(0.0, (reference - created).total_seconds()), 3)
    return ages


@dataclass(frozen=True)
class AgeableRecord:
    """One escalation record reduced to what :func:`escalation_ages_secs` reads."""

    id: str
    created_at: str | None


def as_ageable_records(records: Any) -> list[AgeableRecord]:
    """Normalise any escalation record shape to ``.id`` + ``.created_at``.

    :func:`escalation_ages_secs` reads ``created_at`` — the name
    ``task_ground_truth.EscalationRef`` carries, because that is the shape the
    reconcile-sweep site holds.  An ``escalation.models.Escalation`` (what
    ``EscalationQueue.get_by_task`` returns, and what the scheduler and the
    deterministic-recon sites hold) spells the same fact ``timestamp``.

    Normalising in ONE shared place rather than at each site is the whole
    point of this module: a per-site copy is how the same predicate ends up
    hand-rolled five slightly-different ways.  Keeping it here — instead of
    teaching ``escalation_ages_secs`` two field names and a precedence between
    them — also keeps that helper's record contract single-valued.

    Total and never raises: this feeds telemetry inside recovery sweeps, so a
    corrupt record costs at most its own age, never the pass.
    """
    adapted: list[AgeableRecord] = []
    for record in records or ():
        try:
            adapted.append(AgeableRecord(
                id=str(record.id),
                created_at=(
                    getattr(record, 'created_at', None)
                    or getattr(record, 'timestamp', None)
                ),
            ))
        except Exception:  # noqa: BLE001 — a corrupt record must not abort a sweep
            logger.debug('recovery emission: skipping an unreadable escalation record')
    return adapted


def _jsonable(value: Any) -> Any:
    """Coerce *value* to something ``json.dumps`` accepts, defensively.

    ``EventStore.emit`` JSON-encodes ``data``; an unserialisable member would
    drop the whole row, so an unexpected type degrades to its ``repr`` rather
    than losing the fact.
    """
    try:
        json.dumps(value)
    except (TypeError, ValueError):
        return repr(value)
    return value


def build_recovery_payload(
    *,
    task_id: str | None,
    site: RecoverySite | str,
    shape: str,
    reason: LeaveReason | str | None,
    escalation_ids: Any,
    ages_secs: Any,
    store_unavailable: bool,
    streak: int,
    now: Any,
) -> dict[str, Any]:
    """Build the closed-vocabulary payload both event types carry.

    The key set — ``{task_id, site, shape, reason, escalation_ids, ages_secs,
    measured_at, store_unavailable, streak}`` — is documented alongside the
    ``EventType.recovery_vetoed`` / ``recovery_left`` members and must stay in
    lockstep with them.

    ``escalation_ids`` is a ``{bucket: [id, ...]}`` mapping straight off
    ``escalation.pins.PinReport``'s buckets, with each bucket SORTED: two
    identical vetoes must produce two identical signatures, and store order is
    not guaranteed stable.
    """
    if isinstance(escalation_ids, dict):
        buckets = {
            str(bucket): sorted(str(i) for i in (ids or ()))
            for bucket, ids in escalation_ids.items()
        }
    else:
        buckets = {'queue_handoff': sorted(str(i) for i in (escalation_ids or ()))}

    measured_at = now if isinstance(now, str) else None
    if measured_at is None:
        stamp = now if isinstance(now, datetime) else datetime.now(UTC)
        measured_at = stamp.isoformat()

    return {
        'task_id': task_id,
        'site': str(site),
        'shape': str(shape),
        'reason': None if reason is None else str(reason),
        'escalation_ids': buckets,
        'ages_secs': _jsonable(dict(ages_secs or {})),
        'measured_at': measured_at,
        'store_unavailable': bool(store_unavailable),
        'streak': int(streak),
    }


def emit_recovery_event(
    *,
    event_store: Any,
    event_type: Any,
    task_id: str | None,
    payload: dict[str, Any],
) -> bool:
    """Emit one recovery event, fire-and-forget.

    Wraps the ``if self.event_store: self.event_store.emit(...)`` shape once so
    the five call sites do not each hand-roll the ``None`` check and the
    try/except.  The whole body is guarded — like ``EventStore.emit`` itself —
    so telemetry can NEVER disturb a recovery sweep.

    ``task_id`` is passed as a first-class column (not only inside ``data``) so
    these rows stay joinable against ``task_completed`` / ``escalation_created``.
    ``None`` is legal and means a PROCESS-scoped emission with no single
    subject — the scheduler's queue-absent notice is one.

    Returns ``True`` only when the store accepted the call.
    """
    if event_store is None:
        return False
    try:
        event_store.emit(event_type, task_id=task_id, data=payload)
    except Exception as exc:  # noqa: BLE001 — telemetry is best-effort
        logger.warning(
            'recovery emission for task %s could not be emitted (non-fatal): %s',
            task_id, exc,
        )
        return False
    return True


@dataclass(frozen=True)
class Observation:
    """One site's observation of one task's veto, folded into its streak.

    Returned by :meth:`RecoveryVetoStreakTracker.observe` so a caller can both
    decide whether to emit (:func:`should_emit_event`) and charge the streak
    alarm from a single call.
    """

    site: RecoverySite
    task_id: str
    #: The veto SIGNATURE — ``reason|shape|sorted ids``.  Two observations with
    #: equal signatures are the SAME hold seen twice; an unequal one is a new
    #: fact and restarts the streak.
    signature: str
    #: Consecutive observations of this identical signature, starting at 1.
    streak: int
    #: True when this observation started a streak (new key, or a signature
    #: that differs from the one held).
    changed: bool


@dataclass
class _Streak:
    """One ``(site, task_id)``'s live veto streak."""

    signature: str
    count: int
    #: Clock reading at the FIRST observation of this streak.  Held on the
    #: streak (not per-observation) so the span measures the whole hold, and
    #: dropped with it on reset.
    started_at: float


class RecoveryVetoStreakTracker:
    """Counts CONSECUTIVE IDENTICAL vetoes, per ``(site, task_id)``, with span.

    Pure in-memory state with no I/O — the harness and the scheduler each own
    one instance for the life of the process.  Modelled member-for-member on
    ``zero_progress_requeue.ZeroProgressRequeueTracker``; see this module's
    docstring for why the mechanism exists at all.

    Keyed on ``(site, task_id)`` rather than ``task_id`` alone because the five
    sites run at wildly different frequencies: ``already_landed_gate`` and
    ``scheduler_blocked_redispatch`` fire per dispatch TICK while the reconcile
    sweep fires every ``stranded_reconcile_interval_secs``.  A shared counter
    would let a tick site drive a sweep site's streak past the alarm threshold
    within seconds of a strand appearing.

    Deliberately NOT durable.  A fleet restart re-arms every signature, so the
    first post-deploy sweep re-emits for every live strand — precisely the D5
    "name each currently-stranded task's pinning escalation ids" signal, not a
    lost-state bug.

    Args:
        clock: Monotonic seconds source, injected for testability.
            ``time.monotonic`` (not wall clock) so a span cannot be corrupted
            by an NTP step or a DST jump during a multi-week run.
    """

    def __init__(self, clock: Callable[[], float] = time.monotonic) -> None:
        self._clock = clock
        #: (site, task_id) -> live streak.  Entries are POPPED by :meth:`clear`
        #: rather than zeroed, so the dict stays proportional to the tasks
        #: CURRENTLY held (zero, on a healthy fleet) instead of growing one
        #: entry per task ever swept over a weeks-long run.
        self._streaks: dict[tuple[str, str], _Streak] = {}

    @staticmethod
    def _key(site: RecoverySite | str, task_id: str) -> tuple[str, str]:
        return (str(site), str(task_id))

    def observe(
        self,
        site: RecoverySite | str,
        task_id: str,
        signature: str,
    ) -> Observation:
        """Fold one observed veto into ``(site, task_id)``'s streak.

        Args:
            site: Which veto/LEAVE site is speaking.
            task_id: The task being held.
            signature: The veto signature (``reason|shape|sorted ids``).  Equal
                signatures accumulate; an unequal one RESTARTS the streak, so
                an alarm only ever describes N genuinely identical holds.

        Returns:
            The :class:`Observation`, carrying the new streak and whether this
            observation was a transition.
        """
        key = self._key(site, task_id)
        live = self._streaks.get(key)
        if live is None or live.signature != signature:
            self._streaks[key] = _Streak(
                signature=signature, count=1, started_at=self._clock(),
            )
            changed = True
            count = 1
        else:
            live.count += 1
            changed = False
            count = live.count
        return Observation(
            site=site if isinstance(site, RecoverySite) else RecoverySite(site),
            task_id=str(task_id),
            signature=signature,
            streak=count,
            changed=changed,
        )

    def streak(self, site: RecoverySite | str, task_id: str) -> int:
        """Return the current streak for ``(site, task_id)`` — ``0`` if unheld."""
        live = self._streaks.get(self._key(site, task_id))
        return live.count if live is not None else 0

    def span(self, site: RecoverySite | str, task_id: str) -> float:
        """Return seconds elapsed since this streak began — ``0.0`` if unheld.

        The second half of the alarm predicate (see
        ``config.RecoveryEmissionConfig.veto_streak_min_span_secs``): a streak
        that accrued in seconds is a per-tick site observing the same hold
        rapidly, not a task stuck for a shift.  Never negative.
        """
        live = self._streaks.get(self._key(site, task_id))
        if live is None:
            return 0.0
        return max(0.0, self._clock() - live.started_at)

    def clear(self, site: RecoverySite | str, task_id: str) -> int:
        """POP ``(site, task_id)``'s streak; return the count it dropped.

        Called on the transition where a task STOPS being vetoed.  The return
        value is what the resolve half reports as the streak that ended, and
        the pop is the footprint contract: the reconcile sweep visits every
        in-progress and blocked task, so a tracker that zeroed instead of
        popping would grow one entry per task ever swept.
        """
        dropped = self._streaks.pop(self._key(site, task_id), None)
        return dropped.count if dropped is not None else 0

    def tracked(self) -> Iterable[tuple[str, str]]:
        """Return the ``(site, task_id)`` keys currently holding a streak.

        Introspection hook: exists so tests (and any future operator dump) can
        assert the pop-on-reset footprint contract directly.
        """
        return tuple(self._streaks)


def should_emit_event(observation: Observation, *, threshold: int) -> bool:
    """Decide whether *observation* warrants an event row.

    **This is the canonical statement of the emission cadence**; the five call
    sites reference it rather than restating the rationale.

    True in exactly two cases:

    * ``observation.changed`` — a NEW or CHANGED veto signature.  That is a new
      fact about the strand, and it is what makes the first post-restart sweep
      name every live strand.
    * ``observation.streak == threshold`` — the escalation moment, so the event
      store records the instant the blocking L1 was filed.  ``==`` rather than
      ``>=`` on purpose: a long-lived veto emits here ONCE and then goes quiet
      forever, instead of appending a row on every later observation.

    False for every other quiet repeat.  ``Scheduler._phase_redispatch_stranded
    _blocked`` and ``Harness._already_landed_dispatch_gate`` run per dispatch
    TICK, so unconditional emission would append one SQLite row per tick per
    pinned task indefinitely — the INV-4 storm the dispatch gate's own
    ``note_hold_observed`` / ``clear_hold_observed`` transition-gating already
    exists to avoid at that very site.  The per-sweep operational cadence is
    supplied instead by the reconcile sweep's always-logged summary line, at
    one line per sweep rather than one row per task per tick.
    """
    if observation.changed:
        return True
    return observation.streak == threshold


#: Sentinel task_id prefix for the veto-streak alarm.
#:
#: **This is the canonical statement; other sites point here.**  The alarm MUST
#: NOT be filed against the real task id.  An open escalation on the real id is
#: immediately read by every veto predicate this mechanism observes —
#: ``report.open_escalations`` in the reconcile sweep, ``get_by_task`` in the
#: scheduler's blocked-redispatch phase, ``rows`` in the dispatch gate — so the
#: act of REPORTING a hold would itself deepen that hold, at the very site that
#: filed it.  That converts an observability signal into a disposition change
#: and breaks this task's zero-behavior-change contract outright.  It could
#: additionally suppress the stranded_blocked re-file.
#:
#: A synthetic id also matches what this record IS: a monitor signal ABOUT a
#: task, not work ON one — no per-task steward can be dispatched for it.  Same
#: shape and same reasoning as
#: ``zero_progress_requeue.ZERO_PROGRESS_SENTINEL_PREFIX``.
#:
#: The alarm stays joinable against the task's ``recovery_vetoed`` rows because
#: its body names the REAL task id.
RECOVERY_VETO_STREAK_SENTINEL_PREFIX = '__recovery_veto_streak__'

#: Category used for BOTH the filing and the ``has_open_l1`` signature filter.
#: Filtering on category (task 2757's rationale) keeps an UNRELATED open L1 on
#: this sentinel from silently suppressing the signal.
_STREAK_CATEGORY = 'risk_identified'

#: agent_role stamped on both the alarm and its resolution.
_STREAK_ROLE = 'orchestrator-recovery-veto-streak'


def _fmt_duration(seconds: float) -> str:
    """Render a span the way an operator reads it."""
    if seconds < 90.0:
        return f'{seconds:.0f}s'
    if seconds < 5400.0:
        return f'{seconds / 60.0:.1f} min'
    return f'{seconds / 3600.0:.1f} h'


def _entry_id(entry: Any) -> str:
    """Normalise one entry to an escalation ID string.

    Id-normalisation is part of :func:`_flatten_ids`' contract, not a caller's
    duty — do not "simplify" this back to a bare ``str()``.  Callers legitimately
    hold RECORDS rather than pre-extracted ids (``EscalationRef`` at the
    reconcile sweep, a raw queue ``Escalation`` at the scheduler and
    deterministic-recon sites), and passing those records straight through is
    what lets the tally, :func:`as_ageable_records` and
    :func:`escalation_ages_secs` all read the SAME objects instead of two
    divergently-normalised copies.  Without this, a record folds as its whole
    dataclass repr and the operator-facing summary becomes unreadable.

    Provably inert for the already-correct path: a plain ``str`` has no ``.id``,
    so a bucketed mapping of id strings is byte-identical through here.

    Total by construction — an entry exposing no ``.id`` degrades to its
    ``str()`` rather than raising.  This runs inside a sweep, and a dropped id
    would read as "nothing held this".
    """
    return str(getattr(entry, 'id', entry))


def _flatten_ids(escalation_ids: Any) -> list[str]:
    """Flatten a bucketed id mapping (or a bare sequence) to a sorted list.

    Entries may be id strings OR records carrying an ``.id`` — see
    :func:`_entry_id`.
    """
    if isinstance(escalation_ids, dict):
        flat: list[str] = []
        for ids in escalation_ids.values():
            flat.extend(_entry_id(i) for i in (ids or ()))
        return sorted(set(flat))
    return sorted({_entry_id(i) for i in (escalation_ids or ())})


def emit_recovery_veto_streak_escalation(
    *,
    escalation_queue: Any,
    task_id: str,
    site: RecoverySite | str,
    streak: int,
    threshold: int,
    span_seconds: float,
    min_span_seconds: float,
    reason: LeaveReason | str | None,
    shape: str,
    escalation_ids: Any,
    ages_secs: Any = None,
    filed_at: dict[str, int] | None = None,
) -> bool:
    """File ONE blocking L1 when a veto streak clears BOTH halves of the bar.

    ``streak >= threshold`` AND ``span_seconds >= min_span_seconds``.  See
    ``config.RecoveryEmissionConfig.veto_streak_min_span_secs`` for why a
    streak count alone is not enough, and this module's docstring for why only
    the sweep-frequency sites charge the counter at all.

    Filed against the SENTINEL id — see
    :data:`RECOVERY_VETO_STREAK_SENTINEL_PREFIX` for why that is load-bearing
    rather than cosmetic.

    Deliberately emits NO event of its own.  The payload key set both recovery
    event types carry is closed, so a "paired" row here would be byte-identical
    to the one :func:`should_emit_event`'s threshold clause already emits at
    this exact crossing.  One row, not two.

    Fully fail-open: every external call is individually guarded AND the whole
    body is wrapped again, so this NEVER raises into a recovery sweep.

    Args:
        escalation_queue: The ``EscalationQueue``, or ``None`` (no-op).
        task_id: The REAL task id being held.
        site: Which veto site observed the streak.
        streak: The current consecutive-identical-veto count.
        threshold: ``config.recovery_emission.veto_streak_threshold``.
        span_seconds: Seconds the streak has spanned
            (``RecoveryVetoStreakTracker.span``).
        min_span_seconds: ``config.recovery_emission.veto_streak_min_span_secs``.
        reason: The :class:`LeaveReason` the site classified.
        shape: The rendered ``_shape`` string.
        escalation_ids: The pinning ids — a bucketed mapping or a bare sequence.
        ages_secs: Optional ``{escalation_id: seconds}`` mapping, reported
            inline so the operator does not need a second query to see how old
            the hold is.
        filed_at: Optional caller-owned memo (``task_id -> streak at last disk
            check``).  Keyed on task_id — matching the SENTINEL's own
            granularity — so the memo can never disagree with what
            ``has_open_l1`` would answer.

    Returns:
        ``True`` only when a NEW escalation was filed.
    """
    if escalation_queue is None:
        return False

    # Fire on >= rather than ==: if an operator resolves the alarm while the
    # hold persists, a later re-check re-files, so the signal cannot be
    # permanently silenced by a premature resolve.
    if streak < threshold:
        return False

    # Second half of the predicate, checked BEFORE any filesystem access so the
    # common case stays free.
    if span_seconds < min_span_seconds:
        return False

    sentinel = f'{RECOVERY_VETO_STREAK_SENTINEL_PREFIX}{task_id}'

    try:
        # The memo answers the dedup for free on the sweep path; we only go to
        # disk once every `threshold` further observations, which bounds the
        # pending-queue scan rate while still re-filing (within `threshold`
        # sweeps) if an operator resolves the alarm prematurely.
        if filed_at is not None:
            last_checked = filed_at.get(task_id)
            if last_checked is not None and streak - last_checked < threshold:
                return False

        if escalation_queue.has_open_l1(sentinel, category=_STREAK_CATEGORY):
            if filed_at is not None:
                filed_at[task_id] = streak
            return False

        from escalation.models import Escalation  # noqa: PLC0415 — optional dep

        ids = _flatten_ids(escalation_ids)
        span_str = _fmt_duration(span_seconds)
        ages = dict(ages_secs or {})
        aged = ', '.join(
            f'{i} ({_fmt_duration(ages[i])} old)' if ages.get(i) is not None else f'{i} (age unknown)'
            for i in ids
        ) or '(none recorded)'

        summary = (
            f'Task {task_id} has been held by the same recovery veto '
            f'{streak} consecutive times over {span_str} at site {site} '
            f'(threshold {threshold}) — reason {reason}'
        )
        detail = (
            f'Task: {task_id}\n'
            f'Veto site: {site}\n'
            f'Consecutive IDENTICAL vetoes: {streak}\n'
            f'Elapsed since the streak began: {span_str}\n'
            f'Alert thresholds: {threshold} observations AND '
            f'{_fmt_duration(min_span_seconds)} elapsed\n'
            f'Recovery shape: {shape}\n'
            f'Veto reason: {reason}\n'
            f'Pinning escalations: {aged}\n'
            '\n'
            f'Every one of the last {streak} passes of {site} looked at task '
            f'{task_id}, reached the same conclusion, and left it exactly '
            'where it was.  The task is not progressing and no recovery sweep '
            'will move it while this veto holds.\n'
            '\n'
            'This alarm is filed against a SYNTHETIC sentinel task id '
            f'({sentinel}), never against {task_id} itself: an open record on '
            'the real id would be read by every veto predicate and would '
            'deepen the very hold it reports.  See '
            'orchestrator/src/orchestrator/recovery_emission.py (module '
            'docstring) for the full mechanism.'
        )

        esc = Escalation(
            id=escalation_queue.make_id(sentinel),
            task_id=sentinel,
            agent_role=_STREAK_ROLE,
            severity='blocking',
            level=1,
            category=_STREAK_CATEGORY,
            summary=summary,
            detail=detail,
            suggested_action=(
                f'Resolve or dismiss the pinning escalation(s) above if they '
                f'are stale, or drive them to completion — task {task_id} '
                'cannot move until they clear.  If the hold is legitimate and '
                'long-running, this alarm is the noisy one: retune or silence '
                'it live via the green-tier config section recovery_emission.'
                '{veto_streak_threshold,veto_streak_min_span_secs,'
                'streak_escalation_enabled} — no fleet restart needed.'
            ),
        )
    except Exception as exc:  # noqa: BLE001 — fail-open backstop
        logger.warning(
            'recovery veto streak alarm for task %s could not be built '
            '(non-fatal): %s', task_id, exc,
        )
        return False

    try:
        escalation_queue.submit(esc)
    except Exception as exc:  # noqa: BLE001 — fail-open backstop
        logger.warning(
            'recovery veto streak alarm for task %s could not be filed '
            '(non-fatal): %s', task_id, exc,
        )
        return False

    if filed_at is not None:
        filed_at[task_id] = streak

    logger.warning(
        'Recovery veto streak alarm filed for task %s: %d identical vetoes at '
        '%s over %s (threshold %d, reason=%s, pinned by %s)',
        task_id, streak, site, _fmt_duration(span_seconds), threshold,
        reason, ','.join(_flatten_ids(escalation_ids)) or '(none)',
    )
    return True


def resolve_recovery_veto_streak_escalation(
    *,
    escalation_queue: Any,
    task_id: str,
    recovered_streak: int,
    threshold: int,
    filed_at: dict[str, int] | None = None,
) -> bool:
    """Resolve a filed veto-streak alarm once the veto stops.

    NOT optional polish.  :func:`emit_recovery_veto_streak_escalation` dedups
    on ``has_open_l1``, so an L1 left ``pending`` after the hold cleared would
    (a) leave an operator holding a blocking alarm for a resolved condition and
    (b) permanently suppress the alarm for a genuine LATER strand on the same
    task — the detector would silence itself after one incident per task, for
    the life of the queue.

    Called on the transition where a task stops being vetoed (a sweep that
    visits it and finds no hold).  Deliberately NOT gated on
    ``config.recovery_emission.streak_escalation_enabled``: disabling the
    detector must not strand an already-filed blocking alarm.

    Returns ``True`` when at least one pending sentinel L1 was resolved.
    """
    if escalation_queue is None:
        return False

    # Cheap gate FIRST — every swept task calls this, and touching the
    # filesystem here would make a healthy fleet pay a pending-queue scan per
    # task per sweep.  Go to disk only when THIS process filed (memo hit), or
    # when the streak that just broke was long enough that a PRIOR process
    # could have filed for it before a restart.
    was_filed = filed_at is not None and task_id in filed_at
    if not was_filed and recovered_streak < threshold:
        return False

    if filed_at is not None:
        filed_at.pop(task_id, None)

    try:
        sentinel = f'{RECOVERY_VETO_STREAK_SENTINEL_PREFIX}{task_id}'
        pending = escalation_queue.get_by_task(sentinel, status='pending')
        resolved = 0
        for esc in pending:
            if getattr(esc, 'category', None) != _STREAK_CATEGORY:
                continue
            escalation_queue.resolve(
                esc.id,
                (
                    f'Task {task_id} is no longer held: the '
                    f'{recovered_streak}-observation recovery veto streak '
                    'broke (the pinning escalation cleared, a claimant took '
                    'the task, or the sweep reached a mapped action).  '
                    'Auto-resolved by the recovery-emission detector; it will '
                    're-file if the hold recurs.'
                ),
                resolved_by=_STREAK_ROLE,
            )
            resolved += 1
    except Exception as exc:  # noqa: BLE001 — fail-open backstop
        logger.warning(
            'recovery veto streak alarm for task %s could not be resolved '
            '(non-fatal): %s', task_id, exc,
        )
        return False

    if not resolved:
        return False

    logger.info(
        'Recovery veto streak alarm resolved for task %s after a '
        '%d-observation streak broke', task_id, recovered_streak,
    )
    return True


@dataclass
class RecoverySweepTally:
    """One reconcile pass's aggregate recovery picture.

    Feeds the sweep summary log line, which is the PER-SWEEP half of this
    mechanism's cadence: :func:`should_emit_event` deliberately keeps the event
    store quiet for an unchanged hold (INV-4), so without this line a fleet in
    which every candidate is held would produce neither an event row nor a
    journal line after the first sweep.  One aggregate line per sweep costs a
    fraction of one row per task per tick and is what an operator actually
    reads.

    Deliberately a plain value object passed DOWN as an optional keyword rather
    than hidden mutable harness state: the accumulator is then testable on its
    own, and every existing caller of the per-task reconciler keeps working
    untouched.

    ``held`` counts vetoes (a record actively held something back); ``left``
    counts non-veto fall-throughs, keyed by reason.  Same discriminator as the
    ``recovery_vetoed`` / ``recovery_left`` event types, so the line and the
    rows can be read against each other.
    """

    held: int = 0
    #: The escalation IDS that did the holding, de-duplicated.  Ordered
    #: first-seen ACROSS successive :meth:`record` calls, and sorted WITHIN one
    #: call's own id set (``_flatten_ids`` sorts what it flattens).  Entries
    #: arrive as records or as bare id strings and are normalised to ids by
    #: :func:`_entry_id` — never as reprs.
    pinning_ids: list[str] = field(default_factory=list)
    #: LeaveReason spelling -> count, for every NON-veto fall-through.
    left: dict[str, int] = field(default_factory=dict)
    #: Dispositions that went ahead UNCHANGED while the store was unreadable.
    store_unavailable: int = 0
    #: Every task this pass recorded a hold/fall-through for.  Read as the
    #: COMPLEMENT: a task the pass swept without recording one has stopped
    #: being held, which is the transition that resolves its streak alarm.
    #: Sourced here rather than from a per-task event because the release must
    #: also fire for a task that left the candidate set entirely (it went done,
    #: or was cancelled) and so will never reach the per-task chokepoint again.
    observed_task_ids: set[str] = field(default_factory=set)

    def record(
        self,
        reason: LeaveReason | str | None,
        escalation_ids: Any = (),
        task_id: str | None = None,
    ) -> None:
        """Fold one held/left task into the tally."""
        if reason is None:
            return
        if task_id is not None:
            self.observed_task_ids.add(task_id)
        if str(reason) == LeaveReason.escalation_pinned:
            self.held += 1
            for esc_id in _flatten_ids(escalation_ids):
                if esc_id not in self.pinning_ids:
                    self.pinning_ids.append(esc_id)
            return
        key = str(reason)
        self.left[key] = self.left.get(key, 0) + 1

    def record_store_unavailable(self) -> None:
        """Note a disposition taken on an unreadable store.

        Counted only where the sweep ACTED anyway — a store-unavailable LEAVE
        already appears under ``left``, and counting it twice would overstate
        how often the fleet decided on incomplete information.
        """
        self.store_unavailable += 1

    def render(self) -> str:
        """The operator-facing fragment appended to the sweep summary."""
        held = f'held={self.held}'
        if self.pinning_ids:
            held += f' ({", ".join(self.pinning_ids)})'
        left = f'left={sum(self.left.values())}'
        if self.left:
            breakdown = ', '.join(
                f'{reason}={count}' for reason, count in sorted(self.left.items())
            )
            left += f' ({breakdown})'
        return f'{held}; {left}; store-unavailable={self.store_unavailable}'
