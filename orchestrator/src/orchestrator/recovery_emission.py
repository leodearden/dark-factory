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
from datetime import UTC, datetime
from typing import Any

logger = logging.getLogger(__name__)

__all__ = [
    'LeaveReason',
    'RecoverySite',
    'build_recovery_payload',
    'emit_recovery_event',
    'escalation_ages_secs',
    'render_shape',
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
