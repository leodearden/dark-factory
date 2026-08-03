"""THE shared severity-aware escalation-pin classifier (task 3533).

PRD ``plans/task-escalation-state-graph-prd.md`` D3; spec
``docs/task-escalation-state-spec.md`` S6/E7.

This module is the SINGLE shared pin predicate (INV-5): every
recovery/redispatch veto site asks ``classify_pins`` whether a task's open
escalations pin it, instead of re-deriving ``bool(open_escalations)`` locally
and disagreeing with its neighbours.  It answers two different questions over
one classification — :attr:`PinReport.pins` (the recovery/redispatch veto) and
:attr:`PinReport.vetoes_done_flip` (the deliberately-more-conservative
MARK_DONE veto) — so no consumer has to hand-roll "is this record non-info".

PURE by construction: no I/O, no store binding, no mutation of its inputs.
The CALLER binds the read to the task's OWNING orchestrator's escalation store
(spec S6 store-correctness contract, esc-3163 lesson) and passes the rows in.

Three record classes:
  ``non_pinning``     an ``info``-severity record — never pins, at any level.
  ``queue_handoff``   a live, supervised handoff — PINS.  Covers L1/L2 (queue-
                      backed, consumed by the auto-watcher / a human), a live-
                      filer L0, and every fail-safe fallthrough.
  ``dead_l0``         an L0 whose FILING incarnation is dead — the handoff has
                      no consumer left, so it does NOT pin recovery (conversion
                      proceeds per spec S4) but DOES still veto a done-flip.

...plus a distinguishable THIRD state, ``store_unavailable``: the escalation
store could not be READ, which is never collapsed into "no records" because a
false ``[]`` would route a genuinely-pinned strand into the plain revert
branch (esc-3163).  See :func:`classify_pins` for the full contract.

The precedence chain, evaluated top to bottom (spec S6; the numbered version
with its rationale lives above :func:`_classify_record`):

  1. ``severity == 'info'``                -> ``NON_PINNING``
  2. ``severity`` not in ``KNOWN_SEVERITIES`` -> ``QUEUE_HANDOFF`` (fail-safe)
  3. ``level != 0``                        -> ``QUEUE_HANDOFF``
  3b. ``severity`` in ``BORN_AT_L2_SEVERITIES`` at ``level == 0``
                                           -> ``QUEUE_HANDOFF`` (fail-safe:
                                              contradictory state)
  4. ``level == 0``                        -> filing-incarnation liveness
                                              (``DEAD_L0`` only when the filer
                                              is PROVABLY dead)

This task delivers the types + classifier + tests ONLY.  Rewiring the veto
sites (``task_ground_truth._shape``, the harness reconcile sweeps, the
scheduler's stranded-blocked sweep) is task eta (3541); the structured
``escalation_store_unavailable`` emission is task beta (3535).
"""

from __future__ import annotations

import enum
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

from escalation.models import BORN_AT_L2_SEVERITIES, KNOWN_SEVERITIES

if TYPE_CHECKING:
    from collections.abc import Sequence

__all__ = ['PinClass', 'PinRecord', 'PinReport', 'classify_pins']


class PinClass(enum.StrEnum):
    """The closed vocabulary of pin classes a record can fall into.

    Genuine ``str`` members (mirrors ``shared.task_statuses.TaskStatus`` /
    ``orchestrator.task_ground_truth.RecoveryAction``) so equality against a
    plain string holds without an explicit ``.value``.
    """

    DEAD_L0 = 'dead_l0'
    QUEUE_HANDOFF = 'queue_handoff'
    NON_PINNING = 'non_pinning'


class PinRecord(Protocol):
    """The minimal record surface :func:`classify_pins` reads.

    A STRUCTURAL protocol so BOTH ``escalation.models.Escalation`` (the
    server-side consumer) and ``orchestrator.task_ground_truth.EscalationRef``
    (the resolver-side consumers) satisfy it without ``escalation`` importing
    ``orchestrator`` — that import direction is a layering inversion
    (``orchestrator`` already depends on ``escalation``; ``escalation`` reaches
    for ``orchestrator`` only lazily, inside ``server.py`` function bodies).

    ``filing_claimant_run_id`` semantics are documented once, on
    :func:`classify_pins` (normative source: spec
    ``docs/task-escalation-state-spec.md`` S6).

    The four members are declared READ-ONLY (property form rather than plain
    ``id: str`` attributes) because a protocol attribute is writable-invariant:
    a plain-attribute protocol rejects FROZEN dataclasses, and
    ``EscalationRef`` — the resolver-side record type this classifier exists to
    serve — is ``@dataclass(frozen=True)``.  Read-only members accept both
    frozen and mutable implementations, which is exactly the pairing here
    (``EscalationRef`` frozen, ``Escalation`` mutable).  The classifier only
    ever reads, so nothing is given up.
    """

    @property
    def id(self) -> str: ...

    @property
    def level(self) -> int: ...

    @property
    def severity(self) -> str: ...

    @property
    def filing_claimant_run_id(self) -> str | None: ...


@dataclass(frozen=True)
class PinReport:
    """The classification of one task's open escalations (a frozen snapshot).

    Buckets carry escalation IDs rather than the records themselves, so the
    report stays a comparable, directly-serialisable value object that drops
    straight into a structured ``recovery_vetoed`` / ``recovery_left``
    emission; a caller that needs timestamps or ages already holds the records
    it passed in.  Tuples (not lists) so the frozen dataclass is genuinely
    immutable and no shared consumer can mutate a bucket in place.
    """

    dead_l0: tuple[str, ...]
    queue_handoff: tuple[str, ...]
    non_pinning: tuple[str, ...]
    store_unavailable: bool = False
    task_id: str = ''

    @property
    def pins(self) -> bool:
        """The recovery/redispatch veto: is this task pinned by an escalation?

        True iff a live handoff pins it, or the store could not be read.  A
        ``dead_l0`` deliberately does NOT pin — its handoff has no consumer
        left, so conversion proceeds (spec S4 / PRD boundary row 7).
        """
        return bool(self.store_unavailable or self.queue_handoff)

    @property
    def vetoes_done_flip(self) -> bool:
        """The MARK_DONE veto — deliberately more conservative than :attr:`pins`.

        True for ANY non-info open record (PRD D3: "any non-info open record
        still vetoes MARK_DONE" — phantom-done protection is the half of the
        veto that was always right), so a ``dead_l0`` blocks a done-flip while
        not blocking a conversion.  Also true when the store was unavailable.
        """
        return bool(self.store_unavailable or self.queue_handoff or self.dead_l0)


# ---------------------------------------------------------------------------
# THE precedence chain (spec docs/task-escalation-state-spec.md S6).
# ONE documented ordering, evaluated top to bottom.  Each link is deliberate;
# re-ordering them changes dispositions, so change them only with the spec.
#
#   1. severity == 'info'                -> NON_PINNING
#      An info record never pins, at any level, under any liveness.
#   2. severity not in KNOWN_SEVERITIES  -> QUEUE_HANDOFF   (fail-safe pin)
#      Missing / blank / out-of-vocabulary severity "fails safe to pinning
#      (treated as a handoff), never to conversion".  Deliberately ABOVE the
#      level/L0 links, so an unknown-severity DEAD L0 still pins.
#   3. level != 0                        -> QUEUE_HANDOFF
#      L1/L2 are queue-backed handoffs with supervised consumers; written
#      `!= 0` (not `>= 1`) so a corrupt/out-of-range level fails safe too.
#   3b. level == 0 AND severity in BORN_AT_L2_SEVERITIES -> QUEUE_HANDOFF
#      Same spirit as link 3's `!= 0`: a critical/urgent record is BORN at L2
#      and routed straight to a human, so one sitting at L0 is contradictory
#      state and must not reach the conversion-eligible branch below.
#   4. level == 0, L0-legal non-info severity -> filing-incarnation liveness:
#        no incarnation live at all         -> DEAD_L0
#        live, and both identities known and DIFFERENT -> DEAD_L0
#        live, and identities MATCH         -> QUEUE_HANDOFF
#        live, either identity unknown      -> QUEUE_HANDOFF (fail-safe pin)
#      "unknown" here means None, blank, OR not in compose_claimant_run_id
#      shape — see `_norm_id`: a format mismatch is not PROOF of a dead filer.
# ---------------------------------------------------------------------------


#: The labelled tail every ``shared.task_claimant.compose_claimant_run_id``
#: value ends with (``{run_id}/{session_id}/pid={owner_pid}``).  Its presence
#: is a cheap SHAPE check, not a parse — see :func:`_norm_id`.
_COMPOSED_ID_MARKER = '/pid='


def _norm_id(value: str | None) -> str | None:
    """Normalise a claimant identity to a COMPARABLE string, or ``None``.

    ``None`` means "unknown", which link 4 fails safe to pinning.  Two things
    collapse to unknown:

    * ``None`` or blank-after-strip — there is nothing to compare.
    * a value that is not in ``compose_claimant_run_id`` shape (no ``/pid=``
      marker).  Comparing a composed identity against a bare ``session_id``
      would ALWAYS mismatch, converting a genuinely LIVE filer's L0 to
      ``DEAD_L0`` — the unsafe direction.  The in-repo liveness resolver
      (``orchestrator.task_ground_truth.TaskGroundTruth._resolve_live_claimant``)
      is heterogeneous today: its DB source yields the composed identity, its
      plan.lock source yields a bare ``session_id``, and its in-memory source
      yields ``None``.  A format mismatch is not PROOF that the filer is dead,
      and this module may only convert on proof.

    A SHAPE check, never a parse: the value is compared WHOLE and never
    decomposed (a differing ``pid=`` suffix is a different incarnation).
    """
    if value is None:
        return None
    stripped = value.strip()
    if not stripped or _COMPOSED_ID_MARKER not in stripped:
        return None
    return stripped


def _classify_record(
    record: PinRecord,
    *,
    live_claimant: bool,
    live_claimant_id: str | None,
) -> PinClass:
    """Map one open escalation to its :class:`PinClass` (see the chain above)."""
    # Normalise once.  `or ''` is load-bearing: an `Escalation` rehydrated from
    # JSON on disk can carry a null `severity` despite the `str` annotation.
    sev = str(record.severity or '').strip().lower()

    # Link 1 — spec S6: an info record is an ANNOTATION, not a handoff.
    if sev == 'info':
        return PinClass.NON_PINNING

    # Link 2 — spec S6: an unknown severity "fails safe to pinning (treated as
    # a handoff), never to conversion".  This link sits ABOVE links 3/4 on
    # purpose: the "never to conversion" clause is only meaningful if the
    # fail-safe is evaluated before the L0 branch that produces the
    # convertible DEAD_L0 class, so an unknown-severity dead L0 STILL pins.
    if sev not in KNOWN_SEVERITIES:
        return PinClass.QUEUE_HANDOFF

    # Link 3 — spec S6: L1/L2 are QUEUE-BACKED handoffs whose consumers (the
    # auto-watcher, a human) are supervised and outlive any one workflow
    # incarnation, so they pin regardless of liveness.  Written `!= 0` rather
    # than `>= 1` on purpose: a corrupt / out-of-range / negative level then
    # also fails safe to pinning instead of falling into the conversion-
    # eligible L0 branch below.
    if record.level != 0:
        return PinClass.QUEUE_HANDOFF

    # Link 3b — a BORN_AT_L2 severity (critical/urgent) at level 0 is
    # CONTRADICTORY state: such records are created directly at L2 and routed
    # straight to a human, which is why every in-repo producer stamps
    # `level=2` alongside them and why `workflow.py` already treats the two
    # signals as one (`e.severity in BORN_AT_L2_SEVERITIES or e.level >= 2`).
    # Fail it safe to pinning for exactly the reason link 3 is written `!= 0`
    # — a human-routed record must never fall into the conversion-eligible
    # branch below on the strength of a level field that contradicts it.
    if sev in BORN_AT_L2_SEVERITIES:
        return PinClass.QUEUE_HANDOFF

    # Link 4 — the filing-incarnation rule (spec S6): an L0 is a live handoff
    # ONLY while the incarnation that FILED it lives.  Liveness is judged
    # against that filer, never against "some workflow for this task is alive".
    if not live_claimant:
        # No incarnation is live at all, so the filing one is necessarily dead
        # — identity-independent.  This branch alone covers the 2026-08-02
        # strand shape (7 of 9 tasks pinned by their own dead-steward L0).
        return PinClass.DEAD_L0

    filed_by = _norm_id(record.filing_claimant_run_id)
    live_id = _norm_id(live_claimant_id)
    if filed_by is not None and live_id is not None and filed_by != live_id:
        # A claimant is live, but it is NOT the one that filed this record:
        # "a newer incarnation never keeps a prior incarnation's unconsumed L0
        # alive" (spec S6).  Exact-string comparison — `shared.task_claimant`
        # ships a composer and deliberately no parser, so a differing `pid=`
        # suffix is a DIFFERENT incarnation.
        return PinClass.DEAD_L0

    # Either the identities MATCH (a genuinely live handoff) or one of them is
    # unknown.  On unknown, the classifier may only convert when it can PROVE
    # the filing incarnation is dead — it cannot, so it fails safe to pinning.
    return PinClass.QUEUE_HANDOFF


def classify_pins(
    task_id: str,
    records: Sequence[PinRecord] | None,
    *,
    live_claimant: bool,
    live_claimant_id: str | None = None,
) -> PinReport:
    """Classify *records* — the task's OPEN escalations — into pin classes.

    :param task_id: echoed onto the report for structured-emission attribution.
        Records are NOT filtered against it — the caller owns the read scope.
    :param records: the task's open escalations, in store order, or ``None``
        when the store could not be READ (see the store-correctness contract
        below).  An empty sequence means "read succeeded, no open records" and
        is a genuinely different answer.
    :param live_claimant: whether ANY incarnation currently holds the task.
    :param live_claimant_id: the live incarnation's ``claimant_run_id``, when
        the caller can resolve it.  Needed to tell "the filer is still live"
        from "a DIFFERENT incarnation is live" — the single task-level boolean
        cannot express that distinction, and it is exactly the case the
        dead-L0 rule exists for.  ``None`` (or an unknown filing identity on
        the record) fails safe to pinning.

        PRECONDITION: it must be a full
        ``shared.task_claimant.compose_claimant_run_id`` string
        (``{run_id}/{session_id}/pid={owner_pid}``) — the same form
        ``PinRecord.filing_claimant_run_id`` carries.  A bare ``session_id``
        or any other shape is NOT a valid substitute; it is detected by the
        ``/pid=`` shape check in :func:`_norm_id` and treated as UNKNOWN
        (fail-safe to pinning) rather than silently mismatching every
        composed filing identity and converting live L0s.

    NORMATIVE (spec ``docs/task-escalation-state-spec.md`` S6 — this docstring
    is the single in-code statement of it; other sites cross-reference here).
    ``PinRecord.filing_claimant_run_id`` is the identity of the incarnation
    that FILED the record, in ``compose_claimant_run_id`` format.  Liveness is
    judged against THAT incarnation, never against "some workflow for this
    task is alive": a newer live incarnation never keeps a prior incarnation's
    unconsumed L0 alive.  ``None`` means "filing identity unknown" — legacy
    records, and every producer not yet stamping it — which fails safe to
    PINNING, because the classifier may only convert an L0 when it can PROVE
    the filing incarnation is dead.  The identity is stored and compared
    verbatim, never parsed.

    See the precedence chain above :func:`_classify_record` for the rules, and
    :class:`PinReport` for the buckets and the two derived predicates.

    STORE-CORRECTNESS CONTRACT (spec ``docs/task-escalation-state-spec.md``
    S6; esc-3163 lesson).  Two obligations, both on the CALLER:

    1. Bind the read to the task's OWNING orchestrator's escalation store —
       never the reconciliation store, never "whichever queue is handy".
       esc-3163 was a wrong-store read that reported no open escalations for a
       task that had them.
    2. Pass ``records=None`` when that read could not be performed or failed.
       Never substitute ``[]``: a false "no records" routes a genuinely-pinned
       strand into the plain revert branch.  This is why the third state is a
       field on the returned report rather than an exception — the collapse is
       structurally impossible, and every consumer inherits one fail-safe
       disposition instead of writing its own try/except.

    The caller-side pattern this generalises already exists at
    ``orchestrator/src/orchestrator/scheduler.py`` (the stranded-blocked
    redispatch sweep's ``escalation_queue is None => never flip`` guard):
    without the queue a sweep cannot verify "no open escalation", so it must
    not act.  ``store_unavailable`` makes that guard expressible in the shared
    predicate; task beta (3535) wires the structured
    ``recovery_left(reason=escalation_store_unavailable)`` emission to these
    same semantics rather than re-deriving them.
    """
    if records is None:
        return PinReport((), (), (), store_unavailable=True, task_id=task_id)

    buckets: dict[PinClass, list[str]] = {
        PinClass.DEAD_L0: [],
        PinClass.QUEUE_HANDOFF: [],
        PinClass.NON_PINNING: [],
    }
    for record in records:
        pin_class = _classify_record(
            record, live_claimant=live_claimant, live_claimant_id=live_claimant_id,
        )
        buckets[pin_class].append(record.id)

    return PinReport(
        tuple(buckets[PinClass.DEAD_L0]),
        tuple(buckets[PinClass.QUEUE_HANDOFF]),
        tuple(buckets[PinClass.NON_PINNING]),
        task_id=task_id,
    )
