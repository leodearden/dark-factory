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

...plus a distinguishable THIRD state, ``store_unavailable`` — see
:func:`classify_pins`.

This task delivers the types + classifier + tests ONLY.  Rewiring the veto
sites (``task_ground_truth._shape``, the harness reconcile sweeps, the
scheduler's stranded-blocked sweep) is task eta (3541); the structured
``escalation_store_unavailable`` emission is task beta (3535).
"""

from __future__ import annotations

import enum
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

from escalation.models import KNOWN_SEVERITIES

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

    ``filing_claimant_run_id`` carries the FILING incarnation's identity in
    ``shared.task_claimant.compose_claimant_run_id`` format
    (``{run_id}/{session_id}/pid={owner_pid}``); ``None`` means unknown.

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
#   4. level == 0, known non-info severity -> filing-incarnation liveness:
#        no incarnation live at all         -> DEAD_L0
#        live, and both identities known and DIFFERENT -> DEAD_L0
#        live, and identities MATCH         -> QUEUE_HANDOFF
#        live, either identity unknown      -> QUEUE_HANDOFF (fail-safe pin)
# ---------------------------------------------------------------------------


def _classify_record(
    record: PinRecord,
    *,
    live_claimant: bool,
    live_claimant_id: str | None,
) -> PinClass:
    """Map one open escalation to its :class:`PinClass` (see the chain above)."""
    # Normalise once. `severity` is read defensively (getattr + str()) because
    # records arrive from JSON on disk, where the field may be absent or null.
    sev = str(getattr(record, 'severity', '') or '').strip().lower()

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

    # Links 3 and 4 land in step-8; until then every known non-info severity
    # fails safe to pinning.
    return PinClass.QUEUE_HANDOFF


def classify_pins(
    task_id: str,
    records: Sequence[PinRecord] | None,
    *,
    live_claimant: bool,
    live_claimant_id: str | None = None,
) -> PinReport:
    """Classify *records* — the task's OPEN escalations — into pin classes.

    *records* is ``None`` when the escalation store could not be read (no queue
    bound, read failed).  See :class:`PinReport` for the buckets.
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
