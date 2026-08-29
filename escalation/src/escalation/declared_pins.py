"""THE declared-pin predicate — protection AT RESOLUTION TIME (task 4377).

WHY THIS MODULE EXISTS.  An OPEN escalation record is a PRESERVATION MECHANISM
for its subject task: ``orchestrator/task_ground_truth.py::_RECOVERY`` has no
row for the pinned shape (IN_PROGRESS, no live claimant, exists off-main,
``has_open_escalation=True``), so it falls through to ``RecoveryAction.LEAVE``
and the row survives.  CLOSING the record flips that boolean and the same shape
recovers to REVERT_TO_PENDING.  So closing a record is a state-changing act on
its subject task even under ``action='close_only'`` — and nothing at resolve
time could previously tell that a record was deliberately relied upon.

THE ORIGIN INCIDENT.  On 2026-08-08 an L2 cascade close of the homogeneous
11-member cluster esc-3237-5 dismissed esc-3371-2, the only pin preserving
mu-gate validation specimen task 3371.  The specimen is permanently gone.  All
eleven members were indistinguishable by id, level, category, severity,
agent_role and summary; the one thing that made esc-3371-2 different lived in
prose that nothing linked from.  ``Escalation.pin_declared_by`` is the
machine-readable marker that closes that gap, and this module is the predicate
that reads it.

NOT ``escalation.pins.classify_pins``.  That module answers "does this
ALREADY-OPEN record veto recovery?" from an automatic severity/level/filing-
incarnation policy — CLASSIFICATION AT RECOVERY TIME — and it protects nothing
from being CLOSED.  This module answers a different question, driven by an
explicit human/operator declaration rather than by policy: "may a resolver
close this record at all?".  Folding the two together would blur a module whose
docstring is emphatic about its single question, so they are separate seams by
design.

PURE by construction: no I/O, no store binding, no mutation of its inputs — the
same contract ``escalation.pins`` states, for the same reason.  The CALLER
binds the read (``escalation/server.py::resolve_issue`` builds the candidate
list from the already-fetched target plus each readable cascade member) and
passes the records in.  Keeping the predicate pure is also what lets a SECOND
close path adopt the same rule later by importing it, instead of forking a
divergent copy.

ON THE INPUT TYPE.  The parameter is a concrete ``Sequence[Escalation]``, not a
structural Protocol.  ``pins.py::PinRecord`` exists because a genuine SECOND
implementer in another package (``orchestrator.task_ground_truth.EscalationRef``)
must satisfy it without ``escalation`` importing ``orchestrator`` — a layering
inversion.  There is no second implementer here: the marker is written and read
entirely inside the escalation store, so an unused Protocol would be a
legibility cost with no consumer.  If a cross-package caller ever appears, the
Protocol is the right answer then.

THE ENFORCEMENT SEAM is ``escalation/server.py::resolve_issue``, which consults
:func:`blocking_pin_declarations` as a PRE-FLIGHT over the target AND every
cascade member, before any record is mutated.  ``queue.resolve()`` deliberately
only WARNS (it archives an L2 head before cascading, so a refusal raised there
could only ever half-close a cluster).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Collection, Sequence

    from escalation.models import Escalation

__all__ = ['PinDeclaration', 'blocking_pin_declarations', 'format_refusal']


@dataclass(frozen=True)
class PinDeclaration:
    """One record's declared dependency — a frozen snapshot, not a live view.

    Carries a TUPLE of declarers (not a list) for the same reason
    ``pins.PinReport`` does: the value object is then genuinely immutable and no
    shared consumer can mutate it in place, and it stays directly serialisable
    into the structured ``declared_pins`` payload the refusal returns.
    """

    #: The record whose close is being refused.
    escalation_id: str
    #: WHAT relies on that record staying OPEN — a deviation notice, an
    #: operator gate ('task-3546-second-deviation-notice').  NOT who stamped it.
    #: Normalised: stripped, blanks dropped, declaration order preserved.
    declared_by: tuple[str, ...]
    #: The free-text WHY, verbatim.  May be empty — the declarer list is the
    #: marker, prose alone is not.
    reason: str


def blocking_pin_declarations(
    records: Sequence[Escalation],
    *,
    acknowledged: Collection[str] = (),
) -> tuple[PinDeclaration, ...]:
    """Return one :class:`PinDeclaration` per marked record not yet acknowledged.

    :param records: the candidate records — for a resolve, the target plus every
        readable member of its L2 cluster.  The CALLER binds this read; a member
        the store cannot return is simply absent from the sequence (a record
        that does not exist cannot carry a marker).
    :param acknowledged: escalation ids whose declared pin the caller is
        deliberately spending.  An id that is not blocked is a harmless no-op.
        A PARTIAL acknowledgement still leaves the remainder blocking — that is
        the property that stops a bulk closer from waving a homogeneous cluster
        through on the one id an error message happened to mention first.

    A record is MARKED when ``pin_declared_by`` holds at least one non-blank
    entry.  Blank/whitespace-only declarers are stripped out, and a record left
    with none is not a declaration: an all-blank marker declares nothing.
    ``pin_declared_reason`` is never itself a marker — a reason with no declarer
    names nothing a closer could go consult, so it blocks nothing.

    Input order is preserved, so a refusal reports the target before its members
    and members in cluster order.

    PURE: performs no I/O and does not mutate *records*.  The returned tuples
    are snapshots, so a later append to a record's list does not retroactively
    change a declaration already handed to a caller.
    """
    ack = set(acknowledged)
    declarations: list[PinDeclaration] = []

    for record in records:
        if record.id in ack:
            continue
        declarers = tuple(
            stripped for entry in record.pin_declared_by if (stripped := entry.strip())
        )
        if not declarers:
            continue
        declarations.append(
            PinDeclaration(
                escalation_id=record.id,
                declared_by=declarers,
                reason=record.pin_declared_reason,
            )
        )

    return tuple(declarations)


def format_refusal(declarations: Sequence[PinDeclaration]) -> str:
    """Build the human half of the refusal for *declarations*.

    Names EVERY blocked escalation id with its declarers and reason, and names
    ``acknowledge_declared_pins`` as the deliberate override — so the message
    alone tells a closer what declared the dependency and how to proceed.

    This is the HUMAN half only.  The refusal also returns the same facts
    structurally (a ``declared_pins`` list of
    ``{escalation_id, declared_by, reason}`` dicts), per INV-2: no consumer
    should have to parse prose to recover a fact the emitter held in a variable.
    Emitting the refusal as prose alone would rebuild the exact trap this task
    exists to close — a marker that lives only in text nothing links from.
    """
    lines = [
        'Refusing to close: a declared dependency relies on '
        f'{len(declarations)} record(s) staying OPEN. '
        'An open escalation preserves its subject task, so closing one of these '
        'is a state-changing act on that task.',
    ]
    for decl in declarations:
        detail = f'  - {decl.escalation_id}: declared_by={", ".join(decl.declared_by)}'
        if decl.reason:
            detail += f' — {decl.reason}'
        lines.append(detail)
    lines.append(
        'Go read what pin_declared_by names and consult it. To spend these pins '
        'deliberately, pass acknowledge_declared_pins naming EVERY id above '
        '(a partial acknowledgement still refuses).'
    )
    return '\n'.join(lines)
