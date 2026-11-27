"""The human-curator gate marker's one reading, and the pin its escalation carries.

A deterministic pure gate marked ``metadata.human_curator_gate`` closes only on
a human's CONTENT adjudication of every entry it names (task 3341).  The runner
files its escalation with a declared pin (task 3346), so a resolver cannot close
it without naming it in ``acknowledge_declared_pins``.  Every reader of either
fact keys on this module: the runner that files the pin, and the migration that
lists the resume spending it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from shared.task_metadata import HUMAN_CURATOR_GATE_KEY

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping

__all__ = [
    'curator_gate_declarer',
    'declares_only_the_curator_gate_pin',
    'is_human_curator_gate',
]


def is_human_curator_gate(metadata: Mapping[str, object]) -> bool:
    """True iff *metadata* marks a gate needing human CONTENT adjudication.

    Plain TRUTHINESS, not ``is True``.  This is a SAFETY marker: a false
    NEGATIVE silently reproduces the task-3181 phantom-done, so a
    truthy-but-not-``True`` value (the string ``'true'`` from a hand edit or a
    JSON round-trip) must fail CLOSED and still count as a curator gate.
    """
    return bool(metadata.get(HUMAN_CURATOR_GATE_KEY))


def curator_gate_declarer(task_id: str) -> str:
    """The ``pin_declared_by`` entry the runner files on curator gate *task_id*'s escalation."""
    return f'task-{task_id}-human-curator-gate'


def declares_only_the_curator_gate_pin(declared_by: Iterable[str], task_id: str) -> bool:
    """True iff every non-blank declarer in *declared_by* is the runner's curator pin for *task_id*.

    An empty declaration answers True: there is no other declarer to consult.
    Any other declarer answers False, because acknowledgement is per record, so
    spending the curator pin would spend that declarer's pin unread.
    """
    return {entry.strip() for entry in declared_by if entry.strip()} <= {curator_gate_declarer(task_id)}
