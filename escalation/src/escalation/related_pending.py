"""The sideways census of a resolve: which pending records are its twins (task 4886).

Resolving one record says nothing about the others carrying the same question:
another pending record on the same task, or a pending L2 clustering a member
the resolved record shares. This lists them so the resolver can dispose of
them in the same sitting.

REPORT-ONLY: it mutates nothing, and its caller must not either. Nothing
auto-closes on this evidence, because a pin (esc-3105-3) is indistinguishable
from an answered question on member evidence alone; see the "Ruled-elsewhere
check" in ``skills/escalation-watcher/SKILL.md``.

Deliberately category-agnostic. PENDING-ONLY, so ``[]`` means "no pending
twins", never "no such record ever existed".

The resolved record's own id is a match key too: an L1 resolved directly while
a pending L2 still clusters it is the same twin relation seen from the member.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import TypedDict

from escalation.models import Escalation


class RelatedPendingEntry(TypedDict):
    id: str
    category: str
    severity: str
    level: int
    same_task: bool
    shared_member: str | None  # smallest id shared with the resolved record, if any


def related_pending(
    pending: Iterable[Escalation], *, resolved: Escalation,
) -> list[RelatedPendingEntry]:
    """The pending twins of *resolved* in *pending*, one entry per id, sorted by id."""
    match_keys = set(resolved.members) | {resolved.id}
    entries: dict[str, RelatedPendingEntry] = {}
    for candidate in pending:
        if candidate.id == resolved.id or candidate.status != 'pending':
            continue
        same_task = candidate.task_id == resolved.task_id
        shared = (
            match_keys.intersection(candidate.members) if candidate.level == 2 else set()
        )
        if not (same_task or shared):
            continue
        entries[candidate.id] = {
            'id': candidate.id,
            'category': candidate.category,
            'severity': candidate.severity,
            'level': candidate.level,
            'same_task': same_task,
            'shared_member': min(shared) if shared else None,
        }
    return [entries[esc_id] for esc_id in sorted(entries)]
