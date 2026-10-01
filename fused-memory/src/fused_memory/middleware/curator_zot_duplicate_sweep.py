"""Post-ZOT duplicate sweep: flag near-duplicates the degraded curator let through.

When the task-curator's LLM call hangs with zero output until its timeout (a
ZOT), curator dedupe silently degrades to ``action='create'`` and the caller
sees an ordinary ``created`` result. Two of five confirmed degrade windows
produced a real duplicate. This module is the live owner of that class: for a
task created under a ZOT degrade it runs the curator's own corpus query,
excluding the new task, and names the best surviving near-duplicate.

Detection only. Nothing here combines, cancels or deletes; the caller stamps
the new task and files an operator-visible escalation.

Fail-SAFE direction: an uncertain or errored read is never reported as a
duplicate, because a false duplicate costs human attention. A hit whose status
cannot be confirmed as a live ``TaskStatus`` (an orphan corpus point, the
backend's NULL sentinel) is dropped, and any search or status-read error
yields no finding.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

from shared.task_statuses import TaskStatus

_FLAGGABLE_STATUSES = frozenset(TaskStatus) - {TaskStatus.CANCELLED}


@dataclass(frozen=True)
class DuplicateFinding:
    task_id: str
    duplicate_task_id: str
    duplicate_title: str
    score: float


def _numeric_score(hit: Mapping[str, Any]) -> float | None:
    score = hit.get('score')
    if isinstance(score, bool) or not isinstance(score, int | float):
        return None
    return float(score)


def select_near_duplicate(
    hits: Iterable[Mapping[str, Any]],
    *,
    self_task_id: str,
    statuses: Mapping[str, str],
    threshold: float,
) -> DuplicateFinding | None:
    """Return the highest-scoring flaggable hit, or ``None``.

    A hit is flaggable iff it is well-formed, is not the new task itself, has
    a live non-cancelled status in ``statuses`` and scores ``>= threshold``.
    """
    best: DuplicateFinding | None = None
    for hit in hits:
        raw_id = hit.get('task_id')
        score = _numeric_score(hit)
        if not raw_id or score is None:
            continue
        hit_id = str(raw_id)
        if hit_id == str(self_task_id):
            continue
        if statuses.get(hit_id) not in _FLAGGABLE_STATUSES:
            continue
        if score < threshold:
            continue
        if best is None or score > best.score:
            best = DuplicateFinding(
                task_id=str(self_task_id),
                duplicate_task_id=hit_id,
                duplicate_title=str(hit.get('title') or ''),
                score=score,
            )
    return best
