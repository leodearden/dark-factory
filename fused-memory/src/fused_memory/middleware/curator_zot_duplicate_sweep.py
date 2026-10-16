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

import logging
from collections.abc import Awaitable, Callable, Iterable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Protocol

from shared.task_statuses import TaskStatus

from fused_memory.middleware.task_curator import embedding_text

logger = logging.getLogger(__name__)

DUPLICATE_METADATA_KEY = 'x_zot_duplicate_candidate'

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


class CorpusSearcher(Protocol):
    async def search_corpus(
        self,
        query: str,
        project_id: str,
        *,
        limit: int = ...,
        score_threshold: float = ...,
    ) -> list[dict[str, Any]]: ...


async def sweep_zot_duplicate(
    curator: CorpusSearcher,
    *,
    project_id: str,
    task_id: str,
    title: str,
    description: str,
    files_to_modify: list[str],
    read_statuses: Callable[[list[str]], Awaitable[Mapping[str, str]]],
    threshold: float,
    limit: int,
) -> DuplicateFinding | None:
    """Search the curator corpus for a near-duplicate of a ZOT-degraded create.

    Statuses are read only for the non-self hit ids, so a sweep with nothing
    to flag performs no status read. Any error other than cancellation yields
    ``None``: an errored read is never evidence of a duplicate.
    """
    try:
        hits = await curator.search_corpus(
            embedding_text(title, description, files_to_modify),
            project_id,
            limit=limit,
            score_threshold=threshold,
        )
        candidate_ids = sorted({
            str(hit['task_id'])
            for hit in hits
            if hit.get('task_id') and str(hit['task_id']) != str(task_id)
        })
        if not candidate_ids:
            return None
        statuses = await read_statuses(candidate_ids)
    except Exception:
        logger.warning(
            'zot duplicate sweep failed for project=%s task=%s; reporting no finding',
            project_id, task_id, exc_info=True,
        )
        return None
    return select_near_duplicate(
        hits, self_task_id=task_id, statuses=statuses, threshold=threshold,
    )


def build_duplicate_metadata(
    finding: DuplicateFinding, *, zot_escalation_id: str | None,
) -> dict[str, dict[str, Any]]:
    """Return the ``metadata`` patch that stamps a finding onto the new task."""
    return {
        DUPLICATE_METADATA_KEY: {
            'duplicate_task_id': finding.duplicate_task_id,
            'duplicate_title': finding.duplicate_title,
            'score': float(finding.score),
            'zot_escalation_id': zot_escalation_id,
            'flagged_at': datetime.now(UTC).isoformat(),
        },
    }
