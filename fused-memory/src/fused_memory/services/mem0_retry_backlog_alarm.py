"""Escalate a project whose Mem0 writes have been retrying too long: retries stay unbounded, because a cap would drop the very writes the retry exists to keep, so the backlog's age is the alarm."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Mapping
from datetime import datetime, timedelta
from typing import Any, Protocol

from fused_memory.middleware._folded_escalation import file_folded_escalation
from fused_memory.services.write_journal import Mem0RetryBacklog, WriteJournal

logger = logging.getLogger(__name__)

ANCHOR_TASK_ID = 'mem0-write-retry-backlog'
_AGENT_ROLE = 'fused-memory/mem0-write-retry'
_CATEGORY = 'mem0_write_retry_backlog'
LOG_EVENT = 'mem0_write_retry_backlog'
_EVIDENCE_QUERY = (
    'SELECT id, project_id, attempts, created_at, next_attempt_at, reason '
    "FROM mem0_intents WHERE status='retrying'"
)


class Mem0RetryBacklogChecker(Protocol):
    """Escalates its own Mem0 retry backlog once it is older than its policy allows."""

    async def check_mem0_retry_backlog(self) -> object: ...


def emit_mem0_retry_backlog_escalation(
    project_root: str,
    *,
    backlog: Mem0RetryBacklog,
    now: datetime,
    max_age: timedelta,
) -> str | None:
    """File *backlog* under the alarm's anchor, folding into an open record. Never raises."""
    age_seconds = int(backlog.age(now).total_seconds())
    threshold_seconds = int(max_age.total_seconds())
    detail = '\n'.join((
        f'project_id={backlog.project_id}',
        f'project_root={project_root}',
        f'count={backlog.count}',
        f'oldest_created_at={backlog.oldest_created_at.isoformat()}',
        f'age_seconds={age_seconds}',
        f'max_attempts={backlog.max_attempts}',
        f'threshold_seconds={threshold_seconds}',
        '',
        "These writes' callers were acked stores_pending_retry=['mem0']: Mem0 "
        'refused them on a transient provider error and has refused every '
        're-issue since. Nothing is lost while they stay retrying, because each '
        'mem0_intents row holds the full content and metadata and the startup '
        'prune never touches a retrying row; but they are unsearchable until '
        'they land.',
        '',
        f'Evidence (write_journal.db, read-only): {_EVIDENCE_QUERY}',
        '',
        'This is the ONE open record for this project until it is resolved: '
        'later backlog checks fold into it and file nothing of their own, so the '
        'figures above describe the check that filed it, not a running total.',
    ))

    def _log_fold(existing: Any) -> None:
        logger.debug(
            '%s: %s already open for project_id=%r; folding into it',
            LOG_EVENT, existing.id, backlog.project_id,
        )

    return file_folded_escalation(
        project_root,
        anchor_task_id=ANCHOR_TASK_ID,
        agent_role=_AGENT_ROLE,
        category=_CATEGORY,
        severity='blocking',
        summary=(
            f'{backlog.count} Mem0 write(s) for project_id={backlog.project_id!r} '
            f'still retrying after {age_seconds}s (threshold {threshold_seconds}s): '
            'acked as pending retry, unsearchable until they land'
        ),
        detail=detail,
        suggested_action=(
            'Check the OpenAI provider Mem0 embeds through (credits, quota, API '
            "status; grep the fused-memory log for 'Mem0 retry attempt'). The "
            'retry pass re-issues these writes by itself once the provider '
            'recovers, so no replay is needed: confirm they drain with the '
            f'read-only mem0_intents query {_EVIDENCE_QUERY} against '
            'write_journal.db, and resolve this record once it returns no rows.'
        ),
        logger=logger,
        log_label=LOG_EVENT,
        context=f'project_id={backlog.project_id!r}',
        on_fold=_log_fold,
    )


async def alarm_on_stale_mem0_retries(
    journal: WriteJournal,
    *,
    now: datetime,
    max_age: timedelta,
    project_roots: Mapping[str, str],
) -> list[str]:
    """Escalate each project whose oldest retrying Mem0 write is at least *max_age* old.

    Returns the filed or folded escalation ids. Never raises.
    """
    try:
        backlogs = await journal.get_mem0_retry_backlog()
    except Exception:
        logger.exception(
            '%s: could not read the retrying mem0 intents; no backlog checked', LOG_EVENT
        )
        return []
    filed: list[str] = []
    for backlog in backlogs:
        if backlog.age(now) < max_age:
            continue
        esc_id = await _escalate(backlog, now=now, max_age=max_age, project_roots=project_roots)
        if esc_id is not None:
            filed.append(esc_id)
    return filed


async def _escalate(
    backlog: Mem0RetryBacklog,
    *,
    now: datetime,
    max_age: timedelta,
    project_roots: Mapping[str, str],
) -> str | None:
    project_root = (
        project_roots.get(backlog.project_id) if backlog.project_id is not None else None
    )
    if not project_root:
        logger.warning(
            '%s: cannot resolve project_root for project_id=%r; NOT escalating %d '
            'Mem0 write(s) retrying since %s. Wire '
            'MemoryService.set_known_projects(build_known_projects_map(...)) at '
            'server startup to restore this alarm.',
            LOG_EVENT, backlog.project_id, backlog.count,
            backlog.oldest_created_at.isoformat(),
        )
        return None
    try:
        # The queue write fsyncs, so it runs off the event loop.
        return await asyncio.to_thread(
            emit_mem0_retry_backlog_escalation,
            project_root,
            backlog=backlog,
            now=now,
            max_age=max_age,
        )
    except Exception:
        logger.exception(
            '%s: filing the escalation for project_id=%r failed',
            LOG_EVENT, backlog.project_id,
        )
        return None
