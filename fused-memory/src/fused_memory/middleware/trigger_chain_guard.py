"""One open trigger chain per (project, skill), refused at ``submit_task``.

PRD plans/completion-driven-triggers-prd.md C2: a chain task is refused while
another run of the same skill has a non-terminal, unsuperseded chain task in
the project. When the store read raises, the guard fails OPEN with a WARNING,
under the ``no-silent-fail-soft`` waiver recorded in the PRD's §10 γ row: the
cost is one duplicate chain, which the next run's supersede retires. A page the
guard cannot collate is a code-contract break, not a store error, and raises.
Two concurrent chain submissions can both pass the read (TOCTOU), the bounded
residual task 3588's guard also carries.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

from pydantic import ValidationError
from shared.task_metadata import TRIGGER_CHAIN_KEY, TriggerChain
from shared.task_statuses import ACTIVE, TERMINAL

from fused_memory.backends.metadata_lookup_pages import (
    LOOKUP_PAGE_SIZE,
    MetadataLookupSource,
    metadata_lookup_pages,
)
from fused_memory.backends.task_backend_types import MetadataLookupResult, MetadataLookupRow
from fused_memory.middleware.metadata_dict import raw_metadata_dict

__all__ = [
    'OpenChainTask',
    'find_open_chains',
    'trigger_chain_open_error',
    'trigger_chain_open_refusal',
]

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class OpenChainTask:
    id: str
    role: str
    status: str
    run_id: str

    def as_dict(self) -> dict[str, str]:
        return {'id': self.id, 'role': self.role, 'status': self.status, 'run_id': self.run_id}


def find_open_chains(
    rows: Iterable[MetadataLookupRow], submission: TriggerChain
) -> tuple[OpenChainTask, ...]:
    """The rows holding an open chain of *submission*'s skill from another run."""
    open_chain: list[OpenChainTask] = []
    for row in rows:
        chain = _stored_chain(row)
        if chain is None or not _is_rival(chain, row['status'], submission):
            continue
        open_chain.append(OpenChainTask(str(row['task_id']), chain.role, row['status'], chain.run_id))
    return tuple(open_chain)


def _stored_chain(row: MetadataLookupRow) -> TriggerChain | None:
    try:
        return TriggerChain.model_validate(row.get('matched_value'))
    except (ValidationError, TypeError):
        logger.warning(
            'trigger_chain_guard: task %s carries an unreadable trigger_chain; skipped',
            row['task_id'],
        )
        return None


def _is_rival(chain: TriggerChain, status: str, submission: TriggerChain) -> bool:
    # TERMINAL is re-checked although the query filters it, as find_open_gate does.
    return (
        chain.skill == submission.skill
        and chain.run_id != submission.run_id
        and chain.superseded_by is None
        and status not in TERMINAL
    )


def trigger_chain_open_refusal(
    open_chain: tuple[OpenChainTask, ...], submission: TriggerChain
) -> dict[str, Any]:
    """The structured ``TriggerChainOpen`` refusal naming *open_chain*."""
    run_ids = sorted({task.run_id for task in open_chain})
    task_ids = [task.id for task in open_chain]
    return {
        'error': (
            f'an open {submission.skill!r} trigger chain already exists in this project: '
            f'run {", ".join(run_ids)} on tasks {", ".join(task_ids)}. A project holds one '
            f'open chain per skill, so run {submission.run_id!r} cannot file its own yet.'
        ),
        'error_type': 'TriggerChainOpen',
        'open_chain': [task.as_dict() for task in open_chain],
        'hint': (
            'Supersede the open chain first (plans/completion-driven-triggers-prd.md C5; '
            'the calls are in skills/_shared/filing-the-trigger-chain.md §5): on each '
            f"listed task, update_task metadata.trigger_chain.superseded_by={submission.run_id!r}, "
            'writing the whole trigger_chain dict because metadata merge is shallow; retire '
            'the old human gate before the old completion gate; then resubmit.'
        ),
    }


async def trigger_chain_open_error(
    metadata: str | dict[str, Any] | None,
    project_root: str,
    *,
    source: MetadataLookupSource,
    tag: str | None = None,
) -> dict[str, Any] | None:
    """Refuse a chain task while another run's chain of the same skill is open.

    ``None`` for a submission without a valid ``trigger_chain`` (the
    deterministic guard owns its shape) and, failing open, when the store read
    raises.
    """
    submission = _submitted_chain(metadata)
    if submission is None:
        return None
    try:
        rows = await _read_chain_rows(_StoreReadErrorSource(source), project_root, tag)
    except _StoreReadError:
        logger.warning(
            'trigger_chain_guard fail-open: the trigger_chain read failed for %s; '
            'the submission is accepted unchecked',
            project_root,
            exc_info=True,
        )
        return None
    open_chain = find_open_chains(rows, submission)
    if not open_chain:
        return None
    logger.warning(
        'trigger_chain_guard: refused %r chain run %s; open chain tasks %s',
        submission.skill,
        submission.run_id,
        [task.id for task in open_chain],
    )
    return trigger_chain_open_refusal(open_chain, submission)


def _submitted_chain(metadata: str | dict[str, Any] | None) -> TriggerChain | None:
    raw = raw_metadata_dict(metadata, source='trigger_chain_guard').get(TRIGGER_CHAIN_KEY)
    if raw is None:
        return None
    try:
        return TriggerChain.model_validate(raw)
    except (ValidationError, TypeError):
        return None


class _StoreReadError(Exception):
    """The store's own ``find_tasks_by_metadata`` raised; the one failure the guard fails open on."""


@dataclass(frozen=True)
class _StoreReadErrorSource:
    """*source*, with what its read raises re-raised as :class:`_StoreReadError`.

    A call the source's signature refuses raises before the read starts, so a
    drifted signature propagates rather than failing the guard open.
    """

    source: MetadataLookupSource

    async def find_tasks_by_metadata(
        self,
        project_root: str,
        key: str,
        value: str | int | None = None,
        *,
        statuses: list[str] | None = None,
        limit: int = LOOKUP_PAGE_SIZE,
        offset: int = 0,
        tag: str | None = None,
        include_value: bool = False,
    ) -> MetadataLookupResult:
        read = self.source.find_tasks_by_metadata(
            project_root,
            key,
            value,
            statuses=statuses,
            limit=limit,
            offset=offset,
            tag=tag,
            include_value=include_value,
        )
        try:
            return await read
        except Exception as exc:
            raise _StoreReadError from exc


async def _read_chain_rows(
    source: MetadataLookupSource, project_root: str, tag: str | None
) -> list[MetadataLookupRow]:
    rows: list[MetadataLookupRow] = []
    total = 0
    async for page in metadata_lookup_pages(
        source,
        project_root,
        TRIGGER_CHAIN_KEY,
        statuses=sorted(ACTIVE),
        include_value=True,
        tag=tag,
    ):
        if page['malformed_metadata_ids']:
            logger.warning(
                'trigger_chain_guard: tasks %s carry unreadable metadata; not checked',
                page['malformed_metadata_ids'],
            )
        rows.extend(page['results'])
        total = page['total']
    if len(rows) < total:
        logger.warning(
            'trigger_chain_guard: incomplete read, %d of %d trigger_chain carriers',
            len(rows),
            total,
        )
    return rows
