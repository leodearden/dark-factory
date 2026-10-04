"""Mint the next link of a recurring deterministic chain (task 4866 r2).

docs/prds/recurring-deterministic-tasks.md R-D2/R-D3: completing a carrier
link ``done`` mints exactly one pending successor (C-3), never while another
non-terminal link of the same chain exists (C-2), firing ``interval_secs``
after the predecessor's terminal time (C-6). A mint failure is logged and
never fails the status write that triggered it.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import StrEnum
from typing import Any

from shared.task_metadata import Recurrence
from shared.task_statuses import ACTIVE, TaskStatus

from fused_memory.backends.task_backend_protocol import TaskBackendProtocol
from fused_memory.middleware.deterministic_task_guard import deterministic_task_error

__all__ = [
    'RECURRENCE_MINT_SOURCE',
    'MintOutcome',
    'MintOutcomeKind',
    'mint_successor',
    'mints_successor',
]

logger = logging.getLogger(__name__)

RECURRENCE_MINT_SOURCE = 'recurrence-mint'


class MintOutcomeKind(StrEnum):
    MINTED = 'minted'
    SKIPPED_EXISTING_LINK = 'skipped_existing_link'
    FAILED = 'failed'


@dataclass(frozen=True)
class MintOutcome:
    kind: MintOutcomeKind
    successor_id: str | None = None
    detail: str | None = None


@dataclass(frozen=True)
class _SuccessorSpec:
    title: str
    description: str | None
    details: str | None
    priority: str | None
    metadata: dict[str, Any]


def _metadata_of(task: object) -> Mapping[str, Any]:
    metadata = task.get('metadata') if isinstance(task, Mapping) else None
    return metadata if isinstance(metadata, Mapping) else {}


def _recurrence_of(task: object) -> Mapping[str, Any]:
    recurrence = _metadata_of(task).get('recurrence')
    return recurrence if isinstance(recurrence, Mapping) else {}


def mints_successor(new_status: str, predecessor: object) -> bool:
    """True iff writing *new_status* onto *predecessor* must mint the chain's next link.

    Only ``done`` on a recurrence carrier qualifies: ``cancelled`` ends the
    chain and a non-terminal status leaves it alone (C-2).
    """
    return new_status == TaskStatus.DONE and bool(_recurrence_of(predecessor))


def _run_label(at: str) -> str:
    return f' [due {at}]'


def _successor_title(pred_title: str, pred_at: object, at: str) -> str:
    """The chain's base title plus the successor's own run label.

    The store's candidate_key UNIQUE index covers done rows, so a verbatim
    title copy would collide with the predecessor. The predecessor's label is
    recomputed from its milestone.at and removed, so labels never accumulate.
    """
    base = pred_title.removesuffix(_run_label(pred_at)) if isinstance(pred_at, str) else pred_title
    return base + _run_label(at)


def _build_successor(
    predecessor: Mapping[str, Any], predecessor_id: str, terminal_time: datetime,
) -> _SuccessorSpec:
    pred_metadata = _metadata_of(predecessor)
    recurrence = Recurrence.model_validate(pred_metadata['recurrence'])
    at = (terminal_time + timedelta(seconds=recurrence.interval_secs)).isoformat(timespec='seconds')
    metadata: dict[str, Any] = {
        'task_kind': pred_metadata.get('task_kind'),
        'before_done': pred_metadata.get('before_done'),
        'milestone': {'mode': 'dated', 'at': at},
        'recurrence': {
            'key': recurrence.key,
            'interval_secs': recurrence.interval_secs,
            'minted_from': predecessor_id,
        },
        'source': RECURRENCE_MINT_SOURCE,
    }
    if 'files' in pred_metadata:
        metadata['files'] = pred_metadata['files']
    pred_milestone = pred_metadata.get('milestone')
    pred_at = pred_milestone.get('at') if isinstance(pred_milestone, Mapping) else None
    return _SuccessorSpec(
        title=_successor_title(str(predecessor.get('title') or ''), pred_at, at),
        description=predecessor.get('description'),
        details=predecessor.get('details'),
        priority=predecessor.get('priority'),
        metadata=metadata,
    )


async def _existing_link_id(
    tm: TaskBackendProtocol, project_root: str, tag: str | None, key: str,
) -> str | None:
    active = await tm.get_tasks(project_root, tag, statuses=sorted(s.value for s in ACTIVE))
    for task in active['tasks']:
        if _recurrence_of(task).get('key') == key:
            return str(task['id'])
    return None


def _failed(predecessor_id: str, key: object, reason: str, *, exc_info: bool = False) -> MintOutcome:
    logger.error(
        'recurrence_mint_failed: predecessor=%s key=%s reason=%s',
        predecessor_id, key, reason, exc_info=exc_info,
    )
    return MintOutcome(MintOutcomeKind.FAILED, detail=reason)


async def _mint(
    tm: TaskBackendProtocol,
    *,
    predecessor: Mapping[str, Any],
    predecessor_id: str,
    project_root: str,
    tag: str | None,
    terminal_time: datetime,
) -> MintOutcome:
    spec = _build_successor(predecessor, predecessor_id, terminal_time)
    key = spec.metadata['recurrence']['key']
    # The carrier contract is checked at submit time only (task 3093), so the
    # link being renewed is re-verified rather than trusted.
    carrier_error = deterministic_task_error(spec.metadata['task_kind'], spec.metadata, project_root)
    if carrier_error is not None:
        return _failed(predecessor_id, key, str(carrier_error.get('error')))
    existing_id = await _existing_link_id(tm, project_root, tag, key)
    if existing_id is not None:
        logger.info(
            'recurrence_mint_skipped: predecessor=%s key=%s existing_link=%s',
            predecessor_id, key, existing_id,
        )
        return MintOutcome(MintOutcomeKind.SKIPPED_EXISTING_LINK, detail=existing_id)
    added = await tm.add_task(
        project_root=project_root,
        title=spec.title,
        description=spec.description,
        details=spec.details,
        priority=spec.priority,
        metadata=json.dumps(spec.metadata),
        tag=tag,
        status=TaskStatus.PENDING,
    )
    return MintOutcome(MintOutcomeKind.MINTED, successor_id=str(added['id']))


async def mint_successor(
    tm: TaskBackendProtocol,
    *,
    predecessor: Mapping[str, Any],
    predecessor_id: str,
    project_root: str,
    tag: str | None,
    terminal_time: datetime,
) -> MintOutcome:
    """Mint *predecessor*'s successor link; never raises (cancellation aside).

    The caller holds the project's write lock, so the C-2 scan and the insert
    are atomic with respect to other interceptor writers. Any failure is
    returned as ``FAILED`` after one ``recurrence_mint_failed:`` ERROR line.
    """
    try:
        return await _mint(
            tm,
            predecessor=predecessor,
            predecessor_id=predecessor_id,
            project_root=project_root,
            tag=tag,
            terminal_time=terminal_time,
        )
    except Exception as exc:
        key = _recurrence_of(predecessor).get('key')
        return _failed(predecessor_id, key, f'{type(exc).__name__}: {exc}', exc_info=True)
