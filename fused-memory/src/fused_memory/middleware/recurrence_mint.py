"""Mint the next link of a recurring deterministic chain (task 4866 r2).

docs/prds/recurring-deterministic-tasks.md R-D2/R-D3: completing a carrier
link ``done`` mints exactly one pending successor (C-3), never while another
non-terminal link of the same chain exists (C-2), firing ``interval_secs``
after the predecessor's persisted terminal time (C-6). A mint failure is
logged, never fails the status write that triggered it, and is repaired by
the re-mint sweep.
"""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import Awaitable, Callable, Iterable, Mapping
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import StrEnum
from typing import Any

from shared.recurrence_chains import current_links, recurrence_key
from shared.task_metadata import Recurrence
from shared.task_statuses import ACTIVE, TaskStatus

from fused_memory.backends.task_backend_protocol import TaskBackendProtocol
from fused_memory.middleware.deterministic_task_guard import deterministic_task_error
from fused_memory.models.scope import is_phantom_project_root

__all__ = [
    'RECURRENCE_MINT_SOURCE',
    'REMINT_SWEEP_INTERVAL_SECS',
    'BrokenChain',
    'MintOutcome',
    'MintOutcomeKind',
    'MintTrigger',
    'broken_chains',
    'mint_successor',
    'mints_successor',
    'remint_broken_chain',
    'run_remint_sweep_loop',
]

logger = logging.getLogger(__name__)

RECURRENCE_MINT_SOURCE = 'recurrence-mint'
REMINT_SWEEP_INTERVAL_SECS: float = 3600.0


class MintOutcomeKind(StrEnum):
    MINTED = 'minted'
    SKIPPED_EXISTING_LINK = 'skipped_existing_link'
    SKIPPED_NOT_BROKEN = 'skipped_not_broken'
    FAILED = 'failed'


@dataclass(frozen=True)
class MintOutcome:
    kind: MintOutcomeKind
    successor_id: str | None = None
    detail: str | None = None


class MintTrigger(StrEnum):
    DONE_WRITE = 'done_write'
    REMINT_SWEEP = 'remint_sweep'


@dataclass(frozen=True)
class BrokenChain:
    """A chain whose current link *link_id* is done, with no live link of *key* in *tag*."""

    key: str
    tag: str
    link_id: str


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


def mints_successor(new_status: str, predecessor: object) -> bool:
    """True iff writing *new_status* onto *predecessor* must mint the chain's next link.

    Only ``done`` on a chain link qualifies: ``cancelled`` ends the chain and
    a non-terminal status leaves it alone (C-2).
    """
    return (
        new_status == TaskStatus.DONE
        and isinstance(predecessor, Mapping)
        and recurrence_key(predecessor) is not None
    )


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


def _build_successor(predecessor: Mapping[str, Any], predecessor_id: str) -> _SuccessorSpec:
    pred_metadata = _metadata_of(predecessor)
    recurrence = Recurrence.model_validate(pred_metadata['recurrence'])
    terminal_time = datetime.fromisoformat(predecessor['updatedAt'])
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
        if recurrence_key(task) == key:
            return str(task['id'])
    return None


def _failed(
    predecessor_id: str, predecessor: Mapping[str, Any], reason: str, *, exc_info: bool = False,
) -> MintOutcome:
    logger.error(
        'recurrence_mint_failed: predecessor=%s key=%s reason=%s',
        predecessor_id, recurrence_key(predecessor), reason, exc_info=exc_info,
    )
    return MintOutcome(MintOutcomeKind.FAILED, detail=reason)


async def _mint(
    tm: TaskBackendProtocol,
    *,
    predecessor: Mapping[str, Any],
    predecessor_id: str,
    project_root: str,
    tag: str | None,
) -> MintOutcome:
    spec = _build_successor(predecessor, predecessor_id)
    key = spec.metadata['recurrence']['key']
    # The carrier contract is checked at submit time only (task 3093), so the
    # link being renewed is re-verified rather than trusted. The check stats
    # the before_done script, so it runs off the event loop.
    carrier_error = await asyncio.to_thread(
        deterministic_task_error, spec.metadata['task_kind'], spec.metadata, project_root,
    )
    if carrier_error is not None:
        return _failed(predecessor_id, predecessor, str(carrier_error.get('error')))
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
    write_lock: asyncio.Lock,
    predecessor_id: str,
    project_root: str,
    tag: str | None,
) -> MintOutcome:
    """Mint the completed link *predecessor_id*'s successor; never fails the caller.

    Under *write_lock* (the project's interceptor write lock), the completed
    link is read back, so its persisted ``updatedAt`` anchors the successor's
    cadence and the C-2 scan plus the insert are atomic with respect to other
    interceptor writers. Any failure is returned as ``FAILED`` after one
    ``recurrence_mint_failed:`` ERROR line; cancellation logs that same line
    and is re-raised.
    """
    return await _locked_mint(
        tm,
        write_lock=write_lock,
        predecessor_id=predecessor_id,
        project_root=project_root,
        tag=tag,
        precheck=None,
    )


async def _locked_mint(
    tm: TaskBackendProtocol,
    *,
    write_lock: asyncio.Lock,
    predecessor_id: str,
    project_root: str,
    tag: str | None,
    precheck: Callable[[], Awaitable[MintOutcome | None]] | None,
) -> MintOutcome:
    """:func:`mint_successor`, first running *precheck* under the lock; its outcome skips the mint."""
    predecessor: Mapping[str, Any] = {}
    try:
        async with write_lock:
            if precheck is not None and (skipped := await precheck()) is not None:
                return skipped
            predecessor = await tm.get_task(predecessor_id, project_root, tag)
            return await _mint(
                tm,
                predecessor=predecessor,
                predecessor_id=predecessor_id,
                project_root=project_root,
                tag=tag,
            )
    except asyncio.CancelledError:
        _failed(predecessor_id, predecessor, 'cancelled')
        raise
    except Exception as exc:
        return _failed(predecessor_id, predecessor, f'{type(exc).__name__}: {exc}', exc_info=True)


async def broken_chains(tm: TaskBackendProtocol, project_root: str) -> tuple[BrokenChain, ...]:
    """Every chain in *project_root*, across all tags, whose current link is done.

    A done current link means its mint failed or never ran (a crash between
    the done write and the mint). Read-only and lock-free: each repair
    re-reads its chain under the write lock.
    """
    broken: list[BrokenChain] = []
    for tag in await tm.list_tags(project_root):
        broken.extend(await _broken_chains_in_tag(tm, project_root, tag))
    return tuple(broken)


async def _broken_chains_in_tag(
    tm: TaskBackendProtocol, project_root: str, tag: str,
) -> list[BrokenChain]:
    rows = (await tm.get_tasks(project_root, tag))['tasks']
    return [
        BrokenChain(key=key, tag=tag, link_id=str(link['id']))
        for key, link in current_links(rows).items()
        if link['status'] == TaskStatus.DONE
    ]


async def _skip_unless_still_broken(
    tm: TaskBackendProtocol, chain: BrokenChain, project_root: str,
) -> MintOutcome | None:
    if chain in await _broken_chains_in_tag(tm, project_root, chain.tag):
        return None
    logger.info(
        'recurrence_remint_skipped: predecessor=%s key=%s reason=no_longer_broken',
        chain.link_id, chain.key,
    )
    return MintOutcome(MintOutcomeKind.SKIPPED_NOT_BROKEN)


async def remint_broken_chain(
    tm: TaskBackendProtocol,
    *,
    write_lock: asyncio.Lock,
    chain: BrokenChain,
    project_root: str,
) -> MintOutcome:
    """Mint *chain*'s missing successor if it is still broken under *write_lock*, logging each repair.

    *chain* came from a lock-free read, so its tag is read again under the
    lock: a link reopened, renewed or superseded since then is left alone.
    """
    outcome = await _locked_mint(
        tm,
        write_lock=write_lock,
        predecessor_id=chain.link_id,
        project_root=project_root,
        tag=chain.tag,
        precheck=lambda: _skip_unless_still_broken(tm, chain, project_root),
    )
    if outcome.kind is MintOutcomeKind.MINTED:
        logger.warning(
            'recurrence_remint: predecessor=%s key=%s successor=%s project_root=%s tag=%s',
            chain.link_id, chain.key, outcome.successor_id, project_root, chain.tag,
        )
    return outcome


async def run_remint_sweep_loop(
    remint: Callable[[str], Awaitable[object]],
    project_roots: Iterable[str],
    *,
    interval_secs: float = REMINT_SWEEP_INTERVAL_SECS,
) -> None:
    """Sweep every non-phantom root at once, then every *interval_secs*, until cancelled.

    A phantom root is skipped because reading it would create its task store.
    """
    roots = tuple(project_roots)
    while True:
        for root in roots:
            if await asyncio.to_thread(is_phantom_project_root, root):
                continue
            try:
                await remint(root)
            except Exception:
                logger.exception('recurrence_remint_sweep_failed: project_root=%s', root)
        await asyncio.sleep(interval_secs)
