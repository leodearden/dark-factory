"""Marks each record a write supersedes, at add_memory or when a deferred write's retry lands.

Runs at ``server/tools.py::add_memory`` after the write lands, is metadata-only, and never raises: by then the
correction exists, so failing the call would invent a write failure. A write
whose Mem0 leg is pending server-side retry has no id to mark with yet, so the
tool path reports its targets ``pending_retry`` and
:func:`mark_superseded_after_retry` marks them once the intent lands. It lives
outside ``server/write_triage.py`` because triage never edits a record (that
module's contract C1).
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING, Any

from fused_memory.memory_metadata import SUPERSEDED_BY_METADATA_KEY, normalize_supersedes
from fused_memory.models.memory import Mem0WriteState
from fused_memory.services.consolidation_ops import MemoryUpdater, patch_memory_metadata
from fused_memory.services.write_journal import decode_mem0_intent_metadata
from fused_memory.utils.validation import is_full_uuid

if TYPE_CHECKING:
    from fused_memory.models.memory import AddMemoryResponse
    from fused_memory.services.mem0_add_outcome import Mem0AddOutcome

__all__ = [
    'SUPERSESSION_ACK_KEY',
    'SUPERSESSION_RETRY_SOURCE',
    'SupersessionMarks',
    'UnmarkedReason',
    'UnmarkedSupersession',
    'mark_superseded',
    'mark_superseded_after_retry',
]

logger = logging.getLogger(__name__)

SUPERSESSION_ACK_KEY = 'supersession'

#: The journal source of a mark the retry pass applies; the intent row does not
#: journal the original write's source.
SUPERSESSION_RETRY_SOURCE = 'mem0_write_retry'


class UnmarkedReason(StrEnum):
    INVALID_MEMBER = 'invalid_supersedes_member'
    NO_SUPERSEDING_RECORD = 'no_superseding_mem0_record'
    NO_PROJECT_ID = 'no_project_id'
    PATCH_FAILED = 'patch_failed'


@dataclass(frozen=True)
class UnmarkedSupersession:
    memory_id: Any
    reason: UnmarkedReason
    error: str

    def to_ack(self) -> dict[str, Any]:
        return {'id': self.memory_id, 'reason': self.reason.value, 'error': self.error}


@dataclass(frozen=True)
class SupersessionMarks:
    """What one write's declared supersession did to the records it names."""

    #: The new record's Mem0 id, written as each mark's value; ``None`` when
    #: the write did not land exactly one Mem0 id, so nothing was marked.
    superseded_by: str | None
    marked: tuple[str, ...]
    unmarked: tuple[UnmarkedSupersession, ...]
    #: Targets of a write whose Mem0 leg is queued for retry, marked when it lands.
    pending_retry: tuple[str, ...]

    def to_ack(self) -> dict[str, Any]:
        return {
            SUPERSESSION_ACK_KEY: {
                'superseded_by': self.superseded_by,
                'marked': list(self.marked),
                'unmarked': [entry.to_ack() for entry in self.unmarked],
                'pending_retry': list(self.pending_retry),
            }
        }


async def mark_superseded(
    memory_service: MemoryUpdater,
    *,
    metadata: Any,
    written: AddMemoryResponse,
    project_id: str,
    agent_id: str | None,
    session_id: str | None,
    causation_id: str | None,
    source: str,
) -> SupersessionMarks | None:
    """Stamp each record *metadata* ``supersedes`` with the id *written* landed.

    ``None`` when the write declares no supersession. Otherwise every declared
    member is marked, reported unmarked with its reason, or (for a write
    pending retry) reported ``pending_retry``.
    """
    declared = (
        normalize_supersedes(metadata.get('supersedes'))
        if isinstance(metadata, Mapping)
        else []
    )
    if not declared:
        return None
    if written.mem0_write_state is Mem0WriteState.PENDING_RETRY:
        invalid, targets = _split_declared(declared)
        logger.info(
            'supersession_mark: the write is pending Mem0 retry; %d declared record(s) '
            'are marked when it lands: %s',
            len(targets),
            targets,
        )
        return SupersessionMarks(None, (), tuple(invalid), tuple(targets))
    return await _mark_declared(
        memory_service,
        declared,
        superseding_ids=written.memory_ids,
        project_id=project_id,
        agent_id=agent_id,
        session_id=session_id,
        causation_id=causation_id,
        source=source,
    )


async def mark_superseded_after_retry(
    memory_service: MemoryUpdater,
    intent: Mapping[str, Any],
    outcome: Mem0AddOutcome,
) -> SupersessionMarks | None:
    """Mark the records a deferred write's journaled intent supersedes, once it has landed.

    Attributed to the intent's project, agent, session and causation, under
    :data:`SUPERSESSION_RETRY_SOURCE`. ``None`` when the intent declares no
    supersession; failures are reported in the result, as
    :func:`mark_superseded` reports them. An intent naming no project marks
    nothing: its targets' project is unknown.
    """
    declared = normalize_supersedes(decode_mem0_intent_metadata(intent).get('supersedes'))
    if not declared:
        return None
    return await _mark_declared(
        memory_service,
        declared,
        superseding_ids=outcome.memory_ids,
        project_id=intent.get('project_id') or None,
        agent_id=intent.get('agent_id'),
        session_id=intent.get('session_id'),
        causation_id=intent.get('causation_id'),
        source=SUPERSESSION_RETRY_SOURCE,
    )


def _split_declared(declared: list[Any]) -> tuple[list[UnmarkedSupersession], list[str]]:
    """The declared members that are not full UUIDs, and the rest deduplicated in order."""
    invalid = [
        UnmarkedSupersession(member, UnmarkedReason.INVALID_MEMBER, 'not a full UUID')
        for member in declared
        if not is_full_uuid(member)
    ]
    targets = list(dict.fromkeys(member for member in declared if is_full_uuid(member)))
    return invalid, targets


async def _mark_declared(
    memory_service: MemoryUpdater,
    declared: list[Any],
    *,
    superseding_ids: Sequence[str],
    project_id: str | None,
    agent_id: str | None,
    session_id: str | None,
    causation_id: str | None,
    source: str,
) -> SupersessionMarks:
    unmarked, targets = _split_declared(declared)
    superseding_id = superseding_ids[0] if len(superseding_ids) == 1 else None
    marked: list[str] = []
    if superseding_id is None:
        no_record = (
            f'the write landed {len(superseding_ids)} Mem0 ids; a mark needs exactly one'
        )
        unmarked.extend(
            UnmarkedSupersession(target, UnmarkedReason.NO_SUPERSEDING_RECORD, no_record)
            for target in targets
        )
    elif project_id is None:
        unmarked.extend(
            UnmarkedSupersession(
                target, UnmarkedReason.NO_PROJECT_ID, 'the write names no project to mark in'
            )
            for target in targets
        )
    else:
        for target in targets:
            failure = await patch_memory_metadata(
                memory_service,
                memory_id=target,
                project_id=project_id,
                patch={SUPERSEDED_BY_METADATA_KEY: superseding_id},
                agent_id=agent_id,
                session_id=session_id,
                causation_id=causation_id,
                source=source,
            )
            if failure is None:
                marked.append(target)
            else:
                unmarked.append(UnmarkedSupersession(
                    target,
                    UnmarkedReason.PATCH_FAILED,
                    f"{failure['error_type']}: {failure['error']}",
                ))
    marks = SupersessionMarks(superseding_id, tuple(marked), tuple(unmarked), ())
    if marks.unmarked:
        logger.warning(
            'supersession_mark: superseding=%r left %d declared record(s) unmarked: %s',
            superseding_id,
            len(marks.unmarked),
            [(entry.memory_id, entry.reason.value) for entry in marks.unmarked],
        )
    return marks
