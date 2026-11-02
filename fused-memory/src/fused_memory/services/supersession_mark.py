"""Marks each record a write supersedes.

Runs at ``server/tools.py::add_memory`` after the write lands, is metadata-only,
and never raises: by then the correction exists, so failing the call would
invent a write failure. It lives outside ``server/write_triage.py`` because
triage never edits a record (that module's contract C1).
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING, Any

from fused_memory.memory_metadata import SUPERSEDED_BY_METADATA_KEY, normalize_supersedes
from fused_memory.services.consolidation_ops import patch_memory_metadata
from fused_memory.utils.validation import is_full_uuid

if TYPE_CHECKING:
    from fused_memory.models.memory import AddMemoryResponse
    from fused_memory.services.memory_service import MemoryService

__all__ = [
    'SUPERSESSION_ACK_KEY',
    'SupersessionMarks',
    'UnmarkedReason',
    'UnmarkedSupersession',
    'mark_superseded',
]

logger = logging.getLogger(__name__)

SUPERSESSION_ACK_KEY = 'supersession'


class UnmarkedReason(StrEnum):
    INVALID_MEMBER = 'invalid_supersedes_member'
    NO_SUPERSEDING_RECORD = 'no_superseding_mem0_record'
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

    def to_ack(self) -> dict[str, Any]:
        return {
            SUPERSESSION_ACK_KEY: {
                'superseded_by': self.superseded_by,
                'marked': list(self.marked),
                'unmarked': [entry.to_ack() for entry in self.unmarked],
            }
        }


async def mark_superseded(
    memory_service: MemoryService,
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
    member is either marked or reported unmarked with its reason.
    """
    declared = (
        normalize_supersedes(metadata.get('supersedes'))
        if isinstance(metadata, Mapping)
        else []
    )
    if not declared:
        return None
    unmarked = [
        UnmarkedSupersession(member, UnmarkedReason.INVALID_MEMBER, 'not a full UUID')
        for member in declared
        if not is_full_uuid(member)
    ]
    targets: list[str] = list(dict.fromkeys(member for member in declared if is_full_uuid(member)))
    superseding_id = _sole_mem0_id(written)
    marked: list[str] = []
    if superseding_id is None:
        no_record = (
            f'the write landed {len(written.memory_ids)} Mem0 ids; a mark needs exactly one'
        )
        unmarked.extend(
            UnmarkedSupersession(target, UnmarkedReason.NO_SUPERSEDING_RECORD, no_record)
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
    marks = SupersessionMarks(superseding_id, tuple(marked), tuple(unmarked))
    if marks.unmarked:
        logger.warning(
            'supersession_mark: superseding=%r left %d declared record(s) unmarked: %s',
            superseding_id,
            len(marks.unmarked),
            [(entry.memory_id, entry.reason.value) for entry in marks.unmarked],
        )
    return marks


def _sole_mem0_id(written: AddMemoryResponse) -> str | None:
    return written.memory_ids[0] if len(written.memory_ids) == 1 else None
