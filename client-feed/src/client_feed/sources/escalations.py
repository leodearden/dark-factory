"""The open escalations: the pending escalation/src/escalation/models.py::Escalation records at the queue's root.

escalation/src/escalation/queue.py::EscalationQueue.resolve archives a closed record out of the root; only id, task_id, status, level and members are kept.
"""

from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from fnmatch import fnmatchcase
from pathlib import Path
from types import MappingProxyType
from typing import Final

from pydantic import BaseModel, ConfigDict, ValidationError
from shared.project_layout import ESCALATION_QUEUE_RELPATH

from client_feed.contract import NonNegativeInt, Source
from client_feed.sources.store import StoreProblem, StoreReadError
from client_feed.sources.task_id import task_id_of

QUEUE_RELPATH: Final = Path(ESCALATION_QUEUE_RELPATH)
_RECORD_NAME: Final = 'esc-*.json'


class _Status(StrEnum):
    """escalation/src/escalation/models.py::Escalation.status; any other value is a malformed record, never a closed one."""

    PENDING = 'pending'
    RESOLVED = 'resolved'
    DISMISSED = 'dismissed'


class EscalationSkipReason(StrEnum):
    MALFORMED_RECORD = 'malformed_record'
    UNREADABLE_RECORD = 'unreadable_record'


@dataclass(frozen=True)
class SkippedRecord:
    """A root record that could not be read, by file name, so it neither vanishes nor hides the rest."""

    name: str
    reason: EscalationSkipReason


@dataclass(frozen=True)
class OpenEscalation:
    escalation_id: str
    level: int


@dataclass(frozen=True)
class OpenEscalations:
    """Every pending escalation by task, each listed under its own task and under each pending member's."""

    by_task: Mapping[int, tuple[OpenEscalation, ...]]
    skipped: tuple[SkippedRecord, ...] = ()

    def levels(self, task_id: int) -> tuple[int, ...]:
        """The current level of each of the task's open escalations."""
        return tuple(escalation.level for escalation in self.by_task.get(task_id, ()))


class _Record(BaseModel):
    """The kept fields, each absent one defaulted as the producer's dataclass defaults it."""

    model_config = ConfigDict(frozen=True, extra='ignore')

    id: str
    task_id: str | None
    status: _Status = _Status.PENDING
    level: NonNegativeInt = 0
    members: tuple[str, ...] = ()


def _record_paths(root: Path) -> list[Path]:
    """The root's record files in name order; a queue that cannot be listed raises, never reads as empty."""
    queue = root / QUEUE_RELPATH
    try:
        names = os.listdir(queue)
    except FileNotFoundError as exc:
        raise StoreReadError(
            source=Source.EVENTS, relpath=QUEUE_RELPATH, problem=StoreProblem.MISSING,
            detail='no such directory',
        ) from exc
    except OSError as exc:
        raise StoreReadError(
            source=Source.EVENTS, relpath=QUEUE_RELPATH, problem=StoreProblem.UNREADABLE,
            detail=str(exc),
        ) from exc
    records = sorted(name for name in names if fnmatchcase(name, _RECORD_NAME))
    return [queue / name for name in records]


def _read_record(path: Path) -> _Record | SkippedRecord | None:
    """The record, a skip naming why not, or None when it was archived after listing."""
    try:
        return _Record.model_validate_json(path.read_bytes())
    except FileNotFoundError:
        return None
    except OSError:
        return SkippedRecord(path.name, EscalationSkipReason.UNREADABLE_RECORD)
    except ValidationError:
        return SkippedRecord(path.name, EscalationSkipReason.MALFORMED_RECORD)


def _by_task(pending: Sequence[_Record]) -> dict[int, tuple[OpenEscalation, ...]]:
    """Each record under its own task and under the task of each member that is itself pending."""
    owner = {record.id: task_id_of(record.task_id) for record in pending}
    by_task: dict[int, list[OpenEscalation]] = {}
    for record in pending:
        member_tasks = (owner.get(member) for member in record.members)
        for task_id in {owner[record.id], *member_tasks}:
            if task_id is not None:
                by_task.setdefault(task_id, []).append(OpenEscalation(record.id, record.level))
    return {
        task_id: tuple(sorted(escalations, key=lambda escalation: escalation.escalation_id))
        for task_id, escalations in by_task.items()
    }


def read_open_escalations(root: Path) -> OpenEscalations:
    """The queue root's pending records, by task; reads nothing else, writes nothing and takes no lock."""
    read = [_read_record(path) for path in _record_paths(root)]
    pending = tuple(item for item in read if isinstance(item, _Record) and item.status is _Status.PENDING)
    return OpenEscalations(
        by_task=MappingProxyType(_by_task(pending)),
        skipped=tuple(item for item in read if isinstance(item, SkippedRecord)),
    )
