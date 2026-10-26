"""The task-creation journal row contract, written by the interceptor and read by Stage 2.

TaskInterceptor writes one write_op row per curator or planning-mode creation
OUTCOME, attributed to the submitter's agent_id and to the ``_causation_id``
it put in ``submit_task`` metadata. Stage 2 reads a run's rows by causation to
ground ``tasks_created`` and ``tasks_combined`` in what actually happened.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, TypeAlias

from shared.task_metadata_wire import coerce_task_metadata

if TYPE_CHECKING:
    from fused_memory.services.write_journal import WriteJournal

logger = logging.getLogger(__name__)

TASK_CREATED_OPERATION = 'task_created'
TASK_COMBINED_OPERATION = 'task_combined'

_OPERATION_FOR_STATUS: dict[str, str] = {
    'created': TASK_CREATED_OPERATION,
    'combined': TASK_COMBINED_OPERATION,
}

#: ``(project_id, task_id)``. Taskmaster ids are per-project, so the same id
#: under two projects is two tasks.
TaskKey: TypeAlias = tuple[str | None, str]


def task_key(project_id: object, task_id: object, default_project_id: str | None) -> TaskKey | None:
    """The one normalised key for a task named by a journal row or a Stage 2 record.

    None when *task_id* is missing or blank. Both parts are stripped, and a
    missing or blank *project_id* falls back to *default_project_id*, so an
    omitted project cannot masquerade as a second, cross-project task.
    """
    if task_id is None:
        return None
    task_id_str = str(task_id).strip()
    if not task_id_str:
        return None
    project_key = str(project_id).strip() if project_id is not None else ''
    return (project_key or default_project_id, task_id_str)


def creation_operation_for(status: str | None) -> str | None:
    """The row operation for a curator outcome status; None when no task was minted or folded."""
    if status is None:
        return None
    return _OPERATION_FOR_STATUS.get(status)


def _decode_params(value: object) -> dict[Any, Any] | None:
    """A row's ``params``: stored as JSON text, a dict when already decoded."""
    if isinstance(value, dict):
        return value
    if not isinstance(value, str):
        return None
    try:
        decoded = json.loads(value)
    except ValueError:
        return None
    return decoded if isinstance(decoded, dict) else None


def _non_blank_str(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    return value.strip() or None


@dataclass(frozen=True)
class CreationAttribution:
    """Who submitted a task, and under which run's causation."""

    agent_id: str | None
    causation_id: str | None

    @classmethod
    def from_submission(cls, metadata: object, agent_id: str | None) -> CreationAttribution:
        decoded = coerce_task_metadata(metadata) or {}
        return cls(agent_id=agent_id, causation_id=_non_blank_str(decoded.get('_causation_id')))

    @classmethod
    def from_blob_value(cls, value: object) -> CreationAttribution:
        """Parse the ticket blob's value; tickets that predate the key are unattributed."""
        if not isinstance(value, dict):
            return cls(agent_id=None, causation_id=None)
        return cls(
            agent_id=_non_blank_str(value.get('agent_id')),
            causation_id=_non_blank_str(value.get('causation_id')),
        )

    def to_blob_value(self) -> dict[str, str | None]:
        return {'agent_id': self.agent_id, 'causation_id': self.causation_id}


UNATTRIBUTED = CreationAttribution(agent_id=None, causation_id=None)


@dataclass(frozen=True)
class CreationOutcomes:
    """The task keys one run's submissions created, and the ones they folded into."""

    created: frozenset[TaskKey]
    combined: frozenset[TaskKey]


def _row_key(row: dict[Any, Any], default_project_id: str | None) -> TaskKey | None:
    params = _decode_params(row.get('params'))
    if params is None:
        return None
    return task_key(row.get('project_id'), params.get('task_id'), default_project_id)


async def read_run_creation_outcomes(
    write_journal: WriteJournal | None,
    run_id: str,
    *,
    default_project_id: str | None,
) -> CreationOutcomes | None:
    """The run's journaled creation outcomes; None when the journal cannot be read.

    Rows are keyed by :func:`task_key` under *default_project_id*, the same
    rule a reader of Stage 2's records applies, so one task is one key in both.
    """
    if write_journal is None:
        return None
    try:
        rows = await write_journal.get_ops_by_causation(run_id)
    except Exception:
        logger.warning(
            'task_creation_journal: reading creation rows for run_id=%s failed',
            run_id, exc_info=True,
        )
        return None
    if not isinstance(rows, list):
        return None

    keys: dict[str, set[TaskKey]] = {TASK_CREATED_OPERATION: set(), TASK_COMBINED_OPERATION: set()}
    for row in rows:
        if not isinstance(row, dict):
            continue
        if row.get('layer') != 'write_op' or row.get('causation_id') != run_id:
            continue
        operation_keys = keys.get(str(row.get('operation')))
        if operation_keys is None or not row.get('success'):
            continue
        key = _row_key(row, default_project_id)
        if key is not None:
            operation_keys.add(key)
    return CreationOutcomes(
        created=frozenset(keys[TASK_CREATED_OPERATION]),
        combined=frozenset(keys[TASK_COMBINED_OPERATION]),
    )
