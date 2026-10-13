"""Grounds Stage 2's self-reported task counters in code-side evidence."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, TypeAlias

from fused_memory.reconciliation.flag_dedup import confirm_task_present

#: ``(project_id, task_id)``. Taskmaster ids are per-project, so the same id
#: under two projects is two tasks.
TaskKey: TypeAlias = tuple[str | None, str]

HINTS_UPDATED_ACTION = 'task_hints_updated'


def _dict_records(records: object) -> list[dict[Any, Any]]:
    if not isinstance(records, list):
        return []
    return [record for record in records if isinstance(record, dict)]


def _record_key(record: dict[Any, Any], default_project_id: str | None) -> TaskKey | None:
    """A record's normalised key; a missing or blank project_id falls back to the default.

    The fallback stops an omitted project_id from masquerading as a second,
    cross-project filing of the same task.
    """
    task_id = record.get('task_id')
    if task_id is None:
        return None
    task_id_str = str(task_id).strip()
    if not task_id_str:
        return None
    project_id = record.get('project_id')
    if project_id is None:
        return (default_project_id, task_id_str)
    return (str(project_id).strip() or default_project_id, task_id_str)


def action_record_keys(
    records: object,
    default_project_id: str | None,
    valid_statuses: frozenset[str],
) -> frozenset[TaskKey]:
    """The deduped keys of the action records whose ``status`` is in *valid_statuses*.

    Status matching ignores case and surrounding whitespace. A malformed list
    or record is skipped, never raised: a bad record must not corrupt an
    otherwise-good stage report.
    """
    keys: set[TaskKey] = set()
    for record in _dict_records(records):
        try:
            status = record.get('status')
            if not isinstance(status, str) or status.strip().lower() not in valid_statuses:
                continue
            key = _record_key(record, default_project_id)
        except Exception:
            continue
        if key is not None:
            keys.add(key)
    return frozenset(keys)


def coerce_count(value: object) -> int:
    """Best-effort int coercion of a self-reported counter; anything unusable is 0.

    Accepts an int, a finite float, or a numeric string; a bool, a non-finite
    number (task 4140) and anything else is 0. Coercing rather than rejecting
    keeps a legitimately larger ``"3"`` from looking like an undercount and
    being repaired downward, which the upward-only repair forbids
    (task 2230 / W5-mu).
    """
    if isinstance(value, bool):
        return 0
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value) if math.isfinite(value) else 0
    if isinstance(value, str):
        stripped = value.strip()
        try:
            return int(stripped)
        except ValueError:
            pass
        try:
            return coerce_count(float(stripped))
        except (ValueError, OverflowError):
            return 0
    return 0


@dataclass(frozen=True)
class HintsClaim:
    """The memory_hints entries Stage 2 says it attached to one task."""

    entities: frozenset[str]
    queries: frozenset[str]


def _hint_entries(value: object) -> frozenset[str]:
    """A hints list as comparable strings; non-str entries compare by canonical JSON."""
    if not isinstance(value, list):
        return frozenset()
    return frozenset(
        entry if isinstance(entry, str) else json.dumps(entry, sort_keys=True)
        for entry in value
    )


def hints_update_claims(
    records: object,
    default_project_id: str | None,
) -> Mapping[TaskKey, HintsClaim]:
    """Stage 2's ``task_hints_updated`` records as one claim per task key.

    Records for the same key union their entries. A record naming no entries
    at all claims nothing. Malformed input is skipped, never raised.
    """
    entities: dict[TaskKey, set[str]] = {}
    queries: dict[TaskKey, set[str]] = {}
    for record in _dict_records(records):
        try:
            if record.get('action') != HINTS_UPDATED_ACTION:
                continue
            key = _record_key(record, default_project_id)
            record_entities = _hint_entries(record.get('entities'))
            record_queries = _hint_entries(record.get('queries'))
        except Exception:
            continue
        if key is None or not (record_entities or record_queries):
            continue
        entities.setdefault(key, set()).update(record_entities)
        queries.setdefault(key, set()).update(record_queries)
    return {
        key: HintsClaim(entities=frozenset(entities[key]), queries=frozenset(queries[key]))
        for key in entities
    }


def _decode_object(value: object) -> dict[Any, Any]:
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            decoded = json.loads(value)
        except ValueError:
            return {}
        return decoded if isinstance(decoded, dict) else {}
    return {}


def hints_claim_confirmed(claim: HintsClaim, get_task_result: object) -> bool:
    """True iff the re-read task's memory_hints contain every claimed entry.

    This is the SUPERSET rule the Stage 2 prompt makes the agent apply before
    it counts a hints update. Fails closed and never raises.
    """
    if not isinstance(get_task_result, dict) or not confirm_task_present(get_task_result):
        return False
    try:
        hints = _decode_object(get_task_result.get('metadata')).get('memory_hints')
        if not isinstance(hints, dict):
            return False
        return (
            claim.entities <= _hint_entries(hints.get('entities'))
            and claim.queries <= _hint_entries(hints.get('queries'))
        )
    except Exception:
        return False
