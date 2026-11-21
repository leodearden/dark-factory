"""The preconditions of a top-level task-metadata lookup, in one place.

``SqliteTaskBackend.find_tasks_by_metadata`` calls
:func:`validate_metadata_lookup`, so an in-process caller meets every rule.
The MCP tool checks paging and the status list's shape with the helpers it
shares with ``get_tasks``, then calls :func:`validate_metadata_query` for the
rest, so each rule is checked once per layer. The match semantics these rules
protect are ``plans/task-metadata-lookup-prd.md`` decision 5.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import TypeGuard

from shared.task_statuses import TaskStatus

METADATA_KEY_PATTERN = re.compile(r'[A-Za-z0-9_][A-Za-z0-9_-]*')
DEFAULT_LOOKUP_LIMIT = 50

_SQLITE_INTEGER_RANGE = range(-(2**63), 2**63)
_KNOWN_STATUSES = sorted(status.value for status in TaskStatus)


class InvalidMetadataLookupError(ValueError):
    """A lookup argument breaks the lookup's preconditions; the store was not read."""


def _offending(name: str, value: object, rule: str) -> InvalidMetadataLookupError:
    return InvalidMetadataLookupError(
        f'{name} must be {rule}; got {value!r} ({type(value).__name__})',
    )


def _is_strict_int(value: object) -> TypeGuard[int]:
    return isinstance(value, int) and not isinstance(value, bool)


def validate_metadata_query(
    *, key: object, value: object, statuses: Sequence[str] | None, include_value: object,
) -> None:
    """What is looked up: the key, the value, the status vocabulary and the value opt-in.

    A misspelt status is rejected, because it would otherwise read as a key
    that no task carries.
    """
    if not isinstance(key, str) or METADATA_KEY_PATTERN.fullmatch(key) is None:
        raise _offending('key', key, f'a str matching {METADATA_KEY_PATTERN.pattern}')
    if not (value is None or isinstance(value, str) or _is_strict_int(value)):
        raise _offending('value', value, 'None, a str or an int (not a bool)')
    if _is_strict_int(value) and value not in _SQLITE_INTEGER_RANGE:
        raise _offending('value', value, 'an int within the SQLite INTEGER range')
    unknown = [s for s in statuses or () if s not in _KNOWN_STATUSES]
    if unknown:
        raise _offending('statuses', unknown, f'drawn from {_KNOWN_STATUSES}')
    if not isinstance(include_value, bool):
        raise _offending('include_value', include_value, 'a bool')


def validate_metadata_lookup(
    *,
    key: object,
    value: object,
    statuses: object,
    include_value: object,
    limit: object,
    offset: object,
) -> None:
    """Every precondition of one lookup page, for a caller no tool has checked."""
    if statuses is not None and not (
        isinstance(statuses, list) and all(isinstance(s, str) for s in statuses)
    ):
        raise _offending('statuses', statuses, 'None or a list of status strings')
    validate_metadata_query(key=key, value=value, statuses=statuses, include_value=include_value)
    if not _is_strict_int(limit) or limit < 1:
        raise _offending('limit', limit, 'an int >= 1')
    if not _is_strict_int(offset) or offset < 0:
        raise _offending('offset', offset, 'an int >= 0')
