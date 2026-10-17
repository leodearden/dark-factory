"""The preconditions of a top-level task-metadata lookup, in one place.

Both the MCP tool and ``SqliteTaskBackend.find_tasks_by_metadata`` call
:func:`validate_metadata_lookup`, so an in-process caller meets the same
rules as a remote one. The match semantics these rules protect are
``plans/task-metadata-lookup-prd.md`` decision 5.
"""

from __future__ import annotations

import re
from typing import TypeGuard

METADATA_KEY_PATTERN = re.compile(r'[A-Za-z0-9_][A-Za-z0-9_-]*')
DEFAULT_LOOKUP_LIMIT = 50

_SQLITE_INTEGER_RANGE = range(-(2**63), 2**63)


class InvalidMetadataLookupError(ValueError):
    """A lookup argument breaks the lookup's preconditions; the store was not read."""


def _offending(name: str, value: object, rule: str) -> InvalidMetadataLookupError:
    return InvalidMetadataLookupError(
        f'{name} must be {rule}; got {value!r} ({type(value).__name__})',
    )


def _is_strict_int(value: object) -> TypeGuard[int]:
    return isinstance(value, int) and not isinstance(value, bool)


def validate_metadata_lookup(
    *, key: object, value: object, include_value: object, limit: object, offset: object,
) -> None:
    if not isinstance(key, str) or METADATA_KEY_PATTERN.fullmatch(key) is None:
        raise _offending('key', key, f'a str matching {METADATA_KEY_PATTERN.pattern}')
    if not (value is None or isinstance(value, str) or _is_strict_int(value)):
        raise _offending('value', value, 'None, a str or an int (not a bool)')
    if _is_strict_int(value) and value not in _SQLITE_INTEGER_RANGE:
        raise _offending('value', value, 'an int within the SQLite INTEGER range')
    if not isinstance(include_value, bool):
        raise _offending('include_value', include_value, 'a bool')
    if not _is_strict_int(limit) or limit < 1:
        raise _offending('limit', limit, 'an int >= 1')
    if not _is_strict_int(offset) or offset < 0:
        raise _offending('offset', offset, 'an int >= 0')
