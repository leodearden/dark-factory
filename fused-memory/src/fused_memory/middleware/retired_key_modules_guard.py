"""Reject the retired ``metadata.modules`` key on NEW task submissions.

Wired into ``server/tools.py::submit_task`` after ``inject_task_kind``, which
normalises metadata to a dict; one placement covers both creation paths.

``update_task`` and ``commit_planning`` deliberately do not call it: existing
carriers are re-written whole by amendments, and that must keep working
(plans/metadata-modules-retirement-prd.md decision 3).

There is no bypass flag (PRD open question 2): only the top-level metadata
key trips the guard, so quoting ``metadata.modules`` in prose or nesting it
under an ``x_`` key never does.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

__all__ = ['retired_key_modules_error']

_RETIRED_KEY = 'modules'
_REPLACEMENT_KEY = 'files'


def retired_key_modules_error(metadata: Mapping[str, Any]) -> dict[str, Any] | None:
    """Return a ``RetiredMetadataKey`` error dict iff *metadata* carries ``modules``.

    PRESENCE is the violation, whatever the value: ``modules: []`` would
    still mint a new carrier.
    """
    if _RETIRED_KEY not in metadata:
        return None
    return {
        'error': (
            'metadata.modules is retired and is rejected on new submissions. '
            'Declare scope in metadata.files instead.'
        ),
        'error_type': 'RetiredMetadataKey',
        'retired_key': _RETIRED_KEY,
        'replacement_key': _REPLACEMENT_KEY,
        'hint': (
            'Remove metadata.modules. Put the specific FILE paths you expect '
            'to touch in metadata.files: modules entries are usually '
            'directories, and a directory in metadata.files is rejected by '
            'the lock-charter guard (LockCharterViolation). Under-declaring '
            'is fine; files=[] defers scope to the architect. Lock derivation '
            'reads metadata.files only. Mentioning modules in task prose is '
            'fine; only the metadata key is rejected.'
        ),
    }
