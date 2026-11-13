"""Reject a ``complexity='simple'`` submission that declares no ``metadata.files``.

Why: ``plans/module-tagger-retirement-prd.md`` task δ (decision 2). Only the
author DECLARATION is gated; runtime vetoes such as hard-blocker tokens and
``force_full_path`` are dispatch-time. update_task writes are deliberately not
gated (PRD Open question 1).
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from shared.task_complexity import SIMPLE_COMPLEXITY, is_simple_complexity

from fused_memory.middleware.lock_charter_guard import extract_files

__all__ = ['simple_task_files_error']

_REQUIRED_KEY = 'files'


def simple_task_files_error(metadata: Mapping[str, Any]) -> dict[str, Any] | None:
    """Return a ``SimpleTaskFilesRequired`` error dict iff *metadata* declares
    the simple path but names no file in ``metadata.files``."""
    if not is_simple_complexity(metadata.get('complexity')):
        return None
    if extract_files(dict(metadata)):
        return None
    return {
        'error': (
            "A complexity='simple' task must declare its target files in "
            'metadata.files at submit time: it plans and edits under its '
            'dispatch-time lock, with no architect to derive scope.'
        ),
        'error_type': 'SimpleTaskFilesRequired',
        'complexity': SIMPLE_COMPLEXITY,
        'required_key': _REQUIRED_KEY,
        'hint': (
            'Put the specific FILE paths you will edit in metadata.files (a '
            'directory is rejected by the lock-charter guard, '
            'LockCharterViolation). files=[] does not satisfy this: no '
            "architect runs to defer to. Or omit complexity='simple' to take "
            'the full architect path, which derives scope from the plan.'
        ),
    }
