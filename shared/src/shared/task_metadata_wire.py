"""The one rule for reading a task's ``metadata`` wire value, in every package.

On the wire a task's ``metadata`` is a dict, a JSON-object string, or absent.
:func:`coerce_task_metadata` resolves it to a tri-state:

- ``{}``: absent. It declares nothing.
- a dict: readable.
- ``None``: present but unreadable. Whatever it carried cannot be seen.

Callers choose their own collapse of ``None`` and their own loudness; this
function never logs and never raises.

This is wire-shape resolution only. Schema validation with diagnoses (and its
pydantic dependency) is :func:`shared.task_metadata.parse_metadata`.
"""

from __future__ import annotations

import json
from typing import Any

__all__ = ['coerce_task_metadata']


def coerce_task_metadata(raw: object) -> dict[str, Any] | None:
    """Resolve a raw ``metadata`` value to ``{}`` / a dict / ``None``.

    - ``None`` or exactly ``''``: a fresh ``{}`` (absent).
    - a dict: that same object, not a copy.
    - a str that decodes to a JSON object: the decoded dict.
    - anything else, including a whitespace-only str: ``None`` (unreadable).
    """
    if raw is None or (isinstance(raw, str) and not raw):
        return {}
    if isinstance(raw, dict):
        return raw
    if not isinstance(raw, str):
        return None
    try:
        parsed = json.loads(raw)
    except ValueError:
        return None
    return parsed if isinstance(parsed, dict) else None
