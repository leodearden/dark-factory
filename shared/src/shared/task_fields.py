"""The one reader of a raw task dict's dependency ids, train id and deterministic kind.

Task dicts arrive from ``get_tasks``/``get_task`` with drifting wire shapes
(ints where ids are strings, CSV strings where lists are expected, metadata as
a JSON string).  Every reader here tolerates every known shape and never
raises: callers sit inside broad ``except Exception`` sweeps where a raise
would silently skip the whole sweep rather than one malformed row.

Stdlib only, so any package may import it.
"""
from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

__all__ = ['dependency_ids', 'is_deterministic', 'train_id']

_DETERMINISTIC_KIND = 'deterministic'


def dependency_ids(task: Mapping[str, Any]) -> list[str]:
    """Return *task*'s local dependency ids as a fresh list of strings, order kept.

    ``task['dependencies']`` may be:

    - a list, tuple or set whose entries are ``{'id': x}`` dicts, ints or
      strs, mixed freely — falsy entries are dropped, a dict without an
      ``id`` key is stringified whole; or
    - a CSV string — split on commas, pieces stripped, empty pieces dropped; or
    - any other non-empty shape (a bare int, a dict) — stringified WHOLE into
      one element.  Iterating a dict would silently yield its keys; an id that
      matches nothing fails closed.

    A missing or falsy value returns ``[]``.
    """
    deps = task.get('dependencies') or []
    if isinstance(deps, str):
        return [piece.strip() for piece in deps.split(',') if piece.strip()]
    if not isinstance(deps, (list, tuple, set)):
        return [str(deps)] if deps else []
    return [_dependency_id(entry) for entry in deps if entry]


def _dependency_id(entry: Any) -> str:
    return str(entry.get('id', entry) if isinstance(entry, dict) else entry)


def train_id(task: Mapping[str, Any]) -> str | None:
    """Return ``metadata.train.id`` when it is a non-blank value, else ``None``."""
    metadata = task.get('metadata')
    train = metadata.get('train') if isinstance(metadata, dict) else None
    raw = train.get('id') if isinstance(train, dict) else None
    if raw is None:
        return None
    text = str(raw)
    return text if text.strip() else None


def is_deterministic(task: Mapping[str, Any]) -> bool:
    """Return True iff *task*'s metadata (a dict or a JSON string) has ``task_kind == 'deterministic'``."""
    metadata = task.get('metadata') or {}
    if isinstance(metadata, str):
        try:
            metadata = json.loads(metadata)
        except ValueError:
            return False
    if not isinstance(metadata, dict):
        return False
    return metadata.get('task_kind') == _DETERMINISTIC_KIND
