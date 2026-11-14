"""The one reader of a raw task dict's dependency ids, train id and deterministic kind.

Task dicts arrive from ``get_tasks``/``get_task`` with drifting wire shapes
(ints where ids are strings, CSV strings where lists are expected, metadata as
a JSON string).  Every reader here tolerates every known shape and never
raises: callers sit inside broad ``except Exception`` sweeps where a raise
would silently skip the whole sweep rather than one malformed row.

Stdlib and the stdlib-only `shared.task_metadata_wire` wire rule, so any package may
import it.
"""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from shared.task_metadata_wire import coerce_task_metadata

__all__ = ['dependency_ids', 'is_deterministic', 'train_id']

_DETERMINISTIC_KIND = 'deterministic'


def dependency_ids(task: Mapping[str, Any]) -> list[str]:
    """Return *task*'s local dependency ids as a fresh list of strings, order kept.

    ``task['dependencies']`` may be:

    - a list, tuple or set whose entries are ``{'id': x}`` dicts, ints or
      strs, mixed freely — falsy entries are dropped, a dict without an
      ``id`` key is stringified whole.  The drop tests the raw entry, before
      a dict is unwrapped and without stripping, so a dict whose ``id`` is
      falsy (``{'id': ''}`` gives ``''``, ``{'id': None}`` gives ``'None'``)
      and a whitespace-only string both survive; or
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
        return [str(deps)]
    return [_dependency_id(entry) for entry in deps if entry]


def _dependency_id(entry: Any) -> str:
    return str(entry.get('id', entry) if isinstance(entry, dict) else entry)


def train_id(task: Mapping[str, Any]) -> str | None:
    """Return ``metadata.train.id`` when it is a non-empty value, else ``None``.

    A falsy id (``0``, ``False``, ``[]``) or blank text is ``None``.
    """
    train = _metadata(task).get('train')
    raw = train.get('id') if isinstance(train, dict) else None
    if not raw:
        return None
    text = str(raw)
    return text if text.strip() else None


def is_deterministic(task: Mapping[str, Any]) -> bool:
    """Return True iff *task*'s metadata has ``task_kind == 'deterministic'``."""
    return _metadata(task).get('task_kind') == _DETERMINISTIC_KIND


def _metadata(task: Mapping[str, Any]) -> dict[str, Any]:
    return coerce_task_metadata(task.get('metadata')) or {}
