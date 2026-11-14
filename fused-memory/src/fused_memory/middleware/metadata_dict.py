"""The raw dict a task-metadata wire value denotes.

The middleware guards and readers each receive ``metadata`` as a dict, a JSON
string, ``None`` or, from a stored row, anything at all.
:func:`raw_metadata_dict` resolves it to a dict ONCE, under the one wire rule
:func:`shared.task_metadata_wire.coerce_task_metadata`, so a reader gets every
key it needs from one value and an unreadable value is never discarded
silently.

This module owns the three resolution policies over that rule: raw
(:func:`raw_metadata_dict`), None-preserving (:func:`extract_metadata_dict`)
and copying (:func:`fresh_metadata_copy`).
"""

from __future__ import annotations

import logging
from typing import Any

from shared.task_metadata import parse_metadata
from shared.task_metadata_wire import coerce_task_metadata

logger = logging.getLogger(__name__)

__all__ = ['extract_metadata_dict', 'fresh_metadata_copy', 'raw_metadata_dict']


def extract_metadata_dict(metadata: object, *, source: str) -> dict[str, Any] | None:
    """Return *metadata* as its raw dict, or None when it is unreadable.

    Absent metadata gives ``{}`` and readable metadata its raw dict (the same
    object for a dict input). Unreadable metadata gives None and logs one
    ``task_metadata.schema_warning`` line naming *source*, the value's type
    and ``parse_metadata``'s diagnosis.
    """
    resolved = coerce_task_metadata(metadata)
    if resolved is None:
        _warn_discard(metadata, source=source)
    return resolved


def raw_metadata_dict(metadata: object, *, source: str) -> dict[str, Any]:
    """Return *metadata* as its raw dict, under ``coerce_task_metadata``'s rule.

    Absent metadata gives ``{}``. Readable metadata gives its raw dict (the
    same object for a dict input), never a ``TaskMetadata.model_dump()``, so
    keys the schema does not know survive unchanged. Unreadable metadata
    gives ``{}`` and logs the :func:`extract_metadata_dict` census line.
    """
    resolved = extract_metadata_dict(metadata, source=source)
    return {} if resolved is None else resolved


def fresh_metadata_copy(metadata: object, *, site: str) -> dict[str, Any]:
    """Return a new dict resolved from *metadata*, safe for the caller to mutate.

    Absent metadata gives a fresh ``{}`` and a readable dict is shallow-copied.
    Unreadable metadata is discarded for ``{}``: on top of the census line
    naming *site*, a second WARNING prefixed with *site* carries the original
    value, so the discard can be grepped by caller.
    """
    resolved = extract_metadata_dict(metadata, source=site)
    if resolved is not None:
        return dict(resolved)
    logger.warning(
        '%s: non-dict metadata discarded (type=%s); using fresh dict. Original value: %r',
        site,
        type(metadata).__name__,
        metadata,
    )
    return {}


def _warn_discard(metadata: object, *, source: str) -> None:
    _, warnings = parse_metadata(metadata, direction='read')
    logger.warning(
        'task_metadata.schema_warning source=%s error=%s (type=%s); metadata discarded',
        source,
        '; '.join(w.message for w in warnings) or 'unrecognised shape',
        type(metadata).__name__,
    )
