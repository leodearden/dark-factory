"""The raw dict a task-metadata wire value denotes.

The middleware guards and readers each receive ``metadata`` as a dict, a JSON
string, ``None`` or, from a stored row, anything at all.
:func:`raw_metadata_dict` resolves it to a dict ONCE, under the one wire rule
:func:`shared.task_metadata_wire.coerce_task_metadata`, so a reader gets every
key it needs from one value and an unreadable value is never discarded
silently.
"""

from __future__ import annotations

import logging
from typing import Any

from shared.task_metadata import parse_metadata
from shared.task_metadata_wire import coerce_task_metadata

logger = logging.getLogger(__name__)

__all__ = ['raw_metadata_dict']


def raw_metadata_dict(metadata: Any, *, source: str) -> dict[str, Any]:
    """Return *metadata* as its raw dict, under ``coerce_task_metadata``'s rule.

    Absent metadata gives ``{}``. Readable metadata gives its raw dict (the
    same object for a dict input), never a ``TaskMetadata.model_dump()``, so
    keys the schema does not know survive unchanged. Unreadable metadata
    gives ``{}`` and logs one ``task_metadata.schema_warning`` line naming
    *source*, the value's type and ``parse_metadata``'s diagnosis.
    """
    resolved = coerce_task_metadata(metadata)
    if resolved is not None:
        return resolved
    _, warnings = parse_metadata(metadata, direction='read')
    logger.warning(
        'task_metadata.schema_warning source=%s error=%s (type=%s); metadata discarded',
        source,
        '; '.join(w.message for w in warnings) or 'unrecognised shape',
        type(metadata).__name__,
    )
    return {}
