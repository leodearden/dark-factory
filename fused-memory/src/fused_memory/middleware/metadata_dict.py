"""The raw dict a submit-boundary ``metadata`` argument denotes.

The middleware guards each receive ``metadata`` as a dict, a JSON string or
``None``. :func:`raw_metadata_dict` resolves it to a dict ONCE, under the
malformed-input policy of :func:`shared.task_metadata.parse_metadata`
(``direction='read'``), so a guard reads every key it needs from one value.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from shared.task_metadata import parse_metadata

logger = logging.getLogger(__name__)

__all__ = ['raw_metadata_dict']


def raw_metadata_dict(metadata: str | dict[str, Any] | None, *, source: str) -> dict[str, Any]:
    """Return *metadata* as its raw dict.

    ``None`` and ``''`` are benign-absent and give ``{}``. A string that is
    not a JSON object also gives ``{}``, and logs one
    ``task_metadata.schema_warning`` line naming *source*. The result is the
    raw input, never a ``TaskMetadata.model_dump()``, so keys the schema does
    not know survive unchanged.
    """
    if isinstance(metadata, dict):
        return metadata
    if not metadata:
        return {}
    try:
        decoded = json.loads(metadata)
    except ValueError:
        decoded = None
    if isinstance(decoded, dict):
        return decoded
    _, warnings = parse_metadata(metadata, direction='read')
    logger.warning(
        'task_metadata.schema_warning source=%s error=%s (type=str); metadata discarded',
        source,
        '; '.join(w.message for w in warnings) or 'unrecognised shape',
    )
    return {}
