"""Shared reader for the ``task_metadata.schema_warning`` census lines.

The line shape is the one ``fused_memory/middleware/metadata_dict.py::extract_metadata_dict``
logs when it discards unreadable metadata. Tests that assert a reader discarded
unreadable metadata loudly filter ``caplog`` through here.
"""

from __future__ import annotations

import pytest

SCHEMA_WARNING = 'task_metadata.schema_warning'


def census_lines(
    caplog: pytest.LogCaptureFixture, source: str | None = None
) -> list[str]:
    """The census lines in *caplog*, optionally only those whose source starts with *source*.

    *source* matches as a prefix of the ``source=`` field, so a sub-source such
    as ``reader.sub[id]`` is included in ``reader``'s lines.
    """
    messages = (r.getMessage() for r in caplog.records)
    return [
        m for m in messages
        if SCHEMA_WARNING in m and (source is None or f'source={source}' in m)
    ]
