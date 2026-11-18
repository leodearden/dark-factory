"""Shared reader for the ``task_metadata.schema_warning`` census lines.

The token and the ``source=<reader>`` field are one contract, owned by
``fused_memory.middleware.metadata_dict.raw_metadata_dict``. Tests that assert
a reader discarded unreadable metadata loudly filter ``caplog`` through here.
"""

from __future__ import annotations

SCHEMA_WARNING = 'task_metadata.schema_warning'


def census_lines(caplog, source: str | None = None) -> list[str]:
    """The census lines in *caplog*, optionally only those naming *source*."""
    messages = (r.getMessage() for r in caplog.records)
    return [
        m for m in messages
        if SCHEMA_WARNING in m and (source is None or f'source={source}' in m)
    ]
