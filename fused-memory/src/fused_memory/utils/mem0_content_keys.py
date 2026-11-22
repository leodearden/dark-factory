"""Which raw Qdrant payload keys hold a Mem0 memory's text, and how to read it.

mem0 writes a memory's text under ``data``; older or alternately written
points carry it under ``memory`` or ``content``. The first non-empty string,
in that order, is the memory's text.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

MEM0_CONTENT_KEYS: tuple[str, ...] = ('data', 'memory', 'content')

#: The key mem0 itself writes a memory's text under.
MEM0_TEXT_KEY = MEM0_CONTENT_KEYS[0]


def mem0_payload_text(payload: Mapping[str, Any]) -> str:
    """The text of a raw Qdrant *payload*, or ``''`` when it carries none."""
    for key in MEM0_CONTENT_KEYS:
        value = payload.get(key)
        if isinstance(value, str) and value:
            return value
    return ''
