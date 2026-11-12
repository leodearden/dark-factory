"""How a Stage 1 payload line shows a child memory's link to its parent, read by prompts/stage1.py::CHILD_MEMORY_AUTHORITY_RULE."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from fused_memory.server.grouped_read import PARENT_ID_KEY, child_parent_id

if TYPE_CHECKING:
    from collections.abc import Mapping


def render_parent_link(meta: Mapping[str, Any]) -> str:
    parent_id = child_parent_id(meta)
    if parent_id is None:
        return ''
    return f' {PARENT_ID_KEY}={parent_id} kind={meta["kind"]}'
