"""Shared helpers for tests that check a role's MCP tool grants."""

from __future__ import annotations


def tool_is_granted(tool: str, granted: set[str]) -> bool:
    """True if `tool` is granted directly, or covered by a `mcp__<family>__*` wildcard."""
    if tool in granted:
        return True
    family_prefix, sep, _name = tool.rpartition('__')
    return sep == '__' and f'{family_prefix}__*' in granted
