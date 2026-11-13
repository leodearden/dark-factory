"""A Claude-format mcpServers config holding one stdio plan-tools server, shared by the third-party backend tests.

Each call returns a fresh value, so no test can mutate another's copy.
"""

from __future__ import annotations

from typing import Any


def plan_tools_mcp() -> dict[str, Any]:
    return {'mcpServers': {'plan-tools': {
        'command': '/venv/bin/python', 'args': ['-m', 'plan_tools'],
    }}}
