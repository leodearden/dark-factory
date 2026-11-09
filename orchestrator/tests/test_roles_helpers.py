"""Contract tests for `_roles_helpers.py::tool_is_granted`."""

from __future__ import annotations

import pytest
from _roles_helpers import tool_is_granted


@pytest.mark.parametrize(
    ('tool', 'granted', 'expected'),
    [
        pytest.param(
            'mcp__escalation__escalate_info',
            {'mcp__escalation__escalate_info'},
            True,
            id='direct-grant',
        ),
        pytest.param(
            'mcp__fused-memory__search',
            {'mcp__fused-memory__*'},
            True,
            id='family-wildcard-grant',
        ),
        pytest.param(
            'mcp__fused-memory__search',
            {'mcp__escalation__*'},
            False,
            id='other-family-wildcard',
        ),
        pytest.param(
            'mcp__fused-memory__search',
            {'mcp__fused-memory__add_memory'},
            False,
            id='same-family-sibling-direct-grant',
        ),
        pytest.param(
            'Read',
            {'__*'},
            False,
            id='name-without-double-underscore',
        ),
    ],
)
def test_tool_is_granted(tool: str, granted: set[str], expected: bool) -> None:
    assert tool_is_granted(tool, granted) is expected
