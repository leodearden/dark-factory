"""The fused-memory boundary guard chain (task 4643).

``fused_memory.server.boundary_guards`` builds the ordered members and installs
them as ONE wrapping of ``ToolManager.call_tool``: markup guard outermost, then
the unknown-argument guard, then the uuid-prefix guard when a prefix resolver
is wired.

AUTHORING RULE: never write a raw MCP envelope sentinel literal into this file.
Specimens are BUILT from ``shared.toolcall_markup``'s own constants by
:func:`_leaked`, as ``tests/test_markup_guard_fused_memory.py`` does.
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import AsyncMock

import pytest
from fastmcp.exceptions import ToolError
from shared.mcp_markup_middleware import MarkupGuardMiddleware, RepairPolicy
from shared.toolcall_markup import CANONICAL_OPENER_PREFIX, closer_for
from shared.whole_tree_guard_family import WHOLE_TREE_SCAN_TEST_TIMEOUT

from fused_memory.server import markup_guard
from fused_memory.server.boundary_guards import boundary_guard_chain, install_boundary_guards
from fused_memory.server.tools import create_mcp_server
from fused_memory.server.unknown_argument_guard import UnknownArgumentMiddleware

pytestmark = pytest.mark.timeout(WHOLE_TREE_SCAN_TEST_TIMEOUT)

_PROJECT_ID = 'dark_factory'
_PROJECT_ROOT = '/project'
_KNOWN_PROJECTS = {_PROJECT_ID: _PROJECT_ROOT}
_AGENT_ID = 'claude-task-4643'


def _leaked(clean: str, param: str, value: str) -> str:
    """*clean* text that mis-closed and absorbed *param*, built without a raw sentinel."""
    return (
        clean
        + closer_for('parameter')
        + CANONICAL_OPENER_PREFIX
        + f'"{param}">'
        + value
        + closer_for('parameter')
    )


def _payload(exc_info: pytest.ExceptionInfo[ToolError]) -> dict[str, Any]:
    return json.loads(str(exc_info.value))


class TestTheChainWithoutAPrefixResolver:
    def test_the_members_are_markup_then_unknown_argument(self):
        chain = boundary_guard_chain(known_projects=_KNOWN_PROJECTS, prefix_resolver=None)

        assert [type(member) for member in chain] == [
            MarkupGuardMiddleware,
            UnknownArgumentMiddleware,
        ]
        markup_member = chain[0]
        assert isinstance(markup_member, MarkupGuardMiddleware)
        assert markup_member.policy is RepairPolicy.REJECT_WITH_REPAIR
        assert markup_member.exempt_tools == markup_guard.EXEMPT_TOOLS

    def test_installing_wraps_once_and_warns_once_that_the_uuid_member_is_absent(self, caplog):
        server = create_mcp_server(AsyncMock())

        with caplog.at_level('WARNING', logger='fused_memory.server.boundary_guards'):
            install_boundary_guards(server, known_projects=_KNOWN_PROJECTS, prefix_resolver=None)
            wrapped = server._tool_manager.call_tool
            install_boundary_guards(server, known_projects=_KNOWN_PROJECTS, prefix_resolver=None)

        assert server._tool_manager.call_tool is wrapped
        warnings = [
            record for record in caplog.records
            if record.name == 'fused_memory.server.boundary_guards'
            and record.levelname == 'WARNING'
        ]
        assert len(warnings) == 1
        message = warnings[0].getMessage()
        assert 'uuid-prefix' in message
        assert 'resolver' in message

    @pytest.mark.asyncio
    async def test_markup_sits_outside_the_unknown_argument_guard(self):
        mock_service = AsyncMock()
        server = create_mcp_server(mock_service)
        install_boundary_guards(server, known_projects=_KNOWN_PROJECTS, prefix_resolver=None)

        with pytest.raises(ToolError) as exc_info:
            await server._tool_manager.call_tool(
                'add_memory',
                {
                    'content': _leaked('a clean note', 'agent_id', _AGENT_ID),
                    'project_id': _PROJECT_ID,
                    'categroy': 'observations_and_summaries',
                },
            )

        assert _payload(exc_info)['error_type'] == 'mcp_markup_detected'
        mock_service.add_memory.assert_not_awaited()
