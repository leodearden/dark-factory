"""The escalation server's instructions send clients only to fused-memory tools that exist.

``escalation/src/escalation/server_instructions.py::ESCALATION_SERVER_INSTRUCTIONS``
tells every escalation client that task records are read with
``mcp__fused-memory__get_task`` / ``get_tasks``. If fused-memory renamed or
dropped either tool, those instructions would send clients to a tool that does
not exist, which is the failure they were written to fix.

This check lives here, not in ``escalation/tests/test_server_instructions.py``,
for two reasons. Only fused-memory's test environment can build both servers:
escalation is one of its dev dependencies, but fused-memory is not one of
escalation's. And a rename in fused-memory fails fused-memory's own suite.
"""

from __future__ import annotations

import re
from unittest.mock import AsyncMock

import pytest
from escalation.server_instructions import ESCALATION_SERVER_INSTRUCTIONS

from fused_memory.server.tools import create_mcp_server


@pytest.mark.asyncio
async def test_every_fused_memory_tool_the_escalation_instructions_cite_is_registered():
    cited = set(re.findall(r'mcp__fused-memory__([a-z_]+)', ESCALATION_SERVER_INSTRUCTIONS))
    registered = {tool.name for tool in await create_mcp_server(AsyncMock()).list_tools()}

    assert cited, 'the escalation instructions cite no mcp__fused-memory__<tool> by name'
    assert cited <= registered, (
        f'the escalation instructions cite fused-memory tools that are not registered: '
        f'{sorted(cited - registered)}; rewrite escalation/src/escalation/server_instructions.py'
    )
