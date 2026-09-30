"""The escalation server's self-description to its MCP clients.

Every client of ``escalation/src/escalation/server.py::create_server`` receives
:data:`ESCALATION_SERVER_INSTRUCTIONS` at initialize, and Claude Code shows it
in the agent's system prompt even when this server's tools are deferred.

Two rules keep it worth sending. It must stay under the 2048-char client cap,
because text past the cap never reaches the agent. And every tool it names must
be registered on the server its prefix names. Two test files enforce these:
``escalation/tests/test_server_instructions.py`` checks the cap and the
``mcp__escalation__<tool>`` names, and
``fused-memory/tests/test_escalation_instructions_cite_registered_tools.py``
checks the ``mcp__fused-memory__<tool>`` names.
"""

ESCALATION_SERVER_INSTRUCTIONS = """\
This server holds one escalation queue's records on the L0 -> L1 -> L2 ladder.

MCP tool names are per-server: the mcp__<server>__ prefix names the server \
that owns a tool, and a name on one server says nothing about another.

This server has no `get_task` or `get_tasks`: task records live on fused-memory \
— read one with `mcp__fused-memory__get_task`, list them with \
`mcp__fused-memory__get_tasks`.

Escalation reads on this server:
- `mcp__escalation__get_escalation` — one full record, by escalation id.
- `mcp__escalation__get_pending_escalations` — the pending queue.
- `mcp__escalation__get_task_escalations` and \
`mcp__escalation__get_task_escalation_history` — every escalation ever filed \
for one task, archive included.
- `mcp__escalation__get_task_runtime_state` — the orchestrator's live runtime \
snapshot of in-flight tasks (empty when no orchestrator is attached); not a \
task-record read.
"""
