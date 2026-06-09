"""Regression guard: operator-only escalation tools are out of every agent role.

The halt-direction merge-queue / scheduler levers (and their un-pause siblings)
are OPERATOR-ONLY.  Autonomous agents are spawned with ``--allowed-tools`` set to
their role's explicit allow-list (shared/cli_invoke._invoke_claude), so a tool
that appears in NO role's ``allowed_tools`` is unreachable by them — the CLI is
the hard boundary.  This codifies that invariant: if someone later adds one of
these tool names to a role, this test fails loudly rather than silently handing
autonomous agents a factory-wide stop button.

No server-side role gate is used (that would add an agent_role param + human
caller friction, asymmetric with the un-pause siblings); restriction is purely
by allow-list omission.  See escalation/server.py halt_merge_queue / halt_scheduler.
"""

from __future__ import annotations

from orchestrator.agents.roles import ROLES

# The full operator-only escalation control set: both halt directions and both
# un-pause siblings.  Reversing a halt is just as much an operator action as
# applying one, so the un-pause tools are restricted identically.
OPERATOR_ONLY_ESCALATION_TOOLS = frozenset({
    'mcp__escalation__halt_merge_queue',
    'mcp__escalation__halt_scheduler',
    'mcp__escalation__unhalt_merge_queue',
    'mcp__escalation__resume_scheduler',
})


def test_operator_only_tools_absent_from_every_role_allow_list() -> None:
    """No agent role may list any operator-only escalation tool in allowed_tools."""
    offenders: dict[str, set[str]] = {}
    for name, role in ROLES.items():
        leaked = OPERATOR_ONLY_ESCALATION_TOOLS.intersection(role.allowed_tools)
        if leaked:
            offenders[name] = leaked

    assert offenders == {}, (
        'Operator-only escalation tools leaked into agent allow-lists '
        f'(autonomous agents could call them): {offenders}'
    )
