"""Tests for agent-capability wiring as a role property (W9-η).

Covers the AgentRole.mcp_families / AgentRole.sandboxed capability model:
- The import-time __post_init__ assertion (a role whose allowed_tools
  reference a fused-memory/escalation/plan-tools tool MUST declare the
  matching mcp_families entry, or construction raises ValueError — this is
  the SIMPLE_TASK-fallthrough regression class, reify esc-4943-54).
- The shipped ROLES declarations preserve the EXACT historical wiring of
  the (soon-to-be-deleted) _MCP_CONFIG_ROLES / _PLAN_TOOLS_ROLES tuples and
  the `role.name in ('implementer', 'debugger')` sandbox check in
  workflow.py's _invoke.

A later step in this task adds TestInvokeDerivesGatingFromRole to this same
file, proving _invoke reads these properties off the role object instead of
name-string membership (see test_workflow_status_on_resume.py for the
kwargs-capture pattern that step reuses).
"""

from __future__ import annotations

import pytest


class TestAgentRoleCapabilityFields:
    """AgentRole gains mcp_families/sandboxed fields with safe defaults."""

    def test_defaults_are_empty_family_set_and_unsandboxed(self):
        from orchestrator.agents.roles import AgentRole  # noqa: PLC0415

        role = AgentRole(name='probe', system_prompt='x')

        assert role.mcp_families == frozenset()
        assert role.sandboxed is False


class TestImportTimeCapabilityAssertion:
    """__post_init__ enforces: tool-in-allowed_tools ⟹ matching family declared.

    One-way only — declaring a family with no matching tool is allowed
    (the judge shape: it needs 'orchestrator' wired for jcodemunch even
    though it has no fused-memory/escalation tool in allowed_tools).
    """

    def test_plan_tools_tool_without_plan_tools_family_raises(self):
        from orchestrator.agents.roles import AgentRole  # noqa: PLC0415

        with pytest.raises(ValueError, match='plan_tools'):
            AgentRole(
                name='probe',
                system_prompt='x',
                allowed_tools=['mcp__plan-tools__create_plan'],
            )

    def test_fused_memory_tool_without_orchestrator_family_raises(self):
        from orchestrator.agents.roles import AgentRole  # noqa: PLC0415

        with pytest.raises(ValueError, match='orchestrator'):
            AgentRole(
                name='probe',
                system_prompt='x',
                allowed_tools=['mcp__fused-memory__search'],
            )

    def test_escalation_tool_without_orchestrator_family_raises(self):
        from orchestrator.agents.roles import AgentRole  # noqa: PLC0415

        with pytest.raises(ValueError, match='orchestrator'):
            AgentRole(
                name='probe',
                system_prompt='x',
                allowed_tools=['mcp__escalation__escalate_info'],
            )

    def test_plan_tools_tool_with_matching_family_succeeds(self):
        from orchestrator.agents.roles import AgentRole  # noqa: PLC0415

        role = AgentRole(
            name='probe',
            system_prompt='x',
            allowed_tools=['mcp__plan-tools__create_plan'],
            mcp_families=frozenset({'plan_tools'}),
        )

        assert role.mcp_families == frozenset({'plan_tools'})

    def test_orchestrator_tools_with_matching_family_succeed(self):
        from orchestrator.agents.roles import AgentRole  # noqa: PLC0415

        role = AgentRole(
            name='probe',
            system_prompt='x',
            allowed_tools=[
                'mcp__fused-memory__search', 'mcp__escalation__escalate_info',
            ],
            mcp_families=frozenset({'orchestrator'}),
        )

        assert role.mcp_families == frozenset({'orchestrator'})

    def test_declaring_family_with_no_matching_tool_is_allowed(self):
        """judge shape: mcp_families={'orchestrator'} but no fm/esc tool in
        allowed_tools — the assertion is a lower bound, never the reverse."""
        from orchestrator.agents.roles import AgentRole  # noqa: PLC0415

        role = AgentRole(
            name='judge_like',
            system_prompt='x',
            allowed_tools=['mcp__jcodemunch__*'],
            mcp_families=frozenset({'orchestrator'}),
        )

        assert role.mcp_families == frozenset({'orchestrator'})
        assert not any(
            t.startswith(('mcp__fused-memory__', 'mcp__escalation__'))
            for t in role.allowed_tools
        )


class TestRolesModuleImportsCleanly:
    """Every shipped role in ROLES satisfies the capability invariant at
    import time — importing the module must not raise."""

    def test_roles_import_succeeds_and_is_populated(self):
        from orchestrator.agents.roles import ROLES  # noqa: PLC0415

        assert len(ROLES) == 9
        assert set(ROLES) == {
            'architect', 'implementer', 'debugger', 'merger', 'steward',
            'deep_reviewer', 'reviewer_comprehensive', 'judge', 'simple_task',
        }


class TestShippedRoleCapabilityDeclarations:
    """Preserve the EXACT historical membership of _MCP_CONFIG_ROLES /
    _PLAN_TOOLS_ROLES / ('implementer', 'debugger') as role properties."""

    # role_name -> (mcp_families, sandboxed)
    EXPECTED = {
        'architect': (frozenset({'orchestrator', 'plan_tools'}), False),
        'implementer': (frozenset({'orchestrator', 'plan_tools'}), True),
        'debugger': (frozenset({'orchestrator', 'plan_tools'}), True),
        'merger': (frozenset({'orchestrator'}), False),
        'judge': (frozenset({'orchestrator'}), False),
        'simple_task': (frozenset({'orchestrator', 'plan_tools'}), False),
        'reviewer_comprehensive': (frozenset(), False),
        'steward': (frozenset({'orchestrator'}), False),
        'deep_reviewer': (frozenset({'orchestrator'}), False),
    }

    @pytest.mark.parametrize('role_name', sorted(EXPECTED))
    def test_role_declares_expected_families_and_sandbox(self, role_name):
        from orchestrator.agents.roles import ROLES  # noqa: PLC0415

        role = ROLES[role_name]
        expected_families, expected_sandboxed = self.EXPECTED[role_name]

        assert role.mcp_families == expected_families, (
            f'{role_name}: expected mcp_families={expected_families!r}, '
            f'got {role.mcp_families!r}'
        )
        assert role.sandboxed == expected_sandboxed, (
            f'{role_name}: expected sandboxed={expected_sandboxed!r}, '
            f'got {role.sandboxed!r}'
        )

    def test_simple_task_has_both_families(self):
        """Explicit regression guard for reify esc-4943-54: simple_task's
        prompt embeds escalation/memory instructions AND requires a
        plan-tools-registered plan, so it must declare BOTH families or the
        Lever-C fast-path falls through to the architect silently."""
        from orchestrator.agents.roles import SIMPLE_TASK  # noqa: PLC0415

        assert SIMPLE_TASK.mcp_families == frozenset({'orchestrator', 'plan_tools'})

    def test_only_implementer_and_debugger_are_sandboxed(self):
        from orchestrator.agents.roles import ROLES  # noqa: PLC0415

        sandboxed_names = {name for name, role in ROLES.items() if role.sandboxed}

        assert sandboxed_names == {'implementer', 'debugger'}


class TestInverseCapabilityInvariant:
    """Every ROLES entry that allows a family's tool prefix must declare
    that family — re-expressed against the new model (was
    test_workflow_plan_tools_injection.py::TestMcpRoleGates, which imported
    the now-deleted _PLAN_TOOLS_ROLES / _MCP_CONFIG_ROLES tuples)."""

    def test_every_role_allowing_plan_tools_declares_plan_tools_family(self):
        from orchestrator.agents.roles import ROLES  # noqa: PLC0415

        for name, role in ROLES.items():
            allows_plan_tools = any(
                t.startswith('mcp__plan-tools__') for t in role.allowed_tools
            )
            if allows_plan_tools:
                assert 'plan_tools' in role.mcp_families, (
                    f'{name!r} allows plan-tools tools but does not declare '
                    f"'plan_tools' in mcp_families (reify esc-4943-54 class)"
                )

    def test_every_role_allowing_orchestrator_tools_declares_orchestrator_family(self):
        from orchestrator.agents.roles import ROLES  # noqa: PLC0415

        for name, role in ROLES.items():
            allows_orchestrator = any(
                t.startswith(('mcp__fused-memory__', 'mcp__escalation__'))
                for t in role.allowed_tools
            )
            if allows_orchestrator:
                assert 'orchestrator' in role.mcp_families, (
                    f'{name!r} allows fused-memory/escalation tools but does '
                    f"not declare 'orchestrator' in mcp_families (reify "
                    f'esc-4943-54 class)'
                )
