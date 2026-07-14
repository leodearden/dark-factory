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

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest
from _workflow_helpers import FakeBriefing, FakeMcp, FakeScheduler

from orchestrator.agents.invoke import AgentResult
from orchestrator.config import GitConfig, OrchestratorConfig
from orchestrator.git_ops import GitOps, _run
from orchestrator.scheduler import TaskAssignment
from orchestrator.workflow import TaskWorkflow


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


# ---------------------------------------------------------------------------
# TestInvokeDerivesGatingFromRole fixtures — mirrors
# test_workflow_status_on_resume.py's git_repo/config/git_ops/task_assignment
# fixtures verbatim (per-test-file duplication is the established convention;
# see test_steward_outcome.py's _make_workflow docstring).
# ---------------------------------------------------------------------------


@pytest.fixture
def git_repo(tmp_path: Path) -> Path:
    repo = tmp_path / 'repo'
    repo.mkdir()
    asyncio.run(_init_repo(repo))
    return repo


async def _init_repo(repo: Path) -> None:
    await _run(['git', 'init', '-b', 'main'], cwd=repo)
    await _run(['git', 'config', 'user.email', 'test@test.com'], cwd=repo)
    await _run(['git', 'config', 'user.name', 'Test'], cwd=repo)
    (repo / 'lib.py').write_text('def greet(name): return name\n')
    await _run(['git', 'add', '-A'], cwd=repo)
    await _run(['git', 'commit', '-m', 'Initial commit'], cwd=repo)


@pytest.fixture
def config(git_repo: Path) -> OrchestratorConfig:
    return OrchestratorConfig(
        project_root=git_repo,
        max_concurrent_tasks=1,
        git=GitConfig(
            main_branch='main',
            branch_prefix='task/',
            remote='origin',
            worktree_dir='.worktrees',
        ),
    )


@pytest.fixture
def git_ops(config: OrchestratorConfig) -> GitOps:
    return GitOps(config.git, config.project_root)


@pytest.fixture
def task_assignment() -> TaskAssignment:
    return TaskAssignment(
        task_id='42',
        task={
            'id': '42',
            'title': 'X',
            'description': 'Y',
            'status': 'pending',
            'metadata': {'files': ['lib']},
            'dependencies': [],
        },
        modules=['lib'],
    )


async def _invoke_probe(role, config, git_ops, task_assignment):
    """Build a TaskWorkflow, patch invoke_with_cap_retry, invoke workflow._invoke(role, ...).

    Returns (call_kwargs, workflow) — the kwargs invoke_with_cap_retry was
    awaited with, and the workflow instance (so callers can assert against
    workflow.modules).
    """
    wt_info = await git_ops.create_worktree(task_assignment.task_id)
    cwd = wt_info.path

    config.sandbox.enabled = True

    workflow = TaskWorkflow(
        assignment=task_assignment,
        config=config,
        git_ops=git_ops,
        scheduler=FakeScheduler(),  # type: ignore[arg-type]
        briefing=FakeBriefing(),  # type: ignore[arg-type]
        mcp=FakeMcp(),
    )
    workflow.artifacts = None

    with patch(
        'orchestrator.workflow.invoke_with_cap_retry',
        new_callable=AsyncMock,
        return_value=AgentResult(success=True, output=''),
    ) as mock_cap_retry:
        await workflow._invoke(role, 'PROMPT', cwd)

    assert mock_cap_retry.await_count == 1, 'invoke_with_cap_retry must be called once'
    assert mock_cap_retry.await_args is not None, 'await_args must be set after one await'
    return mock_cap_retry.await_args.kwargs, workflow


@pytest.mark.asyncio
class TestInvokeDerivesGatingFromRole:
    """_invoke must read MCP-family + sandbox gating off the role OBJECT, not
    by role.name string membership in _MCP_CONFIG_ROLES / _PLAN_TOOLS_ROLES /
    ('implementer', 'debugger').

    This refactor is behavior-preserving for the real shipped roles, so a
    probe using e.g. IMPLEMENTER would pass under BOTH the old name-based
    gates and the new property-based ones — not a valid RED/GREEN signal.
    These probe roles are SYNTHETIC (allowed_tools=[], so __post_init__ is
    inert) and deliberately decouple name from property: only a role whose
    NAME does not match the old hardcoded memberships, but whose PROPERTY
    does request wiring, fails on the name-based code and passes once
    _invoke reads the property instead.
    """

    async def test_sandboxed_property_wires_sandbox_modules_regardless_of_name(
        self, config, git_ops, task_assignment,
    ):
        """probe_sbx: name is neither 'implementer' nor 'debugger', but sandboxed=True."""
        from orchestrator.agents.roles import AgentRole  # noqa: PLC0415

        role = AgentRole(
            name='probe_sbx', system_prompt='x', allowed_tools=[], sandboxed=True,
        )
        call_kwargs, workflow = await _invoke_probe(role, config, git_ops, task_assignment)

        assert call_kwargs.get('sandbox_modules') == workflow.modules, (
            f'Expected sandbox_modules == {workflow.modules!r} (role.sandboxed=True) '
            f"but got {call_kwargs.get('sandbox_modules')!r}. _invoke must gate "
            "sandboxing off role.sandboxed, not role.name in ('implementer', 'debugger')."
        )

    async def test_plan_tools_family_wires_plan_tools_server_regardless_of_name(
        self, config, git_ops, task_assignment,
    ):
        """probe_plan: name is not in the historical _PLAN_TOOLS_ROLES tuple,
        but declares mcp_families={'plan_tools'}."""
        from orchestrator.agents.roles import AgentRole  # noqa: PLC0415

        role = AgentRole(
            name='probe_plan', system_prompt='x', allowed_tools=[],
            mcp_families=frozenset({'plan_tools'}),
        )
        call_kwargs, _workflow = await _invoke_probe(role, config, git_ops, task_assignment)

        mcp_config = call_kwargs.get('mcp_config')
        servers = (mcp_config or {}).get('mcpServers', {})
        assert 'plan-tools' in servers, (
            "Expected mcp_config['mcpServers'] to contain 'plan-tools' "
            f"(role.mcp_families={{'plan_tools'}}) but got {mcp_config!r}. _invoke "
            "must gate the plan-tools injection off 'plan_tools' in "
            'role.mcp_families, not role.name in _PLAN_TOOLS_ROLES.'
        )

    async def test_orchestrator_family_wires_mcp_config_regardless_of_name(
        self, config, git_ops, task_assignment,
    ):
        """probe_orch: name is not in the historical _MCP_CONFIG_ROLES tuple,
        but declares mcp_families={'orchestrator'}."""
        from orchestrator.agents.roles import AgentRole  # noqa: PLC0415

        role = AgentRole(
            name='probe_orch', system_prompt='x', allowed_tools=[],
            mcp_families=frozenset({'orchestrator'}),
        )
        call_kwargs, _workflow = await _invoke_probe(role, config, git_ops, task_assignment)

        assert call_kwargs.get('mcp_config') is not None, (
            "Expected mcp_config to be built (role.mcp_families={'orchestrator'}) "
            f"but got {call_kwargs.get('mcp_config')!r}. _invoke must gate the "
            "orchestrator-assembled MCP config off 'orchestrator' in "
            'role.mcp_families, not role.name in _MCP_CONFIG_ROLES.'
        )

    async def test_bare_role_gets_no_sandbox_and_no_plan_tools(
        self, config, git_ops, task_assignment,
    ):
        """Negative control: a role declaring neither family and unsandboxed
        gets no sandbox modules and no plan-tools server wired, exactly as
        it would under the old name-based gates (name matches nothing)."""
        from orchestrator.agents.roles import AgentRole  # noqa: PLC0415

        role = AgentRole(name='probe_bare', system_prompt='x', allowed_tools=[])
        call_kwargs, _workflow = await _invoke_probe(role, config, git_ops, task_assignment)

        assert call_kwargs.get('sandbox_modules') is None, (
            f"Expected sandbox_modules is None (role.sandboxed=False) but got "
            f"{call_kwargs.get('sandbox_modules')!r}."
        )
        mcp_config = call_kwargs.get('mcp_config')
        servers = (mcp_config or {}).get('mcpServers', {})
        assert 'plan-tools' not in servers, (
            f"Expected no 'plan-tools' server wired (role.mcp_families=frozenset()) "
            f'but found one in mcp_config={mcp_config!r}.'
        )
