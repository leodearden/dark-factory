"""The isolation one role's invocation carries under one config (task 5386).

``invocation_scope`` is the single decision ``TaskWorkflow._invoke`` and the
shadow isolation probe both read, so these cases pin it directly.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest

from orchestrator.agents.invocation_scope import InvocationScope, invocation_scope
from orchestrator.agents.roles import ARCHITECT, IMPLEMENTER, ROLES
from orchestrator.config import OrchestratorConfig, SandboxConfig
from orchestrator.evals.profile import apply_eval_profile


@pytest.mark.parametrize('role', list(ROLES.values()), ids=list(ROLES))
def test_production_default_scope_is_each_roles_own(tmp_path: Path, role) -> None:
    config = OrchestratorConfig(project_root=tmp_path)
    assert config.strict_mcp_config is False
    assert config.sandbox.enabled is True

    scope = invocation_scope(config, role)

    assert scope.strict_mcp_config == role.strict_mcp_scope
    assert scope.sandboxed == role.sandboxed


@pytest.mark.parametrize('role', list(ROLES.values()), ids=list(ROLES))
def test_eval_profile_scopes_every_role_strict(tmp_path: Path, role) -> None:
    config = apply_eval_profile(OrchestratorConfig(project_root=tmp_path))

    assert invocation_scope(config, role).strict_mcp_config is True


def test_sandboxed_role_with_sandbox_enabled_gets_the_empty_list_gate(tmp_path: Path) -> None:
    config = OrchestratorConfig(project_root=tmp_path, sandbox=SandboxConfig(enabled=True))

    assert invocation_scope(config, IMPLEMENTER).sandbox_modules == []


def test_sandbox_disabled_leaves_sandboxed_role_ungated(tmp_path: Path) -> None:
    config = OrchestratorConfig(project_root=tmp_path, sandbox=SandboxConfig(enabled=False))

    assert invocation_scope(config, IMPLEMENTER).sandbox_modules is None


def test_unsandboxed_role_is_ungated_even_with_sandbox_enabled(tmp_path: Path) -> None:
    config = OrchestratorConfig(project_root=tmp_path, sandbox=SandboxConfig(enabled=True))

    assert ARCHITECT.sandboxed is False
    assert invocation_scope(config, ARCHITECT).sandbox_modules is None


def test_sandbox_modules_is_a_fresh_list_per_access() -> None:
    scope = InvocationScope(strict_mcp_config=False, sandboxed=True)

    first = scope.sandbox_modules
    assert first is not None
    first.append('leaked')

    assert scope.sandbox_modules == []


def test_invocation_scope_is_frozen() -> None:
    scope = InvocationScope(strict_mcp_config=True, sandboxed=False)

    pytest.raises(dataclasses.FrozenInstanceError, setattr, scope, 'sandboxed', True)
