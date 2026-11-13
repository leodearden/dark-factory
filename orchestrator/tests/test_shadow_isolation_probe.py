"""The deterministic isolation probe a shadow cell is opened behind (task 5386).

The probe reads the legs a representative shadow cell would dispatch, built
from the same producers dispatch uses, and names every gap that would let a
candidate reach production. The coordinator refuses to open a cell while any
gap stands. No host probe runs: every backend resolution is injected.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pytest
import pytest_asyncio
from _shadow_coordinator_doubles import build_coordinator, make_task
from shared.cost_store import CostStore
from shared.eval_lane import is_eval_fixture_task_id

from orchestrator.agents.roles import IMPLEMENTER, ROLES
from orchestrator.agents.sandbox_dispatch import Backend
from orchestrator.config import OrchestratorConfig
from orchestrator.evals.configs import EvalConfig
from orchestrator.evals.live_fixture import build_live_fixture
from orchestrator.evals.profile import apply_eval_profile
from orchestrator.evals.runner import build_eval_orch_config, eval_mcp_lifecycle
from orchestrator.evals.shadow_invocation import build_shadow_orch_config
from orchestrator.evals.shadow_isolation import (
    IsolationGap,
    LegDispatch,
    LegInvocation,
    probe_isolation,
    shadow_isolation_available,
    shadow_leg_invocations,
    workflow_leg,
)
from orchestrator.evals.snapshots import eval_worktree_root
from orchestrator.event_store import EventType
from orchestrator.fm_retry import FM_NULL_SENTINEL_URL
from orchestrator.mcp_lifecycle import McpLifecycle
from orchestrator.workflow import PER_INVOCATION_MCP_SERVERS, PLAN_TOOLS_MCP_SERVER
from orchestrator.workflow_types import WorkflowState

_SHADOW_ID = 'shadow_4242_0123456789abcdef0123456789abcdef'
_IMPL = EvalConfig('probe', 'claude', 'sonnet', 'high')
_SANDBOXED_ROLES = {role.name for role in ROLES.values() if role.sandboxed}


def _landlock() -> Backend:
    return 'landlock'


def _no_backend() -> Backend:
    return 'none'


class _BackendProbeFailed(RuntimeError):
    pass


def _failing_backend() -> Backend:
    raise _BackendProbeFailed('backend probe exploded')


def _shadow_fixture(project_root: Path) -> dict:
    return build_live_fixture(
        {'id': '4242', 'title': 'T', 'metadata': {}},
        base_sha='a' * 40,
        project_root=project_root,
        plan={'steps': [{'id': 'step-1', 'type': 'test', 'description': 'd', 'status': 'pending'}]},
        verify_commands={'test': 'true', 'lint': 'true', 'typecheck': 'true'},
        shape='implementer',
        cell_id='0123456789abcdef0123456789abcdef',
    )


def _eval_cwd(project_root: Path) -> Path:
    return eval_worktree_root(project_root) / _SHADOW_ID


def _production_implementer_leg(tmp_path: Path) -> LegInvocation:
    production = OrchestratorConfig(project_root=tmp_path)
    return workflow_leg(
        production, IMPLEMENTER, task_id=_SHADOW_ID, cwd=_eval_cwd(tmp_path),
        mcp=McpLifecycle(production), resolve_backend=_landlock,
    )


def _gaps(legs: tuple[LegInvocation, ...] | list[LegInvocation]) -> list[IsolationGap]:
    return [finding.gap for finding in probe_isolation(tuple(legs)).findings]


def _workflow_leg_for(legs: tuple[LegInvocation, ...], role: str) -> LegInvocation:
    [leg] = [leg for leg in legs if leg.dispatch is LegDispatch.WORKFLOW and leg.role == role]
    return leg


def test_the_built_shadow_implementer_leg_is_isolated(tmp_path: Path) -> None:
    legs = shadow_leg_invocations(OrchestratorConfig(project_root=tmp_path), resolve_backend=_landlock)

    leg = _workflow_leg_for(legs, 'implementer')
    assert leg.strict_mcp_config is True
    assert leg.memory_url == FM_NULL_SENTINEL_URL
    assert leg.sandbox_modules == ()
    assert leg.sandbox_backend == 'landlock'
    assert is_eval_fixture_task_id(leg.task_id)
    assert leg.cwd.is_relative_to(eval_worktree_root(tmp_path.resolve()))


def test_the_built_legs_cover_every_role_plus_the_architect_eval(tmp_path: Path) -> None:
    legs = shadow_leg_invocations(OrchestratorConfig(project_root=tmp_path), resolve_backend=_landlock)

    workflow_roles = sorted(leg.role for leg in legs if leg.dispatch is LegDispatch.WORKFLOW)
    assert workflow_roles == sorted(role.name for role in ROLES.values())
    [architect_eval] = [leg for leg in legs if leg.dispatch is LegDispatch.ARCHITECT_EVAL]
    assert architect_eval.role == 'architect'
    assert architect_eval.strict_mcp_config is True
    assert architect_eval.sandbox_modules is None


def test_the_built_legs_carry_only_per_invocation_mcp_servers(tmp_path: Path) -> None:
    legs = shadow_leg_invocations(OrchestratorConfig(project_root=tmp_path), resolve_backend=_landlock)

    assert all(leg.mcp_servers <= PER_INVOCATION_MCP_SERVERS for leg in legs)
    assert _workflow_leg_for(legs, 'implementer').mcp_servers == {PLAN_TOOLS_MCP_SERVER}
    [architect_eval] = [leg for leg in legs if leg.dispatch is LegDispatch.ARCHITECT_EVAL]
    assert architect_eval.mcp_servers == {PLAN_TOOLS_MCP_SERVER}


def test_a_shadow_leg_handed_the_production_mcp_lifecycle_is_live_mcp_server(
    tmp_path: Path,
) -> None:
    shadow = build_shadow_orch_config(
        _IMPL, _shadow_fixture(tmp_path), OrchestratorConfig(project_root=tmp_path),
    )
    leg = workflow_leg(
        shadow, IMPLEMENTER, task_id=_SHADOW_ID, cwd=_eval_cwd(tmp_path),
        mcp=McpLifecycle(shadow), resolve_backend=_landlock,
    )

    [finding] = probe_isolation((leg,)).findings

    assert finding.gap is IsolationGap.LIVE_MCP_SERVER
    assert 'fused-memory' in finding.observed
    assert 'escalation' in finding.observed
    assert PLAN_TOOLS_MCP_SERVER not in finding.observed


def test_the_built_shadow_profile_has_no_isolation_gap(tmp_path: Path) -> None:
    base = OrchestratorConfig(project_root=tmp_path)

    verdict = probe_isolation(shadow_leg_invocations(base, resolve_backend=_landlock))

    assert verdict.isolation_unavailable is False
    assert verdict.findings == ()
    assert shadow_isolation_available(base, resolve_backend=_landlock) is True


def test_a_production_profile_leg_is_ambient_mcp_and_live_memory(tmp_path: Path) -> None:
    verdict = probe_isolation((_production_implementer_leg(tmp_path),))

    assert verdict.isolation_unavailable is True
    assert [finding.gap for finding in verdict.findings] == [
        IsolationGap.STRICT_MCP_OFF, IsolationGap.LIVE_MCP_SERVER, IsolationGap.LIVE_MEMORY_ENDPOINT,
    ]


def test_dropping_strict_mcp_alone_is_exactly_strict_mcp_off(tmp_path: Path) -> None:
    config = apply_eval_profile(OrchestratorConfig(project_root=tmp_path)).model_copy(
        update={'strict_mcp_config': False},
    )

    leg = workflow_leg(
        config, IMPLEMENTER, task_id=_SHADOW_ID, cwd=_eval_cwd(tmp_path),
        mcp=eval_mcp_lifecycle(config), resolve_backend=_landlock,
    )

    assert _gaps([leg]) == [IsolationGap.STRICT_MCP_OFF]


def test_a_production_task_id_and_lane_is_escalation_uncontained(tmp_path: Path) -> None:
    cwd = tmp_path / '.worktrees' / '4242'
    config = apply_eval_profile(OrchestratorConfig(project_root=tmp_path))
    leg = workflow_leg(
        config, IMPLEMENTER, task_id='4242', cwd=cwd,
        mcp=eval_mcp_lifecycle(config), resolve_backend=_landlock,
    )

    [finding] = probe_isolation((leg,)).findings

    assert finding.gap is IsolationGap.ESCALATION_UNCONTAINED
    assert '4242' in finding.observed
    assert str(cwd) in finding.observed


def test_a_corpus_profile_sandboxed_leg_is_sandbox_off(tmp_path: Path) -> None:
    corpus = build_eval_orch_config(
        _IMPL, _shadow_fixture(tmp_path), OrchestratorConfig(project_root=tmp_path),
    )

    legs = [
        workflow_leg(corpus, role, task_id=_SHADOW_ID, cwd=_eval_cwd(tmp_path),
                     mcp=eval_mcp_lifecycle(corpus), resolve_backend=_landlock)
        for role in ROLES.values()
    ]
    findings = probe_isolation(tuple(legs)).findings

    assert {finding.gap for finding in findings} == {IsolationGap.SANDBOX_OFF}
    assert {finding.role for finding in findings} == _SANDBOXED_ROLES


def test_a_shadow_profile_leg_under_no_backend_is_sandbox_unwrapped(tmp_path: Path) -> None:
    shadow = build_shadow_orch_config(
        _IMPL, _shadow_fixture(tmp_path), OrchestratorConfig(project_root=tmp_path),
    )

    leg = workflow_leg(
        shadow, IMPLEMENTER, task_id=_SHADOW_ID, cwd=_eval_cwd(tmp_path),
        mcp=eval_mcp_lifecycle(shadow), resolve_backend=_no_backend,
    )

    assert _gaps([leg]) == [IsolationGap.SANDBOX_UNWRAPPED]


def test_no_backend_leaves_the_unsandboxed_legs_without_a_sandbox_gap(tmp_path: Path) -> None:
    legs = shadow_leg_invocations(OrchestratorConfig(project_root=tmp_path), resolve_backend=_no_backend)

    findings = probe_isolation(legs).findings

    assert {finding.gap for finding in findings} == {IsolationGap.SANDBOX_UNWRAPPED}
    assert {finding.role for finding in findings} == _SANDBOXED_ROLES
    assert all(finding.dispatch is LegDispatch.WORKFLOW for finding in findings)


def test_an_empty_probe_refuses_to_claim_isolation() -> None:
    with pytest.raises(ValueError):
        probe_isolation(())


def test_shadow_isolation_available_refuses_and_logs_every_finding(
    tmp_path: Path, caplog: pytest.LogCaptureFixture,
) -> None:
    base = OrchestratorConfig(project_root=tmp_path)
    findings = probe_isolation(shadow_leg_invocations(base, resolve_backend=_no_backend)).findings

    with caplog.at_level(logging.WARNING, logger='orchestrator.evals.shadow_isolation'):
        available = shadow_isolation_available(base, resolve_backend=_no_backend)

    assert available is False
    warnings = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == len(findings) > 0
    for finding, message in zip(findings, warnings, strict=True):
        assert finding.role in message
        assert str(finding.gap) in message


def test_a_probe_that_cannot_build_its_legs_fails_closed(
    tmp_path: Path, caplog: pytest.LogCaptureFixture,
) -> None:
    base = OrchestratorConfig(project_root=tmp_path)

    with caplog.at_level(logging.ERROR, logger='orchestrator.evals.shadow_isolation'):
        available = shadow_isolation_available(base, resolve_backend=_failing_backend)

    assert available is False
    [record] = [r for r in caplog.records if r.levelno == logging.ERROR]
    assert record.exc_info is not None
    assert isinstance(record.exc_info[1], _BackendProbeFailed)


@pytest_asyncio.fixture
async def costs(tmp_path):
    async with CostStore(tmp_path / 'runs.db') as store:
        yield store


@pytest.mark.asyncio
@pytest.mark.parametrize(
    'resolve_backend', [_no_backend, _failing_backend], ids=['gap-found', 'probe-failed'],
)
async def test_the_coordinator_skips_a_cell_whose_profile_does_not_isolate(
    tmp_path: Path, costs: CostStore, resolve_backend,
) -> None:
    base = OrchestratorConfig(project_root=tmp_path)
    rig = build_coordinator(
        tmp_path, costs,
        isolation_probe=lambda: shadow_isolation_available(base, resolve_backend=resolve_backend),
    )

    opened = await rig.coordinator.on_phase(make_task('5387'), WorkflowState.PLAN, entering=False)
    await rig.coordinator.join()

    assert opened == []
    skips = rig.events_of(EventType.shadow_cell_skipped)
    assert [row['data']['reason'] for row in skips] == ['isolation_unavailable']
    assert rig.coordinator.storm_state().paused is False


@pytest.mark.asyncio
async def test_the_coordinator_opens_a_cell_behind_the_shadow_isolation_probe(
    tmp_path: Path, costs: CostStore,
) -> None:
    base = OrchestratorConfig(project_root=tmp_path)
    rig = build_coordinator(
        tmp_path, costs,
        isolation_probe=lambda: shadow_isolation_available(base, resolve_backend=_landlock),
    )

    opened = await rig.coordinator.on_phase(make_task('5387'), WorkflowState.PLAN, entering=False)
    await rig.coordinator.join()

    assert len(opened) == 1
