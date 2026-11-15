"""The deterministic probe a live shadow cell is opened behind.

It builds the legs a representative shadow cell would dispatch, from the same
producers dispatch uses, and names every gap that would let a candidate reach
production. Leaf δ of ``plans/live-shadow-eval-prd.md`` (task 5386).
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from shared.eval_lane import is_eval_fixture_task_id, is_eval_worktree_path

from orchestrator.agents.invocation_scope import invocation_scope
from orchestrator.agents.roles import ARCHITECT, ROLES, AgentRole
from orchestrator.agents.sandbox_dispatch import Backend, resolve_active_backend
from orchestrator.artifacts import TaskArtifacts
from orchestrator.config import OrchestratorConfig
from orchestrator.evals.architect_eval import architect_invocation
from orchestrator.evals.configs import EvalConfig
from orchestrator.evals.live_fixture import build_live_fixture
from orchestrator.evals.runner import eval_mcp_lifecycle
from orchestrator.evals.shadow_invocation import build_shadow_orch_config
from orchestrator.evals.snapshots import eval_worktree_root
from orchestrator.fm_retry import is_fm_null_sentinel
from orchestrator.workflow import (
    PER_INVOCATION_MCP_SERVERS,
    McpConfigSource,
    compose_invocation_mcp_config,
    escalation_mcp_url,
)

logger = logging.getLogger(__name__)

__all__ = [
    'IsolationFinding',
    'IsolationGap',
    'IsolationVerdict',
    'LegDispatch',
    'LegInvocation',
    'architect_eval_leg',
    'probe_isolation',
    'shadow_isolation_available',
    'shadow_leg_invocations',
    'workflow_leg',
]

class LegDispatch(StrEnum):
    """The spawn site a shadow cell's agent invocation goes through."""

    WORKFLOW = 'workflow'
    ARCHITECT_EVAL = 'architect_eval'


class IsolationGap(StrEnum):
    """One way a leg could reach production."""

    STRICT_MCP_OFF = 'strict_mcp_off'
    LIVE_MCP_SERVER = 'live_mcp_server'
    LIVE_MEMORY_ENDPOINT = 'live_memory_endpoint'
    SANDBOX_OFF = 'sandbox_off'
    SANDBOX_UNWRAPPED = 'sandbox_unwrapped'
    ESCALATION_UNCONTAINED = 'escalation_uncontained'


@dataclass(frozen=True)
class LegInvocation:
    """The isolation-bearing values one leg's agent invocation would carry."""

    dispatch: LegDispatch
    role: str
    task_id: str
    cwd: Path
    strict_mcp_config: bool
    mcp_servers: frozenset[str]
    sandboxed_role: bool
    sandbox_modules: tuple[str, ...] | None
    sandbox_backend: Backend | None
    memory_url: str


@dataclass(frozen=True)
class IsolationFinding:
    dispatch: LegDispatch
    role: str
    gap: IsolationGap
    observed: str


@dataclass(frozen=True)
class IsolationVerdict:
    findings: tuple[IsolationFinding, ...]

    @property
    def isolation_unavailable(self) -> bool:
        """What the coordinator reports as ``ShadowSkipReason.ISOLATION_UNAVAILABLE``."""
        return bool(self.findings)


def _server_names(mcp_config: dict | None) -> frozenset[str]:
    return frozenset((mcp_config or {}).get('mcpServers', {}))


def workflow_leg(
    config: OrchestratorConfig,
    role: AgentRole,
    *,
    task_id: str,
    cwd: Path,
    mcp: McpConfigSource,
    resolve_backend: Callable[[], Backend],
) -> LegInvocation:
    """The leg ``TaskWorkflow._invoke`` dispatches for *role* under *config* and *mcp*.

    The escalation server is always offered, so a lifecycle that would add it is caught.
    """
    scope = invocation_scope(config, role)
    mcp_config = compose_invocation_mcp_config(
        role,
        mcp=mcp,
        escalation_url=escalation_mcp_url(config.escalation),
        cwd=cwd,
        task_root=TaskArtifacts.meta_root_for(cwd.parent, cwd.name),
        strict_mcp_config=scope.strict_mcp_config,
    )
    modules = scope.sandbox_modules
    sandbox_modules = None if modules is None else tuple(modules)
    return LegInvocation(
        dispatch=LegDispatch.WORKFLOW,
        role=role.name,
        task_id=task_id,
        cwd=cwd,
        strict_mcp_config=scope.strict_mcp_config,
        mcp_servers=_server_names(mcp_config),
        sandboxed_role=role.sandboxed,
        sandbox_modules=sandbox_modules,
        sandbox_backend=None if sandbox_modules is None else resolve_backend(),
        memory_url=config.fused_memory.url,
    )


_PROBE_PROMPT = 'shadow isolation probe'


def architect_eval_leg(
    task: dict, candidate: EvalConfig, config: OrchestratorConfig, *, cwd: Path,
) -> LegInvocation:
    """The leg ``evals/architect_eval.py`` dispatches for *candidate*'s live architect."""
    kwargs = architect_invocation(task, candidate, _PROBE_PROMPT, cwd)
    modules = kwargs.get('sandbox_modules')
    return LegInvocation(
        dispatch=LegDispatch.ARCHITECT_EVAL,
        role=ARCHITECT.name,
        task_id=str(task['id']),
        cwd=cwd,
        strict_mcp_config=kwargs.get('strict_mcp_config') is True,
        mcp_servers=_server_names(kwargs.get('mcp_config')),
        sandboxed_role=ARCHITECT.sandboxed,
        sandbox_modules=None if modules is None else tuple(modules),
        sandbox_backend=None,
        memory_url=config.fused_memory.url,
    )


_PROBE_TASK = {'id': '0', 'title': 'shadow isolation probe', 'metadata': {}}
_PROBE_PLAN = {
    'steps': [{'id': 'step-1', 'type': 'test', 'description': 'probe', 'status': 'pending'}],
}
_PROBE_GATES = {'test': 'true', 'lint': 'true', 'typecheck': 'true'}
_PROBE_CANDIDATE = EvalConfig('shadow-isolation-probe', 'claude', 'sonnet', 'high')
_PROBE_CELL_ID = '0' * 32


def shadow_leg_invocations(
    base_config: OrchestratorConfig,
    *,
    resolve_backend: Callable[[], Backend] = resolve_active_backend,
) -> tuple[LegInvocation, ...]:
    """Every leg a representative shadow cell on *base_config*'s project would dispatch."""
    fixture = build_live_fixture(
        _PROBE_TASK,
        base_sha='0' * 40,
        project_root=Path(base_config.project_root).resolve(),
        plan=_PROBE_PLAN,
        verify_commands=_PROBE_GATES,
        shape='implementer',
        cell_id=_PROBE_CELL_ID,
    )
    config = build_shadow_orch_config(_PROBE_CANDIDATE, fixture, base_config)
    cwd = eval_worktree_root(fixture['project_root']) / fixture['id']
    mcp = eval_mcp_lifecycle(config)
    workflow_legs = tuple(
        workflow_leg(
            config, role, task_id=fixture['id'], cwd=cwd, mcp=mcp, resolve_backend=resolve_backend,
        )
        for role in ROLES.values()
    )
    return (*workflow_legs, architect_eval_leg(fixture, _PROBE_CANDIDATE, config, cwd=cwd))


def _strict_mcp_off(leg: LegInvocation) -> str | None:
    return None if leg.strict_mcp_config else 'strict_mcp_config=False'


def _live_mcp_server(leg: LegInvocation) -> str | None:
    shared_servers = leg.mcp_servers - PER_INVOCATION_MCP_SERVERS
    return f'mcp_servers={sorted(shared_servers)!r}' if shared_servers else None


def _live_memory_endpoint(leg: LegInvocation) -> str | None:
    return None if is_fm_null_sentinel(leg.memory_url) else f'memory_url={leg.memory_url!r}'


def _sandbox_off(leg: LegInvocation) -> str | None:
    if leg.sandboxed_role and leg.sandbox_modules is None:
        return 'sandboxed role dispatched with sandbox_modules=None'
    return None


def _sandbox_unwrapped(leg: LegInvocation) -> str | None:
    if leg.sandbox_modules is not None and leg.sandbox_backend == 'none':
        return f'sandbox backend resolves to {leg.sandbox_backend!r}'
    return None


def _escalation_uncontained(leg: LegInvocation) -> str | None:
    if is_eval_fixture_task_id(leg.task_id) and is_eval_worktree_path(leg.cwd):
        return None
    return f'task_id={leg.task_id!r} cwd={str(leg.cwd)!r}'


# Ordered like shadow_eval.py::_OPEN_GATES: each rule returns the offending
# value it observed, or None when the leg passes it.
_ISOLATION_RULES: tuple[tuple[IsolationGap, Callable[[LegInvocation], str | None]], ...] = (
    (IsolationGap.STRICT_MCP_OFF, _strict_mcp_off),
    (IsolationGap.LIVE_MCP_SERVER, _live_mcp_server),
    (IsolationGap.LIVE_MEMORY_ENDPOINT, _live_memory_endpoint),
    (IsolationGap.SANDBOX_OFF, _sandbox_off),
    (IsolationGap.SANDBOX_UNWRAPPED, _sandbox_unwrapped),
    (IsolationGap.ESCALATION_UNCONTAINED, _escalation_uncontained),
)


def probe_isolation(legs: tuple[LegInvocation, ...]) -> IsolationVerdict:
    """Every gap in *legs*, in leg order then rule order."""
    if not legs:
        raise ValueError('probe_isolation: no legs to probe; an empty probe cannot claim isolation')
    return IsolationVerdict(tuple(
        IsolationFinding(leg.dispatch, leg.role, gap, observed)
        for leg in legs
        for gap, rule in _ISOLATION_RULES
        if (observed := rule(leg)) is not None
    ))


def shadow_isolation_available(
    base_config: OrchestratorConfig,
    *,
    resolve_backend: Callable[[], Backend] = resolve_active_backend,
) -> bool:
    """Whether a shadow cell on *base_config*'s project may open; the coordinator's isolation dep.

    A probe that cannot build its legs fails closed, so the coordinator still
    names the skip ``isolation_unavailable``.
    """
    try:
        legs = shadow_leg_invocations(base_config, resolve_backend=resolve_backend)
    except Exception:
        logger.exception('shadow isolation unavailable: the probe could not build its legs')
        return False
    verdict = probe_isolation(legs)
    for finding in verdict.findings:
        logger.warning(
            'shadow isolation unavailable: %s leg %s has %s (%s)',
            finding.dispatch, finding.role, finding.gap, finding.observed,
        )
    return not verdict.isolation_unavailable
