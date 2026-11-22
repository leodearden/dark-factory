"""The invocation profile a live shadow cell runs under, and the only entry points that run one.

Leaf δ of ``plans/live-shadow-eval-prd.md`` (task 5386). The probe that refuses
a cell until this profile isolates is ``evals/shadow_isolation.py``.
"""

from __future__ import annotations

from pathlib import Path

from shared.eval_lane import SHADOW_FIXTURE_ID_RE

from orchestrator.config import OrchestratorConfig
from orchestrator.evals.configs import EvalConfig
from orchestrator.evals.eval_gate import GATE_UNSET, InjectedGate
from orchestrator.evals.results import EvalResult, ResultStore
from orchestrator.evals.runner import (
    CellProfile,
    build_eval_orch_config,
    run_end_to_end,
    run_eval,
)
from orchestrator.verify import run_scoped_verification

__all__ = [
    'SHADOW_PROFILE',
    'build_shadow_orch_config',
    'run_shadow_end_to_end',
    'run_shadow_eval',
]


def build_shadow_orch_config(
    config: EvalConfig,
    task: dict,
    base_config: OrchestratorConfig | None = None,
    memory_endpoint: str | None = None,
    architect_config: EvalConfig | None = None,
    judge_config: EvalConfig | None = None,
) -> OrchestratorConfig:
    """The corpus eval config for a shadow fixture; it refuses a memory endpoint and a non-shadow fixture id.

    Strict MCP, the null memory sentinel and the base's sandbox are inherited
    through ``build_eval_orch_config``.
    """
    if memory_endpoint is not None:
        raise ValueError(
            f'build_shadow_orch_config: memory_endpoint={memory_endpoint!r} is refused; '
            f"a shadow cell's harness memory stays on the null sentinel"
        )
    task_id = str(task.get('id') or '')
    if not SHADOW_FIXTURE_ID_RE.fullmatch(task_id):
        raise ValueError(
            f'build_shadow_orch_config: fixture id {task_id!r} is not a shadow fixture id '
            f'(shared/src/shared/eval_lane.py::SHADOW_FIXTURE_ID_RE); only a fixture '
            f'minted by evals/live_fixture.py::build_live_fixture is contained by '
            f'escalation Gate 0'
        )
    return build_eval_orch_config(
        config, task, base_config,
        architect_config=architect_config, judge_config=judge_config,
    )


SHADOW_PROFILE = CellProfile(build_shadow_orch_config, run_scoped_verification)
"""Shadow cells replay live tasks cut from recent main, so production's scoped gate fits."""


async def run_shadow_eval(
    task_path: Path,
    config: EvalConfig,
    base_config: OrchestratorConfig,
    *,
    trial: int = 1,
    timeout_override: int | None = None,
    worktree_path: Path | None = None,
    judge_config: EvalConfig | None = None,
    usage_gate: InjectedGate = GATE_UNSET,
    store: ResultStore | None = None,
) -> EvalResult:
    """``run_eval`` under ``SHADOW_PROFILE``."""
    return await run_eval(
        task_path, config, base_config, trial, timeout_override, worktree_path,
        judge_config=judge_config, usage_gate=usage_gate, store=store,
        profile=SHADOW_PROFILE,
    )


async def run_shadow_end_to_end(
    task_path: Path,
    arch_config: EvalConfig,
    impl_config: EvalConfig,
    base_config: OrchestratorConfig,
    *,
    trial: int = 1,
    timeout_override: int | None = None,
    usage_gate: InjectedGate = GATE_UNSET,
    store: ResultStore | None = None,
) -> EvalResult:
    """``run_end_to_end`` under ``SHADOW_PROFILE``."""
    return await run_end_to_end(
        task_path, arch_config, impl_config, base_config, trial, timeout_override,
        usage_gate=usage_gate, store=store, profile=SHADOW_PROFILE,
    )
