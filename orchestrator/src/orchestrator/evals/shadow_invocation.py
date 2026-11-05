"""The invocation profile a live shadow cell runs under.

Later in this file sits the probe that refuses cells until that profile
isolates. Leaf δ of ``plans/live-shadow-eval-prd.md`` (task 5386).
"""

from __future__ import annotations

from shared.eval_lane import SHADOW_FIXTURE_ID_RE

from orchestrator.config import OrchestratorConfig
from orchestrator.evals.configs import EvalConfig
from orchestrator.evals.runner import build_eval_orch_config

__all__ = ['build_shadow_orch_config']


def build_shadow_orch_config(
    config: EvalConfig,
    task: dict,
    base_config: OrchestratorConfig | None = None,
    memory_endpoint: str | None = None,
    architect_config: EvalConfig | None = None,
    judge_config: EvalConfig | None = None,
) -> OrchestratorConfig:
    """The corpus eval config with sandboxed roles run under the operator's backend.

    Strict MCP and the null memory sentinel are ``EVAL_PROFILE``'s, inherited
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
    evaled = build_eval_orch_config(
        config, task, base_config,
        architect_config=architect_config, judge_config=judge_config,
    )
    assert base_config is not None, 'build_eval_orch_config refuses a None base_config'
    return evaled.model_copy(
        update={'sandbox': base_config.sandbox.model_copy(update={'enabled': True})},
    )
