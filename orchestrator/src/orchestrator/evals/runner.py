"""Main eval orchestrator: run (task, config) pairs through the real workflow."""

from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from orchestrator.agents.briefing import BriefingAssembler
from orchestrator.config import (
    BackendsConfig,
    BudgetsConfig,
    EffortConfig,
    ModelsConfig,
    OrchestratorConfig,
    SandboxConfig,
    load_config,
)
from orchestrator.git_ops import GitOps
from orchestrator.scheduler import TaskAssignment
from orchestrator.workflow import TaskWorkflow, WorkflowOutcome

from .configs import EVAL_CONFIGS, EvalConfig
from .metrics import collect_metrics
from .snapshots import create_eval_worktree

logger = logging.getLogger(__name__)

RESULTS_DIR = Path(__file__).parent / 'results'


@dataclass
class EvalResult:
    """Result of one (task, config) eval run."""

    task_id: str
    config_name: str
    outcome: str
    metrics: dict[str, Any]
    worktree_path: str
    wall_clock_ms: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_task(task_path: Path) -> dict:
    """Load a task definition JSON file."""
    with open(task_path) as f:
        return json.load(f)


def build_eval_orch_config(
    config: EvalConfig,
    task: dict,
    base_config: OrchestratorConfig | None = None,
) -> OrchestratorConfig:
    """Build an OrchestratorConfig override for this eval run.

    Architect always runs on Claude opus-high (constant planning).
    Reviewers always run on Claude sonnet (consistent quality bar).
    Only the implementer varies per eval config.
    """
    base = base_config or load_config()

    models = ModelsConfig(
        architect='opus',
        implementer=config.model,
        debugger=config.model,
        reviewer='sonnet',        # sonnet to match production, save cap budget
        merger='opus',
        module_tagger='sonnet',
    )

    budgets = BudgetsConfig(
        architect=5.0,
        implementer=config.max_budget_usd,
        debugger=config.max_budget_usd / 2,
        reviewer=2.0,             # sonnet reviewers
        merger=5.0,
        module_tagger=2.0,
    )

    effort = EffortConfig(
        architect='high',
        implementer=config.effort or 'high',
        debugger=config.effort or 'high',
        reviewer='medium',         # sonnet reviewers (matches production)
        merger='high',
        module_tagger='medium',
    )

    backends = BackendsConfig(
        architect='claude',
        implementer=config.backend,
        debugger=config.backend,
        reviewer='claude',        # reviewers always on Claude
        merger='claude',
        module_tagger='claude',
    )

    # Create config with overrides
    # max_review_cycles=1: one review pass, no re-architect loop
    return OrchestratorConfig(
        models=models,
        budgets=budgets,
        effort=effort,
        backends=backends,
        max_turns=base.max_turns,
        max_execute_iterations=base.max_execute_iterations,
        max_verify_attempts=base.max_verify_attempts,
        max_review_cycles=1,
        test_command=task.get('verify_commands', {}).get('test', base.test_command),
        lint_command=task.get('verify_commands', {}).get('lint', base.lint_command),
        type_check_command=task.get('verify_commands', {}).get('typecheck', base.type_check_command),
        fused_memory=base.fused_memory,
        sandbox=SandboxConfig(enabled=False),
        escalation=base.escalation,
        git=base.git,
        usage_cap=base.usage_cap,
        project_root=Path(task.get('project_root', str(base.project_root))),
    )


async def run_eval(
    task_path: Path,
    config: EvalConfig,
    base_config: OrchestratorConfig | None = None,
) -> EvalResult:
    """Run one (task, config) pair through PLAN→EXECUTE→VERIFY→REVIEW."""
    task = load_task(task_path)
    task_id = task['id']
    project_root = Path(task['project_root'])

    logger.info(f'Starting eval: {task_id} × {config.name}')
    start_ms = int(time.monotonic() * 1000)

    # 1. Create isolated worktree at pre-task commit (with env setup)
    worktree = await create_eval_worktree(
        project_root, task_id, task['pre_task_commit'],
        setup_commands=task.get('setup_commands'),
    )

    # 2. Build orchestrator config for this eval
    orch_config = build_eval_orch_config(config, task, base_config)

    # 3. Build task assignment
    task_def = task.get('task_definition', {
        'title': task.get('name', task_id),
        'description': task.get('name', ''),
    })
    modules = task.get('modules', [])

    assignment = TaskAssignment(
        task_id=task_id,
        task=task_def,
        modules=list(modules),
    )

    # 4. Set up workflow dependencies
    git_ops = GitOps(orch_config.git, orch_config.project_root)
    scheduler = _EvalScheduler(orch_config)
    briefing = BriefingAssembler(orch_config)
    mcp = _EvalMcpStub(orch_config.fused_memory.url)

    # 5. Load fixed plan if available (skip architect phase)
    initial_plan = task.get('plan')
    if initial_plan:
        logger.info(f'Using fixed plan ({len(initial_plan.get("steps", []))} steps)')
    else:
        logger.info('No fixed plan — architect will generate one')

    # 6. Run the real workflow
    workflow = TaskWorkflow(
        assignment=assignment,
        config=orch_config,
        git_ops=git_ops,
        scheduler=scheduler,  # type: ignore[arg-type]
        briefing=briefing,
        mcp=mcp,  # type: ignore[arg-type]
        initial_plan=initial_plan,
    )

    # Override worktree since we created it ourselves
    workflow.worktree = worktree

    try:
        outcome = await workflow.run()
    except Exception as e:
        logger.error(f'Eval {task_id} × {config.name} failed: {e}')
        outcome = WorkflowOutcome.BLOCKED

    wall_clock_ms = int(time.monotonic() * 1000) - start_ms

    # 6. Collect metrics
    try:
        metrics = await collect_metrics(workflow, worktree, task)
        metrics_dict = metrics.to_dict()
    except Exception as e:
        logger.warning(f'Metric collection failed: {e}')
        metrics_dict = {}

    result = EvalResult(
        task_id=task_id,
        config_name=config.name,
        outcome=outcome.value if isinstance(outcome, WorkflowOutcome) else str(outcome),
        metrics=metrics_dict,
        worktree_path=str(worktree),
        wall_clock_ms=wall_clock_ms,
    )

    # 7. Persist result
    save_result(result)

    logger.info(
        f'Eval complete: {task_id} × {config.name} → {result.outcome} '
        f'({wall_clock_ms / 1000:.1f}s)'
    )
    return result


async def run_eval_matrix(
    task_paths: list[Path],
    configs: list[EvalConfig] | None = None,
    base_config: OrchestratorConfig | None = None,
) -> list[EvalResult]:
    """Run all (task, config) pairs. Sequential to avoid rate limits."""
    configs = configs or EVAL_CONFIGS
    results = []

    for task_path in task_paths:
        for config in configs:
            logger.info(f'Running eval: {task_path.stem} × {config.name}')
            result = await run_eval(task_path, config, base_config)
            results.append(result)

    return results


def save_result(result: EvalResult) -> Path:
    """Write eval result JSON to results directory."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    filename = f'{result.task_id}__{result.config_name}.json'
    path = RESULTS_DIR / filename
    with open(path, 'w') as f:
        json.dump(result.to_dict(), f, indent=2)
    logger.info(f'Saved result: {path}')
    return path


def load_results() -> list[EvalResult]:
    """Load all existing eval results from the results directory."""
    results = []
    if not RESULTS_DIR.exists():
        return results
    for path in sorted(RESULTS_DIR.glob('*.json')):
        with open(path) as f:
            data = json.load(f)
        results.append(EvalResult(**data))
    return results


# ---------------------------------------------------------------------------
# Stubs for eval mode (no real scheduler/MCP needed)
# ---------------------------------------------------------------------------

class _EvalScheduler:
    """Minimal scheduler stub for eval runs."""

    def __init__(self, config: OrchestratorConfig):
        self.config = config

    async def get_tasks(self):
        return []

    async def set_task_status(self, task_id: str, status: str):
        logger.info(f'[eval] Task {task_id} → {status}')

    async def handle_blast_radius_expansion(self, task_id, current, requested):
        return True  # always allow in eval mode


class _EvalMcpStub:
    """Minimal MCP lifecycle stub for eval runs."""

    def __init__(self, url: str):
        self.url = url

    def mcp_config_json(self, escalation_url: str | None = None) -> dict:
        return {}
