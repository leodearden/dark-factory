"""Main eval orchestrator: run (task, config) pairs through the real workflow."""

from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from shared.usage_gate import UsageGate

from orchestrator.agents.briefing import BriefingAssembler
from orchestrator.config import (
    BackendsConfig,
    BudgetsConfig,
    EffortConfig,
    ModelsConfig,
    OrchestratorConfig,
    SandboxConfig,
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
    run_id: str = ''
    trial: int = 1

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _find_repo_root(start: Path) -> Path:
    """Walk up from *start* to find the directory containing ``.git``."""
    current = start.resolve().parent if start.is_file() else start.resolve()
    while current != current.parent:
        if (current / '.git').exists():
            return current
        current = current.parent
    return start.resolve().parent


def load_task(task_path: Path) -> dict:
    """Load a task definition JSON file.

    Resolves ``project_root`` at runtime so task files are portable across
    machines.  If the path in the JSON starts with ``$REPO_ROOT`` it is
    expanded; if the hardcoded absolute path does not exist the discovered
    repository root is used instead.
    """
    with open(task_path) as f:
        task = json.load(f)

    repo_root = _find_repo_root(task_path)
    raw_root = task.get('project_root', '')

    if raw_root.startswith('$REPO_ROOT'):
        suffix = raw_root.replace('$REPO_ROOT/', '').replace('$REPO_ROOT', '')
        task['project_root'] = str(repo_root / suffix)
    elif raw_root and not Path(raw_root).exists():
        task['project_root'] = str(repo_root)

    return task


def build_eval_orch_config(
    config: EvalConfig,
    task: dict,
    base_config: OrchestratorConfig | None = None,
) -> OrchestratorConfig:
    """Build an OrchestratorConfig override for this eval run.

    Architect always runs on Claude opus-high (constant planning).
    Reviewer is the new 1× Opus comprehensive reviewer (matches production
    after the reviewer-panel trial replaced the 5× sonnet panel; merged
    via 594658fbe3 / 2c26a30bca).
    Only the implementer varies per eval config.

    Two task-spec knobs override the defaults below:
      - ``max_execute_iterations``: hard ceiling on implementer iterations.
        Eval default is 20 (was 10) so workstation-tier slow models aren't
        capped before they finish — bumping was confirmed by user 2026-04-08.
      - ``max_review_cycles``: how many re-plan/re-review cycles after
        blocking issues. Eval default is 1; df_task_18 sets 2 because it
        empirically needs a second architect→implement→debug→review pass
        to clear all blockers.
    """
    if base_config is None:
        raise ValueError('build_eval_orch_config requires an explicit base_config')
    base = base_config

    models = ModelsConfig(
        architect='opus',
        implementer=config.model,
        debugger=config.model,
        reviewer='opus',          # 1× opus comprehensive reviewer (production parity)
        merger='opus',
        module_tagger='sonnet',
        judge='sonnet',           # ζ completion judge — read-only, small budget
    )

    budgets = BudgetsConfig(
        architect=5.0,
        implementer=config.max_budget_usd,
        debugger=config.max_budget_usd / 2,
        reviewer=5.0,             # opus reviewer needs more headroom than sonnet
        merger=5.0,
        module_tagger=2.0,
        judge=0.50,
    )

    effort = EffortConfig(
        architect='high',
        implementer=config.effort or 'high',
        debugger=config.effort or 'high',
        reviewer='high',           # opus reviewer at high effort (matches defaults.yaml)
        merger='high',
        module_tagger='medium',
        judge='medium',
    )

    backends = BackendsConfig(
        architect='claude',
        implementer=config.backend,
        debugger=config.backend,
        reviewer='claude',        # reviewers always on Claude
        merger='claude',
        module_tagger='claude',
        judge='claude',           # judge always on Claude (read-only quality call)
    )

    return OrchestratorConfig(
        models=models,
        budgets=budgets,
        effort=effort,
        backends=backends,
        max_turns=base.max_turns,
        max_execute_iterations=task.get('max_execute_iterations', 20),
        max_verify_attempts=base.max_verify_attempts,
        max_review_cycles=task.get('max_review_cycles', 1),
        judge_after_each_iteration=task.get('judge_after_each_iteration', True),
        test_command=task.get('verify_commands', {}).get('test', base.test_command),
        lint_command=task.get('verify_commands', {}).get('lint', base.lint_command),
        type_check_command=task.get('verify_commands', {}).get('typecheck', base.type_check_command),
        fused_memory=base.fused_memory,
        sandbox=SandboxConfig(enabled=False),
        escalation=base.escalation,
        git=base.git,
        usage_cap=base.usage_cap,
        project_root=Path(task.get('project_root', str(base.project_root))),
        env_overrides=config.env_overrides,
    )


async def run_eval(
    task_path: Path,
    config: EvalConfig,
    base_config: OrchestratorConfig | None = None,
    trial: int = 1,
    timeout_override: int | None = None,
) -> EvalResult:
    """Run one (task, config) pair through PLAN→EXECUTE→VERIFY→REVIEW."""
    task = load_task(task_path)
    task_id = task['id']
    project_root = Path(task['project_root'])

    logger.info(f'Starting eval: {task_id} × {config.name} (trial {trial})')
    start_ms = int(time.monotonic() * 1000)

    # 1. Create isolated worktree at pre-task commit (with env setup)
    worktree, run_id = await create_eval_worktree(
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

    # 5. Load fixed plan (required for eval mode)
    initial_plan = task.get('plan')
    if not initial_plan:
        raise ValueError(
            f'Task {task_id} has no embedded plan. '
            f'Run --plan-only to generate one first.'
        )
    logger.info(f'Using fixed plan ({len(initial_plan.get("steps", []))} steps)')

    # 5b. Usage gate for account failover (judge hits Claude API, may cap)
    usage_gate: UsageGate | None = None
    if orch_config.usage_cap.enabled:
        try:
            usage_gate = UsageGate(orch_config.usage_cap)
        except Exception as exc:
            logger.warning(f'Failed to create UsageGate for eval: {exc} — running without failover')

    # 6. Run the real workflow
    workflow = TaskWorkflow(
        assignment=assignment,
        config=orch_config,
        git_ops=git_ops,
        scheduler=scheduler,  # type: ignore[arg-type]
        briefing=briefing,
        mcp=mcp,  # type: ignore[arg-type]
        initial_plan=initial_plan,
        usage_gate=usage_gate,
    )

    # Override worktree since we created it ourselves
    workflow.worktree = worktree

    timeout_minutes = timeout_override or task.get('timeout_minutes', 60)
    try:
        outcome = await asyncio.wait_for(
            workflow.run(), timeout=timeout_minutes * 60,
        )
    except TimeoutError:
        logger.error(
            f'Eval {task_id} × {config.name} timed out after {timeout_minutes}m'
        )
        outcome = 'timeout'
    except Exception as e:
        logger.error(f'Eval {task_id} × {config.name} failed: {e}')
        outcome = WorkflowOutcome.BLOCKED

    wall_clock_ms = int(time.monotonic() * 1000) - start_ms

    # 7. Collect metrics
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
        run_id=run_id,
        trial=trial,
    )

    # 8. Persist result
    save_result(result)

    logger.info(
        f'Eval complete: {task_id} × {config.name} → {result.outcome} '
        f'(total={wall_clock_ms / 1000:.1f}s, '
        f'workflow={metrics_dict.get("workflow_duration_ms", 0) / 1000:.1f}s)'
    )
    return result


async def run_eval_matrix(
    task_paths: list[Path],
    configs: list[EvalConfig] | None = None,
    base_config: OrchestratorConfig | None = None,
    max_parallel: int | None = None,
    trials: int = 1,
    force: bool = False,
    timeout_override: int | None = None,
) -> list[EvalResult]:
    """Run all (task, config, trial) combinations with bounded concurrency."""
    configs = configs or EVAL_CONFIGS

    combos = [
        (task_path, config, t)
        for task_path in task_paths
        for config in configs
        for t in range(1, trials + 1)
    ]

    if max_parallel is None:
        max_parallel = len(combos)
    sem = asyncio.Semaphore(max_parallel)

    async def _run_one(
        task_path: Path, config: EvalConfig, trial: int,
    ) -> EvalResult | None:
        task = load_task(task_path)
        if not force and _result_exists(task['id'], config.name):
            logger.info(f'Skipping existing: {task["id"]} × {config.name}')
            return None
        async with sem:
            return await run_eval(
                task_path, config, base_config,
                trial=trial, timeout_override=timeout_override,
            )

    raw = await asyncio.gather(
        *[_run_one(tp, cfg, t) for tp, cfg, t in combos],
        return_exceptions=True,
    )

    results: list[EvalResult] = []
    for r in raw:
        if isinstance(r, asyncio.CancelledError):
            logger.error(f'Eval cancelled: {r}')
            raise r
        elif isinstance(r, BaseException):
            logger.error(f'Eval failed: {r}')
        elif r is not None:
            results.append(r)
    return results


def _result_exists(task_id: str, config_name: str) -> bool:
    """Check if any result already exists for this (task, config) pair."""
    if not RESULTS_DIR.exists():
        return False
    return any(RESULTS_DIR.glob(f'{task_id}__{config_name}__*.json'))


def save_result(result: EvalResult) -> Path:
    """Write eval result JSON to results directory."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    filename = f'{result.task_id}__{result.config_name}__{result.run_id}.json'
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
        # Filter to known fields so old results with extra/missing keys load
        known = {f.name for f in EvalResult.__dataclass_fields__.values()}
        results.append(EvalResult(**{k: v for k, v in data.items() if k in known}))
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

    async def handle_blast_radius_expansion(self, task_id: str, current: list[str], needed: list[str]) -> bool:
        return True  # always allow in eval mode


class _EvalMcpStub:
    """Minimal MCP lifecycle stub for eval runs."""

    def __init__(self, url: str):
        self.url = url

    def mcp_config_json(self, escalation_url: str | None = None) -> dict:
        return {}


if __name__ == '__main__':
    from orchestrator.cli import eval_cmd
    eval_cmd()
