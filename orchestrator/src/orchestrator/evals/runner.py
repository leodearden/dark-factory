"""Main eval orchestrator: run (task, config) pairs through the real workflow."""

from __future__ import annotations

import asyncio
import json
import logging
import time
from collections.abc import Iterable
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
from orchestrator.scheduler import Scheduler, TaskAssignment
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
    worktree_path: Path | None = None,
) -> EvalResult:
    """Run one (task, config) pair through PLAN→EXECUTE→VERIFY→REVIEW.

    When *worktree_path* is provided, the eval reuses an existing worktree
    instead of creating a fresh one.  The worktree's ``.task/plan.json``
    (with step statuses) is used as the initial plan, so the workflow
    naturally skips already-completed steps — useful for resuming a
    blocked eval from the reviewer phase.
    """
    task = load_task(task_path)
    task_id = task['id']
    project_root = Path(task['project_root'])

    logger.info(f'Starting eval: {task_id} × {config.name} (trial {trial})')
    start_ms = int(time.monotonic() * 1000)

    # 1. Create or reuse worktree
    if worktree_path is not None:
        worktree = worktree_path
        run_id = worktree.name
        logger.info(f'Reusing existing worktree: {worktree}')
    else:
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
    scheduler, _ = _build_eval_scheduler(orch_config, task_id, list(modules))
    briefing = BriefingAssembler(orch_config)
    mcp = _EvalMcpStub(orch_config.fused_memory.url)

    # 5. Load plan — from existing worktree state or task JSON
    if worktree_path is not None:
        import json as _json
        existing_plan_path = worktree / '.task' / 'plan.json'
        if existing_plan_path.exists():
            initial_plan = _json.loads(existing_plan_path.read_text())
            done = sum(1 for s in initial_plan.get('steps', []) if s.get('status') == 'done')
            logger.info(
                f'Using existing plan from worktree '
                f'({done}/{len(initial_plan.get("steps", []))} steps done)'
            )
        else:
            initial_plan = task.get('plan')
            logger.info('No existing plan in worktree — using task JSON plan')
    else:
        initial_plan = task.get('plan')
    if not initial_plan:
        raise ValueError(
            f'Task {task_id} has no embedded plan. '
            f'Run --plan-only to generate one first.'
        )
    if not worktree_path:
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


def _collect_cancel_errors(done: Iterable[asyncio.Task[Any]]) -> list[asyncio.CancelledError]:
    """Return all CancelledErrors from a completed asyncio.wait done-set.

    Iterates every task in *done* and collects cancellation errors so that
    callers can log every failure before raising, rather than discarding all
    but the first (Task 586 fix).

    The primary branch — ``task.cancelled() is True`` — covers all known
    CPython 3.11+ cases, including coroutines that raise CancelledError
    internally without an explicit ``task.cancel()`` call, because the
    runtime transitions the task to the cancelled state in both scenarios.

    Belt-and-suspenders: in current CPython a coroutine raising
    CancelledError causes task.cancelled() to return True, so the
    secondary branch (task.exception() returning CancelledError while
    task.cancelled() is False) is unreachable in practice. Kept in case
    a future runtime routes coroutine-raised CancelledError via
    task.exception() instead of task.cancelled().
    """
    errors: list[asyncio.CancelledError] = []
    for task in done:
        if task.cancelled():
            errors.append(asyncio.CancelledError())
        else:
            exc = task.exception()
            if isinstance(exc, asyncio.CancelledError):
                errors.append(exc)
    return errors


async def run_eval_matrix(
    task_paths: list[Path],
    configs: list[EvalConfig] | None = None,
    base_config: OrchestratorConfig | None = None,
    max_parallel: int | None = None,
    trials: int = 1,
    force: bool = False,
    timeout_override: int | None = None,
) -> list[EvalResult]:
    """Run all (task, config, trial) combinations with bounded concurrency.

    Raises:
        asyncio.CancelledError: if any individual eval coroutine raises
            CancelledError, or if this coroutine itself is cancelled from
            outside.  In either case we log, cancel any still-running
            sibling tasks, await their cleanup, and re-raise.
    """
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

    # Design decision: use asyncio.wait(FIRST_COMPLETED) monitor loop instead of
    # asyncio.gather(return_exceptions=True).
    #
    # asyncio.gather(return_exceptions=True) blocks until ALL tasks complete before
    # the post-gather loop can detect CancelledError and re-raise it.  For a large
    # matrix where one eval is cancelled early, N-1 siblings continue running their
    # full duration — wasting CPU proportional to matrix size × timeout_minutes.
    #
    # asyncio.wait(FIRST_COMPLETED) lets us react to each task completion
    # individually: on CancelledError we immediately cancel all remaining tasks and
    # re-raise, typically within milliseconds.  Non-cancel exceptions are still
    # logged and the loop continues — identical happy-path/error-path semantics to
    # the previous gather loop, with strictly better cancellation behaviour.
    #
    # This is the same pattern used in harness.py (lines 305, 317) for managing
    # concurrent workflow tasks.  Cleanup follows the established pattern from
    # steward.py (lines 101-104): cancel tasks explicitly then await them with
    # return_exceptions=True to ensure clean teardown before re-raising.
    active: set[asyncio.Task] = {
        asyncio.create_task(_run_one(tp, cfg, t))
        for tp, cfg, t in combos
    }
    results: list[EvalResult] = []
    # Distinguish two cancellation scenarios:
    #   Inner-task cancellation — an individual _run_one coroutine was cancelled
    #     or raised CancelledError.  asyncio.wait surfaces this via
    #     task.cancelled() or task.exception() inside the monitor loop below;
    #     we log it, cancel siblings, and re-raise to propagate.
    #   Outer-task cancellation — run_eval_matrix itself was cancelled (e.g.
    #     SIGINT / asyncio.wait_for timeout).  The CancelledError interrupts
    #     the *await asyncio.wait(...)* call directly and is caught by the
    #     outer except clause, which performs the same sibling cleanup.
    try:
        while active:
            done, active = await asyncio.wait(active, return_when=asyncio.FIRST_COMPLETED)
            # Task 586: scan the full done batch for ALL CancelledErrors before
            # processing any results.  Multiple tasks can complete in the same
            # event-loop iteration and land in the same done set (e.g. when a
            # shutdown signal fires while two evals are parked at the same
            # await point).  The old code raised on the first cancel it saw,
            # silently discarding subsequent cancels in the batch.
            cancel_errors = _collect_cancel_errors(done)
            if cancel_errors:
                for ce in cancel_errors:
                    logger.error('Eval cancelled', exc_info=ce)
                for t in active:
                    t.cancel()
                await asyncio.gather(*active, return_exceptions=True)
                active.clear()
                raise cancel_errors[0]
            # No cancellations in this batch — handle results and non-cancel
            # exceptions.  task.cancelled() is False for all remaining tasks so
            # task.exception() / task.result() are safe to call.
            for task in done:
                exc = task.exception()
                if exc is not None:
                    logger.error('Eval failed', exc_info=exc)
                else:
                    r = task.result()
                    if r is not None:
                        results.append(r)
    except asyncio.CancelledError:
        # External cancellation (e.g. SIGINT / asyncio.wait_for timeout).
        # Cancel all remaining sibling tasks and await their cleanup before
        # re-raising so we don't leave orphaned tasks behind.
        for t in active:
            t.cancel()
        await asyncio.gather(*active, return_exceptions=True)
        raise
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
# Stubs and helpers for eval mode (no real MCP HTTP connection needed)
# ---------------------------------------------------------------------------


def _build_eval_scheduler(
    orch_config: OrchestratorConfig,
    task_id: str,
    modules: list[str],
) -> tuple[Scheduler, _StubMcpSession]:
    """Build a production Scheduler wired with an in-memory MCP session stub.

    Pre-installs the module lock for ``task_id`` so that a later
    ``handle_blast_radius_expansion`` call cannot ``KeyError`` (production
    normally installs the lock in ``acquire_next``; eval mode bypasses that).

    Returns ``(scheduler, stub_session)`` so callers can inspect the stub when
    needed (e.g. tests asserting on ``_statuses``).
    """
    stub = _StubMcpSession()
    scheduler = Scheduler(orch_config, mcp_session=stub)
    # Pre-install the module lock so handle_blast_radius_expansion's
    # try_acquire_additional can find the _held[task_id] entry.
    scheduler.lock_table.try_acquire(task_id, modules)
    return scheduler, stub


class _StubMcpSession:
    """In-process MCP session stub for eval runs.

    Mirrors ``McpSession.call_tool``'s signature and JSON-RPC envelope shape
    so the production ``Scheduler`` can use it via duck-typing without any
    changes to its parsing code.

    Currently handles: ``set_task_status``.
    Other tool names raise ``NotImplementedError`` (filled in by later steps).
    """

    def __init__(self) -> None:
        self._statuses: dict[str, str] = {}
        self._request_id: int = 0

    def _next_id(self) -> int:
        self._request_id += 1
        return self._request_id

    def _envelope(self, text: str) -> dict:
        return {
            'jsonrpc': '2.0',
            'id': self._next_id(),
            'result': {
                'content': [
                    {'type': 'text', 'text': text},
                ],
            },
        }

    async def call_tool(
        self,
        name: str,
        arguments: dict,
        timeout: float = 30,
    ) -> dict:
        """Dispatch an in-memory MCP tool call and return a JSON-RPC envelope.

        Supported tools: ``set_task_status``, ``get_task``, ``get_tasks``,
        ``update_task``.  Unknown tool names raise ``NotImplementedError``.

        .. note::
            Terminal-state enforcement is intentionally **not** simulated.
            The production fused-memory ``TaskInterceptor`` rejects transitions
            from terminal states (e.g. ``done`` → ``pending``) unless a
            ``reopen_reason`` is supplied.  This stub silently accepts any
            transition so eval flows are not blocked by status-guard logic.
            If a test needs to verify terminal-state semantics it should target
            the real fused-memory server rather than this stub.
        """
        if name == 'set_task_status':
            task_id = arguments['id']
            status = arguments['status']
            self._statuses[task_id] = status
            return self._envelope(json.dumps({'id': task_id, 'status': status}))
        if name == 'get_task':
            task_id = arguments['id']
            status = self._statuses.get(task_id)
            payload = {'id': task_id, 'status': status} if status is not None else {'id': task_id}
            return self._envelope(json.dumps(payload))
        if name == 'get_tasks':
            return self._envelope(json.dumps({'tasks': []}))
        if name == 'update_task':
            task_id = arguments['id']
            return self._envelope(json.dumps({'id': task_id}))
        raise NotImplementedError(
            f'_StubMcpSession: unknown tool {name!r} — add a branch if this tool is needed'
        )


class _EvalMcpStub:
    """Minimal MCP lifecycle stub for eval runs."""

    def __init__(self, url: str):
        self.url = url

    def mcp_config_json(self, escalation_url: str | None = None) -> dict:
        return {}


if __name__ == '__main__':
    from orchestrator.cli import eval_cmd
    eval_cmd()
