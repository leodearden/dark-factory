"""CLI entry point: `orchestrator run [--prd X]`."""

import asyncio
import json
import logging
import sys
from pathlib import Path

import click
from dotenv import load_dotenv

from orchestrator.config import ConfigRequiredError, load_config

load_dotenv()  # loads .env into os.environ (e.g. CLAUDE_OAUTH_TOKEN_A/B)

LOG_FORMAT = '%(asctime)s %(levelname)-8s [%(name)s] %(message)s'
DATE_FORMAT = '%Y-%m-%d %H:%M:%S'


@click.group()
@click.option('--verbose', is_flag=True, help='Enable debug logging')
def main(verbose: bool):
    """Dark Factory agent orchestrator."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format=LOG_FORMAT,
        datefmt=DATE_FORMAT,
        stream=sys.stderr,
    )


@main.command()
@click.option('--prd', type=click.Path(exists=True, path_type=Path), default=None,
              help='Path to PRD markdown file (omit to run existing tasks)')
@click.option('--config', 'config_path', type=click.Path(exists=True, path_type=Path),
              default=None,
              help='Path to orchestrator config YAML (REQUIRED unless ORCH_CONFIG_PATH '
                   'is set). Selects the target project — sets project_root and '
                   'fused_memory.project_id.')
@click.option('--dry-run', is_flag=True, help='Populate tasks only, do not execute')
@click.option('--delay', default=None,
              help='Delay before executing tasks (e.g. 4h, 30m, 90s). '
                   'Escalation server starts immediately.')
@click.option('--force-dirty-start', is_flag=True,
              help='Start even if project_root has uncommitted changes (risky)')
@click.option('--retag-modules', is_flag=True,
              help='Force re-tag all non-done/cancelled tasks with code modules')
def run(prd: Path | None, config_path: Path | None, dry_run: bool, delay: str | None,
        force_dirty_start: bool, retag_modules: bool):
    """Run the orchestrator against a PRD, or execute existing tasks if no PRD given."""
    from orchestrator.harness import Harness

    delay_secs = _parse_duration(delay) if delay else 0
    try:
        config = load_config(config_path)
    except ConfigRequiredError as e:
        click.echo(f'Error: {e}', err=True)
        sys.exit(1)
    harness = Harness(config)
    report = asyncio.run(harness.run(
        prd, dry_run=dry_run, delay_secs=delay_secs,
        force_dirty_start=force_dirty_start,
        retag_modules=retag_modules,
    ))

    click.echo(report.summary())

    if report.blocked > 0:
        sys.exit(1)


@main.command()
@click.option('--config', 'config_path', type=click.Path(exists=True, path_type=Path),
              default=None,
              help='Path to orchestrator config YAML (REQUIRED unless ORCH_CONFIG_PATH '
                   'is set). Selects the target project — sets project_root and '
                   'fused_memory.project_id.')
def status(config_path: Path | None):
    """Show current task tree and status."""
    from orchestrator.scheduler import Scheduler

    try:
        config = load_config(config_path)
    except ConfigRequiredError as e:
        click.echo(f'Error: {e}', err=True)
        sys.exit(1)
    scheduler = Scheduler(config)

    async def _show():
        tasks = await scheduler.get_tasks()
        if not tasks:
            click.echo('No tasks found.')
            return
        for t in tasks:
            tid = t.get('id', '?')
            title = t.get('title', 'Untitled')
            status = t.get('status', 'unknown')
            modules = t.get('metadata', {}).get('modules', [])
            mod_str = f' [{", ".join(modules)}]' if modules else ''
            click.echo(f'  [{status:12s}] {tid}: {title}{mod_str}')

    asyncio.run(_show())


@main.command('eval')
@click.option('--task', 'task_path', type=click.Path(exists=True, path_type=Path),
              default=None, help='Path to a single task JSON file')
@click.option('--config-name', default=None,
              help='Eval config name (e.g. claude-opus-high) or "all"')
@click.option('--matrix', is_flag=True, help='Run full eval matrix (all tasks × all configs)')
@click.option('--judge', is_flag=True, help='Run Elo-based LLM judge on existing results')
@click.option('--plan-only', is_flag=True, help='Generate plans for tasks (no execution)')
@click.option('--config', 'config_path', type=click.Path(exists=True, path_type=Path),
              default=None,
              help='Path to orchestrator config YAML (REQUIRED unless ORCH_CONFIG_PATH '
                   'is set). Selects the target project — sets project_root and '
                   'fused_memory.project_id.')
@click.option('--max-parallel', type=int, default=None,
              help='Max concurrent eval runs (default: unlimited)')
@click.option('--trials', type=int, default=1,
              help='Number of trials per (task, config) pair')
@click.option('--force', is_flag=True, help='Re-run even if results exist')
@click.option('--cleanup', is_flag=True, help='Remove eval worktrees')
@click.option('--timeout', type=int, default=None,
              help='Timeout in minutes per eval run (overrides task JSON)')
@click.option('--max-rounds', type=int, default=50,
              help='Max judge invocations per task (default: 50)')
@click.option('--reset', is_flag=True, help='Clear judge state and start fresh')
@click.option('--report', 'report_only', is_flag=True,
              help='Generate report from existing state (no new judge calls)')
@click.option('--vllm-url', default=None,
              help='vLLM endpoint URL (e.g. http://workstation:8000)')
def eval_cmd(
    task_path: Path | None,
    config_name: str | None,
    matrix: bool,
    judge: bool,
    plan_only: bool,
    config_path: Path | None,
    max_parallel: int | None,
    trials: int,
    force: bool,
    cleanup: bool,
    timeout: int | None,
    max_rounds: int,
    reset: bool,
    report_only: bool,
    vllm_url: str | None,
):
    """Run multi-provider implementor evaluations."""
    try:
        base_config = load_config(config_path)
    except ConfigRequiredError as e:
        click.echo(f'Error: {e}', err=True)
        sys.exit(1)

    # Inject ANTHROPIC_BASE_URL into vLLM configs when --vllm-url is set
    if vllm_url:
        from orchestrator.evals.configs import VLLM_EVAL_CONFIGS
        for cfg in VLLM_EVAL_CONFIGS:
            cfg.env_overrides['ANTHROPIC_BASE_URL'] = vllm_url

    if cleanup:
        _run_cleanup(base_config)
        return

    if report_only:
        _run_report_cmd()
        return

    if judge:
        _run_judge_cmd(max_rounds=max_rounds, reset=reset)
        return

    if plan_only:
        _run_plan_only(task_path, base_config)
        return

    if matrix:
        _run_matrix_cmd(
            base_config,
            max_parallel=max_parallel, trials=trials,
            force=force, timeout=timeout,
        )
        return

    if task_path is None:
        click.echo('Error: --task is required (or use --matrix / --judge / --plan-only)', err=True)
        sys.exit(1)

    _run_single_eval(task_path, config_name, base_config, force=force, timeout=timeout)


def _run_single_eval(
    task_path: Path, config_name: str | None, base_config,
    force: bool = False, timeout: int | None = None,
):
    """Run eval for a single task with one or all configs."""
    from orchestrator.evals.configs import EVAL_CONFIGS, get_config_by_name
    from orchestrator.evals.runner import run_eval

    all_configs = EVAL_CONFIGS

    if config_name and config_name != 'all':
        cfg = get_config_by_name(config_name)
        if not cfg:
            click.echo(f'Unknown config: {config_name}', err=True)
            click.echo(f'Available: {", ".join(c.name for c in all_configs)}', err=True)
            sys.exit(1)
        configs = [cfg]
    else:
        configs = all_configs

    async def _run():
        for cfg in configs:
            result = await run_eval(
                task_path, cfg, base_config, timeout_override=timeout,
            )
            click.echo(
                f'{result.task_id} × {result.config_name}: '
                f'{result.outcome} ({result.wall_clock_ms / 1000:.1f}s)'
            )

    asyncio.run(_run())


def _run_matrix_cmd(
    base_config,
    max_parallel: int | None = None,
    trials: int = 1,
    force: bool = False,
    timeout: int | None = None,
):
    """Run full eval matrix."""
    from orchestrator.evals.configs import EVAL_CONFIGS
    from orchestrator.evals.runner import run_eval_matrix

    all_configs = EVAL_CONFIGS

    tasks_dir = Path(__file__).parent / 'evals' / 'tasks'
    if not tasks_dir.exists():
        # Try relative to project root
        tasks_dir = base_config.project_root / 'orchestrator' / 'evals' / 'tasks'

    task_paths = sorted(tasks_dir.glob('*.json'))
    if not task_paths:
        click.echo('No task files found in evals/tasks/', err=True)
        sys.exit(1)

    total = len(task_paths) * len(all_configs) * trials
    click.echo(
        f'Running eval matrix: {len(task_paths)} tasks × {len(all_configs)} configs'
        f' × {trials} trials = {total} runs'
        f' (max_parallel={max_parallel or "unlimited"})'
    )

    async def _run():
        results = await run_eval_matrix(
            task_paths, all_configs, base_config,
            max_parallel=max_parallel,
            trials=trials,
            force=force,
            timeout_override=timeout,
        )
        click.echo(f'\nCompleted {len(results)} eval runs:')
        for r in results:
            score = r.metrics.get('composite_score', 0)
            click.echo(
                f'  {r.task_id} × {r.config_name}: '
                f'{r.outcome} (score={score:.2f}, {r.wall_clock_ms / 1000:.1f}s)'
            )

    asyncio.run(_run())


def _run_judge_cmd(max_rounds: int = 50, reset: bool = False):
    """Run Elo-based judge on existing results."""
    from orchestrator.evals.elo import JudgeState, TaskPool, load_state, save_state
    from orchestrator.evals.judge import run_elo_tournament
    from orchestrator.evals.report import build_report, format_markdown, save_report
    from orchestrator.evals.runner import load_results, load_task
    from orchestrator.evals.snapshots import get_diff_between_commits

    # Load or reset state
    if reset:
        state = JudgeState()
        click.echo('Judge state reset.')
    else:
        state = load_state()
        if state.per_task:
            click.echo(f'Resuming from existing state ({len(state.per_task)} tasks)')

    results = load_results()
    if not results:
        click.echo('No existing results found in evals/results/', err=True)
        sys.exit(1)

    # Group by task, filter to passing with existing worktrees
    by_task: dict[str, list] = {}
    for r in results:
        by_task.setdefault(r.task_id, []).append(r)

    passing: dict[str, list[dict]] = {}
    for task_id, task_results in by_task.items():
        p = [r.to_dict() for r in task_results
             if r.metrics.get('tests_pass', False)
             and Path(r.worktree_path).exists()]
        if p:
            passing[task_id] = p
            click.echo(f'  {task_id}: {len(p)} contenders with worktrees')

    if not passing:
        click.echo('No passing results with existing worktrees found', err=True)
        sys.exit(1)

    tasks_dir = Path(__file__).parent / 'evals' / 'tasks'

    async def _run():
        for task_id, result_dicts in passing.items():
            task_file = tasks_dir / f'{task_id}.json'
            if not task_file.exists():
                click.echo(f'Skipping {task_id}: task file not found')
                continue

            task = load_task(task_file)

            # Add reference implementation if post_task_commit exists
            pre = task.get('pre_task_commit')
            post = task.get('post_task_commit')
            if pre and post:
                try:
                    project_root = Path(task['project_root'])
                    ref_diff = await get_diff_between_commits(project_root, pre, post)
                    if ref_diff.strip():
                        result_dicts.append({
                            'config_name': 'reference',
                            'diff': ref_diff,
                            'worktree_path': '',
                        })
                        click.echo(f'  {task_id}: added reference implementation')
                except Exception as e:
                    click.echo(f'  {task_id}: could not compute reference diff: {e}')

            if len(result_dicts) < 2:
                click.echo(f'Skipping {task_id}: need at least 2 contenders')
                continue

            # Get or create task pool
            if task_id not in state.per_task:
                state.per_task[task_id] = TaskPool()
            pool = state.per_task[task_id]

            click.echo(
                f'\nJudging {task_id} '
                f'({len(result_dicts)} contenders, max {max_rounds} rounds)...'
            )
            rounds_used = await run_elo_tournament(
                result_dicts, task, pool, max_rounds,
            )
            click.echo(f'  {task_id}: {rounds_used} judge calls')

            # Save state after each task (crash resilience)
            save_state(state)

        # Generate and print report
        report = build_report(state)
        report_path = save_report(report)
        click.echo(f'\nReport saved to {report_path}')
        click.echo('\n' + format_markdown(report))

    asyncio.run(_run())


def _run_report_cmd():
    """Generate report from existing judge state (no new judge calls)."""
    from orchestrator.evals.elo import load_state
    from orchestrator.evals.report import build_report, format_markdown, save_report

    state = load_state()
    if not state.per_task:
        click.echo('No judge state found. Run --judge first.', err=True)
        sys.exit(1)

    report = build_report(state)
    report_path = save_report(report)
    click.echo(f'Report saved to {report_path}')
    click.echo('\n' + format_markdown(report))


def _run_cleanup(base_config):
    """Remove all eval worktrees."""
    from orchestrator.evals.snapshots import cleanup_eval_worktree

    worktree_root = base_config.project_root / '.eval-worktrees'
    if not worktree_root.exists():
        click.echo('No eval worktrees found.')
        return

    async def _cleanup():
        count = 0
        for task_dir in sorted(worktree_root.iterdir()):
            if not task_dir.is_dir():
                continue
            for run_dir in sorted(task_dir.iterdir()):
                if run_dir.is_dir():
                    await cleanup_eval_worktree(base_config.project_root, run_dir)
                    count += 1
        click.echo(f'Cleaned up {count} eval worktrees.')

    asyncio.run(_cleanup())


def _run_plan_only(task_path: Path | None, base_config):
    """Generate plans for eval tasks using the architect (opus-high).

    Runs the architect against the pre-task commit for each task,
    saves the resulting plan into the task JSON file.
    """
    from orchestrator.evals.runner import load_task
    from orchestrator.evals.snapshots import cleanup_eval_worktree, create_eval_worktree

    tasks_dir = Path(__file__).parent / 'evals' / 'tasks'

    task_paths = [task_path] if task_path else sorted(tasks_dir.glob('*.json'))

    if not task_paths:
        click.echo('No task files found', err=True)
        sys.exit(1)

    async def _run():
        from orchestrator.agents.briefing import BriefingAssembler
        from orchestrator.agents.invoke import invoke_agent
        from orchestrator.agents.roles import ARCHITECT
        from orchestrator.artifacts import TaskArtifacts

        briefing = BriefingAssembler(base_config)

        for tp in task_paths:
            task = load_task(tp)
            task_id = task['id']

            if task.get('plan'):
                click.echo(f'  {task_id}: already has plan ({len(task["plan"].get("steps", []))} steps), skipping')
                continue

            click.echo(f'  {task_id}: generating plan...')
            project_root = Path(task['project_root'])

            # Create worktree at pre-task commit
            worktree, _run_id = await create_eval_worktree(
                project_root, task_id, task['pre_task_commit'],
                setup_commands=task.get('setup_commands'),
            )

            try:
                # Init artifacts so architect has a place to write
                artifacts = TaskArtifacts(worktree)
                artifacts.init(
                    task_id,
                    task.get('task_definition', {}).get('title', ''),
                    task.get('task_definition', {}).get('description', ''),
                    base_commit=task['pre_task_commit'],
                )

                # Build architect prompt
                task_def = task.get('task_definition', {})
                prompt = await briefing.build_architect_prompt(task_def, worktree=worktree)

                # Invoke architect (opus-high, always Claude)
                result = await invoke_agent(
                    prompt=prompt,
                    system_prompt=ARCHITECT.system_prompt,
                    cwd=worktree,
                    model='opus',
                    max_turns=50,
                    max_budget_usd=5.0,
                    allowed_tools=ARCHITECT.allowed_tools or None,
                    disallowed_tools=ARCHITECT.disallowed_tools or None,
                    effort='high',
                    backend='claude',
                )

                if not result.success:
                    click.echo(f'    FAILED: {result.output[:200]}', err=True)
                    continue

                # Read the plan the architect wrote
                plan = artifacts.read_plan()
                if not plan:
                    click.echo('    FAILED: architect produced no plan.json', err=True)
                    continue

                # Save plan into task JSON
                task['plan'] = plan
                with open(tp, 'w') as f:
                    json.dump(task, f, indent=2)
                    f.write('\n')

                step_count = len(plan.get('steps', []))
                click.echo(
                    f'    OK: {step_count} steps, '
                    f'cost=${result.cost_usd:.2f}'
                )

            finally:
                await cleanup_eval_worktree(project_root, worktree)

    asyncio.run(_run())


def _parse_duration(s: str) -> int:
    """Parse a duration string like '4h', '30m', '90s', or bare seconds."""
    s = s.strip().lower()
    if s.endswith('h'):
        return int(s[:-1]) * 3600
    if s.endswith('m'):
        return int(s[:-1]) * 60
    if s.endswith('s'):
        return int(s[:-1])
    return int(s)


if __name__ == '__main__':
    main()
