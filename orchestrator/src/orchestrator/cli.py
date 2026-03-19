"""CLI entry point: `orchestrator run --prd X`."""

import asyncio
import json
import logging
import sys
from pathlib import Path

import click

from orchestrator.config import load_config

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
@click.option('--prd', type=click.Path(exists=True, path_type=Path), required=True,
              help='Path to PRD markdown file')
@click.option('--config', 'config_path', type=click.Path(exists=True, path_type=Path),
              default=None, help='Path to config YAML')
@click.option('--dry-run', is_flag=True, help='Populate tasks only, do not execute')
def run(prd: Path, config_path: Path | None, dry_run: bool):
    """Run the orchestrator against a PRD."""
    from orchestrator.harness import Harness

    config = load_config(config_path)
    harness = Harness(config)
    report = asyncio.run(harness.run(prd, dry_run=dry_run))

    click.echo(report.summary())

    if report.blocked > 0:
        sys.exit(1)


@main.command()
@click.option('--config', 'config_path', type=click.Path(exists=True, path_type=Path),
              default=None, help='Path to config YAML')
def status(config_path: Path | None):
    """Show current task tree and status."""
    from orchestrator.scheduler import Scheduler

    config = load_config(config_path)
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
@click.option('--judge', is_flag=True, help='Run LLM judge on existing results')
@click.option('--plan-only', is_flag=True, help='Generate plans for tasks (no execution)')
@click.option('--config', 'config_path', type=click.Path(exists=True, path_type=Path),
              default=None, help='Path to orchestrator config YAML')
def eval_cmd(
    task_path: Path | None,
    config_name: str | None,
    matrix: bool,
    judge: bool,
    plan_only: bool,
    config_path: Path | None,
):
    """Run multi-provider implementor evaluations."""
    base_config = load_config(config_path)

    if judge:
        _run_judge_cmd()
        return

    if plan_only:
        _run_plan_only(task_path, base_config)
        return

    if matrix:
        _run_matrix_cmd(base_config)
        return

    if task_path is None:
        click.echo('Error: --task is required (or use --matrix / --judge / --plan-only)', err=True)
        sys.exit(1)

    _run_single_eval(task_path, config_name, base_config)


def _run_single_eval(task_path: Path, config_name: str | None, base_config):
    """Run eval for a single task with one or all configs."""
    from orchestrator.evals.configs import EVAL_CONFIGS, get_config_by_name
    from orchestrator.evals.runner import run_eval

    if config_name and config_name != 'all':
        cfg = get_config_by_name(config_name)
        if not cfg:
            click.echo(f'Unknown config: {config_name}', err=True)
            click.echo(f'Available: {", ".join(c.name for c in EVAL_CONFIGS)}', err=True)
            sys.exit(1)
        configs = [cfg]
    else:
        configs = EVAL_CONFIGS

    async def _run():
        for cfg in configs:
            result = await run_eval(task_path, cfg, base_config)
            click.echo(
                f'{result.task_id} × {result.config_name}: '
                f'{result.outcome} ({result.wall_clock_ms / 1000:.1f}s)'
            )

    asyncio.run(_run())


def _run_matrix_cmd(base_config):
    """Run full eval matrix."""
    from orchestrator.evals.configs import EVAL_CONFIGS
    from orchestrator.evals.runner import run_eval_matrix

    tasks_dir = Path(__file__).parent / 'evals' / 'tasks'
    if not tasks_dir.exists():
        # Try relative to project root
        tasks_dir = base_config.project_root / 'orchestrator' / 'evals' / 'tasks'

    task_paths = sorted(tasks_dir.glob('*.json'))
    if not task_paths:
        click.echo('No task files found in evals/tasks/', err=True)
        sys.exit(1)

    click.echo(f'Running eval matrix: {len(task_paths)} tasks × {len(EVAL_CONFIGS)} configs')

    async def _run():
        results = await run_eval_matrix(task_paths, EVAL_CONFIGS, base_config)
        click.echo(f'\nCompleted {len(results)} eval runs:')
        for r in results:
            score = r.metrics.get('composite_score', 0)
            click.echo(
                f'  {r.task_id} × {r.config_name}: '
                f'{r.outcome} (score={score:.2f}, {r.wall_clock_ms / 1000:.1f}s)'
            )

    asyncio.run(_run())


def _run_judge_cmd():
    """Run LLM judge on existing results."""
    from orchestrator.evals.runner import load_results

    results = load_results()
    if not results:
        click.echo('No existing results found in evals/results/', err=True)
        sys.exit(1)

    # Group by task
    by_task: dict[str, list] = {}
    for r in results:
        by_task.setdefault(r.task_id, []).append(r)

    # Filter to passing results only
    passing: dict[str, list] = {}
    for task_id, task_results in by_task.items():
        p = [r for r in task_results if r.metrics.get('tests_pass', False)]
        if len(p) >= 2:
            passing[task_id] = p

    if not passing:
        click.echo('Need at least 2 passing results per task for judge comparison', err=True)
        sys.exit(1)

    from orchestrator.evals.judge import run_tournament
    from orchestrator.evals.runner import load_task

    async def _run():
        for task_id, task_results in passing.items():
            # Load task definition
            tasks_dir = Path(__file__).parent / 'evals' / 'tasks'
            task_file = tasks_dir / f'{task_id}.json'
            if not task_file.exists():
                click.echo(f'Skipping {task_id}: task file not found')
                continue

            task = load_task(task_file)
            result_dicts = [r.to_dict() for r in task_results]

            click.echo(f'\nJudging {task_id} ({len(result_dicts)} contenders)...')
            verdicts = await run_tournament(result_dicts, task)

            # Tally wins
            wins: dict[str, int] = {}
            for v in verdicts:
                if v.winner == 'A':
                    wins[v.config_a] = wins.get(v.config_a, 0) + 1
                elif v.winner == 'B':
                    wins[v.config_b] = wins.get(v.config_b, 0) + 1

            click.echo(f'  Results for {task_id}:')
            for cfg, w in sorted(wins.items(), key=lambda x: -x[1]):
                click.echo(f'    {cfg}: {w} wins')

    asyncio.run(_run())


def _run_plan_only(task_path: Path | None, base_config):
    """Generate plans for eval tasks using the architect (opus-high).

    Runs the architect against the pre-task commit for each task,
    saves the resulting plan into the task JSON file.
    """
    from orchestrator.evals.runner import load_task
    from orchestrator.evals.snapshots import cleanup_eval_worktree, create_eval_worktree

    tasks_dir = Path(__file__).parent / 'evals' / 'tasks'

    if task_path:
        task_paths = [task_path]
    else:
        task_paths = sorted(tasks_dir.glob('*.json'))

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
            worktree = await create_eval_worktree(
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
                    click.echo(f'    FAILED: architect produced no plan.json', err=True)
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


if __name__ == '__main__':
    main()
