"""CLI entry point: `orchestrator run --prd X`."""

import asyncio
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


if __name__ == '__main__':
    main()
