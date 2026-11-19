"""How an orchestrator CLI program obtains its configuration.

``load_program_dotenv`` fills the process env from ``.env`` for a program run.
``pass_config`` gives a command its ``--config`` option and calls it with the
loaded :class:`~orchestrator.config.OrchestratorConfig`; it holds the one
handler for a missing config.
"""

import functools
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import click
from dotenv import load_dotenv

from orchestrator.config import ConfigRequiredError, OrchestratorConfig, load_config

CONFIG_OPTION_HELP_PREFIX = (
    'Path to orchestrator config YAML (REQUIRED unless ORCH_CONFIG_PATH is set).'
)
CONFIG_OPTION_HELP = (
    f'{CONFIG_OPTION_HELP_PREFIX} '
    'Selects the target project — sets project_root and fused_memory.project_id.'
)

CommandCallback = Callable[..., Any]


def load_program_dotenv() -> None:
    """Load .env into os.environ for a program run of the CLI; importing a CLI module never does."""
    load_dotenv()


def pass_config(help: str = CONFIG_OPTION_HELP) -> Callable[[CommandCallback], CommandCallback]:
    """Add ``--config`` where this decorator stands and call the command with ``config=``.

    The callback is wrapped with :func:`functools.wraps`, which shares click's
    ``__click_params__`` list, so ``--config`` keeps this decorator's position
    in ``--help``. With no config to load, the command body never runs: the
    reason is printed as ``Error: ...`` on stderr and the process exits 1.
    """

    def decorate(command: CommandCallback) -> CommandCallback:
        @functools.wraps(command)
        def with_config(*args: Any, config_path: Path | None, **kwargs: Any) -> Any:
            return command(*args, config=_load_or_exit(config_path), **kwargs)

        return click.option(
            '--config', 'config_path', type=click.Path(exists=True, path_type=Path),
            default=None, help=help,
        )(with_config)

    return decorate


def _load_or_exit(config_path: Path | None) -> OrchestratorConfig:
    try:
        return load_config(config_path)
    except ConfigRequiredError as e:
        click.echo(f'Error: {e}', err=True)
        sys.exit(1)
