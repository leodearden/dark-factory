"""Every orchestrator command's usage line and option list match a golden captured before task 6748.

Command descriptions (docstrings) are deliberately not pinned; see fixtures/cli_help_golden/PROVENANCE.md.
"""

import json
import sys
from pathlib import Path

import click
import pytest

from orchestrator.cli import main

GOLDEN_PATH = Path(__file__).parent / 'fixtures' / 'cli_help_golden' / 'help.json'


def _command_paths(group: click.Group, prefix: tuple[str, ...] = ()) -> list[tuple[str, ...]]:
    paths = [prefix]
    for name, command in group.commands.items():
        if isinstance(command, click.Group):
            paths.extend(_command_paths(command, (*prefix, name)))
        else:
            paths.append((*prefix, name))
    return paths


def _context_for(root: click.Group, path: tuple[str, ...]) -> click.Context:
    command: click.Command = root
    ctx = root.make_context(root.name, [], resilient_parsing=True)
    for name in path:
        assert isinstance(command, click.Group), path
        command = command.commands[name]
        ctx = command.make_context(name, [], parent=ctx, resilient_parsing=True)
    return ctx


def render_cli_options(group: click.Group) -> dict[str, dict]:
    """Usage pieces and option help records of *group* and every command beneath it, keyed by space-joined path ('' is the root)."""
    rendered = {}
    for path in _command_paths(group):
        ctx = _context_for(group, path)
        command = ctx.command
        rendered[' '.join(path)] = {
            'usage': ' '.join(command.collect_usage_pieces(ctx)),
            'options': [
                list(record) for param in command.get_params(ctx)
                if (record := param.get_help_record(ctx)) is not None
            ],
        }
    return rendered


@pytest.fixture(scope='module')
def golden() -> dict[str, dict]:
    return json.loads(GOLDEN_PATH.read_text())


@pytest.fixture(scope='module')
def rendered() -> dict[str, dict]:
    return render_cli_options(main)


def test_no_command_disappears_or_is_renamed(golden, rendered):
    assert sorted(rendered) == sorted(golden)


def test_every_command_lists_the_same_usage_and_options(golden, rendered):
    drifted = {
        path: {'golden': record, 'rendered': rendered.get(path)}
        for path, record in golden.items()
        if rendered.get(path) != record
    }
    assert drifted == {}


if __name__ == '__main__':
    if sys.argv[1:] != ['--write']:
        print(f'usage: {sys.argv[0]} --write', file=sys.stderr)
        sys.exit(2)
    import orchestrator.cli

    GOLDEN_PATH.write_text(json.dumps(render_cli_options(main), indent=2, sort_keys=True) + '\n')
    print(GOLDEN_PATH)
    print(orchestrator.cli.__file__)
