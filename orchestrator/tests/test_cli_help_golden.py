"""Every orchestrator command's ``--help`` matches the recorded golden.

The golden (``fixtures/cli_help_golden/help.json``) was captured from the tree
before task 6748 moved the eval commands and folded the config preamble, so
"command names and options stay byte-identical" is checked against a record,
not against the code itself. See that directory's PROVENANCE.md.
"""

import difflib
import json
from pathlib import Path

import click
import pytest
from click.testing import CliRunner

from orchestrator.cli import main

GOLDEN_PATH = Path(__file__).parent / 'fixtures' / 'cli_help_golden' / 'help.json'
HELP_TERMINAL_WIDTH = 100


def _command_paths(group: click.Group, prefix: tuple[str, ...] = ()) -> list[tuple[str, ...]]:
    paths = [prefix]
    for name, command in group.commands.items():
        if isinstance(command, click.Group):
            paths.extend(_command_paths(command, (*prefix, name)))
        else:
            paths.append((*prefix, name))
    return paths


def render_cli_help(group: click.Group) -> dict[str, str]:
    """``--help`` output of *group* and every command beneath it, keyed by space-joined path ('' is the root)."""
    runner = CliRunner()
    rendered = {}
    for path in _command_paths(group):
        result = runner.invoke(group, [*path, '--help'], terminal_width=HELP_TERMINAL_WIDTH)
        assert result.exit_code == 0, (path, result.output)
        rendered[' '.join(path)] = result.stdout
    return rendered


@pytest.fixture(scope='module')
def golden() -> dict[str, str]:
    return json.loads(GOLDEN_PATH.read_text())


@pytest.fixture(scope='module')
def rendered() -> dict[str, str]:
    return render_cli_help(main)


def test_no_command_disappears_or_is_renamed(golden, rendered):
    assert sorted(rendered) == sorted(golden)


def test_every_help_text_matches_the_golden(golden, rendered):
    drifted = {
        path: ''.join(difflib.unified_diff(
            text.splitlines(keepends=True), rendered.get(path, '').splitlines(keepends=True),
            'golden', 'rendered',
        ))
        for path, text in golden.items()
        if rendered.get(path) != text
    }
    assert drifted == {}
