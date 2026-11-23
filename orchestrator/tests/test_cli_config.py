"""``orchestrator.cli_config.pass_config``: the one way a CLI command obtains its config.

The decorator tests drive a throwaway command whose options sit above and below
``@pass_config()``. The uniform-failure test drives every config-loading command
of the real CLI with no config, and pins that each fails through the same single
handler. check-config is deliberately absent: it lints the raw YAML and never
loads a config.
"""

from pathlib import Path

import click
import pytest
import yaml
from click.testing import CliRunner

from orchestrator import cli_config
from orchestrator.cli import main
from orchestrator.config import OrchestratorConfig

MISSING_CONFIG_LINE = 'Error: --config is required (or set ORCH_CONFIG_PATH).'


def _probe_command(received: list, **pass_config_kwargs) -> click.Command:
    @click.command()
    @click.option('--a', default=None, help='Option declared above the decorator.')
    @cli_config.pass_config(**pass_config_kwargs)
    @click.option('--b', default=None, help='Option declared below the decorator.')
    def probe(a, config, b):
        received.append(config)

    return probe


def _flattened(text: str) -> str:
    return ' '.join(text.split())


def test_missing_config_exits_1_with_the_error_and_never_runs_the_body(monkeypatch):
    monkeypatch.delenv('ORCH_CONFIG_PATH', raising=False)
    received: list = []

    result = CliRunner().invoke(_probe_command(received), [])

    assert result.exit_code == 1
    assert MISSING_CONFIG_LINE in result.stderr
    assert received == []


def test_a_real_config_file_reaches_the_body_as_a_loaded_config(monkeypatch, tmp_path: Path):
    config_file = tmp_path / 'dark-factory-orchestrator.yaml'
    config_file.write_text(yaml.safe_dump({'project_root': str(tmp_path)}))
    monkeypatch.setenv('ORCH_CONFIG_PATH', str(config_file))
    received: list = []

    result = CliRunner().invoke(_probe_command(received), ['--config', str(config_file)])

    assert result.exit_code == 0, result.output
    [config] = received
    assert isinstance(config, OrchestratorConfig)
    assert Path(config.project_root) == tmp_path


def test_help_lists_config_where_the_decorator_stands_with_the_default_help():
    result = CliRunner().invoke(_probe_command([]), ['--help'], terminal_width=100)

    assert result.exit_code == 0, result.output
    help_text = result.stdout
    assert help_text.index('--a') < help_text.index('--config') < help_text.index('--b')
    assert _flattened(cli_config.CONFIG_OPTION_HELP) in _flattened(help_text)


def test_help_override_renders_as_the_config_help():
    result = CliRunner().invoke(
        _probe_command([], help='Overridden config help.'), ['--help'], terminal_width=100,
    )

    assert result.exit_code == 0, result.output
    assert 'Overridden config help.' in result.stdout
    assert _flattened(cli_config.CONFIG_OPTION_HELP) not in _flattened(result.stdout)


CONFIG_LOADING_ARGVS = [
    ['run'],
    ['status'],
    ['flake-ledger'],
    ['verify-cache', 'status'],
    ['probe-models'],
    ['verify-merge', '--sha', '0' * 40, '--spec', '{}'],
    ['verify-task', '--sha', '0' * 40, '--spec', '{}'],
    ['cancel-verify', '--request-id', 'r'],
    ['eval'],
    ['eval-ofat'],
    ['eval-matrix'],
    ['eval-confirm', '--arch', 'a', '--impl', 'i'],
    ['eval-shadow', 'report'],
]


@pytest.mark.parametrize('argv', CONFIG_LOADING_ARGVS, ids=' '.join)
def test_every_config_loading_command_fails_the_same_way_without_a_config(monkeypatch, argv):
    monkeypatch.delenv('ORCH_CONFIG_PATH', raising=False)

    result = CliRunner().invoke(main, argv)

    assert result.exit_code == 1, result.output
    assert MISSING_CONFIG_LINE in result.stderr
