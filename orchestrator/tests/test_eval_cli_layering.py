"""The eval CLI sits below the root CLI: nothing under ``orchestrator.evals`` loads ``orchestrator.cli``.

The probes run in a fresh interpreter, because an in-process check would see an
``orchestrator.cli`` that some other test already imported. Each child's cwd is
an empty tmp dir, so the runner entry's program-run ``.env`` load finds nothing.
"""

import json
import subprocess
import sys
from pathlib import Path

from _orch_helpers import sanitized_probe_env
from click.testing import CliRunner

import orchestrator.evals
from orchestrator.cli import main

EVAL_HELP_LINE = 'Run multi-provider implementor evaluations.'
MARKER = 'ROOT_CLI_LOADED='


def _probe(snippet: str, cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, '-c', snippet],
        cwd=cwd,
        env=sanitized_probe_env(),
        capture_output=True,
        text=True,
        timeout=120,
    )


def _root_cli_loaded(stdout: str) -> bool:
    [line] = [line for line in stdout.splitlines() if line.startswith(MARKER)]
    return json.loads(line.removeprefix(MARKER))


def test_importing_the_eval_cli_does_not_load_the_root_cli(tmp_path: Path):
    result = _probe(
        'import json, sys\n'
        'import orchestrator.evals.cli\n'
        f"print({MARKER!r} + json.dumps('orchestrator.cli' in sys.modules))\n",
        tmp_path,
    )

    assert result.returncode == 0, result.stderr
    assert _root_cli_loaded(result.stdout) is False


def test_the_runner_module_entry_does_not_load_the_root_cli(tmp_path: Path):
    result = _probe(
        'import json, runpy, sys\n'
        "sys.argv = ['runner', '--help']\n"
        'try:\n'
        "    runpy.run_module('orchestrator.evals.runner', run_name='__main__')\n"
        'except SystemExit:\n'
        '    pass\n'
        f"print({MARKER!r} + json.dumps('orchestrator.cli' in sys.modules))\n",
        tmp_path,
    )

    assert result.returncode == 0, result.stderr
    assert EVAL_HELP_LINE in result.stdout
    assert _root_cli_loaded(result.stdout) is False


def test_eval_list_fixtures_defaults_to_the_packaged_fixture_dir():
    packaged = Path(orchestrator.evals.__file__).parent / 'tasks'
    fixture_count = len(list(packaged.glob('*.json')))
    runner = CliRunner()

    default = runner.invoke(main, ['eval-list-fixtures'])
    explicit = runner.invoke(main, ['eval-list-fixtures', '--tasks-dir', str(packaged)])

    assert default.exit_code == 0, default.output
    assert explicit.exit_code == 0, explicit.output
    assert default.stdout == explicit.stdout
    assert fixture_count > 0
    assert f'TOTAL: {fixture_count} ' in default.stdout
