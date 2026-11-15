"""Only a program run of the orchestrator CLI loads ``.env`` into ``os.environ``.

Each probe runs in a fresh interpreter whose cwd holds a fixture ``.env``:
under ``python -c`` python-dotenv searches from the cwd, so the child finds
that file and never the checkout's real one. Probes report the NAMES the
snippet added to ``os.environ``, never their values.
"""

import json
import subprocess
import sys
from pathlib import Path

import pytest
from _orch_helpers import sanitized_probe_env

PROBE = 'DF_CLI_DOTENV_PROBE'

_PRELUDE = 'import json, os, sys\nbefore = set(os.environ)\n'
_EPILOGUE = '\nprint(json.dumps(sorted(set(os.environ) - before)))\n'


def _names_added_by(snippet: str, cwd: Path) -> list[str]:
    result = subprocess.run(
        [sys.executable, '-c', _PRELUDE + snippet + _EPILOGUE],
        cwd=cwd,
        env=sanitized_probe_env(drop=(PROBE,)),
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout.splitlines()[-1])


@pytest.fixture
def dotenv_dir(tmp_path: Path) -> Path:
    (tmp_path / '.env').write_text(f'{PROBE}=loaded\n')
    return tmp_path


def test_importing_the_cli_loads_no_dotenv(dotenv_dir: Path):
    assert _names_added_by('import orchestrator.cli', dotenv_dir) == []


def test_an_in_process_cli_invocation_loads_no_dotenv(dotenv_dir: Path):
    snippet = (
        'from click.testing import CliRunner\n'
        'from orchestrator.cli import main\n'
        "CliRunner().invoke(main, ['--help'])\n"
    )
    assert _names_added_by(snippet, dotenv_dir) == []


def test_running_the_cli_as_a_program_loads_the_dotenv(dotenv_dir: Path):
    snippet = (
        'from orchestrator.cli import main\n'
        "sys.argv = ['orchestrator', '--help']\n"
        'try:\n'
        '    main()\n'
        'except SystemExit:\n'
        '    pass\n'
    )
    assert PROBE in _names_added_by(snippet, dotenv_dir)


def test_running_the_eval_runner_module_loads_the_dotenv(dotenv_dir: Path):
    snippet = (
        'import runpy\n'
        "sys.argv = ['runner', '--help']\n"
        'try:\n'
        "    runpy.run_module('orchestrator.evals.runner', run_name='__main__')\n"
        'except SystemExit:\n'
        '    pass\n'
    )
    assert PROBE in _names_added_by(snippet, dotenv_dir)
