"""This suite loads no ``.env`` into ``os.environ``, however a module reaches python-dotenv.

Modules imported during collection call python-dotenv at import time (e.g.
``graphiti_core/driver/driver.py``), and a no-path load walks up to the main
checkout's ``.env``. Verify strips the account pool from its child
(``orchestrator/src/orchestrator/verify.py::_target_subprocess_env``, task
6828), so a worker that reloads it passes tests that a lone re-run fails. The
suite therefore switches python-dotenv off with the library's own
``PYTHON_DOTENV_DISABLED`` before collection.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import dotenv
import pytest
from _probe_session import run_probe_session
from shared.secret_env import is_secret_env_name

PROBE = Path(__file__).with_name('_import_time_dotenv_probe.py')
FIXTURE_POOL_TOKEN = 'CLAUDE_OAUTH_TOKEN_DOTENV_PROBE'
FIXTURE_API_KEY = 'DOTENV_PROBE_API_KEY'


def _secret_names_recorded_by_probe(tmp_path: Path, *extra: str) -> list[str]:
    cwd = tmp_path / 'cwd'
    cwd.mkdir()
    (cwd / '.env').write_text(f'{FIXTURE_POOL_TOKEN}=loaded\n{FIXTURE_API_KEY}=loaded\n')
    basetemp = tmp_path / 'probe-basetemp'

    run = run_probe_session(
        PROBE, basetemp=basetemp, kill_secs=60, backstop_secs=240, cwd=cwd,
        drop_env=('PYTHON_DOTENV_DISABLED', *(name for name in os.environ if is_secret_env_name(name))),
        extra=extra,
    )

    output = run.stdout[-3000:] + run.stderr[-2000:]
    assert run.returncode == 0, output
    assert '1 passed' in run.stdout, output
    [recorded] = basetemp.rglob('secret-names.json')
    return json.loads(recorded.read_text())


def test_collecting_under_the_suites_conftest_loads_no_secret_name(tmp_path: Path) -> None:
    assert _secret_names_recorded_by_probe(tmp_path) == []


def test_without_the_conftest_the_probes_load_takes_effect(tmp_path: Path) -> None:
    recorded = _secret_names_recorded_by_probe(tmp_path, '--noconftest')

    assert {FIXTURE_POOL_TOKEN, FIXTURE_API_KEY} <= set(recorded)


def test_dotenv_loading_is_off_in_this_test_process(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    dotenv_file = tmp_path / '.env'
    dotenv_file.write_text(f'{FIXTURE_POOL_TOKEN}=loaded\n')
    monkeypatch.setenv(FIXTURE_POOL_TOKEN, 'sentinel')

    dotenv.load_dotenv(dotenv_file, override=True)
    value = os.environ.get(FIXTURE_POOL_TOKEN)

    assert value == 'sentinel'
