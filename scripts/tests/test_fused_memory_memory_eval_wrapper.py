"""Tests for scripts/fused-memory-memory-eval.sh, the memory-eval timer's wrapper (task 3211).

Drives the wrapper via subprocess. The MEMORY_EVAL_RUNNER_CMD seam points at a
fake recorder, or a fake ``uv`` stands in for the default prefix; REPO points
at a tmp dir. Real uv, fused_memory and live stores are never touched. The
harness follows scripts/tests/test_flag_marker_sweep_wrapper.py.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

WRAPPER = Path(__file__).parent.parent / 'fused-memory-memory-eval.sh'
RUNNER_TAIL = ('fused-memory', 'scripts', 'memory_eval_runner.py')

# Resolved in the PARENT: a row below hands the child a PATH without bash on it.
BASH = shutil.which('bash') or '/bin/bash'

# Scrubbed from the inherited environment so every row sees only what the wrapper exports.
_SERVICE_ENV = ('CONFIG_PATH', 'PROJECT_ROOT', 'FALKORDB_URI', 'UV_BIN', 'MEMORY_EVAL_RUNNER_CMD')

_RECORDER_SRC = '''"""Records argv[1:] and os.environ into $FAKE_RUNNER_STATE, then exits $FAKE_RUNNER_EXIT_CODE."""
import json
import os
import sys

state_path = os.environ["FAKE_RUNNER_STATE"]
with open(state_path) as f:
    state = json.load(f)
state["calls"].append(sys.argv[1:])
state["envs"].append(dict(os.environ))
with open(state_path, "w") as f:
    json.dump(state, f)
sys.exit(int(os.environ.get("FAKE_RUNNER_EXIT_CODE", "0")))
'''


def _executable(directory: Path, name: str) -> tuple[Path, Path]:
    """An absolute-shebang recorder (it must run under a scrubbed PATH) and its state file."""
    directory.mkdir(parents=True, exist_ok=True)
    fake = directory / name
    fake.write_text(f'#!{sys.executable}\n' + _RECORDER_SRC)
    fake.chmod(0o755)
    state_path = directory / f'{name}-state.json'
    state_path.write_text(json.dumps({'calls': [], 'envs': []}))
    return fake, state_path


def _state(state_path: Path) -> dict:
    return json.loads(state_path.read_text())


def _base_env(repo: Path, state_path: Path, **extra: str) -> dict[str, str]:
    env = {k: v for k, v in os.environ.items() if k not in _SERVICE_ENV}
    env['REPO'] = str(repo)
    env['FAKE_RUNNER_STATE'] = str(state_path)
    env.update(extra)
    return env


def _run(env: dict[str, str], *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [BASH, str(WRAPPER), *args], env=env, capture_output=True, text=True, timeout=30,
    )


def _repo(tmp_path: Path, name: str = 'fake-repo', dotenv: str | None = None) -> Path:
    repo = tmp_path / name
    repo.mkdir()
    if dotenv is not None:
        (repo / '.env').write_text(dotenv)
    return repo


def _seamed_run(tmp_path, *args, dotenv=None, exit_code=0):
    repo = _repo(tmp_path, dotenv=dotenv)
    recorder, state_path = _executable(tmp_path / 'bin', 'fake-runner')
    env = _base_env(
        repo, state_path,
        MEMORY_EVAL_RUNNER_CMD=str(recorder), FAKE_RUNNER_EXIT_CODE=str(exit_code),
    )
    return _run(env, *args), repo, state_path


def test_wrapper_is_executable():
    assert os.access(WRAPPER, os.X_OK), f'chmod +x {WRAPPER}'


def test_the_runner_sees_the_service_environment(tmp_path):
    result, repo, state_path = _seamed_run(
        tmp_path, dotenv='MEMORY_EVAL_WRAPPER_SENTINEL=from-dot-env\n',
    )
    assert result.returncode == 0, f'stdout={result.stdout!r} stderr={result.stderr!r}'

    (argv,) = _state(state_path)['calls']
    assert Path(argv[-1]).parts[-3:] == RUNNER_TAIL, argv
    (env,) = _state(state_path)['envs']
    assert env['CONFIG_PATH'] == str(repo / 'fused-memory' / 'config' / 'config.yaml')
    assert env['PROJECT_ROOT'] == str(repo)
    assert env['FALKORDB_URI'] == 'redis://localhost:6379'
    assert env['MEMORY_EVAL_WRAPPER_SENTINEL'] == 'from-dot-env'


def test_dotenv_overrides_the_store_default(tmp_path):
    result, _, state_path = _seamed_run(tmp_path, dotenv='FALKORDB_URI=redis://test-env-host:1234\n')
    assert result.returncode == 0, result.stderr
    (env,) = _state(state_path)['envs']
    assert env['FALKORDB_URI'] == 'redis://test-env-host:1234'


def test_arguments_pass_through_after_the_runner(tmp_path):
    result, _, state_path = _seamed_run(tmp_path, '--artifact-root', '/tmp/x')
    assert result.returncode == 0, result.stderr
    (argv,) = _state(state_path)['calls']
    assert Path(argv[-3]).parts[-3:] == RUNNER_TAIL, argv
    assert argv[-2:] == ['--artifact-root', '/tmp/x']


@pytest.mark.parametrize('code', [1, 2])
def test_the_runner_exit_status_is_the_wrapper_exit_status(tmp_path, code):
    result, _, _ = _seamed_run(tmp_path, exit_code=code)
    assert result.returncode == code, f'stdout={result.stdout!r} stderr={result.stderr!r}'


def test_the_default_prefix_is_uv_run_no_sync_against_fused_memory(tmp_path):
    """A space in REPO makes a dropped quote around "$FM" show up as a word-split argv."""
    repo = _repo(tmp_path, name='fake repo with spaces')
    uv, state_path = _executable(tmp_path / 'uv-bin', 'uv')
    env = _base_env(repo, state_path, UV_BIN=str(uv))

    result = _run(env)
    assert result.returncode == 0, f'stdout={result.stdout!r} stderr={result.stderr!r}'

    fm = repo / 'fused-memory'
    assert _state(state_path)['calls'] == [[
        'run', '--no-sync', '--project', str(fm), 'python',
        str(fm / 'scripts' / 'memory_eval_runner.py'),
    ]]


def test_an_unresolvable_uv_exits_127_with_an_error_line(tmp_path):
    if os.path.exists('/usr/local/bin/uv'):
        pytest.skip('/usr/local/bin/uv exists on this host, so the last-resort fallback resolves')
    repo = _repo(tmp_path)
    fake_home = tmp_path / 'fake-home'
    fake_home.mkdir()
    env = _base_env(
        repo, tmp_path / 'unused.json', PATH='/usr/bin:/bin', HOME=str(fake_home),
    )

    result = _run(env)
    assert result.returncode == 127, f'stdout={result.stdout!r} stderr={result.stderr!r}'
    assert 'ERROR:' in result.stderr and 'uv' in result.stderr, result.stderr


def test_a_uv_bin_pin_that_is_not_executable_never_falls_through(tmp_path):
    repo = _repo(tmp_path)
    uv, state_path = _executable(tmp_path / 'uv-bin', 'uv')
    bad_uv = tmp_path / 'bad-uv'
    bad_uv.write_text('#!/bin/sh\nexit 0\n')
    bad_uv.chmod(0o644)
    env = _base_env(
        repo, state_path,
        PATH=f'{uv.parent}{os.pathsep}/usr/bin{os.pathsep}/bin', UV_BIN=str(bad_uv),
    )

    result = _run(env)
    assert result.returncode == 127, f'stdout={result.stdout!r} stderr={result.stderr!r}'
    assert str(bad_uv) in result.stderr, result.stderr
    assert _state(state_path)['calls'] == []
