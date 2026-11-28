"""scripts/lock-charter-drift-check.sh, with a fake recorder at its command seam or as ``uv`` (task 11623)."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent.parent
WRAPPER = SCRIPTS_DIR / 'lock-charter-drift-check.sh'

# The workstation's reify main checkout: the sibling the installed timer compares against.
WORKSTATION_REIFY = '/home/leo/src/reify'

_RECORDER_SRC = f'''#!{sys.executable}
"""Fake command: records its argv and the git redirection it inherited, exits $FAKE_RC."""
import json
import os
import sys

with open(os.environ["FAKE_RECORD"], "w") as f:
    json.dump({{"argv": sys.argv[1:], "env": {{k: os.environ.get(k) for k in ("GIT_DIR", "GIT_WORK_TREE")}}}}, f)
sys.exit(int(os.environ.get("FAKE_RC", "0")))
'''


def _recorder(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_RECORDER_SRC)
    path.chmod(0o755)
    return path


def _run_wrapper(tmp_path: Path, *, as_uv: bool = False, **env: str) -> tuple[subprocess.CompletedProcess[str], dict]:
    repo = tmp_path / 'repo'
    repo.mkdir(exist_ok=True)
    record = tmp_path / 'record.json'
    full_env = {**os.environ, 'REPO': str(repo), 'FAKE_RECORD': str(record)}
    full_env.pop('LOCK_CHARTER_DRIFT_CHECK_CMD', None)
    full_env.pop('REIFY_REPO', None)
    if as_uv:
        bin_dir = _recorder(tmp_path / 'bin' / 'uv').parent
        full_env['PATH'] = f'{bin_dir}{os.pathsep}{full_env["PATH"]}'
    else:
        full_env['LOCK_CHARTER_DRIFT_CHECK_CMD'] = str(_recorder(tmp_path / 'fake-heal'))
    full_env.update(env)
    completed = subprocess.run(['bash', str(WRAPPER)], env=full_env, capture_output=True, text=True, timeout=60)
    return completed, json.loads(record.read_text())


def test_wrapper_is_executable():
    assert os.access(WRAPPER, os.X_OK), f'chmod +x {WRAPPER}'


def test_the_override_runs_the_heal_router_on_its_repo_against_the_workstation_reify(tmp_path):
    completed, recorded = _run_wrapper(tmp_path)

    assert completed.returncode == 0, completed.stderr
    repo = str(tmp_path / 'repo')
    assert recorded['argv'] == [
        f'{repo}/scripts/lock_charter_drift_heal.py', '--project-root', repo, '--reify-repo', WORKSTATION_REIFY,
    ]


def test_reify_repo_overrides_the_sibling_compared_against(tmp_path):
    elsewhere = str(tmp_path / 'other-reify')
    completed, recorded = _run_wrapper(tmp_path, REIFY_REPO=elsewhere)

    assert completed.returncode == 0, completed.stderr
    assert recorded['argv'][-2:] == ['--reify-repo', elsewhere]


def test_the_default_prefix_is_uv_run_no_sync_on_the_shared_project(tmp_path):
    completed, recorded = _run_wrapper(tmp_path, as_uv=True)

    assert completed.returncode == 0, completed.stderr
    repo = str(tmp_path / 'repo')
    assert recorded['argv'] == [
        'run', '--no-sync', '--project', f'{repo}/shared', 'python',
        f'{repo}/scripts/lock_charter_drift_heal.py', '--project-root', repo, '--reify-repo', WORKSTATION_REIFY,
    ]


def test_a_failed_route_is_narrated_and_still_exits_zero(tmp_path):
    completed, _ = _run_wrapper(tmp_path, FAKE_RC='2')

    assert completed.returncode == 0, completed.stderr
    assert 'heal=2' in completed.stdout + completed.stderr


def test_the_wrapper_drops_an_inherited_git_redirection(tmp_path):
    _, recorded = _run_wrapper(tmp_path, GIT_DIR=str(tmp_path / 'elsewhere.git'), GIT_WORK_TREE=str(tmp_path))

    assert recorded['env'] == {'GIT_DIR': None, 'GIT_WORK_TREE': None}
