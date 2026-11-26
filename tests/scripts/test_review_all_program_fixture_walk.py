"""``scripts/check_review_all_program_fixture.py``, the review-all program fixture walk, run as a lead runs it.

The walk is itself run ``-I -S`` from a cwd outside the repo, which also proves
the walk is stdlib-only.
"""
from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

from df_pytest_isolation import run_in_new_session

REPO_ROOT = Path(__file__).parents[2]
WALK = REPO_ROOT / 'scripts' / 'check_review_all_program_fixture.py'
EXPECTED = ('B1',)
SCENARIO_LINE = re.compile(r'^(B\d+) (ok|FAIL) ')
SUMMARY_PREFIX = 'fixture walk: '
WALK_TIMEOUT_SECS = 240


def _walk_cwd_env(tmp_path: Path) -> tuple[Path, dict[str, str]]:
    cwd = tmp_path / 'cwd'
    scratch = tmp_path / 'tmp'
    cwd.mkdir(exist_ok=True)
    scratch.mkdir(exist_ok=True)
    return cwd, {'PATH': os.environ['PATH'], 'HOME': str(cwd), 'TMPDIR': str(scratch)}


def _walk(walk: Path, tmp_path: Path) -> subprocess.CompletedProcess[str]:
    cwd, env = _walk_cwd_env(tmp_path)
    return run_in_new_session(
        [sys.executable, '-I', '-S', str(walk)], env=env, cwd=cwd, timeout=WALK_TIMEOUT_SECS,
    )


def _verdicts(stdout: str) -> dict[str, list[str]]:
    verdicts: dict[str, list[str]] = {}
    for line in stdout.splitlines():
        match = SCENARIO_LINE.match(line)
        if match:
            verdicts.setdefault(match.group(1), []).append(match.group(2))
    return verdicts


def _output(proc: subprocess.CompletedProcess[str]) -> str:
    return f'stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}'


def test_the_walk_holds_every_expected_scenario(tmp_path: Path) -> None:
    proc = _walk(WALK, tmp_path)
    assert proc.returncode == 0, _output(proc)
    assert _verdicts(proc.stdout) == {sid: ['ok'] for sid in EXPECTED}, _output(proc)
    stray = [
        line for line in proc.stdout.splitlines()
        if not SCENARIO_LINE.match(line) and not line.startswith(SUMMARY_PREFIX)
    ]
    assert stray == [], _output(proc)


def test_the_walk_runs_as_an_executable(tmp_path: Path) -> None:
    cwd, env = _walk_cwd_env(tmp_path)
    proc = subprocess.run(
        [str(WALK), '--help'], cwd=cwd, env=env, capture_output=True, text=True, timeout=60,
    )
    assert proc.returncode == 0, _output(proc)
