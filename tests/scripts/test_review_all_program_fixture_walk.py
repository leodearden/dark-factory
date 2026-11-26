"""``scripts/check_review_all_program_fixture.py``, the review-all program fixture walk, run as a lead runs it.

The walk is itself run ``-I -S`` from a cwd outside the repo, which also proves
the walk is stdlib-only.
"""
from __future__ import annotations

import contextlib
import os
import re
import shutil
import signal
import subprocess
import sys
from pathlib import Path

import pytest
from shared.whole_tree_guard_family import WHOLE_TREE_SCAN_TEST_TIMEOUT

from df_pytest_isolation import run_in_new_session

pytestmark = pytest.mark.timeout(WHOLE_TREE_SCAN_TEST_TIMEOUT)

REPO_ROOT = Path(__file__).parents[2]
WALK = REPO_ROOT / 'scripts' / 'check_review_all_program_fixture.py'
EXPECTED = ('B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B9', 'B10', 'B11', 'B12', 'B13', 'B14', 'B16')
PROGRAM_DIR = REPO_ROOT / 'skills' / 'review-all' / 'scripts' / 'program'
IMPORT_ANCHOR = 'from __future__ import annotations\n'
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


def test_a_sigterm_mid_walk_leaves_no_process_in_its_group(tmp_path: Path) -> None:
    cwd, env = _walk_cwd_env(tmp_path)
    proc = subprocess.Popen(
        [sys.executable, '-I', '-S', str(WALK)], cwd=cwd, env=env, start_new_session=True,
        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True,
    )
    pgid = proc.pid
    try:
        assert proc.stdout is not None
        first = next((line for line in proc.stdout if SCENARIO_LINE.match(line)), None)
        assert first is not None, 'the walk printed no scenario line'
        os.kill(proc.pid, signal.SIGTERM)
        proc.wait(timeout=60)
        with pytest.raises(ProcessLookupError):
            os.killpg(pgid, 0)
    finally:
        with contextlib.suppress(ProcessLookupError):
            os.killpg(pgid, signal.SIGKILL)
        proc.wait(timeout=60)


def test_the_walk_runs_as_an_executable(tmp_path: Path) -> None:
    cwd, env = _walk_cwd_env(tmp_path)
    proc = subprocess.run(
        [str(WALK), '--help'], cwd=cwd, env=env, capture_output=True, text=True, timeout=60,
    )
    assert proc.returncode == 0, _output(proc)


def _mirror_with_planted_import(mirror: Path) -> Path:
    """A checkout mirror whose program scripts are copies, seam_scan.py importing httpx; returns its walk."""
    program = mirror / 'skills' / 'review-all' / 'scripts' / 'program'
    shutil.copytree(PROGRAM_DIR, program, ignore=shutil.ignore_patterns('__pycache__'))
    (mirror / 'scripts').mkdir()
    walk = Path(shutil.copy2(WALK, mirror / 'scripts' / WALK.name))
    for linked in ('shared', 'orchestrator', 'skills/spawn'):
        (mirror / linked).symlink_to(REPO_ROOT / linked)
    seam_scan = program / 'seam_scan.py'
    source = seam_scan.read_text(encoding='utf-8')
    assert IMPORT_ANCHOR in source, f'{seam_scan} has no {IMPORT_ANCHOR!r} to plant after'
    seam_scan.write_text(source.replace(IMPORT_ANCHOR, IMPORT_ANCHOR + 'import httpx\n', 1), encoding='utf-8')
    return walk


def _lines_of(stdout: str, scenario: str) -> list[str]:
    return [line for line in stdout.splitlines() if line.startswith(f'{scenario} ')]


def test_a_third_party_import_fails_the_walk_at_b14(tmp_path: Path) -> None:
    walk = _mirror_with_planted_import(tmp_path / 'mirror')

    proc = _walk(walk, tmp_path)

    assert proc.returncode == 1, _output(proc)
    (b14,) = _lines_of(proc.stdout, 'B14')
    assert b14.startswith('B14 FAIL ') and 'seam_scan.py' in b14 and 'httpx' in b14, _output(proc)
    assert _verdicts(proc.stdout)['B2'] == ['FAIL'], _output(proc)
    assert _verdicts(proc.stdout)['B1'] == ['ok'], _output(proc)
