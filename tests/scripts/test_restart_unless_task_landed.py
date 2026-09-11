"""Behavioural guard for scripts/restart-orchestrators-unless-task-landed.sh.

The script decides whether a deterministic deploy milestone should restart the
orchestrator fleet at all: it restarts only when its named task has NOT landed
on main. Both directions are load-bearing and fail in opposite ways — a missed
restart silently strands the gated task, a surplus restart kills every in-flight
agent — so both are pinned here.

NOTHING HERE TOUCHES REAL SYSTEMD OR THE REAL REPO. Each test builds a hermetic
git repo under tmp_path, copies the script into its scripts/ dir (the script
resolves the repo from its own location, so the copy IS the isolation), and
injects RESTART_SCRIPT as a recording stub. The stub's argv is written to a file
so "did it restart, and with what" is observable rather than assumed.

THE rc=128 ARM IS THE POINT. Merge cleanup DELETES a landed branch, so the
common post-landing state is "no ref". `git merge-base --is-ancestor` exits 128
there, and the two-way `&& landed || not-landed` idiom reports that as NOT
landed — inverting the truth and triggering a restart that was not needed.
test_landed_via_marker_when_branch_was_deleted is the regression pin for it.
"""

import shutil
import subprocess
from pathlib import Path

import pytest

SCRIPT_NAME = 'restart-orchestrators-unless-task-landed.sh'
REPO_ROOT = Path(__file__).resolve().parents[2]
REAL_SCRIPT = REPO_ROOT / 'scripts' / SCRIPT_NAME


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ['git', *args], cwd=repo, check=True,
        capture_output=True, text=True,
    ).stdout.strip()


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    """A hermetic git repo carrying a copy of the script under scripts/."""
    r = tmp_path / 'repo'
    (r / 'scripts').mkdir(parents=True)
    shutil.copy2(REAL_SCRIPT, r / 'scripts' / SCRIPT_NAME)
    (r / 'scripts' / SCRIPT_NAME).chmod(0o755)

    _git(r.parent, 'init', '--initial-branch=main', str(r))
    _git(r, 'config', 'user.email', 'test@example.invalid')
    _git(r, 'config', 'user.name', 'Test')
    _git(r, 'add', '-A')
    _git(r, 'commit', '-m', 'base')
    return r


@pytest.fixture
def stub(tmp_path: Path) -> tuple[Path, Path]:
    """A recording restart stub: (script_path, argv_log_path)."""
    log = tmp_path / 'restart-argv.log'
    s = tmp_path / 'stub-restart.sh'
    s.write_text(f'#!/bin/sh\nprintf "%s" "$*" > {log}\n')
    s.chmod(0o755)
    return s, log


def _run(repo: Path, stub_path: Path, *args: str, cwd: Path | None = None):
    return subprocess.run(
        [str(repo / 'scripts' / SCRIPT_NAME), *args],
        cwd=cwd or repo, capture_output=True, text=True,
        env={'PATH': '/usr/bin:/bin', 'HOME': str(repo.parent),
             'RESTART_SCRIPT': str(stub_path)},
    )


def test_landed_via_merge_marker_does_not_restart(repo, stub):
    stub_path, log = stub
    _git(repo, 'commit', '--allow-empty', '-m', 'Merge task/777 into main')

    r = _run(repo, stub_path, '777', '--drain')

    assert r.returncode == 0, r.stderr
    assert not log.exists(), f'restart was invoked but must not have been: {r.stderr}'


def test_landed_via_marker_when_branch_was_deleted(repo, stub):
    """The rc=128 trap: landed branches are deleted, ancestry then exits 128."""
    stub_path, log = stub
    _git(repo, 'commit', '--allow-empty', '-m', 'Merge task/778 into main')
    # No refs/heads/task/778 exists — exactly the post-cleanup state.

    r = _run(repo, stub_path, '778', '--drain')

    assert r.returncode == 0, r.stderr
    assert not log.exists(), 'a deleted branch with a marker must read as LANDED'


def test_landed_via_ancestry_without_marker_does_not_restart(repo, stub):
    """Fast-forward / coalesced landings carry no marker of their own."""
    stub_path, log = stub
    _git(repo, 'branch', 'task/779')  # points at main, so it IS an ancestor

    r = _run(repo, stub_path, '779', '--drain')

    assert r.returncode == 0, r.stderr
    assert not log.exists(), 'an ancestor branch must read as LANDED'


def test_unlanded_branch_restarts_and_forwards_args(repo, stub):
    stub_path, log = stub
    _git(repo, 'checkout', '-q', '-b', 'task/780')
    _git(repo, 'commit', '--allow-empty', '-m', 'work not on main')
    _git(repo, 'checkout', '-q', 'main')

    r = _run(repo, stub_path, '780', '--drain')

    assert r.returncode == 0, r.stderr
    assert log.read_text() == '--drain', 'restart args must be forwarded verbatim'


def test_unknown_task_restarts_fail_safe(repo, stub):
    """No marker, no branch: fail SAFE toward restarting."""
    stub_path, log = stub

    r = _run(repo, stub_path, '99999999', '--drain')

    assert r.returncode == 0, r.stderr
    assert log.read_text() == '--drain'


def test_missing_task_id_is_a_usage_error(repo, stub):
    stub_path, log = stub

    r = _run(repo, stub_path)

    assert r.returncode == 2
    assert 'usage:' in r.stderr
    assert not log.exists()


def test_decision_is_independent_of_cwd(repo, stub, tmp_path):
    """The transient systemd unit's cwd defaults to $HOME, not the repo."""
    stub_path, log = stub
    _git(repo, 'commit', '--allow-empty', '-m', 'Merge task/781 into main')
    elsewhere = tmp_path / 'elsewhere'
    elsewhere.mkdir()

    r = _run(repo, stub_path, '781', cwd=elsewhere)

    assert r.returncode == 0, r.stderr
    assert not log.exists(), 'cwd must not affect the landed decision'
