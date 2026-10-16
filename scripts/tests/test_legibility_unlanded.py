"""Tests for scripts/legibility/unlanded.py — landing a legibility write with
one bounded scoped commit, and putting it back when that commit does not land.

Most tests run against a REAL tmp git repo, because the behaviour under test
is what git does to the index and the worktree; the retry-count tests drive
``commit_only`` through its ``runner`` seam instead. The conftest autouse
fixture ``_isolate_legibility_trickle_state`` already points the legibility
state root, and therefore the quarantine, at a per-test tmp dir.
"""
from __future__ import annotations

import dataclasses
import os
import subprocess
import time
from pathlib import Path

import pytest
from legibility import trickle_state, unlanded

_CODEBOOK = 'docs/legibility/confusion-codebook.yaml'
_CENSUS_STATE = 'docs/legibility/census-state.json'
_HEAD_CODEBOOK = b'version: 2\nentries: []\n'
_HEAD_CENSUS_STATE = b'{"last_census_at": "2026-06-01"}\n'
_REFUSED = b'refused bytes\n'


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ['git', '-C', str(repo), *args], check=True, capture_output=True, text=True,
    ).stdout


def _repo(tmp_path: Path) -> Path:
    repo = tmp_path / 'repo'
    (repo / 'docs' / 'legibility').mkdir(parents=True)
    _git(repo, 'init', '-q', '-b', 'main')
    _git(repo, 'config', 'user.email', 'test@example.com')
    _git(repo, 'config', 'user.name', 'Test')
    (repo / _CODEBOOK).write_bytes(_HEAD_CODEBOOK)
    (repo / _CENSUS_STATE).write_bytes(_HEAD_CENSUS_STATE)
    _git(repo, 'add', '.')
    _git(repo, 'commit', '-q', '-m', 'initial')
    return repo


def _head_bytes(repo: Path, rel: str) -> bytes:
    return subprocess.run(
        ['git', '-C', str(repo), 'show', f'HEAD:{rel}'], check=True, capture_output=True,
    ).stdout


def _porcelain(repo: Path, *paths: str) -> str:
    return _git(repo, 'status', '--porcelain', '--', *paths)


def test_roll_back_restores_tracked_paths_to_head_and_quarantines_the_refused_bytes(
    tmp_path,
):
    repo = _repo(tmp_path)
    (repo / _CODEBOOK).write_bytes(_REFUSED + b'codebook\n')
    (repo / _CENSUS_STATE).write_bytes(_REFUSED + b'state\n')
    _git(repo, 'add', _CODEBOOK)

    rb = unlanded.roll_back(
        repo,
        [Path(_CODEBOOK), repo / _CENSUS_STATE],
        project_id='proj',
        label='census-2026-07-14',
    )

    assert rb.restored is True
    assert rb.failure is None
    assert (repo / _CODEBOOK).read_bytes() == _head_bytes(repo, _CODEBOOK)
    assert (repo / _CENSUS_STATE).read_bytes() == _head_bytes(repo, _CENSUS_STATE)
    assert _porcelain(repo, _CODEBOOK, _CENSUS_STATE) == ''
    assert rb.quarantine_dir is not None
    assert rb.quarantine_dir.parent == unlanded.quarantine_root('proj')
    assert rb.quarantine_dir.name.startswith('census-2026-07-14-')
    assert (rb.quarantine_dir / _CODEBOOK).read_bytes() == _REFUSED + b'codebook\n'
    assert (rb.quarantine_dir / _CENSUS_STATE).read_bytes() == _REFUSED + b'state\n'
    assert set(rb.paths) == {_CODEBOOK, _CENSUS_STATE}


def test_roll_back_removes_written_paths_head_does_not_track(tmp_path):
    repo = _repo(tmp_path)
    staged_new = 'plans/confusion-census-2026-07-14.md'
    untracked = 'plans/confusion-census-2026-07-14-payloads.json'
    (repo / 'plans').mkdir()
    (repo / staged_new).write_bytes(b'# report\n')
    (repo / untracked).write_bytes(b'[]\n')
    _git(repo, 'add', staged_new)

    rb = unlanded.roll_back(
        repo, [staged_new, untracked], project_id='proj', label='census-2026-07-14',
    )

    assert rb.restored is True
    assert not (repo / staged_new).exists()
    assert not (repo / untracked).exists()
    assert _git(repo, 'ls-files', '--', staged_new, untracked) == ''
    assert rb.quarantine_dir is not None
    assert (rb.quarantine_dir / staged_new).read_bytes() == b'# report\n'
    assert (rb.quarantine_dir / untracked).read_bytes() == b'[]\n'


def test_roll_back_leaves_every_other_path_alone(tmp_path):
    repo = _repo(tmp_path)
    (repo / 'unrelated-new.txt').write_bytes(b'staged elsewhere\n')
    _git(repo, 'add', 'unrelated-new.txt')
    (repo / _CENSUS_STATE).write_bytes(b'someone else is editing this\n')
    (repo / _CODEBOOK).write_bytes(_REFUSED)

    rb = unlanded.roll_back(repo, [_CODEBOOK], project_id='proj', label='trickle-2026-07-13')

    assert rb.restored is True
    assert _porcelain(repo, 'unrelated-new.txt') == 'A  unrelated-new.txt\n'
    assert (repo / 'unrelated-new.txt').read_bytes() == b'staged elsewhere\n'
    assert _porcelain(repo, _CENSUS_STATE) == f' M {_CENSUS_STATE}\n'
    assert (repo / _CENSUS_STATE).read_bytes() == b'someone else is editing this\n'


def test_roll_back_never_touches_the_checkout_when_the_quarantine_cannot_be_written(
    tmp_path, monkeypatch,
):
    repo = _repo(tmp_path)
    (repo / _CODEBOOK).write_bytes(_REFUSED)
    blocker = tmp_path / 'a-regular-file'
    blocker.write_text('not a directory\n')
    monkeypatch.setenv(trickle_state.STATE_ROOT_ENV, str(blocker / 'root'))

    rb = unlanded.roll_back(repo, [_CODEBOOK], project_id='proj', label='trickle-2026-07-13')

    assert rb.restored is False
    assert rb.quarantine_dir is None
    assert rb.failure
    assert (repo / _CODEBOOK).read_bytes() == _REFUSED


def test_roll_back_keeps_the_quarantine_when_the_restore_fails(tmp_path):
    not_a_repo = tmp_path / 'not-a-repo'
    not_a_repo.mkdir()
    (not_a_repo / 'written.yaml').write_bytes(_REFUSED)

    rb = unlanded.roll_back(
        not_a_repo, ['written.yaml'], project_id='proj', label='trickle-2026-07-13',
    )

    assert rb.restored is False
    assert rb.failure
    assert rb.quarantine_dir is not None
    assert (rb.quarantine_dir / 'written.yaml').read_bytes() == _REFUSED
    assert (not_a_repo / 'written.yaml').read_bytes() == _REFUSED
    assert str(rb.quarantine_dir) in rb.describe()
    assert rb.failure in rb.describe()


def test_roll_back_returns_incomplete_when_git_hangs(tmp_path, monkeypatch):
    repo = _repo(tmp_path)
    (repo / _CODEBOOK).write_bytes(_REFUSED)
    hanging_bin = tmp_path / 'hanging-bin'
    hanging_bin.mkdir()
    hanging_git = hanging_bin / 'git'
    hanging_git.write_text('#!/bin/sh\nexec sleep 30\n')
    hanging_git.chmod(0o755)
    monkeypatch.setenv('PATH', f'{hanging_bin}:{os.environ["PATH"]}')
    monkeypatch.setattr(unlanded, 'GIT_TIMEOUT_SECONDS', 0.5)

    started = time.monotonic()
    rb = unlanded.roll_back(repo, [_CODEBOOK], project_id='proj', label='trickle-2026-07-13')

    assert time.monotonic() - started < 10
    assert rb.restored is False
    assert rb.failure is not None and 'timed out' in rb.failure
    assert rb.quarantine_dir is not None
    assert (rb.quarantine_dir / _CODEBOOK).read_bytes() == _REFUSED


def test_roll_back_refuses_a_path_outside_the_repo(tmp_path):
    repo = _repo(tmp_path)
    (repo / _CODEBOOK).write_bytes(_REFUSED)
    outside = tmp_path / 'outside.txt'
    outside.write_bytes(b'not the repo\'s\n')

    rb = unlanded.roll_back(
        repo, [_CODEBOOK, outside], project_id='proj', label='trickle-2026-07-13',
    )

    assert rb.restored is False
    assert outside.read_bytes() == b'not the repo\'s\n'
    assert (repo / _CODEBOOK).read_bytes() == _REFUSED


def test_quarantine_root_lives_in_the_per_project_legibility_state_dir(
    tmp_path, monkeypatch,
):
    state_root = tmp_path / 'state-root'
    monkeypatch.setenv(trickle_state.STATE_ROOT_ENV, str(state_root))

    root = unlanded.quarantine_root('p')

    assert root.parent == trickle_state.trickle_state_path('p').parent
    assert root.is_relative_to(state_root)


def test_describe_names_the_quarantine_and_the_restored_paths(tmp_path):
    repo = _repo(tmp_path)
    (repo / _CODEBOOK).write_bytes(_REFUSED)
    (repo / _CENSUS_STATE).write_bytes(_REFUSED)

    rb = unlanded.roll_back(
        repo, [_CODEBOOK, _CENSUS_STATE], project_id='proj', label='census-2026-07-14',
    )

    assert rb.restored is True
    description = rb.describe()
    assert str(rb.quarantine_dir) in description
    assert _CODEBOOK in description
    assert _CENSUS_STATE in description


# ---------------------------------------------------------------------------
# commit_only — the one scoped committer the trickle and the census land with
# ---------------------------------------------------------------------------

_REPORT = 'plans/confusion-census-2026-07-14.md'


def _no_sleep(seconds: float) -> None:
    pass


def _lock_released_on_second_backoff(repo: Path):
    """Hold *repo*'s index.lock now, and return the ``sleep`` seam that
    releases it on its 2nd call: attempts 1 and 2 meet the lock, attempt 3
    does not."""
    lock = repo / '.git' / 'index.lock'
    lock.touch()
    backoffs: list[float] = []

    def sleep(seconds: float) -> None:
        backoffs.append(seconds)
        if len(backoffs) == 2:
            lock.unlink()

    return lock, sleep


def _committed_paths(repo: Path) -> set[str]:
    return set(_git(repo, 'show', '--name-only', '--format=', 'HEAD').split())


def _completed(args, returncode: int, *, stdout: str = '', stderr: str = ''):
    return subprocess.CompletedProcess(args, returncode, stdout=stdout, stderr=stderr)


def test_commit_only_commits_only_the_named_path(tmp_path):
    repo = _repo(tmp_path)
    (repo / _CODEBOOK).write_bytes(b'version: 2\nentries: []\ncandidates: []\n')
    (repo / 'scratch.txt').write_text('unrelated wip\n')

    result = unlanded.commit_only(repo, [_CODEBOOK], 'legibility: nightly sightings')

    assert result.ok is True
    assert result.sha == _git(repo, 'rev-parse', 'HEAD').strip()
    assert result.attempts == 1
    assert _committed_paths(repo) == {_CODEBOOK}
    assert _porcelain(repo, 'scratch.txt') == '?? scratch.txt\n'


def test_commit_only_reports_a_no_op_without_retrying(tmp_path):
    repo = _repo(tmp_path)
    head = _git(repo, 'rev-parse', 'HEAD')

    result = unlanded.commit_only(repo, [_CODEBOOK], 'legibility: no-op', sleep=_no_sleep)

    assert result.ok is False
    assert result.sha is None
    assert result.attempts == 1
    assert 'nothing' in result.stderr
    assert _git(repo, 'rev-parse', 'HEAD') == head


def test_commit_only_retries_a_ref_lock_then_lands(tmp_path):
    commits = []

    def runner(args, **kwargs):
        if 'commit' in args:
            commits.append(args)
            if len(commits) < 3:
                return _completed(args, 1, stderr='fatal: cannot lock ref ...')
            return _completed(args, 0)
        if 'rev-parse' in args:
            return _completed(args, 0, stdout='deadbeef\n')
        raise AssertionError(f'unexpected git invocation: {args}')

    result = unlanded.commit_only(tmp_path, [_CODEBOOK], 'msg', runner=runner, sleep=_no_sleep)

    assert (result.ok, result.sha, result.attempts) == (True, 'deadbeef', 3)
    assert len(commits) == 3


def test_commit_only_gives_up_on_a_persistent_lock_after_its_bound(tmp_path):
    """INV-4 storm escape: a lock that never clears costs a bounded number of
    attempts, with one backoff between each pair."""
    backoffs: list[float] = []

    def runner(args, **kwargs):
        if 'commit' in args:
            return _completed(args, 128, stderr="fatal: Unable to create '.git/index.lock'")
        raise AssertionError(f'unexpected git invocation: {args}')

    result = unlanded.commit_only(
        tmp_path, [_CODEBOOK], 'msg', runner=runner, sleep=backoffs.append,
    )

    assert result.ok is False
    assert result.attempts == unlanded.COMMIT_ATTEMPTS == 5
    assert backoffs == [unlanded.COMMIT_BACKOFF_SECONDS] * (unlanded.COMMIT_ATTEMPTS - 1)
    assert 'index.lock' in result.stderr


def test_commit_only_never_stashes(tmp_path):
    repo = _repo(tmp_path)
    (repo / _CODEBOOK).write_bytes(_REFUSED)
    (repo / 'plans').mkdir()
    (repo / _REPORT).write_text('# report\n')
    calls = []

    def spy(args, **kwargs):
        calls.append(args)
        return subprocess.run(args, **kwargs)

    result = unlanded.commit_only(repo, [_CODEBOOK, _REPORT], 'msg', runner=spy)

    assert result.ok is True
    assert calls
    assert not any('stash' in call for call in calls)


def test_commit_only_outlasts_an_index_lock_on_a_tracked_path(tmp_path):
    repo = _repo(tmp_path)
    (repo / _CODEBOOK).write_bytes(_REFUSED)
    lock, sleep = _lock_released_on_second_backoff(repo)

    result = unlanded.commit_only(repo, [_CODEBOOK], 'legibility trickle', sleep=sleep)

    assert (result.ok, result.attempts) == (True, 3)
    assert not lock.exists()
    assert _head_bytes(repo, _CODEBOOK) == _REFUSED
    assert _porcelain(repo) == ''


def test_commit_only_outlasts_an_index_lock_on_a_new_path(tmp_path):
    """The census shape: every census commits a NEW dated report beside a
    tracked file. Under the lock, git rejects the unknown pathspec before it
    reaches the lock, and the scoped add that makes the path known fails on the
    lock; both must be retried, not taken as final."""
    repo = _repo(tmp_path)
    (repo / _CENSUS_STATE).write_bytes(b'{"last_census_at": "2026-07-14"}\n')
    (repo / 'plans').mkdir()
    (repo / _REPORT).write_text('# report\n')
    lock, sleep = _lock_released_on_second_backoff(repo)

    result = unlanded.commit_only(
        repo, [_CENSUS_STATE, _REPORT], 'legibility census 2026-07-14', sleep=sleep,
    )

    assert (result.ok, result.attempts) == (True, 3)
    assert not lock.exists()
    assert _committed_paths(repo) == {_CENSUS_STATE, _REPORT}
    assert _porcelain(repo) == ''


def test_commit_only_lands_a_new_path_within_one_attempt(tmp_path):
    repo = _repo(tmp_path)
    (repo / 'plans').mkdir()
    (repo / _REPORT).write_text('# report\n')

    result = unlanded.commit_only(repo, [_REPORT], 'msg', sleep=_no_sleep)

    assert (result.ok, result.attempts) == (True, 1)
    assert _committed_paths(repo) == {_REPORT}


def test_commit_only_reports_git_that_cannot_be_launched(tmp_path):
    def runner(args, **kwargs):
        raise FileNotFoundError(2, 'No such file or directory', 'git')

    result = unlanded.commit_only(tmp_path, [_CODEBOOK], 'msg', runner=runner, sleep=_no_sleep)

    assert (result.ok, result.sha, result.attempts) == (False, None, 1)
    assert 'No such file or directory' in result.stderr


def test_commit_only_reports_a_refusing_pre_commit_hook_without_retrying(tmp_path):
    repo = _repo(tmp_path)
    _install_pre_commit_hook(tmp_path, repo, 'echo "cited-test-path gate: refused"\nexit 1')
    (repo / _CODEBOOK).write_bytes(_REFUSED)
    head = _git(repo, 'rev-parse', 'HEAD')

    result = unlanded.commit_only(repo, [_CODEBOOK], 'msg', sleep=_no_sleep)

    assert (result.ok, result.attempts) == (False, 1)
    assert 'cited-test-path gate' in result.stderr
    assert _git(repo, 'rev-parse', 'HEAD') == head


def _install_pre_commit_hook(tmp_path: Path, repo: Path, script: str) -> None:
    hooks = tmp_path / 'hooks'
    hooks.mkdir()
    hook = hooks / 'pre-commit'
    hook.write_text(f'#!/bin/sh\n{script}\n')
    hook.chmod(0o755)
    _git(repo, 'config', 'core.hooksPath', str(hooks))


def test_commit_only_reports_both_streams_of_a_failed_commit(tmp_path):
    """git routes hook output to stderr and its own no-op verdict to stdout; the
    diagnostic an operator is escalated must carry both."""
    repo = _repo(tmp_path)
    _install_pre_commit_hook(tmp_path, repo, 'echo "pyright: 3 warnings" >&2\nexit 0')

    result = unlanded.commit_only(repo, [_CODEBOOK], 'legibility: no-op', sleep=_no_sleep)

    assert (result.ok, result.attempts) == (False, 1)
    assert 'pyright: 3 warnings' in result.stderr
    assert 'nothing to commit' in result.stderr


def test_commit_only_does_not_retry_a_hook_refusal_that_mentions_creating(tmp_path):
    """Only git's own lock failure is contention. A hook refusal is final even
    when its text happens to say "unable to create": every retry re-runs the
    whole hook."""
    repo = _repo(tmp_path)
    _install_pre_commit_hook(
        tmp_path, repo, 'echo "error: unable to create the coverage report"\nexit 1',
    )
    (repo / _CODEBOOK).write_bytes(_REFUSED)
    backoffs: list[float] = []

    result = unlanded.commit_only(repo, [_CODEBOOK], 'msg', sleep=backoffs.append)

    assert (result.ok, result.attempts) == (False, 1)
    assert backoffs == []
    assert 'unable to create the coverage report' in result.stderr


@pytest.mark.parametrize('field', ['ok', 'sha', 'stderr', 'attempts'])
def test_a_commit_result_is_frozen(field):
    result = unlanded.CommitResult(ok=True, sha='f' * 40, stderr='', attempts=1)

    with pytest.raises(dataclasses.FrozenInstanceError):
        setattr(result, field, None)
