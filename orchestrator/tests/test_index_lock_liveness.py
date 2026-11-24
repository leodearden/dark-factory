"""Who holds a work tree's index.lock, and what is safe to tell an operator (task 5229).

Which processes a scan LISTS is pinned with real processes in real repos:
the property is what ``/proc`` says about them.  Whether a scan is CONFIRMED,
and every verdict drawn from one, is pinned with a constructed proc root
(``_constructed_proc_table``) instead, because on the real ``/proc`` it
depends on the whole host: inside an agent's landlock sandbox the cwd of
every process outside the sandbox's domain is unreadable (EACCES, measured
2026-11-24), so a real scan there is unconfirmed whenever another agent's git
happens to be running.  The scan's own children are always readable.
"""
from __future__ import annotations

import contextlib
import re
import subprocess
import time
from collections.abc import Iterator
from pathlib import Path

import pytest
from _backdated_index_lock import REIFY_LOCK_AGE_S, index_lock_written_ago
from _blocked_commit_fixture import commit_blocked_in_pre_commit_hook, hermetic_git_env
from _constructed_proc_table import add_git_process, add_process, empty_proc_table
from _git_fixtures import seed_repo

from orchestrator.index_lock_liveness import (
    GitProcess,
    GitProcessScan,
    IndexLockDiagnosis,
    IndexLockVerdict,
    diagnose_index_lock,
    scan_index_lock_holders,
)

_EXIT_TIMEOUT_S = 30.0


@contextlib.contextmanager
def _waiting_on_stdin(argv: list[str], cwd: Path) -> Iterator[subprocess.Popen[bytes]]:
    """*argv* running in *cwd*, blocked reading stdin until the block exits.

    ``Popen`` returns only after ``exec``, so the process's comm and cwd are
    already what the scan will read: no readiness wait is needed.
    """
    proc = subprocess.Popen(
        argv, cwd=cwd, env=hermetic_git_env(cwd),
        stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    try:
        yield proc
    finally:
        assert proc.stdin is not None
        proc.stdin.close()
        try:
            proc.wait(timeout=_EXIT_TIMEOUT_S)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=_EXIT_TIMEOUT_S)


def _git_dir(work_tree: Path) -> Path:
    return Path(subprocess.run(
        ['git', 'rev-parse', '--absolute-git-dir'],
        cwd=work_tree, env=hermetic_git_env(work_tree),
        check=True, capture_output=True, text=True,
    ).stdout.strip())


def _scan_against_a_fresh_lock(work_tree: Path) -> GitProcessScan:
    """A scan of *work_tree* bounded by an index lock written NOW, in its own git dir.

    Every child already running started before that lock, so the start-time
    bound admits all of them and only their placement is under test.
    """
    lock = _git_dir(work_tree) / 'index.lock'
    lock.write_text('')
    try:
        return scan_index_lock_holders(lock, work_tree)
    finally:
        lock.unlink()


def _listed_pids(work_tree: Path) -> set[int]:
    return {process.pid for process in _scan_against_a_fresh_lock(work_tree).processes}


def _nested_worktree(repo: Path) -> Path:
    nested = repo / '.worktrees' / 'wt'
    subprocess.run(
        ['git', 'worktree', 'add', '-q', '-b', 'nested', str(nested)],
        cwd=repo, env=hermetic_git_env(repo), check=True, capture_output=True,
    )
    return nested


class TestScanPlacesGitProcessesInTheWorkTree:
    def test_a_live_git_process_in_the_work_tree_is_reported_until_it_exits(self, tmp_path):
        repo = seed_repo(tmp_path / 'repo')
        with _waiting_on_stdin(['git', 'hash-object', '--stdin'], repo) as proc:
            scan = _scan_against_a_fresh_lock(repo)
            [process] = [p for p in scan.processes if p.pid == proc.pid]
            assert 'hash-object' in process.command
        assert proc.pid not in _listed_pids(repo)

    def test_a_commit_blocked_in_its_pre_commit_hook_is_reported(self, tmp_path):
        repo = seed_repo(tmp_path / 'repo')
        (repo / 'README.md').write_text('# dirtied so --only has a change\n')
        with commit_blocked_in_pre_commit_hook(repo, 'README.md') as git_pid:
            scan = scan_index_lock_holders(repo / '.git' / 'index.lock', repo)
            [process] = [p for p in scan.processes if p.pid == git_pid]
            assert 'commit' in process.command

    def test_a_git_process_in_a_subdirectory_of_the_work_tree_is_reported(self, tmp_path):
        repo = seed_repo(tmp_path / 'repo')
        subdir = repo / 'sub' / 'deeper'
        subdir.mkdir(parents=True)
        with _waiting_on_stdin(['git', 'hash-object', '--stdin'], subdir) as proc:
            assert proc.pid in _listed_pids(repo)

    def test_a_git_process_in_a_nested_linked_worktree_is_not_reported(self, tmp_path):
        repo = seed_repo(tmp_path / 'repo')
        nested = _nested_worktree(repo)
        with _waiting_on_stdin(['git', 'hash-object', '--stdin'], nested) as proc:
            assert proc.pid not in _listed_pids(repo)
            assert proc.pid in _listed_pids(nested)

    def test_a_git_process_below_a_nested_worktree_root_is_not_reported(self, tmp_path):
        repo = seed_repo(tmp_path / 'repo')
        below_nested = _nested_worktree(repo) / 'sub'
        below_nested.mkdir()
        with _waiting_on_stdin(['git', 'hash-object', '--stdin'], below_nested) as proc:
            assert proc.pid not in _listed_pids(repo)

    def test_a_non_git_process_in_the_work_tree_is_not_reported(self, tmp_path):
        repo = seed_repo(tmp_path / 'repo')
        with _waiting_on_stdin(['cat'], repo) as proc:
            assert proc.pid not in _listed_pids(repo)


class TestScanConfirmation:
    """Only a scan that placed every git process that could have written the lock is confirmed."""

    def test_an_unenumerable_proc_root_is_unknown_not_empty(self, tmp_path):
        work_tree = tmp_path / 'wt'
        lock = index_lock_written_ago(work_tree, 0.0)
        scan = scan_index_lock_holders(lock, work_tree, proc_root=tmp_path / 'absent')
        assert scan == GitProcessScan((), confirmed=False)

    def test_an_absent_lock_is_unknown_not_unheld(self, tmp_path):
        work_tree = tmp_path / 'wt'
        work_tree.mkdir()
        proc_root = empty_proc_table(tmp_path)
        scan = scan_index_lock_holders(
            work_tree / '.git' / 'index.lock', work_tree, proc_root=proc_root,
        )
        assert scan == GitProcessScan((), confirmed=False)

    def test_a_process_table_with_no_git_process_is_a_confirmed_empty_scan(self, tmp_path):
        work_tree = tmp_path / 'wt'
        lock = index_lock_written_ago(work_tree, 0.0)
        proc_root = empty_proc_table(tmp_path)
        add_process(proc_root, 4242, comm='cat', cwd=None, command='cat', started_at=time.time())
        scan = scan_index_lock_holders(lock, work_tree, proc_root=proc_root)
        assert scan == GitProcessScan((), confirmed=True)

    def test_an_unplaceable_git_that_predates_the_lock_makes_the_scan_unconfirmed(
        self, tmp_path,
    ):
        work_tree = tmp_path / 'wt'
        lock = index_lock_written_ago(work_tree, REIFY_LOCK_AGE_S)
        proc_root = empty_proc_table(tmp_path)
        add_git_process(
            proc_root, 4242, cwd=None, command='git status',
            started_at=lock.stat().st_mtime - 10,
        )
        scan = scan_index_lock_holders(lock, work_tree, proc_root=proc_root)
        assert scan == GitProcessScan((), confirmed=False)

    def test_an_unplaceable_git_with_an_unreadable_start_makes_the_scan_unconfirmed(
        self, tmp_path,
    ):
        work_tree = tmp_path / 'wt'
        lock = index_lock_written_ago(work_tree, REIFY_LOCK_AGE_S)
        proc_root = empty_proc_table(tmp_path)
        add_git_process(proc_root, 4242, cwd=None, command='git status', started_at=None)
        scan = scan_index_lock_holders(lock, work_tree, proc_root=proc_root)
        assert scan == GitProcessScan((), confirmed=False)

    def test_an_unplaceable_git_started_after_the_lock_leaves_the_scan_confirmed(
        self, tmp_path,
    ):
        """A foreign long-lived git whose cwd is ptrace-gated cannot have
        written a lock older than itself, so it must not keep a crashed
        leftover from being proven."""
        work_tree = tmp_path / 'wt'
        lock = index_lock_written_ago(work_tree, REIFY_LOCK_AGE_S)
        proc_root = empty_proc_table(tmp_path)
        add_git_process(
            proc_root, 4_000_000, cwd=None, command='git fetch --all', started_at=time.time(),
        )
        scan = scan_index_lock_holders(lock, work_tree, proc_root=proc_root)
        assert scan == GitProcessScan((), confirmed=True)
        assert _diagnose(REIFY_LOCK_AGE_S, scan).verdict is IndexLockVerdict.CRASHED_LEFTOVER


class TestScanIndexLockHolders:
    """Only a git process that could have WRITTEN the lock is a holder.

    git creates index.lock after it starts, so a git process that started
    after the lock's mtime cannot be its creator: the merge worker's own
    read-only ``git cherry`` / ``git log`` children are the common case.
    """

    def test_a_git_process_started_before_the_lock_is_listed(self, tmp_path):
        repo = seed_repo(tmp_path / 'repo')
        with _waiting_on_stdin(['git', 'hash-object', '--stdin'], repo) as proc:
            lock = index_lock_written_ago(repo, 0.0)
            scan = scan_index_lock_holders(lock, repo)
            assert proc.pid in {process.pid for process in scan.processes}

    def test_a_git_process_started_after_an_old_lock_is_not_listed(self, tmp_path):
        """The backdate must dwarf the start-time slack: btime is truncated to
        whole seconds, so a computed start reads up to a second early."""
        repo = seed_repo(tmp_path / 'repo')
        lock = index_lock_written_ago(repo, 600.0)
        with _waiting_on_stdin(['git', 'hash-object', '--stdin'], repo) as proc:
            scan = scan_index_lock_holders(lock, repo)
            assert proc.pid not in {process.pid for process in scan.processes}

    def test_a_later_read_only_git_does_not_mask_a_crashed_leftover(self, tmp_path):
        work_tree = tmp_path / 'wt'
        lock = index_lock_written_ago(work_tree, REIFY_LOCK_AGE_S)
        proc_root = empty_proc_table(tmp_path)
        add_git_process(
            proc_root, 4_000_000, cwd=work_tree,
            command='git cherry main abc123', started_at=time.time(),
        )
        scan = scan_index_lock_holders(lock, work_tree, proc_root=proc_root)
        assert scan == GitProcessScan((), confirmed=True)
        assert _diagnose(REIFY_LOCK_AGE_S, scan).verdict is IndexLockVerdict.CRASHED_LEFTOVER

    def test_a_git_process_that_predates_the_lock_is_listed(self, tmp_path):
        work_tree = tmp_path / 'wt'
        lock = index_lock_written_ago(work_tree, REIFY_LOCK_AGE_S)
        proc_root = empty_proc_table(tmp_path)
        add_git_process(
            proc_root, 728900, cwd=work_tree, command='git commit --only plans/x.md',
            started_at=lock.stat().st_mtime - 10,
        )
        scan = scan_index_lock_holders(lock, work_tree, proc_root=proc_root)
        assert [process.pid for process in scan.processes] == [728900]
        assert _diagnose(REIFY_LOCK_AGE_S, scan).verdict is IndexLockVerdict.LIVE_HOLDER

    def test_a_git_process_with_an_unparseable_start_time_stays_listed(self, tmp_path):
        work_tree = tmp_path / 'wt'
        lock = index_lock_written_ago(work_tree, REIFY_LOCK_AGE_S)
        proc_root = empty_proc_table(tmp_path)
        add_git_process(
            proc_root, 728900, cwd=work_tree, command='git commit --only plans/x.md',
            started_at=None,
        )
        scan = scan_index_lock_holders(lock, work_tree, proc_root=proc_root)
        assert [process.pid for process in scan.processes] == [728900]

    def test_a_git_process_stays_listed_when_the_boot_time_is_unknown(self, tmp_path):
        work_tree = tmp_path / 'wt'
        lock = index_lock_written_ago(work_tree, REIFY_LOCK_AGE_S)
        proc_root = empty_proc_table(tmp_path, boot_epoch_known=False)
        add_git_process(
            proc_root, 4_000_000, cwd=work_tree,
            command='git cherry main abc123', started_at=time.time(),
        )
        scan = scan_index_lock_holders(lock, work_tree, proc_root=proc_root)
        assert [process.pid for process in scan.processes] == [4_000_000]


_LOCK = '/r/.git/index.lock'
_REIFY_HOLDER = GitProcess(728900, 'git commit --only plans/x.md')
_NOBODY = GitProcessScan((), confirmed=True)


def _diagnose(initial_age_s, scan, *, grace_s=300.0, stale_floor_s=300.0):
    return diagnose_index_lock(
        initial_age_s=initial_age_s, grace_s=grace_s,
        stale_floor_s=stale_floor_s, scan=scan,
    )


class TestDiagnoseIndexLock:
    def test_an_old_lock_with_a_live_commit_is_named_and_never_removed(self):
        diagnosis = _diagnose(3290.0, GitProcessScan((_REIFY_HOLDER,), confirmed=True))
        assert diagnosis.verdict is IndexLockVerdict.LIVE_HOLDER
        advice = diagnosis.advice(_LOCK)
        assert '728900' in advice
        assert 'git commit --only plans/x.md' in advice
        assert 'wait for that process to exit' in advice
        assert 'rm -f' not in advice

    def test_a_young_lock_with_a_live_holder_names_it_too(self):
        diagnosis = _diagnose(2.0, GitProcessScan((_REIFY_HOLDER,), confirmed=True))
        assert diagnosis.verdict is IndexLockVerdict.LIVE_HOLDER
        assert '728900' in diagnosis.advice(_LOCK)

    def test_a_live_holder_wins_over_an_unconfirmed_scan(self):
        diagnosis = _diagnose(3290.0, GitProcessScan((_REIFY_HOLDER,), confirmed=False))
        assert diagnosis.verdict is IndexLockVerdict.LIVE_HOLDER

    def test_an_old_lock_nobody_holds_is_a_crashed_leftover(self):
        diagnosis = _diagnose(3290.0, _NOBODY)
        assert diagnosis.verdict is IndexLockVerdict.CRASHED_LEFTOVER
        advice = diagnosis.advice(_LOCK)
        assert f'rm -f {_LOCK}' in advice
        assert 'no git process' in advice

    @pytest.mark.parametrize(('grace_s', 'initial_age_s'), [
        (0.0, 0.5), (0.0, 2.0), (0.0, 299.0), (0.0, 300.0), (900.0, 400.0), (900.0, 900.0),
    ])
    def test_a_lock_nobody_holds_is_unproven_until_it_clears_the_age_floor(
        self, grace_s, initial_age_s,
    ):
        diagnosis = _diagnose(initial_age_s, _NOBODY, grace_s=grace_s)
        assert diagnosis.verdict is IndexLockVerdict.UNPROVEN
        assert diagnosis.advice(_LOCK) == ''

    def test_an_unconfirmed_empty_scan_is_never_a_crashed_leftover(self):
        diagnosis = _diagnose(3600.0, GitProcessScan((), confirmed=False))
        assert diagnosis.verdict is IndexLockVerdict.UNPROVEN
        assert 'rm -f' not in diagnosis.advice(_LOCK)

    def test_no_scan_at_all_is_never_a_crashed_leftover(self):
        assert _diagnose(3600.0, None).verdict is IndexLockVerdict.UNPROVEN

    @pytest.mark.parametrize(('initial_age_s', 'grace_s'), [(None, 300.0), (3600.0, None)])
    def test_a_missing_age_fact_is_never_a_crashed_leftover(self, initial_age_s, grace_s):
        diagnosis = _diagnose(initial_age_s, _NOBODY, grace_s=grace_s)
        assert diagnosis.verdict is IndexLockVerdict.UNPROVEN

    def test_many_holders_are_capped_and_counted(self):
        holders = tuple(GitProcess(1000 + n, f'git status #{n}') for n in range(9))
        advice = _diagnose(10.0, GitProcessScan(holders, confirmed=True)).advice(_LOCK)
        assert 'pid 1000 ' in advice
        assert 'pid 1008 ' not in advice
        assert re.search(r'\+\d+ more', advice)
        assert 'wait for those processes to exit' in advice
        assert 'rm -f' not in advice


class TestIndexLockDiagnosisInvariant:
    def test_a_crashed_leftover_with_holders_is_refused(self):
        with pytest.raises(ValueError, match=r'(?s)CRASHED_LEFTOVER.*728900'):
            IndexLockDiagnosis(
                IndexLockVerdict.CRASHED_LEFTOVER, (_REIFY_HOLDER,),
                initial_age_s=3290.0, stale_after_s=300.0,
            )

    def test_a_crashed_leftover_younger_than_its_floor_is_refused(self):
        with pytest.raises(ValueError, match=r'(?s)CRASHED_LEFTOVER.*12\.0.*300\.0'):
            IndexLockDiagnosis(
                IndexLockVerdict.CRASHED_LEFTOVER, (),
                initial_age_s=12.0, stale_after_s=300.0,
            )

    def test_a_crashed_leftover_without_an_age_is_refused(self):
        with pytest.raises(ValueError, match=r'(?s)CRASHED_LEFTOVER.*None'):
            IndexLockDiagnosis(
                IndexLockVerdict.CRASHED_LEFTOVER, (),
                initial_age_s=None, stale_after_s=300.0,
            )

    def test_a_live_holder_without_holders_is_refused(self):
        with pytest.raises(ValueError, match=r'(?s)LIVE_HOLDER.*\(\)'):
            IndexLockDiagnosis(
                IndexLockVerdict.LIVE_HOLDER, (),
                initial_age_s=3290.0, stale_after_s=300.0,
            )
