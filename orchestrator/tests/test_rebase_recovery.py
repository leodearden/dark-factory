"""Tests for ``orchestrator.rebase_recovery`` — guarded rebase/merge abort (task 4797).

Two defects motivate the module under test, and both were reproduced
first-hand in throwaway repos on git 2.43.0 before any code was written:

* A MERGE_RR record naming a conflict id whose ``rr-cache/<id>`` directory is
  absent makes ``git rebase --abort`` die in ``rerere_clear()`` — measured
  ``Segmentation fault (core dumped)``, rc 139, with the rebase state left
  fully in place and a fresh ``MERGE_RR.lock`` created.
* A stale ``MERGE_RR.lock`` with a perfectly INTACT rr-cache makes the same
  abort fail rc 128 with git's "Another git process seems to be running"
  advice.  This is an independent failure, not the crash's residue.

Nothing here asserts the segfault.  It is an upstream git bug that this task
puts out of scope, so pinning rc 139 would turn a future git patch into a red
suite — the fix would read as the regression.  What is asserted is the
version-independent post-condition: the preflight DETECTS the dangling id, and
the guarded abort then returns rc 0 with the evidence preserved.

The paired negative control matters as much as the positive case.  An "abort
works" assertion against a HEALTHY worktree passes vacuously — the unguarded
abort works there too — so every behavioural claim below is made against a
deliberately dangling fixture with an intact-fixture control beside it.

ISOLATION (esc-3072-3).  This module builds real conflicted repositories and
deliberately deletes ``rr-cache`` directories, which is precisely the shape of
write that once landed in a live task worktree: git's repository discovery
walks UP, and pytest's basetemp can sit inside one.  Both layers of the house
defence are applied to every git subprocess spawned here —
:func:`assert_isolated_git_repo` first (pure filesystem, so a rejected call
writes nothing anywhere), then ``env=`` :func:`git_env_with_ceiling` (so escape
is impossible at the git level even if the pre-flight is refactored away).
"""

from __future__ import annotations

import subprocess
from pathlib import Path

from _orch_helpers import assert_isolated_git_repo, git_env_with_ceiling

# ---------------------------------------------------------------------------
# Real-git fixture scaffolding
# ---------------------------------------------------------------------------

_BASE = 'line1\nline2\nline3\n'
_FEATURE = 'line1\nFEATURE\nline3\n'
_MAIN = 'line1\nMAIN\nline3\n'


def _git(repo: Path, *args: str) -> subprocess.CompletedProcess:
    """Run one git command inside *repo*, under both isolation layers."""
    assert_isolated_git_repo(repo)
    return subprocess.run(
        ['git', *args],
        cwd=str(repo),
        capture_output=True,
        text=True,
        env=git_env_with_ceiling(repo),
    )


def _git_ok(repo: Path, *args: str) -> str:
    proc = _git(repo, *args)
    assert proc.returncode == 0, f'git {args!r} failed: {proc.stderr}'
    return proc.stdout


def build_mid_rebase_repo(root: Path, name: str = 'repo') -> tuple[Path, str]:
    """Build an isolated repo left mid-rebase with a populated MERGE_RR.

    Creates a deterministic two-branch content conflict on one line of one
    file, then rebases ``feature`` onto ``main`` so the rebase stops on that
    conflict.  rerere is enabled, so git writes ``MERGE_RR`` naming the
    conflict id and a backing ``rr-cache/<id>/`` directory — the substrate the
    dangling-ref cases manufacture by deleting.

    Returns ``(repo, conflict_id)``.  The conflict id is the LITERAL token from
    MERGE_RR, carrying any rerere variant suffix verbatim.
    """
    repo = root / name
    repo.mkdir(parents=True)

    # The one call that cannot pre-assert: the directory is not a repository
    # until this command makes it one.  ``git init`` creates a repo at the path
    # given rather than reusing an enclosing one, and the ceiling is applied
    # regardless; the assertion inside ``_git`` covers every call after it.
    init = subprocess.run(
        ['git', 'init', '-q', '-b', 'main', '.'],
        cwd=str(repo), capture_output=True, text=True,
        env=git_env_with_ceiling(repo),
    )
    assert init.returncode == 0, f'git init failed: {init.stderr}'
    assert_isolated_git_repo(repo)

    _git_ok(repo, 'config', 'user.email', 'rebase-recovery@test.invalid')
    _git_ok(repo, 'config', 'user.name', 'Rebase Recovery Test')
    _git_ok(repo, 'config', 'commit.gpgsign', 'false')
    _git_ok(repo, 'config', 'rerere.enabled', 'true')
    _git_ok(repo, 'config', 'rerere.autoupdate', 'true')

    (repo / 'f.txt').write_text(_BASE)
    _git_ok(repo, 'add', 'f.txt')
    _git_ok(repo, 'commit', '-qm', 'base')

    _git_ok(repo, 'checkout', '-qb', 'feature')
    (repo / 'f.txt').write_text(_FEATURE)
    _git_ok(repo, 'commit', '-qam', 'feature edit')

    _git_ok(repo, 'checkout', '-q', 'main')
    (repo / 'f.txt').write_text(_MAIN)
    _git_ok(repo, 'commit', '-qam', 'main edit')

    _git_ok(repo, 'checkout', '-q', 'feature')
    rebase = _git(repo, 'rebase', 'main')
    assert rebase.returncode != 0, 'fixture expected a rebase conflict'

    git_dir = repo / '.git'
    assert (git_dir / 'rebase-merge').is_dir(), 'fixture expected mid-rebase state'

    merge_rr = (git_dir / 'MERGE_RR').read_bytes()
    conflict_id = merge_rr.split(b'\x00')[0].split(b'\t')[0].decode()
    assert (git_dir / 'rr-cache' / conflict_id).is_dir(), (
        'fixture expected a backing rr-cache entry'
    )
    return repo, conflict_id
