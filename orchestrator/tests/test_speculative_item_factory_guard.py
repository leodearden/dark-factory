"""I3 factory grep-guard for SpeculativeItem construction sites (task 1990).

step-5  RED   — no production site is marked yet
step-6  GREEN — the 15 legit factory sites carry the marker; the 2 former
                CAS-loop hand-rebuilds are now dataclasses.replace(...) calls
                and no longer match the grep pattern at all.

step-9  RED   — a stale editor/tool temp-backup file
                (test_merge_queue_concurrent_verify.py.tmp.2390297.d4b2fa0bada8)
                was accidentally committed by a WIP-snapshot commit.
step-10 GREEN — the stray file is removed and .gitignore gains a rule, but
                since an already-tracked (or force-added) backup file
                bypasses .gitignore, this repo-hygiene test is the durable
                recurrence guard.
"""

from __future__ import annotations

import fnmatch
import subprocess
from pathlib import Path

import pytest

_MARKER = '# spec-factory'
_SCOPED_PATH = 'orchestrator/src/orchestrator/'
_BACKUP_GLOBS = ('*.tmp.*', '*.py.tmp.*', '*.orig', '*.rej', '*.bak', '*.swp', '*~')


def test_every_speculative_item_construction_in_src_is_marked(repo_root: Path | None) -> None:
    """Every `SpeculativeItem(` construction under orchestrator/src/ must carry
    the `# spec-factory` marker on its construction line.

    Guards I3: a future hand-rebuild that bypasses `dataclasses.replace(...)`
    and reconstructs a SpeculativeItem field-by-field (the task-1928 bug
    class) shows up here as an un-marked hit.
    """
    if repo_root is None:
        pytest.skip('not running inside a git checkout')

    result = subprocess.run(
        ['git', 'grep', '-n', 'SpeculativeItem(', '--', _SCOPED_PATH],
        cwd=repo_root,
        capture_output=True,
        text=True,
    )
    if result.returncode not in (0, 1):
        pytest.fail(f'git grep failed (exit {result.returncode}): {result.stderr}')

    lines = [ln for ln in result.stdout.splitlines() if ln.strip()]
    unmarked = [ln for ln in lines if _MARKER not in ln]
    assert not unmarked, (
        f'Found {len(unmarked)} SpeculativeItem( construction(s) in {_SCOPED_PATH} '
        f'without the {_MARKER!r} factory marker (I3 — every src construction must '
        f'be a whitelisted factory site or dataclasses.replace):\n' + '\n'.join(unmarked)
    )


def test_no_tracked_editor_backup_files_under_orchestrator(repo_root: Path | None) -> None:
    """No editor/tool backup file should ever be git-tracked under orchestrator/.

    A `.gitignore` rule alone only stops an *untracked* backup from being
    casually `git add`ed — it does nothing for a file that is already
    tracked or was force-added (`git add -f`), which is exactly how
    ``test_merge_queue_concurrent_verify.py.tmp.2390297.d4b2fa0bada8`` (a
    ~4193-line stale near-duplicate) slipped in via a WIP-snapshot commit.
    This test scans the tracked file list itself so re-adding a similar
    backup file is caught regardless of .gitignore.
    """
    if repo_root is None:
        pytest.skip('not running inside a git checkout')

    result = subprocess.run(
        ['git', 'ls-files', '--', 'orchestrator/'],
        cwd=repo_root,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        pytest.fail(f'git ls-files failed (exit {result.returncode}): {result.stderr}')

    tracked = [ln for ln in result.stdout.splitlines() if ln.strip()]
    offenders = [
        path
        for path in tracked
        if any(fnmatch.fnmatch(Path(path).name, pattern) for pattern in _BACKUP_GLOBS)
    ]
    assert not offenders, (
        f'Found {len(offenders)} tracked editor/tool backup file(s) under orchestrator/ '
        f'matching {_BACKUP_GLOBS!r} (these must never be committed):\n' + '\n'.join(offenders)
    )
