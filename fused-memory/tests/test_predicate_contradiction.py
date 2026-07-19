"""Tests for the predicate-contradiction filing convention (task 2779).

Gives reconciliation stages a reusable, durable way to file a one-off
``task_kind='deterministic'`` task with ``before_done.kind='predicate'`` when a
recon-stage contradiction can only be settled by live command execution (a
specific pytest test, a git grep/log check) — instead of ad-hoc Mem0
investigation notes/suppression records that silently age across cycles
(motivating precedent: task 2643).

Assertions are pinned to runtime return values (the builder's returned
dataclass/dict, the reference runner's exit codes) and stable load-bearing
substrings within the rendered prompt section — NOT verbatim prompt-text
equality — mirroring the test_recon_self_model.py convention.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

# Repo root: this file is <root>/fused-memory/tests/test_predicate_contradiction.py
_REPO_ROOT = Path(__file__).parents[2]
_SCRIPT = _REPO_ROOT / 'scripts' / 'recon_predicate_check.sh'

# Committed username/email so the hermetic tmp-repo commit does not depend on a
# global git identity being configured in the test environment.
_GIT_ENV_ARGS = [
    '-c',
    'user.email=recon-test@example.com',
    '-c',
    'user.name=recon-test',
    '-c',
    'commit.gpgsign=false',
]


def _init_tmp_git_repo(tmp_path: Path, *, sentinel: str) -> Path:
    """Create a hermetic git repo under *tmp_path* containing a committed file
    whose contents include *sentinel*. Returns the repo path."""
    subprocess.run(['git', 'init', '-q'], cwd=tmp_path, check=True, capture_output=True)
    (tmp_path / 'tracked.txt').write_text(f'preamble\n{sentinel}\ntrailer\n')
    subprocess.run(['git', 'add', 'tracked.txt'], cwd=tmp_path, check=True, capture_output=True)
    subprocess.run(
        ['git', *_GIT_ENV_ARGS, 'commit', '-q', '-m', 'seed'],
        cwd=tmp_path,
        check=True,
        capture_output=True,
    )
    return tmp_path


class TestReconPredicateCheckScript:
    """scripts/recon_predicate_check.sh is a committed, executable reference
    runner whose exit codes honour the predicate contract (0 = settled as
    expected -> task done; non-zero = mismatch -> milestone_check_failed)."""

    def test_script_exists_and_is_executable(self):
        assert _SCRIPT.exists(), f'reference runner missing at {_SCRIPT}'
        assert os.access(_SCRIPT, os.X_OK), f'reference runner not executable: {_SCRIPT}'

    def test_no_args_exits_2(self):
        proc = subprocess.run([str(_SCRIPT)], capture_output=True, text=True)
        assert proc.returncode == 2, proc.stderr

    def test_unknown_mode_exits_2(self):
        proc = subprocess.run([str(_SCRIPT), '--bogus-mode'], capture_output=True, text=True)
        assert proc.returncode == 2, proc.stderr

    def test_git_grep_found_exits_0(self, tmp_path):
        sentinel = 'RECON_SENTINEL_PRESENT'
        repo = _init_tmp_git_repo(tmp_path, sentinel=sentinel)
        proc = subprocess.run(
            [str(_SCRIPT), '--git-grep', sentinel],
            cwd=repo,
            capture_output=True,
            text=True,
        )
        assert proc.returncode == 0, proc.stderr

    def test_git_grep_absent_exits_1(self, tmp_path):
        repo = _init_tmp_git_repo(tmp_path, sentinel='RECON_SENTINEL_PRESENT')
        proc = subprocess.run(
            [str(_SCRIPT), '--git-grep', 'absent-6f1e2d3c-0000-4000-8000-000000000000'],
            cwd=repo,
            capture_output=True,
            text=True,
        )
        assert proc.returncode == 1, proc.stderr
