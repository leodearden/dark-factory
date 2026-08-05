"""Gate-level tests for the THIRD false-positive class of
:func:`orchestrator.merge_gates._check_plan_files_touched_in_branch` —
a declared plan path that was RENAMED on main *after* the architect
declared it (task 3110).

Prior false-positive classes of this same gate:

1. Task 1587 — a ``./``-prefixed declared entry never matched git's
   canonical (prefix-free) touched-set output.  Fixed by
   ``_normalize_plan_path``.
2. Task 3004 — cross-repo declared paths (absolute paths outside
   ``project_root``) were checked against the wrong repository's history.
3. **This class** — the declared path existed when the plan was written,
   then main relocated it (``git mv`` / delete+add) before the branch was
   cut.  The branch dutifully edits the *new* path, so the declared
   pre-rename path appears in no branch commit AND is not a directory in
   the branch tree, and the gate blocks with "no commit on the branch
   touched them" — a confident claim about the branch that was never
   actually tested.

Measured case: reify task 5196 / escalation esc-5196-22.  The reify
harness-layout-consolidation programme moved ``crates/tests/*_e2e.rs``
into ``crates/tests/harness_topo/`` on main; every subsequent task that
had declared a pre-move path was blocked and escalated to a human 22
times over, each time diagnosed as "the implementation has not delivered
against the plan" when the implementation had in fact delivered exactly
what was asked, at the file's current name.

These tests exercise the gate against a REAL git repository (no
monkeypatching whatsoever), so they live in this dedicated file rather
than in the 15k-line ``test_merge_queue.py``.  They import the gate from
``orchestrator.merge_gates`` directly — not through the
``orchestrator.merge_queue`` shim — matching the ``workflow.py`` call
site's precedent of keeping hot ``merge_queue.py`` out of this task's
lock scope.  The workflow-side *message* tests for this class live in
``test_workflow.py`` instead, because
``test_merge_queue_reachback_patch_guard.py``'s ALLOWLIST already carries
the pair ``('test_workflow.py', '_check_plan_files_touched_in_branch')``.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from orchestrator.config import GitConfig
from orchestrator.git_ops import GitOps, _run
from orchestrator.merge_gates import (  # noqa: F401  (used by later steps)
    PlanFilesTouchedResult,
    _check_plan_files_touched_in_branch,
)

# ---------------------------------------------------------------------------
# Fixtures — the standard real-git fixture triple, copied verbatim from
# test_merge_queue.py:88-122.  Per-file duplication (rather than promotion
# to conftest.py) is the established convention across ~60 sibling test
# files in this suite; promoting it would widen this task's lock scope onto
# a shared conftest for no benefit.
# ---------------------------------------------------------------------------


@pytest.fixture
def git_repo(tmp_path: Path) -> Path:
    """Create a temporary git repository with an initial commit."""
    repo = tmp_path / 'repo'
    repo.mkdir()
    asyncio.run(_setup_repo(repo))
    return repo


async def _setup_repo(repo: Path):
    await _run(['git', 'init', '-b', 'main'], cwd=repo)
    await _run(['git', 'config', 'user.email', 'test@test.com'], cwd=repo)
    await _run(['git', 'config', 'user.name', 'Test'], cwd=repo)
    (repo / 'README.md').write_text('# Test\n')
    await _run(['git', 'add', '-A'], cwd=repo)
    await _run(['git', 'commit', '-m', 'Initial commit'], cwd=repo)


@pytest.fixture
def git_config() -> GitConfig:
    return GitConfig(
        main_branch='main',
        branch_prefix='task/',
        remote='origin',
        worktree_dir='.worktrees',
        # Tests use a tmp repo with no real remote; disabling the push avoids
        # per-test subprocess noise.
        push_after_advance=False,
    )


@pytest.fixture
def git_ops(git_config: GitConfig, git_repo: Path) -> GitOps:
    return GitOps(git_config, git_repo)


# ---------------------------------------------------------------------------
# Main-side staging helpers
#
# Every test in this file follows the same three-beat shape that reproduces
# the measured reify-5196 case:
#
#   1. ``_commit_on_main``  — stage the PRE-rename main state (the state the
#      architect saw when it declared ``metadata.files``);
#   2. ``_rename_on_main``  — relocate the file ON MAIN, before any branch
#      exists (this is what makes the class distinct from a branch-side
#      rename, which ``get_files_touched_in_branch``'s ``--no-renames``
#      union already surfaces on both sides);
#   3. ``git_ops.create_worktree(...)`` — cut the task branch from
#      POST-rename main and commit against the NEW path.
#
# Both helpers operate directly in ``git_repo`` (main's checkout), never in
# a worktree, so the rename is unambiguously main-side.
# ---------------------------------------------------------------------------


async def _commit_on_main(
    git_ops: GitOps,
    repo: Path,
    paths_content: dict[str, str],
    msg: str,
) -> str:
    """Write *paths_content* into *repo* on main and commit.  Returns the SHA.

    Keys are repo-relative POSIX paths; parent directories are created as
    needed.  A key mapped to an empty string still produces an (empty) file,
    so callers can stage a path purely for its existence.
    """
    for rel, content in paths_content.items():
        target = repo / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content)
    await _run(['git', 'add', '-A'], cwd=repo)
    rc, _, err = await _run(['git', 'commit', '-m', msg], cwd=repo)
    assert rc == 0, f'commit on main failed: {err}'
    rc, out, _ = await _run(['git', 'rev-parse', 'HEAD'], cwd=repo)
    assert rc == 0
    return out.strip()


async def _rename_on_main(
    git_ops: GitOps,
    repo: Path,
    old: str,
    new: str,
    msg: str,
) -> str:
    """``git mv`` *old* → *new* on main and commit.  Returns the SHA.

    This is the authoritative rename shape — ``git show --name-status -M``
    on the resulting commit yields an ``R<score>\\t<old>\\t<new>`` pair.
    Tests that need the *unrecoverable-pair* shape (separate delete and add
    commits) stage that explicitly with ``git rm`` + ``_commit_on_main``
    rather than through this helper.
    """
    (repo / new).parent.mkdir(parents=True, exist_ok=True)
    rc, _, err = await _run(['git', 'mv', old, new], cwd=repo)
    assert rc == 0, f'git mv {old} -> {new} failed: {err}'
    rc, _, err = await _run(['git', 'commit', '-m', msg], cwd=repo)
    assert rc == 0, f'rename commit failed: {err}'
    rc, out, _ = await _run(['git', 'rev-parse', 'HEAD'], cwd=repo)
    assert rc == 0
    return out.strip()


async def _head_of(worktree: Path) -> str:
    """Return the resolved HEAD SHA of *worktree*."""
    rc, out, _ = await _run(['git', 'rev-parse', 'HEAD'], cwd=worktree)
    assert rc == 0
    return out.strip()
