"""Gate-level tests for a rename false-positive class of
:func:`orchestrator.merge_gates._check_plan_targets_in_tree` — SITE 2.

The defect: the drop-guard intersects ``branch_changed`` (the branch's
add/modify set, ``--no-renames --diff-filter=AM``) with
``dropped_in_merge`` (``--no-renames --diff-filter=D`` over
``task_HEAD..merge_commit``).  Both diffs suppress rename detection, so a
SIBLING/main-side relocation of a path the branch MODIFIED reads as a
delete of that path: the old path is in ``branch_changed`` because the
branch really did modify it, and it is in ``dropped_in_merge`` because
the merge carried it to a new name.  The intersection fires and the guard
blocks with "Merge commit is missing plan target files", even though the
branch's work survived intact at the relocated path.

Measured case: reify esc-6436-4 (2026-09-13/14).

The complementary half of this gate is ALREADY covered:
``test_merge_queue.py::TestCheckPlanTargetsInTree::test_sibling_moved_file_not_flagged``
(L795, esc-3861) pins the case where the sibling moves a path the branch
NEVER touched — there the old path is absent from ``branch_changed`` and
the intersection correctly suppresses it.  This file covers the mirror
case, with exactly one variable flipped: the branch DID modify the
relocated path.  Reading the two together shows immediately which half is
new and why the existing ``--no-renames`` rationale was only half true.

The other rename-aware site — the post-merge equivalence gate, which
resolves renames on the BRANCH range rather than the merge range — lives
in ``test_merge_gates_equivalence_rename.py``.

These tests exercise the guard against a REAL git repository, so they live
in this dedicated file rather than in the 24k-line ``test_merge_queue.py``
— the precedent set by the sibling ``test_merge_gates_plan_files_rename.py``
(its docstring, L29-30/36-39: real-git tests plus keeping hot
``merge_queue.py`` out of this task's lock scope).  They import the guard
from ``orchestrator.merge_gates`` DIRECTLY, never through the
``orchestrator.merge_queue`` shim.  The only deviation from pure real-git
is :class:`_RunSpy`, used narrowly for the one property real git will not
produce on demand: a non-zero rc from a specific git subcommand.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Sequence
from pathlib import Path

import pytest

from orchestrator.config import GitConfig
from orchestrator.git_ops import GitOps, _run

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
# _RunSpy — this file's one, narrowly-scoped deviation from pure real-git
# ---------------------------------------------------------------------------


class _RunSpy:
    """Delegating wrapper around ``merge_gates._run`` with fault injection.

    One guard property is invisible to a pure real-git test: what the
    guard does when a specific git subcommand returns a non-zero rc, which
    real git will not produce on demand.  *fail_when* is a predicate over
    the argv list; a command it matches returns ``(128, '', <fatal>)``
    WITHOUT being executed.  Everything else delegates to the real
    ``_run``, so the surrounding repository work stays real.

    Modelled on ``test_merge_gates_plan_files_rename.py::_RunSpy`` minus
    its ``calls``/``count`` arm, which this file has no use for.
    """

    def __init__(
        self, fail_when: Callable[[Sequence[str]], bool] | None = None,
    ) -> None:
        self._fail_when = fail_when

    async def __call__(
        self, cmd: list[str], cwd: Path | None = None, **kwargs,
    ) -> tuple[int, str, str]:
        if self._fail_when is not None and self._fail_when(cmd):
            return 128, '', 'fatal: injected failure (test fault injection)\n'
        return await _run(cmd, cwd, **kwargs)
