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
import logging
from collections.abc import Callable, Sequence
from pathlib import Path

import pytest

from orchestrator.config import GitConfig
from orchestrator.git_ops import GitOps, _run
from orchestrator.merge_gates import _check_plan_targets_in_tree

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


# ---------------------------------------------------------------------------
# Scenario staging
#
# This is the reify esc-6436-4 shape, and it is the MIRROR of
# test_merge_queue.py::TestCheckPlanTargetsInTree::test_sibling_moved_file_not_flagged
# with exactly one variable flipped: there the branch never touched the
# moved path, here it MODIFIED it.  The 40 spaced lines and the
# top-vs-bottom edit split are lifted from the esc-3843 Cargo.lock case,
# where they guarantee the 3-way merge is clean.
# ---------------------------------------------------------------------------

_BASE_MODULE = ''.join(f'line{i}\n' for i in range(1, 41))


async def _commit_base_module(git_ops: GitOps) -> None:
    """Put ``pkg/a.py`` (40 spaced lines) on main — the shared fork point."""
    (git_ops.project_root / 'pkg').mkdir()
    (git_ops.project_root / 'pkg' / 'a.py').write_text(_BASE_MODULE)
    await _run(['git', 'add', '-A'], cwd=git_ops.project_root)
    await _run(
        ['git', 'commit', '-m', 'Main: add pkg/a.py'],
        cwd=git_ops.project_root,
    )


async def _branch_modifies_the_module(
    git_ops: GitOps, branch: str,
) -> tuple[Path, str]:
    """Cut *branch* and edit ``pkg/a.py`` near the TOP.  Returns (wt, head)."""
    wt = (await git_ops.create_worktree(branch)).path
    (wt / 'pkg' / 'a.py').write_text(
        _BASE_MODULE.replace('line2\n', 'line2\nBRANCH_EDIT\n')
    )
    await git_ops.commit(wt, 'Branch: edit pkg/a.py')
    rc, head_out, _ = await _run(['git', 'rev-parse', 'HEAD'], cwd=wt)
    assert rc == 0
    return wt, head_out.strip()


async def _main_relocates_and_edits(git_ops: GitOps) -> None:
    """A sibling relocates ``pkg/a.py`` on main and edits it near the BOTTOM."""
    rc, _, err = await _run(
        ['git', 'mv', 'pkg/a.py', 'pkg/b.py'], cwd=git_ops.project_root,
    )
    assert rc == 0, f'git mv failed: {err!r}'
    (git_ops.project_root / 'pkg' / 'b.py').write_text(
        _BASE_MODULE.replace('line37\n', 'line37\nMAIN_EDIT\n')
    )
    await _run(['git', 'add', '-A'], cwd=git_ops.project_root)
    await _run(
        ['git', 'commit', '-m', 'Sibling: move pkg/a.py -> pkg/b.py and edit'],
        cwd=git_ops.project_root,
    )


@pytest.mark.asyncio
class TestDropGuardRenameAwareness:
    """A main-side relocation is not a drop of the path it moved from."""

    async def test_sibling_relocation_of_a_branch_modified_file_is_not_flagged(
        self, git_ops: GitOps,
    ):
        """A sibling relocates a file the branch MODIFIED → no drop.

        The regression (reify esc-6436-4).  ``pkg/a.py`` is in
        ``branch_changed`` because the branch really did modify it, and in
        ``dropped_in_merge`` because the merge carried it to ``pkg/b.py``
        under ``--no-renames``.  The intersection fires and the guard
        reports a drop of work that is sitting, intact, at the new path.
        """
        await _commit_base_module(git_ops)
        wt, task_head = await _branch_modifies_the_module(
            git_ops, 'drop-rename',
        )
        await _main_relocates_and_edits(git_ops)
        main_sha = await git_ops.get_main_sha()

        merge_result = await git_ops.merge_to_main(wt, 'drop-rename')
        assert merge_result.success, (
            f'expected clean 3-way merge; details={merge_result.details!r}'
        )
        assert merge_result.merge_commit is not None
        try:
            # Non-vacuous precondition 1: the merge really did relocate the
            # file — so the guard is being asked about a rename, not a no-op.
            rc, tree_out, _ = await _run(
                ['git', 'ls-tree', '-r', '--name-only',
                 merge_result.merge_commit],
                cwd=git_ops.project_root,
            )
            assert rc == 0
            tree = tree_out.split()
            assert 'pkg/b.py' in tree, f'merged tree: {tree!r}'
            assert 'pkg/a.py' not in tree, f'merged tree: {tree!r}'

            # Non-vacuous precondition 2: the branch's work SURVIVED the
            # relocation, so nothing was dropped in fact.
            rc, blob, _ = await _run(
                ['git', 'show', f'{merge_result.merge_commit}:pkg/b.py'],
                cwd=git_ops.project_root,
            )
            assert rc == 0
            assert 'BRANCH_EDIT' in blob, 'branch work missing from merge'
            assert 'MAIN_EDIT' in blob, 'main work missing from merge'

            # Non-vacuous precondition 3 — the pin that proves this test
            # exercises the real mechanism: the raw (unsubtracted) drop set
            # DOES contain the old path, exactly as in the esc-3861 sibling
            # test.  Rename resolution is what must clear it.
            rc, raw_out, _ = await _run(
                ['git', 'diff', '--name-only', '--no-renames',
                 '--diff-filter=D', task_head, merge_result.merge_commit],
                cwd=git_ops.project_root,
            )
            assert rc == 0
            assert 'pkg/a.py' in raw_out, (
                f'expected merge to drop pkg/a.py; raw drop set: {raw_out!r}'
            )

            result = await _check_plan_targets_in_tree(
                merge_result.merge_commit, wt, git_ops, main_sha,
                task_id='drop-rename',
            )
            assert result.dropped == [], (
                f'a relocated file the branch modified must not be flagged; '
                f'got {result.dropped!r}'
            )
        finally:
            if merge_result.merge_worktree:
                await git_ops.cleanup_merge_worktree(merge_result.merge_worktree)

    async def test_genuine_drop_of_a_branch_modified_file_is_still_flagged(
        self, git_ops: GitOps,
    ):
        """A real delete with no pairable rename is STILL flagged.

        The fail-CLOSED discriminator, and the negative guard this gate's
        acceptance requires.  This test PASSES on today's code and that is
        INTENDED — it must stay green through the fix.  What it forbids is
        the cheap over-broad remedy of suppressing every ``D`` whose path
        the branch touched: that would make the regression above pass
        while silently disabling the guard for every file the branch
        edited, which is the one class of work it exists to protect.

        Same base and same branch modification, but main deletes
        ``pkg/a.py`` outright with no replacement, so the merge-side diff
        shows a ``D`` that ``-M`` cannot pair with any ``R``.
        """
        await _commit_base_module(git_ops)
        wt, _task_head = await _branch_modifies_the_module(
            git_ops, 'drop-genuine',
        )

        # Main deletes the file outright — no relocation, nothing to pair.
        (git_ops.project_root / 'pkg' / 'a.py').unlink()
        await _run(['git', 'add', '-A'], cwd=git_ops.project_root)
        await _run(
            ['git', 'commit', '-m', 'Main: delete pkg/a.py'],
            cwd=git_ops.project_root,
        )
        main_sha = await git_ops.get_main_sha()

        # Synthetic merge commit = main's tip: a resolution that took main's
        # deletion and discarded the branch's edit entirely.
        result = await _check_plan_targets_in_tree(
            main_sha, wt, git_ops, main_sha, task_id='drop-genuine',
        )
        assert result.dropped == ['pkg/a.py'], (
            f'a genuine delete with no pairable rename must still be '
            f'flagged; got {result.dropped!r}'
        )

    async def test_drop_guard_rename_map_git_error_fails_open(
        self, git_ops: GitOps, caplog, monkeypatch,
    ):
        """A failing rename-pair diff fails OPEN, loudly.

        Uniform with the guard's four existing ``rc != 0`` arms and with
        its stated policy: flagging a phantom drop on a transient git
        error is worse than missing a real one.
        """
        await _commit_base_module(git_ops)
        wt, _task_head = await _branch_modifies_the_module(
            git_ops, 'drop-rename-failopen',
        )
        await _main_relocates_and_edits(git_ops)
        main_sha = await git_ops.get_main_sha()

        merge_result = await git_ops.merge_to_main(wt, 'drop-rename-failopen')
        assert merge_result.success
        assert merge_result.merge_commit is not None
        try:
            spy = _RunSpy(
                fail_when=lambda cmd: '--name-status' in cmd and '-M' in cmd,
            )
            monkeypatch.setattr('orchestrator.merge_gates._run', spy)

            with caplog.at_level(
                logging.WARNING, logger='orchestrator.merge_queue',
            ):
                result = await _check_plan_targets_in_tree(
                    merge_result.merge_commit, wt, git_ops, main_sha,
                    task_id='drop-rename-failopen',
                )

            assert result.dropped == [], (
                f'an unreadable rename map must fail open; got '
                f'{result.dropped!r}'
            )
            warnings = '\n'.join(
                r.getMessage() for r in caplog.records
                if r.levelno >= logging.WARNING
            )
            assert 'rename-pair diff' in warnings, (
                f'fail-open must name the failed command; got {warnings!r}'
            )
            assert 'failing open' in warnings, warnings
        finally:
            if merge_result.merge_worktree:
                await git_ops.cleanup_merge_worktree(merge_result.merge_worktree)
