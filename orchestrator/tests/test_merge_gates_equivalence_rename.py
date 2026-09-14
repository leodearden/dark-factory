"""Gate-level tests for a rename false-positive class of
:func:`orchestrator.merge_gates._check_post_merge_equivalence` — SITE 1.

The defect: the gate's compare set is built by a raw path-string
subtraction, ``[p for p in branch_touched if p not in main_touched]``,
while all three of its set-building diffs pass ``--no-renames``.  A
branch-side rename is therefore SPLIT into two unrelated path strings
(old and new), and main's concurrent edits to the same file land at the
rename SOURCE.  The new path is absent from ``main_touched``, so it
survives the subtraction and the final scoped diff reports it — the gate
blocks a merge that dropped nothing at all.

Measured case: reify task 5694, merge ``d1d857f43545``, escalation
esc-5694-5.  The branch relocated a file main had concurrently edited at
the old path; the merged tree carried BOTH edits, and the gate still
reported "Conflict resolution likely dropped or rewrote work".  The RCA
that followed read the triage diff in the wrong direction and concluded
the opposite of the truth.

The complementary SITE 2 defect — the plan-target drop-guard's
``branch_changed`` / ``dropped_in_merge`` intersection — lives in
``test_merge_gates_drop_guard_rename.py``.  The two share the
``_rename_pairs`` primitive but resolve renames on OPPOSITE ranges, so
they are kept in separate files.

These tests exercise the gate against a REAL git repository, so they live
in this dedicated file rather than in the 24k-line ``test_merge_queue.py``
— the precedent set by the sibling ``test_merge_gates_plan_files_rename.py``
(its docstring, L29-30/36-39: real-git tests plus keeping hot
``merge_queue.py`` out of this task's lock scope).  They import the gate
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
from orchestrator.merge_gates import _check_post_merge_equivalence

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

    One gate property is invisible to a pure real-git test: what the gate
    does when a specific git subcommand returns a non-zero rc, which real
    git will not produce on demand.  *fail_when* is a predicate over the
    argv list; a command it matches returns ``(128, '', <fatal>)`` WITHOUT
    being executed.  Everything else delegates to the real ``_run``, so
    the surrounding repository work stays real.

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
# Both scenarios share the same base and the same branch-side rename; they
# differ only in what MAIN does concurrently.  The 40 spaced lines and the
# top-vs-bottom edit split are lifted from
# test_merge_queue.py::TestCheckPostMergeEquivalence::
# test_sibling_also_touched_lockfile_not_flagged (the esc-3843 Cargo.lock
# case), where they guarantee the 3-way merge is clean.
# ---------------------------------------------------------------------------

_BASE_MODULE = ''.join(f'line{i}\n' for i in range(1, 41))


async def _commit_base_module(git_ops: GitOps) -> None:
    """Put ``pkg/mod.py`` (40 spaced lines) on main — the shared fork point."""
    (git_ops.project_root / 'pkg').mkdir()
    (git_ops.project_root / 'pkg' / 'mod.py').write_text(_BASE_MODULE)
    await _run(['git', 'add', '-A'], cwd=git_ops.project_root)
    await _run(
        ['git', 'commit', '-m', 'Main: add pkg/mod.py'],
        cwd=git_ops.project_root,
    )


async def _branch_relocates_and_edits(git_ops: GitOps, branch: str) -> Path:
    """Cut *branch*, ``git mv`` the module and edit it near the TOP."""
    wt = (await git_ops.create_worktree(branch)).path
    (wt / 'pkg' / 'sub').mkdir(parents=True)
    rc, _, err = await _run(
        ['git', 'mv', 'pkg/mod.py', 'pkg/sub/mod.py'], cwd=wt,
    )
    assert rc == 0, f'git mv failed: {err!r}'
    (wt / 'pkg' / 'sub' / 'mod.py').write_text(
        _BASE_MODULE.replace('line2\n', 'line2\nBRANCH_EDIT\n')
    )
    await git_ops.commit(wt, 'Branch: relocate pkg/mod.py and edit it')
    return wt


async def _main_edits_the_rename_source(git_ops: GitOps) -> None:
    """Main edits ``pkg/mod.py`` near the BOTTOM — at the rename SOURCE."""
    (git_ops.project_root / 'pkg' / 'mod.py').write_text(
        _BASE_MODULE.replace('line37\n', 'line37\nMAIN_EDIT\n')
    )
    await _run(['git', 'add', '-A'], cwd=git_ops.project_root)
    await _run(
        ['git', 'commit', '-m', 'Main: edit pkg/mod.py'],
        cwd=git_ops.project_root,
    )


@pytest.mark.asyncio
class TestEquivalenceGateRenameAwareness:
    """The gate must follow a branch-side rename back to its source."""

    async def test_rename_with_main_side_source_edit_is_not_flagged(
        self, git_ops: GitOps,
    ):
        """Branch relocates a file main concurrently edited → no divergence.

        The regression (reify task 5694 / esc-5694-5).  ``main_touched``
        carries the rename SOURCE (``pkg/mod.py``) because main edited it
        there, while ``branch_touched`` carries both halves of the split
        rename.  The raw string subtraction leaves the NEW path in the
        compare set, and the scoped diff then reports it even though the
        merged file carries BOTH sides' edits.
        """
        await _commit_base_module(git_ops)
        wt = await _branch_relocates_and_edits(git_ops, 'equiv-rename')
        await _main_edits_the_rename_source(git_ops)
        main_sha = await git_ops.get_main_sha()

        merge_result = await git_ops.merge_to_main(wt, 'equiv-rename')
        assert merge_result.success, (
            f'expected clean 3-way merge; details={merge_result.details!r}'
        )
        assert merge_result.merge_commit is not None
        assert merge_result.merge_worktree is not None
        try:
            outcome = await git_ops.advance_main(
                merge_result.merge_commit, merge_result.merge_worktree,
                branch='equiv-rename', max_attempts=1,
            )
            advanced = outcome.advanced_sha or merge_result.merge_commit
            assert advanced is not None

            # Non-vacuous precondition 1: the merge really did relocate the
            # file — so the gate is being asked about a rename, not a no-op.
            rc, tree_out, _ = await _run(
                ['git', 'ls-tree', '-r', '--name-only', advanced],
                cwd=git_ops.project_root,
            )
            assert rc == 0
            tree = tree_out.split()
            assert 'pkg/sub/mod.py' in tree, f'merged tree: {tree!r}'
            assert 'pkg/mod.py' not in tree, f'merged tree: {tree!r}'

            # Non-vacuous precondition 2: nothing was dropped AND the merged
            # file genuinely differs from the branch tip (it carries main's
            # edit too) — so a passing gate is not passing for want of a diff.
            rc, blob, _ = await _run(
                ['git', 'show', f'{advanced}:pkg/sub/mod.py'],
                cwd=git_ops.project_root,
            )
            assert rc == 0
            assert 'BRANCH_EDIT' in blob, 'branch work missing from merge'
            assert 'MAIN_EDIT' in blob, 'main work missing from merge'

            failed = await _check_post_merge_equivalence(
                wt, advanced, git_ops, main_sha, task_id='equiv-rename',
            )
            assert failed == [], (
                f'a relocated file whose source main edited must not be '
                f'flagged; got {failed!r}'
            )
        finally:
            await git_ops.cleanup_merge_worktree(merge_result.merge_worktree)

    async def test_rename_whose_source_main_never_touched_is_still_compared(
        self, git_ops: GitOps,
    ):
        """A rename main did NOT touch the source of stays in the compare set.

        The fail-CLOSED discriminator.  This test PASSES on today's code and
        that is INTENDED — it is the no-false-negative pin, and it must stay
        green through the fix.  What it forbids is the cheap over-broad
        remedy of suppressing every rename TARGET: that would make the
        regression above pass while silently disabling the gate for every
        relocated file, which is the one class of work it exists to protect.

        Same base and same branch rename as above, but main's concurrent
        commit touches an unrelated file, so no rename source is in
        ``main_touched``.  The gate is then pointed at a synthetic
        ``advanced_sha`` lacking the renamed file — the
        ``test_diverging_tree_flags_files`` trick (test_merge_queue.py
        L9198) — and must still report it.
        """
        await _commit_base_module(git_ops)
        rc, base_out, _ = await _run(
            ['git', 'rev-parse', 'HEAD'], cwd=git_ops.project_root,
        )
        assert rc == 0
        base_sha = base_out.strip()

        wt = await _branch_relocates_and_edits(git_ops, 'equiv-rename-unrelated')

        # Main moves ahead, but nowhere near the rename source.
        (git_ops.project_root / 'other.py').write_text('other = 1\n')
        await _run(['git', 'add', '-A'], cwd=git_ops.project_root)
        await _run(
            ['git', 'commit', '-m', 'Main: add other.py'],
            cwd=git_ops.project_root,
        )
        main_sha = await git_ops.get_main_sha()

        # base_sha predates the branch entirely, so the relocated file is
        # absent from it — exactly what a resolution that dropped the
        # branch's work would look like.
        failed = await _check_post_merge_equivalence(
            wt, base_sha, git_ops, main_sha,
            task_id='equiv-rename-unrelated',
        )
        assert 'pkg/sub/mod.py' in failed, (
            f'a rename whose source main never touched must still be '
            f'compared; got {failed!r}'
        )

    async def test_rename_map_git_error_fails_open(
        self, git_ops: GitOps, caplog, monkeypatch,
    ):
        """A failing rename-pair diff fails OPEN, loudly.

        Uniform with the gate's four existing ``rc != 0`` arms and with its
        stated policy: a transient git error must not block a successful
        merge from being recorded.
        """
        await _commit_base_module(git_ops)
        wt = await _branch_relocates_and_edits(git_ops, 'equiv-rename-failopen')
        await _main_edits_the_rename_source(git_ops)
        main_sha = await git_ops.get_main_sha()

        merge_result = await git_ops.merge_to_main(wt, 'equiv-rename-failopen')
        assert merge_result.success
        assert merge_result.merge_commit is not None
        assert merge_result.merge_worktree is not None
        try:
            outcome = await git_ops.advance_main(
                merge_result.merge_commit, merge_result.merge_worktree,
                branch='equiv-rename-failopen', max_attempts=1,
            )
            advanced = outcome.advanced_sha or merge_result.merge_commit
            assert advanced is not None

            spy = _RunSpy(
                fail_when=lambda cmd: '--name-status' in cmd and '-M' in cmd,
            )
            monkeypatch.setattr('orchestrator.merge_gates._run', spy)

            with caplog.at_level(
                logging.WARNING, logger='orchestrator.merge_queue',
            ):
                failed = await _check_post_merge_equivalence(
                    wt, advanced, git_ops, main_sha,
                    task_id='equiv-rename-failopen',
                )

            assert failed == [], (
                f'an unreadable rename map must fail open; got {failed!r}'
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
            await git_ops.cleanup_merge_worktree(merge_result.merge_worktree)


async def _resolution_discards_the_branch_edit(
    git_ops: GitOps, task_head: str, main_sha: str,
) -> str:
    """Keep the branch's relocation but write MAIN's content at the new name.

    Produces a REAL advanced commit whose ``pkg/sub/mod.py`` carries
    ``MAIN_EDIT`` and NOT ``BRANCH_EDIT``, while ``git diff -M`` still
    pairs ``pkg/mod.py`` -> ``pkg/sub/mod.py`` across the branch range.
    That pair is what reaches the suppression arm.
    """
    await _run(
        ['git', 'merge', '--no-commit', '--no-ff', task_head],
        cwd=git_ops.project_root,
    )
    # Tolerant: rename detection may already have staged the removal.
    await _run(['git', 'rm', '-f', 'pkg/mod.py'], cwd=git_ops.project_root)
    rc, main_blob, err = await _run(
        ['git', 'show', f'{main_sha}:pkg/mod.py'], cwd=git_ops.project_root,
    )
    assert rc == 0, f'reading main blob failed: {err!r}'
    target = git_ops.project_root / 'pkg' / 'sub' / 'mod.py'
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(main_blob + '\n')
    await _run(['git', 'add', '-A'], cwd=git_ops.project_root)
    rc, _, err = await _run(
        ['git', 'commit', '-m', 'Resolution: keep the move, drop branch edit'],
        cwd=git_ops.project_root,
    )
    assert rc == 0, f'resolution commit failed: {err!r}'
    rc, out, _ = await _run(
        ['git', 'rev-parse', 'HEAD'], cwd=git_ops.project_root,
    )
    assert rc == 0
    return out.strip()


@pytest.mark.asyncio
class TestEquivalenceSuppressionIsContentVerified:
    """The symmetric half of the drop-guard's content check.

    ``_rename_aware_compare_set`` excluded a path whenever main had
    touched its rename SOURCE, on the bare existence of a pair.  A
    resolution that keeps the relocation but writes main's content at the
    new name still pairs — and the drop-guard cannot backstop it, because
    the branch's old path is already absent from ``task_head`` and so
    never reaches the ``D`` set.
    """

    async def test_relocation_whose_branch_edit_was_discarded_is_still_flagged(
        self, git_ops: GitOps,
    ):
        """A pairable branch rename that LOST the branch's edit is flagged."""
        await _commit_base_module(git_ops)
        rc, base_out, _ = await _run(
            ['git', 'rev-parse', 'HEAD'], cwd=git_ops.project_root,
        )
        assert rc == 0
        base_sha = base_out.strip()

        wt = await _branch_relocates_and_edits(git_ops, 'equiv-discarded')
        rc, head_out, _ = await _run(['git', 'rev-parse', 'HEAD'], cwd=wt)
        assert rc == 0
        branch_head = head_out.strip()

        await _main_edits_the_rename_source(git_ops)
        main_sha = await git_ops.get_main_sha()
        advanced = await _resolution_discards_the_branch_edit(
            git_ops, branch_head, main_sha,
        )

        # Non-vacuous precondition 1: the relocation was kept.
        rc, tree_out, _ = await _run(
            ['git', 'ls-tree', '-r', '--name-only', advanced],
            cwd=git_ops.project_root,
        )
        assert rc == 0
        tree = tree_out.split()
        assert 'pkg/sub/mod.py' in tree, f'advanced tree: {tree!r}'
        assert 'pkg/mod.py' not in tree, f'advanced tree: {tree!r}'

        # Non-vacuous precondition 2: the branch's work is GENUINELY GONE.
        rc, blob, _ = await _run(
            ['git', 'show', f'{advanced}:pkg/sub/mod.py'],
            cwd=git_ops.project_root,
        )
        assert rc == 0
        assert 'MAIN_EDIT' in blob, 'main work should have survived'
        assert 'BRANCH_EDIT' not in blob, (
            'fixture is wrong: the branch edit was supposed to be discarded'
        )

        # Non-vacuous precondition 3: git DOES pair the branch rename, so
        # the suppression arm is genuinely reached.
        rc, ns_out, _ = await _run(
            ['git', 'diff', '-M', '--name-status', base_sha, branch_head],
            cwd=git_ops.project_root,
        )
        assert rc == 0
        assert any(
            ln.startswith('R') and 'pkg/mod.py' in ln
            and 'pkg/sub/mod.py' in ln
            for ln in ns_out.splitlines()
        ), f'expected a pairable rename; got {ns_out!r}'

        failed = await _check_post_merge_equivalence(
            wt, advanced, git_ops, main_sha, task_id='equiv-discarded',
        )
        assert failed == ['pkg/sub/mod.py'], (
            f'a relocation that discarded the branch edit must still be '
            f'flagged; got {failed!r}'
        )

    async def test_equivalence_content_probe_git_error_fails_closed(
        self, git_ops: GitOps, monkeypatch,
    ):
        """An unverifiable survival claim must NOT suppress.

        The OPPOSITE direction from
        ``test_rename_map_git_error_fails_open``, deliberately, for the
        same reason as the drop-guard's pair: an unreadable rename map
        degrades the whole gate and fails open, while an unproven content
        probe declines only that one suppression, falling back to
        pre-change behaviour that cannot introduce a false block.
        """
        await _commit_base_module(git_ops)
        wt = await _branch_relocates_and_edits(git_ops, 'equiv-probe-error')
        rc, head_out, _ = await _run(['git', 'rev-parse', 'HEAD'], cwd=wt)
        assert rc == 0
        branch_head = head_out.strip()

        await _main_edits_the_rename_source(git_ops)
        main_sha = await git_ops.get_main_sha()
        advanced = await _resolution_discards_the_branch_edit(
            git_ops, branch_head, main_sha,
        )

        spy = _RunSpy(fail_when=lambda cmd: 'apply' in cmd)
        monkeypatch.setattr('orchestrator.merge_gates._run', spy)

        failed = await _check_post_merge_equivalence(
            wt, advanced, git_ops, main_sha, task_id='equiv-probe-error',
        )
        assert failed == ['pkg/sub/mod.py'], (
            f'an unverifiable content probe must not suppress; got {failed!r}'
        )
