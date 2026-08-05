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
import logging
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


# ---------------------------------------------------------------------------
# API surface — the two new PlanFilesTouchedResult fields
# ---------------------------------------------------------------------------


class TestPlanFilesTouchedResultFields:
    """The classification fields this FP class adds to the gate's result.

    ``missing_from_tree`` and ``resolved_renames`` are APPENDED with
    ``field(default_factory=...)`` defaults, so every existing construction
    site — all of which are keyword-only — keeps working untouched.
    """

    def test_defaults_are_empty(self) -> None:
        """A bare result carries three empty containers, not None."""
        result = PlanFilesTouchedResult()
        assert result.not_touched == []
        assert result.missing_from_tree == []
        assert result.resolved_renames == {}

    def test_mutable_defaults_are_per_instance(self) -> None:
        """Each instance owns its containers (pins ``default_factory``).

        A bare ``= []`` / ``= {}`` class attribute would be shared across
        every instance, so one gate invocation's stale paths would leak into
        the next one's result.
        """
        first = PlanFilesTouchedResult()
        first.not_touched.append('leaked.py')
        first.missing_from_tree.append('leaked.py')
        first.resolved_renames['leaked.py'] = 'somewhere/else.py'

        second = PlanFilesTouchedResult()
        assert second.not_touched == []
        assert second.missing_from_tree == []
        assert second.resolved_renames == {}

    def test_legacy_keyword_construction_still_works(self) -> None:
        """The pre-existing single-field construction is unchanged.

        This is the exact shape ``test_workflow.py``'s ``_CheckSequence``
        stubs and ``test_merge_queue.py`` build today.
        """
        result = PlanFilesTouchedResult(not_touched=['a.py'])
        assert result.not_touched == ['a.py']
        assert result.missing_from_tree == []
        assert result.resolved_renames == {}

    def test_field_container_types(self) -> None:
        """``missing_from_tree`` is a list of str; ``resolved_renames`` a dict."""
        result = PlanFilesTouchedResult(
            not_touched=['crates/tests/topo_e2e.rs'],
            missing_from_tree=['crates/tests/topo_e2e.rs'],
            resolved_renames={
                'crates/tests/topo_e2e.rs': 'crates/tests/harness_topo/topo_e2e.rs',
            },
        )
        assert isinstance(result.missing_from_tree, list)
        assert all(isinstance(p, str) for p in result.missing_from_tree)
        assert isinstance(result.resolved_renames, dict)
        assert all(
            isinstance(k, str) and isinstance(v, str)
            for k, v in result.resolved_renames.items()
        )

    def test_missing_from_tree_is_subset_of_not_touched(self) -> None:
        """The documented subset invariant holds for the canonical shape.

        A stale declared entry is reported in BOTH lists — never moved out of
        ``not_touched`` — so every existing ``not_touched``-only consumer
        (workflow's block decision, ``_try_narrow_plan``,
        ``build_plan_tightening_prompt``) keeps its current verdict.
        """
        result = PlanFilesTouchedResult(
            not_touched=['a.py', 'crates/tests/topo_e2e.rs'],
            missing_from_tree=['crates/tests/topo_e2e.rs'],
        )
        assert set(result.missing_from_tree) <= set(result.not_touched)


# ---------------------------------------------------------------------------
# Mechanism 1 — resolution via the deleting commit's rename pair
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestRenamedOnMainResolvesViaDeletingCommit:
    """The measured reify-5196 shape: ``git mv`` on main, branch edits the
    new path, gate must PASS instead of blaming the branch."""

    async def test_renamed_on_main_gate_passes(
        self,
        git_ops: GitOps,
        git_repo: Path,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        old = 'crates/tests/topo_e2e.rs'
        new = 'crates/tests/harness_topo/topo_e2e.rs'

        # 1. Pre-rename main — the state the architect declared against.
        await _commit_on_main(
            git_ops, git_repo, {old: 'fn topo() {}\n'}, 'add topo_e2e',
        )
        # 2. Main relocates it (identical basename — the harness-layout
        #    consolidation shape), before any branch exists.
        rename_sha = await _rename_on_main(
            git_ops, git_repo, old, new, 'harness: consolidate topo tests',
        )
        # 3. Branch cut from POST-rename main; touches only the NEW path.
        wt = (await git_ops.create_worktree('rename-on-main')).path
        base = await _head_of(wt)
        (wt / new).write_text('fn topo() { assert!(true); }\n')
        await git_ops.commit(wt, 'step-2')
        head = await _head_of(wt)

        with caplog.at_level(logging.WARNING, logger='orchestrator.merge_queue'):
            result = await _check_plan_files_touched_in_branch(
                [old], base, head, git_ops, task_id='rename-5196',
            )

        assert result.not_touched == [], (
            'a path renamed on main before the branch was cut must not be '
            'reported as untouched by the branch'
        )
        assert result.missing_from_tree == []
        assert result.resolved_renames == {old: new}

        msg = ' '.join(
            r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING
        )
        assert old in msg, 'resolution log must name the declared path'
        assert new in msg, 'resolution log must name the resolved path'
        assert rename_sha[:12] in msg, (
            'resolution log must name the deleting commit so the rename is auditable'
        )

    async def test_dot_slash_declared_entry_resolves_and_keys_on_original(
        self, git_ops: GitOps, git_repo: Path,
    ) -> None:
        """Composes with task 1587's ``./``-prefix normalization class.

        Matching happens on the normalized form, but ``resolved_renames`` is
        keyed by the ORIGINAL declared string so the diagnostic reflects
        exactly what the architect wrote in plan.json.
        """
        old = 'crates/tests/topo_e2e.rs'
        new = 'crates/tests/harness_topo/topo_e2e.rs'
        declared = f'./{old}'

        await _commit_on_main(
            git_ops, git_repo, {old: 'fn topo() {}\n'}, 'add topo_e2e',
        )
        await _rename_on_main(
            git_ops, git_repo, old, new, 'harness: consolidate topo tests',
        )
        wt = (await git_ops.create_worktree('rename-dot-slash')).path
        base = await _head_of(wt)
        (wt / new).write_text('fn topo() { assert!(true); }\n')
        await git_ops.commit(wt, 'step-2')
        head = await _head_of(wt)

        result = await _check_plan_files_touched_in_branch(
            [declared], base, head, git_ops, task_id='rename-dot-slash',
        )

        assert result.not_touched == []
        assert result.missing_from_tree == []
        assert result.resolved_renames == {declared: new}


# ---------------------------------------------------------------------------
# Mechanism 2 — unique-basename fallback
# ---------------------------------------------------------------------------


async def _relocate_as_delete_then_add(
    git_ops: GitOps,
    repo: Path,
    old: str,
    new: str,
) -> None:
    """Relocate *old* → *new* on main as SEPARATE delete and add commits.

    ``git show --name-status -M`` pairs renames only WITHIN a commit, so
    this shape yields no ``R`` pair for the deleting commit and mechanism 1
    is unrecoverable — which is exactly how the reify harness-consolidation
    programme staged its moves.
    """
    rc, _, err = await _run(['git', 'rm', '--quiet', old], cwd=repo)
    assert rc == 0, f'git rm {old} failed: {err}'
    await _commit_on_main(git_ops, repo, {}, f'remove {old}')
    await _commit_on_main(
        git_ops, repo,
        {new: '// relocated + rewritten\nfn topo() { assert_eq!(1, 1); }\n'},
        f'add {new}',
    )


@pytest.mark.asyncio
class TestUniqueBasenameFallback:
    """When git cannot recover a rename PAIR, a UNIQUE basename match in the
    branch tree resolves the declared path — bounded by the requirement that
    the resolved path additionally be in the touched set."""

    async def test_delete_then_add_resolves_by_unique_basename(
        self,
        git_ops: GitOps,
        git_repo: Path,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        old = 'crates/tests/topo_e2e.rs'
        new = 'crates/tests/harness_topo/topo_e2e.rs'

        await _commit_on_main(
            git_ops, git_repo, {old: 'fn topo() {}\n'}, 'add topo_e2e',
        )
        await _relocate_as_delete_then_add(git_ops, git_repo, old, new)

        wt = (await git_ops.create_worktree('basename-fallback')).path
        base = await _head_of(wt)
        (wt / new).write_text('// branch edit\nfn topo() { assert_eq!(2, 2); }\n')
        await git_ops.commit(wt, 'step-2')
        head = await _head_of(wt)

        with caplog.at_level(logging.WARNING, logger='orchestrator.merge_queue'):
            result = await _check_plan_files_touched_in_branch(
                [old], base, head, git_ops, task_id='basename-5196',
            )

        assert result.not_touched == []
        assert result.missing_from_tree == []
        assert result.resolved_renames == {old: new}

        msg = ' '.join(
            r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING
        )
        assert 'basename' in msg.lower(), (
            'the resolution log must name the mechanism that resolved it, so a '
            f'heuristic pass is auditable; got: {msg!r}'
        )
        assert old in msg
        assert new in msg

    async def test_two_stale_paths_share_one_tree_listing(
        self, git_ops: GitOps, git_repo: Path,
    ) -> None:
        """The literal reify-5196 two-file shape.

        Both declared paths were relocated the same way and both new paths
        are touched, so BOTH must resolve off the single tree listing the
        gate computes lazily once per invocation.
        """
        old_a = 'crates/tests/a_e2e.rs'
        old_b = 'crates/tests/b_e2e.rs'
        new_a = 'crates/tests/harness_topo/a_e2e.rs'
        new_b = 'crates/tests/harness_topo/b_e2e.rs'

        await _commit_on_main(
            git_ops, git_repo,
            {old_a: 'fn a() {}\n', old_b: 'fn b() {}\n'},
            'add a_e2e + b_e2e',
        )
        await _relocate_as_delete_then_add(git_ops, git_repo, old_a, new_a)
        await _relocate_as_delete_then_add(git_ops, git_repo, old_b, new_b)

        wt = (await git_ops.create_worktree('basename-fallback-two')).path
        base = await _head_of(wt)
        (wt / new_a).write_text('// branch edit a\n')
        (wt / new_b).write_text('// branch edit b\n')
        await git_ops.commit(wt, 'step-2')
        head = await _head_of(wt)

        result = await _check_plan_files_touched_in_branch(
            [old_a, old_b], base, head, git_ops, task_id='basename-two',
        )

        assert result.not_touched == []
        assert result.missing_from_tree == []
        assert result.resolved_renames == {old_a: new_a, old_b: new_b}


# ---------------------------------------------------------------------------
# Classification — missing_from_tree, and the two no-false-NEGATIVE guards
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestStalePathClassification:
    """An unresolvable declared path still BLOCKS — this class only changes
    the diagnosis, never the routing."""

    async def test_never_existed_path_is_classified_missing_from_tree(
        self, git_ops: GitOps, git_repo: Path,
    ) -> None:
        declared = 'crates/tests/never_created.rs'

        await _commit_on_main(
            git_ops, git_repo, {'crates/tests/real.rs': 'fn real() {}\n'}, 'add real',
        )
        wt = (await git_ops.create_worktree('never-created')).path
        base = await _head_of(wt)
        (wt / 'crates' / 'tests' / 'real.rs').write_text('fn real() { }\n')
        await git_ops.commit(wt, 'step-2')
        head = await _head_of(wt)

        result = await _check_plan_files_touched_in_branch(
            [declared], base, head, git_ops, task_id='never-created',
        )

        assert result.not_touched == [declared], 'the gate must still block'
        assert result.missing_from_tree == [declared], (
            'a declared path absent from the branch tree is a STALE PATH, not '
            'evidence the branch under-delivered'
        )
        assert result.resolved_renames == {}

    async def test_ambiguous_basename_does_not_resolve(
        self, git_ops: GitOps, git_repo: Path,
    ) -> None:
        """Two candidates sharing the basename ⇒ no resolution, ever.

        Guards the fallback against a coincidental basename silently
        satisfying the gate.
        """
        declared = 'crates/tests/dup_e2e.rs'
        first = 'crates/a/dup_e2e.rs'
        second = 'crates/b/dup_e2e.rs'

        await _commit_on_main(
            git_ops, git_repo,
            {first: 'fn a() {}\n', second: 'fn b() {}\n'},
            'add two dup_e2e files',
        )
        wt = (await git_ops.create_worktree('ambiguous-basename')).path
        base = await _head_of(wt)
        (wt / first).write_text('fn a() { /* edited */ }\n')
        await git_ops.commit(wt, 'step-2')
        head = await _head_of(wt)

        result = await _check_plan_files_touched_in_branch(
            [declared], base, head, git_ops, task_id='ambiguous',
        )

        assert result.not_touched == [declared]
        assert result.missing_from_tree == [declared]
        assert result.resolved_renames == {}, (
            'two basename candidates must NOT resolve — a coincidental '
            'basename can never satisfy the gate'
        )

    async def test_resolved_but_untouched_still_blocks_and_is_not_missing(
        self, git_ops: GitOps, git_repo: Path,
    ) -> None:
        """The rename resolved, but the branch touched neither name.

        Still blocks — the branch genuinely did not deliver — but the entry
        must NOT be labelled "does not exist in the tree": the tree WAS
        consulted and the file exists under a known new name.
        """
        old = 'crates/tests/topo_e2e.rs'
        new = 'crates/tests/harness_topo/topo_e2e.rs'

        await _commit_on_main(
            git_ops, git_repo, {old: 'fn topo() {}\n'}, 'add topo_e2e',
        )
        await _rename_on_main(
            git_ops, git_repo, old, new, 'harness: consolidate topo tests',
        )
        wt = (await git_ops.create_worktree('resolved-untouched')).path
        base = await _head_of(wt)
        (wt / 'unrelated.rs').write_text('fn unrelated() {}\n')
        await git_ops.commit(wt, 'step-2')
        head = await _head_of(wt)

        result = await _check_plan_files_touched_in_branch(
            [old], base, head, git_ops, task_id='resolved-untouched',
        )

        assert result.not_touched == [old], (
            'a resolution alone must never pass the gate — the resolved path '
            'must additionally be in the touched set'
        )
        assert result.resolved_renames == {old: new}
        assert result.missing_from_tree == []
