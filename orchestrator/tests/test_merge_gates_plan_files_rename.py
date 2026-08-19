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

These tests exercise the gate against a REAL git repository, so they live
in this dedicated file rather than in the 15k-line ``test_merge_queue.py``.
The only deviation from pure real-git is :class:`_RunSpy`, a delegating
wrapper around ``merge_gates._run`` used narrowly for the two properties
real git cannot produce on demand: how many times a command was shelled
out (the tree-listing memo), and a non-zero rc from a specific git
subcommand (the two fail-closed arms).  Every other command still runs
for real through it.  They import the gate from
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
from collections.abc import Callable, Sequence
from pathlib import Path

import pytest

from orchestrator.config import GitConfig
from orchestrator.git_ops import GitOps, _run
from orchestrator.merge_gates import (
    PlanFilesTouchedResult,
    _check_plan_files_touched_in_branch,
    _ls_tree_object_type,
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
# _RunSpy — the file's one, narrowly-scoped deviation from pure real-git
# ---------------------------------------------------------------------------


class _RunSpy:
    """Delegating wrapper around ``merge_gates._run`` that records every call.

    Two gate properties are invisible to a pure real-git test: how MANY
    times a command was shelled out (the tree-listing memo — deleting it
    entirely leaves every behavioural assertion green), and what the gate
    does when a specific git subcommand returns a non-zero rc (the two
    fail-closed arms, which real git will not produce on demand).  This
    spy covers exactly those, and delegates everything else to the real
    ``_run``, so the surrounding repository work stays real.

    *fail_when* is an optional predicate over the argv list; a command it
    matches returns ``(128, '', <fatal>)`` without being executed.
    """

    def __init__(
        self, fail_when: Callable[[Sequence[str]], bool] | None = None,
    ) -> None:
        self.calls: list[list[str]] = []
        self._fail_when = fail_when

    async def __call__(
        self, cmd: list[str], cwd: Path | None = None, **kwargs,
    ) -> tuple[int, str, str]:
        self.calls.append(list(cmd))
        if self._fail_when is not None and self._fail_when(cmd):
            return 128, '', 'fatal: injected failure (test fault injection)\n'
        return await _run(cmd, cwd, **kwargs)

    def count(self, predicate: Callable[[Sequence[str]], bool]) -> int:
        """Number of recorded calls whose argv satisfies *predicate*."""
        return sum(1 for cmd in self.calls if predicate(cmd))


def _is_tree_listing(cmd: Sequence[str]) -> bool:
    """True for the resolver's shared ``git ls-tree -r --name-only <head>``."""
    return list(cmd[:4]) == ['git', 'ls-tree', '-r', '--name-only']


def _is_existence_probe(cmd: Sequence[str]) -> bool:
    """True for the per-entry ``git ls-tree <head> -- <path>`` existence probe."""
    return list(cmd[:2]) == ['git', 'ls-tree'] and '--' in cmd and '-r' not in cmd


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


# ---------------------------------------------------------------------------
# ls-tree object-type parsing
# ---------------------------------------------------------------------------


class TestLsTreeObjectType:
    """The existence probe reads the object type out of the TAB-separated
    METADATA half, never by sniffing the whole line for ``' tree '``.

    The type classification gates three arms (directory prefix-match /
    blob / absent-then-rename-resolve), so a blob whose repo-relative path
    merely CONTAINS ``' tree '`` must not be steered onto the directory
    arm.
    """

    def test_blob(self) -> None:
        assert _ls_tree_object_type(
            '100644 blob e69de29bb2d1d6434b8b29ae775ad8c2e48c5391\tsrc/a.py\n',
        ) == 'blob'

    def test_tree(self) -> None:
        assert _ls_tree_object_type(
            '040000 tree 4b825dc642cb6eb9a060e54bf8d69288fbee4904\tsrc/pkg\n',
        ) == 'tree'

    def test_blob_whose_path_contains_the_tree_substring(self) -> None:
        """The regression this parser exists for."""
        line = (
            '100644 blob e69de29bb2d1d6434b8b29ae775ad8c2e48c5391'
            '\tdocs/my tree notes.md\n'
        )
        assert ' tree ' in line, 'fixture must contain the misleading substring'
        assert _ls_tree_object_type(line) == 'blob'

    def test_absent_path_yields_none(self) -> None:
        """``git ls-tree`` exits 0 with EMPTY output for an absent path."""
        assert _ls_tree_object_type('') is None
        assert _ls_tree_object_type('\n') is None

    def test_unparseable_line_yields_none(self) -> None:
        """Defensive: an unrecognised shape is read as 'not a directory'."""
        assert _ls_tree_object_type('garbage\n') is None


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
            git_repo, {old: 'fn topo() {}\n'}, 'add topo_e2e',
        )
        # 2. Main relocates it (identical basename — the harness-layout
        #    consolidation shape), before any branch exists.
        rename_sha = await _rename_on_main(
            git_repo, old, new, 'harness: consolidate topo tests',
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
            git_repo, {old: 'fn topo() {}\n'}, 'add topo_e2e',
        )
        await _rename_on_main(
            git_repo, old, new, 'harness: consolidate topo tests',
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

    async def test_multi_hop_rename_chain_resolves_to_the_live_path(
        self,
        git_ops: GitOps,
        git_repo: Path,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """A relocation staged as SEVERAL ``git mv`` commits on main.

        ``a → b`` then ``b → c``; the branch edits ``c``.  Stopping at the
        first rename pair would answer with the DEAD intermediate ``b`` —
        blocking a branch that delivered exactly what was asked, and, worse,
        naming in the audit trail a second path that does not exist at
        branch HEAD either.  The resolver must chase the chain to the path
        that actually lives in the tree.  Staged multi-step moves are the
        shape the cited reify harness-consolidation programme used, so this
        is not hypothetical.
        """
        first = 'crates/tests/a_e2e.rs'
        middle = 'crates/harness/a_e2e.rs'
        final = 'crates/harness_topo/a_e2e.rs'

        await _commit_on_main(git_repo, {first: 'fn a() {}\n'}, 'add a_e2e')
        await _rename_on_main(git_repo, first, middle, 'harness: hop 1')
        last_sha = await _rename_on_main(git_repo, middle, final, 'harness: hop 2')

        wt = (await git_ops.create_worktree('multi-hop')).path
        base = await _head_of(wt)
        (wt / final).write_text('// branch edit\nfn a() { assert!(true); }\n')
        await git_ops.commit(wt, 'step-2')
        head = await _head_of(wt)

        with caplog.at_level(logging.WARNING, logger='orchestrator.merge_queue'):
            result = await _check_plan_files_touched_in_branch(
                [first], base, head, git_ops, task_id='multi-hop',
            )

        assert result.not_touched == [], (
            'a two-hop relocation on main is one logical move — the branch '
            'delivered against the final path and the gate must PASS'
        )
        assert result.missing_from_tree == []
        assert result.resolved_renames == {first: final}, (
            'the audit trail must name the LIVE path, never the dead '
            f'intermediate {middle!r}'
        )

        msg = ' '.join(
            r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING
        )
        assert middle not in msg, (
            'no log line may point a human at the dead intermediate path'
        )
        assert final in msg
        assert last_sha[:12] in msg, (
            'the mechanism must name the LAST hop, so the chain is auditable'
        )


# ---------------------------------------------------------------------------
# Mechanism 2 — unique-basename fallback
# ---------------------------------------------------------------------------


async def _relocate_as_delete_then_add(
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
    await _commit_on_main(repo, {}, f'remove {old}')
    await _commit_on_main(
        repo,
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
            git_repo, {old: 'fn topo() {}\n'}, 'add topo_e2e',
        )
        await _relocate_as_delete_then_add(git_repo, old, new)

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
        self,
        git_ops: GitOps,
        git_repo: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """The literal reify-5196 two-file shape, plus the memo it shares.

        Both declared paths were relocated the same way and both new paths
        are touched, so BOTH must resolve — and both must resolve off ONE
        ``git ls-tree -r --name-only``.  The invocation count is asserted
        directly: without it, deleting the ``tree_listing`` memo entirely
        (re-shelling per stale entry) would leave every behavioural
        assertion below green, so the caching this test is named for would
        be unprotected.
        """
        old_a = 'crates/tests/a_e2e.rs'
        old_b = 'crates/tests/b_e2e.rs'
        new_a = 'crates/tests/harness_topo/a_e2e.rs'
        new_b = 'crates/tests/harness_topo/b_e2e.rs'

        await _commit_on_main(
            git_repo,
            {old_a: 'fn a() {}\n', old_b: 'fn b() {}\n'},
            'add a_e2e + b_e2e',
        )
        await _relocate_as_delete_then_add(git_repo, old_a, new_a)
        await _relocate_as_delete_then_add(git_repo, old_b, new_b)

        wt = (await git_ops.create_worktree('basename-fallback-two')).path
        base = await _head_of(wt)
        (wt / new_a).write_text('// branch edit a\n')
        (wt / new_b).write_text('// branch edit b\n')
        await git_ops.commit(wt, 'step-2')
        head = await _head_of(wt)

        spy = _RunSpy()
        monkeypatch.setattr('orchestrator.merge_gates._run', spy)
        result = await _check_plan_files_touched_in_branch(
            [old_a, old_b], base, head, git_ops, task_id='basename-two',
        )

        assert result.not_touched == []
        assert result.missing_from_tree == []
        assert result.resolved_renames == {old_a: new_a, old_b: new_b}

        assert spy.count(_is_tree_listing) == 1, (
            'the tree listing must be shelled out exactly ONCE per gate '
            'invocation and shared across every stale entry; got '
            f'{spy.count(_is_tree_listing)} calls'
        )
        # Sanity: the spy really is on the resolution path (both entries were
        # probed for existence through it), so the count above is meaningful.
        assert spy.count(_is_existence_probe) == 2


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
            git_repo, {'crates/tests/real.rs': 'fn real() {}\n'}, 'add real',
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
            git_repo,
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
            git_repo, {old: 'fn topo() {}\n'}, 'add topo_e2e',
        )
        await _rename_on_main(
            git_repo, old, new, 'harness: consolidate topo tests',
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

    async def test_invented_path_does_not_resolve_by_basename(
        self, git_ops: GitOps, git_repo: Path,
    ) -> None:
        """A path with NO history under its declared name is never resolved.

        The basename fallback is a *rename* resolver; a path the architect
        simply invented was never renamed, so calling a coincidentally-unique
        same-basename file its "resolution" is a category error that would
        silently pass the gate on a declaration nothing ever delivered.
        Distinct from ``test_never_existed_path_is_classified_missing_from_tree``,
        where no basename candidate exists at all: here a UNIQUE candidate
        exists AND the branch touched it, so only the history evidence gate
        can stop it.
        """
        declared = 'docs/spec.md'
        elsewhere = 'notes/archive/spec.md'

        await _commit_on_main(
            git_repo, {elsewhere: '# real, unrelated\n'}, 'add archive spec',
        )
        wt = (await git_ops.create_worktree('invented-path')).path
        base = await _head_of(wt)
        (wt / elsewhere).write_text('# real, unrelated — edited\n')
        await git_ops.commit(wt, 'step-2')
        head = await _head_of(wt)

        result = await _check_plan_files_touched_in_branch(
            [declared], base, head, git_ops, task_id='invented',
        )

        assert result.resolved_renames == {}, (
            f'{declared!r} never existed, so {elsewhere!r} is not its rename — '
            'a unique basename must not be enough on its own'
        )
        assert result.not_touched == [declared], 'the gate must still block'
        assert result.missing_from_tree == [declared]

    async def test_mixed_stale_and_untouched_partitions_correctly(
        self, git_ops: GitOps, git_repo: Path,
    ) -> None:
        """The MIXED shape workflow.py's class-scoped message depends on.

        One declared path EXISTS in the branch tree but no commit touched it
        (a genuine under-delivery); a second is stale and unresolvable.  Both
        block, but only the stale one is ``missing_from_tree`` — and the
        documented subset invariant is asserted on the GATE's own output, not
        on a hand-built result.
        """
        existing = 'crates/tests/real.rs'
        stale = 'crates/tests/gone.rs'
        delivered = 'crates/tests/delivered.rs'

        await _commit_on_main(
            git_repo,
            {existing: 'fn real() {}\n', delivered: 'fn delivered() {}\n'},
            'add real + delivered',
        )
        wt = (await git_ops.create_worktree('mixed-classes')).path
        base = await _head_of(wt)
        (wt / delivered).write_text('fn delivered() { assert!(true); }\n')
        await git_ops.commit(wt, 'step-2')
        head = await _head_of(wt)

        result = await _check_plan_files_touched_in_branch(
            [existing, stale], base, head, git_ops, task_id='mixed',
        )

        assert result.not_touched == [existing, stale], (
            'both classes still block, in declaration order'
        )
        assert result.missing_from_tree == [stale], (
            'only the path absent from the branch tree is a stale declaration; '
            'the existing-but-untouched path is a real under-delivery'
        )
        assert set(result.missing_from_tree) <= set(result.not_touched), (
            'the documented subset invariant — a stale entry is reported in '
            'BOTH lists, never moved out of not_touched, so every existing '
            'not_touched-only consumer keeps its current verdict'
        )
        assert result.resolved_renames == {}


# ---------------------------------------------------------------------------
# Fail-closed arms — reachable only by forcing a git error
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestGitErrorArmsFailClosed:
    """Both probe failures degrade LOUDLY and conservatively: the gate still
    blocks, and it never asserts a fact it could not measure."""

    async def test_existence_probe_error_is_not_classified_missing(
        self,
        git_ops: GitOps,
        git_repo: Path,
        caplog: pytest.LogCaptureFixture,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """``ls-tree`` probe rc != 0 ⇒ blocked, but NOT ``missing_from_tree``.

        This is the headline "never claim an absence we did not measure"
        rule: without a successful existence probe the gate cannot know
        whether the declared path is stale, so it must not say so.
        """
        declared = 'crates/tests/topo_e2e.rs'

        await _commit_on_main(
            git_repo, {declared: 'fn topo() {}\n'}, 'add topo_e2e',
        )
        wt = (await git_ops.create_worktree('probe-error')).path
        base = await _head_of(wt)
        (wt / 'unrelated.rs').write_text('fn unrelated() {}\n')
        await git_ops.commit(wt, 'step-2')
        head = await _head_of(wt)

        spy = _RunSpy(fail_when=_is_existence_probe)
        monkeypatch.setattr('orchestrator.merge_gates._run', spy)
        with caplog.at_level(logging.WARNING, logger='orchestrator.merge_queue'):
            result = await _check_plan_files_touched_in_branch(
                [declared], base, head, git_ops, task_id='probe-error',
            )

        assert spy.count(_is_existence_probe) == 1, 'the fault must have fired'
        assert result.not_touched == [declared], (
            'an unmeasurable entry still blocks — the gate fails CLOSED'
        )
        assert result.missing_from_tree == [], (
            'absence was never established, so the entry must NOT be labelled '
            'a stale path'
        )
        assert result.resolved_renames == {}
        msg = ' '.join(
            r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING
        )
        assert 'ls-tree probe failed' in msg, (
            f'the degradation must be loud, not silent; got: {msg!r}'
        )

    async def test_tree_listing_error_blocks_every_stale_entry_once(
        self,
        git_ops: GitOps,
        git_repo: Path,
        caplog: pytest.LogCaptureFixture,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """No tree listing ⇒ nothing resolves, and the memo does not retry.

        Without the listing neither the hop-liveness check nor the basename
        bound can be evaluated, so the resolver fails closed for every stale
        entry.  The memo latches the failure: it must be attempted ONCE per
        gate invocation, not once per stale entry.
        """
        old_a = 'crates/tests/a_e2e.rs'
        old_b = 'crates/tests/b_e2e.rs'
        new_a = 'crates/tests/harness_topo/a_e2e.rs'
        new_b = 'crates/tests/harness_topo/b_e2e.rs'

        await _commit_on_main(
            git_repo,
            {old_a: 'fn a() {}\n', old_b: 'fn b() {}\n'},
            'add a_e2e + b_e2e',
        )
        await _relocate_as_delete_then_add(git_repo, old_a, new_a)
        await _relocate_as_delete_then_add(git_repo, old_b, new_b)

        wt = (await git_ops.create_worktree('tree-listing-error')).path
        base = await _head_of(wt)
        (wt / new_a).write_text('// branch edit a\n')
        (wt / new_b).write_text('// branch edit b\n')
        await git_ops.commit(wt, 'step-2')
        head = await _head_of(wt)

        spy = _RunSpy(fail_when=_is_tree_listing)
        monkeypatch.setattr('orchestrator.merge_gates._run', spy)
        with caplog.at_level(logging.WARNING, logger='orchestrator.merge_queue'):
            result = await _check_plan_files_touched_in_branch(
                [old_a, old_b], base, head, git_ops, task_id='tree-listing-error',
            )

        assert spy.count(_is_tree_listing) == 1, (
            'the failed tree listing must be LATCHED, not retried per stale '
            f'entry; got {spy.count(_is_tree_listing)} calls'
        )
        assert result.not_touched == [old_a, old_b], (
            'without a tree listing nothing resolves — the gate fails CLOSED'
        )
        assert result.missing_from_tree == [old_a, old_b], (
            'the existence probe DID measure the absence, so both entries are '
            'correctly classified as stale paths even though resolution failed'
        )
        assert result.resolved_renames == {}
        msg = ' '.join(
            r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING
        )
        assert 'tree-listing failed' in msg, (
            f'the degradation must be loud, not silent; got: {msg!r}'
        )
