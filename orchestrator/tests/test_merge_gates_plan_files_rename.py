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


def _is_rename_delete_probe(cmd: Sequence[str]) -> bool:
    """True for the rename resolver's ``git log --diff-filter=D ... -- <path>``.

    MUST require ``--diff-filter=D``.  ``_path_existed_in_branch_history``
    also issues a bare ``git log -1 --format=%H <head> -- <path>`` on this
    same code path with no such flag; a predicate that matched both would
    fail THAT history probe closed too, mechanism 2 would then bail for a
    different reason, and a test built on this predicate would pass against
    buggy code without ever exercising the rename-probe arm it targets.
    """
    return list(cmd[:2]) == ['git', 'log'] and '--diff-filter=D' in cmd


def _is_rename_show_probe(cmd: Sequence[str]) -> bool:
    """True for the rename resolver's ``git show --name-status -M <sha>``."""
    return list(cmd[:2]) == ['git', 'show'] and '--name-status' in cmd


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
# Mechanism 2's basename key — the last resolved hop, not the declared path
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestBasenameFallbackUsesLastResolvedHop:
    """Mechanism 2 must try the LAST RESOLVED HOP (``current``) as its
    FIRST basename-lookup key — reached precisely when mechanism 1
    dead-ends on a hop that changed the basename, so the declared name is
    the one name already proven stale.  See
    ``TestDeclaredBasenameFallbackWhenHopHasNoMatch`` for the fallback to
    the originally declared path (``norm``) when the hop key finds no
    unique candidate, and for the tie-break when both keys match.

    The no-chain case (``current == norm``) is already covered by
    ``TestUniqueBasenameFallback.test_delete_then_add_resolves_by_unique_basename``,
    which must stay green — it pins that this class changes nothing when
    the chain never advanced.

    The sibling invariant — that the HISTORY evidence gate
    (:func:`_path_existed_in_branch_history`) stays anchored on the
    originally declared path and does NOT move to the hop — is already
    pinned by
    ``TestStalePathClassification.test_invented_path_does_not_resolve_by_basename``:
    a chain can only ever advance past hop 1 (making ``current != norm``)
    if *norm* itself was deleted by some commit — that is how mechanism 1
    finds a pair to advance on — so "no history under the declared name"
    is necessarily the no-chain (``current == norm``) case that test
    already covers.  Re-anchoring the evidence gate on ``current`` instead
    would make it vacuous (a hop only exists because a commit deleted it),
    which is exactly the regression this reference guards against without
    duplicating the test.
    """

    async def test_dead_ended_chain_resolves_on_the_hop_basename(
        self,
        git_ops: GitOps,
        git_repo: Path,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """The headline false positive this defect produces.

        Hop 1 is a pairable rename that CHANGES the basename
        (``topo_e2e.rs`` -> ``topo_suite.rs``); hop 2 is an unpairable
        delete-then-add, so the chain dead-ends on hop 1's target. Keying
        the basename lookup on the ORIGINAL declared path searches the
        tree for ``topo_e2e.rs`` and finds nothing — the entry is wrongly
        blocked even though the branch delivered exactly what was asked,
        at the file's live name.
        """
        declared = 'crates/tests/topo_e2e.rs'
        hop1 = 'crates/tests/harness_topo/topo_suite.rs'
        final = 'crates/harness/topo_suite.rs'

        await _commit_on_main(
            git_repo, {declared: 'fn topo() {}\n'}, 'add topo_e2e',
        )
        await _rename_on_main(
            git_repo, declared, hop1, 'harness: hop 1 rename + retarget',
        )
        await _relocate_as_delete_then_add(git_repo, hop1, final)

        wt = (await git_ops.create_worktree('dead-ended-hop-basename')).path
        base = await _head_of(wt)
        (wt / final).write_text('// branch edit\nfn topo() { assert_eq!(3, 3); }\n')
        await git_ops.commit(wt, 'step-2')
        head = await _head_of(wt)

        with caplog.at_level(logging.WARNING, logger='orchestrator.merge_queue'):
            result = await _check_plan_files_touched_in_branch(
                [declared], base, head, git_ops, task_id='hop-basename',
            )

        assert result.not_touched == [], (
            'the chain dead-ended on a basename-changing hop, but the branch '
            'delivered against the live path — the gate must PASS'
        )
        assert result.missing_from_tree == []
        assert result.resolved_renames == {declared: final}

        msg = ' '.join(
            r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING
        )
        assert hop1 in msg, (
            'the audit trail must name the HOP the basename matched on, so a '
            'human auditing the PASS can see it is not a coincidental match; '
            f'got: {msg!r}'
        )

    async def test_hop_basename_still_requires_uniqueness(
        self, git_ops: GitOps, git_repo: Path,
    ) -> None:
        """Two candidates sharing the HOP's basename ⇒ still no resolution.

        Guards that this fix does not WEAKEN the heuristic: pins the
        ``len(candidates) == 1`` bound now that the lookup key changed
        from ``norm`` to ``current``.
        """
        declared = 'crates/tests/topo_e2e.rs'
        hop1 = 'crates/tests/harness_topo/topo_suite.rs'
        final = 'crates/harness/topo_suite.rs'
        unrelated = 'crates/other/topo_suite.rs'

        await _commit_on_main(
            git_repo,
            {declared: 'fn topo() {}\n', unrelated: 'fn other() {}\n'},
            'add topo_e2e + unrelated topo_suite',
        )
        await _rename_on_main(
            git_repo, declared, hop1, 'harness: hop 1 rename + retarget',
        )
        await _relocate_as_delete_then_add(git_repo, hop1, final)

        wt = (await git_ops.create_worktree('hop-basename-ambiguous')).path
        base = await _head_of(wt)
        (wt / final).write_text('// branch edit\nfn topo() { assert_eq!(4, 4); }\n')
        await git_ops.commit(wt, 'step-2')
        head = await _head_of(wt)

        result = await _check_plan_files_touched_in_branch(
            [declared], base, head, git_ops, task_id='hop-basename-ambiguous',
        )

        assert result.resolved_renames == {}, (
            'two candidates sharing the hop basename must NOT resolve — a '
            'coincidental basename can never satisfy the gate'
        )
        assert result.not_touched == [declared]
        assert result.missing_from_tree == [declared]


@pytest.mark.asyncio
class TestDeclaredBasenameFallbackWhenHopHasNoMatch:
    """Mechanism 2 tries the LAST RESOLVED HOP's basename FIRST, and falls
    BACK to the ORIGINALLY DECLARED path's basename when the hop key finds
    no unique candidate — rather than replacing one key with the other.

    ``TestBasenameFallbackUsesLastResolvedHop`` (above) pins the case where
    the hop key must win because the declared key finds nothing.  This
    class pins the MIRROR-IMAGE regression: a chain hop can change a file's
    basename and a LATER hop can restore it, so a hop with zero tree
    candidates is not evidence that the declared name has none either.
    Measured on this worktree against real git: keying mechanism 2
    UNCONDITIONALLY on ``current`` (task 4158 step-4) wrongly BLOCKS a
    branch that delivered exactly what was asked, where the pre-step-4
    code (keyed on ``norm``) correctly resolved it. Trying the hop first
    and falling back to the declared path recovers both.

    Read together with ``TestHopBasenameAcceptedTradeoff`` (below), the
    three classes cover: hop-key wins, declared-key fallback, and the
    accepted tradeoff both keys share.
    """

    async def test_hop_with_no_candidates_falls_back_to_the_declared_basename(
        self,
        git_ops: GitOps,
        git_repo: Path,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """The headline regression: the dead-end hop has no tree candidate,
        but the ORIGINALLY DECLARED basename does — because the second hop
        RESTORED it in a new directory.

        Differs from
        ``TestBasenameFallbackUsesLastResolvedHop.test_dead_ended_chain_resolves_on_the_hop_basename``
        only in ``final``'s basename: there it keeps the hop's basename
        (``topo_suite.rs``); here it restores the declared one
        (``topo_e2e.rs``), which is exactly what an unconditional switch to
        ``current`` cannot see.
        """
        declared = 'crates/tests/topo_e2e.rs'
        hop1 = 'crates/tests/harness_topo/topo_suite.rs'
        final = 'crates/harness/topo_e2e.rs'

        await _commit_on_main(
            git_repo, {declared: 'fn topo() {}\n'}, 'add topo_e2e',
        )
        await _rename_on_main(
            git_repo, declared, hop1, 'harness: hop 1 rename + retarget',
        )
        await _relocate_as_delete_then_add(git_repo, hop1, final)

        wt = (await git_ops.create_worktree('hop-no-match-declared-fallback')).path
        base = await _head_of(wt)
        (wt / final).write_text('// branch edit\nfn topo() { assert_eq!(6, 6); }\n')
        await git_ops.commit(wt, 'step-2')
        head = await _head_of(wt)

        with caplog.at_level(logging.WARNING, logger='orchestrator.merge_queue'):
            result = await _check_plan_files_touched_in_branch(
                [declared], base, head, git_ops, task_id='hop-no-match',
            )

        assert result.not_touched == [], (
            'the hop basename has no tree candidate, but the DECLARED '
            'basename does — the gate must fall back and PASS'
        )
        assert result.missing_from_tree == []
        assert result.resolved_renames == {declared: final}

        msg = ' '.join(
            r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING
        )
        assert declared in msg, (
            f'the audit trail must name the declared path that matched; got: {msg!r}'
        )
        assert hop1 in msg, (
            'the audit trail must ALSO name the dead hop that was tried and '
            f'missed, so this PASS is not confused with the hop-key path; '
            f'got: {msg!r}'
        )

    async def test_hop_basename_wins_when_both_keys_have_a_unique_match(
        self,
        git_ops: GitOps,
        git_repo: Path,
    ) -> None:
        """Pins the ORDERING: when both keys resolve uniquely, the HOP wins.

        This is the one genuinely ambiguous case the fallback introduces —
        both ``len(candidates) == 1`` bounds are satisfied, and the
        caller's touched-set requirement cannot break the tie because the
        branch touches BOTH candidates.  The hop is the principled winner:
        mechanism 1 produced AUTHORITATIVE git rename evidence for it,
        whereas ``norm`` is the one name mechanism 1 already PROVED stale.
        Without this test the first-try/fallback order is an accident of
        statement order.
        """
        declared = 'crates/tests/topo_e2e.rs'
        hop1 = 'crates/tests/harness_topo/topo_suite.rs'
        final = 'crates/harness/topo_suite.rs'
        unrelated = 'crates/legacy/topo_e2e.rs'

        await _commit_on_main(
            git_repo, {declared: 'fn topo() {}\n'}, 'add topo_e2e',
        )
        await _rename_on_main(
            git_repo, declared, hop1, 'harness: hop 1 rename + retarget',
        )
        await _relocate_as_delete_then_add(git_repo, hop1, final)
        await _commit_on_main(
            git_repo, {unrelated: 'fn legacy() {}\n'}, 'add unrelated legacy topo_e2e',
        )

        wt = (await git_ops.create_worktree('hop-vs-declared-tiebreak')).path
        base = await _head_of(wt)
        (wt / final).write_text('// branch edit final\nfn topo() { assert_eq!(7, 7); }\n')
        (wt / unrelated).write_text(
            '// branch edit unrelated\nfn legacy() { assert_eq!(8, 8); }\n',
        )
        await git_ops.commit(wt, 'step-2')
        head = await _head_of(wt)

        result = await _check_plan_files_touched_in_branch(
            [declared], base, head, git_ops, task_id='hop-vs-declared',
        )

        assert result.resolved_renames == {declared: final}, (
            'both keys resolve uniquely and both candidates are touched — '
            'the HOP key must win the tie, not the declared-path fallback'
        )
        assert result.not_touched == []
        assert result.missing_from_tree == []

    async def test_declared_basename_fallback_still_requires_uniqueness(
        self,
        git_ops: GitOps,
        git_repo: Path,
    ) -> None:
        """Two candidates sharing the DECLARED basename ⇒ still no
        resolution: the fallback key must not be looser than the hop key.
        """
        declared = 'crates/tests/topo_e2e.rs'
        hop1 = 'crates/tests/harness_topo/topo_suite.rs'
        final = 'crates/harness/topo_e2e.rs'
        unrelated = 'crates/legacy/topo_e2e.rs'

        await _commit_on_main(
            git_repo, {declared: 'fn topo() {}\n'}, 'add topo_e2e',
        )
        await _rename_on_main(
            git_repo, declared, hop1, 'harness: hop 1 rename + retarget',
        )
        await _relocate_as_delete_then_add(git_repo, hop1, final)
        await _commit_on_main(
            git_repo, {unrelated: 'fn legacy() {}\n'}, 'add unrelated legacy topo_e2e',
        )

        wt = (await git_ops.create_worktree('declared-fallback-ambiguous')).path
        base = await _head_of(wt)
        (wt / final).write_text('// branch edit\nfn topo() { assert_eq!(9, 9); }\n')
        await git_ops.commit(wt, 'step-2')
        head = await _head_of(wt)

        result = await _check_plan_files_touched_in_branch(
            [declared], base, head, git_ops, task_id='declared-fallback-ambiguous',
        )

        assert result.resolved_renames == {}, (
            'two candidates sharing the declared basename must NOT resolve — '
            'the len(candidates) == 1 bound applies to the fallback key '
            'exactly as it does to the hop key'
        )
        assert result.not_touched == [declared]
        assert result.missing_from_tree == [declared]


@pytest.mark.asyncio
class TestHopBasenameAcceptedTradeoff:
    """Documents a KNOWN, ACCEPTED limitation of keying mechanism 2's
    basename lookup on ``current`` (the last resolved hop) rather than
    ``norm`` — see the "Known, ACCEPTED limitation" paragraph in
    ``_resolve_renamed_plan_path``'s docstring.

    None of mechanism 2's four bounds can distinguish "the hop's file was
    relocated again as separate delete+add commits" (the case the
    ``current`` key exists to resolve — see
    ``TestBasenameFallbackUsesLastResolvedHop``) from "the hop's file was
    genuinely DELETED OUTRIGHT" (removed, never re-added anywhere).  If an
    UNRELATED file elsewhere in the tree happens to share the dead hop's
    basename, mechanism 2 resolves the declared path to that unrelated
    file even though nothing on the branch delivered against it.

    This is not a NEW class of risk introduced by keying on ``current``:
    the no-chain case (``current == norm``) already has it — a declared
    path that was committed and later deleted outright still has git
    history under its own name, so ``_path_existed_in_branch_history``
    does not stop it either.  Keying on ``current`` only extends the SAME
    accepted tradeoff to chained renames, where a hop's basename is more
    likely to be generic than the originally declared path's.  This test
    pins the CURRENT, INTENTIONAL behaviour so a future change to it is a
    deliberate decision, not an incidental regression.
    """

    async def test_hop_deleted_outright_still_resolves_via_unrelated_basename(
        self,
        git_ops: GitOps,
        git_repo: Path,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        declared = 'crates/tests/topo_e2e.rs'
        hop1 = 'crates/tests/harness_topo/topo_suite.rs'
        unrelated = 'crates/other/topo_suite.rs'

        await _commit_on_main(
            git_repo, {declared: 'fn topo() {}\n'}, 'add topo_e2e',
        )
        await _rename_on_main(
            git_repo, declared, hop1, 'harness: hop 1 rename + retarget',
        )
        # hop1 is deleted OUTRIGHT — no re-add anywhere, unlike
        # `_relocate_as_delete_then_add`'s delete-then-add-elsewhere shape.
        rc, _, err = await _run(['git', 'rm', '--quiet', hop1], cwd=git_repo)
        assert rc == 0, f'git rm {hop1} failed: {err}'
        await _commit_on_main(git_repo, {}, f'remove {hop1} outright')
        # An unrelated file, added independently, happens to share hop1's
        # basename — nothing connects it to the topo_e2e relocation.
        await _commit_on_main(
            git_repo, {unrelated: 'fn other() {}\n'}, 'add unrelated file',
        )

        wt = (await git_ops.create_worktree('hop-deleted-outright')).path
        base = await _head_of(wt)
        (wt / unrelated).write_text('fn other() { /* branch edit */ }\n')
        await git_ops.commit(wt, 'step-2')
        head = await _head_of(wt)

        with caplog.at_level(logging.WARNING, logger='orchestrator.merge_queue'):
            result = await _check_plan_files_touched_in_branch(
                [declared], base, head, git_ops, task_id='hop-deleted-outright',
            )

        # This is the ACCEPTED tradeoff documented on
        # `_resolve_renamed_plan_path`, not a desired outcome: nothing on
        # the branch actually delivered against `declared`, yet the gate
        # PASSES because the dead hop's basename coincidentally matches an
        # unrelated file the branch happened to touch.
        assert result.resolved_renames == {declared: unrelated}, (
            'pins the accepted tradeoff: a coincidental basename match on '
            'the dead hop resolves even though hop1 was deleted outright, '
            'not relocated'
        )
        assert result.not_touched == []
        assert result.missing_from_tree == []

        # Not silent, even though it is a coincidental match: every
        # resolution that passes the gate is logged loudly.
        msg = ' '.join(
            r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING
        )
        assert 'gate PASSES' in msg, (
            f'a resolution that passes the gate must still be logged loudly, '
            f'even under this accepted tradeoff; got: {msg!r}'
        )
        assert declared in msg
        assert unrelated in msg


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

    async def test_rename_delete_probe_error_does_not_fall_through_to_basename(
        self,
        git_ops: GitOps,
        git_repo: Path,
        caplog: pytest.LogCaptureFixture,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A git error on the FIRST rename-probe arm must fail CLOSED.

        Identical repo shape to
        ``TestUniqueBasenameFallback.test_delete_then_add_resolves_by_unique_basename``
        — mechanism 1 is unrecoverable (separate delete/add commits) but
        mechanism 2 WOULD resolve this via a unique basename match.  Forcing
        the ``git log --diff-filter=D`` probe itself to error is a DIFFERENT
        fact from "mechanism 1 found no pair": the resolution is
        unmeasurable, not merely inconclusive, so the gate must not silently
        fall through to the basename heuristic and must block instead.
        """
        old = 'crates/tests/topo_e2e.rs'
        new = 'crates/tests/harness_topo/topo_e2e.rs'

        await _commit_on_main(
            git_repo, {old: 'fn topo() {}\n'}, 'add topo_e2e',
        )
        await _relocate_as_delete_then_add(git_repo, old, new)

        wt = (await git_ops.create_worktree('rename-delete-probe-error')).path
        base = await _head_of(wt)
        (wt / new).write_text('// branch edit\nfn topo() { assert_eq!(2, 2); }\n')
        await git_ops.commit(wt, 'step-2')
        head = await _head_of(wt)

        spy = _RunSpy(fail_when=_is_rename_delete_probe)
        monkeypatch.setattr('orchestrator.merge_gates._run', spy)
        with caplog.at_level(logging.WARNING, logger='orchestrator.merge_queue'):
            result = await _check_plan_files_touched_in_branch(
                [old], base, head, git_ops, task_id='rename-delete-error',
            )

        assert spy.count(_is_rename_delete_probe) >= 1, 'the fault must have fired'
        assert result.not_touched == [old], (
            'an unmeasurable rename probe still blocks — the gate fails CLOSED'
        )
        assert result.resolved_renames == {}, (
            'a git error on the rename probe must never fall through to the '
            'basename heuristic'
        )
        assert result.missing_from_tree == [old], (
            'the existence probe DID succeed and DID measure a genuine absence; '
            'only the rename RESOLUTION was unmeasurable, unlike '
            'test_existence_probe_error_is_not_classified_missing'
        )
        msg = ' '.join(
            r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING
        )
        assert 'resolution abandoned' in msg, (
            f'the degradation must be loud about abandoning resolution, not just '
            f'the per-probe error; got: {msg!r}'
        )

    async def test_rename_show_probe_error_does_not_fall_through_to_basename(
        self,
        git_ops: GitOps,
        git_repo: Path,
        caplog: pytest.LogCaptureFixture,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A git error on the SECOND rename-probe arm must fail CLOSED.

        Identical repo shape to
        ``TestRenamedOnMainResolvesViaDeletingCommit.test_renamed_on_main_gate_passes``
        — a single ``git mv`` commit, so ``git log --diff-filter=D`` finds
        the deleting commit and the ``git show --name-status -M`` call is
        actually reached.  Forcing THAT call to error must not silently
        degrade to the basename heuristic either, even though it too would
        resolve this shape (identical basename on both sides of the move).
        """
        old = 'crates/tests/topo_e2e.rs'
        new = 'crates/tests/harness_topo/topo_e2e.rs'

        await _commit_on_main(
            git_repo, {old: 'fn topo() {}\n'}, 'add topo_e2e',
        )
        await _rename_on_main(
            git_repo, old, new, 'harness: consolidate topo tests',
        )

        wt = (await git_ops.create_worktree('rename-show-probe-error')).path
        base = await _head_of(wt)
        (wt / new).write_text('fn topo() { assert!(true); }\n')
        await git_ops.commit(wt, 'step-2')
        head = await _head_of(wt)

        spy = _RunSpy(fail_when=_is_rename_show_probe)
        monkeypatch.setattr('orchestrator.merge_gates._run', spy)
        with caplog.at_level(logging.WARNING, logger='orchestrator.merge_queue'):
            result = await _check_plan_files_touched_in_branch(
                [old], base, head, git_ops, task_id='rename-show-error',
            )

        assert spy.count(_is_rename_show_probe) >= 1, 'the fault must have fired'
        assert result.not_touched == [old], (
            'an unmeasurable rename probe still blocks — the gate fails CLOSED'
        )
        assert result.resolved_renames == {}, (
            'a git error on the rename probe must never fall through to the '
            'basename heuristic'
        )
        assert result.missing_from_tree == [old], (
            'the existence probe DID succeed and DID measure a genuine absence; '
            'only the rename RESOLUTION was unmeasurable, unlike '
            'test_existence_probe_error_is_not_classified_missing'
        )
        msg = ' '.join(
            r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING
        )
        assert 'resolution abandoned' in msg, (
            f'the degradation must be loud about abandoning resolution, not just '
            f'the per-probe error; got: {msg!r}'
        )

    async def test_rename_delete_probe_error_after_chain_advanced_names_hop_and_declared(
        self,
        git_ops: GitOps,
        git_repo: Path,
        caplog: pytest.LogCaptureFixture,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A git error on HOP 2's delete probe — after the chain already
        advanced past hop 1, so ``current != norm`` — must still fail
        CLOSED.

        The two tests above both inject the fault on hop 1, where
        ``current == norm`` at the point the sentinel is returned, so
        neither could catch a regression that moved the
        ``isinstance(pair, _RenameProbeUnmeasurable)`` check outside the
        loop, or that logged ``norm`` twice instead of naming the hop that
        was actually being probed when resolution failed.  Reuses the
        ``declared -> hop1 -> final`` shape from
        ``TestBasenameFallbackUsesLastResolvedHop``, but the fault fires
        only on the SECOND ``git log --diff-filter=D`` call (keyed on
        ``hop1``), not the first (keyed on ``declared``).
        """
        declared = 'crates/tests/topo_e2e.rs'
        hop1 = 'crates/tests/harness_topo/topo_suite.rs'
        final = 'crates/harness/topo_suite.rs'

        await _commit_on_main(
            git_repo, {declared: 'fn topo() {}\n'}, 'add topo_e2e',
        )
        await _rename_on_main(
            git_repo, declared, hop1, 'harness: hop 1 rename + retarget',
        )
        await _relocate_as_delete_then_add(git_repo, hop1, final)

        wt = (await git_ops.create_worktree('rename-delete-probe-error-hop2')).path
        base = await _head_of(wt)
        (wt / final).write_text('// branch edit\nfn topo() { assert_eq!(5, 5); }\n')
        await git_ops.commit(wt, 'step-2')
        head = await _head_of(wt)

        def _is_declared_delete_probe(cmd: Sequence[str]) -> bool:
            """The FIRST delete probe: what deleted `declared` (resolves hop 1)."""
            return _is_rename_delete_probe(cmd) and declared in cmd

        def _is_hop1_delete_probe(cmd: Sequence[str]) -> bool:
            """The SECOND delete probe: what deleted `hop1` (resolves hop 2)."""
            return _is_rename_delete_probe(cmd) and hop1 in cmd

        spy = _RunSpy(fail_when=_is_hop1_delete_probe)
        monkeypatch.setattr('orchestrator.merge_gates._run', spy)
        with caplog.at_level(logging.WARNING, logger='orchestrator.merge_queue'):
            result = await _check_plan_files_touched_in_branch(
                [declared], base, head, git_ops, task_id='rename-delete-error-hop2',
            )

        # Sanity: the probe for `declared` ran for real and succeeded —
        # otherwise the chain could never have advanced past hop 1, and
        # this test would silently degenerate into the current==norm case
        # the prior two tests already cover.
        assert spy.count(_is_declared_delete_probe) >= 1, (
            'the probe for `declared` must have run for real for the chain '
            'to advance past hop 1'
        )
        assert spy.count(_is_hop1_delete_probe) >= 1, (
            'the fault must have fired on the SECOND (hop1 -> hop2) delete probe'
        )
        assert result.not_touched == [declared], (
            'an unmeasurable rename probe still blocks — the gate fails CLOSED'
        )
        assert result.resolved_renames == {}, (
            'a git error on the rename probe must never fall through to the '
            'basename heuristic, even after the chain has already advanced'
        )
        assert result.missing_from_tree == [declared], (
            'the existence probe DID succeed and DID measure a genuine absence; '
            'only the rename RESOLUTION was unmeasurable'
        )

        msg = ' '.join(
            r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING
        )
        assert hop1 in msg, (
            'the warning must name the HOP (`current`) where resolution was '
            f'abandoned, not just the originally declared path; got: {msg!r}'
        )
        assert declared in msg, (
            f'the warning must ALSO name the originally declared path; got: {msg!r}'
        )
