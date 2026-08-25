"""Boundary suite for ``landing_evidence.branch_work_landed`` (task 4647).

The task's user-observable signal is "boundary rows B1--B6 green against REAL
repo fixtures", so every case in this file runs against a genuine temporary git
repository rather than a ``MagicMock`` ``git_ops``.  That is deliberate and not
merely thorough: the producer under test is defined by what ``git cherry``,
``git merge-base`` and ``git diff --quiet`` actually do to rebased,
sync-merged and no-op-merged histories, and a mock would only re-assert this
file's own assumptions about them.

Fixture scaffolding is MODULE-LOCAL and modelled verbatim on
``test_git_ops.py``'s ``git_repo`` / ``git_config`` / ``git_ops`` triple and
its ``_commit_all`` helper.  Both isolation guards from ``_orch_helpers`` are
applied to every synthetic repo (esc-3072-3): ``assert_isolated_git_repo``
runs BEFORE any subprocess so a rejected call writes nothing anywhere, and
``git_env_with_ceiling`` caps git's upward repo discovery at the fixture root
so escape is impossible at the git level even if the pre-flight is ever
refactored away.  Without both, a pytest basetemp living inside a live task
worktree would send ``git commit`` into the enclosing dark-factory checkout.

The boundary rows the scenario builders below exist to construct:

* **B1** non-decay — a landed branch followed by N unrelated commits that
  TOUCH THE SAME PATHS (:func:`advance_main_touching_same_paths`).
* **B2** sync-merge tip — a branch whose tip is a conflict-resolution merge
  whose ``parents[1]`` is main (:func:`build_sync_merge_branch`).
* **B3** rebase landing — the branch's work on main with REWRITTEN shas, no
  merge commit and no citation (:func:`build_rebased_landing`).
* **B4** no-op landing — a merge commit on main whose net branch diff is
  empty (:func:`build_no_op_landing`).
* **B5** genuinely unlanded — a branch whose commits are absent from main
  (:func:`build_unlanded_branch`).
* **B6** degenerate branch — a tip equal to the recorded ``branch_base_sha``
  (:func:`build_degenerate_branch`).
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass, field
from pathlib import Path

import pytest
from _orch_helpers import (
    NonIsolatedGitRepoError,
    assert_isolated_git_repo,
    git_env_with_ceiling,
)

from orchestrator.config import GitConfig
from orchestrator.git_ops import GitOps

TASK_ID = '4647'
BRANCH = f'task/{TASK_ID}'

#: The two paths every scenario's task work touches.  Shared by construction so
#: B1 can rewrite exactly the lines the task added.
WORK_PATH = 'pkg/feature.py'
OTHER_PATH = 'pkg/helper.py'


def _numbered(prefix: str, count: int, *, start: int = 0) -> str:
    """Return *count* unique, non-blank lines.

    Uniqueness is load-bearing rather than cosmetic (the same reasoning as
    ``test_git_ops.py``'s helper of the same name): patch-ids and line-survival
    are both computed over line CONTENT, so duplicated or short common lines
    can be "present" by pure coincidence and a fixture built from them would
    measure coincidence instead of containment.
    """
    return ''.join(f'{prefix}_{i:04d} = {i * 7919}\n' for i in range(start, start + count))


class _Repo:
    """A synthetic git repository with the isolation guards always applied.

    Every git invocation goes through :meth:`git`, so there is exactly one
    place where ``cwd`` and the ceiling environment are established and no
    scenario builder can accidentally shell out unguarded.
    """

    def __init__(self, root: Path) -> None:
        # FIRST, before any subprocess: a rejected root must write nothing.
        assert_isolated_git_repo(root)
        self.root = root
        self._env = git_env_with_ceiling(root)

    @classmethod
    def init(cls, root: Path) -> _Repo:
        """``git init`` *root* and return the guarded wrapper for it.

        The pre-flight cannot run before ``git init`` — it refuses any
        directory that is not ALREADY a repo root, which is precisely what
        this call creates.  The ceiling environment is therefore applied to
        the bootstrap invocation on its own, and ``git init`` only ever
        writes into its own ``cwd``, so no upward escape is possible even in
        the one call the pre-flight cannot cover.
        """
        proc = subprocess.run(
            ['git', 'init', '-b', 'main'], cwd=str(root), capture_output=True,
            env=git_env_with_ceiling(root), text=True, check=False,
        )
        assert proc.returncode == 0, f'git init failed: {proc.stderr.strip()}'
        return cls(root)

    def git(self, *args: str, check: bool = True) -> str:
        proc = subprocess.run(
            ['git', *args], cwd=str(self.root), capture_output=True,
            env=self._env, text=True, check=False,
        )
        if check:
            assert proc.returncode == 0, (
                f'git {" ".join(args)} failed (rc={proc.returncode}): '
                f'{proc.stderr.strip()}'
            )
        return proc.stdout.strip()

    def git_rc(self, *args: str) -> tuple[int, str]:
        proc = subprocess.run(
            ['git', *args], cwd=str(self.root), capture_output=True,
            env=self._env, text=True, check=False,
        )
        return proc.returncode, proc.stdout.strip()

    def write(self, rel: str, content: str) -> None:
        target = self.root / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content)

    def commit(self, message: str, files: dict[str, str] | None = None) -> str:
        """Write *files*, stage everything and commit; return the new sha."""
        for rel, content in (files or {}).items():
            self.write(rel, content)
        self.git('add', '-A')
        self.git('commit', '-m', message)
        return self.sha('HEAD')

    def commit_empty(self, message: str) -> str:
        self.git('commit', '--allow-empty', '-m', message)
        return self.sha('HEAD')

    def sha(self, ref: str) -> str:
        return self.git('rev-parse', ref)

    def parents(self, ref: str) -> list[str]:
        return self.git('rev-list', '--parents', '-n', '1', ref).split()[1:]

    def subjects(self, ref: str) -> list[str]:
        out = self.git('log', '--format=%s', ref)
        return out.splitlines() if out else []


@dataclass
class Scenario:
    """What a scenario builder hands the test: refs plus the task metadata."""

    repo: _Repo
    branch: str = BRANCH
    branch_tip_sha: str = ''
    branch_base_sha: str = ''
    main_sha: str = ''
    #: The commit on main that carries this task's work, when one exists.
    landed_sha: str | None = None
    #: The paths the task's work touched, for B1's same-path churn.
    work_paths: tuple[str, ...] = (WORK_PATH, OTHER_PATH)
    metadata: dict[str, object] = field(default_factory=dict)

    def refresh(self) -> None:
        self.branch_tip_sha = self.repo.sha(self.branch)
        self.main_sha = self.repo.sha('main')


@pytest.fixture
def git_repo(tmp_path: Path) -> Path:
    """A temporary git repository with an initial commit on ``main``."""
    root = tmp_path / 'repo'
    root.mkdir()
    repo = _Repo.init(root)
    repo.git('config', 'user.email', 'test@test.com')
    repo.git('config', 'user.name', 'Test')
    repo.commit('chore: initial commit', {'README.md': '# Test\n'})
    return root


@pytest.fixture
def git_config() -> GitConfig:
    return GitConfig(
        main_branch='main',
        branch_prefix='task/',
        remote='origin',
        worktree_dir='.worktrees',
        push_after_advance=False,
    )


@pytest.fixture
def git_ops(git_config: GitConfig, git_repo: Path) -> GitOps:
    return GitOps(git_config, git_repo)


@pytest.fixture
def repo(git_repo: Path) -> _Repo:
    """The guarded command wrapper for the same repository ``git_ops`` uses."""
    return _Repo(git_repo)


# --------------------------------------------------------------------------
# Scenario builders — one per boundary row.
# --------------------------------------------------------------------------


def _seed_branch(repo: _Repo, *, branch: str = BRANCH) -> tuple[str, str]:
    """Create *branch* off main with two content-bearing commits.

    Returns ``(branch_base_sha, branch_tip_sha)``.
    """
    base = repo.sha('main')
    repo.git('checkout', '-b', branch)
    repo.commit(
        f'feat({TASK_ID}): add the feature module',
        {WORK_PATH: _numbered('feature', 40)},
    )
    repo.commit(
        f'test({TASK_ID}): cover the feature module',
        {OTHER_PATH: _numbered('helper', 30)},
    )
    tip = repo.sha('HEAD')
    repo.git('checkout', 'main')
    return base, tip


def build_merged_landing(repo: _Repo, *, branch: str = BRANCH) -> Scenario:
    """(a) A task branch merged to main by a no-ff merge carrying a citation."""
    base, tip = _seed_branch(repo, branch=branch)
    repo.git('merge', '--no-ff', branch, '-m', f'Merge {branch} into main')
    merge_sha = repo.sha('main')
    sc = Scenario(
        repo=repo, branch=branch, branch_base_sha=base, landed_sha=merge_sha,
        metadata={'branch_base_sha': base},
    )
    sc.refresh()
    assert sc.branch_tip_sha == tip
    return sc


def build_rebased_landing(repo: _Repo, *, branch: str = BRANCH) -> Scenario:
    """(b) B3 — the same work REBASED onto main: rewritten shas, no merge, no citation.

    Main is advanced first so the replay genuinely rewrites the shas, and the
    replayed commits are re-worded to strip every task citation, so a
    citation-based detector has nothing to find and only patch-id equivalence
    can attribute the landing.
    """
    base, tip = _seed_branch(repo, branch=branch)
    repo.commit('chore: unrelated main advance', {'CHANGELOG.md': _numbered('log', 5)})

    # Replay the branch's content onto main with fresh shas and no citation.
    # ONE REPLAYED COMMIT PER SOURCE COMMIT, deliberately: ``git cherry``
    # compares PER-COMMIT patch-ids, so squashing the branch's two commits
    # into one replay commit produces a patch-id that matches NEITHER source
    # and the fixture would build an "unlanded" shape while claiming to build
    # a rebase landing.  Re-wording each replay does not perturb its patch-id
    # (patch-id is computed over the diff alone), which is what lets this
    # scenario be simultaneously patch-id-contained and citation-free.
    commits = repo.git('rev-list', '--reverse', f'{base}..{tip}').splitlines()
    repo.git('checkout', '-b', '_replay', 'main')
    for i, sha in enumerate(commits):
        repo.git('cherry-pick', sha)
        repo.git('commit', '--amend', '-m', f'chore: unattributed replay {i}')
    repo.git('checkout', 'main')
    repo.git('merge', '--ff-only', '_replay')
    repo.git('branch', '-D', '_replay')

    landed = repo.sha('main')
    sc = Scenario(
        repo=repo, branch=branch, branch_base_sha=base, landed_sha=landed,
        metadata={'branch_base_sha': base},
    )
    sc.refresh()
    assert sc.branch_tip_sha == tip
    return sc


def build_sync_merge_branch(repo: _Repo, *, branch: str = BRANCH) -> Scenario:
    """(c) B2 — a branch tip that is a conflict-resolution merge with parents[1] == main.

    The branch and main both edit ``WORK_PATH``, so merging main INTO the
    branch conflicts and must be resolved; the resulting tip is a real merge
    commit whose second parent is main's tip.  The branch's own work is then
    landed on main by a no-ff merge.
    """
    base, _tip = _seed_branch(repo, branch=branch)
    main_edit = repo.commit(
        'chore: main edits the same file',
        {WORK_PATH: _numbered('feature', 40) + _numbered('mainside', 6)},
    )
    repo.git('checkout', branch)
    rc, _ = repo.git_rc('merge', '--no-ff', 'main', '-m', f'Merge main into {branch}')
    assert rc != 0, 'the sync merge was expected to conflict'
    repo.write(WORK_PATH, _numbered('feature', 40) + _numbered('mainside', 6))
    repo.git('add', '-A')
    repo.git('commit', '--no-edit')
    tip = repo.sha('HEAD')
    assert repo.parents(tip)[1] == main_edit, 'parents[1] must be main'
    repo.git('checkout', 'main')
    repo.git('merge', '--no-ff', branch, '-m', f'Merge {branch} into main')

    sc = Scenario(
        repo=repo, branch=branch, branch_base_sha=base,
        landed_sha=repo.sha('main'), metadata={'branch_base_sha': base},
    )
    sc.refresh()
    assert sc.branch_tip_sha == tip
    return sc


def build_no_op_landing(repo: _Repo, *, branch: str = BRANCH) -> Scenario:
    """(d) B4 — a merge commit on main whose NET branch diff is empty (task-1175 shape).

    The branch adds the feature and then removes it again, so every commit is
    real work but ``merge-base(main, tip)..tip`` is empty: the merge marker on
    main is genuine while the task delivered nothing.
    """
    base = repo.sha('main')
    repo.git('checkout', '-b', branch)
    repo.commit(
        f'feat({TASK_ID}): add the feature module',
        {WORK_PATH: _numbered('feature', 40)},
    )
    repo.git('rm', '-q', WORK_PATH)
    repo.git('commit', '-m', f'fix({TASK_ID}): back the feature out again')
    tip = repo.sha('HEAD')
    repo.git('checkout', 'main')
    repo.git('merge', '--no-ff', branch, '-m', f'Merge {branch} into main')

    sc = Scenario(
        repo=repo, branch=branch, branch_base_sha=base,
        landed_sha=repo.sha('main'), metadata={'branch_base_sha': base},
    )
    sc.refresh()
    assert sc.branch_tip_sha == tip
    return sc


def build_degenerate_branch(repo: _Repo, *, branch: str = BRANCH) -> Scenario:
    """(e) B6 — a branch whose tip equals its recorded ``branch_base_sha``.

    Zero commits were ever pushed beyond the creation point (#1226).  Main is
    then advanced so the branch is parked at an OLD main commit, which is the
    shape that reads as a confident landing to any containment-only check.
    """
    base = repo.sha('main')
    repo.git('branch', branch, base)
    repo.commit('chore: main advances past the parked branch', {WORK_PATH: _numbered('foreign', 20)})

    sc = Scenario(
        repo=repo, branch=branch, branch_base_sha=base, landed_sha=None,
        metadata={'branch_base_sha': base},
    )
    sc.refresh()
    assert sc.branch_tip_sha == base
    return sc


def build_unlanded_branch(repo: _Repo, *, branch: str = BRANCH) -> Scenario:
    """B5 — a branch with real commits that are genuinely absent from main."""
    base, tip = _seed_branch(repo, branch=branch)
    repo.commit('chore: main moves on without the branch', {'CHANGELOG.md': _numbered('log', 5)})

    sc = Scenario(
        repo=repo, branch=branch, branch_base_sha=base, landed_sha=None,
        metadata={'branch_base_sha': base},
    )
    sc.refresh()
    assert sc.branch_tip_sha == tip
    return sc


def advance_main_touching_same_paths(
    sc: Scenario, count: int, *, rewrite: bool = False,
) -> list[str]:
    """(f) B1 — *count* unrelated commits on main that TOUCH THE SAME PATHS.

    With ``rewrite=True`` the final commit REPLACES most of the lines the task
    added, which is the shape that trips the 0.98 aggregate / 0.90 per-file
    line-survival thresholds a decaying effect-present predicate applies.  The
    task's work is not removed — it is churned — so a non-decaying producer
    must still report it landed.
    """
    repo = sc.repo
    repo.git('checkout', 'main')
    shas: list[str] = []
    for i in range(count):
        last = rewrite and i == count - 1
        body = (
            _numbered('rewritten', 40, start=1000) if last
            else _numbered('feature', 40) + _numbered('churn', 4, start=100 * (i + 1))
        )
        shas.append(repo.commit(
            f'chore: unrelated main churn {i}',
            {WORK_PATH: body, OTHER_PATH: _numbered('helper', 30) + _numbered('extra', 2, start=i)},
        ))
    sc.refresh()
    return shas


def revert_landing(sc: Scenario) -> str:
    """Apply a POST-HOC REVERT of the landing on main.

    Measured against real git (both the merged and the rebased landing shape):
    a ``git revert`` does **not** make the task's patch-ids absent from main.
    ``git cherry`` answers "is an equivalent patch anywhere in upstream's
    history", and a revert ADDS an inverse commit rather than removing the
    original — so containment still holds afterwards.  See
    :func:`rewind_landing` for the construction that genuinely removes it.

    This builder therefore exists to PIN that accepted residual, not to flip a
    verdict.  It is the tradeoff D1 already prices in (2 genuine post-hoc
    reverts across both repos in 5.4 months) and is the direct cost of the
    non-decay property B1 demands: a predicate that could see this revert is
    exactly a predicate that decays under ordinary same-path churn.
    """
    repo = sc.repo
    repo.git('checkout', 'main')
    assert sc.landed_sha is not None, 'nothing to revert — this scenario never landed'
    if len(repo.parents(sc.landed_sha)) >= 2:
        repo.git('revert', '--no-edit', '-m', '1', sc.landed_sha)
    else:
        repo.git('revert', '--no-edit', sc.landed_sha)
    sc.refresh()
    return repo.sha('main')


def rewind_landing(sc: Scenario) -> str:
    """B1's paired NEGATIVE — genuinely REMOVE the task's work from main.

    Rewinds main to the commit preceding the landing and advances it with
    unrelated work, so the task's commits are no longer in main's history at
    all and their patch-ids are genuinely absent.  This is the only removal
    shape patch-id containment can observe (see :func:`revert_landing`), and
    it is what keeps B1's non-decay pin from being vacuous: the invariant is
    "must not decay unless the work is genuinely removed", not "always True".
    """
    repo = sc.repo
    repo.git('checkout', 'main')
    assert sc.landed_sha is not None, 'nothing to rewind — this scenario never landed'
    repo.git('reset', '--hard', f'{sc.landed_sha}^1')
    repo.commit('chore: main moves on without the reverted work',
                {'CHANGELOG.md': _numbered('log', 5)})
    sc.landed_sha = None
    sc.refresh()
    return repo.sha('main')


# --------------------------------------------------------------------------
# Fixture smoke tests — each builder produces the git shape it promises.
# --------------------------------------------------------------------------


class TestBoundaryFixtures:
    """The scenario builders construct the real git shapes B1--B6 need.

    These assert on GIT STATE only (no production symbol from this task), so
    they stay green independently of the producer and a later failure in the
    boundary rows can never be blamed on a fixture that silently stopped
    building the shape it claims.
    """

    def test_merged_landing_has_a_citing_merge_commit(self, repo: _Repo) -> None:
        sc = build_merged_landing(repo)
        assert len(repo.parents('main')) == 2, 'no-ff merge expected'
        assert f'Merge {BRANCH} into main' in repo.subjects('main')
        assert repo.git_rc('merge-base', '--is-ancestor', sc.branch_tip_sha, 'main')[0] == 0

    def test_rebased_landing_rewrites_shas_and_drops_the_citation(self, repo: _Repo) -> None:
        sc = build_rebased_landing(repo)
        # The branch tip itself is NOT on main — only its content is.
        assert repo.git_rc('merge-base', '--is-ancestor', sc.branch_tip_sha, 'main')[0] != 0
        assert len(repo.parents('main')) == 1, 'a rebase landing has no merge commit'
        assert not any(TASK_ID in s for s in repo.subjects('main')), 'no citation expected'
        # ...but `git cherry` sees every branch commit as already applied.
        cherry = repo.git('cherry', 'main', sc.branch_tip_sha)
        assert not any(line.startswith('+') for line in cherry.splitlines()), cherry

    def test_sync_merge_branch_tip_has_main_as_second_parent(self, repo: _Repo) -> None:
        sc = build_sync_merge_branch(repo)
        parents = repo.parents(sc.branch_tip_sha)
        assert len(parents) == 2
        assert repo.git_rc('merge-base', '--is-ancestor', parents[1], 'main')[0] == 0

    def test_no_op_landing_merge_has_an_empty_net_branch_diff(self, repo: _Repo) -> None:
        sc = build_no_op_landing(repo)
        base = repo.git('merge-base', 'main', sc.branch_tip_sha)
        assert repo.git_rc('diff', '--quiet', base, sc.branch_tip_sha)[0] == 0
        assert len(repo.parents('main')) == 2, 'the merge marker is still genuine'

    def test_degenerate_branch_tip_equals_its_recorded_base(self, repo: _Repo) -> None:
        sc = build_degenerate_branch(repo)
        assert sc.branch_tip_sha == sc.branch_base_sha
        assert sc.branch_tip_sha != sc.main_sha, 'main must have advanced past it'
        # Degenerate branches are patch-id-contained by construction — the
        # exact trap the ordering rule exists to stop.
        assert repo.git('cherry', 'main', sc.branch_tip_sha) == ''

    def test_unlanded_branch_commits_are_absent_from_main(self, repo: _Repo) -> None:
        sc = build_unlanded_branch(repo)
        cherry = repo.git('cherry', 'main', sc.branch_tip_sha)
        assert [line for line in cherry.splitlines() if line.startswith('+')], cherry

    def test_same_path_churn_keeps_the_landing_but_decays_line_survival(
        self, repo: _Repo,
    ) -> None:
        sc = build_merged_landing(repo)
        shas = advance_main_touching_same_paths(sc, 5, rewrite=True)
        assert len(shas) == 5
        # Still a landing by patch-id...
        assert repo.git('cherry', 'main', sc.branch_tip_sha) == ''
        # ...but most of the lines the task added are gone from main HEAD,
        # which is exactly what makes a survival-threshold predicate decay.
        head_body = (repo.root / WORK_PATH).read_text()
        added = _numbered('feature', 40).splitlines()
        survivors = sum(1 for line in added if line in head_body.splitlines())
        assert survivors / len(added) < 0.90

    @pytest.mark.parametrize('build', [build_merged_landing, build_rebased_landing])
    def test_post_hoc_revert_does_not_remove_the_patch_ids(self, repo: _Repo, build) -> None:
        """The accepted residual, measured rather than assumed.

        A ``git revert`` ADDS an inverse commit; it does not take the original
        out of main's history, so ``git cherry`` still reports the work as
        contained afterwards in BOTH landing shapes.  Pinned here so a future
        reader does not mistake the non-decay property for an oversight.
        """
        sc = build(repo)
        revert_landing(sc)
        cherry = repo.git('cherry', 'main', sc.branch_tip_sha)
        assert not [line for line in cherry.splitlines() if line.startswith('+')], cherry

    def test_rewind_removes_the_work_by_patch_id(self, repo: _Repo) -> None:
        sc = build_merged_landing(repo)
        rewind_landing(sc)
        cherry = repo.git('cherry', 'main', sc.branch_tip_sha)
        assert [line for line in cherry.splitlines() if line.startswith('+')], cherry

    def test_fixtures_never_touch_the_enclosing_checkout(
        self, git_ops: GitOps, git_repo: Path,
    ) -> None:
        """The isolation guards are wired, not merely imported."""
        assert git_ops.project_root == git_repo
        assert (git_repo / '.git').exists(), 'the fixture root must BE a repo root'
        # A nested, non-root directory is refused BEFORE any subprocess runs.
        nested = git_repo / 'pkg'
        nested.mkdir(exist_ok=True)
        with pytest.raises(NonIsolatedGitRepoError):
            _Repo(nested)
        # And the ceiling is really set on the guarded wrapper's environment.
        assert 'GIT_CEILING_DIRECTORIES' in _Repo(git_repo)._env
