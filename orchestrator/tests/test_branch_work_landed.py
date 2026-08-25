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

import inspect
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
from orchestrator.landing_evidence import (
    LandingMethod,
    LandingReason,
    LandingVerdict,
    branch_work_landed,
    format_unattributed_landing_detail,
)

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
    sc: Scenario, count: int, *, rewrite: bool = False, start: int = 0,
) -> list[str]:
    """(f) B1 — *count* unrelated commits on main that TOUCH THE SAME PATHS.

    With ``rewrite=True`` the final commit REPLACES most of the lines the task
    added, which is the shape that trips the 0.98 aggregate / 0.90 per-file
    line-survival thresholds a decaying effect-present predicate applies.  The
    task's work is not removed — it is churned — so a non-decaying producer
    must still report it landed.

    *start* offsets the churn counter so a caller advancing main ONE commit at
    a time (to re-run the producer after each) still writes distinct content
    every time.  Without it a repeated ``count=1`` call rewrites byte-identical
    content and ``git commit`` fails with "nothing to commit" — the fixture
    would silently stop advancing main and the non-decay pin would measure
    nothing.
    """
    repo = sc.repo
    repo.git('checkout', 'main')
    shas: list[str] = []
    for n in range(count):
        i = start + n
        last = rewrite and n == count - 1
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

    def test_git_cherry_skips_merge_commits(self, repo: _Repo) -> None:
        """Step-10(a), MEASURED against real git rather than assumed.

        The containment helper shells ``git cherry``, and the whole
        sync-merge-tip question turns on what it does with the merge commit at
        the tip.  If ``git cherry`` reported the merge as an unlanded commit,
        every branch that ever pulled main in would read as not landed, and
        the producer would need to restrict the question to the branch's own
        non-merge commits by hand.  It does not — so no such filtering exists
        in the producer, and this test is what would notice if that ever
        changed under us.
        """
        base, _tip = _seed_branch(repo)
        main_edit = repo.commit(
            'chore: main edits the same file',
            {WORK_PATH: _numbered('feature', 40) + _numbered('mainside', 6)},
        )
        repo.git('checkout', BRANCH)
        rc, _ = repo.git_rc('merge', '--no-ff', 'main', '-m', f'Merge main into {BRANCH}')
        assert rc != 0
        repo.write(WORK_PATH, _numbered('feature', 40) + _numbered('mainside', 6))
        repo.git('add', '-A')
        repo.git('commit', '--no-edit')
        tip = repo.sha('HEAD')
        assert repo.parents(tip)[1] == main_edit
        assert base

        listed = {line.split()[-1] for line in repo.git('cherry', 'main', tip).splitlines()}
        walked = set(repo.git('rev-list', f'main..{tip}').splitlines())
        assert tip in walked, 'the merge tip IS in the range git cherry walks'
        assert tip not in listed, 'git cherry must skip the merge commit itself'
        assert listed, 'the branch\'s own non-merge commits must still be listed'
        assert listed < walked

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


# --------------------------------------------------------------------------
# The producer's public shape, and the two headline boundary rows (step-05).
# --------------------------------------------------------------------------


class TestBranchWorkLandedSignature:
    """``branch_work_landed``'s public shape, pinned mechanically.

    ``metadata`` is a documented WIDENING of the PRD Contract's sketched
    signature: boundary row B6 needs ``branch_base_sha``, which the sketched
    four-argument form cannot supply.  ``escalation_queue`` is the seam the G7
    storm-escape L1 is filed through.  Both are keyword-only and defaulted, so
    a caller that knows neither still gets exactly the Contract's behaviour.
    """

    def test_signature_is_the_contract_shape(self) -> None:
        params = inspect.signature(branch_work_landed).parameters
        assert list(params) == [
            'git_ops', 'task_id', 'branch',
            'branch_tip_sha', 'metadata', 'escalation_queue',
        ]
        for name in ('git_ops', 'task_id', 'branch'):
            assert params[name].kind is inspect.Parameter.POSITIONAL_OR_KEYWORD, name
        for name in ('branch_tip_sha', 'metadata', 'escalation_queue'):
            assert params[name].kind is inspect.Parameter.KEYWORD_ONLY, name
        # branch_tip_sha is required-by-keyword: a caller must SAY whether it
        # already observed a tip, so the producer never silently re-reads a ref
        # that other checks were already anchored on (the task-3103 hazard
        # branch_is_degenerate documents).
        assert params['branch_tip_sha'].default is inspect.Parameter.empty
        assert params['metadata'].default is None
        assert params['escalation_queue'].default is None

    def test_it_is_a_coroutine_function(self) -> None:
        assert inspect.iscoroutinefunction(branch_work_landed)


@pytest.mark.asyncio
class TestB3RebaseLanding:
    """B3 — work replayed onto main with rewritten shas and NO citation.

    THE headline row.  This is the shape the existing citation-based and
    effect-present detectors cannot see at all, and the whole reason the PRD
    asks for a patch-id producer: ``find_task_citation_commit`` has nothing to
    find, yet every branch commit is present in main as an equivalent patch.
    """

    async def test_rebase_landing_is_accepted_by_patch_id(
        self, git_ops: GitOps, repo: _Repo,
    ) -> None:
        sc = build_rebased_landing(repo)
        verdict = await branch_work_landed(
            git_ops, TASK_ID, sc.branch, branch_tip_sha=sc.branch_tip_sha,
        )
        assert isinstance(verdict, LandingVerdict)
        assert verdict.accepted is True
        assert verdict.reason is LandingReason.landed
        assert verdict.method is LandingMethod.patch_id

    async def test_evidence_sha_is_main_reachable_and_never_a_fallback(
        self, git_ops: GitOps, repo: _Repo,
    ) -> None:
        """Contract invariant 3, in its three separately-checkable halves.

        Main is advanced by one unrelated commit AFTER the replay so that "the
        producer resolved the right evidence sha" and "the producer lazily fell
        back to main's current tip" are distinguishable at all — with the
        replay sitting at main's tip the two are the same sha and the
        invariant would be untestable.
        """
        sc = build_rebased_landing(repo)
        repo.git('checkout', 'main')
        repo.commit('chore: an unrelated follow-up', {'CHANGELOG.md': _numbered('log', 7)})
        sc.refresh()

        verdict = await branch_work_landed(
            git_ops, TASK_ID, sc.branch, branch_tip_sha=sc.branch_tip_sha,
        )
        evidence = verdict.evidence_sha
        assert evidence is not None, 'an accepted verdict must anchor provenance'
        # (1) reachable from main
        assert repo.git_rc('merge-base', '--is-ancestor', evidence, 'main')[0] == 0, (
            f'evidence_sha {evidence} is not reachable from main'
        )
        # (2) never the branch tip, which is NOT on main in this shape
        assert evidence != sc.branch_tip_sha
        # (3) never main's current tip as a lazy fallback
        assert evidence != sc.main_sha
        # ...and it is one of the commits that actually replayed this work.
        assert repo.git('log', '--format=%s', '-1', evidence).startswith(
            'chore: unattributed replay',
        )

    async def test_probe_carries_the_structured_facts(
        self, git_ops: GitOps, repo: _Repo,
    ) -> None:
        """PRD correction 6bis / the manifest FAIL binding.

        A four-field verdict would silently empty the L1 body rendered by
        ``format_unattributed_landing_detail`` and its two sub-renderers, so
        the probe is asserted on the ACCEPT path too: what a call recorded is
        a property of the CALL, not of the outcome.
        """
        sc = build_rebased_landing(repo)
        verdict = await branch_work_landed(
            git_ops, TASK_ID, sc.branch, branch_tip_sha=sc.branch_tip_sha,
        )
        assert verdict.probe, 'the probe must never be empty'
        assert verdict.probe['task_id'] == TASK_ID
        assert verdict.probe['branch'] == sc.branch
        assert verdict.probe['branch_tip_sha'] == sc.branch_tip_sha
        assert verdict.probe['method'] == LandingMethod.patch_id
        assert verdict.probe['reason'] == LandingReason.landed

    async def test_tip_is_resolved_from_the_ref_when_not_supplied(
        self, git_ops: GitOps, repo: _Repo,
    ) -> None:
        sc = build_rebased_landing(repo)
        verdict = await branch_work_landed(
            git_ops, TASK_ID, sc.branch, branch_tip_sha=None,
        )
        assert verdict.accepted is True
        assert verdict.probe['branch_tip_sha'] == sc.branch_tip_sha


@pytest.mark.asyncio
class TestB5GenuinelyUnlanded:
    """B5 — the ordinary negative, which must stay ordinary.

    A producer that is generous enough to see B3 must still be strict enough
    to call an unmerged branch unlanded; otherwise every pending task reads as
    delivered.
    """

    async def test_unlanded_branch_is_not_landed(
        self, git_ops: GitOps, repo: _Repo,
    ) -> None:
        sc = build_unlanded_branch(repo)
        verdict = await branch_work_landed(
            git_ops, TASK_ID, sc.branch, branch_tip_sha=sc.branch_tip_sha,
        )
        assert verdict.accepted is False
        assert verdict.reason is LandingReason.not_landed
        assert verdict.evidence_sha is None, (
            'a rejected verdict must carry no provenance anchor'
        )

    async def test_probe_carries_the_structured_facts(
        self, git_ops: GitOps, repo: _Repo,
    ) -> None:
        sc = build_unlanded_branch(repo)
        verdict = await branch_work_landed(
            git_ops, TASK_ID, sc.branch, branch_tip_sha=sc.branch_tip_sha,
        )
        assert verdict.probe
        for key in ('task_id', 'branch', 'branch_tip_sha', 'method', 'reason'):
            assert key in verdict.probe, key
        assert verdict.probe['reason'] == LandingReason.not_landed

    async def test_rejection_renders_without_an_unrecognized_reason(
        self, git_ops: GitOps, repo: _Repo,
    ) -> None:
        """The G7 contracts-are-machine-checked half, at the RENDER boundary.

        A reason code the shared formatter cannot explain renders the literal
        ``Unrecognized reason code: ...`` into an L1 body a human reads, which
        is how a new vocabulary silently degrades the escalation it was added
        to improve.
        """
        sc = build_unlanded_branch(repo)
        verdict = await branch_work_landed(
            git_ops, TASK_ID, sc.branch, branch_tip_sha=sc.branch_tip_sha,
        )
        summary, detail = format_unattributed_landing_detail(
            TASK_ID, sc.branch, verdict,
        )
        assert 'Unrecognized reason code' not in detail
        assert 'Unrecognized reason code' not in summary
        assert 'genuinely absent from main' in detail
        assert 'not_landed' in summary


# --------------------------------------------------------------------------
# The rejection rows and THE ORDERING RULE (step-07).
# --------------------------------------------------------------------------


def _spy_on_method(monkeypatch: pytest.MonkeyPatch, obj: object, name: str) -> list[tuple]:
    """Wrap ``obj.name`` so calls are recorded, and still run the real thing.

    A WRAPPER, never a stub: the ordering assertions must not change what the
    producer actually decides, or "the guard ran first" would be pinned
    against a fixture the guard never really saw.
    """
    calls: list[tuple] = []
    original = getattr(obj, name)

    async def _recording(*args: object, **kwargs: object) -> object:
        calls.append((args, kwargs))
        return await original(*args, **kwargs)

    monkeypatch.setattr(obj, name, _recording)
    return calls


def _spy_on_patch_content_contained(monkeypatch: pytest.MonkeyPatch) -> list[tuple]:
    """Spy on the lazily-imported ``git cherry`` helper.

    ``branch_work_landed`` imports it INSIDE the function (merge_queue.py
    imports back from landing_evidence at module level, so a top-level reverse
    import would be a cycle), which is exactly what makes patching the module
    attribute effective: the lookup happens at call time.
    """
    import orchestrator.merge_queue as mq

    calls: list[tuple] = []
    original = mq.patch_content_contained

    async def _recording(*args: object, **kwargs: object) -> object:
        calls.append((args, kwargs))
        return await original(*args, **kwargs)

    monkeypatch.setattr(mq, 'patch_content_contained', _recording)
    return calls


@pytest.mark.asyncio
class TestB4NoOpLanding:
    """B4 — a genuine merge marker over a branch that delivered nothing.

    The task-1175 shape: every commit on the branch is real work, but the
    branch's NET contribution relative to its fork point is empty (added then
    removed).  Stamping it done records a task as delivered when nothing
    shipped.
    """

    async def test_no_op_landing_is_rejected(
        self, git_ops: GitOps, repo: _Repo,
    ) -> None:
        sc = build_no_op_landing(repo)
        verdict = await branch_work_landed(
            git_ops, TASK_ID, sc.branch,
            branch_tip_sha=sc.branch_tip_sha, metadata=sc.metadata,
        )
        assert verdict.accepted is False
        assert verdict.reason is LandingReason.no_op_landing
        assert verdict.evidence_sha is None

    async def test_the_merge_marker_and_the_citation_are_both_genuine(
        self, git_ops: GitOps, repo: _Repo,
    ) -> None:
        """What makes B4 hard: nothing about the LANDING is fake.

        A merge commit exists on main and its subject cites the task, so every
        marker-based and citation-based check reads this as a confident
        landing.  Only the net-contribution question separates it from a real
        one.
        """
        sc = build_no_op_landing(repo)
        assert f'Merge {sc.branch} into main' in repo.subjects('main')
        assert await git_ops.find_task_citation_commit(TASK_ID) is not None
        verdict = await branch_work_landed(
            git_ops, TASK_ID, sc.branch,
            branch_tip_sha=sc.branch_tip_sha, metadata=sc.metadata,
        )
        assert verdict.reason is LandingReason.no_op_landing

    async def test_rejection_renders_without_an_unrecognized_reason(
        self, git_ops: GitOps, repo: _Repo,
    ) -> None:
        sc = build_no_op_landing(repo)
        verdict = await branch_work_landed(
            git_ops, TASK_ID, sc.branch,
            branch_tip_sha=sc.branch_tip_sha, metadata=sc.metadata,
        )
        summary, detail = format_unattributed_landing_detail(TASK_ID, sc.branch, verdict)
        assert 'Unrecognized reason code' not in detail
        assert 'Unrecognized reason code' not in summary
        assert 'NO NET CHANGE' in detail


@pytest.mark.asyncio
class TestOrderingRule:
    """THE NORMATIVE ORDERING RULE — Contract invariant 2.

    ``degenerate_branch`` -> ``no_op_landing`` -> patch-id -> landed/not_landed,
    and the order is normative rather than merely defensive.  BOTH guards
    describe states in which the patch-id arm would confidently accept:

    * a no-op merge has every branch commit patch-id-present in main (they
      really were merged) AND an empty net diff;
    * a degenerate branch is patch-id-contained in main BY CONSTRUCTION — it
      is parked at an old main commit and contributes nothing of its own.

    Run in the wrong order the producer attributes a foreign commit's content
    to the task with full confidence, which is worse than any false negative.
    """

    async def test_no_op_outranks_patch_id_acceptance(
        self, git_ops: GitOps, repo: _Repo,
    ) -> None:
        """The adversarial case, with BOTH preconditions asserted first."""
        sc = build_no_op_landing(repo)

        # Precondition 1: the patch-id arm WOULD accept — every branch commit
        # is present in main as an equivalent patch.
        from orchestrator.merge_queue import patch_content_contained
        assert await patch_content_contained(sc.branch_tip_sha, 'main', git_ops) is True
        # Precondition 2: ...and the branch's net contribution is empty.
        assert await git_ops.net_diff_is_empty('main', sc.branch_tip_sha) is True

        verdict = await branch_work_landed(
            git_ops, TASK_ID, sc.branch,
            branch_tip_sha=sc.branch_tip_sha, metadata=sc.metadata,
        )
        assert verdict.reason is LandingReason.no_op_landing, (
            'the no-op guard must run BEFORE patch-id acceptance'
        )
        assert verdict.accepted is False

    async def test_no_op_guard_short_circuits_before_the_patch_id_arm(
        self, git_ops: GitOps, repo: _Repo, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Ordering asserted MECHANICALLY, not just by outcome."""
        sc = build_no_op_landing(repo)
        no_op_calls = _spy_on_method(monkeypatch, git_ops, 'net_diff_is_empty')
        cherry_calls = _spy_on_patch_content_contained(monkeypatch)

        verdict = await branch_work_landed(
            git_ops, TASK_ID, sc.branch,
            branch_tip_sha=sc.branch_tip_sha, metadata=sc.metadata,
        )
        assert verdict.reason is LandingReason.no_op_landing
        assert len(no_op_calls) == 1, 'the no-op guard must run'
        assert cherry_calls == [], (
            'a rejected no-op must never reach the patch-id arm'
        )

    async def test_degenerate_guard_runs_before_the_no_op_check_and_patch_id(
        self, git_ops: GitOps, repo: _Repo, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """The degenerate arm is FIRST, asserted by what never runs.

        A branch parked at an old main commit is patch-id-contained by
        construction, so both later arms would answer about main's history
        rather than about the task.
        """
        sc = build_degenerate_branch(repo)
        no_op_calls = _spy_on_method(monkeypatch, git_ops, 'net_diff_is_empty')
        cherry_calls = _spy_on_patch_content_contained(monkeypatch)

        verdict = await branch_work_landed(
            git_ops, TASK_ID, sc.branch,
            branch_tip_sha=sc.branch_tip_sha, metadata=sc.metadata,
        )
        assert verdict.reason is LandingReason.degenerate_branch
        assert no_op_calls == [], 'the no-op check must not run for a degenerate branch'
        assert cherry_calls == [], 'the patch-id arm must not run for a degenerate branch'


@pytest.mark.asyncio
class TestB6DegenerateBranch:
    """B6 — a provisioning-only branch that never advanced past its base (#1226)."""

    async def test_degenerate_branch_is_rejected_as_degenerate(
        self, git_ops: GitOps, repo: _Repo,
    ) -> None:
        sc = build_degenerate_branch(repo)
        assert sc.branch_tip_sha == sc.branch_base_sha
        verdict = await branch_work_landed(
            git_ops, TASK_ID, sc.branch,
            branch_tip_sha=sc.branch_tip_sha, metadata=sc.metadata,
        )
        assert verdict.accepted is False
        assert verdict.reason is LandingReason.degenerate_branch
        assert verdict.evidence_sha is None

    async def test_it_would_otherwise_read_as_a_confident_landing(
        self, git_ops: GitOps, repo: _Repo,
    ) -> None:
        """Why the row exists: containment says YES for a branch with no work."""
        sc = build_degenerate_branch(repo)
        from orchestrator.merge_queue import patch_content_contained
        assert await patch_content_contained(sc.branch_tip_sha, 'main', git_ops) is True, (
            'a parked branch is patch-id-contained in main by construction'
        )

    async def test_rejection_renders_without_an_unrecognized_reason(
        self, git_ops: GitOps, repo: _Repo,
    ) -> None:
        sc = build_degenerate_branch(repo)
        verdict = await branch_work_landed(
            git_ops, TASK_ID, sc.branch,
            branch_tip_sha=sc.branch_tip_sha, metadata=sc.metadata,
        )
        summary, detail = format_unattributed_landing_detail(TASK_ID, sc.branch, verdict)
        assert 'Unrecognized reason code' not in detail
        assert 'Unrecognized reason code' not in summary
        assert 'branch_base_sha' in detail


@pytest.mark.asyncio
class TestDegenerateArmBackwardCompatibility:
    """Absent or malformed ``branch_base_sha`` must cost nothing.

    The same backward-compatible posture ``branch_is_degenerate`` already
    documents: pre-#1226 tasks and tasks whose metadata write failed
    transiently have no recorded base, and a producer that treated "no base"
    as "degenerate" would reject every one of them.
    """

    async def test_metadata_none_falls_through_to_the_normal_arms(
        self, git_ops: GitOps, repo: _Repo,
    ) -> None:
        sc = build_unlanded_branch(repo)
        verdict = await branch_work_landed(
            git_ops, TASK_ID, sc.branch,
            branch_tip_sha=sc.branch_tip_sha, metadata=None,
        )
        assert verdict.reason is LandingReason.not_landed

    async def test_malformed_branch_base_sha_falls_through(
        self, git_ops: GitOps, repo: _Repo,
    ) -> None:
        sc = build_unlanded_branch(repo)
        for bogus in ('', 'not-a-sha', 'ABCDEF' * 6 + 'abcd', 123, None):
            verdict = await branch_work_landed(
                git_ops, TASK_ID, sc.branch,
                branch_tip_sha=sc.branch_tip_sha,
                metadata={'branch_base_sha': bogus},
            )
            assert verdict.reason is LandingReason.not_landed, bogus

    async def test_a_degenerate_branch_without_metadata_is_not_called_degenerate(
        self, git_ops: GitOps, repo: _Repo,
    ) -> None:
        """The cost of the backward-compatible posture, pinned rather than hidden.

        Without the recorded base the producer CANNOT know the branch never
        advanced, so the degenerate arm is skipped and the arms after it in the
        normative order answer instead.

        The no-op guard does NOT catch it, and the reason is worth stating
        because an earlier revision of this suite asserted the opposite.  A
        parked tip is its own merge-base with main, so
        ``merge-base(main, tip)..tip`` is empty — but so is the merge base of
        main with EVERY branch that has already landed, which is why that
        formula cannot be the no-op guard's baseline (B1 pins that: it reported
        every merged landing as a no-op).  With the baseline ladder in place
        the guard finds no baseline for this shape at all
        (``probe['no_op_baseline'] == 'unavailable'``) and correctly declines
        to answer rather than answering by accident.

        So the verdict comes from attribution: the parked tip IS patch-id
        contained in main, and no main-reachable commit can be attributed to
        the task, giving ``no_attribution``.  The fall-through therefore costs
        LEGIBILITY — the operator is told "nothing on main could be attributed
        to this task" rather than the sharper "this branch never advanced past
        its base" — and never SAFETY: the verdict is still a refusal carrying
        no provenance anchor, never the confident accept that patch-id
        containment alone would have produced for a parked branch.  Supplying
        ``branch_base_sha`` restores the sharp answer, which is why every
        orchestrator-dispatched caller has one.
        """
        sc = build_degenerate_branch(repo)
        verdict = await branch_work_landed(
            git_ops, TASK_ID, sc.branch,
            branch_tip_sha=sc.branch_tip_sha, metadata=None,
        )
        assert verdict.reason is not LandingReason.degenerate_branch
        assert verdict.probe['no_op_baseline'] == 'unavailable'
        assert verdict.reason is LandingReason.no_attribution
        assert verdict.accepted is False
        assert verdict.evidence_sha is None


# --------------------------------------------------------------------------
# B1 non-decay and B2 sync-merge tip (step-09).
# --------------------------------------------------------------------------


@pytest.mark.asyncio
class TestB1NonDecay:
    """B1 — the PRD's HEADLINE invariant, and the one that must not be waived.

    The failure this whole PRD exists to fix is MONOTONIC: the legacy
    effect-present policy compares a landed commit's added lines against main
    HEAD, so every later commit touching those paths makes the next detection
    attempt strictly LESS likely to succeed.  A task that goes unnoticed for a
    day is therefore harder to recover on day two than on day one, and by the
    time anyone looks it is unrecoverable — the shape that stranded tasks 3103
    and 3916.

    A patch-id producer must be immune to that by construction: "is an
    equivalent patch anywhere in main's HISTORY" is a question no later commit
    can un-answer.  These tests hold main still for nothing — they churn
    exactly the paths the task touched, past the 0.98 aggregate / 0.90
    per-file survival thresholds a decaying predicate applies, and demand the
    same verdict every time.
    """

    async def test_landing_survives_five_same_path_churn_commits(
        self, git_ops: GitOps, repo: _Repo,
    ) -> None:
        """Re-run after EVERY commit, not only at the end.

        Asserting once after all five would not distinguish "never decayed"
        from "decayed and recovered", and monotonic decay is precisely a
        property you can only see by sampling the whole sequence.
        """
        sc = build_merged_landing(repo)
        first = await branch_work_landed(
            git_ops, TASK_ID, sc.branch,
            branch_tip_sha=sc.branch_tip_sha, metadata=sc.metadata,
        )
        assert first.accepted is True, first.probe
        assert first.reason is LandingReason.landed

        for i in range(5):
            # The LAST commit rewrites the lines the task added, which is the
            # shape that trips the survival thresholds.
            advance_main_touching_same_paths(sc, 1, rewrite=(i == 4), start=i)
            verdict = await branch_work_landed(
                git_ops, TASK_ID, sc.branch,
                branch_tip_sha=sc.branch_tip_sha, metadata=sc.metadata,
            )
            assert verdict.accepted is True, f'decayed at churn commit {i}: {verdict.probe}'
            assert verdict.reason is LandingReason.landed, i
            assert verdict.method is LandingMethod.patch_id, i
            assert verdict.evidence_sha is not None, i

    async def test_the_churn_really_would_decay_a_survival_predicate(
        self, git_ops: GitOps, repo: _Repo,
    ) -> None:
        """The pin is only meaningful if the fixture is genuinely hostile.

        Measured against the REAL thresholds rather than a restated constant,
        so a future change to either threshold re-tests this fixture instead
        of silently making the row vacuous.
        """
        from orchestrator.git_ops import (
            _EFFECT_SURVIVAL_AGGREGATE_THRESHOLD,
            _EFFECT_SURVIVAL_PER_FILE_THRESHOLD,
        )

        sc = build_merged_landing(repo)
        advance_main_touching_same_paths(sc, 5, rewrite=True)
        added = _numbered('feature', 40).splitlines()
        head_lines = set((repo.root / WORK_PATH).read_text().splitlines())
        survival = sum(1 for line in added if line in head_lines) / len(added)
        assert survival < _EFFECT_SURVIVAL_PER_FILE_THRESHOLD
        assert survival < _EFFECT_SURVIVAL_AGGREGATE_THRESHOLD

        verdict = await branch_work_landed(
            git_ops, TASK_ID, sc.branch,
            branch_tip_sha=sc.branch_tip_sha, metadata=sc.metadata,
        )
        assert verdict.accepted is True, verdict.probe

    async def test_the_rebase_landing_also_does_not_decay(
        self, git_ops: GitOps, repo: _Repo,
    ) -> None:
        """Non-decay is a property of the POLICY, not of one landing shape."""
        sc = build_rebased_landing(repo)
        advance_main_touching_same_paths(sc, 5, rewrite=True)
        verdict = await branch_work_landed(
            git_ops, TASK_ID, sc.branch,
            branch_tip_sha=sc.branch_tip_sha, metadata=sc.metadata,
        )
        assert verdict.accepted is True, verdict.probe
        assert verdict.reason is LandingReason.landed

    async def test_the_pin_is_not_vacuous_genuine_removal_flips_it(
        self, git_ops: GitOps, repo: _Repo,
    ) -> None:
        """B1's PAIRED NEGATIVE — the invariant is conditional, not "always True".

        "Must not decay" would be trivially satisfiable by a producer that
        returned ``landed`` unconditionally, so the row is only a real pin
        alongside a construction that genuinely takes the work out of main.

        The plan's sketch named ``git revert`` as that construction; MEASURED
        against real git (``TestBoundaryFixtures.
        test_post_hoc_revert_does_not_remove_the_patch_ids``) it is not one —
        a revert ADDS an inverse commit and leaves the original patch-ids in
        main's history, so containment still holds.  :func:`rewind_landing`
        is the shape that does remove them, and it is used here instead.  The
        revert residual is pinned separately below as a priced tradeoff.
        """
        sc = build_merged_landing(repo)
        assert (await branch_work_landed(
            git_ops, TASK_ID, sc.branch,
            branch_tip_sha=sc.branch_tip_sha, metadata=sc.metadata,
        )).accepted is True

        rewind_landing(sc)
        verdict = await branch_work_landed(
            git_ops, TASK_ID, sc.branch,
            branch_tip_sha=sc.branch_tip_sha, metadata=sc.metadata,
        )
        assert verdict.accepted is False, verdict.probe
        assert verdict.reason is LandingReason.not_landed
        assert verdict.evidence_sha is None

    async def test_a_post_hoc_revert_is_the_priced_residual_not_a_flip(
        self, git_ops: GitOps, repo: _Repo,
    ) -> None:
        """The direct, deliberate COST of non-decay, recorded rather than hidden.

        A predicate able to notice this revert is exactly a predicate that
        decays under ordinary same-path churn — the two are the same
        measurement pointed in opposite directions, so this residual cannot be
        removed without giving up B1.  D1 prices it (2 genuine post-hoc
        reverts across both repos in 5.4 months) and the PRD accepts it; the
        legacy effect-present producer remains available for callers that need
        the other tradeoff.
        """
        sc = build_merged_landing(repo)
        revert_landing(sc)
        verdict = await branch_work_landed(
            git_ops, TASK_ID, sc.branch,
            branch_tip_sha=sc.branch_tip_sha, metadata=sc.metadata,
        )
        assert verdict.accepted is True, verdict.probe
        assert verdict.reason is LandingReason.landed


@pytest.mark.asyncio
class TestB2SyncMergeTip:
    """B2 — a branch whose TIP is a conflict-resolution merge of main into it.

    Routine in this repo: a long-running branch pulls main in to resolve a
    conflict, so its tip is a merge commit whose ``parents[1]`` is main.  The
    tip therefore carries main's own history as well as the task's work, and a
    producer that asks "is everything reachable from the tip in main?" without
    care will either mis-read main's merged-in commits as unlanded task work
    or compute a diff over main's entire touched-path set.
    """

    async def test_sync_merge_tip_is_accepted(
        self, git_ops: GitOps, repo: _Repo,
    ) -> None:
        sc = build_sync_merge_branch(repo)
        assert len(repo.parents(sc.branch_tip_sha)) == 2, 'the tip must be a merge'
        verdict = await branch_work_landed(
            git_ops, TASK_ID, sc.branch,
            branch_tip_sha=sc.branch_tip_sha, metadata=sc.metadata,
        )
        assert verdict.accepted is True, verdict.probe
        assert verdict.reason is LandingReason.landed
        assert verdict.method is LandingMethod.patch_id

    async def test_the_sync_merge_tip_is_not_a_no_op(
        self, git_ops: GitOps, repo: _Repo,
    ) -> None:
        """Step-10(c): the no-op guard must let this shape through.

        Relative to the branch's FORK POINT the tip is a real net change — it
        carries both the task's work and main's merged-in commit — so the
        no-op arm answers False and the verdict is decided by patch-id, not by
        a guard that fired on the wrong question.
        """
        sc = build_sync_merge_branch(repo)
        verdict = await branch_work_landed(
            git_ops, TASK_ID, sc.branch,
            branch_tip_sha=sc.branch_tip_sha, metadata=sc.metadata,
        )
        assert verdict.reason is not LandingReason.no_op_landing, verdict.probe
        assert verdict.probe.get('patch_id_contained') is True, (
            'the patch-id arm must actually have run and answered'
        )

    async def test_an_unmerged_sync_merge_branch_is_still_not_landed(
        self, git_ops: GitOps, repo: _Repo,
    ) -> None:
        """The negative control for step-10(a).

        ``git cherry`` skips merge commits, so the containment question is
        answered over the branch's OWN non-merge commits.  That must not make
        a sync-merged-but-unlanded branch read as contained just because its
        merge commit is invisible to ``git cherry``: its real work commits are
        still absent from main.
        """
        base, _tip = _seed_branch(repo)
        main_edit = repo.commit(
            'chore: main edits the same file',
            {WORK_PATH: _numbered('feature', 40) + _numbered('mainside', 6)},
        )
        repo.git('checkout', BRANCH)
        rc, _ = repo.git_rc('merge', '--no-ff', 'main', '-m', f'Merge main into {BRANCH}')
        assert rc != 0
        repo.write(WORK_PATH, _numbered('feature', 40) + _numbered('mainside', 6))
        repo.git('add', '-A')
        repo.git('commit', '--no-edit')
        tip = repo.sha('HEAD')
        assert repo.parents(tip)[1] == main_edit
        repo.git('checkout', 'main')

        verdict = await branch_work_landed(
            git_ops, TASK_ID, BRANCH,
            branch_tip_sha=tip, metadata={'branch_base_sha': base},
        )
        assert verdict.accepted is False, verdict.probe
        assert verdict.reason is LandingReason.not_landed

    async def test_the_baseline_ladder_reports_which_rung_answered(
        self, git_ops: GitOps, repo: _Repo,
    ) -> None:
        """The no-op baseline is a structured fact, not an implicit choice.

        No single formula is correct for every landing shape, so which one was
        used decides what an emptiness claim MEANS.  Recording it lets a reader
        of the escalation see what the measurement was taken against instead of
        having to re-derive it from the shape of the repo.
        """
        landed = build_merged_landing(repo)
        verdict = await branch_work_landed(
            git_ops, TASK_ID, landed.branch,
            branch_tip_sha=landed.branch_tip_sha, metadata=landed.metadata,
        )
        assert verdict.probe['no_op_baseline'] == 'recorded_branch_base'

    async def test_a_merged_landing_without_metadata_uses_the_landing_merge(
        self, git_ops: GitOps, repo: _Repo,
    ) -> None:
        """Rung 3 — and it is load-bearing, not a nicety.

        Without the recorded base, rung 2's ``merge-base(main, tip)`` IS the
        tip for any branch that has merged, so the no-op question would answer
        "empty" for EVERY landed branch.  Rung 3 asks the PRD Contract's
        literal ``merge-base(first_parent, tip)..tip`` form about the landing
        merge instead, which stays well-defined after the fact — and it must
        keep distinguishing a real landing from a no-op one WITHOUT the
        metadata, or the task-1175 shape would be accepted whenever a caller
        has none.
        """
        sc = build_merged_landing(repo)
        real = await branch_work_landed(
            git_ops, TASK_ID, sc.branch, branch_tip_sha=sc.branch_tip_sha, metadata=None,
        )
        assert real.probe['no_op_baseline'] == 'landing_merge'
        assert real.accepted is True, real.probe
        assert real.reason is LandingReason.landed

    async def test_a_no_op_landing_without_metadata_is_still_rejected(
        self, git_ops: GitOps, repo: _Repo,
    ) -> None:
        """Rung 3's other half — the guard must not go quiet when metadata is absent.

        Skipping the no-op check whenever the recorded base is missing would
        silently accept the task-1175 shape, and a false ACCEPT closes a task
        that never delivered — strictly worse here than any false negative.
        """
        sc = build_no_op_landing(repo)
        verdict = await branch_work_landed(
            git_ops, TASK_ID, sc.branch, branch_tip_sha=sc.branch_tip_sha, metadata=None,
        )
        assert verdict.probe['no_op_baseline'] == 'landing_merge'
        assert verdict.accepted is False, verdict.probe
        assert verdict.reason is LandingReason.no_op_landing

    @pytest.mark.parametrize(
        'build',
        [
            build_merged_landing,
            build_rebased_landing,
            build_sync_merge_branch,
            build_no_op_landing,
            build_unlanded_branch,
            build_degenerate_branch,
        ],
    )
    async def test_neither_decaying_predicate_is_ever_awaited(
        self, git_ops: GitOps, repo: _Repo, monkeypatch: pytest.MonkeyPatch, build,
    ) -> None:
        """The checkable form of "the ~300-file main-history diff is never computed".

        Asserting a ZERO call count on the real ``GitOps`` is a structural
        pin, not a performance one: as long as neither decaying predicate is
        reachable, the producer's non-decay property cannot be re-introduced
        by accident through an evidence-resolution or preflight leg that
        happens to call one of them.  Both are wrapped rather than stubbed, so
        a call would still behave normally and the failure reported is the
        call itself.

        Parametrised over EVERY boundary row because the hazard is a code
        PATH, and a leg only some rows reach is exactly the one that would
        slip through a single-row assertion.
        """
        branch_content = _spy_on_method(monkeypatch, git_ops, 'branch_content_in_main')
        effect_present = _spy_on_method(monkeypatch, git_ops, 'commit_effect_present_in_main')
        describe_effect = _spy_on_method(monkeypatch, git_ops, 'describe_commit_effect_in_main')

        sc = build(repo)
        await branch_work_landed(
            git_ops, TASK_ID, sc.branch,
            branch_tip_sha=sc.branch_tip_sha, metadata=sc.metadata,
        )
        assert branch_content == [], 'branch_content_in_main (git_ops.py) must never be reached'
        assert effect_present == [], 'commit_effect_present_in_main must never be reached'
        assert describe_effect == [], 'describe_commit_effect_in_main must never be reached'
