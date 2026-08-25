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
from typing import TYPE_CHECKING, Any, cast

import pytest
from _orch_helpers import (
    NonIsolatedGitRepoError,
    assert_isolated_git_repo,
    git_env_with_ceiling,
)

from orchestrator import landing_evidence
from orchestrator.config import GitConfig, RecoveryEmissionConfig
from orchestrator.git_ops import GitOps
from orchestrator.landing_evidence import (
    LANDING_GIT_ERROR_STORM_CATEGORY,
    LANDING_GIT_ERROR_STORM_SENTINEL,
    LandingMethod,
    LandingReason,
    LandingTally,
    LandingVerdict,
    branch_work_landed,
    file_landing_git_error_storm_escalation,
    format_unattributed_landing_detail,
)

if TYPE_CHECKING:
    from escalation.queue import EscalationQueue

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
# The producer itself, and the two headline boundary rows (step-05).
# --------------------------------------------------------------------------


class TestBranchWorkLandedIsAwaitable:
    """``branch_work_landed`` is a coroutine function.

    It widens the PRD Contract's sketched signature in three documented ways —
    ``metadata`` (boundary row B6 needs ``branch_base_sha``),
    ``escalation_queue`` (the seam the G7 storm-escape L1 is filed through) and
    ``recovery_emission`` (the live config submodel supplying that alarm's rate
    and kill switch, which a module-level function cannot read off
    ``self.config``).  The rationale for each lives in ``branch_work_landed``'s
    own docstring, which is where a reader looks; it is deliberately NOT
    re-asserted as a test-suite contract.  Every load-bearing part of it is
    already exercised behaviourally: ``TestB3RebaseLanding``,
    ``TestB6DegenerateBranch``, ``TestGitErrorStormEscape`` et al. all call the
    producer by keyword, so a rename or a defaulting of ``branch_tip_sha``
    fails dozens of tests loudly.  An ``inspect.signature`` equality would add
    nothing but a pin on the ORDER of keyword-only parameters, which no caller
    can observe.
    """

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

    # `Any`, not `object`: unlike the `getattr` capture above, `original` here
    # is the concretely-typed `patch_content_contained`, so forwarding
    # `object`-typed varargs into it is a type error at the call.
    async def _recording(*args: Any, **kwargs: Any) -> Any:
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


# --------------------------------------------------------------------------
# Contract invariant 4 — fail-closed, and git_error is never not_landed
# (step-11).
# --------------------------------------------------------------------------


@pytest.fixture
def broken_main_git_ops(git_repo: Path) -> GitOps:
    """A ``GitOps`` whose configured main branch does not exist.

    The cheapest REAL unhealthy-repo construction available to a test: every
    ref the producer resolves against ``config.main_branch`` genuinely fails,
    with no patching of the code under test, so what is measured is the
    producer's behaviour on a broken repo rather than on a simulated one.
    """
    return GitOps(
        GitConfig(
            main_branch='no-such-main-branch',
            branch_prefix='task/',
            remote='origin',
            worktree_dir='.worktrees',
            push_after_advance=False,
        ),
        git_repo,
    )


@pytest.mark.asyncio
class TestGitErrorIsNeverNotLanded:
    """G7 storm-escape, and the single most consequential rule in this module.

    ``not_landed`` is a claim ABOUT THE TASK: the work is not on main, so
    dispatch it.  ``git_error`` is a claim about the DETECTOR: it could not
    look.  Collapsing the second into the first makes a repo lock, a corrupt
    object or an unresolvable ref read as "this task never landed", so a task
    that DID land is re-dispatched — forever, since the condition that broke
    the check is not fixed by re-running it.  That is the defect this PRD
    exists to fix, so producing it here would be strictly worse than shipping
    nothing.

    Every case therefore asserts the REASON, not merely ``accepted is False``:
    both codes reject, and it is only the code that tells a consumer whether
    to re-dispatch or to escalate.
    """

    async def test_an_unresolvable_branch_ref_is_a_git_error(
        self, git_ops: GitOps, repo: _Repo,
    ) -> None:
        build_unlanded_branch(repo)
        verdict = await branch_work_landed(
            git_ops, TASK_ID, 'task/does-not-exist', branch_tip_sha=None,
        )
        assert verdict.accepted is False
        assert verdict.reason is LandingReason.git_error
        assert verdict.probe['git_error_stage'] == 'resolve_branch_sha'

    async def test_an_unresolvable_tip_sha_is_a_git_error(
        self, git_ops: GitOps, repo: _Repo,
    ) -> None:
        """A caller-supplied tip that is not in this repo at all."""
        sc = build_unlanded_branch(repo)
        verdict = await branch_work_landed(
            git_ops, TASK_ID, sc.branch,
            branch_tip_sha='0' * 39 + '1', metadata=sc.metadata,
        )
        assert verdict.accepted is False
        assert verdict.reason is LandingReason.git_error

    @pytest.mark.parametrize('with_metadata', [True, False])
    async def test_an_unresolvable_main_ref_is_a_git_error(
        self, broken_main_git_ops: GitOps, repo: _Repo, with_metadata: bool,
    ) -> None:
        """Parametrised because the two paths reach the failure DIFFERENTLY.

        Without metadata the baseline ladder resolves against main and fails
        early.  WITH metadata the ladder answers from the recorded base
        without touching main at all, so the first thing to see the broken ref
        is ``git cherry`` — whose fail-OPEN ``False`` is exactly the
        misclassification this class exists to prevent.
        """
        sc = build_unlanded_branch(repo)
        verdict = await branch_work_landed(
            broken_main_git_ops, TASK_ID, sc.branch,
            branch_tip_sha=sc.branch_tip_sha,
            metadata=sc.metadata if with_metadata else None,
        )
        assert verdict.accepted is False
        assert verdict.reason is LandingReason.git_error, verdict.probe
        assert verdict.probe.get('git_error_stage'), 'the failing stage must be named'

    async def test_a_landed_branch_is_a_git_error_not_a_landing_when_main_is_broken(
        self, broken_main_git_ops: GitOps, repo: _Repo,
    ) -> None:
        """The direction that matters most: a REAL landing must not silently reject.

        This branch genuinely landed.  With main unresolvable the containment
        helper returns its fail-open ``False``, and reading that as
        ``not_landed`` would re-dispatch a task whose work is already on main
        — the exact stranding this PRD exists to fix, re-created by the
        recovery path itself.
        """
        sc = build_merged_landing(repo)
        verdict = await branch_work_landed(
            broken_main_git_ops, TASK_ID, sc.branch,
            branch_tip_sha=sc.branch_tip_sha, metadata=sc.metadata,
        )
        assert verdict.reason is LandingReason.git_error, verdict.probe
        assert verdict.reason is not LandingReason.not_landed

    async def test_the_fail_open_containment_answer_is_disambiguated(
        self, broken_main_git_ops: GitOps, repo: _Repo,
    ) -> None:
        """THE fail-open disambiguation, and the probe must SHOW which branch was taken.

        ``merge_queue.patch_content_contained`` is documented fail-OPEN and
        returns ``False`` on ``rc != 0`` — correct for ITS caller, which falls
        through to a full merge attempt, and exactly backwards here, where
        ``False`` means "not landed".  The producer therefore re-probes ref
        health on a ``False`` and records the outcome, so a reader can tell a
        genuine negative from an unreadable repo without re-running git.
        """
        sc = build_merged_landing(repo)
        verdict = await branch_work_landed(
            broken_main_git_ops, TASK_ID, sc.branch,
            branch_tip_sha=sc.branch_tip_sha, metadata=sc.metadata,
        )
        assert verdict.reason is LandingReason.git_error
        assert verdict.probe['containment_recheck'] == 'unhealthy'
        assert verdict.probe['containment_unresolvable'], (
            'the endpoints that failed to re-resolve must be recorded'
        )

    async def test_the_contrasting_positive_control_on_a_healthy_repo(
        self, git_ops: GitOps, repo: _Repo,
    ) -> None:
        """The disambiguation must not have relabelled every rejection.

        Same producer, same ``False`` from the containment helper — but a
        healthy repo, so the answer is trustworthy and the verdict stays the
        ordinary ``not_landed``.  Without this control, "always say git_error
        on False" would pass every other case in this class.
        """
        sc = build_unlanded_branch(repo)
        verdict = await branch_work_landed(
            git_ops, TASK_ID, sc.branch,
            branch_tip_sha=sc.branch_tip_sha, metadata=sc.metadata,
        )
        assert verdict.reason is LandingReason.not_landed, verdict.probe
        assert verdict.probe['containment_recheck'] == 'healthy'
        assert verdict.probe['containment_unresolvable'] == []
        assert 'git_error_stage' not in verdict.probe

    async def test_an_indeterminate_no_op_answer_is_a_git_error(
        self, git_ops: GitOps, repo: _Repo, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """The tri-state's third state must not be read as "not a no-op".

        ``net_diff_is_empty`` returns ``None`` for an unreadable commit or an
        uncomputable merge-base.  Treating that as ``False`` would let the run
        continue to the patch-id arm on a repo that just proved it cannot
        answer questions.
        """
        sc = build_merged_landing(repo)

        async def _indeterminate(*_args: object, **_kwargs: object) -> None:
            return None

        monkeypatch.setattr(git_ops, 'net_diff_is_empty', _indeterminate)
        verdict = await branch_work_landed(
            git_ops, TASK_ID, sc.branch,
            branch_tip_sha=sc.branch_tip_sha, metadata=sc.metadata,
        )
        assert verdict.accepted is False
        assert verdict.reason is LandingReason.git_error
        assert verdict.probe['git_error_stage'] == 'net_diff_is_empty'

    async def test_an_unresolvable_baseline_is_a_git_error(
        self, git_ops: GitOps, repo: _Repo, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        sc = build_unlanded_branch(repo)

        async def _no_merge_base(*_args: object, **_kwargs: object) -> None:
            return None

        monkeypatch.setattr(git_ops, 'merge_base_with_main', _no_merge_base)
        verdict = await branch_work_landed(
            git_ops, TASK_ID, sc.branch, branch_tip_sha=sc.branch_tip_sha, metadata=None,
        )
        assert verdict.reason is LandingReason.git_error
        assert verdict.probe['git_error_stage'] == 'no_op_baseline'


@pytest.mark.asyncio
class TestBranchWorkLandedNeverRaises:
    """It always RETURNS a verdict — it never accepts on doubt and never propagates.

    Its callers are recovery paths: a sweep over stranded tasks, a dispatch
    gate, a merge-status query.  An exception escaping here takes the whole
    sweep down with it, so one unreadable repo would stop every OTHER task
    from being recovered — turning a single broken task into a stalled queue.
    Containing it is not defensive coding; it is the difference between a
    degraded check and a dead one.
    """

    @pytest.mark.parametrize(
        'method',
        [
            'resolve_branch_sha',
            'is_ancestor',
            'merge_base_with_main',
            'net_diff_is_empty',
            'find_task_citation_commit',
            'find_equivalent_commit',
        ],
    )
    async def test_an_unexpected_exception_becomes_a_git_error_verdict(
        self, git_ops: GitOps, repo: _Repo, monkeypatch: pytest.MonkeyPatch, method: str,
    ) -> None:
        sc = build_rebased_landing(repo)

        async def _boom(*_args: object, **_kwargs: object) -> object:
            raise RuntimeError(f'simulated {method} failure')

        monkeypatch.setattr(git_ops, method, _boom)
        verdict = await branch_work_landed(
            git_ops, TASK_ID, sc.branch,
            branch_tip_sha=None if method == 'resolve_branch_sha' else sc.branch_tip_sha,
            metadata=sc.metadata,
        )
        assert isinstance(verdict, LandingVerdict)
        assert verdict.accepted is False, 'never accept on doubt'
        assert verdict.reason is LandingReason.git_error
        assert verdict.evidence_sha is None

    async def test_the_exception_is_recorded_and_logged_not_swallowed(
        self, git_ops: GitOps, repo: _Repo, monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """CONTAINED but not SWALLOWED — the module's existing discipline.

        A caught exception that leaves no trace is indistinguishable from a
        clean negative, so the repr goes in the probe (structured facts for
        the escalation body) AND a warning with a traceback goes to the log
        (for whoever is reading logs rather than escalations).
        """
        sc = build_rebased_landing(repo)

        async def _boom(*_args: object, **_kwargs: object) -> object:
            raise RuntimeError('simulated merge_base failure')

        monkeypatch.setattr(git_ops, 'merge_base_with_main', _boom)
        with caplog.at_level('WARNING', logger='orchestrator.landing_evidence'):
            verdict = await branch_work_landed(
                git_ops, TASK_ID, sc.branch,
                branch_tip_sha=sc.branch_tip_sha, metadata=None,
            )
        assert verdict.reason is LandingReason.git_error
        assert 'simulated merge_base failure' in str(verdict.probe.get('exception'))
        assert verdict.probe['git_error_stage'] == 'unexpected_exception'
        assert any(r.exc_info for r in caplog.records), 'a traceback must be logged'

    async def test_the_verdict_still_renders_without_an_unrecognized_reason(
        self, broken_main_git_ops: GitOps, repo: _Repo,
    ) -> None:
        sc = build_merged_landing(repo)
        verdict = await branch_work_landed(
            broken_main_git_ops, TASK_ID, sc.branch,
            branch_tip_sha=sc.branch_tip_sha, metadata=sc.metadata,
        )
        summary, detail = format_unattributed_landing_detail(TASK_ID, sc.branch, verdict)
        assert 'Unrecognized reason code' not in detail
        assert 'Unrecognized reason code' not in summary
        assert 'git_error' in summary


# ---------------------------------------------------------------------------
# The G7 STORM ESCAPE: a per-reason tally, and a rate-gated escape hatch for
# `git_error`.
#
# `git_error` is the one code whose REPETITION means the detector is broken
# rather than the task being unlanded, and a broken detector is silent by
# construction: every one of its verdicts rejects, and a rejecting detector
# looks exactly like a repo with nothing landed in it.  The tally makes the
# silence audible, and the rate gate escalates it once — not once per verdict,
# which would be the storm the escape hatch exists to prevent.
# ---------------------------------------------------------------------------


class _FakeEscalationQueue:
    """Minimal queue exposing only what the storm filer touches.

    ``has_open_l1`` mirrors the REAL semantics
    (``escalation/src/escalation/queue.py::EscalationQueue.has_open_l1``): a
    category filter narrows the match, and ``category=None`` matches any open
    L1 on the id.  A stub that ignored the category would answer every dedup
    question identically and could not distinguish the narrow guard this task
    requires from the widened one the task-4105 incident forbids.
    """

    def __init__(
        self,
        *,
        submit_raises: bool = False,
        preopen: list[tuple[str, str]] | None = None,
    ) -> None:
        self.submitted: list = []
        #: Every ``(task_id, category)`` pair the filer asked about.
        self.checked: list[tuple[str, str | None]] = []
        self._submit_raises = submit_raises
        self._preopen = list(preopen or [])

    def _open_rows(self) -> list[tuple[str, str]]:
        return self._preopen + [(e.task_id, e.category) for e in self.submitted]

    def has_open_l1(self, task_id: str, *, category: str | None = None) -> bool:
        self.checked.append((task_id, category))
        return any(
            row_task == task_id and (category is None or row_category == category)
            for row_task, row_category in self._open_rows()
        )

    def make_id(self, task_id: str) -> str:
        return f'esc-{task_id}-{len(self.submitted) + 1}'

    def submit(self, escalation) -> None:
        if self._submit_raises:
            raise RuntimeError('escalation queue is down')
        self.submitted.append(escalation)


def _as_queue(queue: _FakeEscalationQueue | None) -> EscalationQueue | None:
    """Present the fake to the production signature's concrete queue type.

    The producers annotate ``escalation_queue: EscalationQueue | None`` (a
    concrete class, not a Protocol), so a structural stand-in is not statically
    assignable.  Cast at the CALL, matching
    ``test_offline_lane_integration.py``, rather than casting where the fake is
    built — the tests then go on to read ``.submitted`` / ``.checked`` off it,
    which the real type does not carry.

    Deliberately not a ``MagicMock`` (the older spelling a sibling test uses):
    the class above mirrors ``has_open_l1``'s real category-filter semantics,
    which is exactly the distinction these tests exist to make.
    """
    return cast('EscalationQueue | None', queue)


def _storm_config(*, rate: int = 3, enabled: bool = True) -> RecoveryEmissionConfig:
    """A deliberately TINY rate so a storm costs four git calls, not eleven.

    The threshold under test is the comparison, not the shipped number — that
    is pinned once, in ``test_config.py``'s ``TestRecoveryEmissionConfig``.
    """
    return RecoveryEmissionConfig(
        landing_git_error_rate_per_hour=rate,
        landing_git_error_escalation_enabled=enabled,
    )


async def _drive_git_errors(
    git_ops: GitOps,
    count: int,
    *,
    queue: _FakeEscalationQueue | None = None,
    recovery_emission: RecoveryEmissionConfig | None = None,
) -> list[LandingVerdict]:
    """Produce *count* genuine ``git_error`` verdicts against a real repo.

    An unresolvable branch ref is the cheapest one — a single ``git rev-parse``
    that legitimately fails — so the storm is driven through the REAL producer
    rather than by calling ``tally.record`` directly.  That matters: the thing
    under test is that ``branch_work_landed`` charges the counter on every
    verdict it returns, which a direct ``record`` call would assume rather than
    check.
    """
    verdicts: list[LandingVerdict] = []
    for _ in range(count):
        verdict = await branch_work_landed(
            git_ops, TASK_ID, 'task/does-not-exist', branch_tip_sha=None,
            escalation_queue=_as_queue(queue), recovery_emission=recovery_emission,
        )
        assert verdict.reason is LandingReason.git_error, verdict.probe
        verdicts.append(verdict)
    return verdicts


@pytest.fixture
def tally(monkeypatch: pytest.MonkeyPatch) -> LandingTally:
    """A FRESH module-level tally for each test.

    A process-lifetime monotonic counter is exactly right in the orchestrator
    and exactly wrong across a test session: shared, every count would depend
    on test ORDER, and the rate gate would fire in whichever test happened to
    run last.  Replacing the module global is the isolation seam.
    """
    fresh = LandingTally()
    monkeypatch.setattr(landing_evidence, 'LANDING_TALLY', fresh)
    return fresh


class TestLandingTally:
    """The tally itself: monotonic, whole-vocabulary, and window-aware."""

    def test_every_reason_in_the_vocabulary_has_a_key(self) -> None:
        """Keyed from the ENUM, so a new member can never go untallied.

        Seeding from ``LandingReason`` rather than from a literal list is the
        whole point: a future reason added to the vocabulary and forgotten here
        would be invisible in the tally, which is the one place an operator
        looks to find out what the detector has been saying.
        """
        assert set(LandingTally().snapshot()) == set(LandingReason)

    def test_unseen_reasons_read_zero_rather_than_missing(self) -> None:
        """A seeded key, not a ``KeyError`` and not an absent row."""
        snapshot = LandingTally().snapshot()
        assert all(count == 0 for count in snapshot.values())
        assert snapshot[LandingReason.no_op_landing] == 0

    def test_counts_accumulate_per_reason(self) -> None:
        subject = LandingTally()
        subject.record(LandingReason.landed)
        subject.record(LandingReason.landed)
        subject.record(LandingReason.git_error)
        snapshot = subject.snapshot()
        assert snapshot[LandingReason.landed] == 2
        assert snapshot[LandingReason.git_error] == 1
        assert snapshot[LandingReason.not_landed] == 0

    def test_counts_are_monotonic(self) -> None:
        """No count ever decreases, sampled across the whole sequence.

        Sampled rather than end-state-checked for the same reason B1 samples
        every churn commit: "never decreased" and "decreased and recovered" are
        indistinguishable from a single final reading.
        """
        subject = LandingTally()
        previous = subject.snapshot()
        for index, reason in enumerate(list(LandingReason) * 3):
            subject.record(reason)
            current = subject.snapshot()
            assert all(
                current[key] >= previous[key] for key in LandingReason
            ), f'a count decreased at step {index} ({previous} -> {current})'
            previous = current
        assert sum(previous.values()) == len(LandingReason) * 3

    def test_the_snapshot_is_a_copy(self) -> None:
        """A caller mutating what it got back must not corrupt the tally."""
        subject = LandingTally()
        subject.record(LandingReason.landed)
        taken = subject.snapshot()
        taken[LandingReason.landed] = 999
        assert subject.snapshot()[LandingReason.landed] == 1

    def test_the_git_error_rate_window_is_the_trailing_hour(self) -> None:
        """The RATE window and the MONOTONIC tally are different questions.

        The tally answers "what has this detector said, ever" and must never
        decrease.  The rate window answers "is it failing RIGHT NOW" and must,
        or a single bad hour would keep the alarm latched for the life of the
        process.
        """
        now = [0.0]
        subject = LandingTally(clock=lambda: now[0])
        for _ in range(5):
            subject.record(LandingReason.git_error)
        assert subject.git_error_count_in_window() == 5

        now[0] = subject.window_secs + 1.0
        assert subject.git_error_count_in_window() == 0, 'the window must slide'
        assert subject.snapshot()[LandingReason.git_error] == 5, (
            'the cumulative tally is monotonic and must NOT slide with it'
        )

    def test_only_git_error_charges_the_rate_window(self) -> None:
        """A repo with genuinely nothing landed in it is not a storm."""
        subject = LandingTally()
        for reason in LandingReason:
            if reason is not LandingReason.git_error:
                subject.record(reason)
        assert subject.git_error_count_in_window() == 0


@pytest.mark.asyncio
class TestTallyIsChargedByEveryVerdict:
    """``branch_work_landed`` charges the tally on EVERY verdict it returns."""

    async def test_an_accept_and_a_git_error_are_both_recorded(
        self, git_ops: GitOps, repo: _Repo, tally: LandingTally,
    ) -> None:
        scenario = build_rebased_landing(repo)
        accepted = await branch_work_landed(
            git_ops, TASK_ID, scenario.branch,
            branch_tip_sha=scenario.branch_tip_sha, metadata=scenario.metadata,
        )
        assert accepted.accepted is True
        assert tally.snapshot()[LandingReason.landed] == 1

        await _drive_git_errors(git_ops, 1)
        snapshot = tally.snapshot()
        assert snapshot[LandingReason.landed] == 1
        assert snapshot[LandingReason.git_error] == 1

    async def test_a_rejection_is_recorded_under_its_own_reason(
        self, git_ops: GitOps, repo: _Repo, tally: LandingTally,
    ) -> None:
        scenario = build_unlanded_branch(repo)
        await branch_work_landed(
            git_ops, TASK_ID, scenario.branch,
            branch_tip_sha=scenario.branch_tip_sha, metadata=scenario.metadata,
        )
        assert tally.snapshot()[LandingReason.not_landed] == 1

    async def test_every_pass_logs_the_snapshot(
        self, git_ops: GitOps, repo: _Repo, tally: LandingTally,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """An operator must be able to see the tally WITHOUT a dashboard.

        A counter nobody can read is not an escape hatch; it is a second
        silence layered on the first.
        """
        scenario = build_rebased_landing(repo)
        with caplog.at_level('INFO', logger='orchestrator.landing_evidence'):
            await branch_work_landed(
                git_ops, TASK_ID, scenario.branch,
                branch_tip_sha=scenario.branch_tip_sha, metadata=scenario.metadata,
            )
        messages = [record.getMessage() for record in caplog.records]
        assert any('landing tally' in message for message in messages), messages
        assert any('landed' in message for message in messages), messages


@pytest.mark.asyncio
class TestGitErrorStormEscape:
    """The rate-gated L1: fires ONCE per storm, and only for ``git_error``."""

    async def test_no_queue_files_nothing_and_never_raises(
        self, git_ops: GitOps, repo: _Repo, tally: LandingTally,
    ) -> None:
        """``escalation_queue=None`` is a SUPPORTED shape, not a degraded one.

        Every verdict this producer returns is correct without a queue (bare
        harness and bare worker unit tests construct exactly that), so a
        missing queue must cost the storm alarm and nothing else — the same
        best-effort guard ``file_unattributed_landing_escalation`` uses.
        """
        build_unlanded_branch(repo)
        verdicts = await _drive_git_errors(
            git_ops, 5, queue=None, recovery_emission=_storm_config(rate=1),
        )
        assert len(verdicts) == 5
        assert tally.git_error_count_in_window() == 5

    async def test_more_than_the_rate_files_exactly_one_blocking_l1(
        self, git_ops: GitOps, repo: _Repo, tally: LandingTally,
    ) -> None:
        build_unlanded_branch(repo)
        queue = _FakeEscalationQueue()
        await _drive_git_errors(
            git_ops, 4, queue=queue, recovery_emission=_storm_config(rate=3),
        )
        assert len(queue.submitted) == 1, (
            'a storm files ONE alarm; one per verdict IS the storm'
        )
        escalation = queue.submitted[0]
        assert escalation.severity == 'blocking'
        assert escalation.level == 1

    async def test_at_or_below_the_rate_files_none(
        self, git_ops: GitOps, repo: _Repo, tally: LandingTally,
    ) -> None:
        """The threshold is a strict EXCEEDANCE, so the rate itself is quiet."""
        build_unlanded_branch(repo)
        queue = _FakeEscalationQueue()
        await _drive_git_errors(
            git_ops, 3, queue=queue, recovery_emission=_storm_config(rate=3),
        )
        assert queue.submitted == []

    async def test_other_reasons_never_file_however_many_there_are(
        self, git_ops: GitOps, repo: _Repo, tally: LandingTally,
    ) -> None:
        """A repo with genuinely nothing landed must never page anyone."""
        scenario = build_unlanded_branch(repo)
        queue = _FakeEscalationQueue()
        for _ in range(6):
            verdict = await branch_work_landed(
                git_ops, TASK_ID, scenario.branch,
                branch_tip_sha=scenario.branch_tip_sha, metadata=scenario.metadata,
                escalation_queue=_as_queue(queue), recovery_emission=_storm_config(rate=1),
            )
            assert verdict.reason is LandingReason.not_landed
        assert queue.submitted == []

    async def test_the_kill_switch_suppresses_the_filing_not_the_tally(
        self, git_ops: GitOps, repo: _Repo, tally: LandingTally,
    ) -> None:
        """The narrow kill switch: silence the alarm, keep the telemetry.

        Same discipline as ``streak_escalation_enabled`` — the only part of the
        mechanism that WRITES to the escalation queue gets its own switch, so a
        noisy alarm can be silenced without losing what explains it.
        """
        build_unlanded_branch(repo)
        queue = _FakeEscalationQueue()
        await _drive_git_errors(
            git_ops, 4, queue=queue,
            recovery_emission=_storm_config(rate=3, enabled=False),
        )
        assert queue.submitted == []
        assert tally.snapshot()[LandingReason.git_error] == 4

    async def test_the_dedup_uses_its_own_category(
        self, git_ops: GitOps, repo: _Repo, tally: LandingTally,
    ) -> None:
        """NEVER ``provenance_unattributed``, and never a bare ``task_id``.

        Category scoping is load-bearing after the task-4105 incident
        (``landing_evidence.py::file_unattributed_landing_escalation``): a
        ``category=None`` dedup matches ANY open L1, so an unrelated escalation
        silently suppresses the filing.  Widening it back would let a broken
        DETECTOR hide behind whatever else happened to be open.
        """
        build_unlanded_branch(repo)
        queue = _FakeEscalationQueue()
        await _drive_git_errors(
            git_ops, 4, queue=queue, recovery_emission=_storm_config(rate=3),
        )
        assert queue.checked, 'the filer must consult has_open_l1'
        for _task_id, category in queue.checked:
            assert category == LANDING_GIT_ERROR_STORM_CATEGORY
            assert category is not None, 'a bare task_id dedup is the 4105 defect'
            assert category != 'provenance_unattributed'
        assert queue.submitted[0].category == LANDING_GIT_ERROR_STORM_CATEGORY

    async def test_a_second_storm_in_the_same_window_files_no_duplicate(
        self, git_ops: GitOps, repo: _Repo, tally: LandingTally,
    ) -> None:
        build_unlanded_branch(repo)
        queue = _FakeEscalationQueue()
        await _drive_git_errors(
            git_ops, 4, queue=queue, recovery_emission=_storm_config(rate=3),
        )
        assert len(queue.submitted) == 1
        await _drive_git_errors(
            git_ops, 4, queue=queue, recovery_emission=_storm_config(rate=3),
        )
        assert len(queue.submitted) == 1, 'has_open_l1 must suppress the re-file'

    async def test_an_open_l1_in_another_category_does_not_suppress_it(
        self, git_ops: GitOps, repo: _Repo, tally: LandingTally,
    ) -> None:
        """The other half of the 4105 lesson, asserted in the live direction."""
        build_unlanded_branch(repo)
        queue = _FakeEscalationQueue(
            preopen=[(LANDING_GIT_ERROR_STORM_SENTINEL, 'task_failure')],
        )
        await _drive_git_errors(
            git_ops, 4, queue=queue, recovery_emission=_storm_config(rate=3),
        )
        assert len(queue.submitted) == 1

    async def test_it_is_filed_against_a_synthetic_sentinel_not_a_task(
        self, git_ops: GitOps, repo: _Repo, tally: LandingTally,
    ) -> None:
        """A storm is a claim about the DETECTOR, so it names no task.

        Filing it against a real task id would be arbitrary (a storm spans
        every task the sweep touched) and actively harmful: an open L1 on a
        task is read by the recovery predicates as a hold, so the alarm would
        deepen the very strand it reports.  Same reasoning and same shape as
        ``recovery_emission.RECOVERY_VETO_STREAK_SENTINEL_PREFIX``.
        """
        build_unlanded_branch(repo)
        queue = _FakeEscalationQueue()
        await _drive_git_errors(
            git_ops, 4, queue=queue, recovery_emission=_storm_config(rate=3),
        )
        escalation = queue.submitted[0]
        assert escalation.task_id == LANDING_GIT_ERROR_STORM_SENTINEL
        assert escalation.task_id != TASK_ID

    async def test_the_detail_names_the_tally_the_window_and_the_threshold(
        self, git_ops: GitOps, repo: _Repo, tally: LandingTally,
    ) -> None:
        build_unlanded_branch(repo)
        queue = _FakeEscalationQueue()
        await _drive_git_errors(
            git_ops, 4, queue=queue, recovery_emission=_storm_config(rate=3),
        )
        detail = queue.submitted[0].detail
        assert 'git_error' in detail
        assert '4' in detail, 'the observed count'
        assert '3' in detail, 'the configured threshold'
        assert 'hour' in detail.lower(), 'the window'
        # The per-reason tally, not just the git_error row.
        for reason in LandingReason:
            assert str(reason) in detail, f'{reason} missing from the tally block'

    async def test_the_detail_says_the_detector_is_broken(
        self, git_ops: GitOps, repo: _Repo, tally: LandingTally,
    ) -> None:
        """The one sentence that decides what the reader does next.

        Without it a human reads a wall of rejected verdicts and concludes the
        tasks never landed — which is precisely the inference this whole PRD
        exists to prevent, made by a person instead of by code.
        """
        build_unlanded_branch(repo)
        queue = _FakeEscalationQueue()
        await _drive_git_errors(
            git_ops, 4, queue=queue, recovery_emission=_storm_config(rate=3),
        )
        escalation = queue.submitted[0]
        blob = f'{escalation.summary}\n{escalation.detail}'.lower()
        assert 'detector' in blob
        assert 'not_landed' in blob, (
            'the detail must say plainly that these are NOT not_landed verdicts'
        )
        assert escalation.suggested_action

    async def test_a_broken_queue_never_breaks_the_verdict(
        self, git_ops: GitOps, repo: _Repo, tally: LandingTally,
    ) -> None:
        """Best-effort, exactly as the sibling filer is: log and continue.

        A recovery sweep that dies because its own alarm could not be filed
        stops recovering every OTHER task in the same pass.
        """
        build_unlanded_branch(repo)
        queue = _FakeEscalationQueue(submit_raises=True)
        verdicts = await _drive_git_errors(
            git_ops, 4, queue=queue, recovery_emission=_storm_config(rate=3),
        )
        assert all(v.reason is LandingReason.git_error for v in verdicts)
        assert queue.submitted == []

    async def test_the_filer_is_directly_callable_and_reports_what_it_did(
        self, tally: LandingTally,
    ) -> None:
        """The filing is a named, testable unit — not an inline block.

        ``file_unattributed_landing_escalation`` is extracted for the same
        reason: two call sites already inline near-verbatim filing boilerplate,
        and a third would have been the copy that drifts.
        """
        for _ in range(4):
            tally.record(LandingReason.git_error)
        queue = _FakeEscalationQueue()
        filed = file_landing_git_error_storm_escalation(
            _as_queue(queue), tally=tally, rate_per_hour=3,
        )
        assert filed is True
        assert len(queue.submitted) == 1
        assert file_landing_git_error_storm_escalation(
            _as_queue(queue), tally=tally, rate_per_hour=3,
        ) is False, 'the dedup answer must be reported, not merely applied'
        assert file_landing_git_error_storm_escalation(
            None, tally=tally, rate_per_hour=3,
        ) is False
