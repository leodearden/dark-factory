"""Real-git pins for the rc=0 group-merge candidate query (tasks 4926, 5539, 5165).

``skills/_shared/deriving-landed-sha.md``'s step 4, rc=0 arm, finds the merge
that brought a landed branch into main with::

    c=$(git rev-list --topo-order --ancestry-path --merges task/<TASK_ID>..main | tail -1)
    git merge-base --is-ancestor task/<TASK_ID> "$c^1"   # "contained-before"

and stamps ``$c`` only on ``contained-before rc=1``.  Each premise of that
pair is pinned here:

* **Ordering.** Without ``--topo-order`` the walk is date-ordered, so clock
  skew or commits sharing one committer-date second make ``tail -1`` a later,
  unrelated merge whose rc=0 silently skips the real one.
* **Merge direction.** Git records none: a branch-side landing is
  byte-identical to a fast-forward landing plus an unrelated merge, so ``^1``
  is main only where merges are made on main; elsewhere rc=0 is undecided.
* **Stacked sync merge.** A branch stacked on this one that merged main in
  sorts older than the landing, so the arm exits rc=0 to the citation gate.
* **Legitimate rc=0.** A branch already in main yields a real candidate the
  arm must NOT stamp.
* **Reachability.** Only the no-op-rebase train member reaches rc=0; the
  ordinary coalesce-absorbed member stays rc=1 (``skills/merge-queue/SKILL.md``
  rule 3).

These are characterization pins, expected to pass on write: a failure means
git's behaviour moved, except in the agreement assertions, where it means the
runbook or production moved.  Prose greps of the runbook are deliberately absent;
see ``orchestrator/tests/test_roles_ancestry_check.py``'s module docstring.
Every fixture repo carries the esc-3072-3 isolation guards through
``_ladder_git.py::new_repo`` and ``_ladder_git.py::git_env``.

``orchestrator/src/orchestrator/git_ops.py::GitOps.landing_merge_for`` runs the
same query for ``orchestrator/src/orchestrator/merge_lane/landing_evidence.py::_no_op_question``
(``--reverse`` with the first element is ``tail -1``), and every candidate here is
asserted to agree with it, as ``test_citation_gate_positive_arm.py`` does for the citation walk.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

from _ladder_git import commit, git, git_rc, new_repo

from orchestrator.config import GitConfig
from orchestrator.git_ops import GitOps

TASK_ID = 'T'

_PINNED_DATE = '2020-01-01T00:00:00 +0000'

# Real-git fixture helpers shared with this ladder's other pins: ``_ladder_git.py``.


def _candidate_date_ordered(root: Path, tid: str) -> str:
    """The rc=0 arm's candidate query WITHOUT ``--topo-order``: date-ordered."""
    out = git(root, 'rev-list', '--ancestry-path', '--merges', f'task/{tid}..main')
    return out.split('\n')[-1] if out else ''


def _candidate_topo(root: Path, tid: str) -> str:
    """The rc=0 arm's candidate query WITH ``--topo-order`` (the fix)."""
    out = git(
        root, 'rev-list', '--topo-order', '--ancestry-path', '--merges', f'task/{tid}..main',
    )
    return out.split('\n')[-1] if out else ''


def _production_landing_merge(root: Path, tid: str) -> str | None:
    """The REAL ``GitOps.landing_merge_for``, called as ``_no_op_question`` calls it."""
    tip = git(root, 'rev-parse', f'task/{tid}')
    git_ops = GitOps(GitConfig(main_branch='main', branch_prefix='task/'), root)
    return asyncio.run(git_ops.landing_merge_for(tip, 'main'))


def _assert_runbook_agrees_with_production(root: Path, tid: str, candidate: str) -> None:
    """Assert the runbook's *candidate* for ``task/<tid>`` equals ``GitOps.landing_merge_for``."""
    assert _production_landing_merge(root, tid) == candidate, (
        "the runbook's rc=0 candidate query (skills/_shared/deriving-landed-sha.md step 4) "
        'and GitOps.landing_merge_for must agree; a disagreement means one of them moved '
        'and the other needs the same change'
    )


def _contained_before(root: Path, tid: str, rev: str) -> int:
    """The arm's ``contained-before`` rc for candidate expression *rev*."""
    return git_rc(root, 'merge-base', '--is-ancestor', f'task/{tid}', rev)


def test_clock_skew_makes_the_date_ordered_candidate_query_pick_the_wrong_merge(
    tmp_path: Path,
) -> None:
    """Without ``--topo-order``, ``tail -1`` can return a LATER unrelated merge.

    The fixture is the minimal shape that actually reproduces the defect,
    and it must NOT be simplified.  Git's default walk pops from a
    date-ordered priority queue seeded only with the TIPS and enqueues a
    parent only after popping one of its children, so a parent can never be
    emitted before its *only* child regardless of date skew -- a linear
    history or a single later merge passes vacuously and pins nothing.  The
    violation needs the bringing-in merge to have TWO children in the walk,
    so one is still queued (with an older date) when the parent is popped.

    Measured against git 2.43.0: default order emits ``MF2, X, MF1`` so
    ``tail -1`` is MF1; ``--topo-order`` emits ``MF2, MF1, X`` so
    ``tail -1`` is X, the merge that actually brought the branch in.
    """
    root = tmp_path / 'repo'
    new_repo(root)

    commit(root, 'base', 'base.txt', date='2020-01-01T00:00:00 +0000')

    git(root, 'checkout', '-b', f'task/{TASK_ID}')
    commit(root, 'work', 'work.txt', date='2020-01-02T00:00:00 +0000')

    # X: the merge that brings task/T in -- committer date skewed NEW
    # (01-10), later than both sibling branch commits below.
    git(root, 'checkout', 'main')
    git(
        root, 'merge', '--no-ff', f'task/{TASK_ID}', '-m', f'Merge task/{TASK_ID} into main',
        date='2020-01-10T00:00:00 +0000',
    )
    x = git(root, 'rev-parse', 'HEAD')

    # f1 and f2 BOTH fork from X, so X has two children in the walk.
    git(root, 'checkout', '-b', 'f1', x)
    commit(root, 'f1work', 'f1.txt', date='2020-01-03T00:00:00 +0000')
    git(root, 'checkout', '-b', 'f2', x)
    commit(root, 'f2work', 'f2.txt', date='2020-01-05T00:00:00 +0000')

    git(root, 'checkout', 'main')
    git(root, 'merge', '--no-ff', 'f1', '-m', 'Merge f1 into main',
        date='2020-01-04T00:00:00 +0000')
    mf1 = git(root, 'rev-parse', 'HEAD')
    git(root, 'merge', '--no-ff', 'f2', '-m', 'Merge f2 into main',
        date='2020-01-20T00:00:00 +0000')
    mf2 = git(root, 'rev-parse', 'HEAD')

    assert len({x, mf1, mf2}) == 3, 'fixture must build three distinct merge commits'

    # (1) The date-ordered form returns f1's merge, NOT X.
    date_ordered = _candidate_date_ordered(root, TASK_ID)
    assert date_ordered == mf1, (
        'expected the date-ordered walk to put an unrelated LATER merge last; if this '
        'stops holding, git changed its default ordering and '
        'skills/_shared/deriving-landed-sha.md step 4 must be re-litigated'
    )
    assert date_ordered != x, (
        'the whole defect is that the date-ordered query does NOT return the merge that '
        'brought the branch in'
    )

    # (2) ...so the arm takes the "unrelated later merge, do not stamp" exit,
    #     silently skipping the true group merge.
    assert _contained_before(root, TASK_ID, f'{date_ordered}^1') == 0, (
        'the wrong candidate yields contained-before rc=0, which the rc=0 arm reads as '
        '"unrelated later merge, do not stamp" -- the true group merge is skipped and '
        'the procedure degrades to the citation path'
    )

    # (3) --topo-order returns exactly X, the bringing-in merge.
    candidate = _candidate_topo(root, TASK_ID)
    assert candidate == x, (
        '--topo-order is what makes `tail -1` genuinely the oldest DESCENDANT merge'
    )
    _assert_runbook_agrees_with_production(root, TASK_ID, candidate)

    # (4) ...whose containment check takes the correct "stamp it" arm.
    assert _contained_before(root, TASK_ID, f'{x}^1') == 1, (
        'contained-before rc=1 means the branch was not in main before X, so X IS the '
        'merge that brought it in -- the stampable arm'
    )


def test_same_second_commits_make_the_date_ordered_candidate_query_pick_the_wrong_merge(
    tmp_path: Path,
) -> None:
    """Commits sharing one committer-date second reproduce the defect with NO skew.

    This is the realistic single-machine cause.  The ``a`` and ``b`` commits
    are load-bearing: with tied dates the walk is first-in-first-out, and X
    is emitted before MF1 only when MF1 sits farther from the tip than X's
    path through f2.  Drop them and the default order happens to be right,
    so the pin would pass vacuously.
    """
    root = tmp_path / 'repo'
    new_repo(root)

    commit(root, 'base', 'base.txt', date=_PINNED_DATE)
    git(root, 'checkout', '-b', f'task/{TASK_ID}')
    commit(root, 'work', 'work.txt', date=_PINNED_DATE)

    git(root, 'checkout', 'main')
    git(
        root, 'merge', '--no-ff', f'task/{TASK_ID}', '-m', f'Merge task/{TASK_ID} into main',
        date=_PINNED_DATE,
    )
    x = git(root, 'rev-parse', 'HEAD')

    git(root, 'checkout', '-b', 'f1', x)
    commit(root, 'f1work', 'f1.txt', date=_PINNED_DATE)
    git(root, 'checkout', '-b', 'f2', x)
    commit(root, 'f2work', 'f2.txt', date=_PINNED_DATE)

    git(root, 'checkout', 'main')
    git(root, 'merge', '--no-ff', 'f1', '-m', 'Merge f1 into main', date=_PINNED_DATE)
    mf1 = git(root, 'rev-parse', 'HEAD')
    commit(root, 'a', 'a.txt', date=_PINNED_DATE)
    commit(root, 'b', 'b.txt', date=_PINNED_DATE)
    git(root, 'merge', '--no-ff', 'f2', '-m', 'Merge f2 into main', date=_PINNED_DATE)

    committer_dates = set(git(root, 'log', '--all', '--format=%ct').split('\n'))
    assert len(committer_dates) == 1, (
        'every commit must share one committer-date second: this pin is about ties, not skew'
    )

    assert _candidate_date_ordered(root, TASK_ID) == mf1, (
        'with tied dates the date-ordered walk puts an unrelated LATER merge last; if this '
        'stops holding, git changed its tie-break order'
    )
    assert _contained_before(root, TASK_ID, f'{mf1}^1') == 0, (
        'the wrong candidate takes the "do not stamp" exit, skipping the true group merge'
    )
    candidate = _candidate_topo(root, TASK_ID)
    assert candidate == x, (
        '--topo-order returns the merge that brought the branch in, ties or not'
    )
    _assert_runbook_agrees_with_production(root, TASK_ID, candidate)
    assert _contained_before(root, TASK_ID, f'{x}^1') == 1


def _build_branch_side_landing(root: Path) -> None:
    """Land ``task/T`` by a merge made on the branch side, fast-forwarded onto main."""
    new_repo(root)
    commit(root, 'base', 'base.txt', date=_PINNED_DATE)
    git(root, 'checkout', '-b', f'task/{TASK_ID}')
    commit(root, 'work', 'work.txt', date=_PINNED_DATE)
    git(root, 'checkout', 'main')
    commit(root, 'side', 'side.txt', date=_PINNED_DATE)

    git(root, 'checkout', '-b', 'integ', f'task/{TASK_ID}')
    git(root, 'merge', '--no-ff', 'main', '-m', 'Merge', date=_PINNED_DATE)
    git(root, 'checkout', 'main')
    assert git_rc(root, 'merge-base', '--is-ancestor', f'task/{TASK_ID}', 'main') == 1, (
        'task/T must not be on main yet, so the branch-side merge is what brings it in'
    )
    git(root, 'merge', '--ff-only', 'integ')


def _build_fast_forward_then_unrelated_merge(root: Path) -> None:
    """Land ``task/T`` by fast-forward, then merge an unrelated branch on main."""
    new_repo(root)
    commit(root, 'base', 'base.txt', date=_PINNED_DATE)
    git(root, 'checkout', '-b', f'task/{TASK_ID}')
    commit(root, 'work', 'work.txt', date=_PINNED_DATE)
    git(root, 'checkout', '-b', 'f1', 'main')
    commit(root, 'side', 'side.txt', date=_PINNED_DATE)

    git(root, 'checkout', 'main')
    git(root, 'merge', '--ff-only', f'task/{TASK_ID}')
    assert git(root, 'rev-parse', 'main') == git(root, 'rev-parse', f'task/{TASK_ID}'), (
        'task/T must land with NO merge commit, so the later merge is unrelated to it'
    )
    git(root, 'merge', '--no-ff', 'f1', '-m', 'Merge', date=_PINNED_DATE)


def test_a_branch_side_landing_is_the_same_history_as_a_fast_forward_then_an_unrelated_merge(
    tmp_path: Path,
) -> None:
    """Git records no merge direction, so ``contained-before rc=0`` can be undecided.

    Story A lands ``task/T`` through a merge built on the branch side: that
    merge IS the landing.  Story B fast-forwards main onto ``task/T``, then
    merges an unrelated branch on main: that merge must NOT be stamped.  With
    one pinned date and identical subjects the two are byte-identical, so no
    git query can tell them apart.  Where merge direction is unknown, rc=0 is
    therefore undecided, and the citation gate is the only exit safe for both.
    """
    a, b = tmp_path / 'a', tmp_path / 'b'
    _build_branch_side_landing(a)
    _build_fast_forward_then_unrelated_merge(b)

    for ref in ('main', f'task/{TASK_ID}'):
        assert git(a, 'rev-parse', ref) == git(b, 'rev-parse', ref), (
            f'the two stories must be one commit graph, sha for sha, at {ref} -- that is '
            'why no git query can tell them apart'
        )

    candidate = _candidate_topo(a, TASK_ID)
    assert candidate == git(a, 'rev-parse', 'main')
    _assert_runbook_agrees_with_production(a, TASK_ID, candidate)
    assert _contained_before(a, TASK_ID, f'{candidate}^1') == 0, (
        'the task tip is ^1, so the arm takes its "do not stamp" exit -- wrong for story '
        'A, right for story B'
    )
    assert _contained_before(a, TASK_ID, f'{candidate}^2') == 1, (
        'a "^2 instead" remedy would stamp -- right for story A, a fabricated stamp for '
        'story B'
    )


def test_a_merge_created_on_main_puts_main_at_caret_1(tmp_path: Path) -> None:
    """The control for the branch-side identity pin: merged FROM main, ``^1`` is main.

    Merged the dark-factory way (``git merge --no-ff task/T`` from main),
    ``^1`` IS main-just-before-the-merge and the containment check renders
    the correct verdict, so the ambiguity
    ``test_a_branch_side_landing_is_the_same_history_as_a_fast_forward_then_an_unrelated_merge``
    pins exists only where merge direction is unknown.
    """
    root = tmp_path / 'repo'
    new_repo(root)

    commit(root, 'base', 'base.txt')
    git(root, 'checkout', '-b', f'task/{TASK_ID}')
    commit(root, 'work', 'work.txt')
    task_tip = git(root, 'rev-parse', f'task/{TASK_ID}')

    git(root, 'checkout', 'main')
    mainadv = commit(root, 'mainadv', 'main.txt')
    git(root, 'merge', '--no-ff', f'task/{TASK_ID}', '-m', f'Merge task/{TASK_ID} into main')
    merge = git(root, 'rev-parse', 'HEAD')

    assert git(root, 'rev-parse', f'{merge}^1') == mainadv, (
        'a merge created on main has main-just-before-the-merge as its first parent'
    )
    assert git(root, 'rev-parse', f'{merge}^2') == task_tip

    candidate = _candidate_topo(root, TASK_ID)
    assert candidate == merge
    _assert_runbook_agrees_with_production(root, TASK_ID, candidate)
    assert _contained_before(root, TASK_ID, f'{candidate}^1') == 1, (
        'merged from main, the documented ^1 check renders the correct "this IS the '
        'merge that brought it in" verdict -- so the ambiguity above is about unknown '
        'merge DIRECTION, not about the check itself'
    )


def test_a_stacked_branch_that_merged_main_first_hides_the_landing_merge(
    tmp_path: Path,
) -> None:
    """A sync merge on a branch stacked on ``task/T`` sorts older than the landing.

    ``task/S`` is stacked on ``task/T``, merges main into itself (Y, as a
    conflict-resolution "merge main into" commit does), then lands via Z
    made on main.  ``tail -1`` returns Y, whose ``^1`` already contains the
    branch, so the arm exits rc=0 to the citation gate: Z is skipped, but
    nothing is wrongly stamped.  Dark-factory produces this shape (a carry
    branch stacked on a task branch that merged main in).
    """
    root = tmp_path / 'repo'
    new_repo(root)

    commit(root, 'base', 'base.txt')
    git(root, 'checkout', '-b', f'task/{TASK_ID}')
    commit(root, 'work', 'work.txt')
    git(root, 'checkout', '-b', 'task/S')
    commit(root, 'swork', 's.txt')

    git(root, 'checkout', 'main')
    commit(root, 'mainadv', 'm.txt')
    git(root, 'checkout', 'task/S')
    git(root, 'merge', '--no-ff', 'main', '-m', 'Y')
    y = git(root, 'rev-parse', 'HEAD')

    git(root, 'checkout', 'main')
    commit(root, 'mainadv2', 'm2.txt')
    git(root, 'merge', '--no-ff', 'task/S', '-m', 'Z')
    z = git(root, 'rev-parse', 'HEAD')

    candidate = _candidate_topo(root, TASK_ID)
    assert candidate == y, (
        'the oldest descendant merge is the stacked branch\'s sync merge, not the landing'
    )
    _assert_runbook_agrees_with_production(root, TASK_ID, candidate)
    assert _contained_before(root, TASK_ID, f'{y}^1') == 0, (
        'Y\'s first parent already contains the branch, so the arm takes its do-not-stamp '
        'exit to the citation gate -- safe, no wrong stamp'
    )
    assert _contained_before(root, TASK_ID, f'{z}^1') == 1, (
        'Z was the stampable landing merge the arm skips'
    )


def test_an_already_landed_branch_yields_the_arms_legitimate_rc0_verdict(
    tmp_path: Path,
) -> None:
    """The arm's CORRECT rc=0 exit: a real candidate exists, and must NOT be stamped.

    The pins above are FALSE or undecided rc=0 verdicts (wrong candidate under
    date order; unknown merge direction) and the ones below assert the positive
    "stamp it" arm, so nothing yet executes the outcome rc=0 actually exists to produce.
    This is that mirror.

    ``task/T`` never advances past its creation point, so it is already
    contained in main by construction -- the fast-forward / phantom-branch
    landing -- and main then advances via an UNRELATED merge.  Two things must
    hold together: the candidate query is NON-EMPTY, so the arm reaches its
    containment check instead of short-circuiting on the ``$c`` empty path, and
    that check returns 0 because the branch really was in main beforehand.  This
    makes executable the second half of step 4's claim that the oldest
    topological descendant merge is "an unrelated later merge only when the
    branch was already in main before it".

    If this ever flipped to rc=1, the suite would otherwise stay green while the
    runbook started stamping ``$c`` as the merge that brought this branch in --
    fabricated landing evidence for a branch carrying none of the task's work,
    which the citation gate below the arm exists to catch.
    """
    root = tmp_path / 'repo'
    new_repo(root)

    base = commit(root, 'base', 'base.txt')
    git(root, 'branch', f'task/{TASK_ID}')  # left AT base; never advanced

    git(root, 'checkout', '-b', 'f1')
    commit(root, 'f1work', 'f1.txt')
    git(root, 'checkout', 'main')
    git(root, 'merge', '--no-ff', 'f1', '-m', 'Merge f1 into main')
    mf1 = git(root, 'rev-parse', 'HEAD')

    assert git(root, 'rev-parse', f'task/{TASK_ID}') == base, (
        'the branch must still sit at its creation point -- that is what makes it '
        'already contained in main, which is the precondition for a legitimate rc=0'
    )
    assert git(root, 'rev-parse', f'{mf1}^1') == base, (
        'the unrelated merge was created on main, so its first parent is '
        'main-just-before-it, which already contains the branch'
    )

    candidate = _candidate_topo(root, TASK_ID)
    assert candidate == mf1, (
        'the arm must reach its containment check with a REAL candidate here: rc=0 has '
        'to be decided by the check, not short-circuited by an empty $c'
    )
    _assert_runbook_agrees_with_production(root, TASK_ID, candidate)
    assert _contained_before(root, TASK_ID, f'{candidate}^1') == 0, (
        "this is the arm's correct 'unrelated later merge, do not stamp' exit -- the "
        'branch was genuinely in main before $c, so no merge commit brought it in and '
        'stamping $c would fabricate landing evidence'
    )


def _build_train(root: Path, *, rebase_rewrites: bool) -> str:
    """Build a two-member merge train and return the tip's merge commit.

    ``task/M`` is the non-tip member, ``task/TIP`` the stacked tip that the
    lane actually merges.  The lane rebases the stack onto main before
    merging (``--onto main task/M~1 task/TIP``) and never advances the
    member's OWN ref -- ``merge-queue/SKILL.md`` rule 3's mechanism.

    *rebase_rewrites* selects the two shapes that differ in whether that
    rebase is a no-op: True forks ``task/M`` BEFORE main advances, so the
    replay writes new shas; False forks it from the current main, so the
    replay has nothing to move and the member's own commits survive
    verbatim into main.
    """
    new_repo(root)
    commit(root, 'base', 'base.txt')

    if not rebase_rewrites:
        # main advances FIRST, so the member is already atop main and the
        # pre-merge rebase has nothing to rewrite.
        commit(root, 'mainmoved', 'moved.txt')

    git(root, 'checkout', '-b', 'task/M')
    commit(root, 'mwork', 'm.txt')

    if rebase_rewrites:
        # main advances AFTER the member forked, so the rebase replays it.
        git(root, 'checkout', 'main')
        commit(root, 'mainmoved', 'moved.txt')

    git(root, 'checkout', '-b', 'task/TIP', 'task/M')
    commit(root, 'tipwork', 'tip.txt')
    git(root, 'rebase', '--onto', 'main', 'task/M~1', 'task/TIP')

    git(root, 'checkout', 'main')
    git(root, 'merge', '--no-ff', 'task/TIP', '-m', 'Merge task/TIP into main')
    return git(root, 'rev-parse', 'HEAD')


def test_the_ordinary_coalesce_absorbed_member_can_never_reach_the_rc0_arm(
    tmp_path: Path,
) -> None:
    """CASE A -- the pre-merge rebase rewrites the member's shas: rc=1, permanently.

    This is the shape ``skills/merge-queue/SKILL.md``'s "Follow the
    superseded successor" rule 3 describes, and the one
    ``skills/_shared/deriving-landed-sha.md`` step 3 names as an
    empty-marker cause.  Its work reaches main only as REWRITTEN commits
    under the tip's merge, and its own ref is never advanced to them, so
    the member never becomes an ancestor of main -- the ancestry check at
    the top of step 4 lands on rc=1 and the rc=0 arm is unreachable.  A
    reader arriving from step 3 therefore belongs on the rc=1 arm, which
    is why the rc=0 rationale must not claim this case.
    """
    root = tmp_path / 'repo'
    _build_train(root, rebase_rewrites=True)

    assert git_rc(root, 'merge-base', '--is-ancestor', 'task/M', 'main') == 1, (
        'the rebase rewrote the member\'s shas and its own ref was never advanced, so '
        'it cannot be an ancestor of main -- rc=1, and the rc=0 arm is unreachable'
    )


def test_the_no_op_rebase_member_is_the_shape_that_reaches_the_rc0_arm(
    tmp_path: Path,
) -> None:
    """CASE B -- the pre-merge rebase is a no-op, so the member IS contained.

    Identical train, except the member forked from the CURRENT main, so
    the replay moves nothing and its own commits survive verbatim into the
    tip's merge.  Now ancestry exits 0, the member reaches the rc=0 arm,
    and the arm works on it: the candidate is the tip's merge commit and
    ``contained-before`` exits 1 -- "this IS the merge that brought it in,
    stamp it".  This is the case the rc=0 rationale should name.
    """
    root = tmp_path / 'repo'
    tip_merge = _build_train(root, rebase_rewrites=False)

    assert git_rc(root, 'merge-base', '--is-ancestor', 'task/M', 'main') == 0, (
        'with a no-op rebase the member\'s own commits are genuinely contained in the '
        'group merge, so ancestry exits 0 and the rc=0 arm IS reached'
    )
    candidate = _candidate_topo(root, 'M')
    assert candidate == tip_merge, (
        "the rc=0 arm's candidate query must find the tip's merge commit -- the group "
        'merge that absorbed this member'
    )
    _assert_runbook_agrees_with_production(root, 'M', candidate)
    assert _contained_before(root, 'M', f'{tip_merge}^1') == 1, (
        'the member was not in main before the tip merge, so that merge IS what brought '
        'it in: the stampable "absorbed into group merge" arm'
    )
