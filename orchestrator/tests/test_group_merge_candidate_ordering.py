"""Root-cause-as-spec pins for the rc=0 group-merge candidate query (task 4926).

``skills/_shared/deriving-landed-sha.md``'s step 4, rc=0 arm, tells an
agent to find the merge that brought a landed branch into main with::

    c=$(git rev-list --ancestry-path --merges task/<TASK_ID>..main | tail -1)
    git merge-base --is-ancestor task/<TASK_ID> "$c^1"   # "contained-before"

and then reads ``contained-before rc=1`` as "``$c`` IS the merge that
brought it in -- stamp it" and rc=0 as "unrelated later merge, do not
stamp".  Two premises hide in that pair, and this module makes both
executable:

* **Ordering.** ``git rev-list`` orders by COMMIT DATE by default, not
  topologically, so ``tail -1`` is only the *oldest descendant merge*
  when dates happen to agree with topology.  With a skewed or rewritten
  committer date the bringing-in merge can sort out of position and
  ``tail -1`` returns a LATER, unrelated merge instead -- whose ``^1``
  containment then exits 0, silently skipping the true group merge and
  degrading the procedure to the citation path.  ``--topo-order`` is what
  makes the "oldest" claim hold.
* **First parent.** ``$c^1`` is "main just before that merge" only when
  the merge was created ON main.  A merge created from the branch side
  has the TASK tip as ``^1``, so the containment check exits 0 trivially
  and the arm falsely concludes "unrelated later merge, do not stamp".

Both of those are FALSE rc=0 verdicts, so a third pin covers the arm's
mirror image: the LEGITIMATE rc=0 exit, where a real candidate exists,
``contained-before`` correctly returns 0 because the branch was already in
main before it, and the agent must NOT stamp.  Without it the suite would
stay green if rc=0 stopped being reachable at all -- while the runbook
began licensing a fabricated merge sha, the very risk the citation gate
below the arm exists to close.

A fourth pin covers the same arm's *rationale*: which train-member shape
can actually REACH rc=0 at all (the no-op-rebase member), versus the
ordinary coalesce-absorbed non-tip member whose shas the pre-merge rebase
rewrites, which is permanently rc=1 -- ``skills/merge-queue/SKILL.md``
rule 3's mechanism.

These are characterization pins, EXPECTED TO PASS ON WRITE, exactly as
``orchestrator/tests/test_citation_gate_positive_arm.py`` (task 4924, same
shared doc) states of its own: they make the failure scenario executable
so a future change to git's behaviour re-litigates this fix's premises
instead of silently invalidating them.  A test failing here on write means
a git-behaviour premise was mis-transcribed, not that the doc edit is
pending.

A test that grepped ``skills/_shared/deriving-landed-sha.md`` or
``skills/*/SKILL.md`` for the corrected query or wording is deliberately
ABSENT.  Per ``orchestrator/tests/test_roles_ancestry_check.py``'s module
docstring, such a test "exercises no runtime behaviour; it only pins
prose, which pressures the wording toward whatever passes the assertion
rather than toward what is clearest to the reader", and it "would couple
this suite to prose in two skill docs that legitimately get rewritten" --
the same docs this task edits.  The real-git tests here are what actually
protect the change.

DELIBERATE DIVERGENCE, do not "restore" the agreement assertion.
``orchestrator/src/orchestrator/git_ops.py::GitOps.landing_merge_for``
runs the same query -- ``git rev-list --ancestry-path --merges --reverse
{head}..{upstream}`` then the first element -- and its docstring makes the
same "oldest merge commit on the ANCESTRY PATH" claim; ``--reverse`` only
reverses the DATE-ordered list, so it carries the ordering defect
identically.  It is orchestrator production code with a live caller
(``orchestrator/src/orchestrator/landing_evidence.py::branch_work_landed``'s
no-op guard) and is deliberately NOT fixed by task 4926, which is a
runbook-wording change; it is filed as a separate follow-up.  So unlike
``test_citation_gate_positive_arm.py``, this module must NOT assert that
the runbook form and the production method AGREE: after this task they
deliberately diverge (the doc gains ``--topo-order``, production does not)
until that follow-up lands.

Every synthetic repo carries the repo's standard git-isolation guards
(esc-3072-3): ``assert_isolated_git_repo`` pre-flights each fixture root
before any subprocess touches it, and ``git_env_with_ceiling`` caps git's
upward repo discovery at that root -- the same pairing
``orchestrator/tests/test_branch_work_landed.py`` and
``orchestrator/tests/test_git_ops_no_op_landing.py`` apply.

No sleeps, no network, no skips: if git is unavailable these fail loudly
rather than silently skipping (the repo's no-silent-fail-soft invariant).
"""

from __future__ import annotations

import subprocess
from pathlib import Path

from _orch_helpers import assert_isolated_git_repo, git_env_with_ceiling

TASK_ID = 'T'

_PINNED_DATE = '2020-01-01T00:00:00 +0000'

# Deterministic identity + no signing, so subjects and shas are stable and
# the tests never depend on the developer's git config.
_GIT_ENV_ARGS = [
    '-c', 'user.name=Test',
    '-c', 'user.email=test@test.com',
    '-c', 'commit.gpgsign=false',
]

# DUPLICATION, deliberately not extracted yet.  ``_GIT_ENV_ARGS``, ``_git``,
# ``_git_rc``, ``_commit`` and ``_new_repo`` are near-verbatim twins of the same
# five helpers in ``test_citation_gate_positive_arm.py`` (this copy generalises
# ``_git`` with ``date=`` and ``_commit`` with a per-branch filename).  Per-file
# ``_git`` helpers are the wider suite's norm, so two copies stay under the bar.
# If a THIRD real-git module for this ladder appears, lift these five -- in the
# generalised form below -- into a small shared fixture module beside
# ``_orch_helpers.py`` and have all three import it, rather than adding a third
# copy.


def _git_env(root: Path, date: str | None = None) -> dict[str, str]:
    """Subprocess environment for a git call in *root*.

    :func:`_orch_helpers.git_env_with_ceiling` caps git's upward repository
    discovery at *root* (incident esc-3072-3), so a fixture command can never
    retarget an ENCLOSING repository -- which is not hypothetical here, because
    pytest's basetemp can sit inside a live task worktree.  The complementary
    hazard -- an inherited ``GIT_DIR`` / ``GIT_WORK_TREE`` / ``GIT_INDEX_FILE``,
    which SKIPS discovery outright rather than walking, so no ceiling can bound
    it -- is stripped from ``os.environ`` for the duration of every test by
    ``df_pytest_isolation.py::_df_git_env_hermetic`` (autouse), which is why the
    ``os.environ`` copy this ceiling is built on is already clean.

    *date* pins both committer and author date, which is how the date-skew
    fixture below is built.
    """
    env = git_env_with_ceiling(root)
    if date is not None:
        env['GIT_COMMITTER_DATE'] = date
        env['GIT_AUTHOR_DATE'] = date
    return env


def _git(root: Path, *args: str, date: str | None = None) -> str:
    """Run git in *root* and return stripped stdout (raises on non-zero).

    *date* pins both committer and author date for the invocation, which is
    how the date-skew fixture below is built.
    """
    return subprocess.run(
        ['git', *_GIT_ENV_ARGS, *args],
        cwd=root, check=True, capture_output=True, text=True, env=_git_env(root, date),
    ).stdout.strip()


def _git_rc(root: Path, *args: str) -> int:
    """Run git in *root* and return only its exit code."""
    return subprocess.run(
        ['git', *_GIT_ENV_ARGS, *args],
        cwd=root, capture_output=True, text=True, env=_git_env(root),
    ).returncode


def _commit(root: Path, subject: str, filename: str, date: str | None = None) -> str:
    """Write *filename*, commit it with *subject*, and return the new sha.

    Each branch writes its OWN file so the sibling merges below never
    conflict -- a conflict would abort the fixture mid-build and the test
    would pin a history other than the one it describes.
    """
    (root / filename).write_text(f'{subject}\n')
    _git(root, 'add', filename)
    _git(root, 'commit', '-m', subject, date=date)
    return _git(root, 'rev-parse', 'HEAD')


def _new_repo(root: Path) -> None:
    """Create *root* as a repository root, then assert the isolation contract.

    :func:`_orch_helpers.assert_isolated_git_repo` is a PURE-FILESYSTEM
    pre-flight: it refuses a directory that merely sits inside some repo instead
    of being one.  It runs immediately after ``git init`` -- the earliest moment
    at which it can hold -- so every later fixture command in this module is
    covered by the pre-flight as well as by the ceiling in :func:`_git_env`.
    Asserting inside ``_new_repo`` rather than at each call site means a test
    added later cannot forget it.
    """
    root.mkdir()
    _git(root, 'init', '-b', 'main')
    assert_isolated_git_repo(root)


def _candidate_date_ordered(root: Path, tid: str) -> str:
    """The rc=0 arm's candidate query WITHOUT ``--topo-order``: date-ordered."""
    out = _git(root, 'rev-list', '--ancestry-path', '--merges', f'task/{tid}..main')
    return out.split('\n')[-1] if out else ''


def _candidate_topo(root: Path, tid: str) -> str:
    """The rc=0 arm's candidate query WITH ``--topo-order`` (the fix)."""
    out = _git(
        root, 'rev-list', '--topo-order', '--ancestry-path', '--merges', f'task/{tid}..main',
    )
    return out.split('\n')[-1] if out else ''


def _contained_before(root: Path, tid: str, rev: str) -> int:
    """The arm's ``contained-before`` rc for candidate expression *rev*."""
    return _git_rc(root, 'merge-base', '--is-ancestor', f'task/{tid}', rev)


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
    _new_repo(root)

    _commit(root, 'base', 'base.txt', date='2020-01-01T00:00:00 +0000')

    _git(root, 'checkout', '-b', f'task/{TASK_ID}')
    _commit(root, 'work', 'work.txt', date='2020-01-02T00:00:00 +0000')

    # X: the merge that brings task/T in -- committer date skewed NEW
    # (01-10), later than both sibling branch commits below.
    _git(root, 'checkout', 'main')
    _git(
        root, 'merge', '--no-ff', f'task/{TASK_ID}', '-m', f'Merge task/{TASK_ID} into main',
        date='2020-01-10T00:00:00 +0000',
    )
    x = _git(root, 'rev-parse', 'HEAD')

    # f1 and f2 BOTH fork from X, so X has two children in the walk.
    _git(root, 'checkout', '-b', 'f1', x)
    _commit(root, 'f1work', 'f1.txt', date='2020-01-03T00:00:00 +0000')
    _git(root, 'checkout', '-b', 'f2', x)
    _commit(root, 'f2work', 'f2.txt', date='2020-01-05T00:00:00 +0000')

    _git(root, 'checkout', 'main')
    _git(root, 'merge', '--no-ff', 'f1', '-m', 'Merge f1 into main',
         date='2020-01-04T00:00:00 +0000')
    mf1 = _git(root, 'rev-parse', 'HEAD')
    _git(root, 'merge', '--no-ff', 'f2', '-m', 'Merge f2 into main',
         date='2020-01-20T00:00:00 +0000')
    mf2 = _git(root, 'rev-parse', 'HEAD')

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
    assert _candidate_topo(root, TASK_ID) == x, (
        '--topo-order is what makes `tail -1` genuinely the oldest DESCENDANT merge'
    )

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
    _new_repo(root)

    _commit(root, 'base', 'base.txt', date=_PINNED_DATE)
    _git(root, 'checkout', '-b', f'task/{TASK_ID}')
    _commit(root, 'work', 'work.txt', date=_PINNED_DATE)

    _git(root, 'checkout', 'main')
    _git(
        root, 'merge', '--no-ff', f'task/{TASK_ID}', '-m', f'Merge task/{TASK_ID} into main',
        date=_PINNED_DATE,
    )
    x = _git(root, 'rev-parse', 'HEAD')

    _git(root, 'checkout', '-b', 'f1', x)
    _commit(root, 'f1work', 'f1.txt', date=_PINNED_DATE)
    _git(root, 'checkout', '-b', 'f2', x)
    _commit(root, 'f2work', 'f2.txt', date=_PINNED_DATE)

    _git(root, 'checkout', 'main')
    _git(root, 'merge', '--no-ff', 'f1', '-m', 'Merge f1 into main', date=_PINNED_DATE)
    mf1 = _git(root, 'rev-parse', 'HEAD')
    _commit(root, 'a', 'a.txt', date=_PINNED_DATE)
    _commit(root, 'b', 'b.txt', date=_PINNED_DATE)
    _git(root, 'merge', '--no-ff', 'f2', '-m', 'Merge f2 into main', date=_PINNED_DATE)

    committer_dates = set(_git(root, 'log', '--all', '--format=%ct').split('\n'))
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
    assert _candidate_topo(root, TASK_ID) == x, (
        '--topo-order returns the merge that brought the branch in, ties or not'
    )
    assert _contained_before(root, TASK_ID, f'{x}^1') == 1


def _build_branch_side_landing(root: Path) -> None:
    """Land ``task/T`` by a merge made on the branch side, fast-forwarded onto main."""
    _new_repo(root)
    _commit(root, 'base', 'base.txt', date=_PINNED_DATE)
    _git(root, 'checkout', '-b', f'task/{TASK_ID}')
    _commit(root, 'work', 'work.txt', date=_PINNED_DATE)
    _git(root, 'checkout', 'main')
    _commit(root, 'side', 'side.txt', date=_PINNED_DATE)

    _git(root, 'checkout', '-b', 'integ', f'task/{TASK_ID}')
    _git(root, 'merge', '--no-ff', 'main', '-m', 'Merge', date=_PINNED_DATE)
    _git(root, 'checkout', 'main')
    assert _git_rc(root, 'merge-base', '--is-ancestor', f'task/{TASK_ID}', 'main') == 1, (
        'task/T must not be on main yet, so the branch-side merge is what brings it in'
    )
    _git(root, 'merge', '--ff-only', 'integ')


def _build_fast_forward_then_unrelated_merge(root: Path) -> None:
    """Land ``task/T`` by fast-forward, then merge an unrelated branch on main."""
    _new_repo(root)
    _commit(root, 'base', 'base.txt', date=_PINNED_DATE)
    _git(root, 'checkout', '-b', f'task/{TASK_ID}')
    _commit(root, 'work', 'work.txt', date=_PINNED_DATE)
    _git(root, 'checkout', '-b', 'f1', 'main')
    _commit(root, 'side', 'side.txt', date=_PINNED_DATE)

    _git(root, 'checkout', 'main')
    _git(root, 'merge', '--ff-only', f'task/{TASK_ID}')
    assert _git(root, 'rev-parse', 'main') == _git(root, 'rev-parse', f'task/{TASK_ID}'), (
        'task/T must land with NO merge commit, so the later merge is unrelated to it'
    )
    _git(root, 'merge', '--no-ff', 'f1', '-m', 'Merge', date=_PINNED_DATE)


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
        assert _git(a, 'rev-parse', ref) == _git(b, 'rev-parse', ref), (
            f'the two stories must be one commit graph, sha for sha, at {ref} -- that is '
            'why no git query can tell them apart'
        )

    candidate = _candidate_topo(a, TASK_ID)
    assert candidate == _git(a, 'rev-parse', 'main')
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
    _new_repo(root)

    _commit(root, 'base', 'base.txt')
    _git(root, 'checkout', '-b', f'task/{TASK_ID}')
    _commit(root, 'work', 'work.txt')
    task_tip = _git(root, 'rev-parse', f'task/{TASK_ID}')

    _git(root, 'checkout', 'main')
    mainadv = _commit(root, 'mainadv', 'main.txt')
    _git(root, 'merge', '--no-ff', f'task/{TASK_ID}', '-m', f'Merge task/{TASK_ID} into main')
    merge = _git(root, 'rev-parse', 'HEAD')

    assert _git(root, 'rev-parse', f'{merge}^1') == mainadv, (
        'a merge created on main has main-just-before-the-merge as its first parent'
    )
    assert _git(root, 'rev-parse', f'{merge}^2') == task_tip

    candidate = _candidate_topo(root, TASK_ID)
    assert candidate == merge
    assert _contained_before(root, TASK_ID, f'{candidate}^1') == 1, (
        'merged from main, the documented ^1 check renders the correct "this IS the '
        'merge that brought it in" verdict -- so the ambiguity above is about unknown '
        'merge DIRECTION, not about the check itself'
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
    _new_repo(root)

    base = _commit(root, 'base', 'base.txt')
    _git(root, 'branch', f'task/{TASK_ID}')  # left AT base; never advanced

    _git(root, 'checkout', '-b', 'f1')
    _commit(root, 'f1work', 'f1.txt')
    _git(root, 'checkout', 'main')
    _git(root, 'merge', '--no-ff', 'f1', '-m', 'Merge f1 into main')
    mf1 = _git(root, 'rev-parse', 'HEAD')

    assert _git(root, 'rev-parse', f'task/{TASK_ID}') == base, (
        'the branch must still sit at its creation point -- that is what makes it '
        'already contained in main, which is the precondition for a legitimate rc=0'
    )
    assert _git(root, 'rev-parse', f'{mf1}^1') == base, (
        'the unrelated merge was created on main, so its first parent is '
        'main-just-before-it, which already contains the branch'
    )

    candidate = _candidate_topo(root, TASK_ID)
    assert candidate == mf1, (
        'the arm must reach its containment check with a REAL candidate here: rc=0 has '
        'to be decided by the check, not short-circuited by an empty $c'
    )
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
    _new_repo(root)
    _commit(root, 'base', 'base.txt')

    if not rebase_rewrites:
        # main advances FIRST, so the member is already atop main and the
        # pre-merge rebase has nothing to rewrite.
        _commit(root, 'mainmoved', 'moved.txt')

    _git(root, 'checkout', '-b', 'task/M')
    _commit(root, 'mwork', 'm.txt')

    if rebase_rewrites:
        # main advances AFTER the member forked, so the rebase replays it.
        _git(root, 'checkout', 'main')
        _commit(root, 'mainmoved', 'moved.txt')

    _git(root, 'checkout', '-b', 'task/TIP', 'task/M')
    _commit(root, 'tipwork', 'tip.txt')
    _git(root, 'rebase', '--onto', 'main', 'task/M~1', 'task/TIP')

    _git(root, 'checkout', 'main')
    _git(root, 'merge', '--no-ff', 'task/TIP', '-m', 'Merge task/TIP into main')
    return _git(root, 'rev-parse', 'HEAD')


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

    assert _git_rc(root, 'merge-base', '--is-ancestor', 'task/M', 'main') == 1, (
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

    assert _git_rc(root, 'merge-base', '--is-ancestor', 'task/M', 'main') == 0, (
        'with a no-op rebase the member\'s own commits are genuinely contained in the '
        'group merge, so ancestry exits 0 and the rc=0 arm IS reached'
    )
    assert _candidate_topo(root, 'M') == tip_merge, (
        "the rc=0 arm's candidate query must find the tip's merge commit -- the group "
        'merge that absorbed this member'
    )
    assert _contained_before(root, 'M', f'{tip_merge}^1') == 1, (
        'the member was not in main before the tip merge, so that merge IS what brought '
        'it in: the stampable "absorbed into group merge" arm'
    )
