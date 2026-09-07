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

A third pin covers the same arm's *rationale*: which train-member shape
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

No sleeps, no network, no skips: if git is unavailable these fail loudly
rather than silently skipping (the repo's no-silent-fail-soft invariant).
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

TASK_ID = 'T'

# Deterministic identity + no signing, so subjects and shas are stable and
# the tests never depend on the developer's git config.
_GIT_ENV_ARGS = [
    '-c', 'user.name=Test',
    '-c', 'user.email=test@test.com',
    '-c', 'commit.gpgsign=false',
]


def _git(root: Path, *args: str, date: str | None = None) -> str:
    """Run git in *root* and return stripped stdout (raises on non-zero).

    *date* pins both committer and author date for the invocation, which is
    how the date-skew fixture below is built.
    """
    env = None
    if date is not None:
        env = {**os.environ, 'GIT_COMMITTER_DATE': date, 'GIT_AUTHOR_DATE': date}
    return subprocess.run(
        ['git', *_GIT_ENV_ARGS, *args],
        cwd=root, check=True, capture_output=True, text=True, env=env,
    ).stdout.strip()


def _git_rc(root: Path, *args: str) -> int:
    """Run git in *root* and return only its exit code."""
    return subprocess.run(
        ['git', *_GIT_ENV_ARGS, *args], cwd=root, capture_output=True, text=True,
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
    root.mkdir()
    _git(root, 'init', '-b', 'main')


def _candidate_documented(root: Path, tid: str) -> str:
    """The rc=0 arm's candidate query WITHOUT ``--topo-order`` (the defect)."""
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


def test_date_skew_makes_the_documented_candidate_query_pick_the_wrong_merge(
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

    # (1) The currently-documented form returns f1's merge, NOT X.
    documented = _candidate_documented(root, TASK_ID)
    assert documented == mf1, (
        'expected the date-ordered walk to put an unrelated LATER merge last; if this '
        'stops holding, git changed its default ordering and '
        'skills/_shared/deriving-landed-sha.md step 4 must be re-litigated'
    )
    assert documented != x, (
        'the whole defect is that the documented query does NOT return the merge that '
        'brought the branch in'
    )

    # (2) ...so the arm takes the "unrelated later merge, do not stamp" exit,
    #     silently skipping the true group merge.
    assert _contained_before(root, TASK_ID, f'{documented}^1') == 0, (
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


def test_a_merge_created_from_the_branch_side_puts_the_task_tip_at_caret_1(
    tmp_path: Path,
) -> None:
    """``$c^1`` is main-just-before-the-merge only when the merge was made ON main.

    Build the merge from the TASK side: ``git checkout -b integ task/T``
    then ``git merge --no-ff main``, and fast-forward main onto it.  The
    merge lands on main and ``task/T``'s own ref never advances -- exactly
    the state the rc=0 arm inspects -- but its first parent is the task
    tip, not main.  The runbook's ``contained-before`` check then exits 0
    trivially and the arm reads "unrelated later merge, do not stamp",
    losing a merge sha it was holding.

    ``orchestrator/src/orchestrator/git_ops.py::GitOps.merge_to_main``
    always merges FROM main, so a dark-factory landing is safe; a target
    project reached through ``skills/orchestrate/SKILL.md``'s call site
    merges by its own convention, which is what makes this reachable.
    """
    root = tmp_path / 'repo'
    _new_repo(root)

    _commit(root, 'base', 'base.txt')
    _git(root, 'checkout', '-b', f'task/{TASK_ID}')
    _commit(root, 'work', 'work.txt')
    task_tip = _git(root, 'rev-parse', f'task/{TASK_ID}')

    # main advances, so the landing genuinely needs a merge commit.
    _git(root, 'checkout', 'main')
    mainadv = _commit(root, 'mainadv', 'main.txt')

    # The merge is created ON THE TASK SIDE, then fast-forwarded onto main.
    _git(root, 'checkout', '-b', 'integ', f'task/{TASK_ID}')
    _git(root, 'merge', '--no-ff', 'main', '-m', 'Merge main into integ')
    merge = _git(root, 'rev-parse', 'HEAD')
    _git(root, 'checkout', 'main')
    _git(root, 'merge', '--ff-only', 'integ')

    assert _git(root, 'rev-parse', 'main') == merge
    assert _git(root, 'rev-parse', f'task/{TASK_ID}') == task_tip, (
        "the task ref must still sit at its pre-merge tip -- that is the state the rc=0 "
        'arm inspects, and it is what makes the ^1 mix-up reachable'
    )

    # (1) The first parent is the TASK side, not main.
    assert _git(root, 'rev-parse', f'{merge}^1') == task_tip
    assert _git(root, 'rev-parse', f'{merge}^2') == mainadv

    candidate = _candidate_topo(root, TASK_ID)
    assert candidate == merge, 'the ladder must select this merge as the candidate'

    # (2) The runbook's check on ^1 gives the FALSE "do not stamp" verdict...
    assert _contained_before(root, TASK_ID, f'{candidate}^1') == 0, (
        'with the task tip at ^1 the containment check exits 0 trivially, which the '
        'rc=0 arm reads as "unrelated later merge, do not stamp" -- a false skip of a '
        'merge sha the agent was already holding'
    )

    # (3) ...while the main-side parent gives the correct one.
    assert _contained_before(root, TASK_ID, f'{candidate}^2') == 1, (
        'against the MAIN-side parent the branch is correctly not-yet-contained: this '
        'IS the merge that brought it in'
    )


def test_a_merge_created_on_main_puts_main_at_caret_1(tmp_path: Path) -> None:
    """The control for the pin above: merge direction is what decides ``^1``.

    Same history, merged the dark-factory way (``git merge --no-ff
    task/T`` from main).  Here ``^1`` IS main-just-before-the-merge and the
    containment check renders the correct verdict, so the previous test
    pins a direction-dependent hazard rather than a universal one.
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
        'merge that brought it in" verdict -- so the hazard above is about merge '
        'DIRECTION, not about the check itself'
    )
