"""The back-filled fixtures carry a landed `reference` block (eval-revival σ).

Guards task 3628's data half. Three fixtures — reify_task_12, reify_task_27,
df_task_18 — shipped with a top-level ``post_task_commit`` but NO ``reference``
block, and ``run_architect_eval`` reads the reference SHA only from
``task['reference']['post_task_commit']`` (runner.py step 6). Those three
therefore reached ``judge_plan_quality(plan, '', task)`` with an EMPTY
reference diff and were scored on plausibility rather than fidelity against
the landed change — half the v1 hard subset, discoverable only by archaeology
(``docs/plan-scoring-and-judge.md`` already cites the incoherent reify_task_12
cell).

Three tests, deliberately split (see the plan's design decision):

* Test A is structural and ALWAYS runs — no git, no network, no LLM. It pins
  that the block exists and that its SHA is copied from the fixture's OWN
  top-level ``post_task_commit``, so an invented or typo'd SHA fails here.
  It is parametrized over the CORPUS — every fixture carrying a top-level
  ``post_task_commit`` — rather than over the three fixtures this task fixed,
  because a guard keyed on the already-fixed set cannot catch the regression
  it exists to prevent: a fixture authored tomorrow in the same defective
  shape would be judged on plausibility with the suite still green. The two
  fixtures that once remained in that shape (``df_task_12``, ``df_task_13``)
  were back-filled by task 3828, so ``_EXEMPT_NO_REFERENCE`` is now EMPTY and
  the guard covers every fixture with a landed commit, no holdouts. The
  exemption machinery survives as the documented escape hatch: a name added
  there is held under a STRICT xfail, so the exemption is visible and shrinks
  loudly the day it is fixed. The emptiness is ASSERTED, not merely described,
  by ``test_the_exempt_set_is_exactly_the_known_defective_fixtures``.
* Test B materializes the real diff through the production helper
  ``snapshots.get_diff_between_commits`` — the PRD's named user-observable
  signal — and SKIPS with an explicit reason when the fixture's checkout or a
  SHA is absent on this machine. Splitting keeps Test A's pin unconditional;
  a single combined test would be vacuously skipped wherever the reify
  checkout is missing.
* Test C pins ACCURACY, which A and B leave open: both accept invented numbers,
  so it re-derives each declared ``diff_stat`` through the production
  ``task_sampler.capture_reference`` and asserts equality. It shares B's
  resolution and skip discipline through ``_resolve_or_skip`` — one definition,
  so a later change to that discipline cannot weaken only one of them.

Follows test_eval_diff_threading.py's convention (drive async entrypoints with
``asyncio.run``, no pytest-asyncio marker) and reuses test_eval_recovery.py's
``TASKS_DIR`` spelling for fixture discovery.
"""

from __future__ import annotations

import asyncio
import json
import subprocess
from functools import cache
from pathlib import Path

import pytest

from orchestrator.evals import runner, snapshots, task_sampler

TASKS_DIR = Path(runner.__file__).parent / 'tasks'

# The three fixtures named by PRD §σ / the capability manifest — the ones this
# task back-filled. Retained only so the corpus scan below can prove it COVERS
# them; the guard itself is deliberately NOT keyed on this list (see
# ``_WITH_POST_TASK_COMMIT``).
_BACKFILLED = ('reify_task_12', 'reify_task_27', 'df_task_18')

# Fixtures exempted from the corpus guard below — now EMPTY, and meant to stay
# that way. Task 3828 back-filled the last two holdouts (``df_task_12``,
# ``df_task_13``), so every fixture carrying a landed ``post_task_commit``
# declares a ``reference``.
#
# Kept rather than deleted because it is the cheapest way to silence a genuine
# corpus-guard failure, and therefore the thing that most needs a tripwire: a
# name added here must be a GENUINELY defective fixture, which
# ``test_the_exempt_set_is_exactly_the_known_defective_fixtures`` enforces —
# not a way to quiet the guard. The STRICT xfail catches the opposite mistake:
# a stale exemption XPASSes and fails this suite until the name is removed.
_EXEMPT_NO_REFERENCE: tuple[str, ...] = ()


def _fixture_names() -> list[str]:
    return sorted(p.stem for p in TASKS_DIR.glob('*.json'))


def _case(name: str):
    """One parametrize case, xfail-marked iff the fixture is a known exemption."""
    if name in _EXEMPT_NO_REFERENCE:
        return pytest.param(
            name,
            marks=pytest.mark.xfail(
                strict=True,
                reason=(
                    f'{name} carries a top-level post_task_commit with no '
                    f'reference block and is a KNOWN-defective holdout — '
                    f'back-fill it and delete it from _EXEMPT_NO_REFERENCE'
                ),
            ),
        )
    return name


def _raw(name: str) -> dict:
    """Load the fixture JSON RAW — deliberately not via ``runner.load_task``.

    ``load_task`` silently rewrites ``project_root`` to the discovered repo
    root when the hardcoded absolute path is missing (runner.py:97-98). Test B
    would then diff reify's SHAs against the dark-factory checkout and report a
    confusing "unknown revision" failure instead of an honest "reify checkout
    absent" skip.
    """
    with open(TASKS_DIR / f'{name}.json') as f:
        return json.load(f)


@cache
def _commit_exists(project_root: Path, sha: str) -> bool:
    # Cached: the two git-materializing tests below probe the SAME
    # (project_root, sha) pairs, so an uncached call costs 4 `git cat-file`
    # subprocesses per fixture where 2 suffice. A checkout cannot gain or lose
    # a commit mid-session, so the cache cannot go stale within a run.
    return subprocess.run(
        ['git', '-C', str(project_root), 'cat-file', '-e', f'{sha}^{{commit}}'],
        capture_output=True,
    ).returncode == 0


def _resolve_or_skip(name: str) -> tuple[Path, str, str]:
    """Resolve a fixture's ``(project_root, pre SHA, reference SHA)``, or SKIP.

    The single definition of this suite's skip discipline, shared by the two
    git-materializing tests below. They are parametrized over the same
    ``_WITH_REFERENCE`` population and had drifted into carrying byte-identical
    preambles, so a future change here — a shallow-clone guard, or treating a
    ``git`` failure as a skip rather than an error — applied to only one of
    them would silently weaken the other.

    SKIPS rather than fails, loudly and naming both the fixture and what is
    missing, when the checkout or either SHA is unavailable on this machine; a
    silent skip would be the same class of defect this suite exists to remove.
    """
    raw = _raw(name)
    project_root = Path(raw['project_root'])
    if not project_root.exists():
        pytest.skip(
            f'{name}: checkout {project_root} is absent on this machine'
        )

    pre = raw['pre_task_commit']
    post = raw['reference']['post_task_commit']
    for label, sha in (('pre_task_commit', pre), ('reference SHA', post)):
        if not _commit_exists(project_root, sha):
            pytest.skip(
                f'{name}: {label} {sha} does not resolve in {project_root}'
            )

    return project_root, pre, post


_WITH_POST_TASK_COMMIT = [
    _case(n) for n in _fixture_names() if _raw(n).get('post_task_commit')
]
_WITH_REFERENCE = [
    n for n in _fixture_names()
    if (_raw(n).get('reference') or {}).get('post_task_commit')
]


@pytest.mark.parametrize('name', _WITH_POST_TASK_COMMIT)
def test_fixture_with_a_landed_commit_carries_a_reference_block(
    name: str,
) -> None:
    """EVERY fixture that landed a commit declares it as its reference.

    Scanned from the corpus, NOT from a list of the fixtures already fixed
    (reviewer: test-coverage). A three-name allowlist can only re-check the
    fixtures someone already noticed; the regression σ exists to prevent is a
    NEW fixture authored tomorrow with a top-level ``post_task_commit`` and no
    ``reference`` block, which would be judged on plausibility with the suite
    still green. Keyed off ``post_task_commit`` because that is exactly the
    signal that a landed diff exists to grade against — a fixture with no
    landed commit is simply not in this guard's population.

    Git-free and unconditional: it must hold on every machine regardless of
    which checkouts are present.
    """
    raw = _raw(name)

    assert 'reference' in raw, (
        f'{name}.json has no `reference` block, so run_architect_eval would '
        f'judge it against an EMPTY reference diff'
    )
    reference = raw['reference']
    assert isinstance(reference, dict)

    # The back-fill introduces no new SHA — it copies the fixture's own
    # top-level field, so a typo'd or invented SHA fails right here.
    assert reference['post_task_commit'] == raw['post_task_commit']

    diff_stat = reference['diff_stat']
    assert isinstance(diff_stat, dict)
    for key in ('files', 'insertions', 'deletions'):
        assert isinstance(diff_stat[key], int), f'{name}: {key} must be an int'
    # A landed reference diff is never empty. Stated as insertions+deletions
    # rather than insertions alone: df_task_2605's landed change is a pure
    # deletion (1 file, 159 deletions), and it is no less a reference for it.
    assert diff_stat['files'] > 0
    assert diff_stat['insertions'] + diff_stat['deletions'] > 0


def test_the_backfilled_three_are_inside_the_corpus_guard() -> None:
    """The task's own three fixtures are covered by the scan, not exempted.

    Ties the PRD's named set to the inverted guard above: if a rename or a
    move ever took one of them out of ``TASKS_DIR``, the guard would quietly
    stop covering it and every parametrized case would still pass.
    """
    scanned = set(_fixture_names())
    assert set(_BACKFILLED) <= scanned
    assert not set(_BACKFILLED) & set(_EXEMPT_NO_REFERENCE)
    # Each is in the guard's population — i.e. each really did land a commit.
    for name in _BACKFILLED:
        assert _raw(name).get('post_task_commit'), name


def test_the_exempt_set_is_exactly_the_known_defective_fixtures() -> None:
    """No silent growth of the exemption list — which is currently EMPTY.

    The xfail marks make a SHRINKING exemption loud (an XPASS fails the
    suite); this makes a GROWING one loud too — adding a name here that is
    not actually defective, to quiet a genuine failure, fails right here.

    The equality assertion pins the state task 3828 reached: ZERO holdouts,
    every fixture with a landed commit subject to the corpus guard. Without it
    the loop below iterates an empty tuple and this test asserts nothing at
    all, so the very invariant the back-fill established would be the one thing
    unguarded. It is deliberately a speed bump, not a ban: a genuinely
    defective fixture may still be exempted, but doing so must edit this line
    and leave a written justification, rather than appending a name in silence.
    """
    assert _EXEMPT_NO_REFERENCE == (), (
        'the corpus guard covers every fixture with a landed post_task_commit; '
        'adding a name here needs a written justification, and the fixture '
        'must be genuinely defective (checked below)'
    )

    for name in _EXEMPT_NO_REFERENCE:
        raw = _raw(name)
        assert raw.get('post_task_commit'), (
            f'{name} is exempted from a guard it is not even subject to'
        )
        assert 'reference' not in raw, (
            f'{name} now carries a reference block — delete it from '
            f'_EXEMPT_NO_REFERENCE (the xfail above is already failing)'
        )


@pytest.mark.parametrize('name', _WITH_REFERENCE)
def test_backfilled_fixture_reference_diff_materializes(name: str) -> None:
    """Every declared reference produces a real, non-empty diff.

    Scanned over the fixtures that DO carry a reference, so a SHA pair that
    resolves but diffs to nothing — a reference block that is present and
    useless — fails here rather than being judged blind at run time.

    Drives the PRODUCTION helper ``snapshots.get_diff_between_commits`` — the
    same call run_architect_eval makes at runner.py step 6 — rather than a
    parallel subprocess reimplementation. Resolution and the skip discipline
    live in ``_resolve_or_skip``, shared with the accuracy guard below.
    """
    project_root, pre, post = _resolve_or_skip(name)

    diff = asyncio.run(
        snapshots.get_diff_between_commits(project_root, pre, post)
    )

    assert diff, f'{name}: reference diff materialized EMPTY ({pre}..{post})'
    assert 'diff --git' in diff, (
        f'{name}: reference diff carries no git header — got {diff[:200]!r}'
    )


@pytest.mark.parametrize('name', _WITH_REFERENCE)
def test_declared_diff_stat_matches_the_landed_diff(name: str) -> None:
    """Every declared ``diff_stat`` is the stat the landed diff actually has.

    The two guards above both accept INVENTED numbers: the structural one
    asserts only that the three values are ints with ``files > 0`` and
    ``insertions + deletions > 0``, and the materializing one asserts only that
    the diff is non-empty — so ``{"files": 99, "insertions": 99,
    "deletions": 99}`` sails through both. Nothing pinned a declared stat to the
    change it claims to describe, and for a HAND-back-filled fixture the
    accuracy of that triple is the entire deliverable (task 3828 measured two of
    them by hand; "do not invent numbers" was until now unenforceable).

    Re-derives the stat through ``task_sampler.capture_reference`` — the same
    helper ``task_sampler.build_fixture_record`` uses to AUTHOR this block for
    sampler-generated fixtures — rather than reimplementing shortstat parsing
    test-side, so there stays exactly one definition of what a diff_stat means.
    Shares ``_resolve_or_skip`` with the test above, so both guards observe one
    skip discipline on machines where a checkout or a SHA is absent.
    """
    project_root, pre, post = _resolve_or_skip(name)

    captured = asyncio.run(
        task_sampler.capture_reference(project_root, pre, post)
    )

    declared = _raw(name)['reference']['diff_stat']
    assert declared == {
        'files': captured.files,
        'insertions': captured.insertions,
        'deletions': captured.deletions,
    }, (
        f'{name}: declared diff_stat does not match the real diff at '
        f'{pre}..{post} in {project_root}'
    )
