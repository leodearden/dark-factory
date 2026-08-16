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

Two tests, deliberately split (see the plan's design decision):

* Test A is structural and ALWAYS runs — no git, no network, no LLM. It pins
  that the block exists and that its SHA is copied from the fixture's OWN
  top-level ``post_task_commit``, so an invented or typo'd SHA fails here.
  It is parametrized over the CORPUS — every fixture carrying a top-level
  ``post_task_commit`` — rather than over the three fixtures this task fixed,
  because a guard keyed on the already-fixed set cannot catch the regression
  it exists to prevent: a fixture authored tomorrow in the same defective
  shape would be judged on plausibility with the suite still green. The two
  fixtures that remain in that shape (``df_task_12``, ``df_task_13``) are
  outside this task's editable scope and are named in ``_EXEMPT_NO_REFERENCE``
  under a STRICT xfail, so the exemption is visible and shrinks loudly.
* Test B materializes the real diff through the production helper
  ``snapshots.get_diff_between_commits`` — the PRD's named user-observable
  signal — and SKIPS with an explicit reason when the fixture's checkout or a
  SHA is absent on this machine. Splitting keeps Test A's pin unconditional;
  a single combined test would be vacuously skipped wherever the reify
  checkout is missing.

Follows test_eval_diff_threading.py's convention (drive async entrypoints with
``asyncio.run``, no pytest-asyncio marker) and reuses test_eval_recovery.py's
``TASKS_DIR`` spelling for fixture discovery.
"""

from __future__ import annotations

import asyncio
import json
import subprocess
from pathlib import Path

import pytest

from orchestrator.evals import runner, snapshots

TASKS_DIR = Path(runner.__file__).parent / 'tasks'

# The three fixtures named by PRD §σ / the capability manifest — the ones this
# task back-filled. Retained only so the corpus scan below can prove it COVERS
# them; the guard itself is deliberately NOT keyed on this list (see
# ``_WITH_POST_TASK_COMMIT``).
_BACKFILLED = ('reify_task_12', 'reify_task_27', 'df_task_18')

# Fixtures that carry a top-level ``post_task_commit`` but no ``reference``
# block, and are NOT fixed here: they are outside this task's locked module set
# (only the three above are editable in this lane), so back-filling them is
# filed as follow-up work rather than smuggled in.
#
# Listed by name, and enforced with a STRICT xfail rather than skipped, so the
# exemption is both visible and shrinkable: the day either one is back-filled
# its case XPASSes and this suite fails until the name is removed. An exemption
# that can be forgotten is how the corpus grew these two in the first place.
_EXEMPT_NO_REFERENCE = ('df_task_12', 'df_task_13')


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
                    f'reference block and is outside task 3628 scope — '
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


def _commit_exists(project_root: Path, sha: str) -> bool:
    return subprocess.run(
        ['git', '-C', str(project_root), 'cat-file', '-e', f'{sha}^{{commit}}'],
        capture_output=True,
    ).returncode == 0


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
    """No silent growth of the exemption list.

    The xfail marks make a SHRINKING exemption loud (an XPASS fails the
    suite); this makes a GROWING one loud too — adding a name here that is
    not actually defective, to quiet a genuine failure, fails right here.
    """
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
    parallel subprocess reimplementation. Skips (loudly, naming the fixture and
    what is missing) when the checkout or a SHA is unavailable here; a silent
    skip would be the same class of defect this task removes.
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

    diff = asyncio.run(
        snapshots.get_diff_between_commits(project_root, pre, post)
    )

    assert diff, f'{name}: reference diff materialized EMPTY ({pre}..{post})'
    assert 'diff --git' in diff, (
        f'{name}: reference diff carries no git header — got {diff[:200]!r}'
    )
