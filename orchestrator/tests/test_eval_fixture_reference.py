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

# The three fixtures named by PRD §σ / the capability manifest. df_task_12 and
# df_task_13 also lack a reference but are deliberately NOT in this task's
# scope, so they are not listed here.
_BACKFILLED = ('reify_task_12', 'reify_task_27', 'df_task_18')


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


@pytest.mark.parametrize('name', _BACKFILLED)
def test_backfilled_fixture_has_reference_block(name: str) -> None:
    """The reference SHA is the fixture's own landed post_task_commit.

    Git-free and unconditional: this is the back-fill's real pin, and it must
    hold on every machine regardless of which checkouts are present.
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
    # A landed reference diff is never empty.
    assert diff_stat['files'] > 0
    assert diff_stat['insertions'] > 0


@pytest.mark.parametrize('name', _BACKFILLED)
def test_backfilled_fixture_reference_diff_materializes(name: str) -> None:
    """The back-filled SHAs produce a real, non-empty diff.

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
