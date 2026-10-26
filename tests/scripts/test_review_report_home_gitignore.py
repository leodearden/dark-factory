"""The /review report home is tracked; only its pre-contract files stay ignored (task 6785).

The invariant is ``docs/quality-findings-contract.md`` §5: ``review/reports/``
is tracked and no /review run record there is ever gitignored. The chosen
answer for the retired three-file format's leftovers is
``skills/review/SKILL.md`` "Report home": those files are not reports, and the
root ``.gitignore`` keeps exactly their two globs ignored.
"""
from __future__ import annotations

from pathlib import Path

import pytest
from git_listing import git, run_git

REPO_ROOT = Path(__file__).parents[2]

REPORT_HOME = 'review/reports/'
FUTURE_RUN_RECORDS: tuple[str, ...] = (
    'review/reports/review-dark_factory-20991231.json',
    'review/reports/review-dark_factory-20991231.md',
    'review/reports/review-dark_factory-20991231-2.json',
    'review/reports/review-dark_factory-20991231-2.md',
    'review/reports/review-cockpit-20991231.json',
    'review/reports/review-cockpit-20991231.md',
)
PRE_CONTRACT_FILES: tuple[str, ...] = (
    'review/reports/summary-20260526T103500.md',
    'review/reports/phase1-20260526T103500.json',
    'review/reports/phase2-20260526T103500.json',
)


def _run_record_paths() -> list[str]:
    listing = git(REPO_ROOT, 'ls-files', '-z', '--', REPORT_HOME)
    tracked = [entry for entry in listing.split('\0') if entry]
    return [*tracked, *FUTURE_RUN_RECORDS]


RUN_RECORD_PATHS = _run_record_paths()


def _ignoring_rule(path: str) -> str | None:
    """The committed rule ``<source>:<line>:<pattern>`` ignoring *path* by pattern alone, else None.

    ``--no-index`` is what makes a tracked path answer by pattern: without it
    check-ignore calls every tracked file unignored. The verdict comes from
    ``-q`` because ``-v`` exits 0 on a negated ``!`` match. A rule whose source
    git prints as an absolute path (``.git/info/exclude``, ``core.excludesFile``)
    is machine-local, not the repo's.
    """
    verdict = run_git(REPO_ROOT, 'check-ignore', '--no-index', '-q', '--', path)
    if verdict.returncode == 1:
        return None
    assert verdict.returncode == 0, (
        f'git check-ignore on {path!r} failed with rc={verdict.returncode}; '
        f'stderr: {verdict.stderr!r}'
    )
    match = git(REPO_ROOT, 'check-ignore', '--no-index', '-v', '--', path)
    rule = match.partition('\t')[0]
    source = rule.split(':', 1)[0]
    return None if Path(source).is_absolute() else rule


@pytest.mark.parametrize('path', RUN_RECORD_PATHS, ids=RUN_RECORD_PATHS)
def test_no_committed_ignore_rule_reaches_a_run_record(path: str) -> None:
    rule = _ignoring_rule(path)
    assert rule is None, (
        f'{path} is ignored by {rule}. /review commits each run record with '
        f'`git add`, which refuses an ignored path, so the next report would stay '
        f'uncommitted and invisible to the next run\'s `since` (contract §5).'
    )


@pytest.mark.parametrize('path', PRE_CONTRACT_FILES, ids=PRE_CONTRACT_FILES)
def test_pre_contract_review_files_stay_ignored(path: str) -> None:
    rule = _ignoring_rule(path)
    assert rule is not None and rule.startswith('.gitignore:'), (
        f'{path} is no longer ignored by the root .gitignore (rule: {rule}). '
        f'skills/review/SKILL.md "Report home" and contract §5 keep these '
        f'pre-contract non-reports ignored; if they are deliberately un-ignored, '
        f'both change with it.'
    )
