"""A branch-swapping quarantine ends the review record's lineage (task 3663).

A quarantine renames ``task/<id>`` aside and the next dispatch cuts a fresh
branch from main, while every review record survives in the task root. These
tests pin the retirement that runs at the next dispatch, using the filesystem
only, with no git and no mocks.
"""
from __future__ import annotations

import json
import shutil
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from orchestrator import review_lineage
from orchestrator.artifacts import TaskArtifacts
from orchestrator.quarantine_provenance import stamp_quarantined_from
from orchestrator.review_lineage import retire_quarantined_lineages
from orchestrator.review_suggestions.prior_round import load_prior_round_suggestions

_PLAN = {
    'task_id': '42',
    'title': 'T',
    'steps': [
        {'id': 'step-1', 'type': 'test', 'description': 'red', 'status': 'done', 'commit': 'abc'},
        {'id': 'step-2', 'type': 'impl', 'description': 'green', 'status': 'pending'},
    ],
}
_PARKING_NAME = '42-20261112T000000Z'
_SUGGESTION = {
    'severity': 'suggestion',
    'location': 'src/foo/bar.py:42',
    'category': 'correctness',
    'description': 'the retry loop swallows the timeout',
    'suggested_fix': 'reraise TimeoutError after the last attempt',
}
_BLOCKING = {
    'severity': 'blocking',
    'location': 'src/foo/baz.py:7',
    'category': 'correctness',
    'description': 'the lock is never released',
    'suggested_fix': 'use a context manager',
}


@pytest.fixture()
def lane(tmp_path: Path) -> Path:
    worktree = tmp_path / 'worktrees' / '_lane-0'
    worktree.mkdir(parents=True)
    return worktree


@pytest.fixture()
def artifacts(lane: Path) -> TaskArtifacts:
    ta = TaskArtifacts.for_task(lane, '42')
    ta.init('42', 'T', 'd')
    ta.write_plan(json.loads(json.dumps(_PLAN)))
    ta.ensure_lane_symlinks()
    return ta


def _stamp(**overrides: object) -> dict[str, object]:
    stamp: dict[str, object] = {
        'branch': f'refs/heads/task/{_PARKING_NAME}',
        'tip_sha': 'abc123',
        'unique_commit_count': 2,
        'quarantined_at': datetime.now(UTC).isoformat(),
        'reason': 'unit-test',
        'parking': f'/q/{_PARKING_NAME}',
        'stored_title': None,
    }
    stamp.update(overrides)
    return stamp


def _write_amend_round(ta: TaskArtifacts, label: str = 'amend-1', issues: list[dict] | None = None) -> Path:
    archive = ta.review_archive_dir(label)
    archive.mkdir(parents=True)
    review = {'verdict': 'ISSUES_FOUND', 'issues': issues if issues is not None else [_SUGGESTION, _BLOCKING]}
    (archive / 'reviewer_comprehensive.json').write_text(json.dumps(review))
    return archive


def _seed_parked_lineage(ta: TaskArtifacts) -> None:
    ta.set_review_counters(amendment_rounds_total=1, review_cycles_total=1)
    _write_amend_round(ta)
    cycle = ta.review_archive_dir('cycle-1')
    cycle.mkdir(parents=True)
    (cycle / 'x.json').write_text('{}')


def _top_level_archives(ta: TaskArtifacts) -> list[str]:
    return sorted(
        p.name for p in ta.review_archives_root.iterdir()
        if p.is_dir() and p.name.startswith(('reviews-amend-', 'reviews-cycle-'))
    )


@pytest.fixture()
def parked(artifacts: TaskArtifacts) -> dict[str, object]:
    _seed_parked_lineage(artifacts)
    stamp = _stamp()
    assert stamp_quarantined_from(artifacts, stamp)
    return stamp


class TestRetirement:
    def test_the_counters_are_zeroed(self, artifacts: TaskArtifacts, parked):
        retire_quarantined_lineages(artifacts)

        assert artifacts.get_amendment_rounds_total() == 0
        assert artifacts.get_review_cycles_total() == 0

    def test_every_round_archive_moves_under_retired_reviews(self, artifacts: TaskArtifacts, parked):
        retire_quarantined_lineages(artifacts)

        assert _top_level_archives(artifacts) == []
        retired = artifacts.artifact_path(f'retired-reviews/{_PARKING_NAME}')
        assert (retired / 'reviews-amend-1' / 'reviewer_comprehensive.json').is_file()
        assert (retired / 'reviews-cycle-1' / 'x.json').is_file()

    def test_task_2523_no_longer_sees_the_parked_lineage(self, artifacts: TaskArtifacts, parked):
        assert load_prior_round_suggestions(artifacts.review_archives_root) != []

        retire_quarantined_lineages(artifacts)

        assert load_prior_round_suggestions(artifacts.review_archives_root) == []

    def test_the_returned_record_describes_the_parked_lineage(self, artifacts: TaskArtifacts, parked):
        (retired,) = retire_quarantined_lineages(artifacts)

        assert retired.branch == parked['branch']
        assert retired.tip_sha == parked['tip_sha']
        assert retired.parking == parked['parking']
        assert retired.quarantined_at == parked['quarantined_at']
        assert retired.amendment_rounds_total == 1
        assert retired.review_cycles_total == 1
        assert retired.archive == artifacts.artifact_path(f'retired-reviews/{_PARKING_NAME}')
        descriptions = [s['description'] for s in retired.suggestions]
        assert descriptions == [_SUGGESTION['description']]
        assert _BLOCKING['description'] not in descriptions

    def test_a_verdict_survives_retirement(self, artifacts: TaskArtifacts, parked):
        artifacts.record_review_verdict('tree-abc', 'PASS', suggestions_routed=False, reviewer_fingerprint='fp')

        retire_quarantined_lineages(artifacts)

        cached = artifacts.get_cached_verdict('tree-abc')
        assert cached is not None
        assert cached['verdict'] == 'PASS'


class TestAcknowledgement:
    def test_an_acknowledged_stamp_leaves_the_new_lineage_alone(self, artifacts: TaskArtifacts, parked):
        retire_quarantined_lineages(artifacts)
        artifacts.set_review_counters(amendment_rounds_total=1)
        _write_amend_round(artifacts)

        assert retire_quarantined_lineages(artifacts) == ()

        assert artifacts.get_amendment_rounds_total() == 1
        assert _top_level_archives(artifacts) == ['reviews-amend-1']

    def test_a_stamp_without_a_branch_retires_nothing(self, artifacts: TaskArtifacts):
        _seed_parked_lineage(artifacts)
        assert stamp_quarantined_from(artifacts, _stamp(branch=None, tip_sha=None, unique_commit_count=None))

        first = retire_quarantined_lineages(artifacts)

        assert all(r.archive is None and r.amendment_rounds_total == 0 for r in first)
        assert artifacts.get_amendment_rounds_total() == 1
        assert artifacts.get_review_cycles_total() == 1
        assert _top_level_archives(artifacts) == ['reviews-amend-1', 'reviews-cycle-1']
        assert retire_quarantined_lineages(artifacts) == ()
        assert artifacts.get_amendment_rounds_total() == 1
        assert _top_level_archives(artifacts) == ['reviews-amend-1', 'reviews-cycle-1']

    def test_no_stamp_is_a_no_op(self, artifacts: TaskArtifacts):
        _seed_parked_lineage(artifacts)

        assert retire_quarantined_lineages(artifacts) == ()

        assert artifacts.get_amendment_rounds_total() == 1
        assert artifacts.get_review_cycles_total() == 1
        assert _top_level_archives(artifacts) == ['reviews-amend-1', 'reviews-cycle-1']

    def test_two_stamps_attribute_the_counters_to_the_first(self, artifacts: TaskArtifacts):
        _seed_parked_lineage(artifacts)
        first_parking = f'/q/{_PARKING_NAME}'
        second_parking = '/q/42-20261113T000000Z'
        assert stamp_quarantined_from(artifacts, _stamp(parking=first_parking))
        assert stamp_quarantined_from(artifacts, _stamp(parking=second_parking))

        first, second = retire_quarantined_lineages(artifacts)

        assert (first.parking, first.amendment_rounds_total, first.review_cycles_total) == (first_parking, 1, 1)
        assert (second.parking, second.amendment_rounds_total) == (second_parking, 0)
        assert artifacts.get_amendment_rounds_total() == 0
        assert retire_quarantined_lineages(artifacts) == ()


class TestRetiredReviewsSince:
    @pytest.fixture()
    def retired(self, artifacts: TaskArtifacts, parked) -> dict[str, object]:
        assert retire_quarantined_lineages(artifacts)
        return parked

    def test_no_since_returns_the_retired_lineage_with_its_suggestions(self, artifacts: TaskArtifacts, retired):
        (review,) = review_lineage.retired_reviews_since(artifacts, None)

        assert review.branch == retired['branch']
        assert review.archive == artifacts.artifact_path(f'retired-reviews/{_PARKING_NAME}')
        (suggestion,) = review.suggestions
        assert suggestion['description'] == _SUGGESTION['description']
        assert suggestion['suggested_fix'] == _SUGGESTION['suggested_fix']

    def test_a_plan_confirmed_after_the_quarantine_has_nothing_pending(self, artifacts: TaskArtifacts, retired):
        after = datetime.fromisoformat(str(retired['quarantined_at'])) + timedelta(minutes=1)

        assert review_lineage.retired_reviews_since(artifacts, after) == ()

    def test_a_plan_confirmed_before_the_quarantine_still_has_it_pending(self, artifacts: TaskArtifacts, retired):
        before = datetime.fromisoformat(str(retired['quarantined_at'])) - timedelta(minutes=1)

        (review,) = review_lineage.retired_reviews_since(artifacts, before)

        assert review.parking == retired['parking']

    def test_an_amend_round_of_only_blocking_issues_carries_nothing(self, artifacts: TaskArtifacts):
        artifacts.set_review_counters(amendment_rounds_total=1)
        _write_amend_round(artifacts, issues=[_BLOCKING])
        assert stamp_quarantined_from(artifacts, _stamp())
        assert retire_quarantined_lineages(artifacts)

        assert review_lineage.retired_reviews_since(artifacts, None) == ()

    def test_a_stamp_without_a_branch_carries_nothing(self, artifacts: TaskArtifacts):
        _seed_parked_lineage(artifacts)
        assert stamp_quarantined_from(artifacts, _stamp(branch=None, tip_sha=None, unique_commit_count=None))
        assert retire_quarantined_lineages(artifacts)

        assert review_lineage.retired_reviews_since(artifacts, None) == ()

    def test_a_deleted_retired_archive_carries_nothing(self, artifacts: TaskArtifacts, retired):
        shutil.rmtree(artifacts.artifact_path('retired-reviews'))

        assert review_lineage.retired_reviews_since(artifacts, None) == ()
