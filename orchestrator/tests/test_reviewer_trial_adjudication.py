"""Tests for the reviewer_trial adjudication log (task 2495 / PRD D-6).

AdjudicationLog records frontier-proposed ground-truth labels per mined
diff, flags a documented spot-check subset for operator sign-off, and
persists as JSONL for auditability (consumed by mining.audit_corpus).
"""

from __future__ import annotations

import json
from pathlib import Path

from orchestrator.evals.reviewer_trial.adjudication import (
    AdjudicationEntry,
    AdjudicationLog,
)

from orchestrator.evals.reviewer_trial.corpus import GroundTruthIssue


def _make_issue(issue_id: str = 'frontier_1') -> GroundTruthIssue:
    return GroundTruthIssue(
        id=issue_id,
        location='src/foo.py:10',
        category='off_by_one',
        severity='blocking',
        description='Loop bound is off by one.',
        mutation_type='logic',
    )


class TestAdjudicationLogAppend:
    def test_append_returns_entry_with_expected_fields(self) -> None:
        log = AdjudicationLog()

        entry = log.append(
            diff_id='d1',
            frontier_proposal=[_make_issue()],
            in_spot_check_subset=True,
            frontier_model='opus',
        )

        assert isinstance(entry, AdjudicationEntry)
        assert entry.diff_id == 'd1'
        assert entry.frontier_model == 'opus'
        assert entry.in_spot_check_subset is True
        assert len(entry.frontier_proposal) == 1
        assert entry.frontier_proposal[0]['id'] == 'frontier_1'
        assert entry.frontier_proposal[0]['category'] == 'off_by_one'

    def test_appended_entry_is_stored_in_log(self) -> None:
        log = AdjudicationLog()

        log.append('d1', [_make_issue()], in_spot_check_subset=False)

        assert len(log.entries) == 1
        assert log.entries[0].diff_id == 'd1'

    def test_spot_check_status_defaults_to_pending_and_reviewer_to_none(self) -> None:
        log = AdjudicationLog()

        entry = log.append('d1', [_make_issue()], in_spot_check_subset=True)

        assert entry.spot_check_status == 'pending'
        assert entry.spot_check_reviewer is None

    def test_notes_default_to_empty_string(self) -> None:
        log = AdjudicationLog()

        entry = log.append('d1', [_make_issue()], in_spot_check_subset=False)

        assert entry.notes == ''


class TestAdjudicationLogSaveLoad:
    def test_save_writes_one_json_object_per_line(self, tmp_path: Path) -> None:
        log = AdjudicationLog()
        log.append('d1', [_make_issue()], in_spot_check_subset=True, frontier_model='opus')
        log.append('d2', [_make_issue()], in_spot_check_subset=False, frontier_model='opus')
        out = tmp_path / 'adjudication_log.jsonl'

        log.save(out)

        lines = out.read_text().strip().splitlines()
        assert len(lines) == 2
        first = json.loads(lines[0])
        assert first['diff_id'] == 'd1'
        second = json.loads(lines[1])
        assert second['diff_id'] == 'd2'

    def test_load_round_trips_entries(self, tmp_path: Path) -> None:
        log = AdjudicationLog()
        log.append(
            'd1', [_make_issue('frontier_1')], in_spot_check_subset=True,
            frontier_model='opus', notes='stratified spot check',
        )
        out = tmp_path / 'adjudication_log.jsonl'
        log.save(out)

        loaded = AdjudicationLog.load(out)

        assert len(loaded.entries) == 1
        entry = loaded.entries[0]
        assert entry.diff_id == 'd1'
        assert entry.frontier_model == 'opus'
        assert entry.in_spot_check_subset is True
        assert entry.spot_check_status == 'pending'
        assert entry.notes == 'stratified spot check'
        assert entry.frontier_proposal[0]['category'] == 'off_by_one'

    def test_load_missing_file_returns_empty_log(self, tmp_path: Path) -> None:
        loaded = AdjudicationLog.load(tmp_path / 'does_not_exist.jsonl')

        assert loaded.entries == []


class TestAdjudicationLogCoverage:
    """coverage(all_diff_ids) reports which diffs have entries, and any gaps."""

    def test_coverage_reports_covered_and_missing(self) -> None:
        log = AdjudicationLog()
        log.append('d1', [_make_issue()], in_spot_check_subset=False)
        log.append('d2', [_make_issue()], in_spot_check_subset=False)

        report = log.coverage(['d1', 'd2', 'd3'])

        assert report.covered == {'d1', 'd2'}
        assert report.missing == {'d3'}
        assert report.ok is False

    def test_coverage_ok_when_no_gaps(self) -> None:
        log = AdjudicationLog()
        log.append('d1', [_make_issue()], in_spot_check_subset=False)

        report = log.coverage(['d1'])

        assert report.missing == set()
        assert report.ok is True

    def test_coverage_over_empty_log(self) -> None:
        log = AdjudicationLog()

        report = log.coverage(['d1', 'd2'])

        assert report.covered == set()
        assert report.missing == {'d1', 'd2'}
        assert report.ok is False


class TestAdjudicationLogSpotCheckSubset:
    """spot_check_subset() returns the flagged subset, status defaulting to 'pending'."""

    def test_returns_only_flagged_entries(self) -> None:
        log = AdjudicationLog()
        log.append('d1', [_make_issue()], in_spot_check_subset=True)
        log.append('d2', [_make_issue()], in_spot_check_subset=False)
        log.append('d3', [_make_issue()], in_spot_check_subset=True)

        subset = log.spot_check_subset()

        assert {e.diff_id for e in subset} == {'d1', 'd3'}

    def test_subset_entries_default_to_pending_status(self) -> None:
        log = AdjudicationLog()
        log.append('d1', [_make_issue()], in_spot_check_subset=True)

        subset = log.spot_check_subset()

        assert all(e.spot_check_status == 'pending' for e in subset)

    def test_empty_log_returns_empty_subset(self) -> None:
        log = AdjudicationLog()

        assert log.spot_check_subset() == []
