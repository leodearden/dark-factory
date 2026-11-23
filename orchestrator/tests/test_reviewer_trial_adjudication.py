"""Tests for the reviewer_trial adjudication log (task 2495 / PRD D-6).

AdjudicationLog records frontier-proposed ground-truth labels per mined
diff, flags a documented spot-check subset for operator sign-off, and
persists as JSONL for auditability (consumed by mining.audit_corpus).
"""

from __future__ import annotations

import json
from collections.abc import Collection
from pathlib import Path

from orchestrator.evals.reviewer_trial.adjudication import (
    AdjudicationEntry,
    AdjudicationLog,
    select_spot_check_subset,
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


class TestSelectSpotCheckSubset:
    """``select_spot_check_subset`` -- deterministic, evenly-spread
    spot-check sampling with a floor and sticky (grow-only) membership."""

    def test_empty_input_returns_empty_set(self) -> None:
        assert select_spot_check_subset([]) == set()

    def test_below_minimum_floor_selects_everything(self) -> None:
        # n=3 < minimum=5 -> k is capped at n (not padded past the pool size).
        diff_ids = ['a', 'b', 'c']
        result = select_spot_check_subset(diff_ids, minimum=5)
        assert result == {'a', 'b', 'c'}

    def test_fraction_rounds_to_target_size(self) -> None:
        diff_ids = [f'd{i:03d}' for i in range(100)]
        result = select_spot_check_subset(diff_ids, fraction=0.1, minimum=5)
        assert len(result) == 10

    def test_deterministic_across_calls_and_input_order(self) -> None:
        diff_ids = [f'd{i:03d}' for i in range(37)]
        first = select_spot_check_subset(diff_ids)
        reordered = select_spot_check_subset(list(reversed(diff_ids)))
        again = select_spot_check_subset(diff_ids)
        assert first == reordered == again

    def test_sticky_ids_survive_even_when_sampling_would_drop_them(self) -> None:
        """A diff flagged in a smaller corpus must stay flagged after the
        corpus grows, even though the deterministic even-spread sample over
        the larger id set no longer lands on it -- this is the bug the
        `sticky` param fixes: prior human sign-off silently disappearing on
        a later `mine` resave."""
        small = [f'd{i:02d}' for i in range(10)]
        prior_subset = select_spot_check_subset(small)
        assert prior_subset == {'d00', 'd02', 'd04', 'd06', 'd08'}

        grown = [f'd{i:02d}' for i in range(40)]
        naive = select_spot_check_subset(grown)  # no sticky: simulates the pre-fix bug
        assert not prior_subset <= naive, 'test premise: naive resample must drop some prior ids'

        result = select_spot_check_subset(grown, sticky=prior_subset)
        assert prior_subset <= result, 'sticky ids must never be dropped'
        assert result - prior_subset, 'new ids should still be sampled to grow the subset'

    def test_sticky_ids_no_longer_in_corpus_are_dropped(self) -> None:
        """A sticky id that no longer exists in diff_ids (e.g. removed from
        the corpus) must not leak into the result."""
        diff_ids = ['a', 'b', 'c']
        result = select_spot_check_subset(diff_ids, sticky={'zzz-not-present'})
        assert 'zzz-not-present' not in result


def _log_over(diff_ids: list[str], flagged: Collection[str] = ()) -> AdjudicationLog:
    log = AdjudicationLog()
    for diff_id in diff_ids:
        log.append(diff_id, [_make_issue()], in_spot_check_subset=diff_id in flagged)
    return log


def _flagged(log: AdjudicationLog) -> set[str]:
    return {e.diff_id for e in log.spot_check_subset()}


class TestGrowSpotCheckSubset:
    """grow_spot_check_subset re-samples over *diff_ids* without dropping a flag or a human's verdict."""

    _GROWN = [f'd{i:02d}' for i in range(40)]

    def test_a_flag_from_a_smaller_sample_survives_growth(self) -> None:
        prior = select_spot_check_subset(self._GROWN[:10])
        assert prior - select_spot_check_subset(self._GROWN), (
            'test premise: the plain 40-id sample drops some prior flags'
        )
        log = _log_over(self._GROWN, flagged=prior)

        log.grow_spot_check_subset(self._GROWN)

        assert prior <= _flagged(log)

    def test_an_entry_a_human_acted_on_becomes_flagged(self) -> None:
        log = _log_over(self._GROWN)
        unsampled = select_spot_check_subset(self._GROWN)
        acted = next(e for e in log.entries if e.diff_id not in unsampled)
        acted.spot_check_status = 'confirmed'

        log.grow_spot_check_subset(self._GROWN)

        assert acted.in_spot_check_subset

    def test_the_flagged_set_is_the_sample_kept_sticky(self) -> None:
        prior = select_spot_check_subset(self._GROWN[:10])
        log = _log_over(self._GROWN, flagged=prior)

        log.grow_spot_check_subset(self._GROWN)

        assert _flagged(log) == select_spot_check_subset(self._GROWN, sticky=prior)
        assert _flagged(log) - prior, 'new ids should be sampled to grow the subset'

    def test_an_entry_outside_diff_ids_keeps_its_flag(self) -> None:
        log = _log_over([*self._GROWN, 'hand_1', 'hand_2'], flagged={'hand_1'})

        log.grow_spot_check_subset(self._GROWN)

        flags = {e.diff_id: e.in_spot_check_subset for e in log.entries}
        assert (flags['hand_1'], flags['hand_2']) == (True, False)

    def test_no_diff_ids_changes_nothing(self) -> None:
        log = _log_over(self._GROWN, flagged={'d00', 'd17'})

        log.grow_spot_check_subset([])

        assert _flagged(log) == {'d00', 'd17'}
