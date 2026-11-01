"""Tests for the reviewer_trial answer-key pooling log and fold."""

from __future__ import annotations

import json
import re
from collections.abc import MutableMapping, MutableSequence
from pathlib import Path
from typing import Any

import pytest

from orchestrator.evals.reviewer_trial.corpus import (
    CorpusDiff,
    CorpusManifest,
    GroundTruthIssue,
)
from orchestrator.evals.reviewer_trial.key_pooling import (
    PoolingLog,
    PoolingRecord,
    audit_pooling,
    fold_pooled_issues,
    is_pooled_issue,
)


def _record(diff_id: str, issue_id: str, **provenance: Any) -> PoolingRecord:
    return PoolingRecord(
        diff_id=diff_id,
        issue_id=issue_id,
        pooling_date='2026-10-22',
        pooling_run='plans/pooling-run',
        provenance=provenance,
    )


class TestPoolingLog:
    def test_save_writes_one_record_per_line_and_load_round_trips(
        self, tmp_path: Path,
    ) -> None:
        rec_a = _record('d1', 'd1_pool_01', confidence='high')
        rec_b = _record('d2', 'd2_pool_01', confidence='medium')
        path = tmp_path / 'pooling_log.jsonl'

        PoolingLog(records=(rec_a, rec_b)).save(path)

        lines = path.read_text().splitlines()
        assert len(lines) == 2
        for line in lines:
            assert set(json.loads(line)) == {
                'diff_id', 'issue_id', 'pooling_date', 'pooling_run', 'provenance',
            }
        assert PoolingLog.load(path) == PoolingLog(records=(rec_a, rec_b))

    def test_load_of_missing_path_is_empty(self, tmp_path: Path) -> None:
        log = PoolingLog.load(tmp_path / 'absent.jsonl')

        assert log.records == ()

    def test_provenance_round_trips_nested_values_verbatim(
        self, tmp_path: Path,
    ) -> None:
        rec = _record(
            'd1', 'd1_pool_01',
            finding_ids=['P3-f2', 'P7-f1'],
            real='yes',
            adjudicated_severity='blocking',
            skeptic_corrections=[],
            corpus_severity='suggestion',
            quality_block='yes',
            confidence='high',
            evidence='line 12 drops the await',
        )
        path = tmp_path / 'pooling_log.jsonl'

        PoolingLog(records=(rec,)).save(path)

        assert PoolingLog.load(path).records[0].provenance == rec.provenance


class TestPoolingRecord:
    def test_provenance_is_detached_from_the_callers_dict(self) -> None:
        provenance: dict[str, Any] = {'finding_ids': ['P1-f1'], 'real': 'yes'}
        rec = _record('d1', 'd1_pool_01', **provenance)

        provenance['real'] = 'no'
        provenance['finding_ids'].append('P2-f1')

        assert rec.provenance == {'finding_ids': ('P1-f1',), 'real': 'yes'}

    def test_provenance_cannot_be_mutated_through_the_record(self) -> None:
        rec = _record('d1', 'd1_pool_01', finding_ids=['P1-f1'], scores={'a': [1]})

        assert not isinstance(rec.provenance, MutableMapping)
        assert not isinstance(rec.provenance['scores'], MutableMapping)
        assert not isinstance(rec.provenance['finding_ids'], MutableSequence)


def _make_gt(id: str, mutation_type: str = 'test_mutation') -> GroundTruthIssue:
    return GroundTruthIssue(
        id=id,
        location='src/foo.py:42',
        category='test_category',
        severity='blocking',
        description='Test issue',
        mutation_type=mutation_type,
    )


def _make_diff(diff_id: str, ground_truth: list[GroundTruthIssue]) -> CorpusDiff:
    return CorpusDiff(
        diff_id=diff_id,
        language='python',
        source='synthetic',
        diff_text='--- a/foo.py\n+++ b/foo.py\n@@ -1 +1 @@\n-old\n+new',
        description='Test diff',
        ground_truth=ground_truth,
    )


def _manifest() -> CorpusManifest:
    return CorpusManifest(diffs=[
        _make_diff('d1', [_make_gt('a')]),
        _make_diff('d2', [_make_gt('b')]),
    ])


def _entry(issue_id: str, **overrides: Any) -> dict[str, Any]:
    entry: dict[str, Any] = {
        'id': issue_id,
        'location': 'src/foo.py:7',
        'category': 'pooled_category',
        'severity': 'suggestion',
        'description': f'pooled issue {issue_id}',
        'mutation_type': 'pooled_test',
        'x_provenance': {
            'finding_ids': [f'{issue_id}-finding'],
            'skeptic_corrections': [],
        },
    }
    entry.update(overrides)
    return entry


def _without(entry: dict[str, Any], key: str) -> dict[str, Any]:
    return {k: v for k, v in entry.items() if k != key}


def _issue_of(entry: dict[str, Any]) -> GroundTruthIssue:
    return GroundTruthIssue(**_without(entry, 'x_provenance'))


def _fold(
    pooled: dict[str, list[dict[str, Any]]],
    manifest: CorpusManifest | None = None,
    log: PoolingLog | None = None,
) -> tuple[CorpusManifest, PoolingLog]:
    return fold_pooled_issues(
        manifest if manifest is not None else _manifest(),
        log if log is not None else PoolingLog(),
        pooled,
        pooling_date='2026-10-22',
        pooling_run='run-x',
    )


def _ground_truth(manifest: CorpusManifest, diff_id: str) -> list[GroundTruthIssue]:
    diff = manifest.get_diff(diff_id)
    assert diff is not None
    return diff.ground_truth


class TestFoldPooledIssues:
    def test_pooled_issues_follow_the_existing_key_in_input_order(self) -> None:
        e1, e2 = _entry('d1_pool_01'), _entry('d1_pool_02', severity='blocking')

        folded, _ = _fold({'d1': [e1, e2]})

        d1 = _ground_truth(folded, 'd1')
        assert [gt.id for gt in d1] == ['a', 'd1_pool_01', 'd1_pool_02']
        assert d1[1:] == [_issue_of(e1), _issue_of(e2)]
        assert _ground_truth(folded, 'd2') == [_make_gt('b')]

    def test_inputs_are_not_mutated(self) -> None:
        manifest = _manifest()
        prior = _record('d2', 'd2_pool_01')
        log = PoolingLog(records=(prior,))
        e1 = _entry('d1_pool_01')
        pooled = {'d1': [e1]}

        _fold(pooled, manifest=manifest, log=log)

        assert _ground_truth(manifest, 'd1') == [_make_gt('a')]
        assert log.records == (prior,)
        assert pooled == {'d1': [_entry('d1_pool_01')]}

    def test_returned_log_appends_one_record_per_pooled_issue(self) -> None:
        prior = _record('d2', 'd2_pool_01', confidence='low')
        e1 = _entry('d1_pool_01')
        e2 = _entry('d1_pool_02', x_provenance={'finding_ids': ['other']})

        _, log = _fold({'d1': [e1, e2]}, log=PoolingLog(records=(prior,)))

        assert log.records == (
            prior,
            PoolingRecord('d1', 'd1_pool_01', '2026-10-22', 'run-x', e1['x_provenance']),
            PoolingRecord('d1', 'd1_pool_02', '2026-10-22', 'run-x', e2['x_provenance']),
        )

    def test_folded_manifest_and_log_round_trip_through_disk(
        self, tmp_path: Path,
    ) -> None:
        folded, log = _fold({'d1': [_entry('d1_pool_01'), _entry('d1_pool_02')]})

        folded.save(tmp_path / 'manifest.json')
        log.save(tmp_path / 'pooling_log.jsonl')

        loaded = CorpusManifest.load(tmp_path / 'manifest.json')
        assert _ground_truth(loaded, 'd1') == _ground_truth(folded, 'd1')
        assert PoolingLog.load(tmp_path / 'pooling_log.jsonl') == log

    @pytest.mark.parametrize(
        ('pooled', 'offending'),
        [
            pytest.param({'nope': [_entry('nope_pool_01')]}, 'nope', id='unknown-diff'),
            pytest.param({'d1': [_entry('a')]}, 'a', id='collides-with-key'),
            pytest.param(
                {'d1': [_entry('d1_pool_01'), _entry('d1_pool_01')]},
                'd1_pool_01',
                id='repeated-in-batch',
            ),
            pytest.param(
                {'d1': [_without(_entry('d1_pool_01'), 'x_provenance')]},
                'x_provenance',
                id='missing-provenance',
            ),
            pytest.param(
                {'d1': [_entry('d1_pool_01', x_extra=1)]}, 'x_extra', id='unexpected-key',
            ),
            pytest.param(
                {'d1': [_without(_entry('d1_pool_01'), 'severity')]},
                'severity',
                id='missing-issue-field',
            ),
            pytest.param(
                {'d1': [_entry('d1_pool_01', mutation_type='logic')]},
                'logic',
                id='mutation-type-not-pooled',
            ),
        ],
    )
    def test_rejects_entries_it_cannot_fold_and_names_the_offender(
        self, pooled: dict[str, list[dict[str, Any]]], offending: str,
    ) -> None:
        with pytest.raises(ValueError, match=re.escape(repr(offending))):
            _fold(pooled)


def _folded_pair() -> tuple[CorpusManifest, PoolingLog]:
    return _fold({'d1': [_entry('d1_pool_01'), _entry('d1_pool_02')]})


class TestAuditPooling:
    def test_fold_output_is_ok(self) -> None:
        audit = audit_pooling(*_folded_pair())

        assert audit.ok is True
        assert audit.duplicate_records == ()
        assert audit.dangling_records == ()
        assert audit.unrecorded_issues == ()

    def test_empty_log_over_an_unpooled_corpus_is_ok(self) -> None:
        assert audit_pooling(_manifest(), PoolingLog()).ok is True

    def test_record_naming_no_key_issue_is_dangling(self) -> None:
        manifest, log = _folded_pair()
        log = PoolingLog(records=(*log.records, _record('d1', 'ghost')))

        audit = audit_pooling(manifest, log)

        assert audit.dangling_records == (('d1', 'ghost'),)
        assert audit.ok is False

    def test_pooled_issue_without_a_record_is_unrecorded(self) -> None:
        manifest, log = _folded_pair()
        log = PoolingLog(records=tuple(r for r in log.records if r.issue_id != 'd1_pool_02'))

        audit = audit_pooling(manifest, log)

        assert audit.unrecorded_issues == (('d1', 'd1_pool_02'),)
        assert audit.ok is False

    def test_repeated_record_is_a_duplicate(self) -> None:
        manifest, log = _folded_pair()
        log = PoolingLog(records=(*log.records, log.records[0]))

        audit = audit_pooling(manifest, log)

        assert audit.duplicate_records == (('d1', 'd1_pool_01'),)
        assert audit.ok is False

    def test_non_pooled_key_issues_are_never_unrecorded(self) -> None:
        manifest, log = _folded_pair()
        recorded = {(r.diff_id, r.issue_id) for r in log.records}
        assert recorded.isdisjoint({('d1', 'a'), ('d2', 'b')})

        assert audit_pooling(manifest, log).unrecorded_issues == ()

    def test_losing_every_record_of_a_run_leaves_all_its_issues_unrecorded(self) -> None:
        manifest, _ = _folded_pair()

        audit = audit_pooling(manifest, PoolingLog())

        assert audit.unrecorded_issues == (('d1', 'd1_pool_01'), ('d1', 'd1_pool_02'))
        assert audit.ok is False

    def test_record_naming_an_unpooled_issue_is_dangling(self) -> None:
        manifest, log = _folded_pair()
        log = PoolingLog(records=(*log.records, _record('d1', 'a')))

        audit = audit_pooling(manifest, log)

        assert audit.dangling_records == (('d1', 'a'),)
        assert audit.unrecorded_issues == ()
        assert audit.ok is False


class TestIsPooledIssue:
    def test_pooled_mutation_type_marks_an_issue_pooled(self) -> None:
        assert is_pooled_issue(_make_gt('x', mutation_type='pooled_2026_10_22')) is True

    def test_other_mutation_types_are_not_pooled(self) -> None:
        assert is_pooled_issue(_make_gt('x', mutation_type='logic')) is False
