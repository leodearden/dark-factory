"""Tests for the reviewer_trial answer-key pooling log and fold."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from orchestrator.evals.reviewer_trial.key_pooling import PoolingLog, PoolingRecord


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
