"""Tests for SQLite-backed persistence of ReconReportState (task 2716).

Covers:
- Config schema field recon_report_persist_enabled (TestConfigSchema)
- ReconReportStore sync lifecycle (TestReconReportStore)
- _serialize_entry / _deserialize_entry round-trip (TestEntrySerialization)
- Write-through on every mutator (TestWriteThrough)
- hydrate_from_store readback after a simulated restart (TestHydrateReadback)
- Run-quiescence GC deletes a run's rows (TestStoreGC)
- Fresh in-process runs stay byte-identical with/without a store (TestByteIdenticalFreshRun)
- _build_recon_report_components / start_persistence / stop_persistence wiring
  (TestReconReportStoreWiredAtBoot)
"""

from __future__ import annotations

import pytest

# ---------------------------------------------------------------------------
# step-1: Config schema — RED until step-2 adds the field
# ---------------------------------------------------------------------------


class TestConfigSchema:
    """Verify the new recon_report_persist_enabled config field."""

    def test_recon_report_persist_enabled_default_true(self):
        from fused_memory.config.schema import ReconciliationConfig

        cfg = ReconciliationConfig()
        assert cfg.recon_report_persist_enabled is True

    def test_recon_report_persist_enabled_roundtrips_false(self):
        from fused_memory.config.schema import ReconciliationConfig

        cfg = ReconciliationConfig(recon_report_persist_enabled=False)
        assert cfg.recon_report_persist_enabled is False


# ---------------------------------------------------------------------------
# step-3: ReconReportStore sync lifecycle — RED until step-4 creates the module
# ---------------------------------------------------------------------------


class TestReconReportStore:
    """Exercise the sync SQLite store's lifecycle against a tmp_path DB."""

    def _make_store(self, tmp_path, name='recon_report_state.db'):
        from fused_memory.server.recon_report_store import ReconReportStore

        return ReconReportStore(tmp_path / name)

    def test_open_creates_db_file(self, tmp_path):
        store = self._make_store(tmp_path)
        db_path = tmp_path / 'recon_report_state.db'
        assert not db_path.exists()
        store.open()
        try:
            assert db_path.exists()
        finally:
            store.close()

    def test_open_creates_table_with_composite_primary_key(self, tmp_path):
        store = self._make_store(tmp_path)
        store.open()
        try:
            conn = store._conn
            cur = conn.execute("PRAGMA table_info(recon_report_state)")
            cols = {row[1]: row for row in cur.fetchall()}
            assert set(cols) >= {
                'run_id', 'stage', 'project_id', 'is_active', 'entry_json', 'updated_at',
            }
            # pk column ordinal (row[5]) is 1-indexed and non-zero for PK members.
            pk_cols = {name for name, row in cols.items() if row[5] > 0}
            assert pk_cols == {'run_id', 'stage'}
        finally:
            store.close()

    def test_open_sets_wal_journal_mode(self, tmp_path):
        store = self._make_store(tmp_path)
        store.open()
        try:
            row = store._conn.execute('PRAGMA journal_mode').fetchone()
            assert row[0] == 'wal'
        finally:
            store.close()

    def test_upsert_then_load_all_returns_row_verbatim(self, tmp_path):
        store = self._make_store(tmp_path)
        store.open()
        try:
            store.upsert_entry(
                run_id='r1',
                stage='memory_consolidator',
                project_id='dark_factory',
                is_active=True,
                entry_json='{"a": 1}',
                updated_at=123.5,
            )
            rows = store.load_all()
            assert len(rows) == 1
            row = rows[0]
            assert row['run_id'] == 'r1'
            assert row['stage'] == 'memory_consolidator'
            assert row['project_id'] == 'dark_factory'
            assert row['is_active'] is True
            assert row['entry_json'] == '{"a": 1}'
            assert row['updated_at'] == 123.5
        finally:
            store.close()

    def test_second_upsert_same_key_replaces_not_duplicates(self, tmp_path):
        store = self._make_store(tmp_path)
        store.open()
        try:
            store.upsert_entry(
                run_id='r1', stage='memory_consolidator', project_id='dark_factory',
                is_active=True, entry_json='{"v": 1}', updated_at=1.0,
            )
            store.upsert_entry(
                run_id='r1', stage='memory_consolidator', project_id='dark_factory',
                is_active=False, entry_json='{"v": 2}', updated_at=2.0,
            )
            rows = store.load_all()
            assert len(rows) == 1
            assert rows[0]['entry_json'] == '{"v": 2}'
            assert rows[0]['is_active'] is False
            assert rows[0]['updated_at'] == 2.0
        finally:
            store.close()

    def test_upsert_distinct_stages_same_run_are_separate_rows(self, tmp_path):
        store = self._make_store(tmp_path)
        store.open()
        try:
            store.upsert_entry(
                run_id='r1', stage='memory_consolidator', project_id='dark_factory',
                is_active=False, entry_json='{}', updated_at=1.0,
            )
            store.upsert_entry(
                run_id='r1', stage='task_knowledge_sync', project_id='dark_factory',
                is_active=True, entry_json='{}', updated_at=2.0,
            )
            rows = store.load_all()
            assert len(rows) == 2
            assert {r['stage'] for r in rows} == {'memory_consolidator', 'task_knowledge_sync'}
        finally:
            store.close()

    def test_delete_run_removes_all_of_that_runs_rows(self, tmp_path):
        store = self._make_store(tmp_path)
        store.open()
        try:
            store.upsert_entry(
                run_id='r1', stage='memory_consolidator', project_id='dark_factory',
                is_active=False, entry_json='{}', updated_at=1.0,
            )
            store.upsert_entry(
                run_id='r1', stage='task_knowledge_sync', project_id='dark_factory',
                is_active=True, entry_json='{}', updated_at=2.0,
            )
            store.upsert_entry(
                run_id='r2', stage='memory_consolidator', project_id='dark_factory',
                is_active=True, entry_json='{}', updated_at=3.0,
            )
            store.delete_run('r1')
            rows = store.load_all()
            assert len(rows) == 1
            assert rows[0]['run_id'] == 'r2'
        finally:
            store.close()

    def test_close_is_idempotent(self, tmp_path):
        store = self._make_store(tmp_path)
        store.open()
        store.close()
        store.close()  # must not raise

    def test_close_before_open_is_safe(self, tmp_path):
        store = self._make_store(tmp_path)
        store.close()  # must not raise even though open() was never called


# ---------------------------------------------------------------------------
# step-5: _serialize_entry / _deserialize_entry round-trip — RED until step-6
# ---------------------------------------------------------------------------


class TestEntrySerialization:
    """Round-trip a _ReportEntry (+ run-level fold-anchor slices) through JSON."""

    def _build_entry(self, *, completed_at):
        from fused_memory.server.recon_report import _Finding, _ReportEntry

        f_real = _Finding(
            finding_id='11111111-1111-1111-1111-111111111111',
            severity='moderate',
            category='memory_stale',
            description='real sig finding',
            suggested_action='investigate',
            actionable=True,
            task_id='42',
            flag_type='orphaned_knowledge',
        )
        f_nullnull = _Finding(
            finding_id='22222222-2222-2222-2222-222222222222',
            severity='low',
            category='other',
            description='null null finding',
            suggested_action='note',
            actionable=False,
            task_id=None,
            flag_type=None,
        )
        f_cited = _Finding(
            finding_id='33333333-3333-3333-3333-333333333333',
            severity='high',
            category='cross_project_routing',
            description='fully cited finding',
            suggested_action='route it',
            actionable=True,
            task_id='99',
            flag_type='cross_project_info',
            cited_entities=[{'entity_uuid': 'e1', 'canonical_name': 'Foo'}],
            cited_edges=[{'edge_uuid': 'ed1', 'fact_text_snapshot': 'fact'}],
            cited_tasks=[{'project_id': 'dark_factory', 'task_id': '99', 'title': 'T'}],
            cited_memories=[{'memory_id': 'm1', 'store': 'mem0', 'metadata_fingerprint': 'fp'}],
            cited_runs=[{'run_id': 'r-cited', 'match_count': 3}],
        )

        entry = _ReportEntry(
            run_id='run-1',
            stage='memory_consolidator',
            project_id='dark_factory',
            findings=[f_real, f_nullnull, f_cited],
            stats={'scanned': 10, 'ratio': 0.5, 'label': 'ok'},
            summary='summary text',
            summary_warnings=['warn 1', 'warn 2'],
            completed_at=completed_at,
            created_at=100.0,
        )
        entry._signature_to_finding = {
            ('42', 'orphaned_knowledge'): f_real.finding_id,
            (None, 'ft'): 'ffffffff-ffff-ffff-ffff-ffffffffffff',
            ('123', None): 'aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa',
        }
        entry._deschash_to_finding = {
            'deadbeef' * 8: f_nullnull.finding_id,
        }
        return entry

    def _slices(self):
        sig_anchor_slice = {
            ('99', 'cross_project_info'): '33333333-3333-3333-3333-333333333333',
            (None, 'other_ft'): 'bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb',
        }
        cited_task_slice = {
            'dark_factory:99': '33333333-3333-3333-3333-333333333333',
        }
        return sig_anchor_slice, cited_task_slice

    def test_roundtrip_preserves_entry_with_completed_at_set(self):
        from fused_memory.server.recon_report import _deserialize_entry, _serialize_entry

        entry = self._build_entry(completed_at=555.5)
        sig_anchor_slice, cited_task_slice = self._slices()
        raw = _serialize_entry(
            entry, sig_anchor_slice=sig_anchor_slice, cited_task_slice=cited_task_slice
        )
        restored = _deserialize_entry(raw)
        assert restored == entry
        assert restored.completed_at == 555.5

    def test_roundtrip_preserves_entry_with_completed_at_none(self):
        from fused_memory.server.recon_report import _deserialize_entry, _serialize_entry

        entry = self._build_entry(completed_at=None)
        sig_anchor_slice, cited_task_slice = self._slices()
        raw = _serialize_entry(
            entry, sig_anchor_slice=sig_anchor_slice, cited_task_slice=cited_task_slice
        )
        restored = _deserialize_entry(raw)
        assert restored == entry
        assert restored.completed_at is None

    def test_roundtrip_preserves_findings_and_all_five_citation_lists(self):
        from fused_memory.server.recon_report import _deserialize_entry, _serialize_entry

        entry = self._build_entry(completed_at=None)
        sig_anchor_slice, cited_task_slice = self._slices()
        raw = _serialize_entry(
            entry, sig_anchor_slice=sig_anchor_slice, cited_task_slice=cited_task_slice
        )
        restored = _deserialize_entry(raw)

        assert [f.finding_id for f in restored.findings] == [f.finding_id for f in entry.findings]
        cited = restored.findings[2]
        expected = entry.findings[2]
        assert cited.cited_entities == expected.cited_entities
        assert cited.cited_edges == expected.cited_edges
        assert cited.cited_tasks == expected.cited_tasks
        assert cited.cited_memories == expected.cited_memories
        assert cited.cited_runs == expected.cited_runs

    def test_roundtrip_preserves_per_entry_dedup_mirrors_with_tuple_keys(self):
        from fused_memory.server.recon_report import _deserialize_entry, _serialize_entry

        entry = self._build_entry(completed_at=None)
        sig_anchor_slice, cited_task_slice = self._slices()
        raw = _serialize_entry(
            entry, sig_anchor_slice=sig_anchor_slice, cited_task_slice=cited_task_slice
        )
        restored = _deserialize_entry(raw)

        assert restored._signature_to_finding == entry._signature_to_finding
        # None must survive as real None, never coerced to the string 'None'.
        assert (None, 'ft') in restored._signature_to_finding
        assert ('123', None) in restored._signature_to_finding
        assert restored._deschash_to_finding == entry._deschash_to_finding

    def test_fold_anchor_slices_roundtrip_preserving_tuple_keys_and_none(self):
        from fused_memory.server.recon_report import (
            _deserialize_fold_anchor_slices,
            _serialize_entry,
        )

        entry = self._build_entry(completed_at=None)
        sig_anchor_slice, cited_task_slice = self._slices()
        raw = _serialize_entry(
            entry, sig_anchor_slice=sig_anchor_slice, cited_task_slice=cited_task_slice
        )

        out_sig_slice, out_cited_task_slice = _deserialize_fold_anchor_slices(raw)
        assert out_sig_slice == sig_anchor_slice
        assert out_cited_task_slice == cited_task_slice
        assert (None, 'other_ft') in out_sig_slice
        # Keys must be real tuples with a real None, not '(None, ...)' strings.
        assert all(isinstance(k, tuple) for k in out_sig_slice)
        assert all(k[0] is None or isinstance(k[0], str) for k in out_sig_slice)

    def test_stats_summary_and_summary_warnings_roundtrip(self):
        from fused_memory.server.recon_report import _deserialize_entry, _serialize_entry

        entry = self._build_entry(completed_at=None)
        sig_anchor_slice, cited_task_slice = self._slices()
        raw = _serialize_entry(
            entry, sig_anchor_slice=sig_anchor_slice, cited_task_slice=cited_task_slice
        )
        restored = _deserialize_entry(raw)

        assert restored.stats == entry.stats
        assert restored.summary == entry.summary
        assert restored.summary_warnings == entry.summary_warnings
        assert restored.created_at == entry.created_at
