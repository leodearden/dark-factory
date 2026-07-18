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
