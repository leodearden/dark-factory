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
            assert conn is not None
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
            conn = store._conn
            assert conn is not None
            row = conn.execute('PRAGMA journal_mode').fetchone()
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

    def test_upsert_many_writes_all_rows_in_one_batch(self, tmp_path):
        store = self._make_store(tmp_path)
        store.open()
        try:
            store.upsert_many([
                {
                    'run_id': 'r1', 'stage': 'memory_consolidator',
                    'project_id': 'dark_factory', 'is_active': False,
                    'entry_json': '{"v": 1}', 'updated_at': 1.0,
                },
                {
                    'run_id': 'r1', 'stage': 'task_knowledge_sync',
                    'project_id': 'dark_factory', 'is_active': True,
                    'entry_json': '{"v": 2}', 'updated_at': 2.0,
                },
            ])
            rows = {r['stage']: r for r in store.load_all()}
            assert set(rows) == {'memory_consolidator', 'task_knowledge_sync'}
            assert rows['memory_consolidator']['entry_json'] == '{"v": 1}'
            assert rows['memory_consolidator']['is_active'] is False
            assert rows['task_knowledge_sync']['entry_json'] == '{"v": 2}'
            assert rows['task_knowledge_sync']['is_active'] is True
        finally:
            store.close()

    def test_upsert_many_conflict_replaces_existing_row(self, tmp_path):
        store = self._make_store(tmp_path)
        store.open()
        try:
            store.upsert_entry(
                run_id='r1', stage='memory_consolidator', project_id='dark_factory',
                is_active=False, entry_json='{"v": 1}', updated_at=1.0,
            )
            # Same (run_id, stage) via the batch path must REPLACE, not duplicate.
            store.upsert_many([
                {
                    'run_id': 'r1', 'stage': 'memory_consolidator',
                    'project_id': 'dark_factory', 'is_active': True,
                    'entry_json': '{"v": 2}', 'updated_at': 5.0,
                },
            ])
            rows = store.load_all()
            assert len(rows) == 1
            assert rows[0]['entry_json'] == '{"v": 2}'
            assert rows[0]['is_active'] is True
            assert rows[0]['updated_at'] == 5.0
        finally:
            store.close()

    def test_upsert_many_empty_is_noop(self, tmp_path):
        store = self._make_store(tmp_path)
        store.open()
        try:
            store.upsert_many([])  # must not raise, opens no transaction
            assert store.load_all() == []
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


# ---------------------------------------------------------------------------
# step-7: write-through persistence on every mutator — RED until step-8
# ---------------------------------------------------------------------------


class _WTFakeMemoryService:
    """Minimal async stub sufficient to drive cite_entity/cite_edge/cite_memory."""

    def __init__(self):
        self.entity_nodes = [{'uuid': 'a1b2c3d4-e5f6-7890-abcd-ef1234567890', 'name': 'Foo'}]
        self.edge_result = {'fact': 'fact text'}
        self.memory_result = {'fingerprint': 'fp'}

    async def get_entity(self, name, project_id):
        return {'nodes': self.entity_nodes, 'edges': []}

    async def get_edge(self, edge_uuid, project_id):
        return self.edge_result

    async def get_memory(self, memory_id, store, project_id):
        return self.memory_result

    async def count_memories_by_metadata(self, project_id, filters):
        return 1


class _WTFakeTaskInterceptor:
    async def get_task(self, task_id, project_root, tag=None):
        return {'data': {'title': 'Some Task'}, 'title': 'Some Task'}


class TestWriteThrough:
    """Every ReconReportState mutator write-throughs to the store immediately."""

    def _make_state(self, tmp_path):
        from fused_memory.server.recon_report import ReconReportState
        from fused_memory.server.recon_report_store import ReconReportStore

        store = ReconReportStore(tmp_path / 'recon_report_state.db')
        store.open()
        t = [0.0]
        state = ReconReportState(
            ttl_seconds=300,
            clock=lambda: t[0],
            memory_service=_WTFakeMemoryService(),
            task_interceptor=_WTFakeTaskInterceptor(),
            store=store,
        )
        state.known_projects = {'dark_factory': '/home/leo/src/dark-factory'}
        return state, store, t

    def _row_for(self, store, run_id, stage):
        rows = store.load_all()
        matches = [r for r in rows if r['run_id'] == run_id and r['stage'] == stage]
        assert len(matches) == 1, (
            f'expected exactly one persisted row for {(run_id, stage)!r}, got {len(matches)}'
        )
        return matches[0]

    def _entry_for(self, store, run_id, stage):
        from fused_memory.server.recon_report import _deserialize_entry

        return _deserialize_entry(self._row_for(store, run_id, stage)['entry_json'])

    @pytest.mark.asyncio
    async def test_full_lifecycle_write_through(self, tmp_path):
        state, store, _t = self._make_state(tmp_path)
        run_id = 'wt-run-1'
        stage = 'memory_consolidator'

        try:
            # start_report
            state.start_report(run_id=run_id, stage=stage, project_id='dark_factory')
            row = self._row_for(store, run_id, stage)
            assert row['is_active'] is True
            assert row['project_id'] == 'dark_factory'
            entry = self._entry_for(store, run_id, stage)
            assert entry.run_id == run_id
            assert entry.findings == []

            # add_finding — real (task_id, flag_type) signature
            result_real = state.add_finding(
                run_id=run_id, severity='moderate', category='memory_stale',
                description='real', suggested_action='act', actionable=True,
                task_id='42', flag_type='orphaned_knowledge',
            )
            fid_real = result_real['finding_id']
            entry = self._entry_for(store, run_id, stage)
            assert {f.finding_id for f in entry.findings} == {fid_real}

            # add_finding — null/null (description-hash dedup)
            result_null = state.add_finding(
                run_id=run_id, severity='low', category='other',
                description='null null observation', suggested_action='note',
                actionable=False,
            )
            fid_null = result_null['finding_id']
            entry = self._entry_for(store, run_id, stage)
            assert {f.finding_id for f in entry.findings} == {fid_real, fid_null}

            # delete_finding — retract the null/null finding.  Done BEFORE
            # complete() below: delete_finding is rejected once the owning
            # entry is completed (report_already_completed guard), so this is
            # the only point in a single-stage lifecycle where a genuine
            # delete is possible.
            delete_result = state.delete_finding(run_id, fid_null)
            assert delete_result == {'status': 'deleted', 'finding_id': fid_null}
            entry = self._entry_for(store, run_id, stage)
            assert {f.finding_id for f in entry.findings} == {fid_real}

            # set_stat
            state.set_stat(run_id, 'scanned', 10)
            entry = self._entry_for(store, run_id, stage)
            assert entry.stats['scanned'] == 10

            # inc_stat
            state.inc_stat(run_id, 'scanned', 5)
            entry = self._entry_for(store, run_id, stage)
            assert entry.stats['scanned'] == 15

            # cite_task
            cite_task_result = await state.cite_task(run_id, fid_real, 'dark_factory', '42')
            assert 'error' not in cite_task_result
            entry = self._entry_for(store, run_id, stage)
            cited_real = next(f for f in entry.findings if f.finding_id == fid_real)
            assert cited_real.cited_tasks and cited_real.cited_tasks[0]['task_id'] == '42'

            # cite_entity
            cite_entity_result = await state.cite_entity(run_id, fid_real, 'foo')
            assert 'error' not in cite_entity_result
            entry = self._entry_for(store, run_id, stage)
            cited_real = next(f for f in entry.findings if f.finding_id == fid_real)
            assert cited_real.cited_entities

            # cite_edge
            cite_edge_result = await state.cite_edge(
                run_id, fid_real, 'a1b2c3d4-e5f6-7890-abcd-ef1234567890'
            )
            assert 'error' not in cite_edge_result
            entry = self._entry_for(store, run_id, stage)
            cited_real = next(f for f in entry.findings if f.finding_id == fid_real)
            assert cited_real.cited_edges

            # cite_memory
            cite_memory_result = await state.cite_memory(
                run_id, fid_real, 'a1b2c3d4-e5f6-7890-abcd-ef1234567890', 'mem0'
            )
            assert 'error' not in cite_memory_result
            entry = self._entry_for(store, run_id, stage)
            cited_real = next(f for f in entry.findings if f.finding_id == fid_real)
            assert cited_real.cited_memories

            # complete
            complete_result = state.complete(run_id, 'summary text')
            assert complete_result['flagged_count'] == 1
            entry = self._entry_for(store, run_id, stage)
            assert entry.completed_at is not None
            assert entry.summary == 'summary text'
        finally:
            store.close()

    @pytest.mark.asyncio
    async def test_recon_lifecycle_filer_direct_sync_add_finding_persists(self, tmp_path):
        """recon_lifecycle_filer.py calls ReconReportState.add_finding directly
        (a sync call, not via an MCP tool wrapper) — confirm that path also
        write-throughs, since it's one of the two non-MCP-wrapper synchronous
        callers named in the design (the other is stages/base.py's start_report)."""
        from fused_memory.middleware.live_task_write_guard import (
            LifecycleResetFinding,
            LiveTaskSnapshot,
        )
        from fused_memory.server.recon_lifecycle_filer import build_lifecycle_reset_filer

        state, store, _t = self._make_state(tmp_path)
        run_id = 'wt-filer-run'
        stage = 'task_knowledge_sync'

        try:
            state.start_report(run_id=run_id, stage=stage, project_id='dark_factory')
            file_finding = build_lifecycle_reset_filer(state)
            finding = LifecycleResetFinding(
                task_id='7',
                project_id='dark_factory',
                op='update_task',
                before=LiveTaskSnapshot(
                    status='in-progress', claimant_run_id='run-x/session-1/pid=1',
                    heartbeat_at='2026-07-15T12:00:00+00:00',
                ),
                after=LiveTaskSnapshot(status='pending', claimant_run_id=None, heartbeat_at=None),
                diverged_fields=('claimant_run_id', 'status'),
            )
            await file_finding(finding)

            entry = self._entry_for(store, run_id, stage)
            assert len(entry.findings) == 1
            assert entry.findings[0].task_id == '7'
        finally:
            store.close()


# ---------------------------------------------------------------------------
# step-9: hydrate_from_store readback after a simulated restart — RED until
# step-10 adds ReconReportState.hydrate_from_store
# ---------------------------------------------------------------------------


class TestHydrateReadback:
    """A FRESH ReconReportState opened on the SAME store rebuilds its in-memory
    state (entries, active-stage pointer, and all four run-level dedup indices)
    from the persisted rows — the ρ observable signal that a mid-run restart
    does not lose previously-filed findings.
    """

    def _make_state(self, store):
        from fused_memory.server.recon_report import ReconReportState

        state = ReconReportState(
            ttl_seconds=300,
            clock=lambda: 0.0,
            memory_service=_WTFakeMemoryService(),
            task_interceptor=_WTFakeTaskInterceptor(),
            store=store,
        )
        state.known_projects = {'dark_factory': '/home/leo/src/dark-factory'}
        return state

    @pytest.mark.asyncio
    async def test_hydrate_rebuilds_state_after_restart(self, tmp_path):
        from fused_memory.server.recon_report_store import ReconReportStore

        db = tmp_path / 'recon_report_state.db'
        run_id = 'hy-run-1'
        s1 = 'memory_consolidator'
        s2 = 'task_knowledge_sync'
        null_desc = 'stage1 informational observation'
        entity_uuid = 'a1b2c3d4-e5f6-7890-abcd-ef1234567890'

        # ---- Phase 1: state A files across TWO live stages of one run ----
        store_a = ReconReportStore(db)
        store_a.open()
        try:
            state_a = self._make_state(store_a)

            # Stage 1 (memory_consolidator)
            state_a.start_report(run_id=run_id, stage=s1, project_id='dark_factory')
            fid_real = state_a.add_finding(
                run_id=run_id, severity='moderate', category='memory_stale',
                description='stage1 real finding', suggested_action='act',
                actionable=True, task_id='42', flag_type='orphaned_knowledge',
            )['finding_id']
            fid_s1_null = state_a.add_finding(
                run_id=run_id, severity='low', category='other',
                description=null_desc, suggested_action='note', actionable=False,
            )['finding_id']
            # Citations on the real stage-1 finding (cite_entity + cite_task).
            assert 'error' not in await state_a.cite_entity(run_id, fid_real, 'Foo')
            assert 'error' not in await state_a.cite_task(run_id, fid_real, 'dark_factory', '42')
            state_a.complete(run_id, 'stage1 summary')

            # Stage 2 (task_knowledge_sync) — becomes the ACTIVE stage.
            state_a.start_report(run_id=run_id, stage=s2, project_id='dark_factory')
            state_a.add_finding(
                run_id=run_id, severity='high', category='systemic_pattern',
                description='stage2 finding', suggested_action='fix',
                actionable=True, task_id='99', flag_type='systemic',
            )
            # Null-task finding + cite_task → registers a project-scoped cited-task
            # fold anchor in _run_cited_task_index (a run-level index that must be
            # rebuilt on hydrate).
            fid_s2_anchor = state_a.add_finding(
                run_id=run_id, severity='low', category='other',
                description='stage2 anchor observation', suggested_action='note',
            )['finding_id']
            assert 'error' not in await state_a.cite_task(
                run_id, fid_s2_anchor, 'dark_factory', '77'
            )
            # In-run cross-stage duplicate of stage-1's (42, orphaned_knowledge).
            dup_a = state_a.add_finding(
                run_id=run_id, severity='moderate', category='memory_stale',
                description='stage2 restatement', suggested_action='act',
                actionable=True, task_id='42', flag_type='orphaned_knowledge',
            )
            assert dup_a.get('error') == 'duplicate_finding'
            assert dup_a['existing_finding_id'] == fid_real

            # Snapshot state A's readback for a byte-for-byte comparison later.
            report_a_s1 = state_a.get_assembled_report(run_id, s1)
            report_a_s2 = state_a.get_assembled_report(run_id, s2)
            findings_a = {
                f['finding_id']: f for f in state_a.get_findings_for_run(run_id)
            }
        finally:
            store_a.close()

        # ---- Phase 2: simulate restart — FRESH state B, SAME store ----
        store_b = ReconReportStore(db)
        store_b.open()
        try:
            state_b = self._make_state(store_b)
            state_b.hydrate_from_store()

            # (1) Readback identical to state A (findings, citation lists, stats).
            assert state_b.get_assembled_report(run_id, s1) == report_a_s1
            assert state_b.get_assembled_report(run_id, s2) == report_a_s2
            findings_b = {
                f['finding_id']: f for f in state_b.get_findings_for_run(run_id)
            }
            assert set(findings_b) == set(findings_a)
            for fid, fa in findings_a.items():
                fb = findings_b[fid]
                assert fb['description'] == fa['description']
                assert fb['cited_entities'] == fa['cited_entities']
                assert fb['cited_tasks'] == fa['cited_tasks']
                assert fb['cited_edges'] == fa['cited_edges']
                assert fb['cited_memories'] == fa['cited_memories']

            # (2) _active restored — active_run_for_stage resolves the LIVE stage
            #     (stage 2), and NOT the completed earlier stage.
            assert state_b.active_run_for_stage(s2, 'dark_factory') == run_id
            assert state_b.active_run_for_stage(s1, 'dark_factory') is None

            # (3) Signature dedup index (_run_sig_index) rebuilt: a follow-up
            #     add_finding duplicating a hydrated (task_id, flag_type) returns
            #     duplicate_finding carrying the ORIGINAL finding_id.
            dup_b = state_b.add_finding(
                run_id=run_id, severity='moderate', category='memory_stale',
                description='post-restart restatement', suggested_action='act',
                actionable=True, task_id='42', flag_type='orphaned_knowledge',
            )
            assert dup_b.get('error') == 'duplicate_finding'
            assert dup_b['existing_finding_id'] == fid_real

            # (3b) Description-hash dedup index (_run_desc_index) rebuilt across
            #      stages: a null/null finding repeating stage-1's description
            #      folds onto the original stage-1 finding.
            dup_desc = state_b.add_finding(
                run_id=run_id, severity='low', category='other',
                description=null_desc, suggested_action='note', actionable=False,
            )
            assert dup_desc.get('error') == 'duplicate_finding'
            assert dup_desc['existing_finding_id'] == fid_s1_null

            # (3c) Cited-task fold anchor index (_run_cited_task_index) rebuilt:
            #      a new null-task finding citing the hydrated anchor's task
            #      folds onto the original anchor finding.
            fid_new_anchor = state_b.add_finding(
                run_id=run_id, severity='low', category='other',
                description='post-restart anchor restatement',
                suggested_action='note',
            )['finding_id']
            fold_res = await state_b.cite_task(run_id, fid_new_anchor, 'dark_factory', '77')
            assert fold_res.get('error') == 'duplicate_finding'
            assert fold_res['existing_finding_id'] == fid_s2_anchor

            # (4) cite_entity on a hydrated STAGE-1 finding_id resolves via the
            #     rebuilt cross-stage _run_finding_index (while stage 2 is active).
            cite_b = await state_b.cite_entity(run_id, fid_real, 'Foo')
            assert 'error' not in cite_b
            assert cite_b['entity_uuid'] == entity_uuid
        finally:
            store_b.close()


# ---------------------------------------------------------------------------
# step-11: run-quiescence GC deletes a run's rows — RED until step-12 adds the
# tick() delete_run hook
# ---------------------------------------------------------------------------


class TestStoreGC:
    """tick() GCs a run's persisted rows at RUN QUIESCENCE (delete_run), not on
    per-entry TTL eviction — a completed-then-evicted stage's findings stay
    durable while a sibling stage of the same run is still live.
    """

    def _make_state(self, store, clock_holder):
        from fused_memory.server.recon_report import ReconReportState

        state = ReconReportState(
            ttl_seconds=300,
            clock=lambda: clock_holder[0],
            memory_service=_WTFakeMemoryService(),
            task_interceptor=_WTFakeTaskInterceptor(),
            store=store,
        )
        state.known_projects = {'dark_factory': '/home/leo/src/dark-factory'}
        return state

    def _add(self, state, run_id, desc):
        return state.add_finding(
            run_id=run_id, severity='low', category='other',
            description=desc, suggested_action='note', actionable=False,
        )['finding_id']

    def _run_ids_in_store(self, store):
        return {r['run_id'] for r in store.load_all()}

    def test_rows_gc_only_at_run_quiescence(self, tmp_path):
        from fused_memory.server.recon_report_store import ReconReportStore

        store = ReconReportStore(tmp_path / 'recon_report_state.db')
        store.open()
        t = [0.0]
        try:
            state = self._make_state(store, t)
            run_r = 'gc-run-R'
            run_live = 'gc-run-LIVE'
            s1 = 'memory_consolidator'
            s2 = 'task_knowledge_sync'

            # run R, stage 1 — completed at t=0.
            state.start_report(run_id=run_r, stage=s1, project_id='dark_factory')
            self._add(state, run_r, 'R stage1')
            state.complete(run_r, 'R s1 done')  # completed_at = 0

            # run R, stage 2 — becomes active; completed LATER at t=100.
            state.start_report(run_id=run_r, stage=s2, project_id='dark_factory')
            self._add(state, run_r, 'R stage2')

            # A DIFFERENT run that never completes → in-progress → immortal.
            state.start_report(run_id=run_live, stage=s1, project_id='dark_factory')
            self._add(state, run_live, 'LIVE stage1')

            # Both runs' rows are persisted up front.
            assert self._run_ids_in_store(store) == {run_r, run_live}

            # Complete R's stage 2 at t=100 (still well within TTL).
            t[0] = 100.0
            state.complete(run_r, 'R s2 done')  # completed_at = 100

            # --- Tick past ONLY stage 1's TTL: stage 2 survives, so run R is
            #     NOT quiescent — its rows must remain durable. ---
            t[0] = 301.0  # 301-0=301 > ttl(300) for s1; 301-100=201 < 300 for s2
            evicted = state.tick()
            assert evicted == 1  # only R/stage1 evicted from memory
            assert (run_r, s2) in state._state  # sibling stage still live
            assert run_r in self._run_ids_in_store(store)  # rows NOT GC'd yet
            assert run_live in self._run_ids_in_store(store)

            # --- Tick past stage 2's TTL as well: run R fully quiesces → its
            #     rows are GC'd (delete_run), the still-live run is untouched. ---
            t[0] = 401.0  # 401-100=301 > ttl(300) for s2
            evicted = state.tick()
            assert evicted == 1  # R/stage2 evicted
            assert run_r not in self._run_ids_in_store(store)  # GC'd with the run
            assert run_live in self._run_ids_in_store(store)  # different run untouched
            # In-memory run-level indices for R released at quiescence.
            assert run_r not in state._run_finding_index
        finally:
            store.close()

    def test_tick_with_store_none_does_not_raise_and_evicts(self, tmp_path):
        """tick()'s GC hook is a full no-op when store is None: eviction still
        happens (byte-identical count) and nothing raises."""
        t = [0.0]
        state = self._make_state(None, t)
        assert state._store is None
        run_id = 'gc-none-run'
        stage = 'memory_consolidator'

        state.start_report(run_id=run_id, stage=stage, project_id='dark_factory')
        self._add(state, run_id, 'finding')
        state.complete(run_id, 'done')  # completed_at = 0

        t[0] = 301.0
        evicted = state.tick()  # must not raise
        assert evicted == 1
        assert (run_id, stage) not in state._state
        assert run_id not in state._run_finding_index


# ---------------------------------------------------------------------------
# step-13: fresh in-process runs stay byte-identical with/without a store —
# RED until step-14 confirms every persistence touchpoint short-circuits on
# store=None.  (The shadow store must NEVER feed back into a live run.)
# ---------------------------------------------------------------------------


class _StoreTouched(BaseException):
    """Raised by :class:`_TripwireStore` on ANY store access.

    Deliberately a ``BaseException`` (not ``Exception``) so it is NOT swallowed
    by the ``except Exception`` guards inside ``_persist_run`` / ``tick`` /
    ``hydrate_from_store``.  That makes it a true tripwire: if a persistence
    touchpoint ever reaches the store when it should have short-circuited, the
    tripwire propagates all the way out and fails the test loudly instead of
    being logged-and-swallowed.
    """


class _TripwireStore:
    """A store stand-in whose every method trips :class:`_StoreTouched`.

    Used as a POSITIVE CONTROL (a non-None store IS reached by the persistence
    code) and, by contrast, to prove a ``store=None`` state never reaches it.
    """

    def open(self):
        raise _StoreTouched('open')

    def close(self):
        raise _StoreTouched('close')

    def upsert_entry(self, **kwargs):
        raise _StoreTouched('upsert_entry')

    def upsert_many(self, *args, **kwargs):
        raise _StoreTouched('upsert_many')

    def delete_run(self, run_id):
        raise _StoreTouched('delete_run')

    def load_all(self):
        raise _StoreTouched('load_all')


class TestByteIdenticalFreshRun:
    """Locks the no-regression constraint: a fresh in-process run behaves
    identically whether or not a shadow store is attached.
    """

    def _make_state(self, store):
        from fused_memory.server.recon_report import ReconReportState

        t = [0.0]
        state = ReconReportState(
            ttl_seconds=300,
            clock=lambda: t[0],
            memory_service=_WTFakeMemoryService(),
            task_interceptor=_WTFakeTaskInterceptor(),
            store=store,
        )
        state.known_projects = {'dark_factory': '/home/leo/src/dark-factory'}
        return state, t

    async def _drive_lifecycle(self, state):
        """Drive a rich single-stage lifecycle and return (run_id, stage).

        Exercises start → add (real + null/null) → duplicate (dedup) → cite →
        set/inc stat → delete → complete.  The delete is done BEFORE complete()
        because delete_finding is rejected once the owning entry is completed
        (report_already_completed guard) — same ordering constraint the
        TestWriteThrough happy path documents.
        """
        run_id = 'bi-run'
        stage = 'memory_consolidator'
        state.start_report(run_id=run_id, stage=stage, project_id='dark_factory')
        fid_real = state.add_finding(
            run_id=run_id, severity='moderate', category='memory_stale',
            description='real', suggested_action='act', actionable=True,
            task_id='42', flag_type='orphaned_knowledge',
        )['finding_id']
        fid_null = state.add_finding(
            run_id=run_id, severity='low', category='other',
            description='null null observation', suggested_action='note',
            actionable=False,
        )['finding_id']
        # Duplicate of the real (42, orphaned_knowledge) signature — returns the
        # ORIGINAL finding_id and allocates no new uuid.
        dup = state.add_finding(
            run_id=run_id, severity='moderate', category='memory_stale',
            description='restatement', suggested_action='act', actionable=True,
            task_id='42', flag_type='orphaned_knowledge',
        )
        assert dup.get('error') == 'duplicate_finding'
        assert dup['existing_finding_id'] == fid_real
        assert 'error' not in await state.cite_task(run_id, fid_real, 'dark_factory', '42')
        assert 'error' not in await state.cite_entity(run_id, fid_real, 'Foo')
        state.set_stat(run_id, 'scanned', 10)
        state.inc_stat(run_id, 'scanned', 5)
        # Genuine deletion of the null/null finding while still in-progress.
        assert state.delete_finding(run_id, fid_null) == {
            'status': 'deleted', 'finding_id': fid_null,
        }
        state.complete(run_id, 'summary text')
        return run_id, stage

    def _patch_deterministic_uuids(self, monkeypatch):
        """Pin recon_report.uuid.uuid4 to a fresh deterministic sequence.

        finding_id is the only source of nondeterminism in the assembled/raw
        readback, so pinning it lets the two runs' outputs be compared for
        FULL dict equality (not merely equal-modulo-finding-id).  Re-called
        before each run so both runs draw the identical uuid sequence.
        """
        import itertools

        import fused_memory.server.recon_report as rr

        counter = itertools.count(1)
        monkeypatch.setattr(rr.uuid, 'uuid4', lambda: rr.uuid.UUID(int=next(counter)))

    @pytest.mark.asyncio
    async def test_outputs_byte_identical_with_and_without_store(self, tmp_path, monkeypatch):
        """(a) get_assembled_report / get_findings_for_run are EQUAL dicts
        whether the run used store=None or a live ReconReportStore — the shadow
        store never alters live-run results."""
        from fused_memory.server.recon_report_store import ReconReportStore

        # Run 1 — no store.
        self._patch_deterministic_uuids(monkeypatch)
        state_none, _ = self._make_state(None)
        run_id, stage = await self._drive_lifecycle(state_none)
        report_none = state_none.get_assembled_report(run_id, stage)
        findings_none = state_none.get_findings_for_run(run_id)

        # Run 2 — live shadow store, identical deterministic uuid sequence.
        self._patch_deterministic_uuids(monkeypatch)
        store = ReconReportStore(tmp_path / 'recon_report_state.db')
        store.open()
        try:
            state_store, _ = self._make_state(store)
            run_id2, stage2 = await self._drive_lifecycle(state_store)
            report_store = state_store.get_assembled_report(run_id2, stage2)
            findings_store = state_store.get_findings_for_run(run_id2)
        finally:
            store.close()

        assert report_none == report_store
        assert findings_none == findings_store
        # And a sanity floor: the run actually produced content to compare.
        assert report_none is not None
        assert len(findings_none) == 1

    @pytest.mark.asyncio
    async def test_non_none_store_IS_reached_positive_control(self, tmp_path):
        """Positive control for (b): a NON-None store IS reached by the
        persistence code — start_report's _persist_run trips the tripwire.
        Without this, the store=None no-op assertion below would be vacuous."""
        state, _ = self._make_state(_TripwireStore())
        with pytest.raises(_StoreTouched):
            state.start_report(run_id='bi-run', stage='memory_consolidator', project_id='dark_factory')

    @pytest.mark.asyncio
    async def test_store_none_never_touches_a_store(self, tmp_path, caplog):
        """(b) With store=None every persistence touchpoint short-circuits
        BEFORE touching a store: the full lifecycle + tick() + hydrate run
        without raising, and no persist-failure WARNING is ever logged (a
        removed `is None` guard would fall into the `except Exception` arm and
        log one per entry)."""
        import logging

        state, t = self._make_state(None)
        assert state._store is None

        with caplog.at_level(logging.WARNING, logger='fused_memory.server.recon_report'):
            run_id, stage = await self._drive_lifecycle(state)
            # hydrate is a no-op when store is None (does not raise, changes nothing).
            state.hydrate_from_store()
            assert (run_id, stage) in state._state
            # Advance past TTL and tick — the GC hook must also short-circuit.
            t[0] = 10_000.0
            evicted = state.tick()
            assert evicted == 1
            assert (run_id, stage) not in state._state

        persist_warnings = [
            r for r in caplog.records if 'failed to persist' in r.getMessage()
            or 'failed to GC' in r.getMessage()
        ]
        assert persist_warnings == []

    def test_store_none_happy_path_subset_unchanged(self):
        """(c) A representative subset of the existing test_recon_report.py
        happy-path assertions still holds verbatim with store=None."""
        from fused_memory.server.recon_report import ReconReportState

        state = ReconReportState(ttl_seconds=300, clock=lambda: 0.0, store=None)

        # start_report → empty assembled report
        state.start_report(run_id='r1', stage='memory_consolidator', project_id='dark_factory')
        report = state.get_assembled_report('r1', 'memory_consolidator')
        assert report == {
            'summary': '', 'stats': {}, 'flagged_items': [], 'summary_warnings': [],
        }

        # add_finding → 36-char uuid finding_id
        first = state.add_finding(
            run_id='r1', severity='moderate', category='memory_stale',
            description='d', suggested_action='a', actionable=True,
            task_id='42', flag_type='orphaned_knowledge',
        )
        assert len(first['finding_id']) == 36

        # second same signature → duplicate_finding with the original id
        second = state.add_finding(
            run_id='r1', severity='moderate', category='memory_stale',
            description='different text same sig', suggested_action='a',
            actionable=True, task_id='42', flag_type='orphaned_knowledge',
        )
        assert second['error'] == 'duplicate_finding'
        assert second['existing_finding_id'] == first['finding_id']

        # set/inc stat + complete → cached flagged_count/stats
        state.set_stat('r1', 'scanned', 100)
        state.inc_stat('r1', 'scanned', 5)
        result = state.complete('r1', 'summary text')
        assert result == {'flagged_count': 1, 'stats': {'scanned': 105}}

        # assembled report full shape after complete
        report = state.get_assembled_report('r1', 'memory_consolidator')
        assert report is not None
        assert report['summary'] == 'summary text'
        assert report['stats'] == {'scanned': 105}
        assert len(report['flagged_items']) == 1
        item = report['flagged_items'][0]
        assert item['finding_id'] == first['finding_id']
        assert item['task_id'] == '42'
        assert item['flag_type'] == 'orphaned_knowledge'
        assert item['cited_entities'] == []
        assert item['cited_tasks'] == []


# ---------------------------------------------------------------------------
# step-15: _build_recon_report_components store wiring + start/stop_persistence
# — RED until step-16 constructs the store and adds the persistence lifecycle.
# Socket-free, mirroring TestReconReportBoot / TestReconReportReaperWiredAtBoot
# in test_recon_report.py.
# ---------------------------------------------------------------------------


class TestReconReportStoreWiredAtBoot:
    """_build_recon_report_components attaches a ReconReportStore (or None when
    disabled), and ReconReportState exposes start_persistence/stop_persistence.
    """

    def _make_config(self, data_dir, *, persist=True, enabled=True, recon_port=8003):
        from fused_memory.config.schema import (
            FusedMemoryConfig,
            ReconciliationConfig,
            ServerConfig,
        )

        return FusedMemoryConfig(
            server=ServerConfig(recon_report_port=recon_port, host='127.0.0.1'),
            reconciliation=ReconciliationConfig(
                enabled=enabled,
                recon_report_persist_enabled=persist,
                recon_report_state_ttl_seconds=300,
                data_dir=str(data_dir),
            ),
        )

    def test_store_attached_at_data_dir_and_not_yet_opened(self, tmp_path):
        from pathlib import Path

        from fused_memory.server.main import _build_recon_report_components
        from fused_memory.server.recon_report_store import ReconReportStore

        config = self._make_config(tmp_path, persist=True, enabled=True)
        state, _, _ = _build_recon_report_components(config)

        assert isinstance(state._store, ReconReportStore)
        # DB is a sibling of reconciliation.db under data_dir.
        assert state._store.db_path == Path(str(tmp_path)) / 'recon_report_state.db'
        # NOT opened at build time — parity with the reaper-not-started-at-build
        # invariant; only run_server()'s start_persistence() opens it.
        assert state._store._conn is None
        # No file materialized just by building.
        assert not (tmp_path / 'recon_report_state.db').exists()

    def test_store_none_when_persist_disabled(self, tmp_path):
        from fused_memory.server.main import _build_recon_report_components

        config = self._make_config(tmp_path, persist=False, enabled=True)
        state, _, _ = _build_recon_report_components(config)
        assert state._store is None

    def test_store_none_when_reconciliation_disabled(self, tmp_path):
        from fused_memory.server.main import _build_recon_report_components

        config = self._make_config(tmp_path, persist=True, enabled=False)
        state, _, _ = _build_recon_report_components(config)
        assert state._store is None

    def test_state_exposes_start_and_stop_persistence(self, tmp_path):
        from fused_memory.server.main import _build_recon_report_components

        config = self._make_config(tmp_path, persist=True, enabled=True)
        state, _, _ = _build_recon_report_components(config)
        assert callable(state.start_persistence)
        assert callable(state.stop_persistence)

    def test_start_persistence_opens_and_hydrates_then_stop_closes(self, tmp_path):
        """start_persistence() opens the store AND runs hydrate_from_store (so a
        pre-populated DB is read back into memory); stop_persistence() closes it
        and is idempotent."""
        from fused_memory.server.main import _build_recon_report_components
        from fused_memory.server.recon_report import (
            _Finding,
            _ReportEntry,
            _serialize_entry,
        )
        from fused_memory.server.recon_report_store import ReconReportStore

        run_id = 'boot-run'
        stage = 'memory_consolidator'
        finding = _Finding(
            finding_id='11111111-1111-1111-1111-111111111111',
            severity='moderate', category='memory_stale',
            description='pre-populated finding', suggested_action='act',
            actionable=True, task_id='42', flag_type='orphaned_knowledge',
        )
        entry = _ReportEntry(
            run_id=run_id, stage=stage, project_id='dark_factory',
            findings=[finding], summary='pre summary', created_at=1.0,
        )

        # Pre-populate exactly the DB path _build_recon_report_components points at.
        db_path = tmp_path / 'recon_report_state.db'
        pre_store = ReconReportStore(db_path)
        pre_store.open()
        try:
            pre_store.upsert_entry(
                run_id=run_id, stage=stage, project_id='dark_factory',
                is_active=True,
                entry_json=_serialize_entry(
                    entry, sig_anchor_slice={}, cited_task_slice={}
                ),
                updated_at=1.0,
            )
        finally:
            pre_store.close()

        config = self._make_config(tmp_path, persist=True, enabled=True)
        state, _, _ = _build_recon_report_components(config)
        assert state._store is not None
        assert state._store._conn is None  # not opened yet

        state.start_persistence()
        try:
            assert state._store._conn is not None  # opened
            # hydrate ran during start_persistence — the pre-populated finding is
            # now readable from in-memory state.
            report = state.get_assembled_report(run_id, stage)
            assert report is not None
            assert [i['finding_id'] for i in report['flagged_items']] == [
                finding.finding_id
            ]
            assert report['summary'] == 'pre summary'
            # is_active row restored the active-stage pointer.
            assert state.active_run_for_stage(stage, 'dark_factory') == run_id
        finally:
            state.stop_persistence()

        assert state._store._conn is None  # closed
        state.stop_persistence()  # idempotent — must not raise

    def test_start_stop_persistence_are_noops_when_store_none(self, tmp_path):
        """With persistence disabled (store=None), start/stop_persistence are a
        total no-op and never raise."""
        from fused_memory.server.main import _build_recon_report_components

        config = self._make_config(tmp_path, persist=False, enabled=True)
        state, _, _ = _build_recon_report_components(config)
        assert state._store is None
        state.start_persistence()  # must not raise
        state.stop_persistence()  # must not raise


# ---------------------------------------------------------------------------
# step-17: hydrate re-anchors each completed entry's TTL baseline across the
# restart boundary, so the reaper still evicts (and GCs) a hydrated completed
# run instead of leaking its rows forever — RED until step-18 adds the re-anchor
# ---------------------------------------------------------------------------


class TestHydrateReanchorEviction:
    """A completed entry hydrated across a restart must have its TTL measured
    from the RESTART baseline, not the prior process's clock value.

    The reaper's default clock is ``asyncio.get_running_loop().time()``
    (monotonic), which restarts near 0 on a fresh event loop.  A verbatim-
    restored ``completed_at`` stamped by the prior process is therefore a large
    STALE value: in ``tick()`` the age ``now - completed_at`` comes out negative
    and never exceeds ``ttl_seconds``, so a hydrated completed run would never
    evict, its run would never quiesce, and ``store.delete_run`` would never fire
    — its ``recon_report_state.db`` rows (and their in-memory entries) leak
    unboundedly across every restart cycle.  This pins the fix: re-anchor each
    hydrated ``completed_at`` to ``_clock()`` inside ``hydrate_from_store`` so the
    TTL counts from restart and the existing run-quiescence GC path reclaims it.
    """

    def _make_state(self, store, clock_holder):
        from fused_memory.server.recon_report import ReconReportState

        state = ReconReportState(
            ttl_seconds=300,
            clock=lambda: clock_holder[0],
            memory_service=_WTFakeMemoryService(),
            task_interceptor=_WTFakeTaskInterceptor(),
            store=store,
        )
        state.known_projects = {'dark_factory': '/home/leo/src/dark-factory'}
        return state

    def test_hydrated_completed_run_evicts_and_gcs_from_restart_baseline(
        self, tmp_path
    ):
        from fused_memory.server.recon_report_store import ReconReportStore

        db = tmp_path / 'recon_report_state.db'
        run_id = 'reanchor-run'
        stage = 'memory_consolidator'

        # ---- PHASE 1: long-lived "old process" with a LARGE monotonic clock ----
        # t_a=500_000 simulates a server whose asyncio loop time has run for days.
        t_a = [500_000.0]
        store_a = ReconReportStore(db)
        state_a = self._make_state(store_a, t_a)
        state_a.start_persistence()  # opens store_a (fresh DB → hydrate is a no-op)
        try:
            state_a.start_report(
                run_id=run_id, stage=stage, project_id='dark_factory'
            )
            fid = state_a.add_finding(
                run_id=run_id, severity='low', category='other',
                description='old-process finding', suggested_action='note',
                actionable=False,
            )['finding_id']
            # completed_at is stamped ~500_000 and write-through persisted.
            state_a.complete(run_id, 'old process done')
            rows_a = [r for r in store_a.load_all() if r['run_id'] == run_id]
            assert len(rows_a) == 1
        finally:
            # Simulate the process dying — close the connection.
            state_a.stop_persistence()

        # ---- PHASE 2: fresh event-loop clock near 0 = "restart" ----
        t_b = [10.0]
        store_b = ReconReportStore(db)
        state_b = self._make_state(store_b, t_b)
        state_b.start_persistence()  # open the SAME db + hydrate_from_store
        try:
            # (a) The completed finding hydrated back into memory.
            findings = state_b.get_findings_for_run(run_id)
            assert [f['finding_id'] for f in findings] == [fid]

            # (b) TTL is measured from the RESTART baseline (10), NOT the stale
            #     500_000: just BEFORE ttl elapses, nothing evicts and the finding
            #     is still present.
            t_b[0] = 10.0 + 300.0 - 1.0  # 309 → 309-10=299 < ttl(300)
            assert state_b.tick() == 0
            assert [
                f['finding_id'] for f in state_b.get_findings_for_run(run_id)
            ] == [fid]

            # (c) Just AFTER ttl elapses (from restart), the run evicts, fully
            #     quiesces, and its persisted rows are GC'd (delete_run fired).
            #     This is RED under a verbatim completed_at restore: 311-500_000 is
            #     always negative, so tick() never evicts, the run never quiesces,
            #     and its recon_report_state.db rows leak forever.
            t_b[0] = 10.0 + 300.0 + 1.0  # 311 → 311-10=301 > ttl(300)
            assert state_b.tick() >= 1
            assert state_b.get_findings_for_run(run_id) == []
            assert [
                r for r in store_b.load_all() if r['run_id'] == run_id
            ] == []
        finally:
            state_b.stop_persistence()
