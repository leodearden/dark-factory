"""Tests for recon_report.start_report's idempotency guard (task 3988).

`ReconReportState.start_report` used to unconditionally overwrite any
existing `(run_id, stage)` entry: a second call for the same pair silently
destroyed every finding/citation/stat recorded so far, orphaned the four
run-scoped dedup indices (a `duplicate_finding` pointer that could no
longer resolve), durably persisted the loss, and reset `completed_at` —
bypassing the post-`complete()` mutation guard. PRD §9.2 already specifies
`start_report` as "Idempotent"; this module pins that contract.

Covers:
- TestStartReportPreservesExistingEntry — step-1 (RED until step-2)
- TestStartReportPreservesExistingEntryDurable — step-1(e) (RED until step-2)
- TestStartReportResultIsUnambiguous — step-3 (RED until step-4)
- TestStartReportActivePointerAndPersistence — step-5 (RED until step-6)
"""

import logging

import pytest

# ---------------------------------------------------------------------------
# step-1: repeat start_report() must preserve the existing entry and its
# dedup indices — RED until step-2 adds the existence-check guard.
# ---------------------------------------------------------------------------


class TestStartReportPreservesExistingEntry:
    """A repeat start_report() for an existing (run_id, stage) must not
    destroy the entry's findings/stats/summary/created_at, must not orphan
    the run-scoped dedup indices, and must not bypass the post-complete()
    mutation guard.
    """

    def _make_state(self):
        from fused_memory.server.recon_report import ReconReportState

        t = [0.0]
        return ReconReportState(ttl_seconds=300, clock=lambda: t[0]), t

    def test_repeat_call_preserves_entry_identity_and_contents(self):
        state, t = self._make_state()
        state.start_report(run_id='r1', stage='memory_consolidator', project_id='dark_factory')

        real = state.add_finding(
            run_id='r1', severity='moderate', category='memory_stale',
            description='real finding', suggested_action='act', actionable=True,
            task_id='42', flag_type='orphaned_knowledge',
        )
        assert 'finding_id' in real, real
        real_id = real['finding_id']

        null = state.add_finding(
            run_id='r1', severity='low', category='other',
            description='null null observation', suggested_action='note',
            actionable=False,
        )
        assert 'finding_id' in null, null
        null_id = null['finding_id']

        state.set_stat('r1', 'scanned', 10)
        state.inc_stat('r1', 'scanned', 5)

        entry_before = state._state[('r1', 'memory_consolidator')]
        findings_before = list(entry_before.findings)
        stats_before = dict(entry_before.stats)
        summary_before = entry_before.summary
        created_at_before = entry_before.created_at

        # Advance the clock so a bug that re-stamps created_at is observable.
        t[0] = 5.0
        state.start_report(run_id='r1', stage='memory_consolidator', project_id='dark_factory')

        # (a) same object, unchanged contents.
        entry_after = state._state[('r1', 'memory_consolidator')]
        assert entry_after is entry_before
        assert entry_after.findings == findings_before
        assert [f.finding_id for f in entry_after.findings] == [real_id, null_id]
        assert entry_after.stats == stats_before
        assert entry_after.summary == summary_before
        assert entry_after.created_at == created_at_before

        # (b) public reads still see both findings.
        report = state.get_assembled_report('r1', 'memory_consolidator')
        assert report is not None
        assert {item['finding_id'] for item in report['flagged_items']} == {real_id, null_id}
        raw = state.get_findings_for_run('r1')
        assert {f['finding_id'] for f in raw} == {real_id, null_id}

        # (c) dedup indices stay coherent — pointers are not orphaned.
        dup_real = state.add_finding(
            run_id='r1', severity='moderate', category='memory_stale',
            description='restatement', suggested_action='act', actionable=True,
            task_id='42', flag_type='orphaned_knowledge',
        )
        assert dup_real == {
            'error': 'duplicate_finding',
            'error_type': 'ReconReportDuplicateFinding',
            'existing_finding_id': real_id,
        }
        report = state.get_assembled_report('r1', 'memory_consolidator')
        assert report is not None
        assert any(item['finding_id'] == real_id for item in report['flagged_items'])

        dup_null = state.add_finding(
            run_id='r1', severity='low', category='other',
            description='null null observation', suggested_action='note',
            actionable=False,
        )
        assert dup_null == {
            'error': 'duplicate_finding',
            'error_type': 'ReconReportDuplicateFinding',
            'existing_finding_id': null_id,
        }
        raw = state.get_findings_for_run('r1')
        assert any(f['finding_id'] == null_id for f in raw)

    def test_repeat_call_after_complete_preserves_completion_guard(self):
        state, _t = self._make_state()
        state.start_report(run_id='r1', stage='memory_consolidator', project_id='dark_factory')
        state.add_finding(
            run_id='r1', severity='low', category='cat', description='d',
            suggested_action='a', task_id='1', flag_type='f',
        )
        state.complete('r1', 'summary text')

        entry_before = state._state[('r1', 'memory_consolidator')]
        completed_at_before = entry_before.completed_at
        summary_before = entry_before.summary
        assert completed_at_before is not None

        state.start_report(run_id='r1', stage='memory_consolidator', project_id='dark_factory')

        entry_after = state._state[('r1', 'memory_consolidator')]
        assert entry_after.completed_at == completed_at_before
        assert entry_after.summary == summary_before

        result = state.add_finding(
            run_id='r1', severity='high', category='cat', description='late finding',
            suggested_action='a',
        )
        assert result['error'] == 'report_already_completed'
        assert result['error_type'] == 'ReconReportAlreadyCompleted'

        assert state.active_run_for_stage('memory_consolidator', 'dark_factory') is None


class TestStartReportPreservesExistingEntryDurable:
    """(e) DURABLE — a repeat start_report's no-op is not merely in-memory:
    the persisted row and a freshly hydrated state both still carry the
    findings filed before the repeat call.
    """

    @pytest.mark.asyncio
    async def test_repeat_call_does_not_destroy_persisted_findings(self, tmp_path):
        from fused_memory.server.recon_report import ReconReportState, _deserialize_entry
        from fused_memory.server.recon_report_store import ReconReportStore

        store = ReconReportStore(tmp_path / 'recon_report_state.db')
        store.open()
        try:
            t = [0.0]
            state = ReconReportState(ttl_seconds=300, clock=lambda: t[0], store=store)
            state.start_report(run_id='r1', stage='memory_consolidator', project_id='dark_factory')

            real = state.add_finding(
                run_id='r1', severity='moderate', category='memory_stale',
                description='real finding', suggested_action='act', actionable=True,
                task_id='42', flag_type='orphaned_knowledge',
            )
            assert 'finding_id' in real, real
            real_id = real['finding_id']

            null = state.add_finding(
                run_id='r1', severity='low', category='other',
                description='null null observation', suggested_action='note',
                actionable=False,
            )
            assert 'finding_id' in null, null
            null_id = null['finding_id']

            # Repeat call — must not destroy the persisted row.
            state.start_report(run_id='r1', stage='memory_consolidator', project_id='dark_factory')

            rows = [
                r for r in store.load_all()
                if r['run_id'] == 'r1' and r['stage'] == 'memory_consolidator'
            ]
            assert len(rows) == 1
            persisted_entry = _deserialize_entry(rows[0]['entry_json'])
            assert {f.finding_id for f in persisted_entry.findings} == {real_id, null_id}

            # A FRESH state hydrated from the SAME store still resolves dedup.
            t2 = [0.0]
            fresh_state = ReconReportState(ttl_seconds=300, clock=lambda: t2[0], store=store)
            fresh_state.hydrate_from_store()
            dup = fresh_state.add_finding(
                run_id='r1', severity='moderate', category='memory_stale',
                description='restatement', suggested_action='act', actionable=True,
                task_id='42', flag_type='orphaned_knowledge',
            )
            assert dup == {
                'error': 'duplicate_finding',
                'error_type': 'ReconReportDuplicateFinding',
                'existing_finding_id': real_id,
            }
        finally:
            store.close()


# ---------------------------------------------------------------------------
# step-3: the response shape must let a caller always distinguish a fresh
# start from a repeat — RED until step-4 enriches both return paths.
# ---------------------------------------------------------------------------


class TestStartReportResultIsUnambiguous:
    """The caller can always tell a fresh start from a repeat, both at the
    state level and through the FastMCP tool surface.
    """

    def _make_state(self):
        from fused_memory.server.recon_report import ReconReportState

        t = [0.0]
        return ReconReportState(ttl_seconds=300, clock=lambda: t[0]), t

    def test_first_call_reports_already_started_false(self):
        state, _t = self._make_state()
        result = state.start_report(run_id='r1', stage='memory_consolidator', project_id='dark_factory')
        assert result['run_id'] == 'r1'
        assert result['stage'] == 'memory_consolidator'
        assert result['already_started'] is False

    def test_repeat_call_reports_census_of_retained_entry(self):
        state, _t = self._make_state()
        state.start_report(run_id='r1', stage='memory_consolidator', project_id='dark_factory')
        state.add_finding(
            run_id='r1', severity='low', category='cat', description='d',
            suggested_action='a', task_id='1', flag_type='f',
        )

        result = state.start_report(run_id='r1', stage='memory_consolidator', project_id='dark_factory')
        assert result['run_id'] == 'r1'
        assert result['stage'] == 'memory_consolidator'
        assert result['already_started'] is True
        assert result['project_id'] == 'dark_factory'
        assert result['finding_count'] == 1
        assert result['completed'] is False
        assert isinstance(result['warning'], str) and result['warning']

    def test_repeat_call_with_completed_entry_reports_completed_true(self):
        state, _t = self._make_state()
        state.start_report(run_id='r1', stage='memory_consolidator', project_id='dark_factory')
        state.complete('r1', 'summary')

        result = state.start_report(run_id='r1', stage='memory_consolidator', project_id='dark_factory')
        assert result['already_started'] is True
        assert result['completed'] is True

    def test_divergent_project_id_retains_original_and_warns(self, caplog):
        state, _t = self._make_state()
        state.start_report(run_id='r1', stage='memory_consolidator', project_id='dark_factory')

        with caplog.at_level(logging.WARNING, logger='fused_memory.server.recon_report'):
            result = state.start_report(
                run_id='r1', stage='memory_consolidator', project_id='other_project',
            )

        assert result['already_started'] is True
        assert result['project_id'] == 'dark_factory'  # retained, NOT overwritten
        assert 'other_project' in result['warning']

        entry = state._state[('r1', 'memory_consolidator')]
        assert entry.project_id == 'dark_factory'  # never overwritten on the entry either

        divergence_warnings = [
            r for r in caplog.records
            if 'dark_factory' in r.getMessage() and 'other_project' in r.getMessage()
        ]
        assert divergence_warnings, [r.getMessage() for r in caplog.records]

        # Fully mirrors complete()'s different-summary handling: the
        # divergence is recorded ON THE ENTRY, not just the transient
        # response/log, so get_assembled_report's summary_warnings surfaces
        # it to every downstream reader.
        assert len(entry.summary_warnings) == 1
        assert 'other_project' in entry.summary_warnings[0]
        assembled = state.get_assembled_report('r1', 'memory_consolidator')
        assert assembled is not None
        assert assembled['summary_warnings'] == entry.summary_warnings

    def test_divergent_project_id_persists_warning_on_entry(self, tmp_path):
        """The project_id-divergence branch is the one exception to the
        pure no-op repeat: it must persist (unlike a same-project_id
        no-op repeat, which stays a zero-store-write no-op — see
        TestStartReportActivePointerAndPersistence.test_pure_noop_repeat_triggers_zero_store_writes).
        """
        from fused_memory.server.recon_report import ReconReportState, _deserialize_entry
        from fused_memory.server.recon_report_store import ReconReportStore

        store = ReconReportStore(tmp_path / 'recon_report_state.db')
        store.open()
        try:
            t = [0.0]
            state = ReconReportState(ttl_seconds=300, clock=lambda: t[0], store=store)
            state.start_report(run_id='r1', stage='memory_consolidator', project_id='dark_factory')

            state.start_report(
                run_id='r1', stage='memory_consolidator', project_id='other_project',
            )

            rows = [
                r for r in store.load_all()
                if r['run_id'] == 'r1' and r['stage'] == 'memory_consolidator'
            ]
            assert len(rows) == 1
            persisted_entry = _deserialize_entry(rows[0]['entry_json'])
            assert persisted_entry.project_id == 'dark_factory'  # retained
            assert len(persisted_entry.summary_warnings) == 1
            assert 'other_project' in persisted_entry.summary_warnings[0]
        finally:
            store.close()

    @pytest.mark.asyncio
    async def test_fastmcp_surface_reports_same_shapes(self):
        from fused_memory.server.recon_report import ReconReportState, create_recon_report_server

        t = [0.0]
        state = ReconReportState(ttl_seconds=300, clock=lambda: t[0])
        mcp = create_recon_report_server(state)
        tm = mcp._tool_manager

        first = await tm.call_tool('start_report', {
            'run_id': 'r1', 'stage': 'memory_consolidator', 'project_id': 'dark_factory',
        })
        assert first['already_started'] is False

        add_result = await tm.call_tool('add_finding', {
            'run_id': 'r1', 'severity': 'low', 'category': 'cat',
            'description': 'd', 'suggested_action': 'a', 'task_id': '1', 'flag_type': 'f',
        })
        assert 'finding_id' in add_result

        second = await tm.call_tool('start_report', {
            'run_id': 'r1', 'stage': 'memory_consolidator', 'project_id': 'dark_factory',
        })
        assert second['already_started'] is True
        assert second['finding_count'] == 1
        assert second['completed'] is False

        report = state.get_assembled_report('r1', 'memory_consolidator')
        assert report is not None
        assert len(report['flagged_items']) == 1


# ---------------------------------------------------------------------------
# step-5: the _active pointer must never be stolen by a stray repeat call
# naming an earlier stage, but MUST be repaired when the run has no active
# stage at all — RED (repair half) until step-6.
# ---------------------------------------------------------------------------


class _CountingStore:
    """Wraps a real ReconReportStore, counting upsert_many calls.

    Delegates every method to the wrapped store so persistence still
    behaves exactly like the real thing (readback via load_all reflects
    every write) — only upsert_many is instrumented, since that is the
    write call under test: a pure no-op repeat start_report must trigger
    ZERO of them, while the _active-repair repeat path must trigger
    exactly one. Mirrors the _TripwireStore shape at
    test_recon_report_store.py:896-931, counting instead of raising.
    """

    def __init__(self, inner):
        self._inner = inner
        self.upsert_many_calls = 0

    def open(self):
        self._inner.open()

    def close(self):
        self._inner.close()

    def upsert_entry(self, **kwargs):
        return self._inner.upsert_entry(**kwargs)

    def upsert_many(self, rows):
        self.upsert_many_calls += 1
        self._inner.upsert_many(rows)

    def delete_run(self, run_id):
        self._inner.delete_run(run_id)

    def load_all(self):
        return self._inner.load_all()


class TestStartReportActivePointerAndPersistence:
    """_active pointer semantics and store side-effects of a repeat call."""

    def _make_state(self):
        from fused_memory.server.recon_report import ReconReportState

        t = [0.0]
        return ReconReportState(ttl_seconds=300, clock=lambda: t[0]), t

    def test_repeat_call_never_steals_active_pointer_from_different_stage(self):
        state, _t = self._make_state()
        state.start_report(run_id='r1', stage='stage_one', project_id='dark_factory')
        finding_one = state.add_finding(
            run_id='r1', severity='low', category='cat', description='d1',
            suggested_action='a', task_id='1', flag_type='f',
        )
        assert 'finding_id' in finding_one, finding_one
        state.complete('r1', 'stage one summary')

        state.start_report(run_id='r1', stage='stage_two', project_id='dark_factory')
        assert state._active['r1'] == 'stage_two'

        before = state.get_assembled_report('r1', 'stage_one')

        # Stray repeat call naming the EARLIER, now-inactive stage.
        result = state.start_report(run_id='r1', stage='stage_one', project_id='dark_factory')
        assert result['already_started'] is True
        assert state._active['r1'] == 'stage_two'  # pointer not stolen

        finding_two = state.add_finding(
            run_id='r1', severity='low', category='cat', description='d2',
            suggested_action='a', task_id='2', flag_type='f',
        )
        assert 'finding_id' in finding_two, finding_two
        # Lands in stage_two's entry (the live one), not stage_one's.
        assert len(state._state[('r1', 'stage_two')].findings) == 1
        assert len(state._state[('r1', 'stage_one')].findings) == 1

        complete_result = state.complete('r1', 'stage two summary')
        assert complete_result['flagged_count'] == 1

        after = state.get_assembled_report('r1', 'stage_one')
        assert after == before  # untouched by the stray call

    def test_repeat_call_repairs_missing_active_pointer(self):
        state, _t = self._make_state()
        state.start_report(run_id='r1', stage='s1', project_id='dark_factory')
        finding = state.add_finding(
            run_id='r1', severity='low', category='cat', description='d',
            suggested_action='a', task_id='1', flag_type='f',
        )
        assert 'finding_id' in finding, finding
        finding_id = finding['finding_id']

        # Simulates tick() evicting the active stage, or a hydrate whose
        # persisted rows carried no is_active=True row for this run.
        del state._active['r1']
        blocked = state.add_finding(
            run_id='r1', severity='low', category='cat', description='d2',
            suggested_action='a', task_id='2', flag_type='f',
        )
        assert blocked == {'error': 'run_id_unknown', 'error_type': 'ReconReportRunUnknown'}

        result = state.start_report(run_id='r1', stage='s1', project_id='dark_factory')
        assert result['already_started'] is True
        assert state._active['r1'] == 's1'  # pointer restored

        resumed = state.add_finding(
            run_id='r1', severity='low', category='cat', description='d3',
            suggested_action='a', task_id='3', flag_type='f',
        )
        assert 'finding_id' in resumed, resumed
        # Lands in the SAME preserved entry — earlier finding still present.
        entry = state._state[('r1', 's1')]
        assert {f.finding_id for f in entry.findings} == {finding_id, resumed['finding_id']}

    def test_pure_noop_repeat_triggers_zero_store_writes(self, tmp_path):
        from fused_memory.server.recon_report import ReconReportState
        from fused_memory.server.recon_report_store import ReconReportStore

        real_store = ReconReportStore(tmp_path / 'recon_report_state.db')
        real_store.open()
        try:
            store = _CountingStore(real_store)
            t = [0.0]
            state = ReconReportState(ttl_seconds=300, clock=lambda: t[0], store=store)
            state.start_report(run_id='r1', stage='s1', project_id='dark_factory')
            state.add_finding(
                run_id='r1', severity='low', category='cat', description='d',
                suggested_action='a', task_id='1', flag_type='f',
            )
            calls_before = store.upsert_many_calls
            assert calls_before > 0  # sanity: setup really did write

            # Pointer is intact (points at s1) — a pure no-op repeat.
            result = state.start_report(run_id='r1', stage='s1', project_id='dark_factory')
            assert result['already_started'] is True

            assert store.upsert_many_calls == calls_before  # zero NEW writes
        finally:
            real_store.close()

    def test_repair_path_persists_and_marks_row_active(self, tmp_path):
        from fused_memory.server.recon_report import ReconReportState
        from fused_memory.server.recon_report_store import ReconReportStore

        real_store = ReconReportStore(tmp_path / 'recon_report_state.db')
        real_store.open()
        try:
            store = _CountingStore(real_store)
            t = [0.0]
            state = ReconReportState(ttl_seconds=300, clock=lambda: t[0], store=store)
            state.start_report(run_id='r1', stage='s1', project_id='dark_factory')
            state.add_finding(
                run_id='r1', severity='low', category='cat', description='d',
                suggested_action='a', task_id='1', flag_type='f',
            )
            del state._active['r1']
            calls_before = store.upsert_many_calls

            result = state.start_report(run_id='r1', stage='s1', project_id='dark_factory')
            assert result['already_started'] is True
            assert store.upsert_many_calls == calls_before + 1  # exactly one repair write

            rows = [
                r for r in real_store.load_all()
                if r['run_id'] == 'r1' and r['stage'] == 's1'
            ]
            assert len(rows) == 1
            assert rows[0]['is_active'] is True
        finally:
            real_store.close()

    def test_repeat_call_repairs_pointer_dropped_by_tick_eviction(self):
        """Organic producer #1 of the missing-pointer state (the tests above
        manufacture it via `del state._active[...]`): `tick()` itself drops
        the pointer when the run's ACTIVE stage is the one that evicts,
        while an in-progress SIBLING entry of the same run survives — the
        scenario the docstring's repair-rationale paragraph cites.
        """
        state, t = self._make_state()
        state.start_report(run_id='r1', stage='s1', project_id='dark_factory')
        finding = state.add_finding(
            run_id='r1', severity='low', category='cat', description='d',
            suggested_action='a', task_id='1', flag_type='f',
        )
        assert 'finding_id' in finding, finding
        finding_id = finding['finding_id']

        # s2 becomes active; s1 stays resident and in-progress (immortal).
        state.start_report(run_id='r1', stage='s2', project_id='dark_factory')
        assert state._active['r1'] == 's2'

        # complete() resolves via _active — completes s2, not s1.
        state.complete('r1', 's2 summary')
        assert state._active['r1'] == 's2'

        t[0] = 301.0  # past TTL for s2's completed_at (0.0)
        evicted = state.tick()
        assert evicted == 1  # only s2 (completed); s1 is immortal in-progress
        assert ('r1', 's2') not in state._state
        assert ('r1', 's1') in state._state  # sibling survives
        assert 'r1' not in state._active  # tick() cleared the pointer it owned

        blocked = state.add_finding(
            run_id='r1', severity='low', category='cat', description='d2',
            suggested_action='a', task_id='2', flag_type='f',
        )
        assert blocked == {'error': 'run_id_unknown', 'error_type': 'ReconReportRunUnknown'}

        result = state.start_report(run_id='r1', stage='s1', project_id='dark_factory')
        assert result['already_started'] is True
        assert state._active['r1'] == 's1'  # pointer restored onto the surviving sibling

        resumed = state.add_finding(
            run_id='r1', severity='low', category='cat', description='d3',
            suggested_action='a', task_id='3', flag_type='f',
        )
        assert 'finding_id' in resumed, resumed
        entry = state._state[('r1', 's1')]
        assert {f.finding_id for f in entry.findings} == {finding_id, resumed['finding_id']}

    def test_repeat_call_repairs_pointer_missing_after_hydrate_with_no_active_row(self, tmp_path):
        """Organic producer #2: `hydrate_from_store()` only sets `_active`
        for a row persisted with `is_active=True`. A run whose persisted
        rows carry no such row (e.g. because the pointer was already
        missing at the last `_persist_run` for that run) hydrates with
        `_state` populated but `_active` unset for that run_id — a second,
        independent real-world path into the same repair branch as the
        tick()-eviction producer above.
        """
        from fused_memory.server.recon_report import ReconReportState
        from fused_memory.server.recon_report_store import ReconReportStore

        store = ReconReportStore(tmp_path / 'recon_report_state.db')
        store.open()
        try:
            t = [0.0]
            state = ReconReportState(ttl_seconds=300, clock=lambda: t[0], store=store)
            state.start_report(run_id='r1', stage='s1', project_id='dark_factory')
            finding = state.add_finding(
                run_id='r1', severity='low', category='cat', description='d',
                suggested_action='a', task_id='1', flag_type='f',
            )
            assert 'finding_id' in finding, finding
            finding_id = finding['finding_id']

            # Drop the pointer and persist through it, so the STORE row
            # itself carries is_active=False for every (r1, *) row.
            del state._active['r1']
            state._persist_run('r1')

            rows = [r for r in store.load_all() if r['run_id'] == 'r1']
            assert rows and all(not r['is_active'] for r in rows)

            # A FRESH state hydrating from that store never sets _active['r1'].
            t2 = [0.0]
            fresh_state = ReconReportState(ttl_seconds=300, clock=lambda: t2[0], store=store)
            fresh_state.hydrate_from_store()
            assert ('r1', 's1') in fresh_state._state
            assert 'r1' not in fresh_state._active

            blocked = fresh_state.add_finding(
                run_id='r1', severity='low', category='cat', description='d2',
                suggested_action='a', task_id='2', flag_type='f',
            )
            assert blocked == {'error': 'run_id_unknown', 'error_type': 'ReconReportRunUnknown'}

            result = fresh_state.start_report(run_id='r1', stage='s1', project_id='dark_factory')
            assert result['already_started'] is True
            assert fresh_state._active['r1'] == 's1'

            resumed = fresh_state.add_finding(
                run_id='r1', severity='low', category='cat', description='d3',
                suggested_action='a', task_id='3', flag_type='f',
            )
            assert 'finding_id' in resumed, resumed
            entry = fresh_state._state[('r1', 's1')]
            assert {f.finding_id for f in entry.findings} == {finding_id, resumed['finding_id']}
        finally:
            store.close()

    def test_ttl_evicted_entry_takes_fresh_create_path_not_already_started(self):
        """Boundary pin for the guard's documented scope edge: once an entry
        has aged out of `_state` via `tick()`, a call naming that same
        (run_id, stage) no longer matches the repeat branch at all — it is
        a legal fresh create and reports `already_started: False`, not
        True. This is the guard's explicit scope boundary (see
        start_report's docstring), not a regression: closing it needs a
        known-stage vocabulary the guard deliberately does not have, and is
        also where a stray call naming an evicted-but-still-active-pointer
        stage could steal `_active` from a live sibling stage (out of scope
        here; tracked in the docstring).
        """
        state, t = self._make_state()
        state.start_report(run_id='r1', stage='s1', project_id='dark_factory')
        state.complete('r1', 'done')
        t[0] = 301.0
        evicted = state.tick()
        assert evicted == 1
        assert ('r1', 's1') not in state._state
        assert 'r1' not in state._active

        result = state.start_report(run_id='r1', stage='s1', project_id='dark_factory')
        assert result['already_started'] is False
