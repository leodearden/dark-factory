"""Tests for the generic per-cycle summary pool-cap core
(fused_memory.reconciliation.summary_pool).

Extracted from task_knowledge_sync.py's Stage-2-specific
``_enforce_stage2_summary_pool_cap`` / ``_pretrim_stage2_summary_pool``
(task 1657 + trim-then-write, task 1831) into a shared, parametrized core so
Stage 1 (memory_consolidator) can enforce an equivalent pool cap without
duplicating the ~90-line async GC logic (task 1942).

Mirrors fused-memory/tests/test_stages.py::TestEnforceStage2SummaryPoolCap(+Resilience)
and ::TestPretrimStage2SummaryPool. Several cases are parametrized over
recon_pool/trim_source (stage1 vs stage2 pool names) specifically to prove the
core is generic and not accidentally hardcoded to the Stage 2 pool.
"""

import json
import logging
from datetime import UTC, datetime
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest
import pytest_asyncio

from fused_memory.models.reconciliation import StageId, StageReport
from fused_memory.reconciliation.recon_ledger import ReconLedgerStore
from fused_memory.reconciliation.summary_pool import (
    enforce_summary_pool_cap,
    write_cycle_summary,
)

_LOGGER = 'fused_memory.reconciliation.summary_pool'

# Parametrize recon_pool/trim_source across both known pools to prove genericity.
_POOL_PARAMS = pytest.mark.parametrize(
    'recon_pool,trim_source',
    [
        ('stage2_cycle_summary', 'stage2_cycle_summary_trim'),
        ('stage1_cycle_summary', 'stage1_cycle_summary_trim'),
    ],
    ids=['stage2', 'stage1'],
)

class TestEnforceSummaryPoolCap:
    """enforce_summary_pool_cap trims oldest pool members to the passed cap."""

    @_POOL_PARAMS
    @pytest.mark.asyncio
    async def test_deletes_two_oldest_when_four_members_exist(self, recon_pool, trim_source):
        members = [
            {'id': 'oldest', 'created_at': '2026-01-01T00:00:00+00:00', 'metadata': {}},
            {'id': 'second', 'created_at': '2026-02-01T00:00:00+00:00', 'metadata': {}},
            {'id': 'third',  'created_at': '2026-03-01T00:00:00+00:00', 'metadata': {}},
            {'id': 'newest', 'created_at': '2026-04-01T00:00:00+00:00', 'metadata': {}},
        ]
        memory_service = AsyncMock()
        memory_service.get_memories_by_metadata = AsyncMock(return_value=members)
        memory_service.delete_memory = AsyncMock(return_value=None)

        result = await enforce_summary_pool_cap(
            memory_service,
            project_id='dark_factory',
            run_id='run-123',
            recon_pool=recon_pool,
            trim_source=trim_source,
            cap=2,
        )

        # Two oldest deleted
        assert result == 2
        assert memory_service.delete_memory.await_count == 2

        deleted_ids = {c.kwargs['memory_id'] for c in memory_service.delete_memory.call_args_list}
        assert deleted_ids == {'oldest', 'second'}
        # Two newest kept
        assert 'third' not in deleted_ids
        assert 'newest' not in deleted_ids

    @_POOL_PARAMS
    @pytest.mark.asyncio
    async def test_delete_memory_called_with_correct_kwargs(self, recon_pool, trim_source):
        members = [
            {'id': 'm1', 'created_at': '2026-01-01T00:00:00+00:00', 'metadata': {}},
            {'id': 'm2', 'created_at': '2026-02-01T00:00:00+00:00', 'metadata': {}},
            {'id': 'm3', 'created_at': '2026-03-01T00:00:00+00:00', 'metadata': {}},
        ]
        memory_service = AsyncMock()
        memory_service.get_memories_by_metadata = AsyncMock(return_value=members)
        memory_service.delete_memory = AsyncMock(return_value=None)

        await enforce_summary_pool_cap(
            memory_service,
            project_id='dark_factory',
            run_id='run-xyz',
            recon_pool=recon_pool,
            trim_source=trim_source,
            cap=2,
        )

        for call in memory_service.delete_memory.call_args_list:
            kwargs = call.kwargs
            assert kwargs.get('store') == 'mem0'
            assert kwargs.get('project_id') == 'dark_factory'
            assert kwargs.get('causation_id') == 'run-xyz'
            assert kwargs.get('_source') == trim_source

    @_POOL_PARAMS
    @pytest.mark.asyncio
    async def test_get_memories_called_with_correct_filter_and_project(self, recon_pool, trim_source):
        memory_service = AsyncMock()
        memory_service.get_memories_by_metadata = AsyncMock(return_value=[])
        memory_service.delete_memory = AsyncMock(return_value=None)

        await enforce_summary_pool_cap(
            memory_service,
            project_id='my_project',
            run_id='run-abc',
            recon_pool=recon_pool,
            trim_source=trim_source,
            cap=2,
        )

        memory_service.get_memories_by_metadata.assert_awaited_once()
        call = memory_service.get_memories_by_metadata.call_args
        kwargs = call.kwargs
        assert kwargs.get('project_id') == 'my_project'
        filters = kwargs.get('filters') or {}
        assert filters.get('recon_pool') == recon_pool

    @_POOL_PARAMS
    @pytest.mark.asyncio
    async def test_at_cap_deletes_nothing_returns_zero(self, recon_pool, trim_source):
        # Exactly 2 members (== cap)
        members = [
            {'id': 'a', 'created_at': '2026-01-01T00:00:00+00:00', 'metadata': {}},
            {'id': 'b', 'created_at': '2026-02-01T00:00:00+00:00', 'metadata': {}},
        ]
        memory_service = AsyncMock()
        memory_service.get_memories_by_metadata = AsyncMock(return_value=members)
        memory_service.delete_memory = AsyncMock(return_value=None)

        result = await enforce_summary_pool_cap(
            memory_service,
            project_id='dark_factory',
            run_id='run-abc',
            recon_pool=recon_pool,
            trim_source=trim_source,
            cap=2,
        )

        assert result == 0
        memory_service.delete_memory.assert_not_awaited()

    @_POOL_PARAMS
    @pytest.mark.asyncio
    async def test_below_cap_deletes_nothing_returns_zero(self, recon_pool, trim_source):
        members = [
            {'id': 'only', 'created_at': '2026-01-01T00:00:00+00:00', 'metadata': {}},
        ]
        memory_service = AsyncMock()
        memory_service.get_memories_by_metadata = AsyncMock(return_value=members)
        memory_service.delete_memory = AsyncMock(return_value=None)

        result = await enforce_summary_pool_cap(
            memory_service,
            project_id='dark_factory',
            run_id='run-abc',
            recon_pool=recon_pool,
            trim_source=trim_source,
            cap=2,
        )

        assert result == 0
        memory_service.delete_memory.assert_not_awaited()


class TestEnforceSummaryPoolCapResilience:
    """enforce_summary_pool_cap resilience: enumeration failure, partial delete failure,
    and created_at ordering edge cases."""

    @pytest.mark.asyncio
    async def test_enumeration_failure_returns_zero_and_logs_warning(self, caplog):
        """When get_memories_by_metadata raises, returns 0, does NOT raise, logs WARNING."""
        memory_service = AsyncMock()
        memory_service.get_memories_by_metadata = AsyncMock(
            side_effect=RuntimeError('qdrant gone')
        )
        memory_service.delete_memory = AsyncMock(return_value=None)

        with caplog.at_level(logging.WARNING, logger=_LOGGER):
            result = await enforce_summary_pool_cap(
                memory_service,
                project_id='dark_factory',
                run_id='run-fail',
                recon_pool='stage1_cycle_summary',
                trim_source='stage1_cycle_summary_trim',
                cap=2,
            )

        assert result == 0
        memory_service.delete_memory.assert_not_awaited()
        warning_records = [r for r in caplog.records if r.levelno >= logging.WARNING]
        assert len(warning_records) >= 1

    @pytest.mark.asyncio
    async def test_partial_delete_failure_excluded_from_count_logs_warning(self, caplog):
        """If one delete raises, that delete is excluded from count; others counted; logs WARNING."""
        # 4 members → 2 to delete (oldest two)
        members = [
            {'id': 'oldest', 'created_at': '2026-01-01T00:00:00+00:00', 'metadata': {}},
            {'id': 'second', 'created_at': '2026-02-01T00:00:00+00:00', 'metadata': {}},
            {'id': 'third',  'created_at': '2026-03-01T00:00:00+00:00', 'metadata': {}},
            {'id': 'newest', 'created_at': '2026-04-01T00:00:00+00:00', 'metadata': {}},
        ]
        memory_service = AsyncMock()
        memory_service.get_memories_by_metadata = AsyncMock(return_value=members)
        # First delete (oldest) succeeds; second delete (second) fails
        memory_service.delete_memory = AsyncMock(
            side_effect=[None, RuntimeError('delete failed')]
        )

        with caplog.at_level(logging.WARNING, logger=_LOGGER):
            result = await enforce_summary_pool_cap(
                memory_service,
                project_id='dark_factory',
                run_id='run-partial',
                recon_pool='stage1_cycle_summary',
                trim_source='stage1_cycle_summary_trim',
                cap=2,
            )

        # Does not raise; only 1 success counted
        assert result == 1
        warning_records = [r for r in caplog.records if r.levelno >= logging.WARNING]
        assert len(warning_records) >= 1

    @pytest.mark.asyncio
    async def test_missing_created_at_sorted_last_and_kept(self):
        """Members with None/missing created_at sort LAST (treated as newest) and are kept."""
        # 4 members: 2 datable + 2 undatable; cap=2 → delete 2 oldest
        # The 2 undatable should be kept; the 2 datable should be the ones deleted
        members = [
            {'id': 'datable-old',  'created_at': '2026-01-01T00:00:00+00:00', 'metadata': {}},
            {'id': 'datable-new',  'created_at': '2026-02-01T00:00:00+00:00', 'metadata': {}},
            {'id': 'no-date-1',    'created_at': None,                          'metadata': {}},
            {'id': 'no-date-2',    'created_at': None,                          'metadata': {}},
        ]
        memory_service = AsyncMock()
        memory_service.get_memories_by_metadata = AsyncMock(return_value=members)
        memory_service.delete_memory = AsyncMock(return_value=None)

        result = await enforce_summary_pool_cap(
            memory_service,
            project_id='dark_factory',
            run_id='run-order',
            recon_pool='stage1_cycle_summary',
            trim_source='stage1_cycle_summary_trim',
            cap=2,
        )

        assert result == 2
        deleted_ids = {c.kwargs['memory_id'] for c in memory_service.delete_memory.call_args_list}
        assert deleted_ids == {'datable-old', 'datable-new'}
        assert 'no-date-1' not in deleted_ids
        assert 'no-date-2' not in deleted_ids

    @pytest.mark.asyncio
    async def test_unparseable_created_at_sorted_last_and_kept(self):
        """Members with unparseable created_at sort LAST (treated as newest) and are kept."""
        members = [
            {'id': 'datable',      'created_at': '2026-01-01T00:00:00+00:00', 'metadata': {}},
            {'id': 'also-datable', 'created_at': '2026-02-01T00:00:00+00:00', 'metadata': {}},
            {'id': 'bad-date-1',   'created_at': 'not-a-date',                'metadata': {}},
            {'id': 'bad-date-2',   'created_at': 12345,                       'metadata': {}},
        ]
        memory_service = AsyncMock()
        memory_service.get_memories_by_metadata = AsyncMock(return_value=members)
        memory_service.delete_memory = AsyncMock(return_value=None)

        result = await enforce_summary_pool_cap(
            memory_service,
            project_id='dark_factory',
            run_id='run-order2',
            recon_pool='stage1_cycle_summary',
            trim_source='stage1_cycle_summary_trim',
            cap=2,
        )

        assert result == 2
        deleted_ids = {c.kwargs['memory_id'] for c in memory_service.delete_memory.call_args_list}
        assert deleted_ids == {'datable', 'also-datable'}
        assert 'bad-date-1' not in deleted_ids
        assert 'bad-date-2' not in deleted_ids


class TestWriteCycleSummaryLedgerWrite:
    """write_cycle_summary (task 2229 W5-λ) writes exactly ONE authoritative
    ``cycle_summary`` ledger row from a ``StageReport`` — the deterministic
    Python replacement for the LLM-driven per-cycle summary write (PRD
    plans/recon-reliability-prd.md §10, boundary test D1).

    This class covers only the AUTHORITATIVE ledger write. The best-effort
    Mem0 mirror + pool-cap trim are covered separately in
    ``TestWriteCycleSummaryMirrorAndTrim`` (step-03/04).
    """

    @pytest_asyncio.fixture
    async def ledger_store(self, tmp_path):
        s = ReconLedgerStore(tmp_path / 'reconciliation.db')
        await s.initialize()
        yield s
        await s.close()

    def _report(self, **overrides) -> StageReport:
        defaults: dict[str, Any] = dict(
            stage=StageId.task_knowledge_sync,
            started_at=datetime(2026, 7, 10, 11, 0, 0, tzinfo=UTC),
            completed_at=datetime(2026, 7, 10, 11, 5, 0, tzinfo=UTC),
            items_flagged=[{'description': 'a'}, {'description': 'b'}],
            stats={'stage2_stage1_dups_suppressed': 1},
            llm_calls=4,
            tokens_used=1234,
        )
        defaults.update(overrides)
        return StageReport(**defaults)

    @pytest.mark.asyncio
    async def test_writes_one_authoritative_ledger_row_from_report(self, ledger_store):
        memory_service = AsyncMock()
        memory_service.recon_ledger = ledger_store
        report = self._report()

        result = await write_cycle_summary(
            memory_service,
            'dark_factory',
            report,
            'run-abc',
            stage='task_knowledge_sync',
            recon_pool='stage2_cycle_summary',
            trim_source='stage2_cycle_summary_trim',
            cap=2,
        )

        assert result is True

        record = await ledger_store.get_by_identity(
            'dark_factory', 'cycle_summary', flag_type='task_knowledge_sync', run_id='run-abc',
        )
        assert record is not None
        assert record.record_kind == 'cycle_summary'
        assert record.task_id == ''
        assert record.flag_type == 'task_knowledge_sync'
        assert record.run_id == 'run-abc'
        assert record.state == 'active'

        payload = json.loads(record.payload_json)
        assert payload['stage'] == 'task_knowledge_sync'
        assert payload['run_id'] == 'run-abc'
        assert payload['stats'] == {'stage2_stage1_dups_suppressed': 1}
        assert payload['items_flagged_count'] == 2
        assert payload['llm_calls'] == 4
        assert payload['tokens_used'] == 1234

    @pytest.mark.asyncio
    async def test_idempotent_repeat_call_same_identity_keeps_one_row_last_write_wins(
        self, ledger_store,
    ):
        """Calling twice with the same (stage, run_id) leaves exactly one row
        in the ledger (upsert PK semantics), carrying the second call's
        payload."""
        memory_service = AsyncMock()
        memory_service.recon_ledger = ledger_store

        await write_cycle_summary(
            memory_service,
            'dark_factory',
            self._report(),
            'run-dup',
            stage='task_knowledge_sync',
            recon_pool='stage2_cycle_summary',
            trim_source='stage2_cycle_summary_trim',
            cap=2,
        )
        result = await write_cycle_summary(
            memory_service,
            'dark_factory',
            self._report(
                items_flagged=[{'description': 'c'}],
                stats={'different': True},
                llm_calls=9,
                tokens_used=999,
            ),
            'run-dup',
            stage='task_knowledge_sync',
            recon_pool='stage2_cycle_summary',
            trim_source='stage2_cycle_summary_trim',
            cap=2,
        )

        assert result is True

        db = ledger_store._db
        cursor = await db.execute(
            "SELECT COUNT(*) FROM recon_ledger WHERE project_id = ? AND record_kind = 'cycle_summary' "
            'AND run_id = ?',
            ('dark_factory', 'run-dup'),
        )
        row = await cursor.fetchone()
        assert row[0] == 1

        record = await ledger_store.get_by_identity(
            'dark_factory', 'cycle_summary', flag_type='task_knowledge_sync', run_id='run-dup',
        )
        payload = json.loads(record.payload_json)
        assert payload['items_flagged_count'] == 1
        assert payload['llm_calls'] == 9
        assert payload['tokens_used'] == 999

    @pytest.mark.asyncio
    async def test_no_ledger_wired_is_noop_returns_false_does_not_raise(self):
        """Scoped to the AUTHORITATIVE ledger write only: no ledger means no
        ledger row and a ``False`` return. This does NOT mean the whole
        function is a no-op — the best-effort Mem0 mirror and pool-cap trim
        still run in this scenario; see
        ``TestWriteCycleSummaryMirrorAndTrim.test_mirror_and_trim_still_run_when_no_ledger_wired``."""
        memory_service = AsyncMock()
        memory_service.recon_ledger = None

        result = await write_cycle_summary(
            memory_service,
            'dark_factory',
            self._report(),
            'run-none',
            stage='task_knowledge_sync',
            recon_pool='stage2_cycle_summary',
            trim_source='stage2_cycle_summary_trim',
            cap=2,
        )

        assert result is False


class TestWriteCycleSummaryRemediationFlag:
    """``write_cycle_summary``'s optional ``remediation`` flag (task 2652)
    stamps the authoritative ledger row's ``payload_json`` and the
    best-effort Mem0 mirror ``metadata``, so
    ``MemoryService.get_cycle_summary_presence`` can later disambiguate a
    Stage-2-only remediation run's expected missing Stage 1 cycle_summary
    from a genuine Stage 1 write failure (see ``prompts/stage3.py``'s
    Stage-2-only remediation run exception).

    Default ``remediation=False`` stamps an explicit ``False`` (not an
    absent key) into both the payload and the mirror metadata — the
    anti-inversion case below — so only rows written before this change lack
    the key entirely (parsed as ``None`` by ``get_cycle_summary_presence``).
    """

    @pytest_asyncio.fixture
    async def ledger_store(self, tmp_path):
        s = ReconLedgerStore(tmp_path / 'reconciliation.db')
        await s.initialize()
        yield s
        await s.close()

    def _report(self, **overrides) -> StageReport:
        defaults: dict[str, Any] = dict(
            stage=StageId.task_knowledge_sync,
            started_at=datetime(2026, 7, 10, 11, 0, 0, tzinfo=UTC),
            completed_at=datetime(2026, 7, 10, 11, 5, 0, tzinfo=UTC),
            items_flagged=[{'description': 'a'}],
            stats={},
            llm_calls=1,
            tokens_used=10,
        )
        defaults.update(overrides)
        return StageReport(**defaults)

    @pytest.mark.asyncio
    async def test_remediation_true_stamps_ledger_payload_and_mirror_metadata(self, ledger_store):
        memory_service = AsyncMock()
        memory_service.recon_ledger = ledger_store
        memory_service.add_system_record = AsyncMock(
            return_value=SimpleNamespace(memory_ids=['m1']),
        )

        result = await write_cycle_summary(
            memory_service,
            'dark_factory',
            self._report(),
            'run-remediation-true',
            stage='task_knowledge_sync',
            recon_pool='stage2_cycle_summary',
            trim_source='stage2_cycle_summary_trim',
            cap=2,
            remediation=True,
        )

        assert result is True

        record = await ledger_store.get_by_identity(
            'dark_factory',
            'cycle_summary',
            flag_type='task_knowledge_sync',
            run_id='run-remediation-true',
        )
        assert record is not None
        payload = json.loads(record.payload_json)
        assert payload['remediation'] is True

        memory_service.add_system_record.assert_awaited_once()
        metadata = memory_service.add_system_record.call_args.kwargs.get('metadata') or {}
        assert metadata.get('remediation') is True

    @pytest.mark.asyncio
    async def test_remediation_default_false_stamps_ledger_payload_and_mirror_metadata(
        self, ledger_store,
    ):
        """Anti-inversion: the default call (no ``remediation`` kwarg) yields
        payload ``remediation=False`` and mirror metadata
        ``['remediation'] is False`` — an explicit marker, not an absent
        key."""
        memory_service = AsyncMock()
        memory_service.recon_ledger = ledger_store
        memory_service.add_system_record = AsyncMock(
            return_value=SimpleNamespace(memory_ids=['m1']),
        )

        result = await write_cycle_summary(
            memory_service,
            'dark_factory',
            self._report(),
            'run-remediation-false',
            stage='task_knowledge_sync',
            recon_pool='stage2_cycle_summary',
            trim_source='stage2_cycle_summary_trim',
            cap=2,
        )

        assert result is True

        record = await ledger_store.get_by_identity(
            'dark_factory',
            'cycle_summary',
            flag_type='task_knowledge_sync',
            run_id='run-remediation-false',
        )
        assert record is not None
        payload = json.loads(record.payload_json)
        assert payload['remediation'] is False

        memory_service.add_system_record.assert_awaited_once()
        metadata = memory_service.add_system_record.call_args.kwargs.get('metadata') or {}
        assert metadata.get('remediation') is False


class TestWriteCycleSummaryMirrorAndTrim:
    """write_cycle_summary's best-effort Mem0 mirror (``add_system_record``)
    and pool-cap trim (``enforce_summary_pool_cap``) — task 2229 W5-λ step-03.

    Both are best-effort: they run UNCONDITIONALLY — regardless of whether a
    ledger is wired at all (see ``test_mirror_and_trim_still_run_when_no_ledger_wired``,
    reviewer finding robustness, amendment pass) and regardless of whether
    the authoritative ledger upsert itself succeeded — and neither can
    change the return value or propagate an exception out of
    ``write_cycle_summary`` (which reflects only the authoritative ledger
    write's own outcome).
    """

    @pytest_asyncio.fixture
    async def ledger_store(self, tmp_path):
        s = ReconLedgerStore(tmp_path / 'reconciliation.db')
        await s.initialize()
        yield s
        await s.close()

    def _report(self, **overrides) -> StageReport:
        defaults: dict[str, Any] = dict(
            stage=StageId.task_knowledge_sync,
            started_at=datetime(2026, 7, 10, 11, 0, 0, tzinfo=UTC),
            completed_at=datetime(2026, 7, 10, 11, 5, 0, tzinfo=UTC),
            items_flagged=[{'description': 'a'}],
            stats={},
            llm_calls=1,
            tokens_used=10,
        )
        defaults.update(overrides)
        return StageReport(**defaults)

    @pytest.mark.asyncio
    async def test_mirror_write_called_once_with_expected_shape(self, ledger_store):
        memory_service = AsyncMock()
        memory_service.recon_ledger = ledger_store
        memory_service.add_system_record = AsyncMock(
            return_value=SimpleNamespace(memory_ids=['m1']),
        )

        await write_cycle_summary(
            memory_service,
            'dark_factory',
            self._report(),
            'run-mirror',
            stage='task_knowledge_sync',
            recon_pool='stage2_cycle_summary',
            trim_source='stage2_cycle_summary_trim',
            cap=2,
        )

        memory_service.add_system_record.assert_awaited_once()
        kwargs = memory_service.add_system_record.call_args.kwargs
        assert kwargs.get('agent_id') == 'recon-stage-task_knowledge_sync'
        assert kwargs.get('category') == 'observations_and_summaries'
        assert kwargs.get('causation_id') == 'run-mirror'
        metadata = kwargs.get('metadata') or {}
        assert metadata.get('kind') == 'cycle_summary'
        assert metadata.get('stage') == 'task_knowledge_sync'
        assert metadata.get('run_id') == 'run-mirror'
        assert 'recon_pool' not in metadata
        assert metadata.get('record_type') == 'ledger_stamp'
        assert 'run-mirror' in kwargs.get('content', '')

    @pytest.mark.asyncio
    async def test_pool_trim_invoked_with_recon_pool_trim_source_cap(self, ledger_store):
        memory_service = AsyncMock()
        memory_service.recon_ledger = ledger_store
        memory_service.add_system_record = AsyncMock(
            return_value=SimpleNamespace(memory_ids=['m1']),
        )

        with patch(
            'fused_memory.reconciliation.summary_pool.enforce_summary_pool_cap',
            AsyncMock(return_value=0),
        ) as mock_trim:
            await write_cycle_summary(
                memory_service,
                'dark_factory',
                self._report(),
                'run-trim',
                stage='task_knowledge_sync',
                recon_pool='stage2_cycle_summary',
                trim_source='stage2_cycle_summary_trim',
                cap=2,
            )

        mock_trim.assert_awaited_once()
        kwargs = mock_trim.call_args.kwargs
        assert kwargs.get('recon_pool') == 'stage2_cycle_summary'
        assert kwargs.get('trim_source') == 'stage2_cycle_summary_trim'
        assert kwargs.get('cap') == 2

    @pytest.mark.asyncio
    async def test_mirror_failure_swallowed_ledger_write_still_returns_true(self, ledger_store):
        memory_service = AsyncMock()
        memory_service.recon_ledger = ledger_store
        memory_service.add_system_record = AsyncMock(side_effect=RuntimeError('mem0 down'))

        result = await write_cycle_summary(
            memory_service,
            'dark_factory',
            self._report(),
            'run-mirror-fail',
            stage='task_knowledge_sync',
            recon_pool='stage2_cycle_summary',
            trim_source='stage2_cycle_summary_trim',
            cap=2,
        )

        assert result is True
        record = await ledger_store.get_by_identity(
            'dark_factory', 'cycle_summary', flag_type='task_knowledge_sync', run_id='run-mirror-fail',
        )
        assert record is not None

    @pytest.mark.asyncio
    async def test_trim_failure_swallowed_ledger_write_still_returns_true(self, ledger_store):
        memory_service = AsyncMock()
        memory_service.recon_ledger = ledger_store
        memory_service.add_system_record = AsyncMock(
            return_value=SimpleNamespace(memory_ids=['m1']),
        )

        with patch(
            'fused_memory.reconciliation.summary_pool.enforce_summary_pool_cap',
            AsyncMock(side_effect=RuntimeError('trim exploded')),
        ):
            result = await write_cycle_summary(
                memory_service,
                'dark_factory',
                self._report(),
                'run-trim-fail',
                stage='task_knowledge_sync',
                recon_pool='stage2_cycle_summary',
                trim_source='stage2_cycle_summary_trim',
                cap=2,
            )

        assert result is True
        record = await ledger_store.get_by_identity(
            'dark_factory', 'cycle_summary', flag_type='task_knowledge_sync', run_id='run-trim-fail',
        )
        assert record is not None

    @pytest.mark.asyncio
    async def test_both_mirror_and_trim_failures_swallowed_together(self, ledger_store):
        """Neither best-effort failure ever propagates — even simultaneously."""
        memory_service = AsyncMock()
        memory_service.recon_ledger = ledger_store
        memory_service.add_system_record = AsyncMock(side_effect=RuntimeError('mem0 down'))

        with patch(
            'fused_memory.reconciliation.summary_pool.enforce_summary_pool_cap',
            AsyncMock(side_effect=RuntimeError('trim exploded')),
        ):
            result = await write_cycle_summary(
                memory_service,
                'dark_factory',
                self._report(),
                'run-both-fail',
                stage='task_knowledge_sync',
                recon_pool='stage2_cycle_summary',
                trim_source='stage2_cycle_summary_trim',
                cap=2,
            )

        assert result is True

    @pytest.mark.asyncio
    async def test_mirror_and_trim_still_run_when_no_ledger_wired(self):
        """A disabled/absent ledger (``recon_ledger_enabled=False``, a
        supported non-default config — server/main.py) must not silence the
        Mem0 mirror too. Stage 3's cycle-summary presence check
        (prompts/stage3.py) reads only the mirror, never the ledger, so if
        the mirror also went dark whenever the ledger was absent, Stage 3
        would false-report "summary missing" every cycle with no fallback
        signal (reviewer finding robustness, task 2229 amendment pass).

        The authoritative ledger upsert is still correctly skipped (``False``)
        — this only proves the best-effort mirror/trim are no longer gated
        behind ledger availability.
        """
        memory_service = AsyncMock()
        memory_service.recon_ledger = None
        memory_service.add_system_record = AsyncMock(
            return_value=SimpleNamespace(memory_ids=['m1']),
        )

        with patch(
            'fused_memory.reconciliation.summary_pool.enforce_summary_pool_cap',
            AsyncMock(return_value=0),
        ) as mock_trim:
            result = await write_cycle_summary(
                memory_service,
                'dark_factory',
                self._report(),
                'run-no-ledger',
                stage='task_knowledge_sync',
                recon_pool='stage2_cycle_summary',
                trim_source='stage2_cycle_summary_trim',
                cap=2,
            )

        assert result is False  # no ledger => authoritative write correctly skipped

        memory_service.add_system_record.assert_awaited_once()
        kwargs = memory_service.add_system_record.call_args.kwargs
        assert kwargs.get('agent_id') == 'recon-stage-task_knowledge_sync'
        metadata = kwargs.get('metadata') or {}
        assert metadata.get('kind') == 'cycle_summary'
        assert metadata.get('run_id') == 'run-no-ledger'
        assert metadata.get('record_type') == 'ledger_stamp'
        assert 'run-no-ledger' in kwargs.get('content', '')

        mock_trim.assert_awaited_once()


class TestEnforceSummaryPoolCapPrecision:
    """The trim must match only real cycle_summary records, and must evict the
    disposable narratives before the ledger_stamp mirrors (task 3041).

    This is the path that consumed the run-84eae9bd anchors reported by recon
    gate 165 / esc-165-1. Two latent precision defects made it worse than the
    designed cap-2 bound alone:

    (a) the enumeration filtered ONLY on recon_pool, with no kind constraint —
        and _apply_cycle_summary_metadata_tagging is additive-only, never
        stripping a caller-supplied recon_pool, so a mis-tagged non-summary
        record could join the pool and be trimmed (or evict a real mirror);
    (b) the sort was record_type-blind, so an LLM-authored 'narrative' copy
        could evict the deterministic 'ledger_stamp' mirror that
        get_cycle_summary_presence parity and any run-scoped audit depend on.
    """

    @pytest.mark.asyncio
    async def test_enumeration_is_constrained_to_kind_cycle_summary(self):
        memory_service = AsyncMock()
        memory_service.get_memories_by_metadata = AsyncMock(return_value=[])
        memory_service.delete_memory = AsyncMock(return_value=None)

        await enforce_summary_pool_cap(
            memory_service,
            project_id='dark_factory',
            run_id='run-deleter',
            recon_pool='stage1_cycle_summary',
            trim_source='stage1_cycle_summary_trim',
            cap=2,
        )

        assert memory_service.get_memories_by_metadata.await_args.kwargs['filters'] == {
            'recon_pool': 'stage1_cycle_summary',
            'kind': 'cycle_summary',
        }

    @pytest.mark.asyncio
    async def test_newest_narrative_is_evicted_before_an_older_ledger_stamp(self):
        """The exact shape that made all three run-84eae9bd anchors vanish."""
        members = [
            {'id': 'stamp-oldest', 'created_at': '2026-01-01T00:00:00+00:00',
             'metadata': {'kind': 'cycle_summary', 'record_type': 'ledger_stamp'}},
            {'id': 'stamp-middle', 'created_at': '2026-02-01T00:00:00+00:00',
             'metadata': {'kind': 'cycle_summary', 'record_type': 'ledger_stamp'}},
            {'id': 'narrative-newest', 'created_at': '2026-03-01T00:00:00+00:00',
             'metadata': {'kind': 'cycle_summary', 'record_type': 'narrative'}},
        ]
        memory_service = AsyncMock()
        memory_service.get_memories_by_metadata = AsyncMock(return_value=members)
        memory_service.delete_memory = AsyncMock(return_value=None)

        result = await enforce_summary_pool_cap(
            memory_service,
            project_id='dark_factory',
            run_id='run-deleter',
            recon_pool='stage1_cycle_summary',
            trim_source='stage1_cycle_summary_trim',
            cap=2,
        )

        assert result == 1
        deleted = {
            call.kwargs.get('memory_id')
            for call in memory_service.delete_memory.call_args_list
        }
        # The disposable LLM-authored copy goes; both ledger_stamp mirrors —
        # the records get_cycle_summary_presence parity depends on — survive.
        assert deleted == {'narrative-newest'}

    @pytest.mark.asyncio
    async def test_older_narrative_evicted_first_when_narratives_are_over_cap(self):
        members = [
            {'id': 'narrative-older', 'created_at': '2026-01-01T00:00:00+00:00',
             'metadata': {'kind': 'cycle_summary', 'record_type': 'narrative'}},
            {'id': 'narrative-newer', 'created_at': '2026-02-01T00:00:00+00:00',
             'metadata': {'kind': 'cycle_summary', 'record_type': 'narrative'}},
            {'id': 'stamp', 'created_at': '2026-03-01T00:00:00+00:00',
             'metadata': {'kind': 'cycle_summary', 'record_type': 'ledger_stamp'}},
        ]
        memory_service = AsyncMock()
        memory_service.get_memories_by_metadata = AsyncMock(return_value=members)
        memory_service.delete_memory = AsyncMock(return_value=None)

        result = await enforce_summary_pool_cap(
            memory_service,
            project_id='dark_factory',
            run_id='run-deleter',
            recon_pool='stage1_cycle_summary',
            trim_source='stage1_cycle_summary_trim',
            cap=2,
        )

        assert result == 1
        deleted = {
            call.kwargs.get('memory_id')
            for call in memory_service.delete_memory.call_args_list
        }
        assert deleted == {'narrative-older'}

    @pytest.mark.asyncio
    async def test_ledger_stamps_still_evict_oldest_first_among_themselves(self):
        members = [
            {'id': 'stamp-oldest', 'created_at': '2026-01-01T00:00:00+00:00',
             'metadata': {'kind': 'cycle_summary', 'record_type': 'ledger_stamp'}},
            {'id': 'stamp-middle', 'created_at': '2026-02-01T00:00:00+00:00',
             'metadata': {'kind': 'cycle_summary', 'record_type': 'ledger_stamp'}},
            {'id': 'stamp-newest', 'created_at': '2026-03-01T00:00:00+00:00',
             'metadata': {'kind': 'cycle_summary', 'record_type': 'ledger_stamp'}},
        ]
        memory_service = AsyncMock()
        memory_service.get_memories_by_metadata = AsyncMock(return_value=members)
        memory_service.delete_memory = AsyncMock(return_value=None)

        result = await enforce_summary_pool_cap(
            memory_service,
            project_id='dark_factory',
            run_id='run-deleter',
            recon_pool='stage1_cycle_summary',
            trim_source='stage1_cycle_summary_trim',
            cap=2,
        )

        assert result == 1
        deleted = {
            call.kwargs.get('memory_id')
            for call in memory_service.delete_memory.call_args_list
        }
        assert deleted == {'stamp-oldest'}

    @pytest.mark.asyncio
    async def test_undatable_member_sorts_last_within_its_own_class(self):
        """Pre-existing invariant: an undatable member is never preferentially
        deleted — now scoped WITHIN its record_type class."""
        members = [
            {'id': 'narrative-dated', 'created_at': '2026-01-01T00:00:00+00:00',
             'metadata': {'kind': 'cycle_summary', 'record_type': 'narrative'}},
            {'id': 'narrative-undatable', 'created_at': None,
             'metadata': {'kind': 'cycle_summary', 'record_type': 'narrative'}},
            {'id': 'narrative-unparseable', 'created_at': 'not-a-timestamp',
             'metadata': {'kind': 'cycle_summary', 'record_type': 'narrative'}},
            {'id': 'stamp', 'created_at': '2026-03-01T00:00:00+00:00',
             'metadata': {'kind': 'cycle_summary', 'record_type': 'ledger_stamp'}},
        ]
        memory_service = AsyncMock()
        memory_service.get_memories_by_metadata = AsyncMock(return_value=members)
        memory_service.delete_memory = AsyncMock(return_value=None)

        result = await enforce_summary_pool_cap(
            memory_service,
            project_id='dark_factory',
            run_id='run-deleter',
            recon_pool='stage1_cycle_summary',
            trim_source='stage1_cycle_summary_trim',
            cap=2,
        )

        assert result == 2
        deleted = {
            call.kwargs.get('memory_id')
            for call in memory_service.delete_memory.call_args_list
        }
        # The dated narrative goes first; one undatable narrative follows only
        # because the pool is still over cap. The ledger_stamp survives.
        assert 'narrative-dated' in deleted
        assert 'stamp' not in deleted

    @pytest.mark.asyncio
    async def test_member_with_no_metadata_key_does_not_raise(self):
        members = [
            {'id': 'no-metadata-1', 'created_at': '2026-01-01T00:00:00+00:00'},
            {'id': 'no-metadata-2', 'created_at': '2026-02-01T00:00:00+00:00'},
            {'id': 'stamp', 'created_at': '2026-03-01T00:00:00+00:00',
             'metadata': {'kind': 'cycle_summary', 'record_type': 'ledger_stamp'}},
        ]
        memory_service = AsyncMock()
        memory_service.get_memories_by_metadata = AsyncMock(return_value=members)
        memory_service.delete_memory = AsyncMock(return_value=None)

        result = await enforce_summary_pool_cap(
            memory_service,
            project_id='dark_factory',
            run_id='run-deleter',
            recon_pool='stage1_cycle_summary',
            trim_source='stage1_cycle_summary_trim',
            cap=2,
        )

        assert result == 1
        deleted = {
            call.kwargs.get('memory_id')
            for call in memory_service.delete_memory.call_args_list
        }
        assert deleted == {'no-metadata-1'}


class TestEnforceSummaryPoolCapTombstones:
    """Every trim eviction leaves a queryable tombstone (task 3041).

    This is the path that consumed the run-84eae9bd anchors reported by recon
    gate 165 / esc-165-1. The fix for its "no audit trail" signature is the
    tombstone, not a retention change: an auditor holding only a memory uuid
    can now find out which sweep took it and on whose run.
    """

    @staticmethod
    def _over_cap_members():
        return [
            {'id': 'trimmed-ok', 'created_at': '2026-01-01T00:00:00+00:00',
             'metadata': {'kind': 'cycle_summary', 'record_type': 'narrative',
                          'recon_pool': 'stage1_cycle_summary', 'run_id': 'run-victim'}},
            {'id': 'trim-fails', 'created_at': '2026-02-01T00:00:00+00:00',
             'metadata': {'kind': 'cycle_summary', 'record_type': 'narrative',
                          'recon_pool': 'stage1_cycle_summary', 'run_id': 'run-victim-2'}},
            {'id': 'kept', 'created_at': '2026-03-01T00:00:00+00:00',
             'metadata': {'kind': 'cycle_summary', 'record_type': 'ledger_stamp',
                          'recon_pool': 'stage1_cycle_summary', 'run_id': 'run-kept'}},
        ]

    @staticmethod
    def _service(members):
        memory_service = AsyncMock()
        memory_service.get_memories_by_metadata = AsyncMock(return_value=members)

        async def _delete(**kwargs):
            if kwargs.get('memory_id') == 'trim-fails':
                raise RuntimeError('mem0 delete exploded')
            return None

        memory_service.delete_memory = AsyncMock(side_effect=_delete)
        return memory_service

    @pytest.mark.asyncio
    async def test_tombstone_written_only_for_the_successful_trim(self):
        import fused_memory.reconciliation.summary_pool as sp

        members = self._over_cap_members()
        memory_service = self._service(members)

        with patch.object(
            sp, 'record_mem0_deletion_tombstone', new=AsyncMock(return_value=True)
        ) as tombstone:
            result = await enforce_summary_pool_cap(
                memory_service,
                project_id='dark_factory',
                run_id='run-deleter',
                recon_pool='stage1_cycle_summary',
                trim_source='stage1_cycle_summary_trim',
                cap=1,
            )

        assert tombstone.await_count == 1
        call = tombstone.await_args
        assert call.args[1] == 'dark_factory'
        assert call.args[2] == 'trimmed-ok'
        assert call.kwargs['deleter'] == 'stage1_cycle_summary_trim'
        # The DELETING run, explicitly distinct from the victim's own
        # metadata['run_id'] — the precise ambiguity that made the original
        # finding unreadable.
        assert call.kwargs['deleting_run_id'] == 'run-deleter'
        assert call.kwargs['victim_metadata']['run_id'] == 'run-victim'
        assert call.kwargs['victim_metadata']['kind'] == 'cycle_summary'
        assert call.kwargs['victim_metadata']['record_type'] == 'narrative'
        assert call.kwargs['victim_metadata']['recon_pool'] == 'stage1_cycle_summary'
        assert call.kwargs['victim_created_at'] == '2026-01-01T00:00:00+00:00'

        # A tombstone must never claim a record that is still alive, and
        # tombstone writing must not perturb the existing accounting.
        assert result == 1

    @pytest.mark.asyncio
    async def test_a_raising_tombstone_helper_cannot_propagate_or_change_the_count(self):
        import fused_memory.reconciliation.summary_pool as sp

        memory_service = self._service(self._over_cap_members())

        with patch.object(
            sp,
            'record_mem0_deletion_tombstone',
            new=AsyncMock(side_effect=RuntimeError('ledger db locked')),
        ):
            result = await enforce_summary_pool_cap(
                memory_service,
                project_id='dark_factory',
                run_id='run-deleter',
                recon_pool='stage1_cycle_summary',
                trim_source='stage1_cycle_summary_trim',
                cap=1,
            )

        assert result == 1

    @pytest.mark.asyncio
    async def test_no_tombstone_when_pool_is_within_cap(self):
        import fused_memory.reconciliation.summary_pool as sp

        memory_service = AsyncMock()
        memory_service.get_memories_by_metadata = AsyncMock(return_value=[
            {'id': 'only', 'created_at': '2026-01-01T00:00:00+00:00',
             'metadata': {'kind': 'cycle_summary', 'record_type': 'ledger_stamp'}},
        ])
        memory_service.delete_memory = AsyncMock(return_value=None)

        with patch.object(
            sp, 'record_mem0_deletion_tombstone', new=AsyncMock(return_value=True)
        ) as tombstone:
            result = await enforce_summary_pool_cap(
                memory_service,
                project_id='dark_factory',
                run_id='run-deleter',
                recon_pool='stage1_cycle_summary',
                trim_source='stage1_cycle_summary_trim',
                cap=2,
            )

        assert result == 0
        tombstone.assert_not_awaited()
