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

import logging
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from fused_memory.reconciliation.summary_pool import (
    enforce_summary_pool_cap,
    pretrim_summary_pool,
    reconstruct_cycle_summary_stub,
    verify_cycle_summary_written,
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

# Parametrize stage across both known stages to prove genericity of the
# absence self-heal functions (verify_cycle_summary_written,
# reconstruct_cycle_summary_stub).
_STAGE_PARAMS = pytest.mark.parametrize(
    'stage',
    ['memory_consolidator', 'task_knowledge_sync'],
    ids=['stage1', 'stage2'],
)

# Parametrize (stage, recon_pool, reconstruct_source) across both known
# stages to prove reconstruct_cycle_summary_stub is generic.
_STAGE_RECON_PARAMS = pytest.mark.parametrize(
    'stage,recon_pool,reconstruct_source',
    [
        ('task_knowledge_sync', 'stage2_cycle_summary', 'stage2_summary_reconstruction'),
        ('memory_consolidator', 'stage1_cycle_summary', 'stage1_summary_reconstruction'),
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


class TestPretrimSummaryPool:
    """pretrim_summary_pool delegates to enforce_summary_pool_cap with
    cap=max(cap-1, 0), reserving one slot for the imminent agent write."""

    def _pool_members(self, count: int, recon_pool: str) -> list:
        return [
            {
                'id': f'summary-{i}',
                'created_at': f'2026-0{i+1}-01T00:00:00+00:00',
                'metadata': {'recon_pool': recon_pool},
            }
            for i in range(count)
        ]

    @_POOL_PARAMS
    @pytest.mark.asyncio
    async def test_two_members_deletes_one_oldest_returns_one(self, recon_pool, trim_source):
        """2 pool members → trim to cap-1=1 → delete 1 oldest, returns 1."""
        memory_service = AsyncMock()
        memory_service.get_memories_by_metadata = AsyncMock(
            return_value=self._pool_members(2, recon_pool)
        )
        memory_service.delete_memory = AsyncMock(return_value=None)

        result = await pretrim_summary_pool(
            memory_service,
            project_id='dark_factory',
            run_id='run-pre',
            recon_pool=recon_pool,
            trim_source=trim_source,
            cap=2,
        )

        assert result == 1
        deleted_ids = {c.kwargs['memory_id'] for c in memory_service.delete_memory.call_args_list}
        assert deleted_ids == {'summary-0'}

    @_POOL_PARAMS
    @pytest.mark.asyncio
    async def test_three_members_deletes_two_oldest_returns_two(self, recon_pool, trim_source):
        """3 pool members → trim to cap-1=1 → delete 2 oldest, returns 2."""
        memory_service = AsyncMock()
        memory_service.get_memories_by_metadata = AsyncMock(
            return_value=self._pool_members(3, recon_pool)
        )
        memory_service.delete_memory = AsyncMock(return_value=None)

        result = await pretrim_summary_pool(
            memory_service,
            project_id='dark_factory',
            run_id='run-pre',
            recon_pool=recon_pool,
            trim_source=trim_source,
            cap=2,
        )

        assert result == 2
        deleted_ids = {c.kwargs['memory_id'] for c in memory_service.delete_memory.call_args_list}
        assert deleted_ids == {'summary-0', 'summary-1'}

    @_POOL_PARAMS
    @pytest.mark.asyncio
    async def test_one_member_deletes_nothing_returns_zero(self, recon_pool, trim_source):
        """1 pool member (already at cap-1=1) → deletes nothing, returns 0."""
        memory_service = AsyncMock()
        memory_service.get_memories_by_metadata = AsyncMock(
            return_value=self._pool_members(1, recon_pool)
        )
        memory_service.delete_memory = AsyncMock(return_value=None)

        result = await pretrim_summary_pool(
            memory_service,
            project_id='dark_factory',
            run_id='run-pre',
            recon_pool=recon_pool,
            trim_source=trim_source,
            cap=2,
        )

        assert result == 0
        memory_service.delete_memory.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_delete_called_with_correct_audit_kwargs(self):
        """delete_memory called with store='mem0', causation_id=run_id, _source=<trim_source>."""
        memory_service = AsyncMock()
        memory_service.get_memories_by_metadata = AsyncMock(
            return_value=self._pool_members(2, 'stage1_cycle_summary')
        )
        memory_service.delete_memory = AsyncMock(return_value=None)

        await pretrim_summary_pool(
            memory_service,
            project_id='dark_factory',
            run_id='run-check',
            recon_pool='stage1_cycle_summary',
            trim_source='stage1_cycle_summary_trim',
            cap=2,
        )

        call_kwargs = memory_service.delete_memory.call_args.kwargs
        assert call_kwargs['store'] == 'mem0'
        assert call_kwargs['causation_id'] == 'run-check'
        assert call_kwargs['_source'] == 'stage1_cycle_summary_trim'

    @pytest.mark.asyncio
    async def test_cap_zero_does_not_go_negative(self):
        """cap=0 → pretrim target max(0-1, 0)==0, not -1 (would over-delete via slicing)."""
        # 1 member, cap=0 -> pretrim target 0 -> the single member is over cap(0) -> deleted.
        memory_service = AsyncMock()
        memory_service.get_memories_by_metadata = AsyncMock(
            return_value=self._pool_members(1, 'stage1_cycle_summary')
        )
        memory_service.delete_memory = AsyncMock(return_value=None)

        result = await pretrim_summary_pool(
            memory_service,
            project_id='dark_factory',
            run_id='run-zero',
            recon_pool='stage1_cycle_summary',
            trim_source='stage1_cycle_summary_trim',
            cap=0,
        )

        assert result == 1


class TestVerifyCycleSummaryWritten:
    """verify_cycle_summary_written: deterministic post-write count check.

    Generalizes task_knowledge_sync.py's _verify_stage2_summary_written
    (task 2366) — parametrized over `stage` to prove the core is generic,
    not hardcoded to either Stage 1 or Stage 2.
    """

    @_STAGE_PARAMS
    @pytest.mark.asyncio
    async def test_returns_positive_count_when_present(self, stage):
        memory_service = AsyncMock()
        memory_service.count_memories_by_metadata = AsyncMock(return_value=1)

        result = await verify_cycle_summary_written(
            memory_service,
            'dark_factory',
            'run-present',
            stage=stage,
        )

        assert result == 1

    @_STAGE_PARAMS
    @pytest.mark.asyncio
    async def test_returns_zero_when_confirmed_absent(self, stage):
        memory_service = AsyncMock()
        memory_service.count_memories_by_metadata = AsyncMock(return_value=0)

        result = await verify_cycle_summary_written(
            memory_service,
            'dark_factory',
            'run-absent',
            stage=stage,
        )

        assert result == 0

    @_STAGE_PARAMS
    @pytest.mark.asyncio
    async def test_returns_none_and_logs_warning_on_transient_failure(self, stage, caplog):
        """A count_memories_by_metadata failure returns None (NOT 0) — absence is
        NOT confirmed on a transient error."""
        memory_service = AsyncMock()
        memory_service.count_memories_by_metadata = AsyncMock(
            side_effect=RuntimeError('qdrant gone')
        )

        with caplog.at_level(logging.WARNING, logger=_LOGGER):
            result = await verify_cycle_summary_written(
                memory_service,
                'dark_factory',
                'run-transient',
                stage=stage,
            )

        assert result is None
        warning_records = [r for r in caplog.records if r.levelno >= logging.WARNING]
        assert len(warning_records) >= 1

    @_STAGE_PARAMS
    @pytest.mark.asyncio
    async def test_count_called_with_exact_triple_filter(self, stage):
        """The count call must use the exact triple filter — no recon_pool key."""
        memory_service = AsyncMock()
        memory_service.count_memories_by_metadata = AsyncMock(return_value=1)

        await verify_cycle_summary_written(
            memory_service,
            'my_project',
            'run-xyz',
            stage=stage,
        )

        memory_service.count_memories_by_metadata.assert_awaited_once()
        call = memory_service.count_memories_by_metadata.call_args
        kwargs = call.kwargs
        assert kwargs.get('project_id') == 'my_project'
        assert kwargs.get('filters') == {
            'kind': 'cycle_summary',
            'run_id': 'run-xyz',
            'stage': stage,
        }


class TestReconstructCycleSummaryStub:
    """reconstruct_cycle_summary_stub: dedup-resilient fallback-stub write.

    Generalizes task_knowledge_sync._reconstruct_stage2_summary (task 2366)
    — parametrized over (stage, recon_pool, reconstruct_source) to prove the
    core is generic, not hardcoded to either Stage 1 or Stage 2.
    """

    @_STAGE_RECON_PARAMS
    @pytest.mark.asyncio
    async def test_first_attempt_lands(self, stage, recon_pool, reconstruct_source):
        memory_service = AsyncMock()
        memory_service.add_memory = AsyncMock(return_value={'memory_ids': ['m1']})

        result = await reconstruct_cycle_summary_stub(
            memory_service,
            'dark_factory',
            'run-1',
            stage=stage,
            recon_pool=recon_pool,
            reconstruct_source=reconstruct_source,
        )

        assert result == 1
        memory_service.add_memory.assert_awaited_once()
        kwargs = memory_service.add_memory.call_args.kwargs
        assert kwargs.get('category') == 'observations_and_summaries'
        assert kwargs.get('project_id') == 'dark_factory'
        assert kwargs.get('causation_id') == 'run-1'
        assert kwargs.get('_source') == reconstruct_source
        assert kwargs.get('metadata') == {
            'kind': 'cycle_summary',
            'stage': stage,
            'run_id': 'run-1',
            'recon_pool': recon_pool,
            'reconstructed': True,
        }

    @_STAGE_RECON_PARAMS
    @pytest.mark.asyncio
    async def test_dedup_noop_then_retry_lands(self, stage, recon_pool, reconstruct_source):
        memory_service = AsyncMock()
        memory_service.add_memory = AsyncMock(
            side_effect=[{'memory_ids': []}, {'memory_ids': ['m2']}]
        )

        result = await reconstruct_cycle_summary_stub(
            memory_service,
            'dark_factory',
            'run-2',
            stage=stage,
            recon_pool=recon_pool,
            reconstruct_source=reconstruct_source,
        )

        assert result == 1
        assert memory_service.add_memory.await_count == 2
        contents = [
            c.kwargs.get('content') for c in memory_service.add_memory.call_args_list
        ]
        assert contents[0] != contents[1], (
            'retry must use a fresh nonce so the content differs from the first attempt'
        )

    @_STAGE_RECON_PARAMS
    @pytest.mark.asyncio
    async def test_dedup_noop_twice_returns_zero_and_logs_warning(
        self, stage, recon_pool, reconstruct_source, caplog
    ):
        memory_service = AsyncMock()
        memory_service.add_memory = AsyncMock(
            side_effect=[{'memory_ids': []}, {'memory_ids': []}]
        )

        with caplog.at_level(logging.WARNING, logger=_LOGGER):
            result = await reconstruct_cycle_summary_stub(
                memory_service,
                'dark_factory',
                'run-3',
                stage=stage,
                recon_pool=recon_pool,
                reconstruct_source=reconstruct_source,
            )

        assert result == 0
        assert memory_service.add_memory.await_count == 2
        warning_records = [r for r in caplog.records if r.levelno >= logging.WARNING]
        assert len(warning_records) >= 1

    @_STAGE_RECON_PARAMS
    @pytest.mark.asyncio
    async def test_add_memory_raises_returns_zero_and_logs_warning(
        self, stage, recon_pool, reconstruct_source, caplog
    ):
        memory_service = AsyncMock()
        memory_service.add_memory = AsyncMock(side_effect=RuntimeError('mem0 down'))

        with caplog.at_level(logging.WARNING, logger=_LOGGER):
            result = await reconstruct_cycle_summary_stub(
                memory_service,
                'dark_factory',
                'run-4',
                stage=stage,
                recon_pool=recon_pool,
                reconstruct_source=reconstruct_source,
            )

        assert result == 0
        warning_records = [r for r in caplog.records if r.levelno >= logging.WARNING]
        assert len(warning_records) >= 1

    @pytest.mark.asyncio
    async def test_content_shape_leads_with_stage1_nonce_and_embeds_run_id(self):
        """content leads with a STAGE1-prefixed nonce and embeds 'run_id: <run_id>'
        so both the metadata triple-filter and Path-1 semantic search self-heal."""
        memory_service = AsyncMock()
        memory_service.add_memory = AsyncMock(return_value={'memory_ids': ['m1']})

        await reconstruct_cycle_summary_stub(
            memory_service,
            'dark_factory',
            'run-shape',
            stage='memory_consolidator',
            recon_pool='stage1_cycle_summary',
            reconstruct_source='stage1_summary_reconstruction',
        )

        content = memory_service.add_memory.call_args.kwargs.get('content')
        first_line = content.splitlines()[0]
        assert first_line.startswith('STAGE1_'), (
            f'expected content to lead with a STAGE1-prefixed nonce, got: {first_line!r}'
        )
        assert 'run_id: run-shape' in content

    @pytest.mark.asyncio
    async def test_extract_response_memory_ids_handles_attr_response(self):
        """The response-memory_ids extraction must accept an AddMemoryResponse-like
        object (attribute access), not just a dict (mirrors production shape)."""
        memory_service = AsyncMock()
        memory_service.add_memory = AsyncMock(
            return_value=SimpleNamespace(memory_ids=['m1'])
        )

        result = await reconstruct_cycle_summary_stub(
            memory_service,
            'dark_factory',
            'run-attr',
            stage='memory_consolidator',
            recon_pool='stage1_cycle_summary',
            reconstruct_source='stage1_summary_reconstruction',
        )

        assert result == 1
        memory_service.add_memory.assert_awaited_once()
