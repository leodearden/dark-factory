"""MemoryService's Mem0 create seams enforce the Stage-1 flag-record write contract (task 4863, #3919).

Drives a REAL MemoryService with mocked backends and asserts on the metadata the
backend is handed — the persisted shape. ``TestOperatorSuppressionPath`` is a
regression guard that is already green before the seam is wired: the operator
mirror written by ``flag_dedup.write_suppression_record`` must keep landing.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
import pytest_asyncio

from fused_memory.models.enums import MemoryCategory
from fused_memory.models.memory import ClassificationResult
from fused_memory.reconciliation import flag_dedup
from fused_memory.reconciliation.flag_record_contract import (
    STAGE1_FLAG_KIND,
    STAGE1_FLAG_MARKER_KIND,
    STAGE1_FLAG_SUPPRESSION_KIND,
    FlagRecordSchemaError,
    ReconStageFlagRecordWriteRefused,
)
from fused_memory.reconciliation.recon_ledger import ReconLedgerStore
from fused_memory.services.memory_service import MemoryService

MEM0_ONLY_CATEGORY = 'observations_and_summaries'
SUPPRESSION_METADATA = {
    'kind': STAGE1_FLAG_SUPPRESSION_KIND,
    'task_id': '6107',
    'flag_types': ['task_memory_mismatch'],
}


@pytest.fixture
def service(mock_config):
    svc = MemoryService(mock_config)
    svc.mem0 = MagicMock()
    svc.mem0.add = AsyncMock(return_value={'results': [{'id': 'mem0-1'}]})
    svc.mem0.add_system_record = AsyncMock(return_value={'results': [{'id': 'sys-1'}]})
    svc.mem0.count_by_metadata = AsyncMock(return_value=0)
    svc.mem0.scroll_by_metadata = AsyncMock(return_value=[])
    svc.durable_queue = MagicMock()
    svc.durable_queue.enqueue = AsyncMock(return_value=1)
    svc.classifier.classify = AsyncMock(return_value=ClassificationResult(
        primary=MemoryCategory.observations_and_summaries, confidence=0.95,
    ))
    return svc


def _persisted_add_metadata(svc: Any) -> dict:
    svc.mem0.add.assert_awaited_once()
    return svc.mem0.add.await_args.kwargs['metadata']


def _persisted_system_record_metadata(svc: Any) -> dict:
    svc.mem0.add_system_record.assert_awaited_once()
    return svc.mem0.add_system_record.await_args.kwargs['metadata']


class TestRecordKindKeysAtSeam:
    """#3919 divergence A: a marker persists with both kind and source."""

    @pytest.mark.asyncio
    async def test_marker_kind_persists_source(self, service):
        await service.add_memory(
            'marker',
            category=MEM0_ONLY_CATEGORY,
            agent_id='claude-interactive',
            metadata={'kind': STAGE1_FLAG_MARKER_KIND, 'task_id': '7', 'flag_type': 'x'},
        )

        assert _persisted_add_metadata(service)['source'] == STAGE1_FLAG_MARKER_KIND

    @pytest.mark.asyncio
    async def test_marker_source_persists_kind(self, service):
        await service.add_memory(
            'marker',
            category=MEM0_ONLY_CATEGORY,
            agent_id='claude-interactive',
            metadata={'source': STAGE1_FLAG_MARKER_KIND, 'task_id': '7', 'flag_type': 'x'},
        )

        assert _persisted_add_metadata(service)['kind'] == STAGE1_FLAG_MARKER_KIND


class TestFlagTypeFieldAtSeam:
    """#3919 divergence B: each kind persists its canonical flag-type spelling."""

    @pytest.mark.asyncio
    async def test_marker_plural_persists_singular(self, service):
        await service.add_memory(
            'marker',
            category=MEM0_ONLY_CATEGORY,
            agent_id='claude-interactive',
            metadata={'kind': STAGE1_FLAG_MARKER_KIND, 'task_id': '7', 'flag_types': ['x']},
        )

        persisted = _persisted_add_metadata(service)
        assert persisted['flag_type'] == 'x'
        assert 'flag_types' not in persisted

    @pytest.mark.asyncio
    async def test_suppression_singular_persists_plural(self, service):
        await service.add_memory(
            'suppression',
            category=MEM0_ONLY_CATEGORY,
            agent_id=None,
            metadata={'kind': STAGE1_FLAG_SUPPRESSION_KIND, 'task_id': '7', 'flag_type': 'x'},
        )

        persisted = _persisted_add_metadata(service)
        assert persisted['flag_types'] == ['x']
        assert 'flag_type' not in persisted

    @pytest.mark.asyncio
    async def test_lossy_marker_is_rejected_before_mem0(self, service):
        with pytest.raises(FlagRecordSchemaError):
            await service.add_memory(
                'marker',
                category=MEM0_ONLY_CATEGORY,
                agent_id='claude-interactive',
                metadata={
                    'kind': STAGE1_FLAG_MARKER_KIND,
                    'task_id': '7',
                    'flag_types': ['x', 'y'],
                },
            )

        service.mem0.add.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_system_record_gets_the_same_normalization(self, service):
        await service.add_system_record(
            'stage1 flag',
            project_id='dark_factory',
            agent_id='recon-stage-task_knowledge_sync',
            category=MEM0_ONLY_CATEGORY,
            metadata={'kind': STAGE1_FLAG_KIND, 'task_id': '7', 'flag_types': ['x']},
        )

        persisted = _persisted_system_record_metadata(service)
        assert persisted['flag_type'] == 'x'
        assert 'flag_types' not in persisted


class TestReconStageRefusalBackstop:
    @pytest.mark.asyncio
    @pytest.mark.parametrize('category', [MEM0_ONLY_CATEGORY, None], ids=['explicit', 'classified'])
    async def test_add_memory_refuses_recon_stage_suppression(self, service, category):
        with pytest.raises(ReconStageFlagRecordWriteRefused):
            await service.add_memory(
                'STAGE 1 FLAG SUPPRESSION task_id=6107',
                category=category,
                agent_id='recon-stage-memory_consolidator',
                metadata=dict(SUPPRESSION_METADATA),
            )

        service.mem0.add.assert_not_awaited()
        if category is None:
            service.classifier.classify.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_add_system_record_refuses_recon_stage_suppression(self, service):
        with pytest.raises(ReconStageFlagRecordWriteRefused):
            await service.add_system_record(
                'STAGE 1 FLAG SUPPRESSION task_id=6107',
                project_id='dark_factory',
                agent_id='recon-stage-task_knowledge_sync',
                category=MEM0_ONLY_CATEGORY,
                metadata=dict(SUPPRESSION_METADATA),
            )

        service.mem0.add_system_record.assert_not_awaited()


@pytest_asyncio.fixture
async def ledger(tmp_path):
    store = ReconLedgerStore(tmp_path / 'reconciliation.db')
    await store.initialize()
    try:
        yield store
    finally:
        await store.close()


class TestOperatorSuppressionPath:
    """Acceptance (c): the operator path still writes the gate's ledger row and its Mem0 mirror."""

    @pytest.mark.asyncio
    async def test_write_suppression_record_lands_ledger_row_and_mirror(self, service, ledger):
        service.recon_ledger = ledger

        result = await flag_dedup.write_suppression_record(
            service,
            project_id='dark_factory',
            task_id=6107,
            flag_types=['task_memory_mismatch'],
        )

        assert await ledger.is_suppressed('dark_factory', '6107', 'task_memory_mismatch') is True
        persisted = _persisted_add_metadata(service)
        assert persisted['kind'] == STAGE1_FLAG_SUPPRESSION_KIND
        assert persisted['task_id'] == '6107'
        assert persisted['flag_types'] == ['task_memory_mismatch']
        assert result.memory_ids == ['mem0-1']
