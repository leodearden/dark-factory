"""Tests for count-snapshot stripping in memory_consolidator payload assembly (task 1547)."""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import AsyncMock, patch

import pytest

from fused_memory.config.schema import ReconciliationConfig
from fused_memory.models.reconciliation import (
    AssembledPayload,
    ContextItem,
    StageId,
    StageReport,
    Watermark,
)
from fused_memory.reconciliation.stages.base import BaseStage
from fused_memory.reconciliation.stages.memory_consolidator import MemoryConsolidator


def _make_consolidator(project_root: str = '') -> MemoryConsolidator:
    """Build a MemoryConsolidator with mocked deps — mirrors test_stage1.py."""
    config = ReconciliationConfig()
    memory_mock = AsyncMock()
    memory_mock.get_episodes = AsyncMock(return_value=[])
    memory_mock.mem0 = AsyncMock()
    memory_mock.mem0.get_all = AsyncMock(return_value={'results': []})
    memory_mock.get_status = AsyncMock(return_value={})

    stage = MemoryConsolidator(
        StageId.memory_consolidator,
        memory_mock,
        AsyncMock(),  # taskmaster
        AsyncMock(),  # journal
        config,
    )
    stage.project_id = 'test_project'
    stage.project_root = project_root
    stage.episode_limit = 5
    stage.memory_limit = 10
    return stage


_SNAPSHOT_LINE = '1505 done / 148 cancelled / 1653 total'
_BENIGN_LINE_A = 'Entity summary line A'
_BENIGN_LINE_B = 'Entity summary line B'


class TestAssemblePayloadSnapshotStripping:
    """Snapshot lines are stripped from episodes, memories, and context items."""

    @pytest.mark.asyncio
    async def test_time_windowed_strips_snapshot_from_episodes_and_memories(self):
        """Time-windowed path: snapshot lines in episodes and mem0 memories are stripped.

        Configures:
        - get_episodes returns one episode whose content contains a snapshot line
          sandwiched between two benign lines.
        - mem0.get_all returns one memory whose 'memory' field contains a benign
          line followed by a snapshot line.

        Asserts:
        - The snapshot line text is absent from the assembled payload.
        - Both benign lines are present.
        - stage._entity_summary_snapshot_lines_stripped == 2 (one from each source).
        """
        stage = _make_consolidator()
        stage.memory.get_episodes = AsyncMock(
            return_value=[
                {
                    'uuid': 'e1',
                    'created_at': '2026-05-28T00:00:00Z',
                    'content': f'{_BENIGN_LINE_A}\n{_SNAPSHOT_LINE}\n{_BENIGN_LINE_B}',
                }
            ]
        )
        stage.memory.mem0.get_all = AsyncMock(
            return_value={
                'results': [
                    {
                        'id': 'm1',
                        'memory': f'Benign mem line\n{_SNAPSHOT_LINE}',
                        'metadata': {'category': 'temporal_facts'},
                    }
                ]
            }
        )

        watermark = Watermark(project_id='test_project')
        result = await stage.assemble_payload(events=[], watermark=watermark, prior_reports=[])

        assert _SNAPSHOT_LINE not in result, (
            f'Snapshot line must be stripped from assembled payload; got result:\n{result!r}'
        )
        assert _BENIGN_LINE_A in result, (
            f'Benign line A must be preserved in assembled payload; got result:\n{result!r}'
        )
        assert _BENIGN_LINE_B in result, (
            f'Benign line B must be preserved in assembled payload; got result:\n{result!r}'
        )
        assert stage._entity_summary_snapshot_lines_stripped == 2, (
            f'Expected _entity_summary_snapshot_lines_stripped=2 (1 from episode + 1 from memory), '
            f'got {stage._entity_summary_snapshot_lines_stripped}'
        )

    @pytest.mark.asyncio
    async def test_assembled_payload_strips_snapshot_from_context_items(self):
        """Assembled-payload path: snapshot lines in context_items.formatted are stripped.

        Sets stage.assembled_payload with a ContextItem whose .formatted contains
        a snapshot line between benign lines.

        Asserts:
        - The snapshot line text is absent from the formatted payload.
        - Benign surrounding lines are present.
        - stage._entity_summary_snapshot_lines_stripped == 1.
        """
        stage = _make_consolidator()
        stage.assembled_payload = AssembledPayload(
            events=[],
            context_items={
                'k': ContextItem(
                    id='k',
                    source='graphiti',
                    formatted='Ctx head\n3355 done, 290 cancelled, 3358 total\nCtx tail',
                )
            },
        )

        watermark = Watermark(project_id='test_project')
        result = await stage.assemble_payload(events=[], watermark=watermark, prior_reports=[])

        assert '3355 done' not in result, (
            f'Snapshot line must be stripped from assembled context; got result:\n{result!r}'
        )
        assert 'Ctx head' in result, (
            f'"Ctx head" must be preserved in assembled payload; got result:\n{result!r}'
        )
        assert 'Ctx tail' in result, (
            f'"Ctx tail" must be preserved in assembled payload; got result:\n{result!r}'
        )
        assert stage._entity_summary_snapshot_lines_stripped == 1, (
            f'Expected _entity_summary_snapshot_lines_stripped=1, '
            f'got {stage._entity_summary_snapshot_lines_stripped}'
        )


class TestRunSurfacesSnapshotStrippedStat:
    """run() copies _entity_summary_snapshot_lines_stripped into report.stats."""

    @pytest.mark.asyncio
    async def test_run_copies_attr_into_stats(self):
        """When _entity_summary_snapshot_lines_stripped is pre-set, run() must copy it to stats.

        Mirrors the census-inconsistency pattern from TestMemoryConsolidatorRunWiring.
        """
        stage = _make_consolidator()
        stage.project_id = 'test_project'
        stage._entity_summary_snapshot_lines_stripped = 3

        base_report = StageReport(
            stage=StageId.memory_consolidator,
            started_at=datetime.now(UTC),
            completed_at=datetime.now(UTC),
            items_flagged=[],
            stats={},
        )

        with patch.object(BaseStage, 'run', new=AsyncMock(return_value=base_report)):
            report = await stage.run(
                events=[],
                watermark=Watermark(project_id='test_project'),
                prior_reports=[],
                run_id='run-delta-stats',
            )

        assert report.stats.get('entity_summary_snapshot_lines_stripped') == 3, (
            f'Expected report.stats["entity_summary_snapshot_lines_stripped"]=3, '
            f'got stats={report.stats!r}'
        )
