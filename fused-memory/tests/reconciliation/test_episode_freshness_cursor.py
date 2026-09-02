"""Tests for the episode/Mem0 freshness cursor comparison (task 4574).

Regression coverage for the 894fbe90 incident: ``assemble_payload``'s "new
episodes since last reconciliation" filter compared timestamps as STRINGS —
``str(watermark.last_episode_timestamp)`` renders with a space separator,
while episode ``created_at`` values arrive from
``services/memory_service.py::_created_at_to_utc_iso`` as ISO-8601 with a
``T`` separator. Because ``'T'`` (0x54) sorts after ``' '`` (0x20), the
lexical ``>`` degenerated to date-granularity: any episode from the
watermark's own calendar day compared as "newer" regardless of its actual
time, so a same-day-earlier episode re-surfaced on every cycle forever.
"""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import AsyncMock

import pytest

from fused_memory.config.schema import ReconciliationConfig
from fused_memory.models.reconciliation import StageId, Watermark
from fused_memory.models.scope import ProjectId, ProjectRoot, ProjectScope
from fused_memory.reconciliation.stages.memory_consolidator import MemoryConsolidator


def _scope(project_id: str, project_root: str) -> ProjectScope:
    """Build a ProjectScope from raw strings — DRYs the many test call sites."""
    return ProjectScope(ProjectId(project_id), ProjectRoot(project_root))


def _make_consolidator(project_root: str = '/tmp/test') -> MemoryConsolidator:
    """Build a MemoryConsolidator with mocked deps — mirrors test_assemble_payload_snapshot_filter.py."""
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
        scope=_scope('test_project', project_root),
    )
    stage.episode_limit = 5
    stage.memory_limit = 10
    return stage


INCIDENT_UUID = '894fbe90-2eee-4329-83b4-eddd9f81e48d'
# LATER the same calendar day as the incident episode's created_at below.
WATERMARK_TS = datetime(2026, 8, 20, 12, 0, 0, tzinfo=UTC)


class TestEpisodeFreshnessCursor:
    """``assemble_payload``'s episode filter must compare instants, not strings."""

    @pytest.mark.asyncio
    async def test_same_day_earlier_episode_excluded(self):
        """The 894fbe90 incident: an episode from earlier the SAME calendar
        day as the watermark must be excluded, not re-surfaced forever."""
        stage = _make_consolidator()
        stage.memory.get_episodes = AsyncMock(
            return_value=[
                {
                    'uuid': INCIDENT_UUID,
                    'created_at': '2026-08-20T01:52:27+00:00',
                    'content': 'incident episode content',
                }
            ]
        )
        watermark = Watermark(project_id='test_project', last_episode_timestamp=WATERMARK_TS)

        result = await stage.assemble_payload(events=[], watermark=watermark, prior_reports=[])

        assert '### New Episodes Since Last Reconciliation (0)' in result, (
            f'Expected header (0); got result:\n{result!r}'
        )
        assert INCIDENT_UUID not in result, (
            f'Incident episode must not re-surface; got result:\n{result!r}'
        )

    @pytest.mark.asyncio
    async def test_genuinely_newer_episode_included(self):
        """An episode from the NEXT calendar day must still be surfaced —
        guards against a fix that degenerates into "filter everything"."""
        stage = _make_consolidator()
        stage.memory.get_episodes = AsyncMock(
            return_value=[
                {
                    'uuid': 'newer-episode',
                    'created_at': '2026-08-21T03:00:00+00:00',
                    'content': 'genuinely newer content',
                }
            ]
        )
        watermark = Watermark(project_id='test_project', last_episode_timestamp=WATERMARK_TS)

        result = await stage.assemble_payload(events=[], watermark=watermark, prior_reports=[])

        assert '### New Episodes Since Last Reconciliation (1)' in result, (
            f'Expected header (1); got result:\n{result!r}'
        )
        assert 'newer-episode' in result

    @pytest.mark.asyncio
    async def test_older_episode_excluded(self):
        """An episode from the PREVIOUS calendar day must be excluded."""
        stage = _make_consolidator()
        stage.memory.get_episodes = AsyncMock(
            return_value=[
                {
                    'uuid': 'older-episode',
                    'created_at': '2026-08-19T23:59:59+00:00',
                    'content': 'older content',
                }
            ]
        )
        watermark = Watermark(project_id='test_project', last_episode_timestamp=WATERMARK_TS)

        result = await stage.assemble_payload(events=[], watermark=watermark, prior_reports=[])

        assert '### New Episodes Since Last Reconciliation (0)' in result, (
            f'Expected header (0); got result:\n{result!r}'
        )
        assert 'older-episode' not in result
