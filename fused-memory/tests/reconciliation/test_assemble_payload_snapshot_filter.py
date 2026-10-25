"""Tests for count-snapshot stripping in memory_consolidator payload assembly (task 1547)."""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import AsyncMock, patch

import pytest
from _clip_assertions import assert_elided

from fused_memory.models.reconciliation import (
    AssembledPayload,
    ContextItem,
    EventSource,
    EventType,
    ReconciliationEvent,
    StageId,
    StageReport,
    Watermark,
)
from fused_memory.reconciliation.stages.base import BaseStage
from reconciliation.consolidator_fixtures import make_consolidator, make_scope

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
        stage = make_consolidator()
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
        stage = make_consolidator()
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


def _buffered_events_block(payload: str) -> str:
    return payload.split('### Buffered Events', 1)[1].split('### Related Context', 1)[0]


class TestAssembledPayloadDisclosesTheTokenBudgetClip:
    """Events the token budget deferred are named under the Buffered Events block."""

    @pytest.mark.asyncio
    async def test_deferred_events_render_a_clip_note(self):
        stage = make_consolidator()
        stage.assembled_payload = AssembledPayload(events=[], events_remaining=7, token_budget=65_000)

        result = await stage.assemble_payload(
            events=[], watermark=Watermark(project_id='test_project'), prior_reports=[],
        )

        [note] = [
            line for line in _buffered_events_block(result).splitlines() if line.startswith('_NOTE:')
        ]
        assert '7 additional buffered event(s)' in note
        assert 'reconciliation.token_budget=65000 cap' in note
        assert 'NOT complete this cycle' in note
        assert 'the newest events (still buffered for a later chunk) were dropped first' in note

    @pytest.mark.asyncio
    async def test_no_deferred_events_render_no_note(self):
        stage = make_consolidator()
        stage.assembled_payload = AssembledPayload(events=[], events_remaining=0, token_budget=65_000)

        result = await stage.assemble_payload(
            events=[], watermark=Watermark(project_id='test_project'), prior_reports=[],
        )

        assert '_NOTE:' not in _buffered_events_block(result)


_EPISODES_HEADER = '### New Episodes Since Last Reconciliation'
_MEMORIES_HEADER = '### New Mem0 Memories Since Last Reconciliation'


def _section(payload: str, header: str) -> str:
    return payload.split(header, 1)[1].split('\n### ', 1)[0]


def _episodes(*contents: str, created_at: tuple[str, ...] = ()) -> list[dict]:
    stamps = created_at or ('2026-05-28T00:00:00Z',) * len(contents)
    return [
        {'uuid': f'e{i}', 'created_at': stamp, 'content': c}
        for i, (c, stamp) in enumerate(zip(contents, stamps, strict=True))
    ]


def _memories(*contents: str, created_at: tuple[str, ...] = ()) -> dict:
    return {'results': [
        {'id': f'm{i}', 'memory': c, 'metadata': {'category': 'temporal_facts'},
         **({'created_at': created_at[i]} if created_at else {})}
        for i, c in enumerate(contents)
    ]}


async def _time_windowed_payload(stage, watermark: Watermark | None = None) -> str:
    return await stage.assemble_payload(
        events=[], watermark=watermark or Watermark(project_id='test_project'), prior_reports=[],
    )


def _rendered_from(section: str, body: str) -> str:
    """The rest of the line in *section* where *body*'s rendering begins."""
    return section[section.index(body[:20]):].split('\n', 1)[0]


def _notes(section: str) -> list[str]:
    return [line for line in section.splitlines() if line.startswith('_NOTE:')]


class TestTimeWindowedPayloadDisclosesClips:
    """Per-record body clips and saturated fetch caps are both visible in the Stage-1 payload."""

    @pytest.mark.asyncio
    async def test_long_records_carry_the_elision_marker_and_short_ones_do_not(self):
        episode_body = 'e' * 10_000
        memory_body = 'm' * 10_000
        stage = make_consolidator()
        stage.memory.get_episodes = AsyncMock(
            return_value=_episodes(f'{_SNAPSHOT_LINE}\n{episode_body}', 'short episode'),
        )
        stage.memory.mem0.get_all = AsyncMock(
            return_value=_memories(f'{memory_body}\n{_SNAPSHOT_LINE}', 'short memory'),
        )

        result = await _time_windowed_payload(stage)

        episodes = _section(result, _EPISODES_HEADER)
        assert_elided(_rendered_from(episodes, episode_body), episode_body)
        assert _rendered_from(episodes, 'short episode') == 'short episode'
        memories = _section(result, _MEMORIES_HEADER)
        assert_elided(_rendered_from(memories, memory_body), memory_body)
        assert _rendered_from(memories, 'short memory') == 'short memory'
        assert stage._entity_summary_snapshot_lines_stripped == 2

    @pytest.mark.asyncio
    async def test_typical_long_records_render_whole(self):
        """Bodies near the measured p90 reach Stage 1 whole (plans/recon-evidence-caps-measurement-2026-10.md, M5)."""
        episode_body = 'e' * 1_950
        memory_body = 'm' * 1_950
        stage = make_consolidator()
        stage.memory.get_episodes = AsyncMock(return_value=_episodes(episode_body))
        stage.memory.mem0.get_all = AsyncMock(return_value=_memories(memory_body))

        result = await _time_windowed_payload(stage)

        assert _rendered_from(_section(result, _EPISODES_HEADER), episode_body) == episode_body
        assert _rendered_from(_section(result, _MEMORIES_HEADER), memory_body) == memory_body

    @pytest.mark.asyncio
    async def test_a_fetch_that_fills_its_limit_renders_a_fetch_cap_note(self):
        stage = make_consolidator()
        stage.episode_limit = 2
        stage.memory_limit = 3
        stage.memory.get_episodes = AsyncMock(return_value=_episodes('a', 'b'))
        stage.memory.mem0.get_all = AsyncMock(return_value=_memories('x', 'y', 'z'))

        result = await _time_windowed_payload(stage)

        [episode_note] = _notes(_section(result, _EPISODES_HEADER))
        assert 'episode_limit=2' in episode_note
        assert 'NOT known complete this cycle' in episode_note
        [memory_note] = _notes(_section(result, _MEMORIES_HEADER))
        assert 'memory_limit=3' in memory_note
        assert 'NOT known complete this cycle' in memory_note

    @pytest.mark.asyncio
    async def test_a_full_episode_fetch_that_reached_the_watermark_renders_no_note(self):
        """Episodes arrive newest-first, so one at or before the watermark proves nothing newer was cut."""
        stage = make_consolidator()
        stage.episode_limit = 2
        stage.memory.get_episodes = AsyncMock(return_value=_episodes(
            'new', 'old', created_at=('2026-05-28T12:00:00Z', '2026-05-27T00:00:00Z'),
        ))
        watermark = Watermark(
            project_id='test_project', last_episode_timestamp=datetime(2026, 5, 28, tzinfo=UTC),
        )

        result = await _time_windowed_payload(stage, watermark)

        episodes = _section(result, _EPISODES_HEADER)
        assert 'new' in episodes and 'old' not in episodes
        assert _notes(episodes) == []

    @pytest.mark.asyncio
    async def test_a_full_mem0_fetch_notes_even_past_the_watermark(self):
        """mem0's get_all is not recency-ordered, so a dropped old record proves nothing about the rest."""
        stage = make_consolidator()
        stage.memory_limit = 2
        stage.memory.mem0.get_all = AsyncMock(return_value=_memories(
            'new', 'old', created_at=('2026-05-28T12:00:00+00:00', '2026-05-27T00:00:00+00:00'),
        ))
        watermark = Watermark(
            project_id='test_project', last_memory_timestamp=datetime(2026, 5, 28, tzinfo=UTC),
        )

        result = await _time_windowed_payload(stage, watermark)

        [memory_note] = _notes(_section(result, _MEMORIES_HEADER))
        assert 'memory_limit=2' in memory_note

    @pytest.mark.asyncio
    async def test_a_fetch_under_its_limit_renders_no_note(self):
        stage = make_consolidator()
        stage.episode_limit = 2
        stage.memory_limit = 3
        stage.memory.get_episodes = AsyncMock(return_value=_episodes('a'))
        stage.memory.mem0.get_all = AsyncMock(return_value=_memories('x', 'y'))

        result = await _time_windowed_payload(stage)

        assert '_NOTE:' not in _section(result, _EPISODES_HEADER)
        assert '_NOTE:' not in _section(result, _MEMORIES_HEADER)


class TestRunSurfacesSnapshotStrippedStat:
    """run() copies _entity_summary_snapshot_lines_stripped into report.stats."""

    @pytest.mark.asyncio
    async def test_run_copies_attr_into_stats(self):
        """When _entity_summary_snapshot_lines_stripped is pre-set, run() must copy it to stats.

        Mirrors the census-inconsistency pattern from TestMemoryConsolidatorRunWiring.
        """
        stage = make_consolidator()
        stage.scope = make_scope('test_project', stage.scope.project_root)
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


def _buffered_event(n: int) -> ReconciliationEvent:
    return ReconciliationEvent(
        id=f'ev{n}', type=EventType.memory_added, source=EventSource.agent,
        project_id='test_project', timestamp=datetime.now(UTC),
    )


async def _run_with_assembled_payload(assembled_payload: AssembledPayload | None) -> StageReport:
    stage = make_consolidator()
    stage.scope = make_scope('test_project', stage.scope.project_root)
    stage.assembled_payload = assembled_payload
    base_report = StageReport(
        stage=StageId.memory_consolidator,
        started_at=datetime.now(UTC),
        completed_at=datetime.now(UTC),
        items_flagged=[],
        stats={},
    )
    with patch.object(BaseStage, 'run', new=AsyncMock(return_value=base_report)):
        return await stage.run(
            events=[],
            watermark=Watermark(project_id='test_project'),
            prior_reports=[],
            run_id='run-assembled-budget',
        )


class TestRunRecordsAssembledBudget:
    """run() records the assembled payload's budget accounting in report.stats.

    This is what the next cap re-evaluation reads from runs.stage_reports
    (plans/recon-evidence-caps-measurement-2026-10.md).
    """

    @pytest.mark.asyncio
    async def test_the_assembled_path_records_its_budget_accounting(self):
        payload = AssembledPayload(
            events=[_buffered_event(0), _buffered_event(1)],
            context_items={f'c{n}': ContextItem(id=f'c{n}', source='mem0', formatted='x') for n in range(3)},
            total_tokens=1234,
            events_remaining=5,
            token_budget=130_000,
        )

        report = await _run_with_assembled_payload(payload)

        assert report.stats['stage1_assembled_budget'] == {
            'estimated_tokens': 1234,
            'token_budget': 130000,
            'events': 2,
            'context_items': 3,
            'events_deferred': 5,
        }

    @pytest.mark.asyncio
    async def test_the_time_windowed_path_records_none(self):
        report = await _run_with_assembled_payload(None)

        assert 'stage1_assembled_budget' in report.stats
        assert report.stats['stage1_assembled_budget'] is None
