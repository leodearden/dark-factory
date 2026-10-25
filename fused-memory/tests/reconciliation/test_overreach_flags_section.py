"""The O4 handoff: shadow overreach flags reach recon Stage 1 (task 6137, design.md §6.1).

A real OverreachShadow over a real sidecar in tmp_path backs the fetch; its
LLM double raises, so a fetch that judged anything would fail loudly, and its
graph double answers the survivor read from a fixed set of live and retired
edges.
"""
from __future__ import annotations

import logging
from collections.abc import Callable, Iterable
from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import pytest_asyncio

from fused_memory.config.schema import (
    FusedMemoryConfig,
    OverreachGuardConfig,
    ReconciliationConfig,
)
from fused_memory.models.reconciliation import (
    AssembledPayload,
    RunStatus,
    StageReport,
    Watermark,
)
from fused_memory.reconciliation.event_buffer import EventBuffer
from fused_memory.reconciliation.harness import ReconciliationHarness
from fused_memory.reconciliation.journal import ReconciliationJournal
from fused_memory.reconciliation.overreach_flags import (
    FLAGS_SECTION_HEADER,
    FLAGS_SECTION_LIMIT,
    acknowledge_overreach_flags,
    fetch_overreach_flags,
    render_overreach_flags_section,
)
from fused_memory.reconciliation.stages.memory_consolidator import MemoryConsolidator
from fused_memory.reconciliation.stages.task_knowledge_sync import (
    IntegrityCheck,
    TaskKnowledgeSync,
)
from fused_memory.services.overreach_guard import PROMPT_VERSION, EdgeJudgement, JudgeEdge
from fused_memory.services.overreach_shadow import (
    JudgedEpisode,
    OverreachFlag,
    OverreachFlagPage,
    OverreachShadow,
    OverreachShadowStore,
    ShadowOutcome,
)
from reconciliation.consolidator_fixtures import make_consolidator

MODULE_LOGGER = 'fused_memory.reconciliation.overreach_flags'
PROJECT, OTHER = 'proj', 'other'
T0 = datetime(2026, 10, 17, 12, 0, 0, tzinfo=UTC)
T1 = T0 + timedelta(minutes=5)


class _NeverJudges:
    async def generate_response(self, *args, **kwargs):
        raise AssertionError('fetching flags must never judge')


class _Graph:
    """The survivor read over a fixed graph: every edge is live unless gone or retired."""

    def __init__(self, *, gone: Iterable[str] = (), retired: Iterable[str] = ()) -> None:
        self.gone, self.retired = set(gone), set(retired)

    async def __call__(self, uuids, *, group_id):
        return [
            {'uuid': uuid, 'fact': f'fact of {uuid}', 'source_name': 'S', 'target_name': 'T',
             'episodes': [], 'active': uuid not in self.retired}
            for uuid in uuids if uuid not in self.gone
        ]


def _clock(*instants: datetime) -> Callable[[], datetime]:
    queue = list(instants)
    return lambda: queue.pop(0) if len(queue) > 1 else queue[0]


def _episode(graph: str, uuid: str) -> JudgedEpisode:
    return JudgedEpisode(
        graph=graph, episode_uuid=uuid, judge_model='judge-m', prompt_version=PROMPT_VERSION,
        classifiers=('category_decisions',), minted_count=2, outcome=ShadowOutcome.judged,
    )


def _judged(edge_uuid: str, label: str) -> tuple[JudgeEdge, EdgeJudgement]:
    return (
        JudgeEdge(uuid=edge_uuid, fact=f'fact of {edge_uuid}', source_name='S', target_name='T'),
        EdgeJudgement(edge_uuid=edge_uuid, label=label, rationale=f'span for {edge_uuid}'),
    )


def _shadow(store: OverreachShadowStore, graph: _Graph | None = None) -> OverreachShadow:
    return OverreachShadow(
        store, _NeverJudges(), 'judge-m', graph or _Graph(), lambda: OverreachGuardConfig(),
    )


async def _seeded_store(tmp_path) -> OverreachShadowStore:
    store = await OverreachShadowStore.open(tmp_path / 'sidecar', clock=_clock(T0, T1, T1))
    await store.record(episode=_episode(PROJECT, 'ep-a'), judgements=[
        _judged('a1', 'overreach'), _judged('a2', 'holding'),
    ])
    await store.record(episode=_episode(PROJECT, 'ep-b'), judgements=[
        _judged('b2', 'overreach'), _judged('b1', 'overreach'),
    ])
    await store.record(episode=_episode(OTHER, 'ep-x'), judgements=[_judged('x1', 'overreach')])
    return store


@pytest_asyncio.fixture
async def shadow(tmp_path):
    runner = _shadow(await _seeded_store(tmp_path))
    yield runner
    await runner.close()


def _keys(page: OverreachFlagPage | None) -> list[tuple[str, str]]:
    assert page is not None
    return [(flag.episode_uuid, flag.edge_uuid) for flag in page.flags]


class _RecordingShadow:
    def __init__(self, error: Exception | None = None) -> None:
        self.error = error
        self.calls: list[tuple[str, int]] = []
        self.acknowledged: list[OverreachFlagPage] = []

    async def overreach_flag_page(self, graph: str, limit: int) -> OverreachFlagPage:
        self.calls.append((graph, limit))
        if self.error is not None:
            raise self.error
        return OverreachFlagPage(graph=graph, flags=(), omitted=0)

    async def acknowledge(self, page: OverreachFlagPage) -> None:
        if self.error is not None:
            raise self.error
        self.acknowledged.append(page)


def _warned(caplog) -> bool:
    return any(r.name == MODULE_LOGGER and r.levelno == logging.WARNING for r in caplog.records)


class TestFetch:
    @pytest.mark.asyncio
    async def test_only_this_projects_flags_in_order(self, shadow) -> None:
        page = await fetch_overreach_flags(shadow, PROJECT)
        assert _keys(page) == [('ep-a', 'a1'), ('ep-b', 'b1'), ('ep-b', 'b2')]

    @pytest.mark.asyncio
    async def test_it_asks_for_one_bounded_page(self) -> None:
        recording = _RecordingShadow()
        await fetch_overreach_flags(recording, PROJECT)
        assert recording.calls == [(PROJECT, FLAGS_SECTION_LIMIT)]
        assert FLAGS_SECTION_LIMIT == 40

    @pytest.mark.parametrize('source', [
        pytest.param(None, id='no-shadow'),
        pytest.param(_RecordingShadow(error=RuntimeError('db locked')), id='raises'),
    ])
    @pytest.mark.asyncio
    async def test_an_unavailable_shadow_is_none_and_warned(self, source, caplog) -> None:
        with caplog.at_level(logging.WARNING, logger=MODULE_LOGGER):
            assert await fetch_overreach_flags(source, PROJECT) is None
        assert _warned(caplog)

    @pytest.mark.asyncio
    async def test_a_flag_whose_edge_is_gone_or_retired_is_not_offered(self, tmp_path) -> None:
        runner = _shadow(await _seeded_store(tmp_path), _Graph(gone=['b1'], retired=['b2']))
        try:
            page = await fetch_overreach_flags(runner, PROJECT)
            await acknowledge_overreach_flags(runner, page)
            after = await fetch_overreach_flags(runner, PROJECT)
        finally:
            await runner.close()
        assert _keys(page) == [('ep-a', 'a1')]
        assert _keys(after) == []


class TestAcknowledge:
    @pytest.mark.asyncio
    async def test_an_acknowledged_page_is_not_offered_again(self, shadow) -> None:
        page = await fetch_overreach_flags(shadow, PROJECT)
        await acknowledge_overreach_flags(shadow, page)
        assert _keys(await fetch_overreach_flags(shadow, PROJECT)) == []
        assert _keys(await fetch_overreach_flags(shadow, OTHER)) == [('ep-x', 'x1')]

    @pytest.mark.asyncio
    async def test_an_unacknowledged_page_is_offered_again(self, shadow) -> None:
        first = await fetch_overreach_flags(shadow, PROJECT)
        assert await fetch_overreach_flags(shadow, PROJECT) == first

    @pytest.mark.asyncio
    async def test_a_missing_page_or_shadow_acknowledges_nothing(self) -> None:
        recording = _RecordingShadow()
        await acknowledge_overreach_flags(recording, None)
        await acknowledge_overreach_flags(None, OverreachFlagPage(graph=PROJECT, flags=(), omitted=0))
        assert recording.acknowledged == []

    @pytest.mark.asyncio
    async def test_a_failing_acknowledgement_is_warned_never_raised(self, caplog) -> None:
        page = OverreachFlagPage(graph=PROJECT, flags=(), omitted=0)
        with caplog.at_level(logging.WARNING, logger=MODULE_LOGGER):
            await acknowledge_overreach_flags(_RecordingShadow(error=RuntimeError('disk full')), page)
        assert _warned(caplog)


class TestCarryOver:
    @pytest.mark.asyncio
    async def test_flags_past_the_page_reach_the_next_cycle(self, tmp_path) -> None:
        overflow = 5
        count = FLAGS_SECTION_LIMIT + overflow
        store = await OverreachShadowStore.open(
            tmp_path / 'burst', clock=_clock(*(T0 + timedelta(seconds=i) for i in range(count))),
        )
        for i in range(count):
            await store.record(episode=_episode(PROJECT, f'ep-{i:02d}'), judgements=[
                _judged(f'e-{i:02d}', 'overreach'),
            ])
        runner = _shadow(store)
        try:
            first = await fetch_overreach_flags(runner, PROJECT)
            await acknowledge_overreach_flags(runner, first)
            second = await fetch_overreach_flags(runner, PROJECT)
            await acknowledge_overreach_flags(runner, second)
            third = await fetch_overreach_flags(runner, PROJECT)
        finally:
            await runner.close()
        assert first is not None and second is not None
        assert (len(first.flags), first.omitted) == (FLAGS_SECTION_LIMIT, overflow)
        assert f'+{overflow} more' in render_overreach_flags_section(first)
        assert (len(second.flags), second.omitted) == (overflow, 0)
        offered = [flag.edge_uuid for page in (first, second) for flag in page.flags]
        assert offered == [f'e-{i:02d}' for i in range(count)]
        assert _keys(third) == []


def _flag(edge_uuid: str, *, fact: str | None = None, rationale: str | None = None) -> OverreachFlag:
    return OverreachFlag(
        episode_uuid=f'ep-of-{edge_uuid}', edge_uuid=edge_uuid,
        fact=fact if fact is not None else f'fact of {edge_uuid}',
        rationale=rationale if rationale is not None else f'span for {edge_uuid}',
        judged_at=T1, prompt_version=PROMPT_VERSION,
    )


PAGE = OverreachFlagPage(graph=PROJECT, flags=(_flag('e1'), _flag('e2')), omitted=0)


class TestRender:
    @pytest.mark.parametrize('page', [None, OverreachFlagPage(graph=PROJECT, flags=(), omitted=0)])
    def test_nothing_to_hand_over_renders_nothing(self, page) -> None:
        assert render_overreach_flags_section(page) == ''

    def test_the_header_and_count_lead(self) -> None:
        section = render_overreach_flags_section(PAGE)
        assert section.lstrip('\n').startswith(f'{FLAGS_SECTION_HEADER} (2)')

    def test_each_flag_line_carries_edge_episode_fact_and_rationale(self) -> None:
        lines = render_overreach_flags_section(PAGE).splitlines()
        for edge in ('e1', 'e2'):
            (line,) = [line for line in lines if edge in line and 'fact of' in line]
            assert f'ep-of-{edge}' in line and f'span for {edge}' in line

    def test_long_text_is_truncated(self) -> None:
        fact, rationale = 'F' * 5000, 'R' * 5000
        section = render_overreach_flags_section(
            OverreachFlagPage(
                graph=PROJECT, flags=(_flag('e1', fact=fact, rationale=rationale),), omitted=0,
            ),
        )
        (line,) = [line for line in section.splitlines() if 'e1' in line]
        assert 'F' * 100 in line and 'R' * 100 in line
        assert fact not in line and rationale not in line
        assert len(line) < 1500

    def test_a_multiline_fact_stays_on_its_flag_line(self) -> None:
        section = render_overreach_flags_section(
            OverreachFlagPage(graph=PROJECT, flags=(_flag('e1', fact='first\nsecond'),), omitted=0),
        )
        (line,) = [line for line in section.splitlines() if 'e1' in line]
        assert 'first' in line and 'second' in line

    def test_omitted_flags_are_counted(self) -> None:
        section = render_overreach_flags_section(
            OverreachFlagPage(graph=PROJECT, flags=PAGE.flags, omitted=3),
        )
        assert '+3 more' in section
        assert section.lstrip('\n').startswith(f'{FLAGS_SECTION_HEADER} (5)')

    def test_it_names_the_only_sanctioned_action(self) -> None:
        section = render_overreach_flags_section(PAGE)
        assert 'update_edge' in section and 'invalid_at' in section


def _stage(flags: OverreachFlagPage | None) -> MemoryConsolidator:
    stage = make_consolidator(project_root='/home/leo/src/test_proj')
    stage.overreach_flags = flags
    return stage


async def _payload(stage: MemoryConsolidator) -> str:
    return await stage.assemble_payload(
        events=[], watermark=Watermark(project_id='test_project'), prior_reports=[],
    )


def _assembled(stage: MemoryConsolidator) -> MemoryConsolidator:
    stage.assembled_payload = AssembledPayload(events=[], context_items={})
    return stage


class TestConsolidatorWiring:
    @pytest.mark.asyncio
    async def test_the_time_windowed_payload_carries_the_section(self) -> None:
        assert FLAGS_SECTION_HEADER in await _payload(_stage(PAGE))

    @pytest.mark.asyncio
    async def test_the_assembled_payload_carries_the_section(self) -> None:
        assert FLAGS_SECTION_HEADER in await _payload(_assembled(_stage(PAGE)))

    @pytest.mark.asyncio
    async def test_the_remediation_payload_does_not(self) -> None:
        stage = _stage(PAGE)
        stage.remediation_findings = [{'description': 'fix this'}]
        assert FLAGS_SECTION_HEADER not in await _payload(stage)

    @pytest.mark.asyncio
    async def test_no_flags_means_no_section_in_either_builder(self) -> None:
        assert FLAGS_SECTION_HEADER not in await _payload(_stage(None))
        assert FLAGS_SECTION_HEADER not in await _payload(_assembled(_stage(None)))

    def test_the_class_default_is_none(self) -> None:
        assert MemoryConsolidator.overreach_flags is None


def _tier() -> MagicMock:
    tier = MagicMock()
    tier.episode_limit, tier.memory_limit = 5, 10
    return tier


class TestHarnessWiring:
    def test_configure_sets_the_flags(self) -> None:
        stage = MagicMock(spec=MemoryConsolidator)
        ReconciliationHarness._configure_consolidator(stage, _tier(), overreach_flags=PAGE)
        assert stage.overreach_flags is PAGE

    def test_configure_defaults_them_to_none(self) -> None:
        stage = MagicMock(spec=MemoryConsolidator)
        ReconciliationHarness._configure_consolidator(stage, _tier())
        assert stage.overreach_flags is None

    @pytest.mark.asyncio
    async def test_a_completed_cycle_acknowledges_what_stage_1_was_shown(
        self, cycle_harness, shadow, stage_1_pages,
    ) -> None:
        cycle_harness.memory.overreach_shadow = shadow
        first = await cycle_harness.run_full_cycle(PROJECT, 'test-trigger')
        second = await cycle_harness.run_full_cycle(PROJECT, 'test-trigger')
        assert (first.status, second.status) == (RunStatus.completed, RunStatus.completed)
        assert [_keys(page) for page in stage_1_pages] == [
            [('ep-a', 'a1'), ('ep-b', 'b1'), ('ep-b', 'b2')], [],
        ]

    @pytest.mark.asyncio
    async def test_a_failed_cycle_leaves_its_flags_for_the_next(
        self, cycle_harness, shadow, stage_1_pages,
    ) -> None:
        cycle_harness.memory.overreach_shadow = shadow
        stage_2_down = AsyncMock(side_effect=RuntimeError('stage 2 down'))
        with patch.object(TaskKnowledgeSync, 'run', stage_2_down), pytest.raises(RuntimeError):
            await cycle_harness.run_full_cycle(PROJECT, 'test-trigger')
        await cycle_harness.run_full_cycle(PROJECT, 'test-trigger')
        assert [_keys(page) for page in stage_1_pages] == [
            [('ep-a', 'a1'), ('ep-b', 'b1'), ('ep-b', 'b2')],
        ] * 2

    @pytest.mark.asyncio
    async def test_no_shadow_runs_the_cycle_without_flags(self, cycle_harness, stage_1_pages) -> None:
        cycle_harness.memory.overreach_shadow = None
        run = await cycle_harness.run_full_cycle(PROJECT, 'test-trigger')
        assert run.status == RunStatus.completed
        assert stage_1_pages == [None]


@pytest_asyncio.fixture
async def cycle_harness(tmp_path):
    journal = ReconciliationJournal(tmp_path / 'journal')
    await journal.initialize()
    buffer = EventBuffer(
        db_path=tmp_path / 'events.db', buffer_size_threshold=2, max_staleness_seconds=3600,
    )
    await buffer.initialize()
    memory = AsyncMock()
    memory.search = AsyncMock(return_value=[])
    memory.get_episodes = AsyncMock(return_value=[])
    memory.get_status = AsyncMock(return_value={
        'graphiti': {'connected': True}, 'mem0': {'connected': True}, 'projects': {},
    })
    memory.get_entity = AsyncMock(return_value={'nodes': [], 'edges': []})
    memory.get_memories_by_metadata = AsyncMock(return_value=[])
    memory.mem0.get_all = AsyncMock(return_value={'results': []})
    config = FusedMemoryConfig(reconciliation=ReconciliationConfig(
        enabled=True, explore_codebase_root='/tmp/test', agent_llm_provider='anthropic',
        agent_llm_model='claude-sonnet-4-20250514', judge_enabled=False,
    ))
    taskmaster = AsyncMock()
    taskmaster.get_tasks = AsyncMock(return_value={'tasks': []})
    yield ReconciliationHarness(
        memory_service=memory, taskmaster=taskmaster, journal=journal, event_buffer=buffer,
        config=config, known_projects={PROJECT: '/tmp/test-project'},
    )
    await buffer.close()
    await journal.close()


@pytest.fixture
def stage_1_pages():
    """Stub every stage's run; record the flag page Stage 1 was configured with."""
    pages: list[OverreachFlagPage | None] = []

    async def fake_run(self, events, watermark, prior_reports, run_id, model=None,
                       resume_session_id=None) -> StageReport:
        if isinstance(self, MemoryConsolidator):
            pages.append(self.overreach_flags)
        now = datetime.now(UTC)
        return StageReport(stage=self.stage_id, started_at=now, completed_at=now)

    with (
        patch.object(MemoryConsolidator, 'run', fake_run),
        patch.object(TaskKnowledgeSync, 'run', fake_run),
        patch.object(IntegrityCheck, 'run', fake_run),
    ):
        yield pages
