"""The O4 handoff: shadow overreach flags reach recon Stage 1 (task 6137, design.md §6.1).

A real OverreachShadow over a real sidecar in tmp_path backs the fetch; its
LLM double raises, so a fetch that judged anything would fail loudly.
"""
from __future__ import annotations

import logging
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
import pytest_asyncio

from fused_memory.config.schema import OverreachGuardConfig
from fused_memory.models.reconciliation import AssembledPayload, Watermark
from fused_memory.reconciliation.harness import ReconciliationHarness
from fused_memory.reconciliation.overreach_flags import (
    FLAGS_SECTION_HEADER,
    FLAGS_SECTION_LIMIT,
    fetch_overreach_flags,
    render_overreach_flags_section,
)
from fused_memory.reconciliation.stages.memory_consolidator import MemoryConsolidator
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


async def _never_reads(uuids, *, group_id):
    raise AssertionError('fetching flags must never read the graph')


def _clock(*instants: datetime) -> Callable[[], datetime]:
    queue = list(instants)
    return lambda: queue.pop(0)


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


@pytest_asyncio.fixture
async def shadow(tmp_path):
    store = await OverreachShadowStore.open(tmp_path / 'sidecar', clock=_clock(T0, T1, T1))
    await store.record(episode=_episode(PROJECT, 'ep-a'), judgements=[
        _judged('a1', 'overreach'), _judged('a2', 'holding'),
    ])
    await store.record(episode=_episode(PROJECT, 'ep-b'), judgements=[
        _judged('b2', 'overreach'), _judged('b1', 'overreach'),
    ])
    await store.record(episode=_episode(OTHER, 'ep-x'), judgements=[_judged('x1', 'overreach')])
    runner = OverreachShadow(
        store, _NeverJudges(), 'judge-m', _never_reads, lambda: OverreachGuardConfig(),
    )
    yield runner
    await runner.close()


def _keys(page: OverreachFlagPage | None) -> list[tuple[str, str]]:
    assert page is not None
    return [(flag.episode_uuid, flag.edge_uuid) for flag in page.flags]


class _RecordingShadow:
    def __init__(self, result: Any = None, error: Exception | None = None) -> None:
        self.result = result if result is not None else OverreachFlagPage(flags=(), omitted=0)
        self.error = error
        self.calls: list[tuple[str, datetime | None, int]] = []

    async def overreach_flags_since(self, graph, since, limit):
        self.calls.append((graph, since, limit))
        if self.error is not None:
            raise self.error
        return self.result


def _warned(caplog) -> bool:
    return any(r.name == MODULE_LOGGER and r.levelno == logging.WARNING for r in caplog.records)


class TestFetch:
    @pytest.mark.asyncio
    async def test_only_this_projects_flags_in_order(self, shadow) -> None:
        page = await fetch_overreach_flags(SimpleNamespace(overreach_shadow=shadow), PROJECT, since=None)
        assert _keys(page) == [('ep-a', 'a1'), ('ep-b', 'b1'), ('ep-b', 'b2')]

    @pytest.mark.asyncio
    async def test_only_flags_judged_after_since(self, shadow) -> None:
        page = await fetch_overreach_flags(SimpleNamespace(overreach_shadow=shadow), PROJECT, since=T0)
        assert _keys(page) == [('ep-b', 'b1'), ('ep-b', 'b2')]

    @pytest.mark.asyncio
    async def test_it_asks_for_one_bounded_page(self) -> None:
        recording = _RecordingShadow()
        await fetch_overreach_flags(SimpleNamespace(overreach_shadow=recording), PROJECT, since=T0)
        assert recording.calls == [(PROJECT, T0, FLAGS_SECTION_LIMIT)]
        assert FLAGS_SECTION_LIMIT == 40

    @pytest.mark.parametrize('memory_service', [
        pytest.param(SimpleNamespace(), id='no-attribute'),
        pytest.param(SimpleNamespace(overreach_shadow=None), id='none'),
        pytest.param(SimpleNamespace(overreach_shadow=_RecordingShadow(error=RuntimeError('db locked'))),
                     id='raises'),
        pytest.param(AsyncMock(), id='not-a-page'),
    ])
    @pytest.mark.asyncio
    async def test_an_unavailable_shadow_is_none_and_warned(self, memory_service, caplog) -> None:
        with caplog.at_level(logging.WARNING, logger=MODULE_LOGGER):
            assert await fetch_overreach_flags(memory_service, PROJECT, since=T0) is None
        assert _warned(caplog)


def _flag(edge_uuid: str, *, fact: str | None = None, rationale: str | None = None) -> OverreachFlag:
    return OverreachFlag(
        episode_uuid=f'ep-of-{edge_uuid}', edge_uuid=edge_uuid,
        fact=fact if fact is not None else f'fact of {edge_uuid}',
        rationale=rationale if rationale is not None else f'span for {edge_uuid}',
        judged_at=T1, prompt_version=PROMPT_VERSION,
    )


PAGE = OverreachFlagPage(flags=(_flag('e1'), _flag('e2')), omitted=0)


class TestRender:
    @pytest.mark.parametrize('page', [None, OverreachFlagPage(flags=(), omitted=0)])
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
            OverreachFlagPage(flags=(_flag('e1', fact=fact, rationale=rationale),), omitted=0),
        )
        (line,) = [line for line in section.splitlines() if 'e1' in line]
        assert 'F' * 100 in line and 'R' * 100 in line
        assert fact not in line and rationale not in line
        assert len(line) < 1500

    def test_a_multiline_fact_stays_on_its_flag_line(self) -> None:
        section = render_overreach_flags_section(
            OverreachFlagPage(flags=(_flag('e1', fact='first\nsecond'),), omitted=0),
        )
        (line,) = [line for line in section.splitlines() if 'e1' in line]
        assert 'first' in line and 'second' in line

    def test_omitted_flags_are_counted(self) -> None:
        section = render_overreach_flags_section(OverreachFlagPage(flags=PAGE.flags, omitted=3))
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
