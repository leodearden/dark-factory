"""Tests for fused_memory.services.overreach_shadow (task 6137, design.md §6.1 Stage S).

A real SQLite sidecar in tmp_path; hand-written fakes for the LLM and the
surviving-edge read, so every call shape is asserted precisely.
"""
from __future__ import annotations

import asyncio
import dataclasses
import logging
import sqlite3
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import pytest
import pytest_asyncio
from pydantic import ValidationError

from fused_memory.config.schema import FusedMemoryConfig, OverreachGuardConfig
from fused_memory.models.enums import MemoryCategory
from fused_memory.services.overreach_guard import (
    PROMPT_VERSION,
    EdgeJudgement,
    JudgeEdge,
    matching_classifiers,
)
from fused_memory.services.overreach_shadow import (
    SIDECAR_FILENAME,
    JudgedEpisode,
    OverreachFlagPage,
    OverreachShadow,
    OverreachShadowStore,
    ShadowJob,
    ShadowOutcome,
    open_overreach_shadow,
    shadow_job_from_write,
)

T0 = datetime(2026, 10, 17, 12, 0, 0, tzinfo=UTC)


class _Clock:
    """Hands out the given instants in order, then repeats the last."""

    def __init__(self, *instants: datetime) -> None:
        self.instants = list(instants)

    def __call__(self) -> datetime:
        return self.instants.pop(0) if len(self.instants) > 1 else self.instants[0]


def _episode(uuid: str = 'ep1', *, graph: str = 'g', outcome: ShadowOutcome = ShadowOutcome.judged,
             minted: int = 2, error: str | None = None) -> JudgedEpisode:
    return JudgedEpisode(
        graph=graph, episode_uuid=uuid, judge_model='judge-m', prompt_version=PROMPT_VERSION,
        classifiers=('category_decisions', 'ruling_lexeme_head'), minted_count=minted,
        outcome=outcome, error=error,
    )


def _judged(edge_uuid: str, label: str, *, fact: str = '') -> tuple[JudgeEdge, EdgeJudgement]:
    return (
        JudgeEdge(uuid=edge_uuid, fact=fact or f'fact of {edge_uuid}', source_name='S', target_name='T'),
        EdgeJudgement(edge_uuid=edge_uuid, label=label, rationale=f'span for {edge_uuid}'),
    )


@pytest_asyncio.fixture
async def store(tmp_path: Path):
    opened = await OverreachShadowStore.open(tmp_path / 'sidecar')
    yield opened
    await opened.close()


# --------------------------------------------------------------------------- #
# The config section
# --------------------------------------------------------------------------- #


class TestConfigSection:
    @pytest.fixture(autouse=True)
    def _pin_config_path(self, tmp_path, monkeypatch):
        monkeypatch.setenv('CONFIG_PATH', str(tmp_path / 'missing.yaml'))

    def test_the_code_defaults(self) -> None:
        section = FusedMemoryConfig().overreach_guard
        assert isinstance(section, OverreachGuardConfig)
        assert section.shadow_enabled is True
        assert section.judge_timeout_seconds == 60.0

    @pytest.mark.parametrize('timeout', [0, -1.0])
    def test_a_non_positive_judge_timeout_is_rejected(self, timeout) -> None:
        with pytest.raises(ValidationError):
            OverreachGuardConfig(judge_timeout_seconds=timeout)

    def test_the_section_is_a_bare_submodel(self) -> None:
        assert FusedMemoryConfig.model_fields['overreach_guard'].annotation is OverreachGuardConfig


# --------------------------------------------------------------------------- #
# The sidecar store
# --------------------------------------------------------------------------- #


class TestStore:
    @pytest.mark.asyncio
    async def test_the_sidecar_is_named_in_its_data_dir(self, tmp_path: Path) -> None:
        opened = await OverreachShadowStore.open(tmp_path / 'data')
        try:
            assert (tmp_path / 'data' / SIDECAR_FILENAME).is_file()
        finally:
            await opened.close()

    @pytest.mark.asyncio
    async def test_judgements_round_trip(self, store) -> None:
        await store.record(episode=_episode(), judgements=[
            _judged('e1', 'holding'), _judged('e2', 'overreach'),
        ])
        got = await store.judgements_for([('g', 'e1'), ('g', 'e2'), ('g', 'missing'), ('other', 'e1')])
        assert set(got) == {('g', 'e1'), ('g', 'e2')}
        stored = got[('g', 'e2')]
        assert (stored.episode_uuid, stored.label, stored.rationale) == ('ep1', 'overreach', 'span for e2')
        assert (stored.judge_model, stored.prompt_version) == ('judge-m', PROMPT_VERSION)

    @pytest.mark.asyncio
    async def test_an_episode_row_round_trips(self, store) -> None:
        await store.record(episode=_episode(), judgements=[])
        assert await store.judged_episode('g', 'ep1') == _episode()
        assert await store.judged_episode('g', 'never') is None

    @pytest.mark.asyncio
    async def test_a_failed_row_keeps_its_error(self, store) -> None:
        failed = _episode(outcome=ShadowOutcome.failed, minted=0, error='RuntimeError: down')
        await store.record(episode=failed, judgements=[])
        assert await store.judged_episode('g', 'ep1') == failed

    def test_the_outcomes(self) -> None:
        assert {o.value for o in ShadowOutcome} == {'judged', 'no_minted', 'failed'}

    @pytest.mark.asyncio
    async def test_a_rejudged_edge_keeps_its_first_judgement(self, store) -> None:
        await store.record(episode=_episode('ep1'), judgements=[_judged('e1', 'holding')])
        await store.record(episode=_episode('ep1'), judgements=[_judged('e1', 'overreach')])
        assert (await store.judgements_for([('g', 'e1')]))[('g', 'e1')].label == 'holding'


async def _seed_flags(tmp_path: Path) -> OverreachShadowStore:
    clock = _Clock(T0, T0 + timedelta(minutes=1), T0 + timedelta(minutes=2))
    opened = await OverreachShadowStore.open(tmp_path / 'flags', clock=clock)
    await opened.record(episode=_episode('ep-a'), judgements=[
        _judged('a2', 'overreach'), _judged('a1', 'overreach'), _judged('a3', 'holding'),
    ])
    await opened.record(episode=_episode('ep-b'), judgements=[_judged('b1', 'overreach')])
    await opened.record(episode=_episode('ep-x', graph='other'), judgements=[_judged('x1', 'overreach')])
    return opened


class TestPendingFlags:
    @pytest.mark.asyncio
    async def test_only_this_graphs_overreach_flags_in_order(self, tmp_path: Path) -> None:
        opened = await _seed_flags(tmp_path)
        try:
            page = await opened.pending_overreach_flags('g', 10)
        finally:
            await opened.close()
        assert isinstance(page, OverreachFlagPage)
        assert page.graph == 'g'
        assert [f.edge_uuid for f in page.flags] == ['a1', 'a2', 'b1']
        assert (page.omitted, page.dropped) == (0, ())

    @pytest.mark.asyncio
    async def test_a_flag_carries_what_stage_1_needs(self, tmp_path: Path) -> None:
        opened = await _seed_flags(tmp_path)
        try:
            page = await opened.pending_overreach_flags('g', 10)
        finally:
            await opened.close()
        flag = page.flags[0]
        assert (flag.episode_uuid, flag.edge_uuid, flag.fact) == ('ep-a', 'a1', 'fact of a1')
        assert (flag.rationale, flag.prompt_version) == ('span for a1', PROMPT_VERSION)
        assert flag.judged_at == T0 and flag.judged_at.tzinfo is not None

    @pytest.mark.asyncio
    async def test_the_page_is_capped_and_counts_the_rest(self, tmp_path: Path) -> None:
        opened = await _seed_flags(tmp_path)
        try:
            page = await opened.pending_overreach_flags('g', 2)
        finally:
            await opened.close()
        assert [f.edge_uuid for f in page.flags] == ['a1', 'a2']
        assert page.omitted == 1

    @pytest.mark.asyncio
    async def test_a_handed_over_flag_leaves_only_its_own_graphs_pending_set(self, tmp_path: Path) -> None:
        opened = await _seed_flags(tmp_path)
        try:
            await opened.record_handoffs('g', ['a1', 'b1', 'x1'])
            await opened.record_handoffs('g', [])
            pending = await opened.pending_overreach_flags('g', 10)
            other = await opened.pending_overreach_flags('other', 10)
        finally:
            await opened.close()
        assert [f.edge_uuid for f in pending.flags] == ['a2']
        assert [f.edge_uuid for f in other.flags] == ['x1']

    @pytest.mark.asyncio
    async def test_the_rest_follows_once_the_page_is_handed_over(self, tmp_path: Path) -> None:
        opened = await _seed_flags(tmp_path)
        try:
            first = await opened.pending_overreach_flags('g', 2)
            await opened.record_handoffs('g', first.handed_over())
            second = await opened.pending_overreach_flags('g', 2)
        finally:
            await opened.close()
        assert ([f.edge_uuid for f in second.flags], second.omitted) == (['b1'], 0)

    @pytest.mark.asyncio
    async def test_handoffs_persist_across_reopen(self, tmp_path: Path) -> None:
        first = await _seed_flags(tmp_path)
        await first.record_handoffs('g', ['a1', 'a2'])
        await first.close()
        second = await OverreachShadowStore.open(tmp_path / 'flags')
        try:
            page = await second.pending_overreach_flags('g', 10)
        finally:
            await second.close()
        assert [f.edge_uuid for f in page.flags] == ['b1']

    @pytest.mark.asyncio
    async def test_instants_in_one_second_order_as_instants(self, tmp_path: Path) -> None:
        whole, fraction = T0, T0 + timedelta(microseconds=500_000)
        opened = await OverreachShadowStore.open(tmp_path / 'same-second', clock=_Clock(fraction, whole))
        try:
            await opened.record(episode=_episode('ep-late'), judgements=[_judged('late', 'overreach')])
            await opened.record(episode=_episode('ep-whole'), judgements=[_judged('whole', 'overreach')])
            every = await opened.pending_overreach_flags('g', 10)
        finally:
            await opened.close()
        assert [f.edge_uuid for f in every.flags] == ['whole', 'late']
        assert [f.judged_at for f in every.flags] == [whole, fraction]


class TestStoreLifecycle:
    @pytest.mark.asyncio
    async def test_concurrent_records_both_land(self, store) -> None:
        await asyncio.gather(
            store.record(episode=_episode('ep1'), judgements=[_judged('e1', 'holding')]),
            store.record(episode=_episode('ep2'), judgements=[_judged('e2', 'overreach')]),
        )
        assert set(await store.judgements_for([('g', 'e1'), ('g', 'e2')])) == {('g', 'e1'), ('g', 'e2')}

    @pytest.mark.asyncio
    async def test_the_read_only_view_reads_and_refuses_writes(self, tmp_path: Path) -> None:
        writer = await OverreachShadowStore.open(tmp_path / 'ro')
        await writer.record(episode=_episode(), judgements=[_judged('e1', 'overreach')])
        await writer.close()
        reader = await OverreachShadowStore.open_read_only(tmp_path / 'ro' / SIDECAR_FILENAME)
        try:
            assert set(await reader.judgements_for([('g', 'e1')])) == {('g', 'e1')}
            with pytest.raises(sqlite3.OperationalError):
                await reader.record(episode=_episode('ep2'), judgements=[])
        finally:
            await reader.close()

    @pytest.mark.asyncio
    async def test_the_read_only_view_never_creates_a_file(self, tmp_path: Path) -> None:
        missing = tmp_path / 'nowhere' / SIDECAR_FILENAME
        with pytest.raises(FileNotFoundError):
            await OverreachShadowStore.open_read_only(missing)
        assert not missing.exists()

    @pytest.mark.asyncio
    async def test_rows_persist_across_reopen_and_close_is_idempotent(self, tmp_path: Path) -> None:
        first = await OverreachShadowStore.open(tmp_path / 'persist')
        await first.record(episode=_episode(), judgements=[_judged('e1', 'overreach')])
        await first.close()
        await first.close()
        second = await OverreachShadowStore.open(tmp_path / 'persist')
        try:
            assert await second.judged_episode('g', 'ep1') == _episode()
        finally:
            await second.close()


# --------------------------------------------------------------------------- #
# The runner
# --------------------------------------------------------------------------- #

JOB = ShadowJob(
    graph='g', episode_uuid='ep1', content='RULING (Leo, 2026-09-01): narrow the gate',
    edge_uuids=('e1', 'e2', 'e3', 'gone'), classifiers=('category_decisions', 'header_ruling'),
)
def _edge_row(uuid: str, *, source: str = 'S', target: str = 'T', episodes: tuple[str, ...] = (),
              active: bool = True) -> dict[str, Any]:
    return {'uuid': uuid, 'fact': f'fact {uuid}', 'source_name': source, 'target_name': target,
            'episodes': list(episodes), 'active': active}


SURVIVORS = [
    _edge_row('e1', source='A', target='B', episodes=('ep1',)),
    _edge_row('e2', source='B', target='C', episodes=('ep1', 'later')),
    _edge_row('e3', source='C', target='D', episodes=('older', 'ep1')),
]


class _EdgeReader:
    def __init__(self, rows: list[dict[str, Any]] | None = None, error: Exception | None = None) -> None:
        self.rows = SURVIVORS if rows is None else rows
        self.error = error
        self.calls: list[tuple[list[str], str]] = []

    async def __call__(self, uuids, *, group_id):
        self.calls.append((list(uuids), group_id))
        if self.error is not None:
            raise self.error
        return [row for row in self.rows if row['uuid'] in uuids]


class _LLM:
    """Answers with *labels* in E-order; or raises; or blocks until released."""

    def __init__(self, labels: tuple[str, ...] = ('holding', 'overreach'), *, reply: Any = None,
                 error: Exception | None = None, block: bool = False) -> None:
        self.labels, self.reply, self.error, self.block = labels, reply, error, block
        self.calls = 0
        self.entered = asyncio.Event()
        self.release = asyncio.Event()

    async def generate_response(self, messages, response_model=None, max_tokens=None, **kwargs):
        self.calls += 1
        self.entered.set()
        if self.block:
            await self.release.wait()
        if self.error is not None:
            raise self.error
        if self.reply is not None:
            return self.reply
        return {'judgements': [
            {'edge': f'E{i}', 'label': label, 'rationale': f'span {i}'}
            for i, label in enumerate(self.labels, start=1)
        ]}


class _Settings:
    def __init__(self, **values: Any) -> None:
        self.current = OverreachGuardConfig(**values)

    def __call__(self) -> OverreachGuardConfig:
        return self.current


def _runner(store, *, llm=None, reader=None, settings=None, max_pending: int = 64) -> OverreachShadow:
    return OverreachShadow(
        store, llm or _LLM(), 'judge-m', reader or _EdgeReader(),
        settings or _Settings(judge_timeout_seconds=5.0), max_pending=max_pending,
    )


class _BrokenStore(OverreachShadowStore):
    async def record(self, *, episode, judgements) -> None:
        raise RuntimeError('disk full')


class TestRunner:
    @pytest.mark.asyncio
    async def test_a_job_judges_only_the_surviving_minted_edges(self, store) -> None:
        llm, reader = _LLM(), _EdgeReader()
        runner = _runner(store, llm=llm, reader=reader)
        assert runner.submit(JOB) is True
        await runner.drain()
        assert reader.calls == [(['e1', 'e2', 'e3', 'gone'], 'g')]
        assert llm.calls == 1
        assert await store.judged_episode('g', 'ep1') == JudgedEpisode(
            graph='g', episode_uuid='ep1', judge_model='judge-m', prompt_version=PROMPT_VERSION,
            classifiers=JOB.classifiers, minted_count=2, outcome=ShadowOutcome.judged, error=None,
        )
        got = await store.judgements_for([('g', 'e1'), ('g', 'e2'), ('g', 'e3'), ('g', 'gone')])
        assert {key: j.label for key, j in got.items()} == {('g', 'e1'): 'holding', ('g', 'e2'): 'overreach'}

    @pytest.mark.asyncio
    async def test_nothing_minted_means_no_call_and_a_no_minted_row(self, store) -> None:
        llm = _LLM()
        runner = _runner(store, llm=llm, reader=_EdgeReader(rows=[SURVIVORS[2]]))
        runner.submit(JOB)
        await runner.drain()
        assert llm.calls == 0
        row = await store.judged_episode('g', 'ep1')
        assert row is not None and (row.outcome, row.minted_count) == (ShadowOutcome.no_minted, 0)

    @pytest.mark.parametrize(('llm', 'reader', 'error_type'), [
        pytest.param(_LLM(error=RuntimeError('down')), _EdgeReader(), 'RuntimeError', id='llm-raises'),
        pytest.param(_LLM(reply={'judgements': []}), _EdgeReader(), 'JudgeReplyError', id='malformed-reply'),
        pytest.param(_LLM(), _EdgeReader(error=ConnectionError('falkor')), 'ConnectionError', id='read-raises'),
    ])
    @pytest.mark.asyncio
    async def test_a_failure_is_recorded_and_never_raised(self, store, llm, reader, error_type) -> None:
        runner = _runner(store, llm=llm, reader=reader)
        assert runner.submit(JOB) is True
        await runner.drain()
        row = await store.judged_episode('g', 'ep1')
        assert row is not None and row.outcome is ShadowOutcome.failed
        assert row.error is not None and row.error.startswith(error_type)
        assert await store.judgements_for([('g', 'e1'), ('g', 'e2')]) == {}

    @pytest.mark.asyncio
    async def test_a_slow_judge_times_out_as_a_failure(self, store) -> None:
        runner = _runner(store, llm=_LLM(block=True), settings=_Settings(judge_timeout_seconds=0.05))
        runner.submit(JOB)
        await asyncio.wait_for(runner.drain(), 2.0)
        row = await store.judged_episode('g', 'ep1')
        assert row is not None and row.outcome is ShadowOutcome.failed
        assert row.error is not None and row.error.startswith('TimeoutError')

    @pytest.mark.asyncio
    async def test_the_kill_switch_is_read_live(self, store) -> None:
        llm, reader, settings = _LLM(), _EdgeReader(), _Settings(shadow_enabled=False)
        runner = _runner(store, llm=llm, reader=reader, settings=settings)
        assert runner.submit(JOB) is False
        await runner.drain()
        assert (reader.calls, llm.calls) == ([], 0)
        assert await store.judged_episode('g', 'ep1') is None
        settings.current = OverreachGuardConfig(shadow_enabled=True)
        assert runner.submit(JOB) is True
        await runner.drain()
        assert await store.judged_episode('g', 'ep1') is not None

    @pytest.mark.asyncio
    async def test_jobs_past_the_pending_bound_are_dropped(self, store, caplog) -> None:
        llm = _LLM(block=True)
        runner = _runner(store, llm=llm, max_pending=1)
        assert runner.submit(JOB) is True
        await llm.entered.wait()
        with caplog.at_level(logging.WARNING, logger='fused_memory.services.overreach_shadow'):
            assert runner.submit(dataclasses.replace(JOB, episode_uuid='ep2')) is False
        assert any(r.levelno == logging.WARNING for r in caplog.records)
        llm.release.set()
        await runner.drain()

    @pytest.mark.asyncio
    async def test_a_store_failure_is_logged_and_swallowed(self, tmp_path: Path, caplog) -> None:
        broken = await _BrokenStore.open(tmp_path / 'broken')
        runner = _runner(broken)
        try:
            with caplog.at_level(logging.WARNING, logger='fused_memory.services.overreach_shadow'):
                assert runner.submit(JOB) is True
                await runner.drain()
            assert any(r.exc_info and 'disk full' in str(r.exc_info[1]) for r in caplog.records)
        finally:
            await runner.close()

    @pytest.mark.asyncio
    async def test_close_cancels_an_in_flight_judge_and_closes_the_store(self, tmp_path: Path) -> None:
        data_dir = tmp_path / 'closing'
        owned = await OverreachShadowStore.open(data_dir)
        llm = _LLM(block=True)
        runner = _runner(owned, llm=llm)
        runner.submit(JOB)
        await llm.entered.wait()
        await asyncio.wait_for(runner.close(), 0.5)
        with pytest.raises(RuntimeError):
            await owned.judged_episode('g', 'ep1')
        tasks_before = asyncio.all_tasks()
        assert runner.submit(JOB) is False
        assert asyncio.all_tasks() == tasks_before
        await runner.close()
        reopened = await OverreachShadowStore.open(data_dir)
        try:
            assert await reopened.judged_episode('g', 'ep1') is None
        finally:
            await reopened.close()



class TestFlagPage:
    @pytest.mark.asyncio
    async def test_a_gone_or_retired_edge_is_dropped_from_the_page(self, tmp_path: Path) -> None:
        reader = _EdgeReader(rows=[_edge_row('a1'), _edge_row('b1', active=False)])
        runner = _runner(await _seed_flags(tmp_path), reader=reader)
        try:
            page = await runner.overreach_flag_page('g', 10)
        finally:
            await runner.close()
        assert reader.calls == [(['a1', 'a2', 'b1'], 'g')]
        assert [f.edge_uuid for f in page.flags] == ['a1']
        assert page.dropped == ('a2', 'b1')
        assert page.handed_over() == ('a1', 'a2', 'b1')

    @pytest.mark.asyncio
    async def test_an_acknowledged_page_is_never_offered_again(self, tmp_path: Path) -> None:
        reader = _EdgeReader(rows=[_edge_row('a1'), _edge_row('a2'), _edge_row('b1', active=False)])
        runner = _runner(await _seed_flags(tmp_path), reader=reader)
        try:
            first = await runner.overreach_flag_page('g', 10)
            again = await runner.overreach_flag_page('g', 10)
            await runner.acknowledge(first)
            after = await runner.overreach_flag_page('g', 10)
        finally:
            await runner.close()
        assert again == first
        assert after == OverreachFlagPage(graph='g', flags=(), omitted=0)

    @pytest.mark.asyncio
    async def test_nothing_pending_reads_no_graph(self, store) -> None:
        reader = _EdgeReader()
        page = await _runner(store, reader=reader).overreach_flag_page('g', 10)
        assert (page.flags, reader.calls) == ((), [])


class TestShadowJobFromWrite:
    def test_a_job_from_an_add_episode_result(self) -> None:
        result = SimpleNamespace(
            episode=SimpleNamespace(uuid='ep-1'),
            edges=[SimpleNamespace(uuid='e1'), SimpleNamespace(uuid='e2')],
        )
        content = 'RULING (Leo, 2026-09-01, esc-1-2): x'
        category = MemoryCategory.decisions_and_rationale
        job = shadow_job_from_write(result, group_id='g', content=content, category=category)
        assert job == ShadowJob(
            graph='g', episode_uuid='ep-1', content=content, edge_uuids=('e1', 'e2'),
            classifiers=matching_classifiers(category, content),
        )

    @pytest.mark.parametrize('result', [
        pytest.param(None, id='no-result'),
        pytest.param(SimpleNamespace(episode=None, edges=[]), id='no-episode'),
        pytest.param(MagicMock(), id='non-str-uuid'),
    ])
    def test_an_unreadable_result_builds_no_job(self, result) -> None:
        job = shadow_job_from_write(
            result, group_id='g', content='x', category=MemoryCategory.decisions_and_rationale,
        )
        assert job is None


class TestOpenOverreachShadow:
    @pytest.mark.asyncio
    async def test_no_llm_means_no_shadow_and_no_sidecar(self, mock_config, tmp_path, caplog) -> None:
        mock_config.queue.data_dir = str(tmp_path)
        mock_config.llm.providers.openai.api_key = ''
        with caplog.at_level(logging.WARNING, logger='fused_memory.services.overreach_shadow'):
            shadow = await open_overreach_shadow(mock_config, read_edges=_EdgeReader())
        assert shadow is None
        assert any(r.levelno == logging.WARNING for r in caplog.records)
        assert not (tmp_path / SIDECAR_FILENAME).exists()

    @pytest.mark.asyncio
    async def test_a_configured_llm_opens_the_sidecar(self, mock_config, tmp_path) -> None:
        mock_config.queue.data_dir = str(tmp_path)
        shadow = await open_overreach_shadow(mock_config, read_edges=_EdgeReader())
        try:
            assert isinstance(shadow, OverreachShadow)
            assert shadow.judge_model == mock_config.llm.model
            assert (tmp_path / SIDECAR_FILENAME).is_file()
        finally:
            if shadow is not None:
                await shadow.close()
