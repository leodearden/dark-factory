"""The restate band (cosine >= t_high) files `restated` only when the write adds nothing.

A high cosine says a write is about the same thing as a stored record, not that
it says nothing new. So a write in the restate band is judged like a
middle-band write unless it repeats the band winner, or the winner's canonical,
near-verbatim. ``decide_band`` stays the cosine BAND label; ``route_band`` is
the routing ``triage_write`` acts on.
"""

from __future__ import annotations

import json
import logging
import types
from dataclasses import replace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from _sdk_client_stubs import sdk_client_double

from fused_memory.models.enums import MemoryCategory, SourceStore
from fused_memory.models.memory import MemoryResult
from fused_memory.server import tools
from fused_memory.server.grouped_read import AMENDMENT_KIND, PARENT_ID_KEY, SIGHTING_KIND
from fused_memory.server.tools import create_mcp_server
from fused_memory.server.write_triage import (
    CANONICAL_ID_KEY,
    OUTCOME_AMENDED,
    OUTCOME_JUDGE,
    OUTCOME_RESTATED,
    OUTCOME_STORED,
    ROUTED_KEY,
    AttachRefusal,
    BandDecision,
    TriageFailOpenCounter,
    TriageJudgeVerdict,
    canonical_id_of,
    decide_band,
    route_band,
    triage_write,
)
from fused_memory.server.write_triage_judge import (
    CANDIDATE_ID_KEY,
    VERDICT_KEY,
    select_judge_candidates,
)
from fused_memory.services.memory_service import RRF_K, SearchResults
from server.write_triage_harness import configure_triage_config, search_candidate

_T_HIGH = 0.90
_T_LOW = 0.70
_PROJECT = 'proj-7155'
_TRIAGE_LOGGER = 'fused_memory.server.write_triage'
_REFILED_PREFIX = 'write_triage judged restatement filed as amended'

_WRITE = 'The nightly verify of the ledger service finishes in 14m22s on the shared build host.'
_WRITE_REFLOWED = 'the nightly verify of the ledger service\nfinishes in 14m22s   on the shared build host.'
_PARAPHRASE = 'On the shared host the ledger verify takes about fourteen minutes each night.'


def _result(
    id_: str,
    cosine: float,
    *,
    content: str | None = None,
    extra_metadata: dict | None = None,
) -> MemoryResult:
    """A post-RRF record: the cosine lives in ``metadata['store_score']``."""
    return MemoryResult(
        id=id_,
        content=content if content is not None else f'content of {id_}',
        category=MemoryCategory.procedural_knowledge,
        source_store=SourceStore.mem0,
        relevance_score=1.0 / (RRF_K + 1),
        metadata={'store_score': cosine, **(extra_metadata or {})},
    )


def _sighting_of(parent_id: str) -> dict:
    return {'kind': SIGHTING_KIND, PARENT_ID_KEY: parent_id}


def _route(results: list[MemoryResult], *, content: str = _WRITE, t_high=_T_HIGH, t_low=_T_LOW):
    return route_band(results, content=content, t_high=t_high, t_low=t_low)


def _band(results: list[MemoryResult], *, t_high=_T_HIGH, t_low=_T_LOW) -> BandDecision:
    return decide_band(results, t_high=t_high, t_low=t_low)


def _svc(results: list[MemoryResult], **write_triage_leaves) -> types.SimpleNamespace:
    service = types.SimpleNamespace(
        config=types.SimpleNamespace(write_triage=types.SimpleNamespace(
            enabled=True, t_high=_T_HIGH, t_low=_T_LOW, candidate_k=20, **write_triage_leaves,
        )),
    )
    service.search = AsyncMock(return_value=SearchResults(results))
    service.cosine_to_memories = AsyncMock(return_value={})
    return service


def _counter() -> TriageFailOpenCounter:
    return TriageFailOpenCounter(time_provider=lambda: 1000.0)


async def _triage(service, judge, counter: TriageFailOpenCounter, *, content: str = _WRITE) -> BandDecision:
    return await triage_write(
        service, content=content, project_id=_PROJECT, counter=counter, judge=judge,
    )


class TestRouteBandKeepsRestatedOnlyForANearVerbatimWrite:

    def test_a_near_verbatim_write_to_the_band_winner_is_restated_as_decided(self) -> None:
        slate = [_result('m1', 0.97, content=_WRITE_REFLOWED), _result('m2', 0.80)]

        routed = _route(slate)

        assert routed.outcome == OUTCOME_RESTATED
        assert routed == _band(slate)

    def test_a_near_verbatim_write_to_the_hoisted_parent_is_restated_against_the_parent(
        self,
    ) -> None:
        slate = [
            _result('c1', 0.95, content=_PARAPHRASE, extra_metadata=_sighting_of('p1')),
            _result('p1', 0.88, content=_WRITE_REFLOWED),
        ]

        routed = _route(slate)

        assert routed.outcome == OUTCOME_RESTATED
        assert routed.canonical_id == 'p1'
        assert routed == _band(slate)

    def test_a_high_band_write_that_is_not_near_verbatim_is_routed_to_the_judge(self) -> None:
        slate = [_result('m1', 0.97, content=_PARAPHRASE), _result('m2', 0.80)]

        routed = _route(slate)

        assert routed == replace(_band(slate), outcome=OUTCOME_JUDGE)
        assert routed.winner_id == 'm1'
        assert routed.canonical_id == 'm1'
        assert routed.similarity == pytest.approx(0.97)

    def test_a_write_near_verbatim_only_to_a_non_winner_is_routed_to_the_judge(self) -> None:
        slate = [_result('m1', 0.97, content=_PARAPHRASE), _result('m2', 0.93, content=_WRITE)]

        assert _route(slate).outcome == OUTCOME_JUDGE

    def test_a_near_verbatim_write_in_the_middle_band_stays_judged(self) -> None:
        slate = [_result('m1', 0.85, content=_WRITE)]

        assert _route(slate) == _band(slate)
        assert _route(slate).outcome == OUTCOME_JUDGE

    @pytest.mark.parametrize(
        ('slate', 't_high'),
        [
            pytest.param([_result('m1', 0.60, content=_WRITE)], _T_HIGH, id='below-t_low'),
            pytest.param([_result('m1', 0.97, content=_PARAPHRASE)], None, id='t_high-none'),
            pytest.param([], _T_HIGH, id='empty-slate'),
        ],
    )
    def test_outside_the_restate_band_the_routing_is_the_band(self, slate, t_high) -> None:
        assert _route(slate, t_high=t_high) == _band(slate, t_high=t_high)


class TestAHighBandWriteIsAJudgedWrite:

    @pytest.mark.asyncio
    async def test_a_high_band_paraphrase_is_judged_with_the_band_winner_on_the_slate(
        self,
    ) -> None:
        slate = [
            _result('m1', 0.97, content=_PARAPHRASE),
            _result('m2', 0.92),
            _result('m3', 0.80),
        ]
        judge = AsyncMock(return_value=TriageJudgeVerdict(OUTCOME_AMENDED, 'm2'))
        counter = _counter()

        decision = await _triage(_svc(slate), judge, counter)

        judge.assert_awaited_once()
        assert judge.await_args is not None
        asked = judge.await_args.kwargs
        assert asked['decision'].outcome == OUTCOME_JUDGE
        assert asked['decision'].canonical_id == 'm1'
        shown = select_judge_candidates(asked['candidates'], 5, canonical_id=asked['decision'].canonical_id)
        assert 'm1' in [candidate.id for candidate in shown]
        assert decision.outcome == OUTCOME_AMENDED
        assert decision.canonical_id == 'm2'
        assert counter.live_count() == 0

    @pytest.mark.asyncio
    async def test_a_high_band_near_verbatim_write_is_restated_without_the_judge(self) -> None:
        slate = [_result('m1', 0.97, content=_WRITE_REFLOWED)]
        judge = AsyncMock(return_value=TriageJudgeVerdict(OUTCOME_AMENDED, 'm1'))

        decision = await _triage(_svc(slate), judge, _counter())

        judge.assert_not_awaited()
        assert decision.outcome == OUTCOME_RESTATED
        assert decision.canonical_id == 'm1'

    @pytest.mark.asyncio
    async def test_a_judge_that_raises_on_a_high_band_write_is_one_counted_fail_open(self) -> None:
        slate = [_result('m1', 0.97, content=_PARAPHRASE)]
        judge = AsyncMock(side_effect=RuntimeError('provider down'))
        counter = _counter()

        decision = await _triage(_svc(slate), judge, counter)

        assert decision.outcome == OUTCOME_STORED
        assert decision.canonical_id is None
        assert counter.live_count() == 1

    @pytest.mark.asyncio
    async def test_the_hoist_gap_refuses_a_judged_high_band_hoist(self) -> None:
        slate = [
            _result('c1', 0.95, content=_PARAPHRASE, extra_metadata=_sighting_of('p1')),
            _result('p1', 0.85),
        ]
        judge = AsyncMock(return_value=TriageJudgeVerdict(OUTCOME_AMENDED, 'c1'))

        decision = await _triage(_svc(slate, hoist_max_cosine_gap=0.05), judge, _counter())

        judge.assert_awaited_once()
        assert decision.outcome == OUTCOME_STORED
        assert decision.attach_refusals == {AttachRefusal.HOIST_COSINE_GAP}


class TestAJudgedRestatementThatAddsSomethingIsFiledAmended:

    @staticmethod
    def _slate(winner_cosine: float, *, m2_content: str = _PARAPHRASE) -> list[MemoryResult]:
        """The band winner m1, and m2: a sighting child of p2, which is not on the slate."""
        return [
            _result('m1', winner_cosine, content='an unrelated record about the hook'),
            _result('m2', 0.80, content=m2_content, extra_metadata=_sighting_of('p2')),
        ]

    @pytest.mark.asyncio
    @pytest.mark.parametrize('winner_cosine', [0.97, 0.85], ids=['high-band', 'middle-band'])
    async def test_a_restates_verdict_on_a_write_that_is_not_near_verbatim_is_amended(
        self, winner_cosine: float,
    ) -> None:
        slate = self._slate(winner_cosine)
        judge = AsyncMock(return_value=TriageJudgeVerdict(OUTCOME_RESTATED, 'm2'))
        counter = _counter()

        decision = await _triage(_svc(slate), judge, counter)

        assert decision.outcome == OUTCOME_AMENDED
        assert decision.canonical_id == canonical_id_of(slate[1]) == 'p2'
        assert decision.judged_candidate_id == 'm2'
        assert counter.live_count() == 0

    @pytest.mark.asyncio
    async def test_a_restates_verdict_on_a_near_verbatim_write_stays_restated(self) -> None:
        slate = self._slate(0.85, m2_content=_WRITE_REFLOWED)
        judge = AsyncMock(return_value=TriageJudgeVerdict(OUTCOME_RESTATED, 'm2'))

        decision = await _triage(_svc(slate), judge, _counter())

        assert decision.outcome == OUTCOME_RESTATED
        assert decision.canonical_id == 'p2'

    @pytest.mark.asyncio
    async def test_a_bare_word_restated_is_checked_against_the_band_winner(self) -> None:
        slate = [_result('m1', 0.97, content=_PARAPHRASE), _result('m2', 0.80, content=_WRITE)]
        judge = AsyncMock(return_value=OUTCOME_RESTATED)

        decision = await _triage(_svc(slate), judge, _counter())

        assert decision.outcome == OUTCOME_AMENDED
        assert decision.canonical_id == 'm1'

    @pytest.mark.asyncio
    async def test_a_refiled_restatement_is_one_info_line(self, caplog) -> None:
        judge = AsyncMock(return_value=TriageJudgeVerdict(OUTCOME_RESTATED, 'm2'))
        caplog.set_level(logging.INFO, logger=_TRIAGE_LOGGER)

        await _triage(_svc(self._slate(0.97)), judge, _counter())

        [line] = [
            record for record in caplog.records
            if record.name == _TRIAGE_LOGGER and record.getMessage().startswith(_REFILED_PREFIX)
        ]
        assert line.levelno == logging.INFO
        message = line.getMessage()
        for fragment in (_PROJECT, "'m2'", "'p2'", 'not near-verbatim'):
            assert fragment in message, (fragment, message)


def _model_answering(answer: dict) -> MagicMock:
    """A fake ``AsyncOpenAI`` whose Responses API answers *answer*: only the model is faked."""
    client = sdk_client_double()
    client.responses.create = AsyncMock(return_value=types.SimpleNamespace(
        output_text=json.dumps(answer),
        status='completed',
        incomplete_details=None,
        usage=types.SimpleNamespace(
            input_tokens=900, output_tokens=60,
            output_tokens_details=types.SimpleNamespace(reasoning_tokens=40),
        ),
    ))
    return client


async def _add_memory(content: str, model: MagicMock, monkeypatch) -> tuple[dict, dict, TriageFailOpenCounter]:
    """Write *content* through the tool against one 0.97 candidate m1; return the ack, the child's metadata, the counter."""
    service = AsyncMock()
    configure_triage_config(service, judge_enabled=True)
    written = MagicMock()
    written.model_dump.return_value = {
        'id': 'new-id', 'category': 'procedural_knowledge', 'stored_in': ['mem0'],
    }
    service.add_memory.return_value = written
    service.search.return_value = SearchResults([search_candidate('m1', 0.97, content=_WRITE)])
    counter = _counter()
    monkeypatch.setattr(tools, 'TriageFailOpenCounter', lambda: counter)
    with patch('openai.AsyncOpenAI', return_value=model):
        ack = await create_mcp_server(service)._tool_manager.call_tool('add_memory', {
            'content': content,
            'category': 'procedural_knowledge',
            'agent_id': 'claude-interactive',
            'project_id': _PROJECT,
        })
    persisted = service.add_memory.await_args.kwargs.get('metadata') or {}
    return ack, persisted, counter


class TestAddMemoryAcksRestatedOnlyForAWriteThatAddsNothing:

    @pytest.mark.asyncio
    async def test_a_high_band_paraphrase_the_model_calls_a_restatement_is_an_amendment(
        self, monkeypatch,
    ) -> None:
        model = _model_answering({VERDICT_KEY: 'restates', CANDIDATE_ID_KEY: 'm1'})

        ack, persisted, counter = await _add_memory(_PARAPHRASE, model, monkeypatch)

        model.responses.create.assert_awaited_once()
        assert ack[ROUTED_KEY] == OUTCOME_AMENDED, f'{ack!r}'
        assert ack[CANONICAL_ID_KEY] == 'm1', f'{ack!r}'
        assert persisted['kind'] == AMENDMENT_KIND, f'{persisted!r}'
        assert counter.live_count() == 0

    @pytest.mark.asyncio
    async def test_a_verbatim_repeat_is_a_sighting_without_asking_the_model(self, monkeypatch) -> None:
        model = _model_answering({VERDICT_KEY: 'amends', CANDIDATE_ID_KEY: 'm1'})

        ack, persisted, _ = await _add_memory(_WRITE, model, monkeypatch)

        model.responses.create.assert_not_awaited()
        assert ack[ROUTED_KEY] == OUTCOME_RESTATED, f'{ack!r}'
        assert ack[CANONICAL_ID_KEY] == 'm1', f'{ack!r}'
        assert persisted['kind'] == SIGHTING_KIND, f'{persisted!r}'
