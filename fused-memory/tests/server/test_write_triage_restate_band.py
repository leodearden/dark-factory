"""The restate band (cosine >= t_high) files `restated` only when the write adds nothing.

A high cosine says a write is about the same thing as a stored record, not that
it says nothing new. So a write in the restate band is judged like a
middle-band write unless it repeats the band winner, or the winner's canonical,
near-verbatim. ``decide_band`` stays the cosine BAND label; ``route_band`` is
the routing ``triage_write`` acts on.
"""

from __future__ import annotations

import types
from dataclasses import replace
from unittest.mock import AsyncMock

import pytest

from fused_memory.models.enums import MemoryCategory, SourceStore
from fused_memory.models.memory import MemoryResult
from fused_memory.server.grouped_read import PARENT_ID_KEY, SIGHTING_KIND
from fused_memory.server.write_triage import (
    OUTCOME_AMENDED,
    OUTCOME_JUDGE,
    OUTCOME_RESTATED,
    OUTCOME_STORED,
    AttachRefusal,
    BandDecision,
    TriageFailOpenCounter,
    TriageJudgeVerdict,
    decide_band,
    route_band,
    triage_write,
)
from fused_memory.server.write_triage_judge import select_judge_candidates
from fused_memory.services.memory_service import RRF_K, SearchResults

_T_HIGH = 0.90
_T_LOW = 0.70
_PROJECT = 'proj-7155'

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
