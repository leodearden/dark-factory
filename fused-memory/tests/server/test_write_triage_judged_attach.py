"""The record a judged verdict is filed against is the record the judge named.

A middle-band write is shown to the judge alongside a slate of candidates, and
the judge answers with a verdict AND the id of the candidate that verdict is
about. That named candidate — not the band's max-cosine winner — is what the
write attaches to. These tests pin that at the ``triage_write`` seam.

Helpers are local rather than imported from the sibling triage suites, which
are each written to stand alone.
"""

from __future__ import annotations

import types
from unittest.mock import AsyncMock

import pytest

from fused_memory.models.enums import MemoryCategory, SourceStore
from fused_memory.models.memory import MemoryResult
from fused_memory.server.grouped_read import AMENDMENT_KIND, PARENT_ID_KEY
from fused_memory.server.write_triage import (
    OUTCOME_AMENDED,
    OUTCOME_CONTESTED,
    OUTCOME_RESTATED,
    JudgeVerdict,
    TriageFailOpenCounter,
    triage_write,
)
from fused_memory.services.memory_service import RRF_K, SearchResults

_T_HIGH = 0.90
_T_LOW = 0.70
_BAND_WINNER = 'm1'


def _result(
    id_: str,
    cosine: float,
    *,
    extra_metadata: dict | None = None,
) -> MemoryResult:
    """A post-RRF record: the cosine lives in ``metadata['store_score']``."""
    return MemoryResult(
        id=id_,
        content=f'content of {id_}',
        category=MemoryCategory.procedural_knowledge,
        source_store=SourceStore.mem0,
        relevance_score=1.0 / (RRF_K + 1),
        metadata={'store_score': cosine, **(extra_metadata or {})},
    )


def _middle_band_slate(**m3_metadata) -> list[MemoryResult]:
    """Four candidates, all inside [t_low, t_high); the band's winner is m1."""
    return [
        _result('m1', 0.85),
        _result('m2', 0.83),
        _result('m3', 0.80, extra_metadata=m3_metadata or None),
        _result('m4', 0.78),
    ]


def _svc(results: list[MemoryResult]) -> types.SimpleNamespace:
    service = types.SimpleNamespace(
        config=types.SimpleNamespace(write_triage=types.SimpleNamespace(
            enabled=True, t_high=_T_HIGH, t_low=_T_LOW, candidate_k=20,
        )),
    )
    service.search = AsyncMock(return_value=SearchResults(results))
    return service


def _counter() -> TriageFailOpenCounter:
    return TriageFailOpenCounter(time_provider=lambda: 1000.0)


def _judge_answering(answer: object) -> AsyncMock:
    return AsyncMock(return_value=answer)


async def _triage(results: list[MemoryResult], judge, counter: TriageFailOpenCounter):
    return await triage_write(
        _svc(results), content='the new entry', project_id='p',
        counter=counter, judge=judge,
    )


class TestTheJudgedCandidateIsTheAttachTarget:

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        'outcome', [OUTCOME_RESTATED, OUTCOME_AMENDED, OUTCOME_CONTESTED],
    )
    async def test_the_write_attaches_to_the_candidate_the_judge_named(
        self, outcome,
    ) -> None:
        counter = _counter()

        decision = await _triage(
            _middle_band_slate(), _judge_answering(JudgeVerdict(outcome, 'm3')), counter,
        )

        assert decision.outcome == outcome
        assert decision.canonical_id == 'm3', (
            f'filed against the band winner {_BAND_WINNER!r}, not the named m3'
        )
        assert decision.judged_candidate_id == 'm3'
        assert counter.live_count() == 0

    @pytest.mark.asyncio
    async def test_the_similarity_stays_the_cosine_that_routed_the_write(self) -> None:
        decision = await _triage(
            _middle_band_slate(),
            _judge_answering(JudgeVerdict(OUTCOME_AMENDED, 'm3')),
            _counter(),
        )

        assert decision.similarity == pytest.approx(0.85)

    @pytest.mark.asyncio
    async def test_a_judged_child_is_hoisted_to_its_parent(self) -> None:
        decision = await _triage(
            _middle_band_slate(kind=AMENDMENT_KIND, **{PARENT_ID_KEY: 'parent-P'}),
            _judge_answering(JudgeVerdict(OUTCOME_AMENDED, 'm3')),
            _counter(),
        )

        assert decision.canonical_id == 'parent-P', 'a child is never an attach target'
        assert decision.judged_candidate_id == 'm3'

    @pytest.mark.asyncio
    async def test_a_plain_pair_is_read_as_a_verdict(self) -> None:
        counter = _counter()

        decision = await _triage(
            _middle_band_slate(), _judge_answering(('amended', 'm3')), counter,
        )

        assert decision.outcome == OUTCOME_AMENDED
        assert decision.canonical_id == 'm3'
        assert decision.judged_candidate_id == 'm3'
        assert counter.live_count() == 0

    @pytest.mark.asyncio
    async def test_a_bare_word_names_no_candidate_and_attaches_to_the_bands_winner(
        self,
    ) -> None:
        counter = _counter()

        decision = await _triage(
            _middle_band_slate(), _judge_answering('amended'), counter,
        )

        assert decision.outcome == OUTCOME_AMENDED
        assert decision.canonical_id == _BAND_WINNER
        assert decision.judged_candidate_id is None
        assert counter.live_count() == 0

    @pytest.mark.asyncio
    async def test_the_deterministic_band_never_asks_the_judge(self) -> None:
        judge = _judge_answering(JudgeVerdict(OUTCOME_AMENDED, 'm3'))
        slate = [_result('m1', 0.95), *_middle_band_slate()[1:]]

        decision = await _triage(slate, judge, _counter())

        assert decision.outcome == OUTCOME_RESTATED
        assert decision.canonical_id == _BAND_WINNER
        assert decision.judged_candidate_id is None
        judge.assert_not_awaited()
