"""The record a judged verdict is filed against is the record the judge named.

A middle-band write is shown to the judge alongside a slate of candidates, and
the judge answers with a verdict AND the id of the candidate that verdict is
about. That named candidate — not the band's max-cosine winner — is what the
write attaches to. These tests pin that at the ``triage_write`` seam, and at
the ``add_memory`` tool with only the model's answer faked.

Helpers are local rather than imported from the sibling triage suites, which
are each written to stand alone.
"""

from __future__ import annotations

import json
import logging
import subprocess
import sys
import types
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from fused_memory.models.enums import MemoryCategory, SourceStore
from fused_memory.models.memory import MemoryResult
from fused_memory.server import tools
from fused_memory.server.grouped_read import (
    AMENDMENT_KIND,
    PARENT_ID_KEY,
    SIGHTING_KIND,
    is_contested_child,
)
from fused_memory.server.tools import create_mcp_server
from fused_memory.server.write_triage import (
    CANONICAL_ID_KEY,
    OUTCOME_AMENDED,
    OUTCOME_CONTESTED,
    OUTCOME_RESTATED,
    OUTCOME_STORED,
    ROUTED_KEY,
    TriageFailOpenCounter,
    TriageJudgeVerdict,
    triage_write,
)
from fused_memory.server.write_triage_judge import CANDIDATE_ID_KEY, VERDICT_KEY
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
            _middle_band_slate(), _judge_answering(TriageJudgeVerdict(outcome, 'm3')), counter,
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
            _judge_answering(TriageJudgeVerdict(OUTCOME_AMENDED, 'm3')),
            _counter(),
        )

        assert decision.similarity == pytest.approx(0.85)

    @pytest.mark.asyncio
    async def test_a_judged_child_is_hoisted_to_its_parent(self) -> None:
        decision = await _triage(
            _middle_band_slate(kind=AMENDMENT_KIND, **{PARENT_ID_KEY: 'parent-P'}),
            _judge_answering(TriageJudgeVerdict(OUTCOME_AMENDED, 'm3')),
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
        judge = _judge_answering(TriageJudgeVerdict(OUTCOME_AMENDED, 'm3'))
        slate = [_result('m1', 0.95), *_middle_band_slate()[1:]]

        decision = await _triage(slate, judge, _counter())

        assert decision.outcome == OUTCOME_RESTATED
        assert decision.canonical_id == _BAND_WINNER
        assert decision.judged_candidate_id is None
        judge.assert_not_awaited()


def _logged_exception(record: logging.LogRecord) -> str:
    assert record.exc_info is not None, 'a fail-open is logged with its exception'
    return str(record.exc_info[1])


class TestABreachedVerdictFailsOpenOnce:
    """Every judged-band contract breach stores the write and counts ONE fail-open."""

    @staticmethod
    async def _triage_breach(answer: object, caplog) -> logging.LogRecord:
        counter = _counter()
        with caplog.at_level(logging.DEBUG, logger=triage_write.__module__):
            decision = await _triage(
                _middle_band_slate(), _judge_answering(answer), counter,
            )

        assert decision.outcome == OUTCOME_STORED
        assert decision.canonical_id is None
        assert decision.judged_candidate_id is None
        assert counter.live_count() == 1
        fail_opens = [
            record for record in caplog.records
            if 'fail-open at stage=judge' in record.getMessage()
        ]
        assert len(fail_opens) == 1, caplog.text
        return fail_opens[0]

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ('answer', 'offending'),
        [
            pytest.param(
                TriageJudgeVerdict(OUTCOME_AMENDED, 'not-retrieved'), 'not-retrieved',
                id='an id naming no retrieved record',
            ),
            pytest.param(
                TriageJudgeVerdict(OUTCOME_STORED, 'm3'), 'm3',
                id='a stored verdict naming a candidate',
            ),
            pytest.param((OUTCOME_AMENDED, 7), 7, id='a non-str id'),
            pytest.param(TriageJudgeVerdict(OUTCOME_AMENDED, ''), '', id='an empty id'),
            pytest.param(
                TriageJudgeVerdict('superseded', 'm3'), 'superseded',
                id='an outcome outside the vocabulary',
            ),
        ],
    )
    async def test_a_value_breach_is_a_warning_naming_the_value(
        self, answer, offending, caplog,
    ) -> None:
        record = await self._triage_breach(answer, caplog)

        assert record.levelno == logging.WARNING, caplog.text
        assert repr(offending) in _logged_exception(record)

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        'answer',
        [
            pytest.param(('amended', 'm3', 'x'), id='a triple'),
            pytest.param({'outcome': 'amended', 'candidate_id': 'm3'}, id='a dict'),
            pytest.param(None, id='None'),
        ],
    )
    async def test_a_malformed_answer_shape_is_counted(self, answer, caplog) -> None:
        await self._triage_breach(answer, caplog)


def _tool_service(results: list[MemoryResult]) -> AsyncMock:
    """A memory service whose config is REAL namespaces and whose writes dump.

    Namespaces, not an unspecced mock: an auto-generated attribute reads as a
    truthy Mock, which the triage resolvers refuse. The judge is enabled on the
    openai arm; the provider itself is faked per test.
    """
    service = AsyncMock()
    service.config = types.SimpleNamespace(
        write_triage=types.SimpleNamespace(
            enabled=True, candidate_k=20, t_high=_T_HIGH, t_low=_T_LOW,
            judge_enabled=True, judge_provider='openai', judge_model='test-model',
        ),
        reconciliation=types.SimpleNamespace(
            procedural_knowledge_near_dup_guard_enabled=True,
            procedural_knowledge_near_dup_threshold=0.90,
            procedural_knowledge_topic_guard_clusters=[],
        ),
    )
    written = MagicMock()
    written.model_dump.return_value = {
        'id': 'new-id', 'category': 'procedural_knowledge', 'stored_in': ['mem0'],
    }
    service.add_memory.return_value = written
    service.search.return_value = SearchResults(results)
    return service


def _provider_answering(answer: dict) -> MagicMock:
    """A fake ``AsyncOpenAI``, its own async context manager as the SDK's is."""
    message = types.SimpleNamespace(content=json.dumps(answer))
    client = MagicMock()
    client.__aenter__ = AsyncMock(return_value=client)
    client.__aexit__ = AsyncMock(return_value=False)
    client.chat.completions.create = AsyncMock(return_value=types.SimpleNamespace(
        choices=[types.SimpleNamespace(message=message)],
    ))
    return client


async def _add_memory(
    service: AsyncMock, answer: dict, monkeypatch,
) -> tuple[dict, dict, TriageFailOpenCounter]:
    """Write through the tool; return the ack, the persisted metadata, the counter."""
    counter = _counter()
    monkeypatch.setattr(tools, 'TriageFailOpenCounter', lambda: counter)
    with patch('openai.AsyncOpenAI', return_value=_provider_answering(answer)):
        ack = await create_mcp_server(service)._tool_manager.call_tool('add_memory', {
            'content': 'the new entry',
            'category': 'procedural_knowledge',
            'agent_id': 'claude-interactive',
            'project_id': 'dark_factory',
        })
    persisted = service.add_memory.await_args.kwargs.get('metadata') or {}
    return ack, persisted, counter


class TestAddMemoryFilesTheVerdictAgainstTheNamedCandidate:
    """The tool-level signal: the ack and the persisted child name the judged record."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ('word', 'outcome', 'kind'),
        [
            ('amends', OUTCOME_AMENDED, AMENDMENT_KIND),
            ('restates', OUTCOME_RESTATED, SIGHTING_KIND),
            ('contests', OUTCOME_CONTESTED, AMENDMENT_KIND),
        ],
    )
    async def test_the_ack_and_the_child_name_the_judged_candidate(
        self, word, outcome, kind, monkeypatch,
    ) -> None:
        ack, persisted, counter = await _add_memory(
            _tool_service(_middle_band_slate()),
            {VERDICT_KEY: word, CANDIDATE_ID_KEY: 'm3'},
            monkeypatch,
        )

        assert ack[ROUTED_KEY] == outcome, f'{ack!r}'
        assert ack[CANONICAL_ID_KEY] == 'm3', f'{ack!r}'
        assert persisted[PARENT_ID_KEY] == 'm3', (
            f'filed under the band winner {_BAND_WINNER!r}, not the judged m3: '
            f'{persisted!r}'
        )
        assert persisted['kind'] == kind, f'{persisted!r}'
        assert is_contested_child(persisted) is (outcome == OUTCOME_CONTESTED)
        assert counter.live_count() == 0

    @pytest.mark.asyncio
    async def test_a_candidate_not_on_the_slate_stores_the_write_standalone(
        self, monkeypatch,
    ) -> None:
        ack, persisted, counter = await _add_memory(
            _tool_service(_middle_band_slate()),
            {VERDICT_KEY: 'amends', CANDIDATE_ID_KEY: 'not-on-the-slate'},
            monkeypatch,
        )

        assert ack[ROUTED_KEY] == OUTCOME_STORED, f'{ack!r}'
        assert CANONICAL_ID_KEY not in ack, f'{ack!r}'
        assert PARENT_ID_KEY not in persisted, f'{persisted!r}'
        assert counter.live_count() == 1


#: The repo root, reached from `<repo>/fused-memory/tests/server/`.
_REPO_ROOT = Path(__file__).resolve().parents[3]

#: Gate item 5's own consumption probe (task 4949), run as the oracle.
_CONSUMPTION_PROBE = _REPO_ROOT / 'scripts' / 'check_write_triage_attach_consumption.py'


class TestGateItemFiveMeasuresTheConsumption:
    """The flip gate's item-5 probe, run against this worktree's source.

    It must reach its MEASURED branch — the attach tracking a judge-side swap
    of the named candidate — not the branch that holds only by construction
    while the prompt still marks the band's winner.
    """

    def test_the_probe_measures_the_judged_candidate_consumed(self) -> None:
        if not _CONSUMPTION_PROBE.exists():
            pytest.skip(f'attach-consumption probe not present at {_CONSUMPTION_PROBE}')
        completed = subprocess.run(
            [
                sys.executable, str(_CONSUMPTION_PROBE),
                '--src-root', str(_REPO_ROOT / 'fused-memory' / 'src'),
                '--extra-path', str(_REPO_ROOT / 'shared' / 'src'),
            ],
            capture_output=True,
            text=True,
            timeout=120,
        )
        assert completed.returncode == 0, completed.stdout + completed.stderr
        assert (
            'PASS  the judge-bound candidate is CONSUMED by the attach' in completed.stdout
        ), completed.stdout
        assert 'ITEM5-BRANCH  judge-side designation swap' in completed.stdout, (
            completed.stdout
        )
