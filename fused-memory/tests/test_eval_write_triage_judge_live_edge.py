"""Tests for eval_write_triage_judge.py's live edge and CLI.

The script's pure core is `test_eval_write_triage_judge.py`'s subject. This
module is the other partition: `build_judge_fn` and `build_retrieved_judge_fn`
driving the SHIPPED `judge_write` -> `_call_llm` -> openai SDK chain, and
`main()` driven through `sys.argv`. The SDK is a fake async-context client
patched in at `openai.AsyncOpenAI`, as `tests/server/test_write_triage_judge.py`
does, and every CLI run is a `--dry-run` on a synthetic fixture, so nothing
here needs a key, a network or Qdrant.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import re
import sys
import types
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from _fm_helpers import load_script_module
from _write_triage_store_fake import FakeMemoryService

from fused_memory.models.enums import SourceStore
from fused_memory.models.memory import MemoryResult
from fused_memory.server.write_triage import (
    OUTCOME_AMENDED,
    OUTCOME_JUDGE,
    OUTCOME_RESTATED,
    OUTCOME_STORED,
)
from fused_memory.server.write_triage_judge import (
    CANDIDATE_ID_KEY,
    JUDGE_VERDICTS,
    VERDICT_KEY,
)
from fused_memory.services.memory_service import SearchResults

SCRIPT_PATH = Path(__file__).parent.parent / 'scripts' / 'eval_write_triage_judge.py'

#: The committed curator-labelled corpus the seeded eval runs over.
SEEDED_FIXTURE = Path(__file__).parent / 'fixtures' / 'write_triage_calibration.jsonl'

#: What the fake provider bills for every response, in the Responses API's shape.
_BILLED = types.SimpleNamespace(
    input_tokens=321, output_tokens=7,
    output_tokens_details=types.SimpleNamespace(reasoning_tokens=0),
)

#: The same bill as the eval artifact's per-case ``usage`` row records it:
#: only the keys a consumer reads, so the reasoning split is not carried.
_USAGE_ROW = {'prompt_tokens': 321, 'completion_tokens': 7, 'total_tokens': 328}

T_HIGH = 0.9
T_LOW = 0.5


def _mod() -> types.ModuleType:
    return load_script_module(SCRIPT_PATH, 'eval_write_triage_judge')


def _rec(memory_id: str, cluster_id: str, label: str) -> dict:
    return {
        'memory_id': memory_id, 'content': f'content of {memory_id}',
        'category': 'procedural_knowledge', 'cluster_id': cluster_id, 'label': label,
    }


#: Three clusters, grouped in file order as the committed fixture is. In this
#: shape a head-N slice is one cluster, which is why `--limit` round-robins.
_CORPUS = (
    _rec('a-canon', 'a-canon', 'canonical'),
    _rec('a-dup-1', 'a-canon', 'duplicate'),
    _rec('a-dup-2', 'a-canon', 'duplicate'),
    _rec('b-canon', 'b-canon', 'canonical'),
    _rec('b-dup-1', 'b-canon', 'duplicate'),
    _rec('b-distinct', 'b-canon', 'distinct'),
    _rec('c-canon', 'c-canon', 'canonical'),
    _rec('c-dup-1', 'c-canon', 'duplicate'),
    _rec('c-pseudo', 'c-canon', 'pseudo_contradiction'),
)


#: A candidate line of the rendered user prompt, capturing its id.
_ID_LINE = re.compile(r'^- id: (\S+)$', re.MULTILINE)


def _openai(word: str, *, named: int = 0, billed: object = _BILLED) -> MagicMock:
    """A fake `AsyncOpenAI` that is its own async context manager, as the SDK is.

    Every Responses call answers *word* about the candidate at position *named*
    of its own prompt (an attach verdict must name one; `distinct` names none)
    and bills *billed*. The response is plain namespaces because a MagicMock
    `usage` would hand `int()` a 1. The chat endpoint is a bare `AsyncMock`, so
    a call on the wrong arm is visible.
    """
    async def _respond(**kwargs) -> types.SimpleNamespace:
        answer = {VERDICT_KEY: word}
        if JUDGE_VERDICTS[word] != OUTCOME_STORED:
            answer[CANDIDATE_ID_KEY] = _ID_LINE.findall(kwargs['input'])[named]
        return types.SimpleNamespace(
            output_text=json.dumps(answer), status='completed',
            incomplete_details=None, usage=billed,
        )

    client = MagicMock()
    client.__aenter__ = AsyncMock(return_value=client)
    client.__aexit__ = AsyncMock(return_value=False)
    client.responses.create = AsyncMock(side_effect=_respond)
    client.chat.completions.create = AsyncMock()
    return client


def _judge_config() -> types.SimpleNamespace:
    """Everything `judge_write` reads: the openai arm, enabled, five wide."""
    return types.SimpleNamespace(
        write_triage=types.SimpleNamespace(
            judge_enabled=True, judge_provider='openai', judge_model='test-model',
            judge_candidate_count=5,
        ),
        llm=types.SimpleNamespace(provider='openai', model='test-model', providers=None),
    )


def _prompt_ids(create: AsyncMock) -> list[list[str]]:
    """The candidate ids each awaited Responses call's user prompt named, in order."""
    return [_ID_LINE.findall(call.kwargs['input']) for call in create.await_args_list]


def _resolved(plan, index: int) -> list:
    return list(plan.candidate_records[index])


class TestTheSeededLiveEdge:
    """`build_judge_fn`: one shipped-judge call per case, on the slate built for it."""

    @staticmethod
    def _plan():
        return _mod().seeded_plan(list(_CORPUS), distractors=2)

    def test_every_case_is_one_completion_naming_its_slate_in_order(
        self, tmp_path: Path,
    ) -> None:
        plan = self._plan()
        client = _openai('amends')
        with patch('openai.AsyncOpenAI', return_value=client):
            _mod().run_judge_eval(
                plan=plan, judge_fn=_mod().build_judge_fn(_judge_config()),
                report_path=tmp_path / 'r.json', provenance={},
            )
        create = client.responses.create
        assert create.await_count == len(plan.cases)
        assert _prompt_ids(create) == [case['candidates'] for case in plan.cases]

    def test_the_answer_is_the_parsed_verdict(self) -> None:
        plan = self._plan()
        case = plan.cases[0]
        with patch('openai.AsyncOpenAI', return_value=_openai('amends')):
            answer = _mod().build_judge_fn(_judge_config())(case, _resolved(plan, 0))
        assert (answer.outcome, answer.verdict) == (JUDGE_VERDICTS['amends'],) * 2
        assert answer.candidate_id == case['candidates'][0]

    def test_a_recorded_run_reports_what_the_provider_billed(self) -> None:
        plan = self._plan()
        case = plan.cases[0]
        with patch('openai.AsyncOpenAI', return_value=_openai('restates')):
            answer = _mod().build_judge_fn(_judge_config())(case, _resolved(plan, 0))
        assert answer.usage == _USAGE_ROW

    def test_a_run_with_no_reported_usage_is_unpriced(self) -> None:
        """The artifact says 'unpriced', never zeros."""
        plan = self._plan()
        case = plan.cases[0]
        with patch('openai.AsyncOpenAI', return_value=_openai('restates', billed=None)):
            answer = _mod().build_judge_fn(_judge_config())(case, _resolved(plan, 0))
        assert answer.usage is None


def _deterministic_answer(system: str, user: str) -> str:
    """One verdict per (system, user) text pair, so any wire difference changes a verdict."""
    digest = hashlib.sha256(f'{system}\x00{user}'.encode()).digest()
    word = sorted(JUDGE_VERDICTS)[digest[0] % len(JUDGE_VERDICTS)]
    answer = {VERDICT_KEY: word}
    if JUDGE_VERDICTS[word] != OUTCOME_STORED:
        ids = _ID_LINE.findall(user)
        answer[CANDIDATE_ID_KEY] = ids[digest[1] % len(ids)]
    return json.dumps(answer)


def _answering_both_apis() -> MagicMock:
    """A fake `AsyncOpenAI` serving both OpenAI APIs from :func:`_deterministic_answer`."""
    async def _respond(**kwargs) -> types.SimpleNamespace:
        return types.SimpleNamespace(
            output_text=_deterministic_answer(kwargs['instructions'], kwargs['input']),
            status='completed', incomplete_details=None, usage=None,
        )

    async def _complete(**kwargs) -> types.SimpleNamespace:
        system, user = (message['content'] for message in kwargs['messages'])
        return types.SimpleNamespace(
            choices=[types.SimpleNamespace(message=types.SimpleNamespace(
                content=_deterministic_answer(system, user),
            ))],
            usage=None,
        )

    client = MagicMock()
    client.__aenter__ = AsyncMock(return_value=client)
    client.__aexit__ = AsyncMock(return_value=False)
    client.responses.create = AsyncMock(side_effect=_respond)
    client.chat.completions.create = AsyncMock(side_effect=_complete)
    return client


class TestTheDefaultModelsSeededVerdictsAreUnchanged:
    """Over the committed seeded fixture, the Responses arm reproduces main's chat verdicts.

    A live run is not byte-reproducible even at temperature 0, so the
    regression is pinned offline: both arms put the same prompt bytes and the
    same temperature on the wire, and one answer function of those bytes yields
    the same verdict per case. The chat arm is main's request, kept for
    ``openai_generic`` endpoints.
    """

    def test_both_arms_send_the_same_bytes_and_read_the_same_verdicts(self) -> None:
        mod = _mod()
        plan = mod.seeded_plan(mod.load_fixture(SEEDED_FIXTURE), distractors=4)
        chat_config = _judge_config()
        chat_config.llm.client_class = 'openai_generic'
        verdicts: dict[str, list] = {}
        clients: dict[str, MagicMock] = {}
        for arm, config in (('responses', _judge_config()), ('chat', chat_config)):
            client = _answering_both_apis()
            judge_fn = mod.build_judge_fn(config)
            with patch('openai.AsyncOpenAI', return_value=client):
                answers = [
                    judge_fn(case, _resolved(plan, index))
                    for index, case in enumerate(plan.cases)
                ]
            verdicts[arm] = [(answer.outcome, answer.candidate_id) for answer in answers]
            clients[arm] = client

        sent = [
            (call.kwargs['instructions'], call.kwargs['input'], call.kwargs['temperature'])
            for call in clients['responses'].responses.create.await_args_list
        ]
        main = [
            (*(message['content'] for message in call.kwargs['messages']),
             call.kwargs['temperature'])
            for call in clients['chat'].chat.completions.create.await_args_list
        ]
        assert len(sent) == len(plan.cases)
        assert sent == main
        assert verdicts['responses'] == verdicts['chat']
        assert len({outcome for outcome, _ in verdicts['chat']}) > 1, (
            'precondition: the answer function varies, or the comparison is vacuous'
        )
        clients['responses'].chat.completions.create.assert_not_awaited()
        clients['chat'].responses.create.assert_not_awaited()


def _hit(memory_id: str, cosine: float) -> MemoryResult:
    """A store row as `MemoryService.search` returns it, scored for one query."""
    return MemoryResult(
        id=memory_id, content=f'content of {memory_id}', source_store=SourceStore.mem0,
        metadata={'store_score': cosine},
    )


class _Store:
    """The two reads the retrieval edge makes: one search per query, one liveness probe."""

    def __init__(self, rows_by_query: dict[str, list[MemoryResult]]) -> None:
        self.search = AsyncMock(
            side_effect=lambda **kwargs: SearchResults(rows_by_query[kwargs['query']]),
        )

    async def get_memory_by_id(self, project_id: str, memory_id: str) -> dict:
        return {'id': memory_id}


def _shipped_plan(records: list[dict], rows_by_query: dict[str, list[MemoryResult]]):
    """The shipped retrieval, bands and trim over *rows_by_query*, paired with the labels."""
    retrieval = _mod().load_retrieval()
    labelled = [record for record in records if record['label'] != 'canonical']
    retrievals = asyncio.run(retrieval.prefetch_retrievals(
        _Store(rows_by_query), labelled, project_id='reify', k=5,
    ))
    slates = retrieval.retrieved_slates(
        labelled, retrievals, t_high=T_HIGH, t_low=T_LOW, judge_candidate_count=5,
    )
    return _mod().plan_from_slates(records, slates, provenance={})


class TestTheRetrievedLiveEdge:
    """`build_retrieved_judge_fn`: only the middle band reaches the provider."""

    @staticmethod
    def _case(cosine: float) -> tuple[dict, list]:
        """A duplicate retrieving its canonical at *cosine*, routed by the shipped bands."""
        plan = _shipped_plan(
            [_rec('canon', 'canon', 'canonical'), _rec('dup', 'canon', 'duplicate')],
            {'content of dup': [_hit('canon', cosine)]},
        )
        [case] = plan.cases
        return case, _resolved(plan, 0)

    @pytest.mark.parametrize(('cosine', 'band'), [
        (0.95, OUTCOME_RESTATED),
        (0.3, OUTCOME_STORED),
    ])
    def test_a_band_that_decided_itself_makes_no_call(
        self, cosine: float, band: str,
    ) -> None:
        case, candidates = self._case(cosine)
        assert case['band'] == band, 'precondition: the shipped bands routed it'
        client = _openai('amends')
        with patch('openai.AsyncOpenAI', return_value=client):
            answer = _mod().build_retrieved_judge_fn(_judge_config())(case, candidates)
        assert (answer.outcome, answer.verdict) == (band, None)
        assert answer.candidate_id is None
        client.responses.create.assert_not_awaited()

    def test_a_middle_band_case_is_one_completion(self) -> None:
        case, candidates = self._case(0.7)
        assert case['band'] == OUTCOME_JUDGE, 'precondition: the shipped bands routed it'
        client = _openai('amends')
        with patch('openai.AsyncOpenAI', return_value=client):
            answer = _mod().build_retrieved_judge_fn(_judge_config())(case, candidates)
        assert client.responses.create.await_count == 1
        assert (answer.outcome, answer.verdict) == (JUDGE_VERDICTS['amends'],) * 2

    def test_the_answer_carries_the_candidate_the_verdict_named(self) -> None:
        plan = _shipped_plan(
            [_rec('canon', 'canon', 'canonical'), _rec('dup', 'canon', 'duplicate')],
            {'content of dup': [_hit('canon', 0.8), _hit('other', 0.7)]},
        )
        [case] = plan.cases
        assert case['band'] == OUTCOME_JUDGE, 'precondition: the shipped bands routed it'
        assert case['candidates'] == ['canon', 'other'], 'precondition: the retrieval order'
        with patch('openai.AsyncOpenAI', return_value=_openai('amends', named=1)):
            answer = _mod().build_retrieved_judge_fn(_judge_config())(case, _resolved(plan, 0))
        assert (answer.outcome, answer.candidate_id) == (OUTCOME_AMENDED, 'other')

    def test_a_distinct_verdict_names_no_candidate(self) -> None:
        case, candidates = self._case(0.7)
        assert case['band'] == OUTCOME_JUDGE, 'precondition: the shipped bands routed it'
        with patch('openai.AsyncOpenAI', return_value=_openai('distinct')):
            answer = _mod().build_retrieved_judge_fn(_judge_config())(case, candidates)
        assert (answer.outcome, answer.candidate_id) == (OUTCOME_STORED, None)


class TestEachPromptRendersItsOwnRetrieval:
    """The `MemoryResult`s handed to `judge_write` carry each case's own cosines.

    `judge_write` re-sorts its slate by `store_score`, so the order the provider
    sees is decided by whichever query's cosine a row carries. Two duplicates
    retrieving the same two records at opposite cosines make a borrowed row
    visible in the rendered prompt.
    """

    def test_each_prompt_names_its_own_slate_in_its_own_order(self, tmp_path: Path) -> None:
        plan = _shipped_plan(
            [
                _rec('canon', 'canon', 'canonical'),
                _rec('dup-a', 'canon', 'duplicate'),
                _rec('dup-b', 'canon', 'duplicate'),
            ],
            {
                'content of dup-a': [_hit('canon', 0.80), _hit('other', 0.70)],
                'content of dup-b': [_hit('other', 0.85), _hit('canon', 0.75)],
            },
        )
        assert [case['band'] for case in plan.cases] == [OUTCOME_JUDGE, OUTCOME_JUDGE], (
            'precondition: the shipped bands routed both to the judge'
        )
        assert [case['candidates'] for case in plan.cases] == [
            ['canon', 'other'], ['other', 'canon'],
        ], 'precondition: each retrieval ranked its own slate'
        client = _openai('amends')
        with patch('openai.AsyncOpenAI', return_value=client):
            _mod().run_judge_eval(
                plan=plan, judge_fn=_mod().build_retrieved_judge_fn(_judge_config()),
                report_path=tmp_path / 'r.json', provenance={},
            )
        assert _prompt_ids(client.responses.create) == [
            case['candidates'] for case in plan.cases
        ]


class TestTheLimitedCli:
    """`main()` end to end on a `--dry-run`: the `--limit` draw, and the empty-slate guard."""

    @staticmethod
    def _argv(tmp_path: Path, *extra: str) -> list[str]:
        fixture = tmp_path / 'corpus.jsonl'
        fixture.write_text(''.join(json.dumps(record) + '\n' for record in _CORPUS))
        return [
            'eval_write_triage_judge.py', '--dry-run', '--fixture', str(fixture),
            '--report-path', str(tmp_path / 'r.json'),
            '--cases-path', str(tmp_path / 'cases.jsonl'), *extra,
        ]

    def test_a_limit_draws_across_clusters_and_every_slate_resolves(
        self, tmp_path: Path, monkeypatch,
    ) -> None:
        monkeypatch.setattr(sys, 'argv', self._argv(tmp_path, '--limit', '3', '--distractors', '2'))
        assert _mod().main() == 0
        provenance = json.loads((tmp_path / 'r.json').read_text())['provenance']
        assert provenance['limit'] == 3
        assert provenance['candidate_count_min'] == 3, 'every slate is distractors + 1 wide'
        rows = [json.loads(line) for line in (tmp_path / 'cases.jsonl').read_text().splitlines()]
        assert len({row['cluster_id'] for row in rows}) >= 2

    def test_a_one_record_limit_is_refused_before_anything_is_written(
        self, tmp_path: Path, monkeypatch,
    ) -> None:
        """One cluster leaves the control an empty slate, which no judge ever sees."""
        monkeypatch.setattr(sys, 'argv', self._argv(tmp_path, '--limit', '1'))
        with pytest.raises(ValueError, match='a-dup-1'):
            _mod().main()
        assert [path.name for path in tmp_path.iterdir()] == ['corpus.jsonl']


class TestTheRetrievedCli:
    """`main()` on a retrieved `--dry-run` over a stub store: what the plan is built from."""

    @staticmethod
    def _main(tmp_path: Path, monkeypatch, *extra: str, rows=()) -> None:
        """Every search answers *rows*."""

        class _LiveStore(FakeMemoryService):
            def __init__(self, config) -> None:
                super().__init__(rows)

            async def initialize(self) -> None:
                return None

            async def close(self) -> None:
                return None

        monkeypatch.setattr('fused_memory.services.memory_service.MemoryService', _LiveStore)
        monkeypatch.setattr(
            sys, 'argv', TestTheLimitedCli._argv(tmp_path, '--slate-mode', 'retrieved', *extra),
        )
        assert _mod().main() == 0

    def test_a_limited_run_describes_its_targets_from_the_whole_fixture(
        self, tmp_path: Path, monkeypatch,
    ) -> None:
        """`--limit 1` measures a-dup-1 alone; its target b-dup-1 is still a fixture record."""
        self._main(tmp_path, monkeypatch, '--limit', '1', rows=[_hit('b-dup-1', 0.7)])
        [row] = [json.loads(line) for line in (tmp_path / 'cases.jsonl').read_text().splitlines()]
        assert row['attach_target_id'] == 'b-dup-1', 'precondition: the shipped bands attached it'
        assert (row['attach_target_cluster_id'], row['attach_target_label']) == (
            'b-canon', 'duplicate',
        )


def _wording() -> types.ModuleType:
    return load_script_module(
        SCRIPT_PATH.parent / 'write_triage_judge_wording.py', 'write_triage_judge_wording',
    )


class TestTheWordingCli:
    """`main()` on a live seeded `--limit` run: `--wording` reaches the shipped call."""

    @staticmethod
    def _instructions(tmp_path: Path, monkeypatch, *extra: str) -> list[str]:
        """The system prompt every Responses call carried, for one live seeded run."""
        fixture = tmp_path / 'corpus.jsonl'
        fixture.write_text(''.join(json.dumps(record) + '\n' for record in _CORPUS))
        monkeypatch.setattr(sys, 'argv', [
            'eval_write_triage_judge.py', '--fixture', str(fixture),
            '--report-path', str(tmp_path / 'r.json'),
            '--limit', '3', '--distractors', '2', *extra,
        ])
        client = _openai('amends')
        with patch('openai.AsyncOpenAI', return_value=client):
            assert _mod().main() == 0
        create = client.responses.create
        assert create.await_count > 0, 'precondition: the shipped judge was reached'
        return [call.kwargs['instructions'] for call in create.await_args_list]

    def test_a_pre_psi_run_sends_the_pre_psi_prompt_and_restores_the_shipped_one(
        self, tmp_path: Path, monkeypatch,
    ) -> None:
        from fused_memory.server import write_triage_judge  # noqa: PLC0415

        sent = self._instructions(tmp_path, monkeypatch, '--wording', 'pre-psi')
        assert set(sent) == {_wording().PRE_PSI_JUDGE_SYSTEM_PROMPT}
        assert write_triage_judge.JUDGE_SYSTEM_PROMPT is _wording().system_prompt('shipped')

    def test_the_default_is_the_shipped_wording(self, tmp_path: Path, monkeypatch) -> None:
        sent = self._instructions(tmp_path, monkeypatch)
        assert set(sent) == {_wording().system_prompt('shipped')}
