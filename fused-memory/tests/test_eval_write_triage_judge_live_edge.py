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
    OUTCOME_JUDGE,
    OUTCOME_RESTATED,
    OUTCOME_STORED,
)
from fused_memory.server.write_triage_judge import JUDGE_VERDICTS, VERDICT_KEY
from fused_memory.services.memory_service import SearchResults

SCRIPT_PATH = Path(__file__).parent.parent / 'scripts' / 'eval_write_triage_judge.py'

#: What the fake provider bills for every completion.
_USAGE = {'prompt_tokens': 321, 'completion_tokens': 7, 'total_tokens': 328}

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


def _openai(word: str) -> MagicMock:
    """A fake `AsyncOpenAI` that is its own async context manager, as the SDK is.

    Every completion answers *word* and bills :data:`_USAGE`. The response is
    plain namespaces because a MagicMock `usage` would hand `int()` a 1.
    """
    response = types.SimpleNamespace(
        choices=[types.SimpleNamespace(
            message=types.SimpleNamespace(content=json.dumps({VERDICT_KEY: word})),
        )],
        usage=types.SimpleNamespace(**_USAGE),
    )
    client = MagicMock()
    client.__aenter__ = AsyncMock(return_value=client)
    client.__aexit__ = AsyncMock(return_value=False)
    client.chat.completions.create = AsyncMock(return_value=response)
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
    """The candidate ids each awaited completion's user prompt named, in order."""
    prompts = [
        next(m['content'] for m in call.kwargs['messages'] if m['role'] == 'user')
        for call in create.await_args_list
    ]
    return [re.findall(r'^- id: (\S+)$', prompt, re.MULTILINE) for prompt in prompts]


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
        create = client.chat.completions.create
        assert create.await_count == len(plan.cases)
        assert _prompt_ids(create) == [case['candidates'] for case in plan.cases]

    def test_the_answer_is_the_parsed_verdict(self) -> None:
        plan = self._plan()
        case = plan.cases[0]
        with patch('openai.AsyncOpenAI', return_value=_openai('amends')):
            answer = _mod().build_judge_fn(_judge_config())(case, _resolved(plan, 0))
        assert (answer.outcome, answer.verdict) == (JUDGE_VERDICTS['amends'],) * 2

    def test_a_recorded_run_reports_what_the_provider_billed(self) -> None:
        plan = self._plan()
        case = plan.cases[0]
        with (
            patch('openai.AsyncOpenAI', return_value=_openai('restates')),
            _mod().usage_recording_openai() as recorded,
        ):
            answer = _mod().build_judge_fn(_judge_config(), recorded)(
                case, _resolved(plan, 0),
            )
        assert answer.usage == _USAGE


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
        client.chat.completions.create.assert_not_awaited()

    def test_a_middle_band_case_is_one_completion(self) -> None:
        case, candidates = self._case(0.7)
        assert case['band'] == OUTCOME_JUDGE, 'precondition: the shipped bands routed it'
        client = _openai('amends')
        with patch('openai.AsyncOpenAI', return_value=client):
            answer = _mod().build_retrieved_judge_fn(_judge_config())(case, candidates)
        assert client.chat.completions.create.await_count == 1
        assert (answer.outcome, answer.verdict) == (JUDGE_VERDICTS['amends'],) * 2


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
        assert _prompt_ids(client.chat.completions.create) == [
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
    """`main()` on a retrieved `--dry-run` over a stub store: what the plan is built under, and from.

    A retrieved plan builds a `MemoryService`, whose stores construct SDK
    clients of their own, so it must be built outside `usage_recording_openai`,
    which swaps `openai.AsyncOpenAI` for a plain factory.
    """

    @staticmethod
    def _main(tmp_path: Path, monkeypatch, *extra: str, rows=()) -> list:
        """Every search answers *rows*. Returns the SDK class each store was built under."""
        import openai  # noqa: PLC0415

        built_under: list = []

        class _LiveStore(FakeMemoryService):
            def __init__(self, config) -> None:
                super().__init__(rows)
                built_under.append(openai.AsyncOpenAI)

            async def initialize(self) -> None:
                return None

            async def close(self) -> None:
                return None

        monkeypatch.setattr('fused_memory.services.memory_service.MemoryService', _LiveStore)
        monkeypatch.setattr(
            sys, 'argv', TestTheLimitedCli._argv(tmp_path, '--slate-mode', 'retrieved', *extra),
        )
        assert _mod().main() == 0
        return built_under

    def test_the_store_is_built_under_the_sdks_own_client_class(
        self, tmp_path: Path, monkeypatch,
    ) -> None:
        import openai  # noqa: PLC0415

        assert self._main(tmp_path, monkeypatch) == [openai.AsyncOpenAI]

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
