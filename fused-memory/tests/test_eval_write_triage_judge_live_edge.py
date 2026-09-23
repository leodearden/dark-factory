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


def _resolved(plan, case) -> list:
    return [plan.records_by_id[cid] for cid in case['candidates']]


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
            answer = _mod().build_judge_fn(_judge_config())(case, _resolved(plan, case))
        assert (answer.outcome, answer.verdict) == (JUDGE_VERDICTS['amends'],) * 2

    def test_a_recorded_run_reports_what_the_provider_billed(self) -> None:
        plan = self._plan()
        case = plan.cases[0]
        with (
            patch('openai.AsyncOpenAI', return_value=_openai('restates')),
            _mod().usage_recording_openai() as recorded,
        ):
            answer = _mod().build_judge_fn(_judge_config(), recorded)(
                case, _resolved(plan, case),
            )
        assert answer.usage == _USAGE


class _Store:
    """The two reads the retrieval edge makes: one search, one liveness probe."""

    def __init__(self, rows: list[MemoryResult]) -> None:
        self.search = AsyncMock(return_value=SearchResults(rows))

    async def get_memory_by_id(self, project_id: str, memory_id: str) -> dict:
        return {'id': memory_id}


class TestTheRetrievedLiveEdge:
    """`build_retrieved_judge_fn`: only the middle band reaches the provider."""

    @staticmethod
    def _case(cosine: float) -> tuple[dict, list]:
        """A duplicate retrieving its canonical at *cosine*, routed by the shipped bands."""
        retrieval = _mod().load_retrieval()
        records = [_rec('canon', 'canon', 'canonical'), _rec('dup', 'canon', 'duplicate')]
        labelled = records[1:]
        store = _Store([MemoryResult(
            id='canon', content='content of canon', source_store=SourceStore.mem0,
            metadata={'store_score': cosine},
        )])
        retrievals = asyncio.run(
            retrieval.prefetch_retrievals(store, labelled, project_id='reify', k=5),
        )
        slates = retrieval.retrieved_slates(
            labelled, retrievals, t_high=T_HIGH, t_low=T_LOW, judge_candidate_count=5,
        )
        plan = _mod().plan_from_slates(records, slates, provenance={})
        [case] = plan.cases
        return case, _resolved(plan, case)

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
