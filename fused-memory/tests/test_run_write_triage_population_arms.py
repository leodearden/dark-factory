"""Tests for run_write_triage_population_arms.py — the π judge arms (task 6151).

Every arm call goes through the SHIPPED ``write_triage_judge._call_llm``, with
the OpenAI SDK faked at ``openai.AsyncOpenAI``. Rows are checked against ι's
``JudgedCase.from_row`` so μ can score them unchanged. No network, no key.
"""
from __future__ import annotations

import asyncio
import functools
import json
import re
import types
from collections.abc import Callable
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from _fm_helpers import load_script_module

from fused_memory.config.schema import FusedMemoryConfig
from fused_memory.models.enums import SourceStore
from fused_memory.models.memory import MemoryResult
from fused_memory.server import write_triage_judge
from fused_memory.server.write_triage_judge import _ELIDED_MARKER, resolve_judge_field_chars

SCRIPTS = Path(__file__).parent.parent / 'scripts'

#: A candidate line of the rendered user prompt, capturing its id.
_ID_LINE = re.compile(r'^- id: (\S+)$', re.MULTILINE)

SNAPSHOT_SHA = 'a' * 64


@functools.cache
def _mod() -> types.ModuleType:
    return load_script_module(
        SCRIPTS / 'run_write_triage_population_arms.py', 'run_write_triage_population_arms',
    )


def _wording() -> types.ModuleType:
    return load_script_module(
        SCRIPTS / 'write_triage_judge_wording.py', 'write_triage_judge_wording',
    )


def _scorer() -> types.ModuleType:
    return load_script_module(SCRIPTS / 'score_write_triage_pairs.py', 'score_write_triage_pairs')


def _retrieval() -> types.ModuleType:
    return load_script_module(
        SCRIPTS / 'eval_write_triage_retrieval.py', 'eval_write_triage_retrieval',
    )


def _service() -> types.SimpleNamespace:
    return types.SimpleNamespace(config=FusedMemoryConfig())


def _candidate(memory_id: str, cosine: float, content: str | None = None, **metadata: Any) -> dict:
    """A frozen slate record: ``normalize()`` of a store row plus its created_at."""
    row = MemoryResult(
        id=memory_id, content=content or f'content of {memory_id}',
        source_store=SourceStore.mem0, metadata={'store_score': cosine, **metadata},
        created_at='2026-09-30T00:00:00+00:00',
    )
    return {**_retrieval().normalize(row), 'created_at': row.created_at}


def _slate_of_20() -> list[dict]:
    """c00 is a sighting of c12, so the band winner is c12, twelve places down."""
    return [
        _candidate(
            f'c{i:02d}', round(0.85 - 0.01 * i, 2),
            content=('x' * 5_000) if i == 1 else None,
            **({'kind': 'sighting', 'parent_id': 'c12'} if i == 0 else {}),
        )
        for i in range(20)
    ]


def _write(candidates: list[dict] | None = None, band_winner_id: str = 'c12') -> dict:
    return {
        'memory_id': 'w', 'project_id': 'reify', 'category': 'procedural_knowledge',
        'created_at': '2026-10-01T00:00:00+00:00', 'content': 'y' * 5_000,
        'metadata': {}, 'recon_marker': True, 'declares_attach_keys': False,
        'band': 'judge', 'band_winner_id': band_winner_id, 'similarity': 0.85,
        'retrieved_count': 20, 'self_retrieved': False, 'own_children_dropped': 0,
        'candidates': _slate_of_20() if candidates is None else candidates,
    }


def _response(
    output_text: str, *, status: str = 'completed', input_tokens: int = 1_000,
    output_tokens: int = 50, reasoning_tokens: int | None = 3,
) -> types.SimpleNamespace:
    return types.SimpleNamespace(
        output_text=output_text,
        status=status,
        incomplete_details=(
            types.SimpleNamespace(reason='max_output_tokens') if status == 'incomplete' else None
        ),
        usage=types.SimpleNamespace(
            input_tokens=input_tokens, output_tokens=output_tokens,
            output_tokens_details=types.SimpleNamespace(reasoning_tokens=reasoning_tokens),
        ),
    )


def _client(respond: Callable[..., Any]) -> MagicMock:
    """A fake ``AsyncOpenAI`` that is its own async context manager, as the SDK is."""
    client = MagicMock()
    client.__aenter__ = AsyncMock(return_value=client)
    client.__aexit__ = AsyncMock(return_value=False)
    client.responses.create = AsyncMock(side_effect=respond)
    client.chat.completions.create = AsyncMock()
    return client


def _answering(text: str, **response: Any) -> MagicMock:
    async def _respond(**kwargs: Any) -> types.SimpleNamespace:
        return _response(text, **response)

    return _client(_respond)


def _clock(*readings: float) -> Callable[[], float]:
    values = iter(readings)
    return lambda: next(values)


def _judge(client: MagicMock, arm: Any, write: dict | None = None, clock=None) -> dict:
    async def _go() -> dict:
        with _wording().judge_wording(arm.wording):
            return await _mod().judge_for_arm(
                _write() if write is None else write, arm, service=_service(),
                snapshot_sha256=SNAPSHOT_SHA,
                clock=clock or _clock(10.0, 12.5),
            )

    with patch('openai.AsyncOpenAI', return_value=client):
        return asyncio.run(_go())


def _arm(name: str) -> Any:
    [arm] = [a for a in _mod().ARMS if a.name == name]
    return arm


class TestArms:
    def test_the_ten_arms_in_cost_order(self) -> None:
        assert [arm.name for arm in _mod().ARMS] == [
            'gpt-4o-mini@5', 'gpt-4o-mini@5+pre-psi',
            'gpt-4o-mini@20', 'gpt-4o-mini@20+pre-psi',
            'gpt-6-luna:low@5', 'gpt-6-luna:low@20',
            'gpt-5.6-terra:none@5', 'gpt-5.6-terra:none@20',
            'gpt-6.1-sol:low@5', 'gpt-6.1-sol:low@20',
        ]

    def test_each_arm_is_its_fields(self) -> None:
        assert [
            (a.model, a.reasoning_effort, a.width, a.wording) for a in _mod().ARMS
        ] == [
            ('gpt-4o-mini', None, 5, 'shipped'), ('gpt-4o-mini', None, 5, 'pre-psi'),
            ('gpt-4o-mini', None, 20, 'shipped'), ('gpt-4o-mini', None, 20, 'pre-psi'),
            ('gpt-6-luna', 'low', 5, 'shipped'), ('gpt-6-luna', 'low', 20, 'shipped'),
            ('gpt-5.6-terra', 'none', 5, 'shipped'), ('gpt-5.6-terra', 'none', 20, 'shipped'),
            ('gpt-6.1-sol', 'low', 5, 'shipped'), ('gpt-6.1-sol', 'low', 20, 'shipped'),
        ]

    def test_names_are_unique_and_filename_safe(self) -> None:
        names = [arm.name for arm in _mod().ARMS]
        assert len(set(names)) == len(names)
        assert all(re.fullmatch(r'[A-Za-z0-9.:@+-]+', name) for name in names)

    def test_every_model_is_priced(self) -> None:
        assert {arm.model for arm in _mod().ARMS} <= set(_scorer().LIST_PRICES)

    def test_an_arm_is_immutable(self) -> None:
        import dataclasses  # noqa: PLC0415

        with pytest.raises(dataclasses.FrozenInstanceError):
            _mod().ARMS[0].width = 7  # type: ignore[misc]


class TestTheWire:
    @pytest.mark.parametrize('name', [arm_name for arm_name in (
        'gpt-4o-mini@5', 'gpt-4o-mini@20+pre-psi', 'gpt-6-luna:low@5',
        'gpt-5.6-terra:none@20', 'gpt-6.1-sol:low@20',
    )])
    def test_the_request_is_the_arms_model_sampling_and_wording(self, name: str) -> None:
        arm = _arm(name)
        client = _answering('{"verdict": "distinct"}')
        _judge(client, arm)
        kwargs = client.responses.create.call_args.kwargs
        assert kwargs['model'] == arm.model
        if arm.reasoning_effort is None:
            assert (kwargs['temperature'], 'reasoning' in kwargs) == (0.0, False)
        else:
            assert kwargs['reasoning'] == {'effort': arm.reasoning_effort}
            assert 'temperature' not in kwargs
        assert kwargs['instructions'] == _wording().system_prompt(arm.wording)

    @pytest.mark.parametrize(('name', 'shown'), [
        ('gpt-4o-mini@5', ['c00', 'c01', 'c02', 'c03', 'c12']),
        ('gpt-4o-mini@20', [f'c{i:02d}' for i in range(20)]),
    ])
    def test_the_slate_is_the_top_width_with_the_band_winner_guaranteed(
        self, name: str, shown: list[str],
    ) -> None:
        client = _answering('{"verdict": "distinct"}')
        _judge(client, _arm(name))
        assert _ID_LINE.findall(client.responses.create.call_args.kwargs['input']) == shown

    def test_every_field_is_cut_at_the_configured_cap(self) -> None:
        cap = resolve_judge_field_chars(_service())
        client = _answering('{"verdict": "distinct"}')
        _judge(client, _arm('gpt-4o-mini@5'))
        rendered = client.responses.create.call_args.kwargs['input']
        for field in ('y' * 5_000, 'x' * 5_000):
            assert field[:cap] + _ELIDED_MARKER in rendered
            assert field[:cap + 1] not in rendered

    def test_only_the_openai_provider_is_judged(self) -> None:
        service = _service()
        service.config.write_triage.judge_provider = 'anthropic'
        client = _answering('{"verdict": "distinct"}')
        with patch('openai.AsyncOpenAI', return_value=client), pytest.raises(ValueError):
            asyncio.run(_mod().judge_for_arm(
                _write(), _arm('gpt-4o-mini@5'), service=service,
                snapshot_sha256=SNAPSHOT_SHA,
            ))
        client.responses.create.assert_not_awaited()


class TestTheRow:
    def test_an_attach_naming_a_child_files_against_its_hoisted_parent(self) -> None:
        arm = _arm('gpt-6.1-sol:low@5')
        text = '{"verdict": "amends", "candidate_id": "c00"}'
        row = _judge(_answering(text), arm)
        price = _scorer().LIST_PRICES[arm.model]
        assert row == {
            'arm': arm.name, 'judge_model': arm.model, 'reasoning_effort': 'low',
            'width': 5, 'wording': 'shipped', 'snapshot_sha256': SNAPSHOT_SHA,
            'memory_id': 'w', 'project_id': 'reify', 'category': 'procedural_knowledge',
            'recon_marker': True, 'declares_attach_keys': False,
            'band': 'judge', 'band_winner_id': 'c12',
            'outcome': 'amended', 'verdict_candidate_id': 'c00', 'judged_candidate_id': 'c12',
            'raw_text': text,
            'usage': {'prompt_tokens': 1_000, 'completion_tokens': 50, 'reasoning_tokens': 3},
            'usd': price.usd(1_000, 50),
            'judge_seconds': 2.5,
            'parse_failure': False, 'failure': None,
            'attempts': 1, 'transport_failures': [],
        }

    @pytest.mark.parametrize(('word', 'outcome'), [
        ('restates', 'restated'), ('amends', 'amended'), ('contests', 'contested'),
    ])
    def test_each_attach_word_maps_to_its_ack(self, word: str, outcome: str) -> None:
        row = _judge(
            _answering(json.dumps({'verdict': word, 'candidate_id': 'c03'})),
            _arm('gpt-4o-mini@5'),
        )
        assert (row['outcome'], row['verdict_candidate_id'], row['judged_candidate_id']) == (
            outcome, 'c03', 'c03',
        )

    def test_the_seconds_are_the_provider_call_alone(self) -> None:
        readings: list[float] = []

        def clock() -> float:
            readings.append(float(len(readings)))
            return readings[-1]

        row = _judge(_answering('{"verdict": "distinct"}'), _arm('gpt-4o-mini@5'), clock=clock)
        assert len(readings) == 2
        assert row['judge_seconds'] == 1.0

    def test_distinct_is_stored_naming_nothing(self) -> None:
        row = _judge(_answering('{"verdict": "distinct"}'), _arm('gpt-4o-mini@5'))
        assert (row['outcome'], row['verdict_candidate_id'], row['judged_candidate_id']) == (
            'stored', None, None,
        )
        assert row['parse_failure'] is False

    def test_an_unparseable_reply_is_a_priced_fail_open(self) -> None:
        arm = _arm('gpt-4o-mini@5')
        row = _judge(_answering('{"verdict":"maybe"}'), arm)
        assert (row['outcome'], row['judged_candidate_id'], row['parse_failure']) == (
            'stored', None, True,
        )
        assert row['raw_text'] == '{"verdict":"maybe"}'
        assert row['usage'] == {
            'prompt_tokens': 1_000, 'completion_tokens': 50, 'reasoning_tokens': 3,
        }
        assert row['usd'] == _scorer().LIST_PRICES[arm.model].usd(1_000, 50)
        assert 'JudgeOutputError' in row['failure']

    def test_an_incomplete_response_is_an_unpriced_fail_open(self) -> None:
        row = _judge(
            _answering('{"verdict": "dist', status='incomplete'), _arm('gpt-6.1-sol:low@20'),
        )
        assert (row['outcome'], row['parse_failure']) == ('stored', True)
        assert (row['raw_text'], row['usage'], row['usd']) == (None, None, None)
        assert 'JudgeOutputError' in row['failure']
        assert row['judge_seconds'] == 2.5

    @pytest.mark.parametrize('text', [
        '{"verdict": "amends", "candidate_id": "c00"}',
        '{"verdict": "contests", "candidate_id": "c05"}',
        '{"verdict": "distinct"}',
        '{"verdict":"maybe"}',
    ])
    def test_every_row_is_one_iota_scores_unchanged(self, text: str) -> None:
        row = _judge(_answering(text), _arm('gpt-4o-mini@20'))
        case = _scorer().JudgedCase.from_row(row)
        assert (case.arm, case.memory_id, case.in_judge_band) == ('gpt-4o-mini@20', 'w', True)

    def test_an_incomplete_row_is_one_iota_scores_unchanged(self) -> None:
        row = _judge(_answering('', status='incomplete'), _arm('gpt-4o-mini@5'))
        assert _scorer().JudgedCase.from_row(row).parse_failure is True

    def test_the_shipped_prompt_is_back_after_a_pre_psi_arm(self) -> None:
        _judge(_answering('{"verdict": "distinct"}'), _arm('gpt-4o-mini@5+pre-psi'))
        assert write_triage_judge.JUDGE_SYSTEM_PROMPT is _wording().system_prompt('shipped')
