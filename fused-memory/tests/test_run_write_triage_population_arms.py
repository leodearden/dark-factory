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

import httpx
import openai
import pytest
from _fm_helpers import load_script_module

from fused_memory.config.schema import FusedMemoryConfig
from fused_memory.models.enums import SourceStore
from fused_memory.models.memory import MemoryResult
from fused_memory.server import write_triage_judge
from fused_memory.server.write_triage_judge import (
    _ELIDED_MARKER,
    resolve_judge_field_chars,
    resolve_judge_timeout,
)

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


def _write(candidates: list[dict] | None = None, band_winner_id: str | None = 'c12') -> dict:
    return {
        'memory_id': 'w', 'project_id': 'reify', 'category': 'procedural_knowledge',
        'created_at': '2026-10-01T00:00:00+00:00', 'content': 'y' * 5_000,
        'metadata': {}, 'recon_marker': True, 'declares_attach_keys': False,
        'band': 'judge', 'band_winner_id': band_winner_id, 'similarity': 0.85,
        'retrieved_count': 20, 'self_retrieved': False, 'own_children_dropped': 0,
        'candidates': _slate_of_20() if candidates is None else candidates,
        'slates': 'frozen',
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
            'field_chars': resolve_judge_field_chars(_service()),
            'timeout_seconds': resolve_judge_timeout(_service()),
            'memory_id': 'w', 'project_id': 'reify', 'category': 'procedural_knowledge',
            'recon_marker': True, 'declares_attach_keys': False,
            'band': 'judge', 'band_winner_id': 'c12', 'slates': 'frozen',
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


# ---------------------------------------------------------------------------
# The runner
# ---------------------------------------------------------------------------

_REQUEST = httpx.Request('POST', 'https://api.openai.com/v1/responses')


def _api_error(cls: type, status: int) -> Exception:
    return cls(f'status {status}', response=httpx.Response(status, request=_REQUEST), body=None)


def _freeze_mod() -> types.ModuleType:
    return load_script_module(
        SCRIPTS / 'freeze_write_triage_population.py', 'freeze_write_triage_population',
    )


def _population_write(memory_id: str, band: str = 'judge') -> dict:
    write = _write(
        candidates=[_candidate('a', 0.7), _candidate('b', 0.6)],
        band_winner_id='a' if band != 'stored' else None,
    )
    return {**write, 'memory_id': memory_id, 'content': f'entry {memory_id}', 'band': band}


def _snapshot_file(tmp_path: Path, n_judge: int = 8, candidate_k: int = 20) -> Path:
    writes = [_population_write(f'w{i:02d}') for i in range(n_judge)]
    writes += [_population_write('det', 'restated'), _population_write('low', 'stored')]
    return _freeze_mod().write_snapshot(tmp_path / 'data', {
        'schema_version': 1, 'frozen_at': '2026-10-05T07:00:00+00:00',
        'candidate_k': candidate_k, 'writes': writes, 'targets': {},
    })


class _Provider:
    """The fake Responses endpoint: per-write scripted failures, concurrency observed."""

    def __init__(self, script: dict[str, list] | None = None, delay: float = 0.0) -> None:
        self._script = {key: list(steps) for key, steps in (script or {}).items()}
        self._delay = delay
        self.calls: list[tuple[str, str, str]] = []
        self._in_flight = 0
        self.max_in_flight = 0

    async def _create(self, **kwargs: Any) -> types.SimpleNamespace:
        memory_id = kwargs['input'].split('\n')[1].removeprefix('entry ')
        self.calls.append((memory_id, kwargs['instructions'], write_triage_judge.JUDGE_SYSTEM_PROMPT))
        self._in_flight += 1
        self.max_in_flight = max(self.max_in_flight, self._in_flight)
        try:
            await asyncio.sleep(self._delay)
            steps = self._script.get(memory_id)
            if steps:
                step = steps.pop(0)
                if isinstance(step, BaseException):
                    raise step
            return _response('{"verdict": "distinct"}')
        finally:
            self._in_flight -= 1

    def client(self) -> MagicMock:
        return _client(self._create)

    def called(self) -> list[str]:
        return [memory_id for memory_id, _, _ in self.calls]


def _run_arms(
    snapshot_path: Path,
    provider: _Provider,
    *,
    arms: list[str] | None = None,
    max_writes: int | None = None,
    concurrency: int = 2,
    budget_usd: float = 40.0,
    max_attempts: int = 4,
    sleep: AsyncMock | None = None,
) -> dict:
    snapshot, sha = _freeze_mod().load_snapshot(snapshot_path)
    with patch('openai.AsyncOpenAI', return_value=provider.client()):
        return asyncio.run(_mod().run_arms(
            snapshot, sha, snapshot_path.parent / 'arms',
            arms=[_arm(name) for name in (arms or ['gpt-4o-mini@5'])],
            service=_service(), max_writes=max_writes, concurrency=concurrency,
            budget_usd=budget_usd, sleep=sleep or AsyncMock(), max_attempts=max_attempts,
        ))


def _rows(snapshot_path: Path, arm_name: str) -> list[dict]:
    path = snapshot_path.parent / 'arms' / f'{arm_name}.jsonl'
    return [json.loads(line) for line in path.read_text().splitlines()]


def _order(snapshot_path: Path) -> list[str]:
    snapshot, sha = _freeze_mod().load_snapshot(snapshot_path)
    return [w['memory_id'] for w in _mod().judge_band_order(snapshot, sha)]


_ONE_CALL_USD = _scorer().LIST_PRICES['gpt-4o-mini'].usd(1_000, 50)


class TestTheRunSet:
    def test_the_order_is_the_snapshot_hash_order_of_the_judge_band(self, tmp_path: Path) -> None:
        import hashlib  # noqa: PLC0415

        path = _snapshot_file(tmp_path)
        snapshot, sha = _freeze_mod().load_snapshot(path)
        expected = sorted(
            (w['memory_id'] for w in snapshot['writes'] if w['band'] == 'judge'),
            key=lambda mid: hashlib.sha256(f'{sha}:{mid}'.encode()).hexdigest(),
        )
        assert _order(path) == expected
        assert {'det', 'low'}.isdisjoint(expected)
        shuffled = {**snapshot, 'writes': list(reversed(snapshot['writes']))}
        assert [w['memory_id'] for w in _mod().judge_band_order(shuffled, sha)] == expected

    def test_each_arm_judges_the_prefix_one_row_per_write(self, tmp_path: Path) -> None:
        path = _snapshot_file(tmp_path)
        provider = _Provider()
        summary = _run_arms(path, provider, max_writes=5)
        assert [row['memory_id'] for row in _rows(path, 'gpt-4o-mini@5')] == _order(path)[:5]
        assert sorted(provider.called()) == sorted(_order(path)[:5])
        assert summary['arms']['gpt-4o-mini@5'] == {
            'complete': True, 'rows': 5, 'missing': 0, 'written_now': 5,
        }
        assert summary['budget_exhausted'] is False
        assert summary['spent_usd'] == pytest.approx(5 * _ONE_CALL_USD)

    def test_no_limit_means_the_whole_judge_band(self, tmp_path: Path) -> None:
        path = _snapshot_file(tmp_path)
        _run_arms(path, _Provider())
        assert len(_rows(path, 'gpt-4o-mini@5')) == 8

    def test_an_arm_wider_than_the_frozen_slate_is_refused_before_any_call(
        self, tmp_path: Path,
    ) -> None:
        path = _snapshot_file(tmp_path, candidate_k=5)
        provider = _Provider()
        with pytest.raises(ValueError, match='gpt-4o-mini@20'):
            _run_arms(path, provider, arms=['gpt-4o-mini@5', 'gpt-4o-mini@20'])
        assert provider.calls == []


class TestResume:
    def test_a_second_run_fills_only_the_missing_rows(self, tmp_path: Path) -> None:
        path = _snapshot_file(tmp_path)
        _run_arms(path, _Provider(), max_writes=3)
        provider = _Provider()
        summary = _run_arms(path, provider, max_writes=6)
        assert sorted(provider.called()) == sorted(_order(path)[3:6])
        assert sorted(r['memory_id'] for r in _rows(path, 'gpt-4o-mini@5')) == sorted(
            _order(path)[:6],
        )
        assert summary['arms']['gpt-4o-mini@5']['written_now'] == 3

    def test_rows_from_another_snapshot_are_refused_naming_the_arm(self, tmp_path: Path) -> None:
        path = _snapshot_file(tmp_path)
        arms_dir = path.parent / 'arms'
        arms_dir.mkdir()
        (arms_dir / 'gpt-4o-mini@5.jsonl').write_text(
            json.dumps({'memory_id': 'w00', 'snapshot_sha256': 'f' * 64, 'usd': 0.0}) + '\n',
        )
        with pytest.raises(ValueError, match='gpt-4o-mini@5'):
            _run_arms(path, _Provider())


class TestTransportFailures:
    def test_transient_errors_are_retried_with_doubling_backoff_and_recorded(
        self, tmp_path: Path,
    ) -> None:
        path = _snapshot_file(tmp_path, n_judge=1)
        sleep = AsyncMock()
        provider = _Provider({'w00': [
            TimeoutError(),
            openai.APIConnectionError(request=_REQUEST),
            _api_error(openai.RateLimitError, 429),
            _api_error(openai.InternalServerError, 500),
        ]})
        _run_arms(path, provider, sleep=sleep, max_attempts=5)
        [row] = _rows(path, 'gpt-4o-mini@5')
        assert row['attempts'] == 5
        assert row['transport_failures'] == [
            'TimeoutError', 'APIConnectionError', 'RateLimitError', 'InternalServerError',
        ]
        assert [call.args[0] for call in sleep.await_args_list] == [2.0, 4.0, 8.0, 16.0]

    def test_an_exhausted_write_gets_no_row_and_the_arm_is_incomplete(
        self, tmp_path: Path,
    ) -> None:
        path = _snapshot_file(tmp_path, n_judge=3)
        stuck = _order(path)[1]
        provider = _Provider({stuck: [TimeoutError(), TimeoutError()]})
        summary = _run_arms(path, provider, max_attempts=2)
        assert stuck not in {r['memory_id'] for r in _rows(path, 'gpt-4o-mini@5')}
        assert len(_rows(path, 'gpt-4o-mini@5')) == 2
        assert summary['arms']['gpt-4o-mini@5'] == {
            'complete': False, 'rows': 2, 'missing': 1, 'written_now': 2,
        }

    @pytest.mark.parametrize('error', [
        _api_error(openai.AuthenticationError, 401),
        ValueError('a mis-configuration'),
    ])
    def test_a_non_transient_error_aborts_after_in_flight_calls_finish(
        self, tmp_path: Path, error: Exception,
    ) -> None:
        path = _snapshot_file(tmp_path)
        order = _order(path)
        provider = _Provider({order[0]: [error]}, delay=0.01)
        with pytest.raises(type(error)):
            _run_arms(path, provider, concurrency=2)
        assert [r['memory_id'] for r in _rows(path, 'gpt-4o-mini@5')] == [order[1]]
        assert sorted(provider.called()) == sorted(order[:2])


class TestBudget:
    def test_dispatch_stops_once_the_spend_reaches_the_cap(self, tmp_path: Path) -> None:
        path = _snapshot_file(tmp_path)
        summary = _run_arms(path, _Provider(), concurrency=1, budget_usd=2.5 * _ONE_CALL_USD)
        assert len(_rows(path, 'gpt-4o-mini@5')) == 3
        assert summary['budget_exhausted'] is True
        assert summary['arms']['gpt-4o-mini@5']['complete'] is False

    def test_calls_in_flight_when_the_cap_is_crossed_still_land(self, tmp_path: Path) -> None:
        path = _snapshot_file(tmp_path)
        cap = 0.5 * _ONE_CALL_USD
        summary = _run_arms(path, _Provider(delay=0.05), concurrency=4, budget_usd=cap)
        assert len(_rows(path, 'gpt-4o-mini@5')) == 4
        assert summary['budget_exhausted'] is True
        assert summary['spent_usd'] == pytest.approx(4 * _ONE_CALL_USD)
        assert cap < summary['spent_usd'] < cap + 4 * _ONE_CALL_USD

    def test_prior_spend_in_any_arm_file_counts_toward_the_cap(self, tmp_path: Path) -> None:
        path = _snapshot_file(tmp_path)
        _run_arms(path, _Provider(), arms=['gpt-4o-mini@20'], max_writes=2)
        provider = _Provider()
        summary = _run_arms(path, provider, budget_usd=2 * _ONE_CALL_USD)
        assert provider.calls == []
        assert summary['budget_exhausted'] is True
        assert summary['spent_usd'] == pytest.approx(2 * _ONE_CALL_USD)


class TestConcurrencyAndWording:
    def test_in_flight_calls_never_exceed_the_bound(self, tmp_path: Path) -> None:
        path = _snapshot_file(tmp_path)
        provider = _Provider(delay=0.01)
        _run_arms(path, provider, concurrency=3)
        assert 1 < provider.max_in_flight <= 3

    def test_arms_run_one_after_another_each_under_its_own_wording(
        self, tmp_path: Path,
    ) -> None:
        path = _snapshot_file(tmp_path)
        provider = _Provider(delay=0.001)
        _run_arms(path, provider, arms=['gpt-4o-mini@5+pre-psi', 'gpt-4o-mini@5'], concurrency=4)
        pre_psi = _wording().PRE_PSI_JUDGE_SYSTEM_PROMPT
        shipped = _wording().system_prompt('shipped')
        assert [sent for _, sent, _ in provider.calls] == [pre_psi] * 8 + [shipped] * 8
        assert all(sent == in_force for _, sent, in_force in provider.calls)
        assert write_triage_judge.JUDGE_SYSTEM_PROMPT is shipped


# ---------------------------------------------------------------------------
# Publish
# ---------------------------------------------------------------------------

def _calibrate() -> types.ModuleType:
    return load_script_module(SCRIPTS / 'calibrate_write_triage.py', 'calibrate_write_triage')


def _sha(data: bytes | str) -> str:
    import hashlib  # noqa: PLC0415

    return hashlib.sha256(data.encode() if isinstance(data, str) else data).hexdigest()


_LONG = 'z' * 5_000


def _publish_write(memory_id: str, band: str = 'judge', created_at: str | None = None) -> dict:
    write = _write(
        candidates=[
            _candidate('a', 0.7),
            _candidate('kid', 0.65, kind='sighting', parent_id='P'),
            _candidate('b', 0.6, content=_LONG),
        ],
        band_winner_id='a' if band != 'stored' else None,
    )
    return {
        **write, 'memory_id': memory_id, 'content': f'entry {memory_id}', 'band': band,
        'project_id': 'reify' if memory_id.endswith(('1', '3')) else 'dark_factory',
        'category': 'observations_and_summaries' if memory_id == 'w02' else 'procedural_knowledge',
        'recon_marker': memory_id in {'w00', 'det'},
        'declares_attach_keys': memory_id == 'w04',
        'created_at': created_at or '2026-10-01T00:00:00+00:00',
    }


class _Published:
    """A frozen snapshot of six judge-band writes, with arm files over a 4-write prefix."""

    def __init__(self, tmp_path: Path) -> None:
        (tmp_path / '.git').mkdir()
        writes = [
            _publish_write(f'w{i:02d}', created_at='2026-09-29T12:00:00-07:00' if i < 2 else None)
            for i in range(6)
        ]
        writes += [_publish_write('det', 'restated'), _publish_write('low', 'stored')]
        self.path = _freeze_mod().write_snapshot(tmp_path / 'data', {
            'schema_version': 2, 'frozen_at': '2026-10-05T07:00:00+00:00',
            'projects': ['dark_factory', 'reify'], 'candidate_k': 20, 'writes': writes,
            'targets': {'P': {'project_id': 'reify', 'content': 'the parent'}},
            'excluded_writes': {'undated': 1, 'vanished': 2},
            'slate_rows_dropped': {'self': 4, 'own_children': 3},
        })
        self.snapshot, self.sha = _freeze_mod().load_snapshot(self.path)
        self.order = [w['memory_id'] for w in _mod().judge_band_order(self.snapshot, self.sha)]
        self.run_set = self.order[:4]
        self.rows = {arm.name: self._stored_rows(arm) for arm in _mod().ARMS}
        r0, r1, r2, r3 = self.run_set
        self._answer('gpt-4o-mini@5', {
            r0: ('contested', 'a'), r1: ('contested', 'a'), r2: ('amended', 'a'),
        })
        self._answer('gpt-4o-mini@5+pre-psi', {
            r1: ('contested', 'a'), r2: ('contested', 'a'), r3: ('contested', 'b'),
        })
        self._answer('gpt-6.1-sol:low@20', {r0: ('amended', 'P'), r1: ('restated', 'gone')})
        mini = self.rows['gpt-4o-mini@5']
        for row, seconds in zip(mini, (1.0, 2.0, 3.0, 4.0), strict=True):
            row['judge_seconds'] = seconds
        mini[0]['transport_failures'] = ['TimeoutError', 'RateLimitError']
        mini[1]['transport_failures'] = ['TimeoutError']
        mini[2]['usd'] = None
        mini[3]['parse_failure'] = True

    def _stored_rows(self, arm: Any) -> list[dict]:
        return [{
            'arm': arm.name, 'judge_model': arm.model, 'reasoning_effort': arm.reasoning_effort,
            'width': arm.width, 'wording': arm.wording, 'snapshot_sha256': self.sha,
            'field_chars': 4_000, 'timeout_seconds': 15.0,
            'memory_id': memory_id, 'band': 'judge', 'band_winner_id': 'a',
            'outcome': 'stored', 'verdict_candidate_id': None, 'judged_candidate_id': None,
            'raw_text': '{"verdict": "distinct"}', 'usage': None, 'usd': 0.001,
            'judge_seconds': 1.0, 'parse_failure': False, 'failure': None,
            'attempts': 1, 'transport_failures': [],
        } for memory_id in self.run_set]

    def _answer(self, arm_name: str, answers: dict[str, tuple[str, str]]) -> None:
        for row in self.rows[arm_name]:
            if row['memory_id'] in answers:
                outcome, target = answers[row['memory_id']]
                row.update(outcome=outcome, verdict_candidate_id=target, judged_candidate_id=target)

    def judge_a_write_outside_the_snapshot(self, arm_name: str) -> None:
        self.rows[arm_name][0].update(
            memory_id='not-frozen', outcome='amended',
            verdict_candidate_id='a', judged_candidate_id='a',
        )

    def write_arm_files(self) -> None:
        arms_dir = self.path.parent / 'arms'
        arms_dir.mkdir(exist_ok=True)
        for name, rows in self.rows.items():
            _mod().arm_path(arms_dir, name).write_text(
                ''.join(json.dumps(row, sort_keys=True) + '\n' for row in rows),
            )

    def artifact(self, **kwargs: Any) -> dict:
        self.write_arm_files()
        return _mod().build_population_artifact(
            self.snapshot, self.sha, self.path, self.rows, budget_usd=40.0, **kwargs,
        )

    def pairs(self, already_rated: Any = frozenset()) -> tuple[list[dict], dict]:
        return _mod().build_pairs_to_rate(
            self.snapshot, self.sha, self.rows, already_rated=already_rated,
        )


@pytest.fixture
def published(tmp_path: Path) -> _Published:
    return _Published(tmp_path)


class TestThePopulationBlock:
    def test_counts_over_the_frozen_population_and_the_run_set(
        self, published: _Published,
    ) -> None:
        population = published.artifact()['population']
        later = sum(1 for m in published.run_set if m in {'w00', 'w01'})
        assert population == {
            'n_writes': 8,
            'n_judge_band': 4,
            'n_judge_band_frozen': 6,
            'judge_band_sample': {
                'order': 'sha256(snapshot_sha256:memory_id)', 'size': 4, 'of': 6,
            },
            'projects': ['dark_factory', 'reify'],
            'frozen_at': '2026-10-05T07:00:00+00:00',
            'snapshot_sha256': published.sha,
            'snapshot_path': 'data/write-triage-population-2026-10-05/snapshot.json',
            'by_category': {'observations_and_summaries': 1, 'procedural_knowledge': 7},
            'by_project': {'dark_factory': 6, 'reify': 2},
            'by_band': {'judge': 6, 'restated': 1, 'stored': 1},
            'recon_marker_writes': 2,
            'recon_marker_run_set': sum(1 for m in published.run_set if m == 'w00'),
            'declares_attach_keys_writes': 1,
            'excluded_writes': {'undated': 1, 'vanished': 2},
            'slate_rows_dropped': {'self': 4, 'own_children': 3},
            'judge_band_with_later_candidates': later,
        }

    def test_an_unsampled_run_records_no_sample(self, published: _Published) -> None:
        published.run_set = published.order
        published.rows = {arm.name: published._stored_rows(arm) for arm in _mod().ARMS}
        population = published.artifact()['population']
        assert (population['n_judge_band'], population['judge_band_sample']) == (6, None)


class TestTheArmRows:
    def test_one_row_per_arm_in_arms_order(self, published: _Published) -> None:
        arms = published.artifact()['arms']
        assert [row['arm'] for row in arms] == [arm.name for arm in _mod().ARMS]
        for row, arm in zip(arms, _mod().ARMS, strict=True):
            assert (row['model'], row['reasoning_effort'], row['width'], row['wording']) == (
                arm.model, arm.reasoning_effort, arm.width, arm.wording,
            )
            assert row['system_prompt_sha256'] == _sha(_wording().system_prompt(arm.wording))
            assert row['calls'] == 4

    def test_an_arms_provenance_and_measurements(self, published: _Published) -> None:
        [row] = [r for r in published.artifact()['arms'] if r['arm'] == 'gpt-4o-mini@5']
        cases = published.path.parent / 'arms' / 'gpt-4o-mini@5.jsonl'
        seconds = _calibrate().summarize_distribution([1.0, 2.0, 3.0, 4.0])
        assert row == {
            'arm': 'gpt-4o-mini@5', 'model': 'gpt-4o-mini', 'provider': 'openai',
            'reasoning_effort': None, 'width': 5, 'wording': 'shipped',
            'system_prompt_sha256': _sha(_wording().system_prompt('shipped')),
            'field_chars': 4_000, 'timeout_seconds': 15.0,
            'calls': 4, 'parse_failures': 1,
            'transport_failures': 3,
            'transport_failures_by_type': {'RateLimitError': 1, 'TimeoutError': 2},
            'usd': 0.003, 'unpriced_calls': 1,
            'p50_seconds': seconds['median'], 'p95_seconds': seconds['p95'],
            'outcomes': {'amended': 1, 'contested': 2, 'stored': 1},
            'cases_path': 'data/write-triage-population-2026-10-05/arms/gpt-4o-mini@5.jsonl',
            'cases_sha256': _sha(cases.read_bytes()),
        }

    def test_rows_disagreeing_on_the_field_cap_are_refused(self, published: _Published) -> None:
        published.rows['gpt-6-luna:low@5'][2]['field_chars'] = 1_200
        with pytest.raises(ValueError, match='gpt-6-luna:low@5'):
            published.artifact()


class TestPublishRefuses:
    def test_a_missing_arm(self, published: _Published) -> None:
        del published.rows['gpt-5.6-terra:none@20']
        with pytest.raises(ValueError, match='gpt-5.6-terra:none@20'):
            published.artifact()

    def test_a_row_from_another_snapshot(self, published: _Published) -> None:
        published.rows['gpt-6-luna:low@20'][1]['snapshot_sha256'] = 'f' * 64
        with pytest.raises(ValueError, match='gpt-6-luna:low@20'):
            published.artifact()

    def test_an_arm_covering_another_set_of_writes(self, published: _Published) -> None:
        published.rows['gpt-6.1-sol:low@5'][3]['memory_id'] = published.order[4]
        with pytest.raises(ValueError, match='gpt-6.1-sol:low@5'):
            published.artifact()

    def test_an_arm_with_a_write_judged_twice(self, published: _Published) -> None:
        rows = published.rows['gpt-6.1-sol:low@5']
        rows.append(dict(rows[0]))
        with pytest.raises(ValueError, match='gpt-6.1-sol:low@5'):
            published.artifact()

    def test_a_run_set_that_is_not_the_order_prefix(self, published: _Published) -> None:
        for rows in published.rows.values():
            rows[0]['memory_id'] = published.order[5]
        with pytest.raises(ValueError, match='prefix'):
            published.artifact()

    def test_the_pairs_refuse_a_row_of_a_write_outside_the_snapshot(
        self, published: _Published,
    ) -> None:
        published.judge_a_write_outside_the_snapshot('gpt-6-luna:low@5')
        with pytest.raises(ValueError, match='gpt-6-luna:low@5'):
            published.pairs()

    def test_a_row_judged_on_write_time_slates(self, published: _Published) -> None:
        published.rows['gpt-6-luna:low@5'][2]['slates'] = 'write-time'
        with pytest.raises(ValueError, match='gpt-6-luna:low@5'):
            published.artifact()
        with pytest.raises(ValueError, match='gpt-6-luna:low@5'):
            published.pairs()


def test_rows_saying_frozen_publish_beside_rows_that_predate_the_field(
    published: _Published,
) -> None:
    for rows in published.rows.values():
        rows[0]['slates'] = 'frozen'
    assert len(published.artifact()['arms']) == len(_mod().ARMS)
    published.pairs()


class TestThePublishCommand:
    @staticmethod
    def _publish(published: _Published, out: Path, population_out: Path | None = None) -> int:
        return _mod().main([
            'publish', '--snapshot', str(published.path),
            '--population-out', str(population_out or out / 'population.json'),
            '--pairs-out', str(out / 'pairs.jsonl'),
            '--already-rated', str(out / 'no-verdicts.jsonl'),
        ])

    def test_the_artifact_names_the_pairs_file_written_beside_it(
        self, published: _Published, tmp_path: Path,
    ) -> None:
        published.write_arm_files()
        assert self._publish(published, tmp_path) == 0
        artifact = json.loads((tmp_path / 'population.json').read_text(encoding='utf-8'))
        body = (tmp_path / 'pairs.jsonl').read_bytes()
        assert artifact['pairs_to_rate']['path'] == 'pairs.jsonl'
        assert artifact['pairs_to_rate']['sha256'] == _sha(body)
        assert list(tmp_path.glob('*.tmp')) == []

    def test_a_row_of_a_write_outside_the_snapshot_is_refused_before_anything_is_written(
        self, published: _Published, tmp_path: Path,
    ) -> None:
        published.judge_a_write_outside_the_snapshot('gpt-6-luna:low@5')
        published.write_arm_files()
        with pytest.raises(ValueError, match='gpt-6-luna:low@5'):
            self._publish(published, tmp_path)
        assert not (tmp_path / 'population.json').exists()
        assert not (tmp_path / 'pairs.jsonl').exists()

    def test_an_artifact_that_cannot_be_written_leaves_the_pairs_file_unwritten(
        self, published: _Published, tmp_path: Path,
    ) -> None:
        published.write_arm_files()
        with pytest.raises(FileNotFoundError):
            self._publish(published, tmp_path, population_out=tmp_path / 'absent' / 'p.json')
        assert not (tmp_path / 'pairs.jsonl').exists()
        assert list(tmp_path.glob('*.tmp')) == []


class TestTheSpend:
    def test_the_total_and_the_price_table_it_was_priced_at(self, published: _Published) -> None:
        spend = published.artifact()['spend']
        assert spend == {
            'usd_total': pytest.approx(0.001 * 39),
            'budget_usd': 40.0,
            'list_prices_as_of': _scorer().LIST_PRICES_AS_OF,
            'list_prices_source': _scorer().LIST_PRICES_SOURCE,
        }


class TestThePairsToRate:
    def test_one_blind_row_per_distinct_judged_pair(self, published: _Published) -> None:
        rows, stats = published.pairs()
        r0, r1, r2, r3 = published.run_set
        assert {(row['entry_id'], row['target_id']) for row in rows} == {
            (r0, 'a'), (r1, 'a'), (r2, 'a'), (r3, 'b'), (r0, 'P'), (r1, 'gone'),
        }
        assert all(
            set(row) == {'entry_id', 'target_id', 'entry_text', 'target_text'} for row in rows
        )
        assert stats['n_pairs'] == 6

    def test_texts_come_from_the_slate_then_the_targets_and_are_cut_at_4000(
        self, published: _Published,
    ) -> None:
        rows, stats = published.pairs()
        text = {(row['entry_id'], row['target_id']): row for row in rows}
        r0, r1, _, r3 = published.run_set
        assert text[(r0, 'a')]['entry_text'] == f'entry {r0}'
        assert text[(r0, 'a')]['target_text'] == 'content of a'
        assert text[(r0, 'P')]['target_text'] == 'the parent'
        assert text[(r3, 'b')]['target_text'] == _LONG[:4_000] + '…[truncated, 5000 chars total]'
        assert text[(r1, 'gone')]['target_text'] is None
        assert stats['missing_target_text'] == 1

    def test_the_order_is_the_snapshot_hash_order_of_the_pair(self, published: _Published) -> None:
        rows, stats = published.pairs()
        keys = [f'{published.sha}:{row["entry_id"]}:{row["target_id"]}' for row in rows]
        assert keys == sorted(keys, key=_sha)
        assert stats['order'] == 'sha256(snapshot_sha256:entry_id:target_id)'

    def test_pairs_already_rated_are_left_out_and_counted(self, published: _Published) -> None:
        r2 = published.run_set[2]
        rows, stats = published.pairs(already_rated={(r2, 'a')})
        assert (r2, 'a') not in {(row['entry_id'], row['target_id']) for row in rows}
        assert (stats['n_pairs'], stats['excluded_already_rated']) == (5, 1)

    def test_the_artifact_carries_the_pairs_block_it_was_given(
        self, published: _Published,
    ) -> None:
        block = {'n_pairs': 5, 'sha256': 'b' * 64}
        assert published.artifact(pairs_to_rate=block)['pairs_to_rate'] == block


class TestTheWordingReadout:
    def test_the_population_half_is_a_label_free_paired_readout(
        self, published: _Published,
    ) -> None:
        readout = published.artifact()['wording_attribution']['population']
        assert set(readout) == {'gpt-4o-mini@5+pre-psi', 'gpt-4o-mini@20+pre-psi'}
        five = readout['gpt-4o-mini@5+pre-psi']
        assert (five['shipped_arm'], five['other_arm']) == ('gpt-4o-mini@5', 'gpt-4o-mini@5+pre-psi')
        assert five['shipped'] == {
            'outcomes': {'amended': 1, 'contested': 2, 'stored': 1},
            'attaches': 3, 'contested': 2, 'contested_share_of_attaches': pytest.approx(2 / 3),
        }
        assert five['pre-psi'] == {
            'outcomes': {'contested': 3, 'stored': 1},
            'attaches': 3, 'contested': 3, 'contested_share_of_attaches': 1.0,
        }
        assert (five['contested_only_shipped'], five['contested_only_pre_psi']) == (1, 2)
        assert five['mcnemar_p'] == 1.0
        assert readout['gpt-4o-mini@20+pre-psi']['shipped']['contested_share_of_attaches'] is None

    def test_no_fixture_reports_means_no_fixture_half(self, published: _Published) -> None:
        assert published.artifact()['wording_attribution']['fixture'] is None


def _fixture_report(wording: str, **provenance: Any) -> dict:
    contested = {'shipped': 18, 'pre-psi': 9}[wording]
    return {
        'per_class': {
            'duplicate': {'n': 75, 'correct': 55, 'accuracy': 0.7333},
            'pseudo_contradiction': {'n': 6, 'correct': 3, 'accuracy': 0.5},
        },
        'confusion': {
            'duplicate': {'amended': 34, 'contested': contested, 'restated': 21, 'stored': 2},
            'pseudo_contradiction': {'amended': 2, 'contested': 3, 'restated': 1, 'stored': 0},
        },
        'false_contested': contested + 3,
        'production_shape': {'middle_band': {'n': 73, 'correct': 48, 'accuracy': 0.6575}},
        'provenance': {
            'field_chars': 4_000, 'slate_mode': 'retrieved', 'project_id': 'reify',
            'judge_model': 'gpt-4o-mini',
            'judge_system_prompt_sha256': _sha(_wording().system_prompt(wording)),
            **provenance,
        },
    }


class TestTheFixtureReadout:
    def test_per_wording_at_matched_width(self, published: _Published) -> None:
        fixture = published.artifact(fixture_reports={
            'shipped': _fixture_report('shipped'), 'pre-psi': _fixture_report('pre-psi'),
        })['wording_attribution']['fixture']
        assert fixture['shipped'] == {
            'duplicates_n': 75, 'duplicates_contested': 18, 'duplicates_contested_rate': 0.24,
            'false_contested': 21,
            'middle_band': {'n': 73, 'correct': 48, 'accuracy': 0.6575},
            'pseudo_contradiction_n': 6, 'pseudo_contradiction_contested': 3,
            'field_chars': 4_000, 'judge_model': 'gpt-4o-mini',
            'judge_system_prompt_sha256': _sha(_wording().system_prompt('shipped')),
        }
        assert (fixture['pre-psi']['duplicates_contested'], fixture['pre-psi']['false_contested']) == (
            9, 12,
        )

    @pytest.mark.parametrize('field', ['field_chars', 'slate_mode', 'project_id', 'judge_model'])
    def test_reports_that_are_not_matched_are_refused(
        self, published: _Published, field: str,
    ) -> None:
        mismatched = _fixture_report('pre-psi', **{field: 'other'})
        with pytest.raises(ValueError, match=field):
            published.artifact(fixture_reports={
                'shipped': _fixture_report('shipped'), 'pre-psi': mismatched,
            })

    def test_a_report_measured_under_the_other_wording_is_refused(
        self, published: _Published,
    ) -> None:
        with pytest.raises(ValueError, match='pre-psi'):
            published.artifact(fixture_reports={
                'shipped': _fixture_report('shipped'), 'pre-psi': _fixture_report('shipped'),
            })


# ---------------------------------------------------------------------------
# The committed artifacts
# ---------------------------------------------------------------------------

_PACKAGE = Path(__file__).parent.parent
_COMMITTED_POPULATION = _PACKAGE / 'calibration' / 'write_triage_population.json'
_COMMITTED_PAIRS = _PACKAGE / 'calibration' / 'write_triage_pairs_to_rate.jsonl'
_SEED_VERDICTS = Path(__file__).parent / 'fixtures' / 'write_triage_pair_verdicts_seed.jsonl'
_TRUNCATION = re.compile(r'…\[truncated, \d+ chars total\]$')


@functools.cache
def _committed() -> dict:
    return json.loads(_COMMITTED_POPULATION.read_text(encoding='utf-8'))


@functools.cache
def _committed_pairs() -> tuple[bytes, tuple[dict, ...]]:
    body = _COMMITTED_PAIRS.read_bytes()
    return body, tuple(json.loads(line) for line in body.decode('utf-8').splitlines())


class TestCommittedPopulationIsTraceable:
    """The committed π artifacts say which arms ran, how, and over what.

    No numeric floor is asserted here (e.g. ``n_judge_band >= 300``): that is
    the flip gate's predicate (PRD D3/D10), and a floor in a test would make
    the test the decision.
    """

    def test_the_arms_are_the_arm_table_in_order(self) -> None:
        rows = _committed()['arms']
        assert [row['arm'] for row in rows] == [arm.name for arm in _mod().ARMS]
        for row, arm in zip(rows, _mod().ARMS, strict=True):
            assert (row['model'], row['reasoning_effort'], row['width'], row['wording']) == (
                arm.model, arm.reasoning_effort, arm.width, arm.wording,
            )

    def test_each_arm_names_the_prompt_its_wording_sends(self) -> None:
        for row in _committed()['arms']:
            assert row['system_prompt_sha256'] == _sha(
                _wording().system_prompt(row['wording']),
            ), row['arm']

    def test_every_arm_ran_the_whole_run_set(self) -> None:
        n_judge_band = _committed()['population']['n_judge_band']
        for row in _committed()['arms']:
            assert {'calls', 'parse_failures', 'usd'} <= set(row), row['arm']
            assert row['calls'] == n_judge_band, row['arm']

    def test_the_population_says_what_was_frozen(self) -> None:
        population = _committed()['population']
        assert {
            'n_writes', 'n_judge_band', 'frozen_at', 'by_category', 'recon_marker_writes',
            'excluded_writes', 'slate_rows_dropped',
        } <= set(population)
        assert 'excluded' not in population
        assert population['projects'] == ['dark_factory', 'reify']
        assert re.fullmatch(r'[0-9a-f]{64}', population['snapshot_sha256'])

    def test_the_wording_readout_is_keyed_by_each_non_shipped_arm(self) -> None:
        readout = _committed()['wording_attribution']['population']
        assert set(readout) == {arm.name for arm in _mod().ARMS if arm.wording != 'shipped'}
        for name, entry in readout.items():
            assert entry['other_arm'] == name

    def test_the_pairs_file_is_the_one_the_artifact_names(self) -> None:
        body, rows = _committed_pairs()
        block = _committed()['pairs_to_rate']
        assert len(rows) == block['n_pairs']
        assert _sha(body) == block['sha256']

    def test_every_pair_row_is_blind(self) -> None:
        _, rows = _committed_pairs()
        for row in rows:
            assert set(row) == {'entry_id', 'target_id', 'entry_text', 'target_text'}, row

    def test_no_text_exceeds_the_rater_cap(self) -> None:
        _, rows = _committed_pairs()
        for row in rows:
            for key in ('entry_text', 'target_text'):
                text = row[key]
                if text is not None and len(text) > 4_000:
                    assert len(_TRUNCATION.sub('', text)) == 4_000, (row['entry_id'], key)
                    assert _TRUNCATION.search(text), (row['entry_id'], key)

    def test_no_seed_pair_is_asked_again(self) -> None:
        seed = {
            (row['entry_id'], row['target_id'])
            for row in map(json.loads, _SEED_VERDICTS.read_text().splitlines())
        }
        _, rows = _committed_pairs()
        assert not seed & {(row['entry_id'], row['target_id']) for row in rows}
