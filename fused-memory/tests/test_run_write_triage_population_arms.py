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


def _write(candidates: list[dict] | None = None, band_winner_id: str | None = 'c12') -> dict:
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
