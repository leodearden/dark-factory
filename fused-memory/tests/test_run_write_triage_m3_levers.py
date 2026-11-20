"""Tests for run_write_triage_m3_levers.py — the lever replay's LLM edge (task 7519).

No network: every call goes through the SHIPPED ``write_triage_judge.judge_write``
into a faked ``openai.AsyncOpenAI`` whose Responses call answers from a script.
"""
from __future__ import annotations

import asyncio
import functools
import hashlib
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
from _sdk_client_stubs import sdk_client_double

from fused_memory.config.schema import FusedMemoryConfig
from fused_memory.server import write_triage_judge

SCRIPTS = Path(__file__).parent.parent / 'scripts'


@functools.cache
def _mod() -> types.ModuleType:
    return load_script_module(
        SCRIPTS / 'run_write_triage_m3_levers.py', mod_name='run_write_triage_m3_levers',
    )


@functools.cache
def _cases_mod() -> types.ModuleType:
    return load_script_module(
        SCRIPTS / 'write_triage_m3_replay_cases.py', mod_name='write_triage_m3_replay_cases',
    )


@functools.cache
def _wording() -> types.ModuleType:
    return load_script_module(
        SCRIPTS / 'write_triage_judge_wording.py', mod_name='write_triage_judge_wording',
    )


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


_SHIPPED = write_triage_judge.JUDGE_SYSTEM_PROMPT
_REQUEST = httpx.Request('POST', 'https://api.openai.com/v1/responses')
_CASES_SHA = 'c' * 64


def _config(min_parent_cosine: float | None = None) -> Any:
    config = FusedMemoryConfig()
    return config.model_copy(update={'write_triage': config.write_triage.model_copy(update={
        'judge_enabled': True, 'judge_provider': 'openai',
        'min_parent_cosine': min_parent_cosine, 'hoist_max_cosine_gap': None,
    })})


def _case(
    entry: str = 'E1', *, route: str = 'judge', cos: float | None = 0.8,
    soak_band: str = 'judge', soak_outcome: str = 'amended',
    child: str = 'a child entry', parent: str = 'a parent entry',
) -> Any:
    cases = _cases_mod()
    features = cases.CaseFeatures(
        entry_id=entry, target_id=f'T-{entry}', stratum='amended-sample', project='dark_factory',
        soak_outcome=soak_outcome, soak_band=soak_band, via_sibling=False, cross_category=False,
        parent_edited_after_attach=False, supersedes_target=False, route=cases.Route(route),
        cos_parent=cos, source_cosine=cos, metadata_recovered=True,
        child_chars=len(child), parent_chars=len(parent),
    )
    return cases.ReplayCase(features, child, parent)


def _reply(verdict: str, candidate: str | None = None, *, tokens: int = 1000) -> Any:
    body: dict[str, Any] = {'verdict': verdict}
    if candidate is not None:
        body['candidate_id'] = candidate
    return types.SimpleNamespace(
        output_text=json.dumps(body),
        usage=types.SimpleNamespace(
            input_tokens=tokens, output_tokens=tokens // 10, output_tokens_details=None,
        ),
        status='completed',
        incomplete_details=None,
    )


def _client(answer: Callable[[dict[str, Any]], Any]) -> MagicMock:
    """A fake AsyncOpenAI whose Responses call answers ``answer(kwargs)`` (or raises it)."""
    client = sdk_client_double()

    async def create(**kwargs: Any) -> Any:
        result = answer(kwargs)
        if isinstance(result, BaseException):
            raise result
        return result

    client.responses.create = AsyncMock(side_effect=create)
    return client


def _amends(kwargs: dict[str, Any]) -> Any:
    return _reply('amends', _target(kwargs))


def _target(kwargs: dict[str, Any]) -> str:
    match = re.search(r'- id: (\S+)', kwargs['input'])
    assert match is not None
    return match.group(1)


def _arm(name: str = 'f3p') -> Any:
    return next(arm for arm in _mod().ARMS if arm.name == name)


async def _judge(case: Any, arm: Any, client: MagicMock, *, config: Any = None,
                 sleep: Any = None) -> dict[str, Any]:
    service = _mod().arm_service(config or _config(), arm)
    with patch('openai.AsyncOpenAI', return_value=client), \
            _wording().judge_system_prompt(arm.prompt, label=arm.name) as prompt_sha:
        return await _mod().judge_case(
            case, arm, service, cases_sha256=_CASES_SHA, prompt_sha256=prompt_sha,
            sleep=sleep or AsyncMock(),
        )


def _run(cases: list[Any], arms: list[Any], arms_dir: Path, client: MagicMock, *,
         budget_usd: float = 100.0, concurrency: int = 1, config: Any = None) -> Any:
    with patch('openai.AsyncOpenAI', return_value=client):
        return asyncio.run(_mod().run_arms(
            cases, arms, config=config or _config(), arms_dir=arms_dir, budget_usd=budget_usd,
            cases_sha256=_CASES_SHA, concurrency=concurrency, sleep=AsyncMock(),
        ))


def _rows(arms_dir: Path, arm: str) -> list[dict[str, Any]]:
    path = arms_dir / f'{arm}.jsonl'
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


class TestTheArmTable:
    def test_names_are_unique_and_filename_safe(self) -> None:
        names = [arm.name for arm in _mod().ARMS]
        assert len(names) == len(set(names))
        assert all(re.fullmatch(r'[a-z0-9][a-z0-9-]*', name) for name in names)

    def test_kinds_and_parents(self) -> None:
        seen: set[str] = set()
        for arm in _mod().ARMS:
            assert arm.kind in {'judge', 'second_pass', 'transform'}
            if arm.kind == 'judge':
                assert arm.parent is None
            else:
                assert arm.parent in seen, arm.name
            assert arm.scope in {'all', 'soak-judge-band'}
            seen.add(arm.name)

    def test_only_the_shipped_prompt_is_shared(self) -> None:
        prompted = [arm for arm in _mod().ARMS if arm.kind != 'transform']
        assert all(arm.prompt for arm in prompted)
        by_sha: dict[str, list[str]] = {}
        for arm in prompted:
            by_sha.setdefault(_sha(arm.prompt), []).append(arm.name)
        shared = [names for names in by_sha.values() if len(names) > 1]
        assert shared == [['f3p', 'f3p-repeat', 'b1-sol-medium', 'b2-sol-high']]
        assert _arm('f3p').prompt is _SHIPPED
        assert _arm('psi-repro').prompt == _wording().PSI_JUDGE_SYSTEM_PROMPT

    def test_the_baseline_arms_run_the_shipped_model_at_each_effort(self) -> None:
        assert [(a.model, a.effort, a.scope) for a in map(_arm, ('psi-repro', 'f3p', 'f3p-repeat'))] == [
            ('gpt-6.1-sol', 'low', 'soak-judge-band'),
            ('gpt-6.1-sol', 'low', 'all'), ('gpt-6.1-sol', 'low', 'all'),
        ]
        assert (_arm('b1-sol-medium').effort, _arm('b2-sol-high').effort) == ('medium', 'high')


class TestJudgeCase:
    def test_drives_the_shipped_judge_with_a_one_candidate_slate(self) -> None:
        client = _client(_amends)
        row = asyncio.run(_judge(_case(), _arm('b1-sol-medium'), client))
        sent = client.responses.create.call_args.kwargs
        assert sent['instructions'] is _SHIPPED
        assert sent['model'] == 'gpt-6.1-sol'
        assert re.findall(r'- id: (\S+)', str(sent['input'])) == ['T-E1']
        assert row['judge_outcome'] == 'amended'
        assert row['candidate_id'] == 'T-E1'
        assert row['routed_outcome'] == 'amended'
        assert (row['arm'], row['entry_id'], row['target_id']) == ('b1-sol-medium', 'E1', 'T-E1')
        assert (row['model'], row['effort']) == ('gpt-6.1-sol', 'medium')
        assert row['prompt_sha256'] == _sha(_SHIPPED)
        assert row['cases_sha256'] == _CASES_SHA
        assert (row['calls'], row['attempts']) == (1, 1)
        assert row['seconds'] >= 0
        assert (row['input_tokens'], row['output_tokens']) == (1000, 100)
        assert row['usd'] == pytest.approx((1000 * 2.00 + 100 * 10.00) / 1e6)
        assert row['error_type'] is None

    def test_the_candidate_is_shown_even_without_a_measured_cosine(self) -> None:
        client = _client(_amends)
        row = asyncio.run(_judge(_case(cos=None), _arm(), client))
        assert row['calls'] == 1
        assert row['judge_outcome'] == 'amended'

    def test_an_unreadable_answer_fails_open(self) -> None:
        client = _client(lambda _: _reply('maybe'))
        row = asyncio.run(_judge(_case(), _arm(), client))
        assert row['judge_outcome'] == 'stored'
        assert row['routed_outcome'] == 'stored'
        assert row['error_type'] == 'JudgeOutputError'
        assert row['error']
        assert client.responses.create.await_count == 1

    def test_a_timeout_that_survives_the_retries_fails_open(self) -> None:
        client = _client(lambda _: TimeoutError())
        sleep = AsyncMock()
        row = asyncio.run(_judge(_case(), _arm(), client, sleep=sleep))
        assert row['judge_outcome'] == 'stored'
        assert row['error_type'] == 'TimeoutError'
        assert row['attempts'] == 4
        assert [call.args[0] for call in sleep.await_args_list] == [2.0, 4.0, 8.0]

    def test_a_rate_limit_is_retried(self) -> None:
        answers = iter([
            openai.RateLimitError('429', response=httpx.Response(429, request=_REQUEST), body=None),
        ])
        client = _client(lambda kwargs: next(answers, None) or _amends(kwargs))
        row = asyncio.run(_judge(_case(), _arm(), client))
        assert row['judge_outcome'] == 'amended'
        assert row['attempts'] == 2
        assert row['transport_failures'] == ['RateLimitError']

    @pytest.mark.parametrize(('route', 'outcome'), [
        ('supersession', 'stored'), ('force_stored', 'stored'), ('near_verbatim', 'restated'),
    ])
    def test_a_route_that_needs_no_judge_makes_no_call(self, route: str, outcome: str) -> None:
        client = _client(_amends)
        row = asyncio.run(_judge(_case(route=route), _arm(), client))
        assert client.responses.create.await_count == 0
        assert (row['calls'], row['usd'], row['judge_outcome']) == (0, 0.0, None)
        assert row['routed_outcome'] == outcome

    def test_the_routed_outcome_applies_the_configs_attach_guard(self) -> None:
        client = _client(_amends)
        row = asyncio.run(_judge(_case(cos=0.59), _arm(), client, config=_config(0.60)))
        assert row['judge_outcome'] == 'amended'
        assert row['routed_outcome'] == 'stored'

    def test_the_soak_judge_band_scope_judges_whatever_the_deployed_route(self) -> None:
        client = _client(_amends)
        row = asyncio.run(_judge(_case(route='force_stored'), _arm('psi-repro'), client))
        assert row['calls'] == 1
        assert row['judge_outcome'] == 'amended'


class TestRunArms:
    def test_writes_one_row_per_case_and_is_complete(self, tmp_path: Path) -> None:
        cases = [_case('E1'), _case('E2', route='force_stored')]
        result = _run(cases, [_arm()], tmp_path, _client(_amends))
        assert result.complete == ('f3p',)
        assert result.partial == ()
        assert result.exit_code == 0
        assert [row['entry_id'] for row in _rows(tmp_path, 'f3p')] == ['E1', 'E2']

    def test_the_arms_prompt_is_in_force_only_while_it_runs(self, tmp_path: Path) -> None:
        client = _client(_amends)
        _run([_case()], [_arm('psi-repro')], tmp_path, client)
        assert client.responses.create.call_args.kwargs['instructions'] == (
            _wording().PSI_JUDGE_SYSTEM_PROMPT
        )
        assert write_triage_judge.JUDGE_SYSTEM_PROMPT is _SHIPPED

    def test_a_soak_judge_band_arm_covers_only_those_cases(self, tmp_path: Path) -> None:
        cases = [_case('E1'), _case('E2', soak_band='restated')]
        result = _run(cases, [_arm('psi-repro')], tmp_path, _client(_amends))
        assert result.complete == ('psi-repro',)
        assert [row['entry_id'] for row in _rows(tmp_path, 'psi-repro')] == ['E1']

    def test_a_rerun_skips_the_rows_already_written(self, tmp_path: Path) -> None:
        cases = [_case('E1'), _case('E2')]
        _run(cases[:1], [_arm()], tmp_path, _client(_amends))
        client = _client(_amends)
        _run(cases, [_arm()], tmp_path, client)
        assert client.responses.create.await_count == 1
        assert [row['entry_id'] for row in _rows(tmp_path, 'f3p')] == ['E1', 'E2']

    @pytest.mark.parametrize('key', ['cases_sha256', 'prompt_sha256'])
    def test_a_foreign_row_is_refused_naming_both_digests(self, tmp_path: Path, key: str) -> None:
        _run([_case('E1')], [_arm()], tmp_path, _client(_amends))
        path = tmp_path / 'f3p.jsonl'
        row = json.loads(path.read_text())
        row[key] = 'f' * 64
        path.write_text(json.dumps(row) + '\n')
        with pytest.raises(ValueError, match='f' * 64) as refusal:
            _run([_case('E1'), _case('E2')], [_arm()], tmp_path, _client(_amends))
        expected = _CASES_SHA if key == 'cases_sha256' else _sha(_SHIPPED)
        assert expected in str(refusal.value)

    def test_the_spend_cap_stops_dispatch_and_names_the_partial_arms(self, tmp_path: Path) -> None:
        cases = [_case(f'E{index}') for index in range(4)]
        client = _client(lambda kwargs: _reply('amends', _target(kwargs), tokens=1_000_000))
        result = _run(cases, [_arm(), _arm('f3p-repeat')], tmp_path, client, budget_usd=4.5)
        assert client.responses.create.await_count == 2
        assert result.partial == ('f3p', 'f3p-repeat')
        assert result.exit_code != 0
        assert 'f3p-repeat' in result.message
        assert result.spent_usd >= 4.5

    def test_the_cap_counts_the_spend_already_recorded(self, tmp_path: Path) -> None:
        client = _client(lambda _: _reply('amends', 'T-E0', tokens=1_000_000))
        _run([_case('E0')], [_arm()], tmp_path, client)
        again = _client(_amends)
        result = _run([_case('E0'), _case('E1')], [_arm()], tmp_path, again, budget_usd=1.0)
        assert again.responses.create.await_count == 0
        assert result.partial == ('f3p',)


def _second_pass_arm() -> Any:
    return _mod().Arm(
        name='dispute-test', kind='second_pass', prompt='DISPUTE CHECK', model='gpt-6.1-sol',
        effort='low', parent='f3p', scope='all', description='a synthetic second pass',
    )


class TestSecondPass:
    def test_calls_only_on_the_parents_attached_amendments_and_restatements(
        self, tmp_path: Path,
    ) -> None:
        first = {'E1': 'amends', 'E2': 'restates', 'E3': 'contests', 'E4': 'distinct'}

        def answer(kwargs: dict[str, Any]) -> Any:
            target = _target(kwargs)
            entry = target.removeprefix('T-')
            if kwargs['instructions'] == 'DISPUTE CHECK':
                return _reply('contests' if entry == 'E1' else 'amends', target)
            verdict = first.get(entry, 'amends')
            return _reply(verdict, None if verdict == 'distinct' else target)

        cases = [_case('E1'), _case('E2'), _case('E3'), _case('E4'), _case('E5', cos=0.5),
                 _case('E6', route='near_verbatim')]
        client = _client(answer)
        result = _run(cases, [_arm(), _second_pass_arm()], tmp_path, client,
                      config=_config(0.60))
        assert result.complete == ('f3p', 'dispute-test')
        dispute = {row['entry_id']: row for row in _rows(tmp_path, 'dispute-test')}
        assert {entry for entry, row in dispute.items() if row['calls']} == {'E1', 'E2'}
        assert dispute['E1']['dispute_marked'] is True
        assert dispute['E2']['dispute_marked'] is False
        assert dispute['E5']['dispute_marked'] is False
        parent = {row['entry_id']: row for row in _rows(tmp_path, 'f3p')}
        for entry, row in dispute.items():
            assert row['judge_outcome'] == parent[entry]['judge_outcome']
            assert row['routed_outcome'] == parent[entry]['routed_outcome']
        assert dispute['E1']['prompt_sha256'] == _sha('DISPUTE CHECK')

    def test_waits_for_a_complete_parent(self, tmp_path: Path) -> None:
        cases = [_case('E1'), _case('E2')]
        client = _client(lambda _: _reply('amends', 'T-E1', tokens=1_000_000))
        result = _run(cases, [_arm(), _second_pass_arm()], tmp_path, client, budget_usd=1.0)
        assert result.partial == ('f3p', 'dispute-test')
        assert not (tmp_path / 'dispute-test.jsonl').exists() or not _rows(
            tmp_path, 'dispute-test',
        )
