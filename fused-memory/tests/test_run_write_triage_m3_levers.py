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
        near_verbatim=route == 'near_verbatim', cos_parent=cos, source_cosine=cos,
        metadata_recovered=True,
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


class TestParentDigest:
    def test_covers_what_a_child_reads_and_not_row_order(self) -> None:
        rows = [_parent_row('E1', 'amended', 'amended'), _parent_row('E2', 'contested', 'stored')]
        digest = _mod().parent_digest(rows)
        assert _mod().parent_digest(rows[::-1]) == digest
        for field, value in (('judge_outcome', 'restated'), ('routed_outcome', 'stored'),
                             ('candidate_id', 'T-E9'), ('dispute_marked', True)):
            changed = [{**rows[0], field: value}, rows[1]]
            assert _mod().parent_digest(changed) != digest, field

    def test_ignores_what_a_child_does_not_read(self) -> None:
        rows = [_parent_row('E1', 'amended', 'amended')]
        timed = [{**rows[0], 'seconds': 9.9, 'usd': 1.0}]
        assert _mod().parent_digest(timed) == _mod().parent_digest(rows)


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

    def test_an_unknown_kind_or_scope_is_refused_at_construction(self) -> None:
        with pytest.raises(ValueError, match='bogus'):
            _mod().Arm('x', 'bogus', 'P', 'gpt-6.1-sol', 'low', None, 'all', 'x')
        with pytest.raises(ValueError, match='everything'):
            _mod().Arm('x', 'judge', 'P', 'gpt-6.1-sol', 'low', None, 'everything', 'x')

    def test_a_kind_and_scope_given_as_text_are_held_as_their_enums(self) -> None:
        arm = _mod().Arm('x', 'second_pass', 'P', 'gpt-6.1-sol', 'low', 'f3p', 'soak-judge-band',
                         'x')
        assert arm.kind is _mod().ArmKind.SECOND_PASS
        assert arm.scope is _mod().ArmScope.SOAK_JUDGE_BAND

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

    def test_a_judged_restatement_is_routed_as_deployed_triage_files_it(self) -> None:
        client = _client(lambda kwargs: _reply('restates', _target(kwargs)))
        row = asyncio.run(_judge(_case(), _arm(), client))
        assert (row['judge_outcome'], row['routed_outcome']) == ('restated', 'amended')

    def test_a_judge_row_names_no_parent(self) -> None:
        row = asyncio.run(_judge(_case(), _arm(), _client(_amends)))
        assert (row['parent'], row['parent_sha256']) == (None, None)

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
        assert dispute['E2']['routed_outcome'] == 'amended', 'a refiled restatement is re-asked'
        assert {row['parent'] for row in dispute.values()} == {'f3p'}
        assert {row['parent_sha256'] for row in dispute.values()} == {
            _mod().parent_digest(_rows(tmp_path, 'f3p')),
        }
        assert dispute['E1']['dispute_marked'] is True
        assert dispute['E2']['dispute_marked'] is False
        assert dispute['E5']['dispute_marked'] is False
        parent = {row['entry_id']: row for row in _rows(tmp_path, 'f3p')}
        for entry, row in dispute.items():
            assert row['judge_outcome'] == parent[entry]['judge_outcome']
            assert row['routed_outcome'] == parent[entry]['routed_outcome']
        assert dispute['E1']['prompt_sha256'] == _sha('DISPUTE CHECK')

    def test_a_parent_rerun_under_new_verdicts_is_refused(self, tmp_path: Path) -> None:
        cases = [_case('E1'), _case('E2')]
        _run(cases, [_arm(), _second_pass_arm()], tmp_path, _client(_amends))
        (tmp_path / 'f3p.jsonl').unlink()
        contests = _client(lambda kwargs: _reply('contests', _target(kwargs)))
        with pytest.raises(ValueError, match='parent_sha256'):
            _run(cases, [_arm(), _second_pass_arm()], tmp_path, contests)

    def test_waits_for_a_complete_parent(self, tmp_path: Path) -> None:
        cases = [_case('E1'), _case('E2')]
        client = _client(lambda _: _reply('amends', 'T-E1', tokens=1_000_000))
        result = _run(cases, [_arm(), _second_pass_arm()], tmp_path, client, budget_usd=1.0)
        assert result.partial == ('f3p', 'dispute-test')
        assert not (tmp_path / 'dispute-test.jsonl').exists() or not _rows(
            tmp_path, 'dispute-test',
        )


_LEVERS = ('a1-partial-dispute', 'c1-dispute-marker', 'd1-stale-detail-check')
_GREENHOUSE = write_triage_judge.JUDGE_EXEMPLARS[0].candidate


class TestTheLeverArms:
    def test_the_lever_arms_and_their_kinds(self) -> None:
        a1, c1, d1 = map(_arm, _LEVERS)
        assert (a1.kind, a1.parent, a1.scope) == ('judge', None, 'all')
        assert (c1.kind, c1.parent, c1.scope) == ('second_pass', 'f3p', 'all')
        assert (d1.kind, d1.parent, d1.scope) == ('judge', None, 'all')
        for arm in (a1, c1, d1):
            assert (arm.model, arm.effort) == ('gpt-6.1-sol', 'low')
            assert arm.prompt != _SHIPPED

    @pytest.mark.parametrize('name', _LEVERS)
    def test_each_lever_prompt_keeps_the_shipped_reply_contract(self, name: str) -> None:
        prompt = _arm(name).prompt
        for word in write_triage_judge.JUDGE_VERDICTS:
            assert f'"{word}"' in prompt, word
        assert write_triage_judge.JUDGE_REPLY_SHAPE in prompt
        exemplars = _mod().LEVER_EXEMPLARS[name]
        assert write_triage_judge.render_judge_exemplars(exemplars) in prompt
        for exemplar in exemplars:
            assert isinstance(exemplar, write_triage_judge.JudgeExemplar)
            assert exemplar.verdict in write_triage_judge.JUDGE_VERDICTS

    @pytest.mark.parametrize('name', ['a1-partial-dispute', 'd1-stale-detail-check'])
    def test_a_first_pass_lever_adds_one_contests_exemplar_on_the_greenhouse_candidate(
        self, name: str,
    ) -> None:
        exemplars = _mod().LEVER_EXEMPLARS[name]
        shipped = write_triage_judge.JUDGE_EXEMPLARS
        assert exemplars[:len(shipped)] == shipped
        (added,) = exemplars[len(shipped):]
        assert (added.candidate, added.verdict) == (_GREENHOUSE, 'contests')
        assert added not in shipped

    def test_the_first_pass_levers_add_different_exemplars(self) -> None:
        a1, d1 = (_mod().LEVER_EXEMPLARS[name][-1] for name in _LEVERS[::2])
        assert a1.entry != d1.entry

    def test_the_dispute_marker_answers_contests_or_amends(self) -> None:
        verdicts = {exemplar.verdict for exemplar in _mod().LEVER_EXEMPLARS['c1-dispute-marker']}
        assert verdicts == {'amends', 'contests'}

    def test_the_shipped_parser_reads_a_lever_answer(self) -> None:
        client = _client(lambda kwargs: _reply('contests', _target(kwargs)))
        row = asyncio.run(_judge(_case(), _arm('a1-partial-dispute'), client))
        assert client.responses.create.call_args.kwargs['instructions'] == (
            _arm('a1-partial-dispute').prompt
        )
        assert (row['judge_outcome'], row['routed_outcome']) == ('contested', 'contested')
        assert row['prompt_sha256'] == _sha(_arm('a1-partial-dispute').prompt)


def _sentence_of(prompt: str, tokens: int) -> str:
    """The last run of *tokens* word tokens in *prompt*'s instructions, before its examples.

    Joined by spaces, they tokenise back to exactly those tokens.
    """
    found = re.findall(r'\w+|[^\w\s]', prompt.split('Worked examples:')[0].lower())
    for start in range(len(found) - tokens, -1, -1):
        window = found[start:start + tokens]
        if all(re.fullmatch(r'\w+', token) for token in window):
            return ' '.join(window)
    raise AssertionError(f'no run of {tokens} word tokens')


def _shares_a_shingle(span_text: str, message: str) -> bool:
    words = span_text.lower().split()
    return any(' '.join(words[i:i + 8]) in message.lower() for i in range(len(words) - 7))


class TestTheCorpusTextGuard:
    @pytest.mark.parametrize('name', _LEVERS)
    def test_refuses_a_prompt_sharing_an_eight_token_shingle_with_a_case(
        self, tmp_path: Path, name: str,
    ) -> None:
        copied = _sentence_of(_arm(name).prompt, 12)
        cases = [_case('E1'), _case('E2', child=f'An entry. {copied.upper()} More text.')]
        client = _client(_amends)
        with pytest.raises(ValueError, match=name) as refusal:
            _run(cases, [_arm(name)], tmp_path, client)
        assert _shares_a_shingle(copied, str(refusal.value))
        assert client.responses.create.await_count == 0

    def test_a_parent_text_is_checked_too(self, tmp_path: Path) -> None:
        copied = _sentence_of(_arm('d1-stale-detail-check').prompt, 9)
        cases = [_case('E1', parent=copied)]
        with pytest.raises(ValueError, match='d1-stale-detail-check'):
            _run(cases, [_arm('d1-stale-detail-check')], tmp_path, _client(_amends))

    def test_a_seven_token_overlap_is_allowed(self, tmp_path: Path) -> None:
        copied = _sentence_of(_arm('a1-partial-dispute').prompt, 7)
        cases = [_case('E1', child=f'zzz {copied} qqq')]
        result = _run(cases, [_arm('a1-partial-dispute')], tmp_path, _client(_amends))
        assert result.complete == ('a1-partial-dispute',)

    def test_the_shingle_guard_is_public_and_names_the_span(self) -> None:
        copied = _sentence_of(_arm('a1-partial-dispute').prompt, 8)
        with pytest.raises(ValueError, match='a1-partial-dispute'):
            _mod().refuse_corpus_text([_arm('a1-partial-dispute')], [_case(child=copied)])
        _mod().refuse_corpus_text(list(_mod().ARMS), [_case()])


class TestCombinationArms:
    def _combination(self) -> Any:
        c1 = _arm('c1-dispute-marker')
        return _mod().Arm(
            name='c1-over-a1', kind='second_pass', prompt=c1.prompt, model=c1.model,
            effort=c1.effort, parent='a1-partial-dispute', scope='all',
            description='the dispute marker over the boundary lever',
        )

    def test_a_second_pass_over_a_lever_arm_is_accepted(self) -> None:
        _mod().validate_arms([*_mod().ARMS, self._combination()])

    def test_it_runs_over_the_lever_arms_rows(self, tmp_path: Path) -> None:
        marker = _arm('c1-dispute-marker').prompt

        def answer(kwargs: dict[str, Any]) -> Any:
            target = _target(kwargs)
            if kwargs['instructions'] == marker:
                return _reply('contests', target)
            return _reply('amends' if target == 'T-E1' else 'contests', target)

        cases = [_case('E1'), _case('E2')]
        result = _run(cases, [_arm('a1-partial-dispute'), self._combination()], tmp_path,
                      _client(answer))
        assert result.complete == ('a1-partial-dispute', 'c1-over-a1')
        rows = {row['entry_id']: row for row in _rows(tmp_path, 'c1-over-a1')}
        assert (rows['E1']['calls'], rows['E1']['dispute_marked']) == (1, True)
        assert (rows['E2']['calls'], rows['E2']['judge_outcome']) == (0, 'contested')


def _parent_row(entry: str, judge: str | None, routed: str, **extra: Any) -> dict[str, Any]:
    return {'entry_id': entry, 'target_id': f'T-{entry}', 'route': 'judge',
            'judge_outcome': judge, 'candidate_id': None, 'routed_outcome': routed,
            'dispute_marked': None, **extra}


class TestTheTransformArms:
    def test_the_guard_mark_arms_are_transforms_over_f3p_and_d1(self) -> None:
        e1, combined = _arm('e1-mark-refused-contests'), _arm('d1-e1-combined')
        assert (e1.kind, e1.parent, e1.prompt) == ('transform', 'f3p', None)
        assert (combined.kind, combined.parent, combined.prompt) == (
            'transform', 'd1-stale-detail-check', None,
        )
        assert e1.transform is combined.transform is _mod().mark_refused_contest
        alternative = _arm('a1-e1-combined')
        assert (alternative.kind, alternative.parent, alternative.transform) == (
            'transform', 'a1-partial-dispute', _mod().mark_refused_contest,
        )
        _mod().validate_arms(_mod().ARMS)

    @pytest.mark.parametrize(('judge', 'routed', 'marked'), [
        ('contested', 'stored', True),
        ('contested', 'contested', False),
        ('amended', 'stored', False),
        ('amended', 'amended', False),
        ('stored', 'stored', False),
        (None, 'stored', False),
    ])
    def test_marks_only_a_contest_the_attach_guard_stored(
        self, judge: str | None, routed: str, marked: bool,
    ) -> None:
        assert _mod().mark_refused_contest(_parent_row('E1', judge, routed)) is marked

    def test_a_transform_needs_a_function_and_no_prompt(self) -> None:
        arm = _mod().Arm('t', 'transform', None, None, None, 'f3p', 'all', 'x')
        with pytest.raises(ValueError, match="'t'"):
            _mod().validate_arms([_arm('f3p'), arm])
        prompted = _mod().Arm('j', 'judge', 'P', 'gpt-6.1-sol', 'low', None, 'all', 'x',
                              _mod().mark_refused_contest)
        with pytest.raises(ValueError, match="'j'"):
            _mod().validate_arms([prompted])

    def test_runs_over_its_parents_rows_without_a_call(self, tmp_path: Path) -> None:
        def answer(kwargs: dict[str, Any]) -> Any:
            target = _target(kwargs)
            return _reply('contests' if target in ('T-E1', 'T-E2') else 'amends', target)

        cases = [_case('E1', cos=0.5), _case('E2'), _case('E3', cos=0.5),
                 _case('E4', route='force_stored')]
        client = _client(answer)
        result = _run(cases, [_arm('f3p')], tmp_path, client, config=_config(0.60))
        calls_before = client.responses.create.await_count
        result = _run(cases, [_arm('f3p'), _arm('e1-mark-refused-contests')], tmp_path, client,
                      config=_config(0.60))
        assert client.responses.create.await_count == calls_before
        assert result.complete == ('f3p', 'e1-mark-refused-contests')
        parent = {row['entry_id']: row for row in _rows(tmp_path, 'f3p')}
        rows = {row['entry_id']: row for row in _rows(tmp_path, 'e1-mark-refused-contests')}
        assert {entry for entry, row in rows.items() if row['dispute_marked']} == {'E1'}
        digest = _mod().parent_digest(parent.values())
        for entry, row in rows.items():
            assert (row['judge_outcome'], row['routed_outcome']) == (
                parent[entry]['judge_outcome'], parent[entry]['routed_outcome'],
            )
            assert (row['calls'], row['usd'], row['prompt_sha256']) == (0, 0.0, None)
            assert row['arm'] == 'e1-mark-refused-contests'
            assert (row['parent'], row['parent_sha256']) == ('f3p', digest)
        assert set(rows['E1']) == set(parent['E1']), 'a transform row has the judge row schema'

    def test_a_parent_rerun_under_new_verdicts_is_refused(self, tmp_path: Path) -> None:
        cases = [_case('E1'), _case('E2')]
        _run(cases, [_arm('f3p'), _arm('e1-mark-refused-contests')], tmp_path, _client(_amends))
        (tmp_path / 'f3p.jsonl').unlink()
        contests = _client(lambda kwargs: _reply('contests', _target(kwargs)))
        with pytest.raises(ValueError, match='parent_sha256'):
            _run(cases, [_arm('f3p'), _arm('e1-mark-refused-contests')], tmp_path, contests)

    def test_keeps_a_mark_its_parent_already_carries(self) -> None:
        row = _parent_row('E1', 'amended', 'amended', dispute_marked=True)
        assert _mod().transform_row(row, _arm('e1-mark-refused-contests'), cases_sha256=_CASES_SHA,
                                    parent_sha256='d' * 64)['dispute_marked'] is True

    def test_waits_for_a_complete_parent(self, tmp_path: Path) -> None:
        result = _run([_case('E1')], [_arm('e1-mark-refused-contests')], tmp_path,
                      _client(_amends))
        assert result.partial == ('e1-mark-refused-contests',)
