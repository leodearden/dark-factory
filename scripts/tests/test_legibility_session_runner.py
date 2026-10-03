"""The legibility session runner: every trickle/census call goes through the
orchestrator's runner and account pool (task 6042, Leo's 2026-09-29 ruling).

Driven end to end over ``fake_claude_cli`` (a JSON-mode fake first on PATH,
scripted per leased token) and ``pool_roster`` (a hermetic roster), with a REAL
``UsageGate`` from ``account_pool.build_pool``. Nothing here can reach the real
CLI or the operator's own login.

ACCEPTANCE (a), pinned below: a call runs under the runner's own
``CLAUDE_CONFIG_DIR`` carrying the LEASED token, while ``HOME`` and the ambient
``CLAUDE_CONFIG_DIR`` point at a sentinel login that must never be read, and an
empty roster never falls back to that login.
"""
from __future__ import annotations

import dataclasses
import json
import os
import shutil
from pathlib import Path

import pytest
from legibility import account_pool, session_runner
from legibility.session_runner import (
    CLASSIFIER,
    READ_ONLY_EXPLORER,
    SessionRunner,
    StageSpec,
)
from shared.cap_markers import REAL_CLI_CAP_HIT_MESSAGES, REAL_CLI_NEAR_CAP_MESSAGES

pytestmark = pytest.mark.timeout(60)


_CLASSIFIER_STAGE = StageSpec(
    name='s', cwd=None, timeout_secs=30, max_turns=2, max_budget_usd=1.0,
    tools=CLASSIFIER,
)


def _flag_values(argv, flag):
    """The values following *flag* in *argv*, up to the next ``--`` flag."""
    start = argv.index(flag) + 1
    values = []
    for arg in argv[start:]:
        if arg.startswith('--'):
            break
        values.append(arg)
    return values


def test_a_call_runs_as_the_leased_account_never_the_operator_login(
    fake_claude_cli, pool_roster, sentinel_login,
):
    accounts_file, env_file = pool_roster('a', 'b')
    fake_claude_cli.plan(default={'result': 'the verdict text'})

    gate = account_pool.build_pool(accounts_file=accounts_file, env_file=env_file)
    with SessionRunner(gate, label='t') as runner:
        reply = runner.invoker(_CLASSIFIER_STAGE)('the prompt', 'haiku')

    assert reply == 'the verdict text'
    [call] = fake_claude_cli.calls()
    config_dir = Path(call['env']['CLAUDE_CONFIG_DIR'])
    assert config_dir != sentinel_login / '.claude'
    assert not config_dir.is_relative_to(sentinel_login)
    assert call['credentials'] == {'claudeAiOauth': {'accessToken': pool_roster.token('a')}}
    assert call['env']['CLAUDE_CODE_OAUTH_TOKEN'] == pool_roster.token('a')
    assert call['env']['ANTHROPIC_API_KEY_present'] is False

    argv = call['argv']
    assert _flag_values(argv, '--output-format') == ['json']
    assert _flag_values(argv, '--model') == ['haiku']
    assert _flag_values(argv, '--max-turns') == ['2']
    assert '--max-budget-usd' in argv
    assert '--system-prompt-file' in argv and call['system_prompt']
    assert _flag_values(argv, '--disallowed-tools') == ['*']
    assert '--mcp-config' in argv and '--strict-mcp-config' in argv
    assert call['stdin'] == 'the prompt'
    assert os.listdir(call['cwd']) == [], 'a classifier runs in an empty neutral cwd'


def test_a_read_only_explorer_runs_in_its_project_dir_and_cannot_write(
    fake_claude_cli, pool_roster, sentinel_login, tmp_path,
):
    accounts_file, env_file = pool_roster('a')
    project = tmp_path / 'project'
    project.mkdir()
    stage = dataclasses.replace(
        _CLASSIFIER_STAGE, name='verify', cwd=project, tools=READ_ONLY_EXPLORER,
    )

    gate = account_pool.build_pool(accounts_file=accounts_file, env_file=env_file)
    with SessionRunner(gate, label='t') as runner:
        runner.invoker(stage)('explore this', 'sonnet')

    [call] = fake_claude_cli.calls()
    assert Path(call['cwd']) == project
    assert _flag_values(call['argv'], '--allowed-tools') == ['Read', 'Grep', 'Glob']
    assert _flag_values(call['argv'], '--permission-mode') == ['dontAsk']


def test_an_unresolvable_roster_raises_no_headroom_and_never_rides_the_operator_login(
    fake_claude_cli, pool_roster, sentinel_login,
):
    accounts_file, env_file = pool_roster('a', 'b', resolve_tokens=False)

    gate = account_pool.build_pool(accounts_file=accounts_file, env_file=env_file)
    with SessionRunner(gate, label='t') as runner:
        invoke = runner.invoker(_CLASSIFIER_STAGE)
        with pytest.raises(session_runner.NoHeadroom) as excinfo:
            invoke('the prompt', 'haiku')

    assert 'no pool accounts resolved' in str(excinfo.value)
    assert fake_claude_cli.calls() == []


# ---------------------------------------------------------------------------
# ACCEPTANCE (c): auth rejections fail over, exhaustion is typed, nothing parks.
#
# Response shapes measured on the real CLI in JSON mode (CLI 2.1.287 for the
# 401, 2.1.168 for the org-disabled 403, task 5947): rc 1, is_error true,
# subtype "success", and the HTTP status in api_error_status.
# ---------------------------------------------------------------------------

_AUTH_401 = {
    'is_error': True, 'rc': 1, 'api_error_status': 401,
    'result': 'Failed to authenticate. API Error: 401 OAuth access token is invalid.',
}
_AUTH_403 = {
    'is_error': True, 'rc': 1, 'api_error_status': 403,
    'result': (
        'Your organization has disabled Claude subscription access for Claude '
        'Code · Use an Anthropic API key instead, or ask your admin to enable access'
    ),
}
_CAPPED = {'is_error': True, 'rc': 1, 'result': REAL_CLI_CAP_HIT_MESSAGES[2]}
"""Strictly a cap, and its "resets in 3 hours" parses to a FUTURE reset, so the
gate's own reset sweep cannot reopen the account mid-test."""
_NEAR_CAP = {'is_error': True, 'rc': 1, 'result': REAL_CLI_NEAR_CAP_MESSAGES[0]}
_EXIT_ZERO_BANNER = {'result': REAL_CLI_CAP_HIT_MESSAGES[2]}
"""The CLI declining to answer WITHOUT failing: the banner as an rc-0,
``is_error`` false reply (task 5637's route)."""
_PLAIN_VERDICT = '{"matches": [], "candidates": []}'


def _is_json_object(reply):
    try:
        return isinstance(json.loads(reply), dict)
    except ValueError:
        return False


_JSON_STAGE = dataclasses.replace(_CLASSIFIER_STAGE, is_usable_reply=_is_json_object)

_P, _Q = 'max-p', 'max-q'


def _run(fake_claude_cli, pool_roster, by_token, *names, stage=_CLASSIFIER_STAGE, calls=1):
    """Script the fake per account, run *calls* invocations on one runner and
    return ``(outcomes, gate)`` — each outcome a reply or the raised exception."""
    accounts_file, env_file = pool_roster(*names)
    fake_claude_cli.plan(
        {pool_roster.token(name): response for name, response in by_token.items()},
        default={'result': _PLAIN_VERDICT},
    )
    gate = account_pool.build_pool(accounts_file=accounts_file, env_file=env_file)
    outcomes = []
    with SessionRunner(gate, label='t') as runner:
        invoke = runner.invoker(stage)
        for i in range(calls):
            try:
                outcomes.append(invoke(f'prompt {i}', 'haiku'))
            except session_runner.InvocationFailed as exc:
                outcomes.append(exc)
    return outcomes, gate


def _tokens_called(fake_claude_cli):
    return [call['env']['CLAUDE_CODE_OAUTH_TOKEN'] for call in fake_claude_cli.calls()]


@pytest.mark.parametrize('rejection', [_AUTH_401, _AUTH_403], ids=['401', '403'])
def test_an_auth_rejected_account_fails_over_and_the_call_completes_next_door(
    fake_claude_cli, pool_roster, sentinel_login, rejection,
):
    [reply], gate = _run(fake_claude_cli, pool_roster, {_P: rejection}, _P, _Q)

    assert reply == _PLAIN_VERDICT
    assert gate.auth_failed_account_names == (_P,)
    assert _tokens_called(fake_claude_cli) == [pool_roster.token(_P), pool_roster.token(_Q)]


def test_a_pool_whose_every_account_rejects_its_credentials_fails_loud_not_deferred(
    fake_claude_cli, pool_roster, sentinel_login,
):
    outcomes, gate = _run(
        fake_claude_cli, pool_roster, {_P: _AUTH_401, _Q: _AUTH_403}, _P, _Q, calls=2,
    )

    for exc in outcomes:
        assert isinstance(exc, session_runner.InvocationFailed), exc
        assert not isinstance(exc, session_runner.NoHeadroom), exc
        message = str(exc)
        assert _P in message and _Q in message, message
        assert 'credentials rejected' in message, message
        assert 'will not clear at the weekly reset' in message, message
        assert 'capped' not in message.lower(), message
    assert len(fake_claude_cli.calls()) == 2, (
        'every account is already AUTH_FAILED, so the next call must not spend '
        'a single CLI call finding that out again'
    )


def test_a_pool_whose_every_account_is_capped_defers(
    fake_claude_cli, pool_roster, sentinel_login,
):
    [exc], _gate = _run(fake_claude_cli, pool_roster, {_P: _CAPPED, _Q: _CAPPED}, _P, _Q)

    assert isinstance(exc, session_runner.NoHeadroom), exc
    assert 'all 2 pool accounts capped' in str(exc)


def test_a_partly_auth_failed_pool_defers_without_claiming_every_account_is_capped(
    fake_claude_cli, pool_roster, sentinel_login,
):
    [exc], _gate = _run(fake_claude_cli, pool_roster, {_P: _AUTH_401, _Q: _CAPPED}, _P, _Q)

    assert isinstance(exc, session_runner.NoHeadroom), exc
    message = str(exc)
    assert _P in message, message
    assert 'all 2 pool accounts capped' not in message, message
    assert 'will not clear' in message, (
        f'the auth-failed account must not be promised back at the reset; got {message!r}'
    )


def test_a_near_cap_pool_never_claims_a_cap_that_did_not_happen(
    fake_claude_cli, pool_roster, sentinel_login,
):
    """A near-cap verdict takes no phase transition, so the bounded retry runs
    out while the gate still considers an account live. That is not a cap and
    will not clear at a reset, and the reason must not say otherwise."""
    [exc], gate = _run(
        fake_claude_cli, pool_roster, {_P: _NEAR_CAP, _Q: _NEAR_CAP}, _P, _Q,
    )

    assert isinstance(exc, session_runner.NoHeadroom), exc
    message = str(exc)
    assert gate.active_account_name is not None
    assert gate.active_account_name in message, message
    assert 'capped' not in message.lower(), message


def test_an_ordinary_failure_names_the_label_account_subtype_and_both_streams(
    fake_claude_cli, pool_roster, sentinel_login,
):
    failure = {
        'is_error': True, 'rc': 1, 'subtype': 'error_max_turns',
        'total_cost_usd': 0.12, 'num_turns': 3,
        'result': 'STDOUT-TAIL-MARKER the model ran out of turns',
        'stderr': 'STDERR-TAIL-MARKER warning from the CLI',
    }
    [exc], _gate = _run(fake_claude_cli, pool_roster, {_P: failure}, _P, _Q)

    assert isinstance(exc, session_runner.InvocationFailed), exc
    assert not isinstance(exc, session_runner.NoHeadroom), exc
    message = str(exc)
    assert 't[s]' in message, message
    assert _P in message, message
    assert 'error_max_turns' in message, message
    assert 'stdout=' in message and 'STDOUT-TAIL-MARKER' in message, message
    assert 'stderr=' in message and 'STDERR-TAIL-MARKER' in message, message
    assert len(fake_claude_cli.calls()) == 1, 'an ordinary failure does not fail over'


@pytest.mark.parametrize(
    'stage', [_CLASSIFIER_STAGE, _JSON_STAGE], ids=['every-reply-usable', 'json-reply-usable'],
)
def test_a_successful_verdict_quoting_a_cap_banner_is_a_verdict(
    fake_claude_cli, pool_roster, sentinel_login, stage,
):
    """The task-5691 shape: this codebook is full of usage-limit clusters, so a
    verdict QUOTING a banner must come back as the verdict it is."""
    verdict = json.dumps({
        'matches': [{
            'cluster_id': 'usage-limit-stall',
            'evidence_quote': REAL_CLI_CAP_HIT_MESSAGES[0],
        }],
        'candidates': [],
    })
    [reply], gate = _run(
        fake_claude_cli, pool_roster, {_P: {'result': verdict}}, _P, _Q, stage=stage,
    )

    assert reply == verdict
    assert len(fake_claude_cli.calls()) == 1
    assert gate.active_account_name == _P
    assert gate.soonest_resets_at is None


def test_an_exit_zero_banner_the_stage_cannot_use_rotates_and_completes_next_door(
    fake_claude_cli, pool_roster, sentinel_login,
):
    """Task 5637's route: a banner that arrives as an ordinary reply is offered
    to the gate because the stage cannot use it, so the account is capped and
    the SAME call completes on the next one instead of the next call landing
    on a still-AVAILABLE capped account."""
    [reply], gate = _run(
        fake_claude_cli, pool_roster, {_P: _EXIT_ZERO_BANNER}, _P, _Q, stage=_JSON_STAGE,
    )

    assert reply == _PLAIN_VERDICT
    assert _tokens_called(fake_claude_cli) == [pool_roster.token(_P), pool_roster.token(_Q)]
    assert gate.active_account_name == _Q


def test_a_missing_claude_binary_is_a_loud_failure_naming_the_cwd(
    pool_roster, sentinel_login, tmp_path, monkeypatch,
):
    empty_bin = tmp_path / 'empty-bin'
    empty_bin.mkdir()
    monkeypatch.setenv('PATH', f'{empty_bin}{os.pathsep}/usr/bin')
    assert shutil.which('claude') is None, 'the premise: no claude is reachable'
    project = tmp_path / 'project'
    project.mkdir()
    accounts_file, env_file = pool_roster(_P)

    gate = account_pool.build_pool(accounts_file=accounts_file, env_file=env_file)
    with SessionRunner(gate, label='t') as runner:
        invoke = runner.invoker(dataclasses.replace(_CLASSIFIER_STAGE, cwd=project))
        with pytest.raises(session_runner.InvocationFailed) as excinfo:
            invoke('the prompt', 'haiku')

    assert not isinstance(excinfo.value, session_runner.NoHeadroom)
    assert 'could not be started' in str(excinfo.value)
    assert str(project) in str(excinfo.value)


def test_a_call_that_outlives_its_stage_timeout_fails_naming_the_timeout(
    fake_claude_cli, pool_roster, sentinel_login,
):
    """6s, not less: a failure under 5s with no cost reads to the shared
    runner's heuristic net as an unrecognised cap."""
    stage = dataclasses.replace(_CLASSIFIER_STAGE, timeout_secs=6)
    [exc], _gate = _run(
        fake_claude_cli, pool_roster, {_P: {'sleep_secs': 30}}, _P, stage=stage,
    )

    assert isinstance(exc, session_runner.InvocationFailed), exc
    assert not isinstance(exc, session_runner.NoHeadroom), exc
    assert 'timed out' in str(exc), exc
