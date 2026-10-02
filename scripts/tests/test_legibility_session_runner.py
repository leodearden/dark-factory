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

import json
import os
from pathlib import Path

import pytest
from legibility import account_pool, session_runner
from legibility.session_runner import (
    CLASSIFIER,
    READ_ONLY_EXPLORER,
    SessionRunner,
    StageSpec,
)

from shared import usage_gate as usage_gate_mod

_SENTINEL_TOKEN = 'sk-ant-oat01-SENTINEL-operator-login'

pytestmark = pytest.mark.timeout(60)


@pytest.fixture
def sentinel_login(tmp_path, monkeypatch):
    """The operator's interactive login, faked: HOME, the ambient config dir
    and an API key all point somewhere a legibility call must never use."""
    home = tmp_path / 'sentinel-home'
    claude_dir = home / '.claude'
    claude_dir.mkdir(parents=True)
    credentials = claude_dir / '.credentials.json'
    credentials.write_text(json.dumps({'claudeAiOauth': {'accessToken': _SENTINEL_TOKEN}}))
    (home / '.claude.json').write_text('{}')
    monkeypatch.setenv('HOME', str(home))
    monkeypatch.setenv('CLAUDE_CONFIG_DIR', str(claude_dir))
    monkeypatch.setenv('ANTHROPIC_API_KEY', 'sk-ant-api-SENTINEL')
    monkeypatch.setattr(usage_gate_mod, 'CREDENTIALS_PATH', credentials)
    return home


def _classifier_stage(**overrides):
    fields = dict(
        name='s', cwd=None, timeout_secs=30, max_turns=2, max_budget_usd=1.0,
        tools=CLASSIFIER,
    )
    fields.update(overrides)
    return StageSpec(**fields)


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
        reply = runner.invoker(_classifier_stage())('the prompt', 'haiku')

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
    stage = _classifier_stage(name='verify', cwd=project, tools=READ_ONLY_EXPLORER)

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
        invoke = runner.invoker(_classifier_stage())
        with pytest.raises(session_runner.NoHeadroom) as excinfo:
            invoke('the prompt', 'haiku')

    assert 'no pool accounts resolved' in str(excinfo.value)
    assert fake_claude_cli.calls() == []
