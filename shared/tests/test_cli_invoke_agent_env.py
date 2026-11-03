"""The non-sandbox Claude spawn path hands its child one credential and no Python-env selection.

``shared.cli_invoke.invoke_claude_agent`` is the path fused-memory
reconciliation and the legibility runner use. The spawning process holds the
OAuth pool and API keys; its child must see only ``CLAUDE_CODE_OAUTH_TOKEN``.
The observation is the env the real child saw: a fake `claude` named by
CLAUDE_BINARY dumps `env -0`.
"""

from __future__ import annotations

import json
import os
import stat
from collections.abc import Callable
from pathlib import Path

import pytest

from shared.cli_invoke import invoke_claude_agent, resolve_claude_binary

_SUCCESS_JSON = json.dumps({
    'result': 'ok',
    'is_error': False,
    'subtype': 'success',
    'cost_usd': 0.0,
    'duration_ms': 1,
    'num_turns': 1,
    'session_id': 's',
})

POOL_A = 'pool-a-' + 'A' * 41
POOL_B = 'pool-b-' + 'B' * 41
AMBIENT_SESSION_TOKEN = 'ambient-' + 'm' * 40
SELECTED_TOKEN = 'selected-' + 's' * 39


def _credential_shaped(name: str) -> bool:
    return name.startswith('CLAUDE_OAUTH_TOKEN_') or name.endswith(('_API_KEY', '_TOKEN'))


@pytest.fixture
def child_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Callable[[], dict[str, str]]:
    """Point CLAUDE_BINARY at a fake `claude` that dumps its env; return its reader."""
    bin_dir = tmp_path / 'bin'
    bin_dir.mkdir()
    dump = tmp_path / 'env-dump'
    script = bin_dir / 'fake-claude'
    script.write_text(
        '#!/bin/sh\n'
        'cat >/dev/null\n'
        'env -0 > "$FAKE_CLAUDE_ENV_DUMP"\n'
        f"printf '%s\\n' '{_SUCCESS_JSON}'\n"
    )
    script.chmod(script.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)

    monkeypatch.setenv('CLAUDE_BINARY', str(script))
    monkeypatch.setenv('FAKE_CLAUDE_ENV_DUMP', str(dump))
    for var in ('DF_AGENT_CPU_NICE', 'DF_AGENT_CPU_GOVERN'):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv('CLAUDE_OAUTH_TOKEN_A', POOL_A)
    monkeypatch.setenv('CLAUDE_OAUTH_TOKEN_B', POOL_B)
    monkeypatch.setenv('OPENAI_API_KEY', 'openai-' + 'o' * 41)
    monkeypatch.setenv('ANTHROPIC_API_KEY', 'anthropic-' + 'k' * 38)
    monkeypatch.setenv('CLAUDE_CODE_OAUTH_TOKEN', AMBIENT_SESSION_TOKEN)

    assert resolve_claude_binary() == str(script), 'the fake must be the CLI that runs'

    def read() -> dict[str, str]:
        entries = dump.read_bytes().split(b'\0')
        return dict(e.decode().split('=', 1) for e in entries if e)

    return read


@pytest.mark.parametrize(
    ('oauth_token', 'expected_credential'),
    [
        pytest.param(SELECTED_TOKEN, SELECTED_TOKEN, id='selected-token'),
        pytest.param(None, AMBIENT_SESSION_TOKEN, id='ambient-token'),
    ],
)
async def test_child_inherits_only_its_session_credential(
    child_env, tmp_path, oauth_token, expected_credential,
):
    workdir = tmp_path / 'work'
    workdir.mkdir()

    result = await invoke_claude_agent(
        prompt='hi',
        system_prompt='s',
        cwd=workdir,
        model='opus',
        max_turns=1,
        max_budget_usd=0.01,
        oauth_token=oauth_token,
    )
    assert result.success, f'fake claude dispatch did not succeed: {result!r}'
    env = child_env()

    assert [k for k in env if k.startswith('CLAUDE_OAUTH_TOKEN_')] == []
    assert 'OPENAI_API_KEY' not in env
    assert 'ANTHROPIC_API_KEY' not in env
    assert env['CLAUDE_CODE_OAUTH_TOKEN'] == expected_credential
    assert {k for k in env if _credential_shaped(k)} == {'CLAUDE_CODE_OAUTH_TOKEN'}
    assert os.environ['CLAUDE_OAUTH_TOKEN_A'] == POOL_A, 'the parent keeps its pool'


async def test_child_inherits_no_python_env_selection(child_env, tmp_path, monkeypatch):
    original_path = os.environ['PATH']
    orch_venv = tmp_path / 'orch-venv'
    monkeypatch.setenv('VIRTUAL_ENV', str(orch_venv))
    monkeypatch.setenv('UV_PROJECT_ENVIRONMENT', str(orch_venv))
    monkeypatch.setenv('UV_RUN_RECURSION_DEPTH', '1')
    monkeypatch.setenv('PATH', str(orch_venv / 'bin') + os.pathsep + original_path)
    workdir = tmp_path / 'work'
    workdir.mkdir()

    result = await invoke_claude_agent(
        prompt='hi',
        system_prompt='s',
        cwd=workdir,
        model='opus',
        max_turns=1,
        max_budget_usd=0.01,
    )
    assert result.success, f'fake claude dispatch did not succeed: {result!r}'
    env = child_env()

    for name in ('VIRTUAL_ENV', 'UV_PROJECT_ENVIRONMENT', 'UV_RUN_RECURSION_DEPTH'):
        assert name not in env
    assert env['PATH'].split(os.pathsep) == original_path.split(os.pathsep)
