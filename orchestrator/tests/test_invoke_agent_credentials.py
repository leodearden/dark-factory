"""A dispatched agent child inherits only the credentials its backend names, never the pool.

The orchestrator holds the whole OAuth pool and every API key in its own
os.environ (it loads the repo .env in-process), and must keep them there for
usage-gate refreshes. A Claude child sees exactly the token selected for it.
A codex, gemini or pi child sees only its own provider keys. No child sees the
``CLAUDE_OAUTH_TOKEN_*`` pool. The observation is the env the real child
process saw: a fake CLI (`claude` named by CLAUDE_BINARY, or `codex`/`gemini`/
`pi` on PATH) dumps `env -0`, driven through the public ``invoke_agent``.
"""

from __future__ import annotations

import json
import os
import stat
from collections.abc import Callable, Iterator
from contextlib import AbstractContextManager, contextmanager, nullcontext
from pathlib import Path
from unittest.mock import patch

import pytest
from shared.cli_invoke import resolve_claude_binary

from orchestrator.agents.invoke import invoke_agent

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


@contextmanager
def _bwrap_identity_sandbox() -> Iterator[None]:
    """Reach the sandboxed branch through public sandbox_dispatch names only."""
    with (
        patch('orchestrator.agents.sandbox_dispatch.resolve_active_backend', return_value='bwrap'),
        patch(
            'orchestrator.agents.sandbox_dispatch.wrap_command',
            side_effect=lambda cmd, *args, **kwargs: cmd,
        ),
    ):
        yield


_DISPATCH_PATHS = pytest.mark.parametrize(
    ('sandbox_modules', 'dispatch_context'),
    [
        pytest.param(None, nullcontext, id='unsandboxed'),
        pytest.param(['shared'], _bwrap_identity_sandbox, id='sandboxed'),
    ],
)


async def _dispatch(
    workdir: Path,
    sandbox_modules: list[str] | None,
    dispatch_context: Callable[[], AbstractContextManager[None]],
    env_overrides: dict[str, str] | None = None,
) -> None:
    workdir.mkdir(exist_ok=True)
    with dispatch_context():
        result = await invoke_agent(
            prompt='hi',
            system_prompt='s',
            cwd=workdir,
            model='opus',
            max_turns=1,
            max_budget_usd=0.01,
            sandbox_modules=sandbox_modules,
            oauth_token=SELECTED_TOKEN,
            env_overrides=env_overrides,
        )
    assert result.success, f'fake claude dispatch did not succeed: {result!r}'


@pytest.mark.asyncio
@_DISPATCH_PATHS
async def test_child_inherits_only_the_selected_credential(
    child_env, tmp_path, sandbox_modules, dispatch_context,
):
    await _dispatch(tmp_path / 'work', sandbox_modules, dispatch_context)
    env = child_env()

    assert [k for k in env if k.startswith('CLAUDE_OAUTH_TOKEN_')] == []
    assert 'OPENAI_API_KEY' not in env
    assert 'ANTHROPIC_API_KEY' not in env
    assert env['CLAUDE_CODE_OAUTH_TOKEN'] == SELECTED_TOKEN
    assert {k for k in env if _credential_shaped(k)} == {'CLAUDE_CODE_OAUTH_TOKEN'}
    assert os.environ['CLAUDE_OAUTH_TOKEN_A'] == POOL_A, 'the parent keeps its pool'


@pytest.mark.asyncio
@_DISPATCH_PATHS
async def test_explicit_env_override_reaches_the_child(
    child_env, tmp_path, sandbox_modules, dispatch_context,
):
    await _dispatch(
        tmp_path / 'work', sandbox_modules, dispatch_context,
        env_overrides={'ANTHROPIC_API_KEY': 'dummy'},
    )

    assert child_env()['ANTHROPIC_API_KEY'] == 'dummy'


GITHUB_TOKEN_VALUE = 'github-' + 'h' * 40

_BACKEND_ENV = {
    'CLAUDE_OAUTH_TOKEN_A': POOL_A,
    'CLAUDE_OAUTH_TOKEN_B': POOL_B,
    'CLAUDE_CODE_OAUTH_TOKEN': AMBIENT_SESSION_TOKEN,
    'OPENAI_API_KEY': 'openai-' + 'o' * 41,
    'ANTHROPIC_API_KEY': 'anthropic-' + 'k' * 38,
    'GEMINI_API_KEY': 'gemini-' + 'g' * 41,
    'GOOGLE_API_KEY': 'google-' + 'q' * 41,
    'ANTHROPIC_OAUTH_TOKEN': 'ambient-oauth-' + 'u' * 34,
    'GITHUB_TOKEN': GITHUB_TOKEN_VALUE,
    'KRAKEN_API_KEY': 'kraken-' + 'r' * 41,
}


@pytest.fixture
def backend_child(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Callable[[], dict[str, str]]:
    """Put env-dumping fakes named codex, gemini and pi first on PATH; return the dump's reader."""
    bin_dir = tmp_path / 'backend-bin'
    bin_dir.mkdir()
    dump = tmp_path / 'backend-env-dump'
    for name in ('codex', 'gemini', 'pi'):
        script = bin_dir / name
        script.write_text(
            '#!/bin/sh\n'
            'cat >/dev/null\n'
            'env -0 > "$FAKE_AGENT_ENV_DUMP"\n'
            "printf '%s\\n' 'not-backend-json'\n"
        )
        script.chmod(script.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)

    monkeypatch.setenv('PATH', os.pathsep.join([str(bin_dir), os.environ['PATH']]))
    monkeypatch.setenv('FAKE_AGENT_ENV_DUMP', str(dump))
    for var in ('DF_AGENT_CPU_NICE', 'DF_AGENT_CPU_GOVERN'):
        monkeypatch.delenv(var, raising=False)
    for name, value in _BACKEND_ENV.items():
        monkeypatch.setenv(name, value)

    def read() -> dict[str, str]:
        entries = dump.read_bytes().split(b'\0')
        return dict(e.decode().split('=', 1) for e in entries if e)

    return read


async def _dispatch_backend(
    workdir: Path,
    backend: str,
    oauth_token: str | None = None,
    env_overrides: dict[str, str] | None = None,
) -> None:
    workdir.mkdir(exist_ok=True)
    await invoke_agent(
        prompt='hi',
        system_prompt='s',
        cwd=workdir,
        backend=backend,
        model='m',
        max_turns=1,
        max_budget_usd=0.01,
        timeout_seconds=30,
        oauth_token=oauth_token,
        env_overrides=env_overrides,
    )


@pytest.mark.spawns_agent_cli
@pytest.mark.asyncio
@pytest.mark.parametrize(
    ('backend', 'oauth_token', 'expected_credentials'),
    [
        pytest.param('codex', None, {'OPENAI_API_KEY'}, id='codex'),
        pytest.param('gemini', None, {'GEMINI_API_KEY', 'GOOGLE_API_KEY'}, id='gemini'),
        pytest.param(
            'pi', None, {'ANTHROPIC_API_KEY', 'OPENAI_API_KEY', 'GEMINI_API_KEY'}, id='pi',
        ),
        pytest.param(
            'pi', SELECTED_TOKEN, {'ANTHROPIC_OAUTH_TOKEN', 'OPENAI_API_KEY', 'GEMINI_API_KEY'},
            id='pi-oauth',
        ),
    ],
)
async def test_backend_child_inherits_only_its_provider_keys(
    backend_child, tmp_path, backend, oauth_token, expected_credentials,
):
    await _dispatch_backend(tmp_path / 'work', backend, oauth_token=oauth_token)
    env = backend_child()

    assert {k for k in env if _credential_shaped(k)} == expected_credentials
    assert [k for k in env if k.startswith('CLAUDE_OAUTH_TOKEN_')] == []
    for name in expected_credentials - {'ANTHROPIC_OAUTH_TOKEN'}:
        assert env[name] == _BACKEND_ENV[name]
    if oauth_token is not None:
        assert env['ANTHROPIC_OAUTH_TOKEN'] == oauth_token
    assert os.environ['CLAUDE_OAUTH_TOKEN_A'] == POOL_A, 'the parent keeps its pool'


@pytest.mark.spawns_agent_cli
@pytest.mark.asyncio
async def test_pi_env_override_reaches_the_child(backend_child, tmp_path):
    zai_key = 'zai-' + 'z' * 44
    await _dispatch_backend(tmp_path / 'work', 'pi', env_overrides={'ZAI_API_KEY': zai_key})

    assert backend_child()['ZAI_API_KEY'] == zai_key
