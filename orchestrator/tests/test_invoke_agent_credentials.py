"""A dispatched Claude child inherits exactly one credential: the token selected for it.

The orchestrator holds the whole OAuth pool and every API key in its own
os.environ (it loads the repo .env in-process), and must keep them there for
usage-gate refreshes. The child must not see them. The observation is the env
the real child process saw: a fake `claude` named by CLAUDE_BINARY dumps
`env -0`, driven through the public ``invoke_agent``.
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
