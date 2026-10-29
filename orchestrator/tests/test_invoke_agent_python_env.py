"""A dispatched agent child inherits none of the orchestrator's Python-env selection.

Production shape, measured 2026-10-28 in a dispatched session: VIRTUAL_ENV is
main's ``.venv``, PATH is fronted by its ``bin``, and UV_RUN_RECURSION_DEPTH=1.
The observation is the env the real child process saw: a fake CLI dumps
`env -0`, driven through the public ``invoke_agent``.
"""

from __future__ import annotations

import json
import os
import stat
from collections.abc import Callable, Iterator
from contextlib import AbstractContextManager, contextmanager, nullcontext
from dataclasses import dataclass
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

_SELECTION_NAMES = ('VIRTUAL_ENV', 'UV_PROJECT_ENVIRONMENT', 'UV_RUN_RECURSION_DEPTH')
_PATH_NAMED_BACKENDS = ('codex', 'gemini', 'pi')


@dataclass(frozen=True)
class AgentChild:
    expected_path: list[str]
    read_env: Callable[[], dict[str, str]]


def _write_env_dumping_fake(script: Path, stdout_line: str) -> None:
    script.write_text(
        '#!/bin/sh\n'
        'cat >/dev/null\n'
        'env -0 > "$FAKE_AGENT_ENV_DUMP"\n'
        f"printf '%s\\n' '{stdout_line}'\n"
    )
    script.chmod(script.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)


@pytest.fixture
def agent_child(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> AgentChild:
    """Give this process main's venv selection and install env-dumping fake CLIs."""
    orch_venv = tmp_path / 'orch-venv'
    (orch_venv / 'bin').mkdir(parents=True)
    tmp_bin = tmp_path / 'bin'
    tmp_bin.mkdir()
    dump = tmp_path / 'env-dump'
    original_path = os.environ['PATH'].split(os.pathsep)

    fake_claude = tmp_bin / 'fake-claude'
    for script in (fake_claude, *(tmp_bin / name for name in _PATH_NAMED_BACKENDS)):
        _write_env_dumping_fake(script, _SUCCESS_JSON)

    monkeypatch.setenv('VIRTUAL_ENV', str(orch_venv))
    monkeypatch.setenv('UV_PROJECT_ENVIRONMENT', str(orch_venv))
    monkeypatch.setenv('UV_RUN_RECURSION_DEPTH', '1')
    monkeypatch.setenv('PATH', os.pathsep.join([str(orch_venv / 'bin'), str(tmp_bin), *original_path]))
    monkeypatch.setenv('CLAUDE_BINARY', str(fake_claude))
    monkeypatch.setenv('FAKE_AGENT_ENV_DUMP', str(dump))
    for var in ('DF_AGENT_CPU_NICE', 'DF_AGENT_CPU_GOVERN'):
        monkeypatch.delenv(var, raising=False)

    assert resolve_claude_binary() == str(fake_claude), 'the fake must be the CLI that runs'

    def read() -> dict[str, str]:
        entries = dump.read_bytes().split(b'\0')
        return dict(e.decode().split('=', 1) for e in entries if e)

    return AgentChild(expected_path=[str(tmp_bin), *original_path], read_env=read)


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


@dataclass(frozen=True)
class SpawnPath:
    backend: str
    model: str
    fake_reports_success: bool
    sandbox_modules: list[str] | None = None
    context: Callable[[], AbstractContextManager[None]] = nullcontext


_SPAWN_PATHS = [
    pytest.param(SpawnPath('claude', 'opus', True), id='claude-unsandboxed'),
    pytest.param(
        SpawnPath('claude', 'opus', True, ['shared'], _bwrap_identity_sandbox),
        id='claude-sandboxed',
    ),
    *(pytest.param(SpawnPath(name, 'm', False), id=name) for name in _PATH_NAMED_BACKENDS),
]


@pytest.mark.asyncio
@pytest.mark.parametrize('spawn_path', _SPAWN_PATHS)
async def test_child_inherits_no_python_env_selection(agent_child, tmp_path, spawn_path):
    workdir = tmp_path / 'work'
    workdir.mkdir()

    with spawn_path.context():
        result = await invoke_agent(
            prompt='hi',
            system_prompt='s',
            cwd=workdir,
            backend=spawn_path.backend,
            model=spawn_path.model,
            max_turns=1,
            max_budget_usd=0.01,
            sandbox_modules=spawn_path.sandbox_modules,
            timeout_seconds=30,
        )
    if spawn_path.fake_reports_success:
        assert result.success, f'fake {spawn_path.backend} dispatch did not succeed: {result!r}'
    env = agent_child.read_env()

    for name in _SELECTION_NAMES:
        assert name not in env
    assert env['PATH'].split(os.pathsep) == agent_child.expected_path
    assert 'VIRTUAL_ENV' in os.environ, 'the orchestrator keeps its own venv'
