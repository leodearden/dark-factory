"""invoke_agent hands every spawn to one injected runner (PRD D12, B11 d).

A generic fake CLI, installed under each backend's name on PATH, answers
from FAKE_* env vars. A spy runner awaits the real
``shared.cli_invoke.run_cli_subprocess`` and records what each backend asked
it to run, so these tests pin both the call contract and the result shape
the five non-claude backends see.
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
from _operator_home_fixture import OPERATOR_ENV_VARS, write_operator_home
from _zcode_fixture import BASE_URL as ZCODE_BASE_URL
from shared.cli_invoke import SubprocessResult, run_cli_subprocess

from orchestrator.agents.invoke import invoke_agent
from orchestrator.agents.mimo_backend import SESSION_READ_TIMEOUT_SECS

pytestmark = pytest.mark.asyncio

MIMO_BASE_URL = 'https://token-plan-ams.xiaomimimo.com/anthropic'
FAKE_STDERR = 'fake-stderr'
PLAIN_STDOUT = 'plain-text-output'
MIMO_SESSION_LIST = json.dumps([{'id': 'ses-1'}])
TIMEOUT_SECS = 30.0
NON_CLAUDE_BACKENDS = ('codex', 'gemini', 'pi', 'zcode', 'mimo')
SINGLE_SPAWN_BACKENDS = ('codex', 'gemini', 'pi', 'zcode')

_FAKE_CLI = '''#!{interpreter} -I
import os
import sys
import time

time.sleep(float(os.environ.get('FAKE_SLEEP', '0')))
sys.stdout.write(os.environ.get('FAKE_STDOUT', ''))
sys.stderr.write(os.environ.get('FAKE_STDERR', ''))
sys.exit(int(os.environ.get('FAKE_RC', '0')))
'''

_CODEX_TURN = (
    json.dumps({'type': 'thread.started', 'thread_id': 'tid-1'}) + '\n'
    + json.dumps({'type': 'item.completed', 'item': {'type': 'agent_message', 'text': 'hello'}})
    + '\n'
    + json.dumps({'type': 'turn.completed', 'usage': {'input_tokens': 10, 'output_tokens': 5}})
    + '\n'
)


@dataclass(frozen=True)
class _Spawn:
    cmd: list[str]
    kw: dict[str, Any]
    result: SubprocessResult


class _SpyRunner:
    def __init__(self) -> None:
        self.spawns: list[_Spawn] = []

    async def __call__(
        self, cmd: list[str], cwd: Path, env: dict[str, str], **kw: Any,
    ) -> SubprocessResult:
        result = await run_cli_subprocess(cmd, cwd, env, **kw)
        self.spawns.append(_Spawn(cmd=list(cmd), kw=kw, result=result))
        return result


@pytest.fixture(autouse=True)
def fake_clis(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    bin_dir = tmp_path / 'bin'
    bin_dir.mkdir()
    for name in NON_CLAUDE_BACKENDS:
        executable = bin_dir / name
        executable.write_text(_FAKE_CLI.format(interpreter=sys.executable))
        executable.chmod(0o755)
    monkeypatch.setenv('PATH', f'{bin_dir}{os.pathsep}{os.environ["PATH"]}')
    for var in ('DF_AGENT_CPU_NICE', 'DF_AGENT_CPU_GOVERN'):
        monkeypatch.delenv(var, raising=False)
    operator = write_operator_home(tmp_path / 'operator')
    monkeypatch.setenv('HOME', str(operator))
    monkeypatch.setenv('GIT_CONFIG_NOSYSTEM', '1')
    for name in OPERATOR_ENV_VARS:
        monkeypatch.delenv(name, raising=False)


@pytest.fixture
def worktree(tmp_path: Path) -> Path:
    wt = tmp_path / 'wt'
    wt.mkdir()
    (wt / 'README.md').write_text('fixture\n')
    return wt


def _fake_answers(
    monkeypatch: pytest.MonkeyPatch, *, stdout: str, rc: int, sleep: float = 0.0,
) -> None:
    monkeypatch.setenv('FAKE_STDOUT', stdout)
    monkeypatch.setenv('FAKE_STDERR', FAKE_STDERR)
    monkeypatch.setenv('FAKE_RC', str(rc))
    monkeypatch.setenv('FAKE_SLEEP', str(sleep))


def _endpoint_overrides(backend: str) -> dict[str, str] | None:
    base_url = {'zcode': ZCODE_BASE_URL, 'mimo': MIMO_BASE_URL}.get(backend)
    if base_url is None:
        return None
    return {'ANTHROPIC_BASE_URL': base_url, 'ANTHROPIC_AUTH_TOKEN': 'k-secret'}


async def _invoke(backend: str, worktree: Path, runner: Any, **overrides: Any):
    kwargs: dict[str, Any] = {
        'prompt': 'do step-1',
        'system_prompt': 'SYS',
        'cwd': worktree,
        'model': 'model-x',
        'backend': backend,
        'sandbox_modules': None,
        'env_overrides': _endpoint_overrides(backend),
        'timeout_seconds': TIMEOUT_SECS,
        'runner': runner,
    }
    kwargs.update(overrides)
    return await invoke_agent(**kwargs)


def _canned_stdout(backend: str) -> tuple[str, int]:
    return (MIMO_SESSION_LIST, 0) if backend == 'mimo' else (PLAIN_STDOUT, 3)


class TestNonClaudeBackendsSpawnThroughTheRunner:
    @pytest.mark.timeout(60)
    @pytest.mark.parametrize('backend', NON_CLAUDE_BACKENDS)
    async def test_every_spawn_is_a_flat_runner_call(
        self, backend: str, worktree: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        stdout, rc = _canned_stdout(backend)
        _fake_answers(monkeypatch, stdout=stdout, rc=rc)
        spy = _SpyRunner()

        await _invoke(backend, worktree, spy)

        assert spy.spawns
        for spawn in spy.spawns:
            assert spawn.kw['backend'] == backend
            assert spawn.kw['transcript'] is None
            assert spawn.cmd[0] == backend
            assert spawn.result == SubprocessResult(
                stdout=stdout, stderr=FAKE_STDERR, returncode=rc,
                duration_ms=spawn.result.duration_ms,
            )

    @pytest.mark.timeout(60)
    @pytest.mark.parametrize('backend', SINGLE_SPAWN_BACKENDS)
    async def test_single_spawn_backends_pass_their_stdin_and_timeout(
        self, backend: str, worktree: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        _fake_answers(monkeypatch, stdout=PLAIN_STDOUT, rc=3)
        spy = _SpyRunner()

        await _invoke(backend, worktree, spy)

        expected_stdin = b'SYS\n\ndo step-1' if backend == 'codex' else None
        assert [(s.kw.get('stdin_data'), s.kw['timeout_seconds']) for s in spy.spawns] == [
            (expected_stdin, TIMEOUT_SECS),
        ]

    @pytest.mark.timeout(60)
    async def test_mimo_runs_then_reads_its_session_store(
        self, worktree: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        _fake_answers(monkeypatch, stdout=MIMO_SESSION_LIST, rc=0)
        spy = _SpyRunner()

        await _invoke('mimo', worktree, spy)

        assert [(s.cmd[1], s.kw['stdin_data'], s.kw['timeout_seconds']) for s in spy.spawns] == [
            ('run', b'do step-1', TIMEOUT_SECS),
            ('session', b'', SESSION_READ_TIMEOUT_SECS),
            ('export', b'', SESSION_READ_TIMEOUT_SECS),
        ]


class TestNonClaudeTimeouts:
    @pytest.mark.timeout(60)
    @pytest.mark.parametrize('backend', SINGLE_SPAWN_BACKENDS)
    async def test_a_timeout_is_the_flat_killed_result(
        self, backend: str, worktree: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        _fake_answers(monkeypatch, stdout=PLAIN_STDOUT, rc=0, sleep=30)
        spy = _SpyRunner()

        agent = await _invoke(backend, worktree, spy, timeout_seconds=0.5)

        [spawn] = spy.spawns
        assert spawn.result == SubprocessResult(
            stdout='',
            stderr='Process killed after 0.5s timeout (SIGTERM+SIGKILL)',
            returncode=1,
            duration_ms=spawn.result.duration_ms,
            timed_out=True,
        )
        assert agent.timed_out is True
        assert agent.success is False


async def test_the_agent_result_is_the_parsers_reading_of_the_runner_result(
    worktree: Path,
) -> None:
    async def canned(cmd: list[str], cwd: Path, env: dict[str, str], **kw: Any) -> SubprocessResult:
        return SubprocessResult(stdout=_CODEX_TURN, stderr='', returncode=0, duration_ms=100)

    agent = await _invoke(
        'codex', worktree, canned,
        prices={'model-x': {'input_per_1m': 100.0, 'output_per_1m': 100.0}},
    )

    assert agent.success is True
    assert agent.input_tokens == 10
    assert agent.output_tokens == 5
    assert agent.cost_usd == pytest.approx((10 * 100.0 + 5 * 100.0) / 1_000_000)
