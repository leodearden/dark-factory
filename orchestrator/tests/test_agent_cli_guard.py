"""Tests for the agent-CLI spawn guard (task 4701): ``_agent_cli_guard.py``
and the autouse ``_forbid_agent_cli_spawn`` fixture in ``conftest.py``."""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest
from _agent_cli_guard import AGENT_CLI_BINARIES, AgentCliSpawnGuard, agent_cli_in
from _operator_home_fixture import OPERATOR_ENV_VARS, write_operator_home
from _zcode_fixture import BASE_URL as ZCODE_BASE_URL
from shared.backends import AGENT_CLI_BACKENDS
from shared.cli_invoke import build_claude_argv

import orchestrator.workflow
from orchestrator.agents import invoke, sandbox_dispatch
from orchestrator.agents.landlock import build_landlock_command
from orchestrator.agents.mimo_backend import (
    mimo_export_argv,
    mimo_run_argv,
    mimo_session_list_argv,
)
from orchestrator.agents.zcode_backend import ZCodeRunHome, zcode_argv

_WORKSPACE_PACKAGES = ('shared', 'orchestrator')

_ZCODE_HOME = ZCodeRunHome(
    root=Path('/run-home'),
    provider_config_path=Path('/run-home/provider_config.json'),
    operator_home=Path('/home/op'),
    workspace=Path('/run-home/workspace'),
)

_UNMOCKED_INVOKE_KWARGS: dict[str, dict[str, Any]] = {
    'claude': {},
    'codex': {},
    'gemini': {},
    'pi': {},
    'zcode': {
        'env_overrides': {'ANTHROPIC_BASE_URL': ZCODE_BASE_URL, 'ANTHROPIC_AUTH_TOKEN': 'k-secret'},
        'timeout_seconds': 30.0,
    },
    'mimo': {
        'env_overrides': {
            'ANTHROPIC_BASE_URL': 'https://token-plan-ams.xiaomimimo.com/anthropic',
            'ANTHROPIC_AUTH_TOKEN': 'k-secret',
        },
        'timeout_seconds': 30.0,
    },
}


def _real_claude_argv() -> list[str]:
    argv, temp_files = build_claude_argv(
        model='sonnet', max_budget_usd=1.0, system_prompt='s', max_turns=1,
        permission_mode='bypassPermissions', allowed_tools=None,
        disallowed_tools=None, mcp_config=None, output_schema=None, effort=None,
        resume_session_id=None, session_id=None,
    )
    for path in temp_files:
        os.unlink(path)
    return argv


class TestAgentCliIn:
    def test_the_guard_covers_every_production_agent_cli_backend(self):
        assert {backend.value for backend in AGENT_CLI_BACKENDS} == AGENT_CLI_BINARIES

    def test_production_claude_argv_is_classified_as_claude(self):
        assert agent_cli_in(_real_claude_argv()) == 'claude'

    def test_claude_behind_cpu_govern_and_nice_prefixes_is_classified(self):
        argv = [
            '/opt/x/cpu-governed-exec.sh', '--role', 'task', '--',
            'nice', '-n', '5', *_real_claude_argv(),
        ]
        assert agent_cli_in(argv) == 'claude'

    def test_landlock_wrapped_claude_is_classified(self, tmp_path: Path):
        argv = build_landlock_command(['claude', '--print'], tmp_path, ['mod_a'])
        assert agent_cli_in(argv) == 'claude'

    def test_absolute_path_to_the_cli_is_classified(self):
        assert agent_cli_in(['/usr/local/bin/claude', '--print']) == 'claude'

    def test_tuple_args_as_asyncio_passes_them_are_classified(self):
        assert agent_cli_in(('claude', '--print')) == 'claude'

    @pytest.mark.parametrize(
        ('args', 'expected'),
        [
            ('claude', 'claude'),
            (b'claude', 'claude'),
            (Path('/usr/bin/codex'), 'codex'),
        ],
        ids=['str', 'bytes', 'pathlike'],
    )
    def test_a_single_program_argument_is_classified(self, args, expected):
        assert agent_cli_in(args) == expected

    @pytest.mark.parametrize(
        ('argv', 'expected'),
        [
            (['codex', 'exec', '-'], 'codex'),
            (['gemini', '-p', 'x'], 'gemini'),
            (['pi', '--model', 'm'], 'pi'),
            (zcode_argv('p', _ZCODE_HOME, None), 'zcode'),
            (mimo_run_argv('mimo-v2.6-pro', Path('/wt'), 'high'), 'mimo'),
            (mimo_session_list_argv(), 'mimo'),
            (mimo_export_argv('ses-1'), 'mimo'),
        ],
        ids=['codex', 'gemini', 'pi', 'zcode', 'mimo-run', 'mimo-session-list', 'mimo-export'],
    )
    def test_every_non_claude_backend_argv_is_classified(self, argv, expected):
        assert agent_cli_in(argv) == expected

    @pytest.mark.parametrize(
        'args',
        [
            ('git', 'rev-parse', 'HEAD'),
            ['uv', 'sync', '--project', 'x'],
            ['ls', '/home/u/.claude/projects'],
            ['python', '-c', 'import math; print(math.pi)'],
            ['claude-code'],
            [],
        ],
        ids=['git', 'uv', 'dot-claude-path', 'math-pi', 'claude-prefix', 'empty'],
    )
    def test_non_agent_argv_is_not_classified(self, args):
        assert agent_cli_in(args) is None


_NODEID = 'tests/test_x.py::test_y'
_NICE_CLAUDE = ('nice', '-n', '5', 'claude', '--print')


class TestAgentCliSpawnGuard:
    @pytest.fixture
    def calls(self) -> list[tuple]:
        return []

    @pytest.fixture
    def guard(self) -> AgentCliSpawnGuard:
        return AgentCliSpawnGuard(_NODEID)

    @pytest.fixture
    def wrapped(self, guard, calls):
        def fake_init(popen_self, args, *a, **kw):
            calls.append((args, a, kw))

        return guard.guarded(fake_init)

    def test_non_agent_spawns_are_delegated_unchanged(self, guard, wrapped, calls):
        wrapped(object(), ['git', 'status'], cwd='/tmp')

        assert calls == [(['git', 'status'], (), {'cwd': '/tmp'})]
        assert guard.violation_message() is None

    def test_an_agent_spawn_is_blocked_at_the_seam(self, wrapped, calls):
        with pytest.raises(pytest.fail.Exception) as exc:
            wrapped(object(), _NICE_CLAUDE)

        assert calls == []
        message = str(exc.value)
        assert _NODEID in message
        assert "'claude'" in message
        assert repr(_NICE_CLAUDE) in message
        assert 'spawns_agent_cli' in message
        assert 'orchestrator.workflow.invoke_agent' in message
        assert not isinstance(exc.value, Exception)

    def test_a_blocked_spawn_is_recorded(self, guard, wrapped):
        with pytest.raises(pytest.fail.Exception):
            wrapped(object(), _NICE_CLAUDE)

        message = guard.violation_message()
        assert message is not None
        assert _NODEID in message
        assert repr(_NICE_CLAUDE) in message

    def test_a_one_shot_argv_iterator_is_delegated_intact(self, wrapped, calls):
        wrapped(object(), iter(['git', 'status']))

        [(args, _, _)] = calls
        assert list(args) == ['git', 'status']

    def test_a_one_shot_agent_argv_iterator_is_recorded_intact(self, guard, wrapped):
        with pytest.raises(pytest.fail.Exception, match=re.escape(repr(_NICE_CLAUDE))):
            wrapped(object(), iter(_NICE_CLAUDE))

        assert guard.take_attempts() == [_NICE_CLAUDE]

    def test_taking_the_attempts_acknowledges_them(self, guard, wrapped):
        with pytest.raises(pytest.fail.Exception):
            wrapped(object(), _NICE_CLAUDE)

        assert guard.take_attempts() == [_NICE_CLAUDE]
        assert guard.take_attempts() == []
        assert guard.violation_message() is None

    def test_every_blocked_spawn_is_recorded_in_order(self, guard, wrapped):
        second = ('/usr/bin/codex', 'exec', '-')
        for argv in (_NICE_CLAUDE, second):
            with pytest.raises(pytest.fail.Exception):
                wrapped(object(), argv)

        message = guard.violation_message()
        assert message is not None
        assert repr(_NICE_CLAUDE) in message
        assert repr(second) in message
        assert guard.take_attempts() == [_NICE_CLAUDE, second]


class TestForbidAgentCliSpawnFixture:
    def test_sync_agent_spawn_fails_at_the_seam(self, _forbid_agent_cli_spawn, request):
        with pytest.raises(pytest.fail.Exception, match=re.escape(request.node.nodeid)):
            subprocess.run(['claude', '--version'])

        assert _forbid_agent_cli_spawn.take_attempts() == [('claude', '--version')]

    @pytest.mark.asyncio
    async def test_async_agent_spawn_fails_at_the_seam(self, _forbid_agent_cli_spawn, request):
        with pytest.raises(pytest.fail.Exception, match=re.escape(request.node.nodeid)):
            await asyncio.create_subprocess_exec(
                'claude', '--print',
                stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
            )

        assert _forbid_agent_cli_spawn.take_attempts() == [('claude', '--print')]

    @pytest.mark.asyncio
    @pytest.mark.parametrize('backend', sorted(AGENT_CLI_BACKENDS))
    async def test_unmocked_invoke_agent_is_stopped_before_spawning(
        self, _forbid_agent_cli_spawn, request, monkeypatch, tmp_path, backend,
    ):
        empty_bin = tmp_path / 'empty-bin'
        empty_bin.mkdir()
        monkeypatch.setenv('PATH', str(empty_bin))
        monkeypatch.setenv('HOME', str(write_operator_home(tmp_path / 'operator')))
        for name in OPERATOR_ENV_VARS:
            monkeypatch.delenv(name, raising=False)

        with pytest.raises(pytest.fail.Exception, match=re.escape(request.node.nodeid)):
            await invoke.invoke_agent(
                prompt='p', system_prompt='s', cwd=tmp_path, backend=backend,
                **_UNMOCKED_INVOKE_KWARGS[backend],
            )

        [attempt] = _forbid_agent_cli_spawn.take_attempts()
        assert agent_cli_in(attempt) == backend

    @pytest.mark.asyncio
    async def test_the_sandbox_branch_is_stopped_before_spawning(
        self, _forbid_agent_cli_spawn, monkeypatch, tmp_path,
    ):
        monkeypatch.setattr(sandbox_dispatch, 'resolve_active_backend', lambda: 'landlock')

        with pytest.raises(pytest.fail.Exception):
            await invoke.invoke_agent(
                prompt='p', system_prompt='s', cwd=tmp_path, backend='claude',
                sandbox_modules=['mod_a'],
            )

        [attempt] = _forbid_agent_cli_spawn.take_attempts()
        assert agent_cli_in(attempt) == 'claude'

    @pytest.mark.asyncio
    async def test_the_task_882_stale_patch_is_caught_at_the_seam(
        self, _forbid_agent_cli_spawn, tmp_path,
    ):
        with (
            patch('orchestrator.agents.invoke.invoke_agent', new=AsyncMock()) as stale,
            pytest.raises(pytest.fail.Exception),
        ):
            await orchestrator.workflow.invoke_agent(prompt='p', system_prompt='s', cwd=tmp_path)

        stale.assert_not_awaited()
        [attempt] = _forbid_agent_cli_spawn.take_attempts()
        assert agent_cli_in(attempt) == 'claude'

    def test_a_swallowed_spawn_is_still_recorded(self, _forbid_agent_cli_spawn):
        with contextlib.suppress(BaseException):
            subprocess.run(['claude'])

        assert _forbid_agent_cli_spawn.take_attempts() == [('claude',)]

    def test_non_agent_sync_spawns_still_run(self):
        assert subprocess.run(['true']).returncode == 0

    @pytest.mark.asyncio
    async def test_non_agent_async_spawns_still_run(self):
        process = await asyncio.create_subprocess_exec('true')

        assert await process.wait() == 0

    @pytest.mark.spawns_agent_cli
    def test_spawns_agent_cli_marker_restores_real_spawning(self, tmp_path):
        fake_claude = tmp_path / 'claude'
        fake_claude.write_text('#!/bin/sh\necho fake-claude\n')
        fake_claude.chmod(0o755)

        result = subprocess.run([str(fake_claude)], capture_output=True, text=True)

        assert result.returncode == 0
        assert result.stdout.strip() == 'fake-claude'


def test_importing_the_guard_loads_no_workspace_package() -> None:
    probe = (
        'import json, sys, _agent_cli_guard; '
        f'print(json.dumps(sorted(set({_WORKSPACE_PACKAGES!r}) & set(sys.modules))))'
    )
    child = subprocess.run(
        [sys.executable, '-c', probe],
        cwd=Path(__file__).parent,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert child.returncode == 0, child.stderr

    leaked = json.loads(child.stdout)
    assert leaked == [], (
        f'importing _agent_cli_guard loaded {leaked}; conftest.py imports it '
        'before putting this worktree\'s src dirs on sys.path'
    )
