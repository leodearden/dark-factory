"""Tests for the agent-CLI spawn guard (task 4701): ``_agent_cli_guard.py``
and the autouse ``_forbid_agent_cli_spawn`` fixture in ``conftest.py``."""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from _agent_cli_guard import AGENT_CLI_BINARIES, agent_cli_in
from shared.cli_invoke import build_claude_argv

from orchestrator.agents.landlock import build_landlock_command


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
    def test_the_backend_set_is_the_four_invoke_agent_backends(self):
        assert frozenset({'claude', 'codex', 'gemini', 'pi'}) == AGENT_CLI_BINARIES

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
        ],
        ids=['codex', 'gemini', 'pi'],
    )
    def test_every_backend_binary_is_classified(self, argv, expected):
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
