"""Probe-shaped ZCode values and an operator home, shared by the zcode backend and invoke tests.

The shapes follow plans/zcode-harness-probe-2026-11.md. Each builder returns
a fresh value, so no test can mutate another's copy.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from orchestrator.agents.zcode_backend import OPERATOR_TOOL_HOMES

BASE_URL = 'https://api.z.ai/api/anthropic'
OPERATOR_EMAIL = 'op@example.test'
OPERATOR_IGNORED_FILE = 'dropped.operator-ignored'
OPERATOR_ENV_VARS = (*(name for name, _ in OPERATOR_TOOL_HOMES), 'GIT_CONFIG_GLOBAL')


def measured_output() -> dict[str, Any]:
    """The ``--output-format json`` object of one successful run."""
    return {
        'sessionId': 'sess-123',
        'traceId': 'trace-1',
        'turnId': 'turn-1',
        'response': 'probe complete',
        'usage': {
            'source': 'provider',
            'modelRequestCount': 2,
            'inputTokens': 1000,
            'outputTokens': 50,
            'totalTokens': 1050,
            'cacheReadTokens': 300,
            'cacheWriteTokens': 100,
            'reasoningTokens': 0,
        },
        'eventCount': 12,
        'projection': {'status': 'completed', 'turnCount': 1},
    }


def plan_tools_mcp() -> dict[str, Any]:
    return {'mcpServers': {'plan-tools': {
        'command': '/venv/bin/python', 'args': ['-m', 'plan_tools'],
    }}}


def write_operator_home(root: Path, *, with_gitconfig: bool = True) -> Path:
    """An operator home whose git identity is OPERATOR_EMAIL and whose excludes drop OPERATOR_IGNORED_FILE."""
    root.mkdir()
    if with_gitconfig:
        (root / '.gitconfig').write_text(
            f'[user]\n\tname = Op Erator\n\temail = {OPERATOR_EMAIL}\n',
        )
    ignore = root / '.config' / 'git' / 'ignore'
    ignore.parent.mkdir(parents=True)
    ignore.write_text(f'{OPERATOR_IGNORED_FILE}\n')
    return root
