"""The lint gate bans asyncio.get_event_loop() under orchestrator/tests.

Task 8395 replaces task 1711's AST guard with ruff's TID251; the replacement
idioms live in the banned-api msg in orchestrator/pyproject.toml.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ORCH_ROOT = Path(__file__).resolve().parents[1]

_PLANTED_SOURCE = (
    'import asyncio\n'
    'from asyncio import get_event_loop\n'
    '\n'
    '\n'
    'async def planted():\n'
    '    asyncio.get_event_loop()\n'
    '    asyncio.events.get_event_loop()\n'
    '    asyncio.get_running_loop()\n'
    '    return get_event_loop\n'
)


def _banned_api_findings(source: str) -> dict[int, str]:
    """Lint *source* as a file under orchestrator/tests with the package's own config; TID251 row -> message."""
    proc = subprocess.run(
        [
            sys.executable, '-m', 'ruff', 'check', '--no-cache',
            '--output-format', 'json',
            '--stdin-filename', 'tests/test_planted_get_event_loop.py', '-',
        ],
        input=source, cwd=ORCH_ROOT, capture_output=True, text=True,
    )
    assert proc.returncode in (0, 1), f'ruff failed (rc={proc.returncode}): {proc.stderr}'
    assert proc.stdout.strip(), f'ruff produced no JSON (rc={proc.returncode}): {proc.stderr}'
    return {
        finding['location']['row']: finding['message']
        for finding in json.loads(proc.stdout)
        if finding['code'] == 'TID251'
    }


def test_ruff_flags_a_planted_get_event_loop_under_orchestrator_tests() -> None:
    findings = _banned_api_findings(_PLANTED_SOURCE)

    assert sorted(findings) == [2, 6, 7], (
        'expected TID251 on the from-import (row 2) and both call spellings (rows 6, 7), '
        f'and not on get_running_loop (row 8); got {findings}'
    )
    for row, message in findings.items():
        assert 'get_running_loop' in message, f'row {row} message names no remedy: {message!r}'
