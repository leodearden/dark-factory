"""Probe-shaped MiMo Code values, shared by the mimo backend and invoke tests.

The shapes follow plans/mimo-code-harness-probe-2026-11.md. Each builder
returns a fresh value, so no test can mutate another's copy.
"""

from __future__ import annotations

from typing import Any


def _assistant_message(
    *, finish: str | None, error: dict[str, Any] | None, parts: list[dict[str, Any]],
) -> dict[str, Any]:
    info: dict[str, Any] = {
        'role': 'assistant',
        'tokens': {'input': 700, 'output': 30, 'reasoning': 20, 'cache': {'read': 300, 'write': 0}},
        'finish': finish,
    }
    if error is not None:
        info['error'] = error
    return {'info': info, 'parts': parts}


def measured_export(
    session_id: str = 'ses_x',
    *,
    error: dict[str, Any] | None = None,
    finish: str | None = 'stop',
    parent_id: str | None = None,
) -> dict[str, Any]:
    """One ``mimo export <id>`` document: a user message and two assistant steps."""
    info: dict[str, Any] = {'id': session_id}
    if parent_id is not None:
        info['parentID'] = parent_id
    return {
        'info': info,
        'messages': [
            {'info': {'role': 'user'}, 'parts': [{'type': 'text', 'text': 'do step-1'}]},
            _assistant_message(finish='tool-calls', error=None, parts=[{'type': 'tool'}]),
            _assistant_message(
                finish=finish, error=error, parts=[{'type': 'text', 'text': 'probe complete'}],
            ),
        ],
    }
