"""Probe-shaped ZCode values, shared by the zcode backend and invoke tests.

The shapes follow plans/zcode-harness-probe-2026-11.md. Each builder returns
a fresh value, so no test can mutate another's copy.
"""

from __future__ import annotations

from typing import Any

BASE_URL = 'https://api.z.ai/api/anthropic'


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
