"""Shared primitive for extracting a structured verdict from an agent result payload.

Replaces the silent `.get('verdict', default)` fall-through pattern with a loud,
distinguishable sentinel so that a failed agent run is never silently laundered
into a neutral result.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

__all__ = ['AgentVerdict', 'extract_agent_verdict']


@dataclass
class AgentVerdict:
    """Structured result from extract_agent_verdict.

    On success, `failed` is False and `verdict`/`summary` carry the agent's output.
    On failure, `failed` is True, `summary` starts with 'agent-failed:<token>', and
    a WARNING has been emitted so the failure is never silent.
    """

    verdict: str
    summary: str
    failed: bool = False
    raw: dict | None = None


def extract_agent_verdict(
    result: object,
    *,
    default_verdict: str,
    error_summary: str,
) -> AgentVerdict:
    """Extract a structured verdict from a raw agent result payload.

    SUCCESS: ``result`` is a dict with a truthy ``'verdict'`` key.
    Returns ``AgentVerdict(verdict=result['verdict'], summary=result.get('summary',''),
    failed=False, raw=result)``.  No log is emitted on the success path.

    FAILURE: everything else (None, non-dict, dict missing 'verdict', dict with a
    ``'warning'`` key from AgentLoop).  Emits ``logger.warning(...)`` and returns
    ``AgentVerdict(verdict=default_verdict, summary='agent-failed:<token>',
    failed=True, raw=...)``.

    The distinguishing signal lives in ``summary`` (the ``agent-failed:`` prefix) AND
    the ``failed`` flag — never in the ``verdict`` field alone, which equals the
    caller-supplied ``default_verdict`` and may coincide with a legitimate value.
    """
    # ── FAILURE branch ──────────────────────────────────────────────────────────
    # Determine the token: use the 'warning' key when present (matches the shapes
    # emitted by AgentLoop.run(): {'warning': 'no_tool_calls'} / {'warning':
    # 'max_steps_reached'}), otherwise fall back to the caller-supplied error_summary.
    # (SUCCESS branch added in step-4.)
    if isinstance(result, dict) and 'warning' in result:
        token = result['warning']
    else:
        token = error_summary

    output_prefix = str(result)[:200]
    logger.warning(
        'agent produced no parseable verdict: %s | %s',
        token,
        output_prefix,
    )
    return AgentVerdict(
        verdict=default_verdict,
        summary=f'agent-failed:{token}',
        failed=True,
        raw=result if isinstance(result, dict) else None,
    )
