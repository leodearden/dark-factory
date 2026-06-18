"""Shared primitive for extracting a structured verdict from an agent result payload.

Replaces the silent `.get('verdict', default)` fall-through pattern with a loud,
distinguishable sentinel so that a failed agent run is never silently laundered
into a neutral result.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

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
    # ── GUARD: 2-tuple misuse ───────────────────────────────────────────────────
    # AgentLoop.run() returns (result_dict, journal_entries).  A caller who
    # passes the whole tuple instead of unpacking it gets isinstance(result,
    # dict)==False and silently falls to the failure sentinel with an unhelpful
    # token.  Detect this early and emit a distinct warning so the misuse is
    # loud rather than indistinguishable from a real agent failure.
    if isinstance(result, tuple) and len(result) == 2 and isinstance(result[0], dict):
        logger.warning(
            'extract_agent_verdict received a 2-tuple; did you forget to unpack'
            ' AgentLoop.run()? Pass result[0] directly. Treating as failure.'
        )
        # Intentionally do NOT recover — the caller must fix the call site.
        # Fall through to the failure branch below (the tuple is not a dict).

    # ── SUCCESS branch ──────────────────────────────────────────────────────────
    # NOTE: a truthy 'verdict' key takes precedence over any 'warning' key that
    # may also be present in the payload.  When both keys coexist the payload is
    # treated as a successful (though possibly partial) result and returned
    # without a log.  Callers that need to inspect the warning in this case can
    # read it from result.raw after the call.
    if isinstance(result, dict) and result.get('verdict'):
        return AgentVerdict(
            verdict=result['verdict'],
            summary=result.get('summary', ''),
            failed=False,
            raw=result,
        )

    # ── FAILURE branch ──────────────────────────────────────────────────────────
    # Determine the token: use the 'warning' key when present (matches the shapes
    # emitted by AgentLoop.run(): {'warning': 'no_tool_calls'} / {'warning':
    # 'max_steps_reached'}), otherwise fall back to the caller-supplied error_summary.
    token = result['warning'] if isinstance(result, dict) and 'warning' in result else error_summary

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
