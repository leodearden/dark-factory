"""The task-3118 refused AgentResult every scoring-site test drives, and a parseable answer beside it."""

from __future__ import annotations

import json

from shared.cli_invoke import AgentResult

REFUSAL_COST_USD = 0.31
REFUSAL_TEXT = "You've hit your session limit · resets 8pm"


def refused_agent_result(cost_usd: float = REFUSAL_COST_USD) -> AgentResult:
    return AgentResult(
        success=False,
        subtype='error',
        api_error_status=429,
        output=REFUSAL_TEXT,
        cost_usd=cost_usd,
        duration_ms=1200,
        turns=0,
    )


def answered(payload: dict, cost_usd: float = 0.0) -> AgentResult:
    return AgentResult(
        success=True,
        structured_output=payload,
        output=json.dumps(payload),
        cost_usd=cost_usd,
        duration_ms=1000,
        turns=1,
    )
