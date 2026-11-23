"""A MagicMock stand-in for shared.cli_invoke.AgentResult that classify_agent_failure() reads faithfully.

Shared by test_dry_run_unblock.py and the test_b3_gate*.py modules, which feed it
to dry_run_unblock._build_entry. Extend it here when AgentResult gains or renames
a field that classify_agent_failure() consults.
"""

from __future__ import annotations

from unittest.mock import MagicMock


def make_agent_result(*, success=True, cost_usd=0.50, structured_output=None,
                      subtype='', output='', duration_ms=1000,
                      timed_out=False, transcript_turns=None, session_id='',
                      stderr='', turns=0, api_error_status=None,
                      schema_salvaged=False, output_tokens=None,
                      ended_awaiting_background=False):
    """A MagicMock AgentResult with every field classify_agent_failure() reads set explicitly.

    A bare MagicMock auto-vivifies every unset attribute as a truthy MagicMock,
    which would corrupt classify_agent_failure()'s decision rules: a truthy
    .ended_awaiting_background misclassifies every failure as
    ENDED_AWAITING_BACKGROUND, a truthy .timed_out as TIMED_OUT, and a truthy
    .schema_salvaged a generic failure as STRUCTURAL instead of UNKNOWN.
    .output_tokens is read when formatting the MAX_TURNS summary.
    """
    r = MagicMock()
    r.success = success
    r.cost_usd = cost_usd
    r.structured_output = structured_output
    r.subtype = subtype
    r.output = output
    r.duration_ms = duration_ms
    r.timed_out = timed_out
    r.transcript_turns = transcript_turns
    r.session_id = session_id
    r.stderr = stderr
    r.turns = turns
    r.api_error_status = api_error_status
    r.schema_salvaged = schema_salvaged
    r.output_tokens = output_tokens
    r.ended_awaiting_background = ended_awaiting_background
    return r
