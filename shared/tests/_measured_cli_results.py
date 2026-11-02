"""``claude --print --output-format json`` results measured live.

One copy of each measurement for every suite that replays it, so a
re-measurement updates them all at once. Consumers import by bare module name
-- ``conftest.py`` prepends ``shared/tests`` to ``sys.path`` (same convention
as ``_usage_gate_test_helpers.py``). The leading underscore keeps pytest from
collecting this module as a test file.
"""

from __future__ import annotations

import json

from shared.cli_invoke import AgentResult, parse_claude_cli_output

ORG_DISABLED_TEXT = (
    'Your organization has disabled Claude subscription access for Claude '
    'Code · Use an Anthropic API key instead, or ask your admin to enable '
    'access'
)

# Measured 2026-09-26 (CLI 2.1.283) for the org-disabled max-b/max-c
# accounts, via the production probe command.
MEASURED_ORG_DISABLED_403 = json.dumps({
    'type': 'result',
    'subtype': 'success',
    'is_error': True,
    'api_error_status': 403,
    'result': ORG_DISABLED_TEXT,
    'num_turns': 1,
    'total_cost_usd': 0,
})


# Measured 2026-10-31 (CLI 2.1.314) with an empty CLAUDE_CONFIG_DIR and no
# credentials -- the CLI resolves the session before auth -- via
# `claude --print --output-format json --resume <uuid>`: stdout 0 bytes, the
# text below on stderr, exit 1, ~1.0s. Matches the 2.1.236 measurement in
# plans/session-resume-eligibility-seam-prd.md.
MEASURED_RESUME_REJECTED_RETURNCODE = 1


def measured_resume_rejected_stderr(session_id: str) -> str:
    return f'No conversation found with session ID: {session_id}'


def measured_resume_rejection(
    session_id: str = '4aed993b-20c0-4b91-a8dd-60180e7db2e0',
) -> AgentResult:
    """The CLI's rejection of an unresolvable ``--resume``, as production parses it.

    A fresh object per call: several consumers mutate the result.
    """
    return parse_claude_cli_output(
        stdout='',
        stderr=measured_resume_rejected_stderr(session_id),
        returncode=MEASURED_RESUME_REJECTED_RETURNCODE,
    )
