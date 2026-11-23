"""The CLI's rejection of an unresolvable ``--resume``, measured live.

One copy of the measurement for every suite that replays it, so a
re-measurement updates them all at once. It lives in the shipped ``shared``
package, beside :mod:`shared.testing_fake_claude`, because ``shared.tests`` is
not importable from the orchestrator pytest rootdir and both suites drive it.
"""

from __future__ import annotations

from shared.cli_invoke import AgentResult, parse_claude_cli_output

__all__ = ['measured_resume_rejection']

# Measured 2026-10-31 (CLI 2.1.314) with an empty CLAUDE_CONFIG_DIR and no
# credentials -- the CLI resolves the session before auth -- via
# `claude --print --output-format json --resume <uuid>`: stdout 0 bytes, the
# text below on stderr, exit 1, ~1.0s. Matches the 2.1.236 measurement in
# plans/session-resume-eligibility-seam-prd.md.
_MEASURED_RETURNCODE = 1


def measured_resume_rejection(
    session_id: str = '4aed993b-20c0-4b91-a8dd-60180e7db2e0',
) -> AgentResult:
    """The CLI's rejection of a ``--resume`` of *session_id*, as production parses it.

    A fresh object per call: several consumers mutate the result.
    """
    return parse_claude_cli_output(
        stdout='',
        stderr=f'No conversation found with session ID: {session_id}',
        returncode=_MEASURED_RETURNCODE,
    )
