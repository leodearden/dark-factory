"""Verdict-tools MCP server -- stdio tool server for verdict-artifact writes.

Spawned per-agent invocation by the orchestrator (mirrors plan_tools.py).
A role-specific agent (reviewer/judge/triage/merger) submits its structured
verdict via the single tool registered for its ``--verdict-role``. The tool
writes a schema-versioned envelope to ``verdicts/<role>.json`` under the
task's ``TaskArtifacts`` root — the role-derived filename is authoritative
(never an agent-supplied field), so a reviewer cannot misname its artifact
onto a sibling's path. For reviewer verdicts, the ``reviewer`` payload field
is also validated to match ``--verdict-role``: a mismatch is rejected
(``status: error``) rather than written, so the payload can never disagree
with the filename it lands in either.

Usage (stdio transport, spawned by orchestrator):
    <sys.executable> -m orchestrator.mcp.verdict_tools --worktree /path/to/worktree \
        --verdict-role judge --session-id <session_id>
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from fastmcp import FastMCP

from orchestrator.artifacts import TaskArtifacts, _validate_verdict_role

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Envelope helper
# ---------------------------------------------------------------------------

SCHEMA_VERSION = 1


def _envelope(role: str, session_id: str, payload: dict) -> dict:
    """Wrap *payload* in the schema-versioned verdict envelope."""
    return {
        'role': role,
        'schema_version': SCHEMA_VERSION,
        'session_id': session_id,
        'emitted_at': datetime.now(UTC).isoformat(),
        'verdict': payload,
    }


# ---------------------------------------------------------------------------
# Standalone implementation functions (testable without MCP transport)
#
# Only _submit_review_verdict has constrained/validated fields (the
# reviewer==role identity check and the verdict enum below) — the other
# three payloads (completion/triage/merge) have no enum-like fields today,
# so they write whatever they receive. Booleans and lists are still
# type-checked at the pydantic MCP tool boundary in create_server(). Add
# parallel validation here if any of those payloads gain a constrained
# field in the future.
# ---------------------------------------------------------------------------


def _submit_review_verdict(
    artifacts: TaskArtifacts,
    role: str,
    session_id: str,
    reviewer: str,
    verdict: str,
    issues: list[dict],
    summary: str,
) -> dict[str, Any]:
    if reviewer != role:
        return {
            'status': 'error',
            'message': (
                f'reviewer {reviewer!r} must match --verdict-role {role!r} — '
                'the artifact filename is authoritative for this role; a '
                'mismatched payload is rejected rather than written'
            ),
        }
    if verdict not in {'PASS', 'ISSUES_FOUND'}:
        return {
            'status': 'error',
            'message': f'verdict must be one of PASS/ISSUES_FOUND, got {verdict!r}',
        }
    payload = {
        'reviewer': reviewer,
        'verdict': verdict,
        'issues': issues,
        'summary': summary,
    }
    artifacts.write_verdict(role, _envelope(role, session_id, payload))
    return {'status': 'ok', 'role': role}


def _submit_completion_verdict(
    artifacts: TaskArtifacts,
    role: str,
    session_id: str,
    complete: bool,
    reasoning: str,
    uncovered_plan_steps: list[str],
    substantive_work: bool,
) -> dict[str, Any]:
    payload = {
        'complete': complete,
        'reasoning': reasoning,
        'uncovered_plan_steps': uncovered_plan_steps,
        'substantive_work': substantive_work,
    }
    artifacts.write_verdict(role, _envelope(role, session_id, payload))
    return {'status': 'ok', 'role': role}


def _submit_triage(
    artifacts: TaskArtifacts,
    role: str,
    session_id: str,
    accepted: list[dict],
    skipped: list[dict],
    proposed_task_groups: list[dict],
) -> dict[str, Any]:
    payload = {
        'accepted': accepted,
        'skipped': skipped,
        'proposed_task_groups': proposed_task_groups,
    }
    artifacts.write_verdict(role, _envelope(role, session_id, payload))
    return {'status': 'ok', 'role': role}


def _submit_merge_disposition(
    artifacts: TaskArtifacts,
    role: str,
    session_id: str,
    blocked: bool,
    reason: str,
) -> dict[str, Any]:
    payload = {
        'blocked': blocked,
        'reason': reason,
    }
    artifacts.write_verdict(role, _envelope(role, session_id, payload))
    return {'status': 'ok', 'role': role}


# ---------------------------------------------------------------------------
# FastMCP server factory
# ---------------------------------------------------------------------------

# Roles that get a dedicated singleton tool. Any role NOT in this set is
# treated as a reviewer-panel member's name (defined in roles.py, task β) —
# verdict_tools stays decoupled from that list and falls back to
# submit_review_verdict for it.
_SINGLETON_ROLE_TOOLS = frozenset({'judge', 'triage', 'merger'})


def create_server(artifacts: TaskArtifacts, role: str, session_id: str = '') -> FastMCP:
    """Create the verdict-tools MCP server with EXACTLY ONE tool registered,
    selected by *role*.
    """
    mcp = FastMCP('verdict-tools')

    if role == 'judge':
        @mcp.tool()
        def submit_completion_verdict(
            complete: bool,
            reasoning: str,
            uncovered_plan_steps: list[str],
            substantive_work: bool,
        ) -> dict[str, Any]:
            """Submit the completion-judge verdict for this task.

            Args:
                complete: Whether the plan's substantive work is complete.
                reasoning: Explanation for the verdict.
                uncovered_plan_steps: Plan step ids not covered by the diff.
                substantive_work: Whether the diff contains substantive work.
            """
            return _submit_completion_verdict(
                artifacts, role, session_id,
                complete, reasoning, uncovered_plan_steps, substantive_work,
            )
    elif role == 'triage':
        @mcp.tool()
        def submit_triage(
            accepted: list[dict],
            skipped: list[dict],
            proposed_task_groups: list[dict],
        ) -> dict[str, Any]:
            """Submit the triage verdict for this task.

            Args:
                accepted: Suggestions accepted for follow-up tasks.
                skipped: Suggestions skipped, with reasons.
                proposed_task_groups: Proposed grouping of accepted suggestions.
            """
            return _submit_triage(
                artifacts, role, session_id,
                accepted, skipped, proposed_task_groups,
            )
    elif role == 'merger':
        @mcp.tool()
        def submit_merge_disposition(
            blocked: bool,
            reason: str,
        ) -> dict[str, Any]:
            """Submit the merge disposition verdict for this task.

            Args:
                blocked: Whether the merge should be blocked.
                reason: Explanation for the disposition (may be empty when
                    not blocked).
            """
            return _submit_merge_disposition(
                artifacts, role, session_id, blocked, reason,
            )
    else:
        # A reviewer-panel member's name — see _SINGLETON_ROLE_TOOLS above.
        assert role not in _SINGLETON_ROLE_TOOLS

        @mcp.tool()
        def submit_review_verdict(
            reviewer: str,
            verdict: str,
            issues: list[dict],
            summary: str,
        ) -> dict[str, Any]:
            """Submit a reviewer's verdict for this task.

            Args:
                reviewer: The reviewer's name. Must equal --verdict-role;
                    a mismatch is rejected (status: error) rather than
                    written.
                verdict: One of "PASS" or "ISSUES_FOUND".
                issues: Structured list of issues found (empty if PASS).
                summary: One-paragraph summary of the review.
            """
            return _submit_review_verdict(
                artifacts, role, session_id, reviewer, verdict, issues, summary,
            )

    return mcp


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def _artifacts_from_args(
    argv: list[str] | None = None,
) -> tuple[TaskArtifacts, str, str]:
    """Parse ``--worktree``/``--meta-root``/``--verdict-role``/``--session-id``
    and return the corresponding ``(TaskArtifacts, role, session_id)``.

    The worktree/meta-root resolution mirrors plan_tools.py's
    ``_artifacts_from_args`` byte-identically (parity signal) so both sides
    resolve the IDENTICAL meta_root for a given worktree. When
    ``--meta-root`` is omitted, the root dir checked for existence is the
    legacy ``<worktree>/.task``. ``--verdict-role`` is validated eagerly
    here (``_validate_verdict_role``) so a malformed role aborts server
    startup with a clear message, instead of surfacing as an opaque
    ValueError on every subsequent tool call.
    """
    parser = argparse.ArgumentParser(description='Verdict-tools MCP server (stdio)')
    parser.add_argument(
        '--worktree', type=Path, required=True,
        help='Path to the git worktree containing .task/',
    )
    parser.add_argument(
        '--meta-root', type=Path, default=None,
        help=(
            'Optional `.task-meta` root (sibling of the worktree). When '
            'omitted, verdicts/ et al. are read/written at the legacy '
            '<worktree>/.task location.'
        ),
    )
    parser.add_argument(
        '--verdict-role', type=str, required=True,
        help='The role submitting a verdict (judge/triage/merger, or a '
             'reviewer-panel name) — selects the one registered tool and '
             'the authoritative verdicts/<role>.json filename.',
    )
    parser.add_argument(
        '--session-id', type=str, default='',
        help='Optional session id recorded in the verdict envelope.',
    )
    args = parser.parse_args(argv)

    try:
        _validate_verdict_role(args.verdict_role)
    except ValueError as exc:
        print(f'Error: {exc}', file=sys.stderr)
        sys.exit(1)

    worktree = args.worktree.resolve()
    meta_root = args.meta_root.resolve() if args.meta_root is not None else None
    root_to_check = meta_root if meta_root is not None else worktree / '.task'
    if not root_to_check.is_dir():
        print(f'Error: {root_to_check} does not exist', file=sys.stderr)
        sys.exit(1)

    return TaskArtifacts(worktree, meta_root), args.verdict_role, args.session_id


def main() -> None:
    """Parse CLI args and run the stdio MCP server."""
    artifacts, role, session_id = _artifacts_from_args()
    server = create_server(artifacts, role, session_id)
    server.run(transport='stdio')


if __name__ == '__main__':
    main()
