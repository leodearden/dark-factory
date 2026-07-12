"""Verdict-tools MCP server -- stdio tool server for verdict-artifact writes.

Spawned per-agent invocation by the orchestrator (mirrors plan_tools.py).
A role-specific agent (reviewer/judge/triage/merger) submits its structured
verdict via the single tool registered for its ``--verdict-role``. The tool
writes a schema-versioned envelope to ``verdicts/<role>.json`` under the
task's ``TaskArtifacts`` root — the role-derived filename is authoritative
(never an agent-supplied field), so a reviewer cannot misname its artifact
onto a sibling's path.

Usage (stdio transport, spawned by orchestrator):
    <sys.executable> -m orchestrator.mcp.verdict_tools --worktree /path/to/worktree \
        --verdict-role judge --session-id <session_id>
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import Any

from fastmcp import FastMCP

from orchestrator.artifacts import TaskArtifacts

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
                reviewer: The reviewer's name (matches --verdict-role).
                verdict: One of "PASS" or "ISSUES_FOUND".
                issues: Structured list of issues found (empty if PASS).
                summary: One-paragraph summary of the review.
            """
            return _submit_review_verdict(
                artifacts, role, session_id, reviewer, verdict, issues, summary,
            )

    return mcp
