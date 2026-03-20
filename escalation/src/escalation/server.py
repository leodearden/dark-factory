"""Escalation MCP server — FastMCP tools for agents and handlers."""

from __future__ import annotations

from typing import Any

from fastmcp import FastMCP

from escalation.models import Escalation
from escalation.queue import EscalationQueue

CATEGORIES = [
    'scope_violation',
    'design_concern',
    'cleanup_needed',
    'dependency_discovered',
    'risk_identified',
    'infra_issue',
    'task_failure',
    # Reconciliation categories
    'recon_failure',
    'recon_backlog_overflow',
    'recon_stale_run',
    'recon_integrity_issue',
]


def create_server(queue: EscalationQueue) -> FastMCP:
    """Create the escalation MCP server with all tools registered."""
    mcp = FastMCP('escalation')

    # --- Agent-side tools ---

    @mcp.tool()
    def escalate_info(
        task_id: str,
        agent_role: str,
        category: str,
        summary: str,
        detail: str = '',
        suggested_action: str = '',
        worktree: str | None = None,
        workflow_state: str | None = None,
    ) -> dict[str, Any]:
        """Report a non-blocking observation. The agent continues working after this call.

        Categories: scope_violation, design_concern, cleanup_needed,
        dependency_discovered, risk_identified, infra_issue.
        """
        esc = Escalation(
            id=queue.make_id(task_id),
            task_id=task_id,
            agent_role=agent_role,
            severity='info',
            category=category,
            summary=summary,
            detail=detail,
            suggested_action=suggested_action,
            worktree=worktree,
            workflow_state=workflow_state,
        )
        esc_id = queue.submit(esc)
        return {'id': esc_id, 'status': 'queued'}

    @mcp.tool()
    def escalate_blocker(
        task_id: str,
        agent_role: str,
        category: str,
        summary: str,
        detail: str = '',
        suggested_action: str = '',
        worktree: str | None = None,
        workflow_state: str | None = None,
    ) -> dict[str, Any]:
        """Report a blocking problem. After calling this, commit any in-progress work,
        log your iteration, and STOP. Do NOT retry — the handler will resolve the issue
        and you will be re-invoked.

        Categories: scope_violation, design_concern, cleanup_needed,
        dependency_discovered, risk_identified, infra_issue.
        """
        esc = Escalation(
            id=queue.make_id(task_id),
            task_id=task_id,
            agent_role=agent_role,
            severity='blocking',
            category=category,
            summary=summary,
            detail=detail,
            suggested_action=suggested_action,
            worktree=worktree,
            workflow_state=workflow_state,
        )
        esc_id = queue.submit(esc)
        return {'id': esc_id, 'status': 'queued', 'action': 'terminate_cleanly'}

    # --- Handler-side tools ---

    @mcp.tool()
    def resolve_issue(
        escalation_id: str,
        resolution: str,
        terminate: bool = False,
    ) -> dict[str, Any]:
        """Resolve or dismiss an escalation. The resolution text is injected into the
        agent's briefing when the task resumes.

        Set terminate=true to abandon the task rather than resume it.
        """
        esc = queue.resolve(escalation_id, resolution, dismiss=terminate)
        if esc is None:
            return {'error': f'Escalation {escalation_id} not found'}
        return esc.to_dict()

    @mcp.tool()
    def get_pending_escalations(
        task_id: str | None = None,
    ) -> list[dict[str, Any]]:
        """List all pending escalations, optionally filtered by task ID."""
        if task_id:
            escalations = queue.get_by_task(task_id, status='pending')
        else:
            escalations = queue.get_pending()
        return [e.to_dict() for e in escalations]

    @mcp.tool()
    def get_escalation(
        escalation_id: str,
    ) -> dict[str, Any]:
        """Get a single escalation by ID."""
        esc = queue.get(escalation_id)
        if esc is None:
            return {'error': f'Escalation {escalation_id} not found'}
        return esc.to_dict()

    return mcp
