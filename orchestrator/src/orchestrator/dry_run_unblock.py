"""Autonomous dry-run unblock — fire-and-forget investigation at block time.

Invoked by workflow._mark_blocked when a task transitions to blocked.
Runs the unblock-auto skill read-only, then appends a proposal entry to
metadata.dry_run_proposals[] via scheduler.update_task(append=True).
Never mutates task state; all writes go through the parent process.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from orchestrator.agents.invoke import invoke_agent  # noqa: E402

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Schema and constants
# ---------------------------------------------------------------------------

DRY_RUN_PROPOSAL_SCHEMA: dict[str, Any] = {
    'type': 'object',
    'required': ['proposal_text', 'risk_label', 'files_referenced'],
    'properties': {
        'proposal_text': {
            'type': 'string',
            'description': 'Concrete description of the proposed action and root cause.',
        },
        'risk_label': {
            'type': 'string',
            'enum': ['low', 'medium', 'human-review-required'],
        },
        'files_referenced': {
            'type': 'array',
            'items': {'type': 'string'},
            'description': 'Files that the proposed action would need to modify.',
        },
    },
    'additionalProperties': False,
}

_RISK_LABELS: frozenset[str] = frozenset({'low', 'medium', 'human-review-required'})
_HUMAN_REVIEW_REQUIRED: str = 'human-review-required'

# Read-only tools the dry-run agent is allowed to use
_ALLOWED_TOOLS: list[str] = [
    'Read',
    'Glob',
    'Grep',
    'Bash(git:*)',
    'Bash(cargo:*)',
    'Bash(pytest:*)',
    'mcp__fused-memory__search',
    'mcp__fused-memory__get_task',
    'mcp__fused-memory__get_tasks',
]

# Mutating tools explicitly blocked
_DISALLOWED_TOOLS: list[str] = [
    'Edit',
    'Write',
    'mcp__fused-memory__set_task_status',
    'mcp__fused-memory__update_task',
    'mcp__fused-memory__delete_memory',
    'mcp__fused-memory__remove_task',
]

# ---------------------------------------------------------------------------
# SKILL.md loader
# ---------------------------------------------------------------------------

def _load_skill_system_prompt() -> str:
    """Load skills/unblock-auto/SKILL.md from repo root, strip frontmatter."""
    skill_path = Path(__file__).parent.parent.parent.parent.parent / 'skills' / 'unblock-auto' / 'SKILL.md'
    if not skill_path.exists():
        # Fallback: search upward for the skills directory
        here = Path(__file__).resolve()
        for parent in here.parents:
            candidate = parent / 'skills' / 'unblock-auto' / 'SKILL.md'
            if candidate.exists():
                skill_path = candidate
                break

    raw = skill_path.read_text()
    if not raw.startswith('---'):
        return raw
    parts = raw.split('---', maxsplit=2)
    return parts[2].strip() if len(parts) >= 3 else raw


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

async def run_dry_run_unblock(
    *,
    task_id: str,
    worktree: str,
    reason: str,
    detail: str,
    scheduler: Any,
    mcp: Any,
    config: Any,
    event_store: Any = None,
    usage_gate: Any = None,
) -> None:
    """Investigate a blocked task and write a proposal to metadata.

    Fire-and-forget: called via asyncio.create_task from _mark_blocked.
    Any exception is caught and written as a fallback entry so failures
    are always visible in metadata.dry_run_proposals[].
    """
    import time

    ua_cfg = config.unblock_auto
    start_time = time.monotonic()

    try:
        system_prompt = _load_skill_system_prompt()
        user_prompt = (
            f'Task ID: {task_id}\n'
            f'Worktree: {worktree}\n'
            f'Block reason: {reason}\n'
            f'Detail: {detail or "(none)"}\n\n'
            'Investigate and emit your structured proposal.'
        )

        mcp_config = mcp.mcp_config_json() if mcp is not None else None

        result = await invoke_agent(
            prompt=user_prompt,
            system_prompt=system_prompt,
            cwd=Path(worktree),
            model=ua_cfg.model,
            max_turns=ua_cfg.max_turns,
            max_budget_usd=ua_cfg.budget_usd,
            allowed_tools=_ALLOWED_TOOLS,
            disallowed_tools=_DISALLOWED_TOOLS,
            mcp_config=mcp_config,
            output_schema=DRY_RUN_PROPOSAL_SCHEMA,
            effort=ua_cfg.effort,
            backend=ua_cfg.backend,
            timeout_seconds=ua_cfg.timeout_seconds,
        )

        entry = _build_entry(result, reason=reason, budget_usd=ua_cfg.budget_usd)

    except Exception as exc:
        logger.warning('dry_run_unblock: unexpected error for task %s: %s', task_id, exc)
        entry = {
            'status': 'investigation_failed',
            'proposal_text': f'Investigation failed (unexpected error): {exc}',
            'risk_label': _HUMAN_REVIEW_REQUIRED,
            'files_referenced': [],
            'block_reason': reason,
            'investigated_at': _now_iso(),
            'timestamp': _now_iso(),
        }
        result = None

    try:
        await scheduler.update_task(
            task_id,
            {'dry_run_proposals': [entry]},
            append=True,
        )
    except Exception as exc:
        logger.error('dry_run_unblock: failed to persist proposal for task %s: %s', task_id, exc)

    if event_store and result is not None:
        try:
            from orchestrator.event_store import EventType
            event_store.emit(
                EventType.invocation_end,
                task_id=task_id,
                phase='blocked',
                role='unblock_auto',
                cost_usd=getattr(result, 'cost_usd', 0.0),
                duration_ms=getattr(result, 'duration_ms', int((time.monotonic() - start_time) * 1000)),
                data={
                    'dry_run': True,
                    'risk_label': entry.get('risk_label', _HUMAN_REVIEW_REQUIRED),
                    'success': getattr(result, 'success', False),
                    'status': entry.get('status', 'ok'),
                },
            )
        except Exception as exc:
            logger.warning('dry_run_unblock: failed to emit event for task %s: %s', task_id, exc)


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _build_entry(result: Any, *, reason: str, budget_usd: float) -> dict[str, Any]:
    """Convert an AgentResult to a proposal entry dict."""
    now = _now_iso()

    if not result.success:
        if _is_budget_exhausted(result, budget_usd):
            return {
                'status': 'budget_exhausted',
                'proposal_text': (
                    f'Budget exhausted, proposal incomplete. '
                    f'cost_usd={result.cost_usd:.3f}, subtype={result.subtype}'
                ),
                'risk_label': _HUMAN_REVIEW_REQUIRED,
                'files_referenced': [],
                'block_reason': reason,
                'cost_usd': result.cost_usd,
                'investigated_at': now,
                'timestamp': now,
            }
        return {
            'status': 'investigation_failed',
            'proposal_text': (
                f'Investigation failed: subtype={result.subtype}; '
                f'output={result.output[:200]}'
            ),
            'risk_label': _HUMAN_REVIEW_REQUIRED,
            'files_referenced': [],
            'block_reason': reason,
            'investigated_at': now,
            'timestamp': now,
        }

    # Success path — parse structured_output
    raw = result.structured_output or {}
    risk = raw.get('risk_label', '')
    if risk not in _RISK_LABELS:
        risk = _HUMAN_REVIEW_REQUIRED

    return {
        'proposal_text': raw.get('proposal_text', ''),
        'risk_label': risk,
        'files_referenced': raw.get('files_referenced', []),
        'block_reason': reason,
        'investigated_at': now,
        'timestamp': now,
    }


def _is_budget_exhausted(result: Any, budget_usd: float) -> bool:
    subtype = getattr(result, 'subtype', '') or ''
    cost = getattr(result, 'cost_usd', 0.0)
    return 'budget' in subtype.lower() or cost >= budget_usd
