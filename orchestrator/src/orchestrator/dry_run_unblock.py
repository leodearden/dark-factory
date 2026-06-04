"""Autonomous dry-run unblock — fire-and-forget investigation at block time.

Invoked by workflow._mark_blocked when a task transitions to blocked.
Runs the unblock-auto skill read-only, then appends a proposal entry to
metadata.dry_run_proposals[] via scheduler.update_task(append=True).
Never mutates task state; all writes go through the parent process.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from orchestrator.agents.invoke import invoke_agent  # noqa: E402
from orchestrator.agents.skill_prompt import load_skill_system_prompt

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

# Stable subtype values emitted by the CLI when the cost cap fires.
# Mirrors the canonical value used in shared/src/shared/usage_gate.py and
# fused-memory/src/fused_memory/middleware/task_curator.py — the Claude CLI
# emits ``error_max_budget_usd`` (with the ``_usd`` suffix) on cost-cap
# exhaustion, not ``error_max_budget``.
_BUDGET_SUBTYPES: frozenset[str] = frozenset({'error_max_budget_usd'})


def _is_budget_exhausted(result: Any, budget_usd: float) -> bool:  # noqa: ARG001
    """Return True only when the agent's subtype explicitly signals budget exhaustion.

    We avoid the ``cost >= budget_usd`` heuristic: an agent can spend close to
    the cap before failing for an unrelated reason (e.g. max_turns, tool error),
    which would mis-classify those failures as 'budget_exhausted' and produce
    misleading operator-visible status fields in metadata.
    """
    subtype = getattr(result, 'subtype', '') or ''
    return subtype in _BUDGET_SUBTYPES


# Read-only tools the dry-run agent is allowed to use.
# Bash(pytest:*) and Bash(cargo:*) are intentionally omitted: both can
# write .pyc/__pycache__/target/ files or fetch from the network, which
# contradicts the read-only safety contract advertised in SKILL.md.
# The agent can read existing test output from .task/iterations.jsonl instead.
# Explicit read-only git subcommands instead of a wildcard — Bash(git:*) would
# permit mutating subcommands (commit, push, reset, checkout, restore, clean)
# which contradict the read-only contract for this autonomous, no-human-
# checkpoint invocation where the threat profile is higher than interactive use.
_ALLOWED_TOOLS: list[str] = [
    'Read',
    'Glob',
    'Grep',
    'Bash(git log:*)',
    'Bash(git diff:*)',
    'Bash(git status:*)',
    'Bash(git show:*)',
    'Bash(git rev-parse:*)',
    'Bash(git branch:*)',
    'Bash(git ls-files:*)',
    'mcp__fused-memory__search',
    'mcp__fused-memory__get_task',
    'mcp__fused-memory__get_tasks',
]

# Mutating tools explicitly blocked.
# Defence-in-depth: list the most dangerous git mutations alongside the
# write-capable MCP tools so the allowlist and denylist are consistent.
_DISALLOWED_TOOLS: list[str] = [
    'Edit',
    'Write',
    'Bash(git commit:*)',
    'Bash(git push:*)',
    'Bash(git reset:*)',
    'Bash(git checkout:*)',
    'Bash(git restore:*)',
    'Bash(git clean:*)',
    'Bash(git merge:*)',
    'Bash(git rebase:*)',
    'mcp__fused-memory__set_task_status',
    'mcp__fused-memory__update_task',
    'mcp__fused-memory__delete_memory',
    'mcp__fused-memory__remove_task',
]

# ---------------------------------------------------------------------------
# SHA capture helper
# ---------------------------------------------------------------------------

async def _capture_worktree_shas(worktree: str) -> tuple[str | None, str | None]:
    """Capture HEAD and main branch SHAs from the worktree via git.

    Fully defensive: any non-zero exit code, subprocess exception, or
    non-repo worktree yields None for that sha — never raises.
    Both shas are always returned (keys always present in the stamped entry)
    so the entry shape stays consistent across repo and non-repo worktrees.
    """
    async def _rev_parse(ref: str) -> str | None:
        try:
            proc = await asyncio.create_subprocess_exec(
                'git', '-C', worktree, 'rev-parse', ref,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, _ = await proc.communicate()
            if proc.returncode != 0:
                return None
            return stdout.decode().strip() or None
        except Exception:
            return None

    head_sha, main_sha = await asyncio.gather(_rev_parse('HEAD'), _rev_parse('main'))
    return head_sha, main_sha


# ---------------------------------------------------------------------------
# SKILL.md loader
# ---------------------------------------------------------------------------

def _load_skill_system_prompt() -> str:
    """Load skills/unblock-auto/SKILL.md from repo root, strip frontmatter.

    Delegates to the shared ``load_skill_system_prompt`` helper so the
    parent-walk and frontmatter-strip logic is not duplicated here.
    Raises FileNotFoundError with a clear message if the file cannot be found.
    """
    return load_skill_system_prompt('unblock-auto')


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
) -> None:
    """Investigate a blocked task and write a proposal to metadata.

    Fire-and-forget: called via asyncio.create_task from _mark_blocked.
    Any exception is caught and written as a fallback entry so failures
    are always visible in metadata.dry_run_proposals[].
    """
    import time

    ua_cfg = config.unblock_auto
    start_time = time.monotonic()

    # Capture worktree SHAs before invoke_agent so both the success and
    # exception paths can stamp the same anchor values.
    head_sha, main_sha = await _capture_worktree_shas(worktree)

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
            'cost_usd': 0.0,
            'investigated_at': _now_iso(),
            'timestamp': _now_iso(),
        }
        result = None

    # Stamp the git anchor onto the entry at a single point so all four
    # shapes (ok, investigation_failed, budget_exhausted, exception fallback)
    # carry head_sha/main_sha. DRY_RUN_PROPOSAL_SCHEMA stays unchanged
    # (additionalProperties:False, no sha keys) so the agent cannot forge these.
    entry['head_sha'] = head_sha
    entry['main_sha'] = main_sha

    try:
        await scheduler.update_task(
            task_id,
            {'dry_run_proposals': [entry]},
            append=True,
        )
    except Exception as exc:
        logger.error('dry_run_unblock: failed to persist proposal for task %s: %s', task_id, exc)

    # Best-effort keep-last-N trim: read the full metadata blob, slice
    # dry_run_proposals to the most recent keep_last entries, rewrite the
    # whole blob (append=False preserves sibling keys like memory_hints/files).
    # Wrapped in its own try/except — a trim failure never crashes the hook.
    # keep_last <= 0 disables trimming.
    # Note: existing MagicMock-scheduler tests stay green because their
    # scheduler.get_task is non-awaitable — raises TypeError, is caught here,
    # and only the single append=True call persists.
    #
    # Concurrency note: this read-modify-write is not atomic. If another writer
    # mutates the task metadata between get_task and update_task(append=False),
    # that concurrent change is clobbered (stale-plus-trimmed blob wins).
    # Concurrent dry-runs on the same blocked task are extremely unlikely — the
    # scheduler dispatches at most one hook per task — so this is acceptable as
    # a best-effort trim. A missed trim cycle is preferable to coordination
    # overhead.
    try:
        keep_last = ua_cfg.b3_proposal_keep_last
        if keep_last and keep_last > 0:
            task = await scheduler.get_task(task_id)
            if task:
                metadata = task.get('metadata') or {}
                proposals = metadata.get('dry_run_proposals') or []
                if len(proposals) > keep_last:
                    metadata['dry_run_proposals'] = proposals[-keep_last:]
                    await scheduler.update_task(task_id, metadata, append=False)
    except Exception as exc:
        logger.warning(
            'dry_run_unblock: trim failed for task %s (best-effort, continuing): %s',
            task_id, exc,
        )

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
            'cost_usd': getattr(result, 'cost_usd', 0.0),
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
