"""CLI-native stage execution — invokes Claude CLI with MCP tools for each stage."""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from shared.cli_invoke import AgentResult, invoke_with_cap_retry

if TYPE_CHECKING:
    from shared.usage_gate import UsageGate

    from fused_memory.config.schema import ReconciliationConfig

logger = logging.getLogger(__name__)

# ── Disallowed tool lists ──────────────────────────────────────────────

# Non-MCP built-in tools that stages should never use
DISALLOW_BUILTIN = ['Bash', 'Edit', 'Write', 'NotebookEdit']

# Task write tools (disallowed in Stage 1 — memory consolidation only)
DISALLOW_TASK_WRITES = [
    'mcp__fused-memory__set_task_status',
    'mcp__fused-memory__add_task',
    'mcp__fused-memory__update_task',
    'mcp__fused-memory__add_subtask',
    'mcp__fused-memory__remove_task',
    'mcp__fused-memory__add_dependency',
    'mcp__fused-memory__remove_dependency',
    'mcp__fused-memory__expand_task',
    'mcp__fused-memory__parse_prd',
]

# Memory write tools (disallowed in Stage 3 — read-only integrity check)
DISALLOW_MEMORY_WRITES = [
    'mcp__fused-memory__add_episode',
    'mcp__fused-memory__add_memory',
    'mcp__fused-memory__delete_memory',
    'mcp__fused-memory__delete_episode',
    'mcp__fused-memory__replay_to_graphiti',
    'mcp__fused-memory__replay_dead_letters',
]

# Per-stage disallowed lists
STAGE1_DISALLOWED = DISALLOW_TASK_WRITES + DISALLOW_BUILTIN
STAGE2_DISALLOWED = DISALLOW_BUILTIN  # Full access to memory + task tools
STAGE3_DISALLOWED = DISALLOW_TASK_WRITES + DISALLOW_MEMORY_WRITES + DISALLOW_BUILTIN

# Output schema for stage reports
STAGE_REPORT_SCHEMA: dict[str, Any] = {
    'type': 'object',
    'properties': {
        'flagged_items': {
            'type': 'array',
            'items': {'type': 'object'},
            'description': 'Items flagged for the next stage or next cycle',
        },
        'stats': {
            'type': 'object',
            'description': 'Counts and metrics from this stage',
        },
        'summary': {
            'type': 'string',
            'description': 'Human-readable summary of what was done',
        },
    },
    'required': ['summary'],
}


@dataclass
class StageResult:
    """Result from a single CLI stage invocation."""

    report: dict = field(default_factory=dict)
    llm_calls: int = 0
    tokens_used: int = 0
    cost_usd: float = 0.0
    duration_ms: int = 0
    model: str = ''
    success: bool = False
    error: str = ''


async def run_stage_via_cli(
    system_prompt: str,
    payload: str,
    disallowed_tools: list[str],
    config: ReconciliationConfig,
    mcp_config: dict,
    usage_gate: UsageGate | None = None,
    model: str | None = None,
    cwd: Path | None = None,
) -> StageResult:
    """Invoke a reconciliation stage via Claude CLI with MCP tools.

    The stage agent has access to fused-memory (and optionally escalation)
    MCP tools, with per-stage restrictions via ``--disallowedTools``.
    """
    effective_model = model or config.agent_llm_model
    effective_cwd = cwd or Path(config.explore_codebase_root)

    start_ms = int(time.monotonic() * 1000)

    try:
        agent_result: AgentResult = await invoke_with_cap_retry(
            usage_gate=usage_gate,
            label=f'Reconciliation stage ({effective_model})',
            prompt=payload,
            system_prompt=system_prompt,
            cwd=effective_cwd,
            model=effective_model,
            max_turns=config.agent_max_steps,
            max_budget_usd=5.0,
            disallowed_tools=disallowed_tools,
            mcp_config=mcp_config,
            output_schema=STAGE_REPORT_SCHEMA,
            permission_mode='bypassPermissions',
            timeout_seconds=float(config.stage_timeout_seconds),
        )
    except Exception as e:
        duration_ms = int(time.monotonic() * 1000) - start_ms
        return StageResult(
            error=str(e),
            duration_ms=duration_ms,
            model=effective_model,
        )

    duration_ms = int(time.monotonic() * 1000) - start_ms

    # Parse report from structured output or raw text
    report = _extract_report(agent_result)

    return StageResult(
        report=report,
        llm_calls=agent_result.turns,
        tokens_used=0,  # CLI doesn't expose token counts
        cost_usd=agent_result.cost_usd,
        duration_ms=duration_ms,
        model=effective_model,
        success=agent_result.success,
        error='' if agent_result.success else (agent_result.output or 'Agent failed'),
    )


def _extract_report(result: AgentResult) -> dict:
    """Extract the stage report from agent output, with fallback parsing."""
    # Prefer structured output (from --json-schema)
    if result.structured_output:
        if isinstance(result.structured_output, dict):
            return result.structured_output
        if isinstance(result.structured_output, str):
            try:
                return json.loads(result.structured_output)
            except json.JSONDecodeError:
                pass

    # Try parsing the full output as JSON
    if result.output:
        try:
            parsed = json.loads(result.output)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            pass

        # Fallback: wrap raw text as summary
        return {'summary': result.output[:2000]}

    return {'summary': 'No output from agent'}
