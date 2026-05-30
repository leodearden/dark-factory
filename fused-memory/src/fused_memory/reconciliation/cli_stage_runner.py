"""CLI-native stage execution — invokes Claude CLI with MCP tools for each stage."""

from __future__ import annotations

import json
import logging
import secrets
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from shared.cli_invoke import AgentResult, AllAccountsCappedException, invoke_with_cap_retry

from fused_memory.reconciliation import _RECONCILIATION_STAGE_CAP_WAIT_SANITY_SECS

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
    'mcp__fused-memory__submit_task',
    'mcp__fused-memory__resolve_ticket',
    'mcp__fused-memory__update_task',
    'mcp__fused-memory__add_subtask',
    'mcp__fused-memory__remove_task',
    'mcp__fused-memory__add_dependency',
    'mcp__fused-memory__remove_dependency',
]

# Memory write tools (disallowed in Stage 3 — read-only integrity check)
DISALLOW_MEMORY_WRITES = [
    'mcp__fused-memory__add_episode',
    'mcp__fused-memory__add_memory',
    'mcp__fused-memory__delete_memory',
    'mcp__fused-memory__delete_episode',
    'mcp__fused-memory__replay_to_graphiti',
    'mcp__fused-memory__replay_dead_letters',
    'mcp__fused-memory__refresh_entity_summary',
    'mcp__fused-memory__merge_entities',
    'mcp__fused-memory__rebuild_entity_summaries',
    'mcp__fused-memory__update_edge',
]

# Subtask creation (disallowed in Stage 2 — orchestrator scheduler is top-level-only;
# new subtasks would be silently orphaned and never dispatched, see procedural memory fca61c20)
DISALLOW_SUBTASK_CREATE = ['mcp__fused-memory__add_subtask']

# Per-stage disallowed lists
STAGE1_DISALLOWED = DISALLOW_TASK_WRITES + DISALLOW_BUILTIN
STAGE2_DISALLOWED = DISALLOW_BUILTIN + DISALLOW_SUBTASK_CREATE  # Full memory + task tools EXCEPT add_subtask (see DISALLOW_SUBTASK_CREATE)
STAGE3_DISALLOWED = DISALLOW_TASK_WRITES + DISALLOW_MEMORY_WRITES + DISALLOW_BUILTIN
# NOTE: `mcp__recon-report__*` tools are intentionally NOT included in DISALLOW_MEMORY_WRITES
# or DISALLOW_TASK_WRITES and therefore NOT in STAGE3_DISALLOWED.  These tools write only
# to in-process ReconReportState (not Graphiti / Mem0 / Taskmaster) so they do not violate
# Stage 3's read-only contract.  Do NOT add them to either disallow list.
# See PRD §9.1 / §11 task γ for the rationale.

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

# Structured schema for individual finding items (Stage 3)
# PRD §9.3: four typed citation lists replace the retired `affected_ids` field.
# Top-level task_id / flag_type / actionable are preserved for flag_dedup and
# stats_verifier compatibility.
#
# NOTE — ASSEMBLED REPORT SHAPE (suggestion schema_prompt_mismatch):
# This schema describes the shape of data in the ASSEMBLED report (after the
# recon_report server has resolved citations).  It is NOT the shape the LLM
# emits: the stage prompts instruct the agent to call the cite_* MCP tools
# (e.g. cite_entity(name=...)); the server resolves UUIDs and fingerprints
# server-side.  Fields marked "server-resolved" below are therefore optional
# in the LLM output but present in the assembled dict.
FINDING_ITEM_SCHEMA: dict[str, Any] = {
    'type': 'object',
    'properties': {
        'description': {
            'type': 'string',
            'description': 'What the inconsistency is, with specific IDs and evidence',
        },
        'severity': {
            'type': 'string',
            'enum': ['minor', 'moderate', 'serious'],
            'description': 'Severity level of the finding',
        },
        'actionable': {
            'type': 'boolean',
            'description': 'True if Stage 1/Stage 2 can fix it automatically',
        },
        'task_id': {
            'type': ['string', 'null'],
            'description': (
                'Task ID associated with this finding. '
                'Dedup contract: either set task_id + flag_type here, OR include '
                'at least one cited_tasks entry — otherwise compute_flag_signature '
                'returns None and this finding will not deduplicate across runs.'
            ),
        },
        'flag_type': {
            'type': ['string', 'null'],
            'description': (
                'Machine-readable flag type for deduplication. '
                'Required alongside task_id (or cited_tasks fallback) for '
                'cross-run recurrence tracking to work correctly.'
            ),
        },
        'category': {
            'type': 'string',
            'enum': [
                'memory_stale',
                'memory_duplicate',
                'memory_contradiction',
                'task_memory_mismatch',
                'missing_knowledge',
                'cross_store_inconsistency',
                'systemic_pattern',
                'cross_project_routing',
                'other',
            ],
            'description': 'Category of the finding',
        },
        'suggested_action': {
            'type': 'string',
            'description': 'What the remediation stage should do to fix this finding',
        },
        # PRD §9.3 typed citation lists — replace retired `affected_ids`
        'cited_entities': {
            'type': 'array',
            'description': 'Entities cited as evidence for this finding',
            'items': {
                'type': 'object',
                'properties': {
                    # entity_uuid is server-resolved: the agent calls cite_entity(name=...)
                    # and the server looks up the UUID.  Present in assembled output, but
                    # the LLM never produces it directly — do NOT add to required.
                    'entity_uuid': {'type': 'string', 'description': 'UUID of the entity node (server-resolved)'},
                    'canonical_name': {'type': 'string', 'description': 'Entity canonical name'},
                },
                'required': ['canonical_name'],
            },
        },
        'cited_edges': {
            'type': 'array',
            'description': 'Graphiti edges cited as evidence for this finding',
            'items': {
                'type': 'object',
                'properties': {
                    'edge_uuid': {'type': 'string', 'description': 'UUID of the edge'},
                    'fact_text_snapshot': {'type': 'string', 'description': 'Snapshot of the edge fact text'},
                },
                'required': ['edge_uuid', 'fact_text_snapshot'],
            },
        },
        'cited_tasks': {
            'type': 'array',
            'description': 'Tasks cited as evidence for this finding',
            'items': {
                'type': 'object',
                'properties': {
                    'project_id': {'type': 'string', 'description': 'Project ID containing the task'},
                    'task_id': {'type': 'string', 'description': 'Task ID'},
                    'title': {'type': 'string', 'description': 'Task title'},
                },
                'required': ['project_id', 'task_id', 'title'],
            },
        },
        'cited_memories': {
            'type': 'array',
            'description': 'Mem0 memory entries cited as evidence for this finding',
            'items': {
                'type': 'object',
                'properties': {
                    'memory_id': {'type': 'string', 'description': 'Memory ID'},
                    'store': {'type': 'string', 'description': 'Store type (mem0/graphiti)'},
                    # metadata_fingerprint is server-derived — present in assembled output
                    # but not emitted by the LLM (agent calls cite_memory(memory_id, store)).
                    'metadata_fingerprint': {'type': 'string', 'description': 'Fingerprint of memory metadata (server-derived)'},
                },
                'required': ['memory_id', 'store'],
            },
        },
    },
    'required': ['description', 'severity'],
}

# Stage 3-specific report schema with structured finding items
STAGE3_REPORT_SCHEMA: dict[str, Any] = {
    'type': 'object',
    'properties': {
        'flagged_items': {
            'type': 'array',
            'items': FINDING_ITEM_SCHEMA,
            'description': 'Findings from the integrity check',
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


def generate_summary_nonce() -> str:
    """Return an 8-char lowercase-hex nonce for per-cycle summary uniqueness.

    Generated from Python's CSPRNG (secrets module) OUTSIDE the LLM so the
    token carries genuine entropy regardless of the LLM's output distribution.
    Injected into the Stage 2 payload once per cycle (in assemble_payload) so
    the Stage 2 agent can prepend it to its per-cycle summary content.

    Rationale (task 1572): ISO 8601 uniqueness_token alone (tasks 1473/1488)
    produced insufficient cosine-distance separation — consecutive timestamps
    embed nearly identically in Mem0's embedding space, causing per-cycle
    summaries to be silently deduplicated (memory_ids=[]) in 8+ confirmed
    recurrences.  A 32-bit random hex prefix shifts the leading content far
    enough to defeat the ~0.92 cosine-similarity dedup threshold.

    Returns:
        8-character lowercase hex string (e.g. 'a3f7c21b'), i.e.
        secrets.token_hex(4).
    """
    return secrets.token_hex(4)


def build_summary_nonce_section() -> str:
    """Return the full '### Per-Cycle Summary Nonce' payload section string.

    Shared by Stage 1 (memory_consolidator) and Stage 2 (task_knowledge_sync)
    so both stages produce identical section wording and cannot silently drift.
    Generates a fresh CSPRNG nonce on each call via generate_summary_nonce().

    Returns:
        Multi-line string ready to embed in a payload f-string, e.g.::

            \\n### Per-Cycle Summary Nonce\\n
            summary_nonce: a3f7c21b\\n
            (Prepend this nonce as the FIRST line of your per-cycle summary content.)\\n
    """
    return (
        f'\n### Per-Cycle Summary Nonce\n'
        f'summary_nonce: {generate_summary_nonce()}\n'
        f'(Prepend this nonce as the FIRST line of your per-cycle summary content.)\n'
    )


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
    output_schema: dict | None = None,
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
            output_schema=output_schema,  # PRD γ: pass None straight through when caller opts out
            permission_mode='bypassPermissions',
            timeout_seconds=float(config.stage_timeout_seconds),
            cap_wait_sanity_secs=_RECONCILIATION_STAGE_CAP_WAIT_SANITY_SECS,
        )
    except AllAccountsCappedException:
        logger.warning(
            'Stage %s: all accounts capped — propagating to harness for deferral',
            effective_model,
        )
        raise
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


def _normalize_report(report: dict) -> dict:
    """Normalize a raw stage report dict.

    Handles two key remapping cases:
    1. LLM used 'findings' instead of 'flagged_items': remap.
    2. Both keys present but 'flagged_items' is empty: use 'findings' as fallback.

    Also filters placeholder findings (where description is missing or '?').
    """
    findings = report.get('findings')
    flagged = report.get('flagged_items')

    if findings is not None and (flagged is None or (isinstance(flagged, list) and len(flagged) == 0)):
        # Remap findings → flagged_items
        report = dict(report)
        report['flagged_items'] = findings
        report.pop('findings', None)

    # Filter placeholder findings
    raw_items = report.get('flagged_items', [])
    if isinstance(raw_items, list):
        filtered = [
            item for item in raw_items
            if isinstance(item, dict)
            and item.get('description')
            and item['description'] != '?'
        ]
        n_removed = len(raw_items) - len(filtered)
        if n_removed > 0:
            logger.warning('Filtered %d placeholder finding(s) from stage report', n_removed)
        report = dict(report)
        report['flagged_items'] = filtered

    return report


def _extract_report(result: AgentResult) -> dict:
    """Extract the stage report from agent output, with fallback parsing."""
    # Prefer structured output (from --json-schema)
    if result.structured_output:
        if isinstance(result.structured_output, dict):
            return _normalize_report(result.structured_output)
        if isinstance(result.structured_output, str):
            try:
                return _normalize_report(json.loads(result.structured_output))
            except json.JSONDecodeError:
                pass

    # Try parsing the full output as JSON
    if result.output:
        try:
            parsed = json.loads(result.output)
            if isinstance(parsed, dict):
                return _normalize_report(parsed)
        except json.JSONDecodeError:
            pass

        # Fallback: wrap raw text as summary
        return {'summary': result.output[:2000]}

    # PRD γ: return {} instead of a placeholder so BaseStage.run's recon_report
    # fallback path (empty assembled dict) triggers cleanly rather than masking
    # a missing report with a fake 'No output' summary.
    # Behavior-change note (reviewer finding behavior_change): prior code returned
    # {'summary': 'No output from agent'} here.  All callers use report.get('summary')
    # (never report['summary']), so {} is safe.  Non-recon callers that depend on a
    # non-empty summary string should pass output_schema=STAGE_REPORT_SCHEMA so the
    # LLM-emitted JSON path fires before this fallback is reached.
    return {}
