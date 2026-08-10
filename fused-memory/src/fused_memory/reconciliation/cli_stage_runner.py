"""CLI-native stage execution — invokes Claude CLI with MCP tools for each stage."""

from __future__ import annotations

import json
import logging
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from shared.cli_invoke import AgentResult, AllAccountsCappedException, invoke_with_cap_retry

from fused_memory.reconciliation import _RECONCILIATION_STAGE_CAP_WAIT_SANITY_SECS
from fused_memory.reconciliation.sandbox_guard import (
    RemediationSandboxUnavailable,
    resolve_recon_sandbox_wrap,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from shared.config_dir import TaskConfigDir
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
    'mcp__fused-memory__remove_task',
    'mcp__fused-memory__add_dependency',
    'mcp__fused-memory__remove_dependency',
]

# Memory write tools (disallowed in Stage 3 — read-only integrity check)
#
# update_memory (task 3088, esc-3623-3) is an in-place SILENT-REWRITE primitive
# — it changes what a Mem0 record says while preserving its point id and
# created_at — and shipped unclassified: it was in NO disallow list at all, so
# Stage 3 (the read-only integrity check) was neither denied it here nor
# turned away by the mem0_update authz gate, whose shipped default
# content_amend_allowed_agent_prefixes=['recon-stage-'] admits every recon
# stage's f'recon-stage-{stage_id}' agent_id. Stage 1 and Stage 2 KEEP the
# tool because neither STAGE1_DISALLOWED nor STAGE2_DISALLOWED (below) folds
# DISALLOW_MEMORY_WRITES — the very asymmetry the AMEND_AND_EPISODE_TOOLS_BLOCK
# prompt addition (prompts/__init__.py) now advertises.
DISALLOW_MEMORY_WRITES = [
    'mcp__fused-memory__add_episode',
    'mcp__fused-memory__add_memory',
    'mcp__fused-memory__delete_memory',
    'mcp__fused-memory__update_memory',
    'mcp__fused-memory__delete_episode',
    'mcp__fused-memory__redact_episode_content',
    'mcp__fused-memory__replay_to_graphiti',
    'mcp__fused-memory__replay_dead_letters',
    'mcp__fused-memory__refresh_entity_summary',
    'mcp__fused-memory__set_entity_summary',
    'mcp__fused-memory__rename_entity',
    'mcp__fused-memory__merge_entities',
    'mcp__fused-memory__reassign_edge',
    'mcp__fused-memory__delete_entity',
    'mcp__fused-memory__rebuild_entity_summaries',
    'mcp__fused-memory__update_edge',
]

# Recon-report tools that perform DURABLE writes (disallowed outside Stage 2).
# task 2895 β: write_entity_standing_decision is the first recon-report tool that
# writes past in-process ReconReportState — it upserts a row into the durable
# SQLite reconciliation ledger. It must be blocked in Stage 1 and Stage 3 but
# remain callable in Stage 2 (where the entity standing decision is authored), so
# it rides its own additive sublist rather than DISALLOW_MEMORY_WRITES (which
# would miss Stage 1) or DISALLOW_TASK_WRITES (semantically wrong).
DISALLOW_RECON_REPORT_LEDGER_WRITES = [
    'mcp__recon-report__write_entity_standing_decision',
]

# Escalation READ tools (disallowed in every stage — task 3163,
# plans/escalation-store-ambiguity-prd.md task α).
#
# A stage's `escalation` MCP server is wired to the RECONCILIATION escalation
# queue, never to a project's queue: harness.py `_start_escalation_server`
# builds it over `config.escalation_queue_dir` and publishes the resulting URL,
# which stages/base.py injects as the stage's `escalation` server. So a stage
# asking "was an escalation ever filed for task X?" gets [] CATEGORICALLY — the
# per-task record it is looking for lives in a different store and was never in
# this one. That is not a race, and three incidents read the empty result as
# positive proof of absence. Denying the read tools removes the false answer at
# the source; the boundary paragraph in the stage prompts
# (prompts/__init__.py ESCALATION_BOUNDARY_NOTE) explains the absence.
#
# The escalation WRITE tool (escalate_blocker) is deliberately NOT denied: it is
# the sole sanctioned recon escalation use — Stage 2's Stale Flag Escalation
# (FIX D) — and it writes to the reconciliation store, which is the correct
# destination for it. Over-denying here breaks FIX D.
#
# PRD open question 4 (reject-vs-omit) resolved during α: `--disallowed-tools`
# OMITS a denied MCP tool from the agent's tool listing rather than surfacing it
# and rejecting the call. Verified against five live recon-stage transcripts
# whose `deferred_tools_delta` → `addedNames` payload excludes exactly each
# stage's denied names (Stage 3: 80 tools, no builtins / delete_entity /
# submit_task; Stage 1: 93, no submit_task; Stage 2: 100, builtins only). Because
# it is omission and not rejection, the agent gets no explanation for the missing
# tool — which is what makes ESCALATION_BOUNDARY_NOTE load-bearing rather than
# decorative.
# `get_task_escalations` (task 3023) is denied for the SAME reason and is the
# sharpest case of it: `escalation.server.create_server` backs BOTH escalation
# servers — the orchestrator queue (port 8102) and the reconciliation queue
# (port 8103, started by reconciliation/harness.py `_start_escalation_server`)
# — so registering an archive-inclusive per-task lookup there exposes it to
# recon stages pointed at the WRONG store. Its own docstring says an empty
# result IS evidence of absence (true for the queue that server is backed by),
# so a stage calling it against the reconciliation queue would read a
# categorical [] as proof that an orchestrator gate record was never written —
# re-arming the 16-instance false positive this task exists to kill, with more
# authority than the pending-only probe had. The gating here is a DENY list
# only (there is no allow-list; `_run_stage(disallowed_tools=...)`), so a new
# read tool on the shared server is exposed to every stage until it is named
# here — see test_stages.py::test_every_escalation_server_tool_is_classified,
# which fails on any unclassified addition.
#
# `get_task_escalation_history` (task 3164) is the first tool that guard
# actually caught, and it is the same hazard intensified. It DELEGATES to
# `get_task_escalations`, so it inherits the identical archive-inclusive
# cross-store read — but it wraps the result in an envelope that echoes
# `task_id` and `level_filter` back to the caller. Against the reconciliation
# queue that turns a categorical `count: 0` into what reads as an attributable,
# query-specific answer ("no record was ever filed for this task at this
# level"), when it is only the artefact of asking the wrong store. A bare []
# at least looks anonymous; a self-describing zero invites being believed. Both
# per-task archive-inclusive reads therefore sit adjacent below, as one class.
DISALLOW_ESCALATION_READS = [
    'mcp__escalation__get_pending_escalations',
    'mcp__escalation__get_escalation',
    'mcp__escalation__get_task_escalations',
    'mcp__escalation__get_task_escalation_history',
]

# Per-stage disallowed lists
STAGE1_DISALLOWED = (
    DISALLOW_TASK_WRITES
    + DISALLOW_RECON_REPORT_LEDGER_WRITES
    + DISALLOW_ESCALATION_READS
    + DISALLOW_BUILTIN
)
# Stage 2 keeps full memory + task write access; only built-ins and the
# escalation reads are blocked.
STAGE2_DISALLOWED = DISALLOW_ESCALATION_READS + DISALLOW_BUILTIN
STAGE3_DISALLOWED = (
    DISALLOW_TASK_WRITES
    + DISALLOW_MEMORY_WRITES
    + DISALLOW_RECON_REPORT_LEDGER_WRITES
    + DISALLOW_ESCALATION_READS
    + DISALLOW_BUILTIN
)
# NOTE: the IN-PROCESS-STATE `mcp__recon-report__*` tools (start_report, add_finding,
# set_stat, cite_*, complete, …) are intentionally NOT in any disallow list: they write
# only to in-process ReconReportState (not Graphiti / Mem0 / Taskmaster / the ledger), so
# they do not violate Stage 3's read-only contract. Do NOT add those to any disallow list.
# See PRD §9.1 / §11 task γ for the rationale.
# CARVE-OUT (task 2895 β): write_entity_standing_decision is the exception — the first
# recon-report tool with a durable SQLite-ledger write. It IS blocked in Stage 1 and
# Stage 3 (via DISALLOW_RECON_REPORT_LEDGER_WRITES above) and callable only in Stage 2.

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
                'returns None and this finding will not deduplicate across runs. '
                'cited_tasks is the AUTHORITATIVE dedup key: this field\'s shape '
                '(None, a single id, a comma-joined multi-id string, or a '
                'foreign-project id) is normalized to the cited-task set for '
                'both in-run (cite_task) and cross-cycle (compute_flag_signature) '
                'dedup, so it need not exactly match a prior call\'s task_id as '
                'long as the cited tasks agree.'
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
            'description': (
                'Tasks cited as evidence for this finding. AUTHORITATIVE dedup '
                'key (task-2432): in-run (cite_task) and cross-cycle '
                '(compute_flag_signature) dedup normalize the top-level task_id '
                'shape (None / single / comma-joined / foreign) against this '
                'cited-task set, rather than fingerprinting the raw task_id.'
            ),
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


def recon_config_base_dir(data_dir: Path) -> Path:
    """Base dir under which per-run recon CLI config dirs live (task 2744).

    Single source of truth for the config-dir root so the creator
    (``BaseStage.run``) and the GC (``harness``) derive the same path from
    ``(journal.data_dir, run_id)`` without passing a ``TaskConfigDir`` across
    scopes.
    """
    return data_dir / 'recon-config'


def gc_run_config_dir(data_dir: Path, run_id: str) -> None:
    """Remove a run's per-run recon CLI config dir, if present (task 2744).

    ``TaskConfigDir(task_id=run_id, base_dir=recon_config_base_dir(data_dir)).path``
    is ``recon_config_base_dir(data_dir) / f'claude-config-{run_id}'`` by
    construction (config_dir.py's deterministic naming); this rmtrees exactly
    that path. ``ignore_errors=True`` makes it a silent no-op when the dir was
    never created (e.g. a run that short-circuited before any stage launch).
    """
    shutil.rmtree(
        recon_config_base_dir(data_dir) / f'claude-config-{run_id}',
        ignore_errors=True,
    )


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
    session_id: str | None = None,
    config_dir: TaskConfigDir | None = None,
    resume_session_id: str | None = None,
    resume_delivers_prompt: bool = False,
) -> StageResult:
    """Invoke a reconciliation stage via Claude CLI with MCP tools.

    The stage agent has access to fused-memory (and optionally escalation)
    MCP tools, with per-stage restrictions via ``--disallowedTools``.

    ``session_id`` / ``config_dir`` (task 2744) are forwarded straight to
    ``invoke_with_cap_retry``, which writes ``CLAUDE_CONFIG_DIR`` + ``--session-id``
    and activates ``_run_subprocess``'s transcript-liveness watchdog (it only
    engages when BOTH are set). They default to None so existing cutover/sandbox
    call sites keep today's behavior. BaseStage.run mints/persists the session
    before calling here (mint-before-spawn); this runner stays generic.

    ``resume_session_id`` / ``resume_delivers_prompt`` (task 2717 σ) are likewise
    forwarded straight to ``invoke_with_cap_retry`` for the startup
    adopt-and-resume path: when set, the stage subprocess ``--resume``s an
    in-flight session and (with ``resume_delivers_prompt=True``) receives the
    caller's real ``payload`` as its continuation prompt instead of the generic
    CRASH_RECOVERY placeholder. Both default to None/False so non-resume call
    sites are byte-identical. BaseStage.run validates the transcript exists and
    builds the recovery prompt before setting these; this runner stays generic.
    """
    effective_model = model or config.agent_llm_model
    # Task 1989 (sweep verdict): kept at explore_codebase_root, NOT switched to
    # a neutral cwd. This is the GENERIC stage runner — disallowed_tools is a
    # per-call param (not hardcoded ['*']) and stages get fused-memory/
    # escalation MCP tools, so it is not a pure prompt-contained classifier;
    # changing the default cwd here would affect tool-enabled stages.
    effective_cwd = cwd or Path(config.explore_codebase_root)

    start_ms = int(time.monotonic() * 1000)

    # ── Defense-in-depth Landlock confinement (task 1935) ─────────────────────
    # When sandbox_recon_agents=True, resolve a sandbox-wrap callable that
    # confines the entire claude process tree (kernel-level Landlock or bwrap).
    # Fail-CLOSED: if confinement is requested but no backend is available,
    # return an error StageResult WITHOUT calling invoke_with_cap_retry (never
    # run an unconfined agent when confinement is explicitly enabled).
    sandbox_wrap: Callable[[list[str]], list[str]] | None = None
    if config.sandbox_recon_agents:
        try:
            sandbox_wrap = resolve_recon_sandbox_wrap(
                effective_cwd,
                list(config.sandbox_recon_writable_extras),
            )
        except RemediationSandboxUnavailable as exc:
            logger.error(
                'Reconciliation agent NOT launched (fail-closed): %s', exc,
            )
            duration_ms = int(time.monotonic() * 1000) - start_ms
            return StageResult(
                error=str(exc),
                duration_ms=duration_ms,
                model=effective_model,
            )

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
            sandbox_wrap=sandbox_wrap,
            session_id=session_id,
            config_dir=config_dir,
            resume_session_id=resume_session_id,
            resume_delivers_prompt=resume_delivers_prompt,
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
