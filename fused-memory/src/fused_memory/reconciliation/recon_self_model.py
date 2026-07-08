"""Single-source-of-truth self-model for recon's control-plane mechanisms
(task 2220, W5-β, PRD plans/recon-reliability-prd.md §8.4, stream W5
foundations phase).

This module is the canonical description of recon's control-plane
mechanics: the record_kind vocabulary and marker lifecycle (§8.1), the
fingerprint-identity fields the escalation/dedup layer reads, the
execution_class contract (§8.5), rendered prompt sections describing these
mechanisms, and a premise-lint that flags task descriptions containing
known-false premises about them.

FOUNDATIONS-FIRST / import-light: the stage prompts
(reconciliation/prompts/stage1.py, stage2.py) will import this module for
its rendered sections at a later task (ξ). Those prompts currently import
nothing from harness.py/flag_dedup.py/recon_ledger.py, so — to stay safely
importable from the prompt-import path without pulling in aiosqlite
(recon_ledger's dependency) or other reconciliation internals — this module
imports ONLY reconciliation.recon_pool_map (itself a leaf — see that
module's docstring) plus stdlib. Consistency with recon_ledger.MARKER_KINDS,
harness._derive_affected_ids, and flag_dedup's content-fingerprint fallback
is cross-checked by tests (fused-memory/tests/test_recon_self_model.py),
which may import those modules freely.

This task builds ONLY this module + its unit tests. The prompt cutover
(stage1.py/stage2.py importing these rendered sections) and the
premise-lint wiring at the recon submit path are task ξ; this module
therefore does not modify, and is not yet imported by, any prompt file.
"""

from __future__ import annotations

from dataclasses import dataclass

from fused_memory.reconciliation.recon_pool_map import (
    CYCLE_SUMMARY_STAGE_TO_RECON_POOL,
    STAGE1_CYCLE_SUMMARY_RECON_POOL,
    STAGE2_CYCLE_SUMMARY_RECON_POOL,
)

# --------------------------------------------------------------------------- #
# §8.1 record_kind vocabulary
# --------------------------------------------------------------------------- #

# Full record_kind vocabulary for the recon_ledger control plane (PRD §8.1).
# Broader than recon_ledger.MARKER_KINDS, which is only the per-task-marker
# subset gc() deletes when a task goes terminal (stage1_flag_suppression and
# cycle_summary are excluded there — see recon_ledger.py's module comment).
# test_recon_self_model.py asserts the GC-on-terminal subset of
# MARKER_LIFECYCLE (below) equals recon_ledger.MARKER_KINDS exactly, so
# drift between the two constants fails a test rather than silently
# diverging.
MARKER_KINDS = (
    'stage1_flag_marker',
    'stage1_flag_suppression',
    'stage2_persistence_marker',
    'flag_for_stage2',
    'cycle_summary',
)

# --------------------------------------------------------------------------- #
# Marker lifecycle: who writes each record_kind, and who deletes it
# --------------------------------------------------------------------------- #

# Deleter sentinels — see MarkerLifecycle.deleter. These are descriptive
# tags (not machine identifiers) naming the mechanism that removes a record.
DELETER_GC = 'gc-on-terminal-or-ttl'
DELETER_POOL_TRIM = 'cycle-summary-pool-cap-trim-or-ttl'
DELETER_TTL = 'ttl-expiry'


@dataclass(frozen=True)
class MarkerLifecycle:
    """Who writes a record_kind, and what deletes it.

    ``writer``/``deleter`` are short human-readable descriptions documenting
    the lifecycle contract for a record_kind — this is not a runtime
    dispatch table.
    """

    writer: str
    deleter: str


# One entry per MARKER_KINDS value. The GC-on-terminal subset (deleter ==
# DELETER_GC) is exactly recon_ledger.MARKER_KINDS — the per-task marker
# kinds ReconLedgerStore.gc() deletes once their task_id goes terminal.
# stage1_flag_suppression and cycle_summary are NOT per-task markers, so
# they expire via TTL / pool-cap trim instead (see recon_ledger.py:62-66 and
# recon_pool_map.py's docstring for the corresponding cycle_summary pool-cap
# trim mechanism). test_recon_self_model.py cross-checks this subset against
# recon_ledger.MARKER_KINDS to catch drift between the two modules.
MARKER_LIFECYCLE: dict[str, MarkerLifecycle] = {
    'stage1_flag_marker': MarkerLifecycle(
        writer=(
            'Stage 1 flag_dedup post-processor (dedup_flags), one marker per '
            '(task_id, flag_type)'
        ),
        deleter=DELETER_GC,
    ),
    'flag_for_stage2': MarkerLifecycle(
        writer=(
            'Stage 1 flag_dedup post-processor, for flagged items carrying '
            'metadata.flag_for_stage2=true'
        ),
        deleter=DELETER_GC,
    ),
    'stage2_persistence_marker': MarkerLifecycle(
        writer='Stage 2 / TaskKnowledgeSync post-processor',
        deleter=DELETER_GC,
    ),
    'stage1_flag_suppression': MarkerLifecycle(
        writer=(
            'Operators / remediation hooks, via '
            'flag_dedup.write_suppression_record'
        ),
        deleter=DELETER_TTL,
    ),
    'cycle_summary': MarkerLifecycle(
        writer=(
            "Python, from each stage's StageReport — one per stage per cycle "
            "(metadata.kind='cycle_summary')"
        ),
        deleter=DELETER_POOL_TRIM,
    ),
}

# --------------------------------------------------------------------------- #
# Fingerprint identity — single-sourced against the LIVE logic in
# harness._derive_affected_ids (harness.py:191) and flag_dedup's
# content-fingerprint fallback (compute_content_fingerprint_signature /
# _content_fingerprint, flag_dedup.py:1442-1498). This module does not
# import either (see the import-light rationale in the module docstring);
# test_recon_self_model.py cross-checks this tuple against both functions
# directly so drift between this description and the live code fails a test.
# --------------------------------------------------------------------------- #

FINGERPRINT_IDENTITY_FIELDS = (
    # Legacy field: takes precedence over the four typed citation containers
    # below when present (pre-recon_report-cutover journal rows).
    'affected_ids',
    # Typed citation containers _derive_affected_ids flattens, in read order:
    # cited_tasks.task_id, cited_entities.canonical_name|entity_uuid,
    # cited_edges.edge_uuid, cited_memories.memory_id.
    'cited_tasks',
    'cited_entities',
    'cited_edges',
    'cited_memories',
    # Content-fingerprint fallback inputs (compute_content_fingerprint_signature),
    # used only when no task anchor exists (task_id is None and cited_tasks
    # yields no task_id): the normalized description, and flag_type (or the
    # _CONTENT_FP_FLAG_TYPE sentinel when flag_type is absent).
    'description',
    'flag_type',
)

# --------------------------------------------------------------------------- #
# §8.5 execution_class contract
# --------------------------------------------------------------------------- #

EXECUTION_CLASSES = ('code_tdd', 'operational', 'decision')

# --------------------------------------------------------------------------- #
# Recon tool-surface call shapes (hand-transcribed from the stage prompts —
# reconciliation/prompts/stage1.py, stage2.py, stage3.py, __init__.py — and
# from the MCP tool definitions in server/tools.py). Kept here so the exact
# call shape recon relies on is single-sourced instead of re-transcribed
# ad hoc wherever it's discussed (e.g. by a premise-lint consumer at ξ).
# --------------------------------------------------------------------------- #

MCP_CALL_SIGNATURES: dict[str, str] = {
    'submit_task': (
        "submit_task(project_root, title, description, priority, metadata) "
        "-> {'ticket': 'tkt_...'}"
    ),
    'resolve_ticket': (
        'resolve_ticket(ticket, project_root) -> '
        "{'status': 'created'|'combined'|'failed', 'task_id'?: ..., 'reason'?: ...}"
    ),
    'add_finding': (
        'add_finding(severity, category, flag_type, actionable, description, '
        "suggested_action, task_id) -> {'finding_id': ...}"
    ),
    'cite_task': (
        "cite_task(finding_id, project_id, task_id) -> {'ok': True}  "
        '# dedup anchor: _derive_affected_ids reads cited_tasks, not add_finding.task_id'
    ),
    'cite_entity': (
        "cite_entity(finding_id, name) -> {'ok': True}  "
        '# canonical entity NAME, not a UUID — server resolves it'
    ),
    'cite_edge': (
        "cite_edge(finding_id, edge_uuid) -> {'ok': True}  "
        '# full 36-char UUID, never truncated/constructed'
    ),
    'cite_memory': (
        "cite_memory(finding_id, memory_id, store) -> {'ok': True}  "
        "# store in {'mem0', 'graphiti'}; memory_id is the full 36-char UUID"
    ),
    'add_memory': (
        'add_memory(content, project_id, category=None, agent_id=None, metadata=None) '
        "-> {'memory_ids': [...]}"
    ),
    'search': (
        'search(query, project_id, categories=None, stores=None, limit=10) '
        "-> {'results': [...]}"
    ),
    'count_memories_by_metadata': (
        "count_memories_by_metadata(project_id, filters) -> {'count': N}"
    ),
}

# --------------------------------------------------------------------------- #
# Rendered prompt sections
# --------------------------------------------------------------------------- #


def _kinds_with_deleter(deleter: str) -> tuple[str, ...]:
    """Return the MARKER_KINDS entries whose MARKER_LIFECYCLE.deleter matches
    *deleter*, in MARKER_KINDS order."""
    return tuple(kind for kind in MARKER_KINDS if MARKER_LIFECYCLE[kind].deleter == deleter)


def render_marker_lifecycle_section() -> str:
    """Render the marker-lifecycle / run_id-fresh-per-cycle section, faithful
    to reconciliation/prompts/stage1.py:562-592's Flag Deduplication and
    Stage 2 Flag Relay prose."""
    gc_kinds = _kinds_with_deleter(DELETER_GC)
    kinds_str = ', '.join(f'`{kind}`' for kind in gc_kinds)
    return (
        '## Marker Lifecycle\n'
        "Stage 1's flag emission is post-processed by an automatic deduplicator "
        'that writes a `stage1_flag_marker` Mem0 memory per (task_id, flag_type), '
        'and — for items carrying `metadata.flag_for_stage2=true` — a '
        '`flag_for_stage2` marker so Stage 2 can pick it up; Stage 2 / '
        'TaskKnowledgeSync likewise writes a `stage2_persistence_marker`. '
        f'These per-task marker kinds ({kinds_str}) share one lifecycle rule: '
        'GC deletes them once their task_id goes terminal, and nothing else '
        '(not Stage 3 remediation, not the LLM) deletes them early.\n\n'
        'Every marker write MUST include `metadata.run_id=<current_run_id>`. '
        '`run_id` is minted fresh per run and is never persisted across cycles: '
        'the Mem0 marker channel is intentionally single-cycle. Any marker whose '
        'run_id does not match the current cycle — including one left over from '
        'a prior cycle whose consumer crashed before processing it — is '
        'unconditionally swept by Python and never reaches the LLM; it is not '
        'retried. The `flagged_items` structured-output field is the durable '
        'delivery channel that survives this sweep — the marker is only a '
        'same-cycle relay.'
    )


def render_suppression_schema_section() -> str:
    """Render the canonical suppression-record schema section, faithful to
    reconciliation/prompts/stage1.py:498-560."""
    return (
        '## Suppression Schema\n'
        'Canonical suppression record schema (Mem0, observations_and_summaries '
        "category) — the producer's contract read by the Stage 1 post-processor:\n"
        '  - `metadata.kind = "stage1_flag_suppression"`\n'
        '  - `metadata.task_id = <N>` (int)\n'
        '  - `metadata.flag_types = [<str>, ...]` (OPTIONAL scoping allowlist)\n'
        '  - content: `"STAGE 1 FLAG SUPPRESSION task_id=<N>"`\n\n'
        'Scoped vs. legacy/blanket suppression: a record WITH a non-empty '
        '`metadata.flag_types` is a scoped record that suppresses ONLY those '
        '(task_id, flag_type) pairs, leaving other flag_types for the same '
        'task_id free to surface. A record WITHOUT `flag_types` (absent, None, '
        'or empty — the legacy shape) is a blanket record that suppresses ALL '
        'flag_types for that task_id. When both a scoped and a legacy/blanket '
        'record exist for the same task_id, the blanket record wins (union '
        'semantics) — a blanket suppression cannot be narrowed by a more '
        'specific scoped record.'
    )
