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
