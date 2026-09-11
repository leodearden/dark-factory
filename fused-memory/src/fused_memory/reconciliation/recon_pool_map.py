"""Canonical stage -> recon_pool map for per-cycle reconciliation summaries.

LEAF module: must NOT import anything from ``fused_memory`` (not even
``fused_memory.reconciliation``). That rule is enforced, not just documented:
``tests/test_mem0_tombstone.py::TestReconPoolMapIsImportFreeLeaf`` imports this
module in a fresh interpreter and fails if doing so loads any other
``fused_memory.*`` module. This map has two independent consumers,
each keying on a different half of it, for ``metadata.kind ==
'cycle_summary'`` writes: reconciliation/summary_pool.py trims a pool by
the ``recon_pool`` *value* (passed in as a parameter — see
``filters={'recon_pool': recon_pool}``), while
scripts/prune_recon_cycle_summaries.py buckets records by
``metadata.stage`` against this map's *keys* (``_POOL_STAGES =
('memory_consolidator', 'task_knowledge_sync')``). Both derive from this
single map, which is what makes tagging independent of LLM prompt
compliance (task 2077).

Before task 2140 these values were duplicated three ways — once here (well,
once in services/memory_service.py) and once each in
reconciliation/stages/memory_consolidator.py and
reconciliation/stages/task_knowledge_sync.py — because importing the
per-stage constants directly from services/memory_service.py created a real
circular import (memory_consolidator -> task_knowledge_sync ->
services.live_workflow_detector -> services/__init__ -> memory_service).
This module sits below all of that: reconciliation/__init__.py is
import-free, and this module imports nothing, so every one of those three
call sites can import from here without re-creating the cycle. Do NOT change
any of these string values — the ops prune script and the summary_pool trim
filter match on them exactly.
"""

from __future__ import annotations

STAGE1_CYCLE_SUMMARY_RECON_POOL = 'stage1_cycle_summary'
STAGE2_CYCLE_SUMMARY_RECON_POOL = 'stage2_cycle_summary'

CYCLE_SUMMARY_STAGE_TO_RECON_POOL: dict[str, str] = {
    'memory_consolidator': STAGE1_CYCLE_SUMMARY_RECON_POOL,
    'task_knowledge_sync': STAGE2_CYCLE_SUMMARY_RECON_POOL,
}

# --- cycle_summary metadata discriminators (task 3041 amendment pass) -------
#
# The `kind` and `record_type` halves of the same Mem0 metadata vocabulary the
# map above indexes, hoisted here for the same reason the map itself is here:
# they have consumers on BOTH sides of an import edge that cannot be traversed
# in both directions.
#
#   reconciliation/summary_pool.py -------> reconciliation/mem0_tombstone.py
#       (imports the tombstone writer for the trim's audit trail)
#
# summary_pool needs both literals for its enumeration filter and its
# record_type-aware eviction order; mem0_tombstone needs both for
# is_protected_mirror_record. The reverse import would be a cycle, so before
# this change mem0_tombstone carried private copies kept in sync "BY
# CONVENTION" with nothing pinning them equal — a future edit to one side
# would have silently disabled half the protected-mirror guard. That is the
# lockstep literal duplication INV-5 forbids (see standing_decision_constants
# and RECORD_KIND_MEM0_TOMBSTONE in recon_ledger.py for the same fix applied
# to the other direction). This module imports nothing at all, so both sides
# can reach it unconditionally.
#
# Do NOT change these string values: they are matched exactly by live Qdrant
# payloads written by every prior cycle, and — like the pool names above — by
# scripts/prune_recon_cycle_summaries.py.
CYCLE_SUMMARY_KIND = 'cycle_summary'

# record_type vocabulary for kind='cycle_summary' Mem0 writes (task 2468).
# There are two distinct writers: summary_pool.write_cycle_summary's
# deterministic, terse, auto-generated mirror of the authoritative ledger row
# (LEDGER_STAMP), and the LLM-authored reconstruction/self-heal write in
# reconciliation/prompts/stage2.py (NARRATIVE).
#
# NARRATIVE still has no Python consumer: prompts/stage2.py and
# recon_self_model.py hardcode the literal in prose/f-string text rather than
# importing it, because those prompt modules are deliberately import-light.
# They COULD import from here (this module imports nothing), but that is a
# separate cleanup with its own review surface — until it happens, keeping the
# prompt-side literal in sync is a reviewed invariant, not an enforced one.
CYCLE_SUMMARY_RECORD_TYPE_LEDGER_STAMP = 'ledger_stamp'
CYCLE_SUMMARY_RECORD_TYPE_NARRATIVE = 'narrative'
