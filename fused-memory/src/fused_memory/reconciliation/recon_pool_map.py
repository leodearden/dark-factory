"""Canonical stage -> recon_pool map for per-cycle reconciliation summaries.

LEAF module: must NOT import anything from ``fused_memory`` (not even
``fused_memory.reconciliation``). This map has two independent consumers,
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
