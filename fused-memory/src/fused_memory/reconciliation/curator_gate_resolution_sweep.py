"""Resolved human-curator-gate detection via a Mem0 ``source`` sweep — task 3084.

Stage 1 (``MemoryConsolidator``) has no deterministic sweep that notices when
a human-curator gate task (``metadata.operational_mode == 'gate'``) has in
fact already been resolved.  Today that detection is an ad-hoc Stage-3
spot-check, and it misses roughly a quarter of cases: reify run ec45eed0
(Stage 1 finding c8e9b86e) closed 3 gates and flagged 6, but two more —
gates 5561 and 5563 — were resolved-but-stale and went undetected.

The evidence needed to close that gap is already deterministic and already
in Mem0.  When the reify curator rules on a gate it writes its ruling
stamped ``metadata.source == f'curator_gate_{task_id}'``
(``fused-memory/tests/fixtures/README.md``: "each canonical identified by
``metadata.source == 'curator_gate_NNNN'``", and the same section enumerates
the 21 resolved ``milestone_gate`` escalations including 5561 and 5563).
Nothing reads that key back.  This module is the deterministic sweep that
does, emitting a Stage-1 flag so Stage 2 — which, unlike Stage 1, holds
``set_task_status``/``submit_task`` — can act on it.

Design decisions (captured in plan.json):

- The Mem0 filter is ``{'source': curator_gate_source(task_id)}`` ONLY.  No
  ``task_id`` key is ANDed in: Qdrant payload filters AND their conditions,
  so an extra ``task_id`` condition would silently miss any curator entry
  whose writer omitted that field — and the source key already encodes the
  task id, so the extra condition buys nothing and can only lose recall.
  Missing a resolved gate is precisely the failure this module exists to fix.
- Reads go through ``MemoryService.count_memories_by_metadata`` /
  ``get_memories_by_metadata``, which talk to Qdrant's count/scroll API with
  an exact payload filter — deterministic key-equality, explicitly NOT
  semantic search, so a resolved gate can never be lost to top-N truncation.
- The source-key spelling has exactly ONE owner in the tree
  (``CURATOR_GATE_SOURCE_TEMPLATE``/``curator_gate_source``, INV-5).  A
  divergent second copy would silently make the sweep zero-recall.
- Detection only; no task writes.  Stage 1 runs under ``DISALLOW_TASK_WRITES``,
  so this module emits a flag into ``report.items_flagged`` and Stage 2 acts
  — the ``flag_for_stage2`` relay contract that
  ``recon_self_model.render_source_completion_section`` documents.
- Best-effort throughout, in the fail-SAFE direction: an errored read is
  never treated as evidence of resolution (see
  ``sweep_resolved_curator_gates``).
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

# The single owner of the curator-gate source-key spelling (INV-5).  This is a
# ``metadata.source`` value; ``source`` is a BLESSED free-form metadata key
# (shared/src/shared/memory_metadata.py), so a dynamic per-task value needs no
# vocabulary-registry change.  The value is written OUTSIDE this repo by the
# reify curator workflow — we do not get to choose the format, we read it.
CURATOR_GATE_SOURCE_TEMPLATE: str = 'curator_gate_{task_id}'


# ── Pure helpers ─────────────────────────────────────────────────────────────


def curator_gate_source(task_id) -> str:
    """Return the ``metadata.source`` value the curator stamps for *task_id*.

    ``curator_gate_source(5561) == curator_gate_source('5561') ==
    'curator_gate_5561'``.  The id is coerced via ``str()`` BEFORE formatting
    so an int-typed task id (common when ids come straight off a task dict)
    cannot produce a key that diverges from the str-typed spelling — a
    divergence would make the sweep silently match nothing.

    Pure: no I/O, no side effects.
    """
    return CURATOR_GATE_SOURCE_TEMPLATE.format(task_id=str(task_id))
