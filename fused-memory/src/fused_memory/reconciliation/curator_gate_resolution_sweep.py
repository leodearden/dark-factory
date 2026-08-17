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

# Single owners of the emitted flag's dedup key components.  ``flag_type`` is
# free-form per ``FINDING_ITEM_SCHEMA`` (cli_stage_runner.py); ``category`` must
# be a member of that schema's category enum, and ``task_memory_mismatch`` is
# exactly what this finding is — a task whose state does not reflect the memory
# corpus.  Together they form the ``compute_flag_signature`` key that gives the
# flag cross-cycle recurrence tracking and suppression.
GATE_RESOLUTION_FLAG_TYPE: str = 'task_completed_not_reflected'
GATE_RESOLUTION_FLAG_CATEGORY: str = 'task_memory_mismatch'


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


def extract_open_gate_task_ids(tasks: list[dict]) -> list[str]:
    """Return sorted, deduped str ids of open human-curator gate tasks.

    A task qualifies when ALL hold:

    - it is a ``dict`` (a non-dict element is skipped, never fed to ``.get``);
    - ``task['metadata']`` is a ``dict`` (absent/``None``/str/list metadata is
      skipped — a gate's markers are only readable off a parsed dict);
    - ``metadata['operational_mode'] == 'gate'`` — an EXACT, value-sensitive
      match, mirroring ``TaskInterceptor._is_gate_metadata``'s deliberate
      strictness.  ``'llm'``, ``'GATE'`` and ``None`` are all non-matches;
    - ``task['id']`` is not ``None`` (coerced to ``str``).  A gate with a
      missing/``None`` id is dropped rather than contributing a spurious id:
      the id is formatted straight into the Mem0 source key, so a wrong id
      queries the wrong key.

    Callers pass ``filtered_task_tree.active_tasks`` (the dataclass field on
    ``task_filter.FilteredTaskTree``), which excludes done/cancelled gates for
    free while still including ``status == 'blocked'`` — the state a filed
    human gate sits in.

    Structure, first-seen dedup, ``str(id)`` coercion and sorted return mirror
    ``stage1_stall_detector.extract_stalled_gate_backlog_task_ids``.  Two
    deliberate divergences from that helper:

    1. It keys on the ``gate_escalated_at`` stamp and explicitly does NOT
       filter on ``operational_mode`` (see its rationale block under
       ``gate_escalated_age_secs``: the operational-routing contract can coerce
       an ``operational_mode='llm'`` task into a pure gate while leaving the
       ``'llm'`` value in place, so a mode filter would miss that population).
       Here the mode filter IS the right key and the divergence is deliberate,
       not an oversight: this sweep's evidence is a ``curator_gate_{id}`` Mem0
       entry, which only exists for tasks the reify curator actually gated —
       exactly the ``operational_mode == 'gate'`` population — and a broader
       selection would just spend one Qdrant count per non-gate task to learn
       nothing.
    2. It uses bare ``task.get(...)`` with no ``isinstance(task, dict)`` guard;
       the guard added here is a strictly-safer divergence.

    Pure: no I/O, no side effects.  Empty input returns ``[]``.
    """
    seen: set[str] = set()
    for task in tasks:
        if not isinstance(task, dict):
            continue
        metadata = task.get('metadata')
        if not isinstance(metadata, dict):
            continue
        if metadata.get('operational_mode') != 'gate':
            continue
        raw_tid = task.get('id')
        if raw_tid is None:
            continue
        seen.add(str(raw_tid))
    return sorted(seen)


def build_gate_resolution_flag(task_id, memories, *, task: dict | None = None) -> dict:
    """Build the Stage-1 flag announcing that gate *task_id* is already resolved.

    Stage 1 runs under ``DISALLOW_TASK_WRITES`` and so cannot close the gate
    itself; it emits this flag into ``report.items_flagged`` and Stage 2 —
    which holds ``set_task_status``/``submit_task`` — acts on it.  That relay
    is the ``flag_for_stage2`` contract
    ``recon_self_model.render_source_completion_section`` documents.

    The emitted dict follows ``FINDING_ITEM_SCHEMA`` (cli_stage_runner.py):

    - ``flag_type``/``category`` are ``GATE_RESOLUTION_FLAG_TYPE`` /
      ``GATE_RESOLUTION_FLAG_CATEGORY``, and ``task_id`` is ``str``-coerced —
      together these are the ``compute_flag_signature`` key, so the flag
      dedupes across cycles (gaining a ``stage1_flag_marker`` recurrence row
      and honouring explicit suppression) instead of re-emitting unmarked
      forever.
    - ``description`` names the exact ``metadata.source`` key that was matched
      and how many entries matched (plus the task title when *task* is given),
      so a Stage-2 reader can re-derive the evidence deterministically rather
      than trusting the flag's assertion.
    - ``cited_memories`` carries one ``{'memory_id', 'store': 'mem0'}`` entry
      per input memory, in input order.  A memory dict with a missing or
      ``None`` ``'id'`` contributes NO entry rather than raising — a malformed
      row must not cost the whole finding.

    Args:
        task_id: The gate task's id (``str`` or ``int``; coerced to ``str``).
        memories: Mem0 memory dicts matching the gate's curator source key,
            as returned by ``MemoryService.get_memories_by_metadata``.
        task: Optional task dict, used only to enrich the description.

    Pure: no I/O, no side effects.
    """
    tid = str(task_id)
    source = curator_gate_source(tid)

    cited_memories = []
    for memory in memories:
        mid = memory.get('id') if isinstance(memory, dict) else None
        if mid is None:
            continue
        cited_memories.append({'memory_id': str(mid), 'store': 'mem0'})

    title = None
    if isinstance(task, dict):
        raw_title = task.get('title')
        if isinstance(raw_title, str) and raw_title:
            title = raw_title

    description = (
        f'Human-curator gate task {tid}'
        + (f' ("{title}")' if title else '')
        + f' is still open, but the curator has already ruled on it: '
        f'{len(memories)} Mem0 entr{"y" if len(memories) == 1 else "ies"} '
        f"carry metadata.source == '{source}' (deterministic Qdrant payload-filter "
        'match, not semantic search). The task state does not reflect the '
        'recorded resolution — read the cited memories for the ruling, then '
        'close or update the gate task accordingly.'
    )

    return {
        'description': description,
        'severity': 'moderate',
        'actionable': True,
        'task_id': tid,
        'flag_type': GATE_RESOLUTION_FLAG_TYPE,
        'category': GATE_RESOLUTION_FLAG_CATEGORY,
        'suggested_action': (
            f'Verify the ruling in the cited memories (metadata.source == '
            f"'{source}'), then set task {tid} to its resolved status and record "
            'the decision, so the gate stops appearing as an open human decision.'
        ),
        'cited_memories': cited_memories,
    }
