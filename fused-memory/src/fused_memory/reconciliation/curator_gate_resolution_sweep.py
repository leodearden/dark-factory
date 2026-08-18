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

Evidence for the source-key format, and the zero-recall risk it carries
(reviewer finding "correctness-risk", amendment pass):

    The ``curator_gate_{task_id}`` spelling has NO producer in this repo.  The
    writer is the reify MEMORY-curation session, and the only in-tree record of
    what it stamps is ``fused-memory/tests/fixtures/README.md`` (~line 100:
    "each canonical identified by ``metadata.source == 'curator_gate_NNNN'``"),
    which describes that session labelling the canonical of an adjudicated
    duplicate-memory cluster.  Two things therefore cannot be settled from
    inside this repo, and both are handled rather than assumed away:

    1. If the real writer's spelling — or its id BASIS (escalation id vs task
       id) — differs, this sweep is permanently zero-recall and reports
       ``scanned=N, flags_emitted=0``, byte-identical to a healthy cycle in
       which no gate happened to be resolved.  That is the exact silent miss
       the sweep exists to fix, so it is made VISIBLE two ways:
       ``sweep_resolved_curator_gates`` logs a distinct warning whenever it
       scanned gates and matched nothing without erroring, and the Stage-1 call
       site surfaces ``curator_gate_resolution_errors`` alongside the scanned/
       emitted counts, so "scanned > 0 and emitted == 0 and errors == 0" — the
       zero-recall signature — is derivable straight from ``report.stats``.
       To CONFIRM the format against live data, probe the reify corpus for the
       two gates the task names:
       ``count_memories_by_metadata(project_id='reify',
       filters={'source': 'curator_gate_5561'})`` (and ``..._5563``); a
       non-zero count is direct observation of the spelling.
    2. The key may be stamped for gates whose memories were merely CURATED
       rather than gates that were RULED ON.  The flag text therefore asserts
       only what was OBSERVED — that N entries carrying the key exist — and
       tells the reader to check the cited memories before acting (see
       ``build_gate_resolution_flag``).  Stage 2 holds ``set_task_status``, so
       an over-claiming description could get a still-open human gate closed.
"""

from __future__ import annotations

import asyncio
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
      than trusting the flag's assertion.  It states only what was OBSERVED —
      that N entries carrying the key exist — and NOT that the gate is
      resolved: the entries are matched on the key alone and their content is
      never read here, and the key's producer lives outside this repo (see the
      module docstring).  Stage 2 holds ``set_task_status``, so an
      over-claiming description could get a still-open human decision gate
      closed on a curated-but-unruled cluster.
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
        + ' is still open, and '
        f'{len(memories)} Mem0 entr{"y" if len(memories) == 1 else "ies"} '
        f"stamped metadata.source == '{source}' exist"
        f'{"s" if len(memories) == 1 else ""} '
        '(deterministic Qdrant payload-filter match, not semantic search). '
        'That source key is what the reify curator stamps when it records a '
        'ruling on a gate, so this is EVIDENCE the gate may already be '
        'resolved — not proof: the match is on the key alone, and the entry '
        'content was not read here. Read the cited memories; if they record '
        'a ruling on this gate, the task state does not reflect it.'
    )

    return {
        'description': description,
        'severity': 'moderate',
        'actionable': True,
        'task_id': tid,
        'flag_type': GATE_RESOLUTION_FLAG_TYPE,
        'category': GATE_RESOLUTION_FLAG_CATEGORY,
        'suggested_action': (
            f'Read the cited memories (metadata.source == \'{source}\') and check '
            f'whether they record a ruling on gate {tid}. If they do, set the task '
            'to its resolved status and record the decision, so the gate stops '
            'appearing as an open human decision. If they only curate memories '
            'about the gate without ruling on it, dismiss this flag — do NOT '
            'close a human decision that has not been made.'
        ),
        'cited_memories': cited_memories,
    }


# ── Async I/O helpers ────────────────────────────────────────────────────────


async def sweep_resolved_curator_gates(
    memory_service,
    project_id: str,
    task_ids: list[str],
    *,
    tasks_by_id: dict[str, dict] | None = None,
    log: logging.Logger = logger,
) -> dict:
    """Flag every open gate in *task_ids* that already has a curator ruling.

    For each ``task_id``, counts Mem0 entries stamped
    ``metadata.source == curator_gate_source(task_id)`` via
    ``MemoryService.count_memories_by_metadata`` — Qdrant's count API with an
    exact payload filter, NOT semantic search, so a resolved gate can never be
    lost to top-N truncation.  On a positive count the matching memories are
    enumerated with ``get_memories_by_metadata`` (Qdrant's scroll API, same
    filter) purely to cite them, and one flag is built per resolved gate.

    The scroll alone could answer both questions, so the count is NOT kept for
    determinism (the scroll is equally deterministic) — it is kept solely to
    avoid payload transfer on the common path.  Most open gates are unresolved,
    and for those the count is a payload-free probe that returns 0 and moves no
    memory bodies; only a gate that actually has evidence pays a second round
    trip.

    The payload filter carries ONLY the ``source`` key.  Qdrant ANDs filter
    conditions, so adding a ``task_id`` condition would silently miss any
    curator entry whose writer omitted that field — and the source key already
    encodes the id, so the extra condition buys nothing and can only lose
    recall.

    Args:
        memory_service: Object exposing ``count_memories_by_metadata`` and
            ``get_memories_by_metadata``.
        project_id: Project scope for both reads.
        task_ids: Str gate task ids (typically from
            ``extract_open_gate_task_ids``).
        tasks_by_id: Optional ``{str(task_id): task_dict}`` map, used ONLY to
            enrich an emitted flag's description with the gate's title.  A
            missing id — or ``None`` — simply omits the title.  The map is
            never consulted for selection, so a stale or partial map cannot
            change which gates are swept or which are flagged.
        log: Logger to use (default: this module's logger).

    Returns:
        dict with ``flags`` (the Stage-1 flag dicts to append to
        ``report.items_flagged``) and int counts ``scanned`` (gates checked),
        ``resolved`` (gates with >= 1 curator entry), ``errors``.

    Best-effort, and fail-SAFE in one specific direction: a backend failure at
    EITHER the count or the citation-fetch step is caught, logged, tallied into
    ``stats['errors']``, and skips flag emission for that gate — it never
    aborts the sweep for the remaining gates, and an errored read is NEVER
    treated as evidence of resolution.  That asymmetry is deliberate and
    mirrors ``sweep_degenerate_task_nodes``' "uncertain is never treated as
    degenerate" invariant: a false "resolved" flag could get a still-open
    human decision gate closed, which is strictly worse than missing one
    cycle's detection (the next cycle re-checks).  Note that a Qdrant
    read-timeout PROPAGATES out of ``count_memories_by_metadata`` /
    ``get_memories_by_metadata`` rather than returning empty, so it arrives
    here as an exception and is tallied — never silently seen as "no curator
    entry".  ``asyncio.CancelledError``/``KeyboardInterrupt``/``SystemExit``
    are re-raised unchanged (never swallowed as best-effort).

    The per-task loop is sequential rather than an ``asyncio.gather``: the open
    gate population is small, and a serial loop keeps per-task error
    attribution exact.  Empty *task_ids* short-circuits to all-zero stats with
    no backend calls.
    """
    stats: dict = {'flags': [], 'scanned': 0, 'resolved': 0, 'errors': 0}

    if not task_ids:
        return stats

    for task_id in task_ids:
        stats['scanned'] += 1
        source = curator_gate_source(task_id)

        try:
            count = await memory_service.count_memories_by_metadata(
                project_id=project_id, filters={'source': source},
            )
        except (asyncio.CancelledError, KeyboardInterrupt, SystemExit):
            raise
        except Exception:
            # Includes a propagated Qdrant read-timeout.  Uncertain is never
            # "no curator entry" and never "resolved" — tally and move on.
            log.exception(
                'curator_gate_resolution_sweep: count_memories_by_metadata failed '
                'for source=%s project_id=%s',
                source, project_id,
            )
            stats['errors'] += 1
            continue

        if not count:
            continue

        try:
            memories = await memory_service.get_memories_by_metadata(
                project_id=project_id, filters={'source': source},
            )
        except (asyncio.CancelledError, KeyboardInterrupt, SystemExit):
            raise
        except Exception:
            # The count says evidence exists but we could not read it.  Emitting
            # a flag we cannot cite would let Stage 2 close a gate on unverified
            # evidence — skip this gate and re-check next cycle.
            log.exception(
                'curator_gate_resolution_sweep: get_memories_by_metadata failed '
                'for source=%s project_id=%s',
                source, project_id,
            )
            stats['errors'] += 1
            continue

        if not memories:
            # Count/scroll divergence (TOCTOU).  The count said evidence exists
            # but the scroll read none back — the curator entry was deleted or
            # GC'd between the two reads, or the count was answered from a
            # stale segment.  An empty list is NOT an exception, so without this
            # guard it falls straight through and emits a flag reading "0 Mem0
            # entries ... exist" with an EMPTY cited_memories: the same
            # uncitable claim the fetch-failure branch above exists to prevent,
            # handed to a stage that holds set_task_status.  A gate is only
            # flagged once at least one citable memory was actually READ.
            # Tallied as an error — an anomalous read, not a clean "no
            # evidence" — and skipped; the next cycle re-checks.
            log.warning(
                'curator_gate_resolution_sweep: count/scroll divergence for '
                'source=%s project_id=%s (count=%s, scroll returned 0 rows) — '
                'not flagging; a gate needs at least one citable memory',
                source, project_id, count,
            )
            stats['errors'] += 1
            continue

        stats['resolved'] += 1
        stats['flags'].append(
            build_gate_resolution_flag(
                task_id, memories, task=(tasks_by_id or {}).get(str(task_id)),
            ),
        )

    if stats['scanned'] and not stats['resolved'] and not stats['errors']:
        # Zero-recall canary (reviewer finding "correctness-risk").  A clean
        # sweep that matched nothing is indistinguishable, in the stats alone,
        # from a sweep whose source-key spelling (or id basis) does not match
        # what the curator actually writes — and that writer lives outside this
        # repo (see the module docstring).  Log the exact key SHAPE that was
        # probed so a format drift is greppable instead of silent; the Stage-1
        # call site surfaces the same signature on report.stats as
        # scanned > 0 / flags_emitted == 0 / errors == 0.
        log.warning(
            'curator_gate_resolution_sweep: scanned %d open gate(s) in '
            'project_id=%s with no errors and matched no curator entries '
            '(probed keys of the form %r) — expected when no gate is resolved, '
            'but also the signature of a source-key format drift that would '
            'make this sweep permanently zero-recall',
            stats['scanned'], project_id, CURATOR_GATE_SOURCE_TEMPLATE,
        )

    return stats
