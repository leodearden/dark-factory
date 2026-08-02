"""Citation integrity for Stage-1 reconciliation (tasks 2978, 3108).

This module owns one invariant end-to-end: **a cited memory id must resolve.**
It covers both halves of that invariant, so there is one owner rather than two
mechanisms that can drift.

Half 1 — recon-report citations (task 2978). ``verify_cited_memories`` walks
each finding's ``cited_memories`` list and re-resolves every cited Mem0 id
against the live store, so a finding's claim can never be silently backed by an
id that does not (or no longer) exist. Its stats carry the ``stage1_`` prefix
because ``MemoryConsolidator.run()`` merges them into ``report.stats``.

Half 2 — task-metadata citations (task 3108). ``find_citation_occurrences`` /
``find_live_citation_occurrences`` / ``repoint_metadata`` /
``repoint_tombstone_chain`` / ``repoint_task_citations`` find and rewrite live
pointers to a memory id that is about to be deleted, so a consolidation delete
repoints citations BEFORE the irreversible destruction rather than leaving
dangling pointers behind it. These run from the ``delete_memory`` MCP tool
handler's citation-repoint gate, NOT from ``MemoryConsolidator.run()``: the
consolidator never deletes from Python, so a sweep inside ``run()`` would
execute AFTER the delete and could only report damage. Their stats therefore
ride back on the ``delete_memory`` response under ``citation_repoint`` and
deliberately carry NO ``stage1_`` prefix — they are tool-response stats, not
stage-report stats.

**A tombstone is provenance, never a live pointer.** This is the one rationale
the rest of the module refers back to rather than restating.
``X_CITATION_TOMBSTONE_KEY`` records exist precisely to name a dead id (that is
what ``superseded_memory_id`` is *for*), so counting one as a citation would
make every already-repointed task an outstanding citer forever — and a retry
would rewrite ``superseded_memory_id`` to the survivor (destroying the
forwarding provenance), append a further record whose ``paths`` names ledger
internals rather than any real citation, and inflate the repoint count,
unbounded on every further pass. So the ledger is excluded both from detection
(``find_live_citation_occurrences``) and from the rewrite, which is what makes
a retried sweep idempotent instead of self-amplifying; only a *stale
destination* (``replacement_memory_id``) is carried forward, by
``repoint_tombstone_chain``.

These are small, single-purpose helpers in the ``reconciliation/`` convention
(cf. ``flag_dedup``/``task_filter``).
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from fused_memory.middleware.task_interceptor import interceptor_write_succeeded
from fused_memory.reconciliation.task_filter import INACTIVE_TASK_STATUSES

logger = logging.getLogger(__name__)

# The recon-stage caller identity the repoint writes are attributed to.
# Matches the `recon-stage-` prefix that recon_write_policy scopes on — see
# `task_interceptor.update_task`'s `is_recon_stage_write` predicate. (Symbol
# names, not file:line refs, on purpose: line numbers rot within a release.)
REPOINT_AGENT_ID = 'recon-stage-memory_consolidator'

# Canonical 36-char UUID, the ONLY shape accepted as a forwarding pointer.
# Anchored end-to-end so a uuid embedded in prose does not qualify: a
# replacement value must BE an id, not merely mention one.
_CANONICAL_UUID_RE = re.compile(
    r'^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-'
    r'[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$'
)

# Tier-C (``x_``) metadata key holding the old->new forwarding records. The
# ``x_`` namespace is silently admitted by ``shared.task_metadata.parse_metadata``
# with no registration and no SchemaWarning, and
# ``recon_write_policy.is_terminal_annotation_add`` already blesses ``x_``
# annotation adds, so no blessed-key change is needed.
X_CITATION_TOMBSTONE_KEY = 'x_memory_citation_tombstones'


async def verify_cited_memories(
    findings: list[dict[str, Any]],
    memory_service: Any,
    project_id: str,
) -> dict[str, int]:
    """Verify each finding's cited Mem0 memories still resolve; drop phantoms.

    For every ``cited_memories`` entry, resolve its ``memory_id`` via
    ``memory_service.get_memory_by_id(project_id, memory_id)`` (the
    no-silent-fail raw Qdrant point read):

    - resolves (truthy record) -> KEEP the citation and count it verified;
    - genuine not-found (``None``) -> DROP it from ``cited_memories`` and append
      a ``{memory_id, store, reason: 'memory_not_found'}`` marker to the
      finding's ``citation_failures`` list, so the phantom claim is surfaced
      rather than silently retained.

    Only ``store == 'mem0'`` entries carrying a truthy ``memory_id`` are
    resolved. ``store == 'graphiti'`` citations (and any malformed/id-less
    entries) are left UNTOUCHED and never looked up: ``get_memory_by_id`` is a
    Mem0/Qdrant-only point read, so graphiti-store verification is intentionally
    out of scope (it would need a different graph primitive and would otherwise
    false-flag every graphiti citation as a phantom).

    Mirrors ``standing_decision_writer.resolve_evidence_refs``'s found/None
    branching. Returns ``stage1_*`` stats for ``report.stats``.
    """
    # Why re-verify at all, at report-assembly time? Two root causes this pass
    # closes that a cite-time check cannot:
    #   (1) The LLM stage hallucinating/typo-ing an id: the structured-output
    #       JSON fallback (BaseStage.run builds items_flagged straight from the
    #       model's flagged_items) bypasses recon_report.cite_memory's existence
    #       check entirely, so a fabricated id reaches cited_memories unchecked.
    #   (2) Citing an id whose queued add_memory write later FAILED: a TOCTOU
    #       the cite-time-only check structurally cannot catch (it validates at
    #       cite-time, not at report-assembly-time). This run()-time
    #       re-verification is the only check that closes it.
    stats = {
        'stage1_phantom_citations_dropped': 0,
        'stage1_citations_verified': 0,
        'stage1_citation_verification_errors': 0,
    }
    for finding in findings:
        cited = finding.get('cited_memories') or []
        if not cited:
            # Nothing to verify — leave the finding ENTIRELY untouched. In
            # particular, do NOT add an empty ``cited_memories`` key to a finding
            # that never carried one: this pass runs over every flagged finding,
            # and mutating citation-less findings would surprise unrelated
            # consumers (and their tests).
            continue
        kept: list[Any] = []
        for entry in cited:
            # Skip anything we cannot — or must not — resolve, preserving it
            # verbatim and never counting it verified/dropped/errored:
            #   * a non-dict entry (malformed);
            #   * a dict with no truthy memory_id (nothing to look up);
            #   * a non-mem0-store citation. get_memory_by_id is Mem0/Qdrant-only
            #     (a raw point-id read), so resolving a graphiti edge uuid through
            #     it would return not-found for EVERY graphiti citation and
            #     false-flag legitimate graph evidence as a phantom.
            if (
                not isinstance(entry, dict)
                or entry.get('store') != 'mem0'
                or not entry.get('memory_id')
            ):
                kept.append(entry)
                continue
            memory_id = entry.get('memory_id')
            store = entry.get('store')
            try:
                record = await memory_service.get_memory_by_id(project_id, memory_id)
            except Exception as exc:
                # A raised backend error is 'unknown', not 'absent': dropping the
                # citation here would itself be a silent-fail (the exact
                # anti-pattern this fix forbids). KEEP it, surface the
                # uncertainty via a marker, and never propagate — the stage must
                # not crash on a check error.
                kept.append(entry)
                finding.setdefault('citation_failures', []).append(
                    {
                        'memory_id': memory_id,
                        'store': store,
                        'reason': 'verification_error',
                        'error_type': type(exc).__name__,
                    },
                )
                stats['stage1_citation_verification_errors'] += 1
                continue
            if record:
                kept.append(entry)
                stats['stage1_citations_verified'] += 1
            else:
                finding.setdefault('citation_failures', []).append(
                    {'memory_id': memory_id, 'store': store, 'reason': 'memory_not_found'},
                )
                stats['stage1_phantom_citations_dropped'] += 1
        finding['cited_memories'] = kept
    return stats


# --------------------------------------------------------------------------- #
# Task-metadata citation scanning (task 3108)
# --------------------------------------------------------------------------- #


def find_citation_occurrences(metadata: Any, memory_id: str) -> list[str]:
    """Return a dotted/indexed path for EVERY occurrence of ``memory_id``.

    A pure, side-effect-free recursive walk over ``metadata``:

    - **dicts** are descended by key (``memory_hints`` -> ``memory_hints.x``);
    - **lists/tuples** are descended by index (``queries`` -> ``queries[0]``);
    - **strings** match on SUBSTRING containment, so an id embedded in free
      prose (``'see canonical entry <uuid> for ...'``) is found, not just a
      scalar whose whole value is the id;
    - **dict KEYS** are matched too, reported as ``<path>#key``. A uuid-keyed
      map (``{'x_cluster': {'<uuid>': 'note'}}``) is a plausible shape for the
      open Tier-C namespace, and a key-position citation dangles after a delete
      exactly as a value-position one does.

    Never mutates its input, and never raises on malformed input: a ``None``,
    a bare string, an int or a top-level list all return ``[]``.

    **Why this is a mechanical all-keys scan and not a known-field lookup.**
    Incident failure mode (1) was exactly a known-field/known-task
    enumeration: a hand-written pass found 3 of 8 citing tasks, and the 5 it
    missed included the pending/dispatchable ones — i.e. the ones that
    mattered, because they were still going to be dispatched against a dead
    pointer. The reflex fix ("just list the citation-bearing keys") is worse
    than it looks here: the known citation key names from that incident
    (``mem0_canonical_entry``, ``mem0_cluster_entries``,
    ``x_memory_write_caution``) are *reify-project* task-DB keys with ZERO
    occurrences in this repo, so an allowlist built from them would be
    empty-by-construction and would silently pass every delete. Metadata is
    ``extra='allow'`` with a wide-open Tier-C ``x_`` namespace, so the set of
    keys — or key POSITIONS — that may carry a citation is not knowable in
    advance. Scan everything instead.

    ``memory_id`` is matched in full. A truncated 8-char prefix therefore
    matches nothing, which is deliberate: two distinct UUIDs can share a
    prefix (the hazard the Stage-1 prompt warns about), and a prefix match
    would repoint an unrelated entry.
    """
    if not memory_id or not isinstance(memory_id, str):
        return []
    if not isinstance(metadata, dict):
        # A non-dict blob carries no addressable top-level keys. Returning []
        # (rather than raising) keeps the scan callable on whatever the task
        # backend hands back, including a missing/NULL metadata column.
        return []

    paths: list[str] = []

    def _walk(node: Any, path: str) -> None:
        if isinstance(node, str):
            if memory_id in node:
                paths.append(path)
            return
        if isinstance(node, dict):
            for key, value in node.items():
                child = f'{path}.{key}' if path else str(key)
                # A citation can sit in KEY position as easily as in value
                # position, and a `#key` suffix keeps the two distinguishable
                # (a uuid-keyed entry whose value also mentions the id yields
                # both `x.<uuid>#key` and `x.<uuid>`).
                if memory_id in str(key):
                    paths.append(f'{child}#key')
                _walk(value, child)
            return
        if isinstance(node, (list, tuple)):
            for index, value in enumerate(node):
                _walk(value, f'{path}[{index}]')
            return
        # Any other scalar (int/float/bool/None) cannot contain a uuid.

    _walk(metadata, '')
    return paths


def find_live_citation_occurrences(metadata: Any, memory_id: str) -> list[str]:
    """Like :func:`find_citation_occurrences`, minus the tombstone ledger.

    Applies the module header's "a tombstone is provenance, never a live
    pointer" rule: ``X_CITATION_TOMBSTONE_KEY`` is dropped before scanning.

    The exclusion lives HERE, in a caller-facing wrapper, rather than inside
    :func:`find_citation_occurrences`: that function is a deliberately generic
    mechanical scanner, and its "the dead id survives ONLY in the labelled
    provenance slot" assertion is how the tombstone contract is proven. This
    wrapper is what "cites" means for the sweep and the gate — both must agree
    on it, or the gate demands a repoint the sweep then reports zero work for.
    """
    if not isinstance(metadata, dict):
        return find_citation_occurrences(metadata, memory_id)
    return find_citation_occurrences(
        {k: v for k, v in metadata.items() if k != X_CITATION_TOMBSTONE_KEY},
        memory_id,
    )


def repoint_tombstone_chain(
    tombstones: Any,
    old_id: str,
    new_id: str,
) -> list[dict[str, Any]]:
    """Forward each existing record's stale DESTINATION, and nothing else.

    A prior record forwarding ``A -> old_id`` would, left verbatim, park a
    dangling pointer in a field literally named ``replacement_memory_id`` — the
    exact harm this module prevents, merely relocated. So that slot is
    transitively repointed to ``new_id`` and the chain stays resolvable.

    Every other field is copied byte-identical, per the module header's
    provenance rule: ``superseded_memory_id`` MUST keep naming a dead id, so a
    wholesale :func:`repoint_metadata` pass over the ledger is exactly wrong —
    it would turn an ``old_id -> new_id`` record into a self-referential
    ``new_id -> new_id`` one.

    Pure: neither the list nor its records are mutated. Non-list input yields
    ``[]``, so a malformed/absent ledger degrades to "no prior records" rather
    than raising mid-sweep.
    """
    if not isinstance(tombstones, list):
        return []
    out: list[dict[str, Any]] = []
    for record in tombstones:
        if not isinstance(record, dict):
            # Preserve whatever is there rather than dropping it — this ledger
            # is provenance, and silently discarding an unrecognised record
            # would be the silent-fail this module exists to prevent.
            out.append(record)
            continue
        if record.get('replacement_memory_id') == old_id:
            forwarded = dict(record)
            forwarded['replacement_memory_id'] = new_id
            out.append(forwarded)
        else:
            out.append(dict(record))
    return out


def _deep_copy(node: Any) -> Any:
    """Structural deep copy over the same node kinds :func:`repoint_metadata`
    traverses, with no dependence on any id.

    Kept as its own walker rather than ``copy.deepcopy`` so the degenerate-input
    path shares the rewrite path's traversal shape exactly and cannot raise on
    an exotic leaf (leaves are returned as-is, never copied).
    """
    if isinstance(node, dict):
        return {key: _deep_copy(value) for key, value in node.items()}
    if isinstance(node, list):
        return [_deep_copy(value) for value in node]
    if isinstance(node, tuple):
        return tuple(_deep_copy(value) for value in node)
    return node


def repoint_metadata(
    metadata: Any,
    old_id: str,
    new_id: str,
) -> tuple[Any, int]:
    """Rewrite every occurrence of ``old_id`` to ``new_id``; return ``(blob, count)``.

    Pure: the input is deep-copied, never mutated, so a caller whose
    subsequent write fails is left holding its original object rather than a
    half-rewritten one.

    Rewrite rules mirror :func:`find_citation_occurrences`'s match rules
    exactly, and the count is computed on the SAME traversal, so the two
    functions cannot drift apart:

    - a string equal to ``old_id`` becomes ``new_id``;
    - any other string gets ``str.replace(old_id, new_id)``, so an id embedded
      in free prose is repointed with the surrounding text preserved verbatim;
    - a dict KEY containing ``old_id`` is rewritten too, the dict rebuilt
      around it — UNLESS the rewritten key already exists in that same dict.
      A collision is left verbatim rather than collapsed: overwriting would
      silently drop one of two distinct entries, and leaving the doomed id in
      place instead keeps :func:`repoint_task_citations`' post-rewrite residual
      check able to see it and refuse the delete;
    - dicts and lists are descended; every other scalar is returned as-is.

    A string containing ``old_id`` more than once counts as ONE occurrence, to
    stay path-for-path consistent with the scanner (which reports one path per
    string, not one per byte offset). A rewritten key counts as one more.
    ``count`` is therefore rewrites-performed, not occurrences-present: a
    skipped collision is deliberately NOT counted, so it cannot read as done.

    Degenerate input is returned unchanged (deep-copied) with ``count == 0``,
    never raised on and never rewritten — matching
    :func:`find_citation_occurrences`, which returns ``[]`` rather than raising
    on a malformed id. In particular an empty ``old_id`` must NOT reach the
    rewrite path: ``''.replace('', x)`` inserts ``x`` between every character,
    which would silently corrupt every string in the blob.
    """
    if not isinstance(old_id, str) or not old_id or not isinstance(metadata, dict):
        # Nothing addressable to rewrite. Still deep-copy dict input so the
        # return value is never an alias of the caller's object.
        return (_deep_copy(metadata) if isinstance(metadata, dict) else metadata), 0

    count = 0

    def _rewrite(node: Any) -> Any:
        nonlocal count
        if isinstance(node, str):
            if old_id in node:
                count += 1
                return node.replace(old_id, new_id)
            return node
        if isinstance(node, dict):
            rebuilt: dict[Any, Any] = {}
            for key, value in node.items():
                target = key
                if isinstance(key, str) and old_id in key:
                    candidate = key.replace(old_id, new_id)
                    # Never collapse two entries into one. If the destination
                    # key already exists (here or already emitted), keep the
                    # original key verbatim: both values survive, the doomed id
                    # stays visible, and the caller's residual check refuses the
                    # delete rather than silently permitting a half-repoint.
                    if candidate not in node and candidate not in rebuilt:
                        target = candidate
                        count += 1
                rebuilt[target] = _rewrite(value)
            return rebuilt
        if isinstance(node, list):
            return [_rewrite(value) for value in node]
        if isinstance(node, tuple):
            return tuple(_rewrite(value) for value in node)
        return node

    return _rewrite(metadata), count


def is_concrete_memory_id(value: Any) -> bool:
    """Return True only for a well-formed canonical 36-char memory UUID.

    This is the mechanical guard against **incident failure mode (2)**. During
    the incident Stage 2 wrote a "correction" instructing dispatch to re-derive
    the canonical entry via ``search(query=...)``. Running that query live
    returned only superseded cluster members, routing dispatch straight back
    into the contradictory advice that consolidation existed to collapse. A
    forwarding pointer is only a pointer if it is a concrete id; prose that
    describes how to *find* an id is not one.

    Making UUID-shape a hard precondition — of both
    :func:`build_citation_tombstone` and the delete-side guard — turns "never
    emit a re-derive-via-search instruction" from a prose rule into a checkable
    one. The full-36-char requirement also aligns with the truncated-UUID
    hazard the Stage-1 prompt already warns about: an 8-char prefix is not a
    valid delete id and is not a valid forwarding pointer either.
    """
    return isinstance(value, str) and bool(_CANONICAL_UUID_RE.match(value))


def build_citation_tombstone(
    superseded_id: str,
    replacement_id: str,
    paths: list[str],
    run_id: str | None = None,
) -> dict[str, Any]:
    """Build the old->new forwarding record for ``X_CITATION_TOMBSTONE_KEY``.

    Written in the SAME ``update_task`` call as the repoint itself, so a task
    can never end up rewritten-but-unlabelled. It preserves the provenance the
    repoint would otherwise destroy — which id used to be cited, which id
    replaced it, exactly where, and in which run — while confining every
    remaining occurrence of the dead id to one explicitly-labelled field.

    Raises ``ValueError`` when ``replacement_id`` is not concrete
    (see :func:`is_concrete_memory_id`). This refusal is the point: a tombstone
    whose replacement is a search instruction would preserve the dangling
    pointer under a new name rather than close it.
    """
    if not is_concrete_memory_id(replacement_id):
        raise ValueError(
            'replacement_memory_id must be a concrete 36-char UUID, got '
            f'{replacement_id!r}. A search instruction (e.g. '
            "'re-derive ... via search(query=...)') is never an acceptable "
            'forwarding pointer — it re-derives to the superseded entries the '
            'consolidation was collapsing.'
        )
    return {
        'superseded_memory_id': superseded_id,
        'replacement_memory_id': replacement_id,
        'paths': list(paths),
        'run_id': run_id,
    }


async def repoint_task_citations(
    task_interceptor: Any,
    project_root: str,
    *,
    memory_id: str,
    replacement_id: str,
    run_id: str | None = None,
    tasks: list[Any] | None = None,
    log: logging.Logger | None = None,
) -> dict[str, Any]:
    """Repoint every LIVE task-metadata citation of ``memory_id``, before a delete.

    One task-tree snapshot yields both worklists, partitioned in Python by
    ``INACTIVE_TASK_STATUSES`` (``task_filter``'s ``{'done', 'cancelled'}``):

    - **non-terminal** citers are REPOINTED: their metadata is rewritten to
      ``replacement_id`` and a :func:`build_citation_tombstone` record is
      appended, in ONE ``update_task(..., metadata_mode='merge')`` write, so a
      task can never end up rewritten-but-unlabelled;
    - **terminal** citers are REPORTED into ``terminal_citations`` and never
      written. Rewriting a done task's record would falsify history, and
      ``recon_write_policy`` Gate 1 refuses recon-stage writes to terminal
      tasks anyway. Surfacing them keeps a terminal dangler visible instead of
      silent.

    ``tasks`` is that snapshot. A caller that has ALREADY read the task tree —
    the ``delete_memory`` citation gate, which must scan before it can decide a
    repoint is needed — passes its list in, so the gate and this sweep provably
    operate on one snapshot rather than two reads that can diverge between the
    "who cites this?" decision and the rewrite. When omitted, one unfiltered
    ``get_tasks(project_root)`` read is performed here. Either way it is a
    SINGLE unfiltered read rather than two status-filtered ones, so both
    worklists come from the same snapshot and pending tasks are guaranteed
    covered — the tasks the incident's manual pass missed were pending, i.e.
    still dispatchable, i.e. the ones that mattered.

    Per the module header's provenance rule, "cites" is
    :func:`find_live_citation_occurrences` (tombstone ledger excluded), and the
    rewrite runs over the same tombstone-free view — which is what makes a
    retry after a partial failure genuinely idempotent: an already-repointed
    task is not a citer, so it issues NO write, so the ledger cannot grow.
    Existing records are carried forward by :func:`repoint_tombstone_chain`,
    which forwards only a stale ``replacement_memory_id`` destination.

    Every rewrite is verified BEFORE its write: the blob the shallow merge will
    actually produce is re-scanned, and any surviving live occurrence of
    ``memory_id`` (a colliding uuid KEY, a top-level uuid key a merge cannot
    remove, any shape the rewriter does not handle) is recorded as a repoint
    FAILURE and the write is skipped. So "repointed" means checked, not
    assumed, and a blob that cannot be fully repointed is never half-written.

    Deliberate residual: a task carrying ONLY a stale tombstone destination
    (``replacement_memory_id == memory_id``) and no live citation is not a
    citer, so this sweep does not repair it. That is a provenance-chain
    degradation, not a live dispatch pointer — nothing reads it to decide what
    to do — and widening the citer definition to cover it would put a write
    back on the no-live-citer path and destroy the idempotency above.

    **Merge is shallow last-write-wins** (see ``update_task``'s
    ``metadata_mode`` contract): supplied keys overwrite wholesale and omitted
    keys are preserved. So the payload carries WHOLE top-level key values —
    repointing ``memory_hints.queries[0]`` resends the entire ``memory_hints``
    value, never a nested fragment, or the write would wipe its siblings. An
    explicit ``metadata_mode`` is mandatory rather than stylistic: a bare
    ``append=False`` metadata write is rejected by the backend
    (``_resolve_metadata_mode``) after the task-2180 metadata-wipe incident.

    Write success is read via ``interceptor_write_succeeded``, NOT try/except:
    interceptor gates REFUSE BY RETURNING ``{'success': False, 'error_type':
    ...}`` (e.g. ``recon_write_policy``'s ``ReconTerminalWriteRejected``) and
    never raise, so a truthy-dict check would mistake a rejection for a success
    and let the delete proceed.

    Error posture, deliberately asymmetric:

    - a **per-task** write failure (returned rejection, raised error, or a
      failed residual check) is tallied into ``unrepointed`` and the sweep
      continues, per the prevailing best-effort sweep convention
      (``stale_status_snapshot_edge_sweep``);
    - the **bulk** ``get_tasks`` read failure PROPAGATES. It means "unknown",
      and unknown must not be read as "no citations" when the caller is about
      to perform an irreversible delete — the caller fails closed on it.

    Known limitation: ``get_tasks`` defaults to a single tag, so a multi-tag
    project would need ``list_tags()`` aggregation. Out of scope here.

    Returns a stats dict whose keys are ALWAYS present, on every path. Scalar
    counters carry the module's ``stage1_*`` prefix (as
    :func:`verify_cited_memories`'s do), because they are merged into the same
    flat Stage-1 stats block where an unprefixed name would collide; the
    detail LISTS (``terminal_citations``, ``unrepointed``) are consumed
    structurally by the delete gate and stay unprefixed.
    """
    log = log or logger
    stats: dict[str, Any] = {
        'stage1_citation_tasks_scanned': 0,
        'stage1_citations_repointed': 0,
        'stage1_citation_tasks_repointed': 0,
        'stage1_citation_repoint_failures': 0,
        'stage1_terminal_citations_reported': 0,
        'terminal_citations': [],
        'unrepointed': [],
    }

    if tasks is None:
        # Deliberately NOT wrapped: a failed bulk read must reach the caller so
        # it can refuse the delete rather than treat 'unknown' as 'nothing to do'.
        tasks_data = await task_interceptor.get_tasks(project_root)
        tasks = (tasks_data or {}).get('tasks') or []
    if not isinstance(tasks, list):
        return stats

    for task in tasks:
        if not isinstance(task, dict):
            continue
        stats['stage1_citation_tasks_scanned'] += 1
        metadata = task.get('metadata')
        # LIVE occurrences only: the tombstone ledger names dead ids by design,
        # so scanning it would make an already-repointed task a citer forever.
        paths = find_live_citation_occurrences(metadata, memory_id)
        if not paths or not isinstance(metadata, dict):
            # find_live_citation_occurrences only reports paths for a dict blob,
            # so the isinstance is redundant at runtime — it is the narrowing the
            # rewrite below needs to be provably safe rather than incidentally so.
            continue

        task_id = str(task.get('id'))
        status = task.get('status')

        if status in INACTIVE_TASK_STATUSES:
            stats['terminal_citations'].append(
                {'task_id': task_id, 'status': status, 'paths': paths},
            )
            stats['stage1_terminal_citations_reported'] += 1
            continue

        try:
            # Rewrite the LIVE half of the blob only (module header: a
            # tombstone is provenance, never a live pointer).
            live_view = {
                key: value
                for key, value in metadata.items()
                if key != X_CITATION_TOMBSTONE_KEY
            }
            repointed, count = repoint_metadata(live_view, memory_id, replacement_id)
            tombstone = build_citation_tombstone(
                superseded_id=memory_id,
                replacement_id=replacement_id,
                paths=paths,
                run_id=run_id,
            )
            # Existing tombstones must be resent: merge overwrites the key
            # wholesale, so omitting them would drop prior provenance. Each is
            # forwarded through repoint_tombstone_chain, which repoints ONLY a
            # stale replacement_memory_id destination — a prior record
            # forwarding to the id being deleted would otherwise park a
            # dangling pointer in a field literally named
            # replacement_memory_id, the exact harm this sweep prevents merely
            # relocated. superseded_memory_id, paths and run_id are preserved
            # byte-identical, so the appended record below is the only place
            # the collapsed hop is recorded and the ledger cannot grow on a
            # retry (a fully-repointed task is not a citer at all, so it never
            # reaches this branch).
            tombstones = repoint_tombstone_chain(
                metadata.get(X_CITATION_TOMBSTONE_KEY), memory_id, replacement_id,
            )
            tombstones.append(tombstone)

            # Send ONLY the top-level keys that actually changed, each as a
            # WHOLE value (shallow merge), so untouched keys are preserved by
            # omission rather than by round-tripping them through JSON.
            changed_top_level = {
                key: value
                for key, value in repointed.items()
                if value != metadata.get(key)
            }
            changed_top_level[X_CITATION_TOMBSTONE_KEY] = tombstones

            # VERIFY BEFORE WRITING, on the blob the shallow merge will
            # actually produce. Anything the rewriter could not reach — a uuid
            # KEY whose destination already exists, a TOP-LEVEL uuid key (merge
            # adds the new key but cannot delete the old one), any future shape
            # — still names the doomed id here. That is a repoint FAILURE, not
            # a success: report it and skip the write, so the gate refuses the
            # delete instead of landing it over a half-repointed blob.
            residual = find_live_citation_occurrences(
                {**metadata, **changed_top_level}, memory_id,
            )
            if residual:
                log.warning(
                    'citation repoint incomplete for task %s (%s -> %s): '
                    '%d occurrence(s) survive the rewrite at %s',
                    task_id, memory_id, replacement_id, len(residual), residual,
                )
                stats['stage1_citation_repoint_failures'] += 1
                stats['unrepointed'].append({
                    'task_id': task_id,
                    'status': status,
                    'paths': paths,
                    'error': (
                        f'{len(residual)} occurrence(s) of {memory_id} survive '
                        f'the rewrite at {residual}; no write was issued'
                    ),
                    'error_type': 'CitationResidualAfterRewrite',
                    'residual_paths': residual,
                })
                continue

            resp = await task_interceptor.update_task(
                task_id=task_id,
                project_root=project_root,
                metadata=json.dumps(changed_top_level),
                metadata_mode='merge',
                agent_id=REPOINT_AGENT_ID,
            )
        except (KeyboardInterrupt, SystemExit):
            raise
        except Exception as exc:  # noqa: BLE001 — best-effort per-task sweep
            log.warning(
                'citation repoint failed for task %s (%s -> %s): %s',
                task_id, memory_id, replacement_id, exc,
            )
            stats['stage1_citation_repoint_failures'] += 1
            stats['unrepointed'].append({
                'task_id': task_id,
                'status': status,
                'paths': paths,
                'error': str(exc),
                'error_type': type(exc).__name__,
            })
            continue

        if interceptor_write_succeeded(resp):
            stats['stage1_citation_tasks_repointed'] += 1
            stats['stage1_citations_repointed'] += count
        else:
            # A REFUSAL, not an exception — see the docstring.
            log.warning(
                'citation repoint rejected for task %s (%s -> %s): %s',
                task_id, memory_id, replacement_id, resp,
            )
            stats['stage1_citation_repoint_failures'] += 1
            stats['unrepointed'].append({
                'task_id': task_id,
                'status': status,
                'paths': paths,
                'error': (resp or {}).get('error') if isinstance(resp, dict) else None,
                'error_type': (
                    (resp or {}).get('error_type') if isinstance(resp, dict) else None
                ),
            })

    return stats
