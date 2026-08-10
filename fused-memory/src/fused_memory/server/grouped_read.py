"""Server-side grouped reads: canonical + amendment digests + sighting count.

An amendment or sighting (task 3195's ``KIND_REGISTRY`` members) is an
ordinary Mem0 entry with its own UUID whose ``metadata`` carries
``parent_id`` — the ADD-ONLY child representation.  Read back naively, those
children are indistinguishable peers of the canonical they attach to, which
is the esc-5541 failure mode: untagged survivors outrank the entry they
amend.  This module composes the three landed Mem0 read primitives
(``count_memories_by_metadata`` / ``get_memories_by_metadata`` /
``get_memory_by_id``) into ONE grouped document per canonical — bounded
amendment digests plus an EXACT sighting count.

LAYERING CONSTRAINT (load-bearing — see ``server/tools.py``'s call sites and
``docs/prds/memory-metadata-vocabulary.md``):

    The child-suppression filter lives HERE and is called ONLY from the MCP
    tool wrappers.  ``MemoryService.search`` is deliberately NOT touched.
    Pushing the filter down into the service would strip child records from
    ``reconciliation/mem0_dedup.py::find_prior_memories``, whose per-record
    ``task_id``/``kind`` post-filter iterates RAW service results — hiding
    duplicates from the recon dedup detector and candidates from the
    near-duplicate write guard.  A live test pins this, not just this
    comment.

Shaped as a sibling of ``server/near_duplicate_guard.py``: duck-typed
service parameter, no runtime ``MemoryService`` import for typing, caps as
named module constants.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from fused_memory.models.enums import MemoryCategory, SourceStore
from fused_memory.models.memory import MemoryResult

# The canonical 'data' → 'memory' → 'content' payload-key fallback that turns
# a raw Qdrant payload into its text.  IMPORTED rather than re-declared: the
# search path and the point-id path already disagree about which key holds
# the text, and a third private copy would be a fourth place to drift (INV-5).
from fused_memory.services.memory_service import _MEM0_CONTENT_KEYS
from fused_memory.utils.async_utils import gather_collect

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

logger = logging.getLogger(__name__)

#: The two ``KIND_REGISTRY`` members that attach to a parent (task 3195).
AMENDMENT_KIND = 'amendment'
SIGHTING_KIND = 'sighting'
CHILD_KINDS = frozenset({AMENDMENT_KIND, SIGHTING_KIND})

#: The metadata key carrying the parent's full-UUID id (shape-validated by
#: ``memory_metadata``'s ``invalid_parent_id_shape``, liveness-validated at
#: the write seam by task 3197).
PARENT_ID_KEY = 'parent_id'

#: Max characters of an amendment body carried in a digest.  A digest is a
#: pointer, not a copy — the full text stays one ``get_memory_by_id`` away.
_DIGEST_CHARS = 240

#: Appended when a body was cut, so a truncated digest is never mistaken for
#: the whole amendment.
_DIGEST_ELLIPSIS = '…'

#: Max digests listed for one canonical.  A wider fan-out is reported by the
#: EXACT count plus a ``truncated`` marker rather than by a longer list.
_AMENDMENT_DIGEST_CAP = 10

#: Set on a grouped block whose child reads FAILED.  Its presence is the record
#: that the grouping is INCOMPLETE — a block carrying it is never used to
#: suppress a child, because we never actually learned what that parent's
#: children are.
CHILDREN_UNAVAILABLE_KEY = 'children_unavailable'

#: Where a SWALLOWED child hit is pinned into its parent's block, carrying its
#: FULL body.  Deliberately separate from ``amendments`` so a sighting is never
#: misfiled as an amendment and ``amendment_count`` / ``truncated`` keep meaning
#: exactly what the count API said.  See :func:`_pin_matched_child`.
MATCHED_CHILDREN_KEY = 'matched_children'

#: Metadata flag marking a child that CONTESTS its parent rather than merely
#: extending it.  This module is the single home of the predicate below because
#: no landed vocabulary key carries adjudication state and the producer (leaf
#: γ's write-triage judge) has not shipped — so without a read-side definition
#: the D6 carve-out would be untestable.  The ``x_`` experimental namespace
#: (``memory_metadata.EXPERIMENTAL_KEY_PREFIX``) is deliberate: it needs no
#: amendment to task 3195's closed five-key ``RESERVED_VOCABULARY_KEYS`` set
#: and produces no unknown-key census noise, while giving leaf γ exactly ONE
#: constant to stamp.
CONTESTED_METADATA_KEY = 'x_contested'


def is_contested_child(meta: Mapping[str, Any] | None) -> bool:
    """True when *meta* marks a child as CONTESTING its parent.

    A contested child is never suppressed into a digest: demoting a correction
    to truncated text under the very entry it contests is the esc-5712
    five-week-wrong-appendix shape.
    """
    return bool((meta or {}).get(CONTESTED_METADATA_KEY))


def _child_text(payload: Mapping[str, Any]) -> str:
    """First non-empty string among the canonical mem0 content keys."""
    for key in _MEM0_CONTENT_KEYS:
        value = payload.get(key)
        if isinstance(value, str) and value:
            return value
    return ''


def _digest(text: str) -> str:
    """Truncate *text* to ``_DIGEST_CHARS``, marking a cut with an ellipsis."""
    if len(text) <= _DIGEST_CHARS:
        return text
    return text[:_DIGEST_CHARS] + _DIGEST_ELLIPSIS


def _digest_entry(row: Mapping[str, Any]) -> dict[str, Any]:
    """Build one digest entry from a scrolled child row."""
    payload = row.get('metadata') or {}
    entry = {
        'id': row.get('id'),
        'digest': _digest(_child_text(payload)),
        'created_at': row.get('created_at'),
        'kind': payload.get('kind', AMENDMENT_KIND),
    }
    # A contested child appears in BOTH places — here AND as a surviving
    # top-level hit — which is strictly more informative than either alone.
    # Omitted (never False) when uncontested, per the fault-only convention.
    if is_contested_child(payload):
        entry['contested'] = True
    return entry


def _digest_sort_key(row: Mapping[str, Any]) -> tuple[str, str]:
    """Deterministic (created_at, id) ordering; a missing timestamp sorts first.

    Coerced to str so a ``None`` created_at cannot raise comparing None to str
    — two runs over the same corpus must agree on the ORDER of the rows they
    were given.

    This orders the sample; it does not choose it.  The backend caps the scroll
    at ``_AMENDMENT_DIGEST_CAP`` BEFORE this module ever sorts, so WHICH
    cap-sized subset of a wide fan-out comes back is arbitrary.  That
    arbitrariness is precisely why a swallowed child can never be assumed
    present in the digest list — see :func:`_pin_matched_child`.
    """
    return (row.get('created_at') or '', row.get('id') or '')


async def build_grouped_document(
    service: Any,
    project_id: str,
    canonical_id: str,
) -> dict[str, Any] | None:
    """Build the grouped block for *canonical_id*, or ``None`` when childless.

    Returns ``{'amendments': [...], 'amendment_count': int, 'sighting_count':
    int}``.  The sighting count always comes from the EXACT count API — never
    ``len(scroll)``, which would silently under-report a fan-out wider than
    the scroll limit.

    COST ORDERING (mirrors task 3197's delete gate, for the same reason): the
    un-kinded TOTAL child count runs FIRST and short-circuits on zero, so the
    common path — every canonical on today's corpus, where ``parent_id`` has
    zero live footprint — pays one cheap exact count and issues no scroll at
    all.  The detail reads are paid only when they buy something the total
    cannot: the per-kind split and the child bodies the digests need.

    FAULT CONTAINMENT: a failed child read returns
    ``{'children_unavailable': True, 'error_type': ...}`` — never ``None``
    (which is the wire shape for "genuinely childless") and never a fabricated
    zero.  ``Mem0Backend`` propagates a Qdrant read-timeout as ``TimeoutError``
    exactly so this distinction survives; collapsing it here would tell an
    operator a canonical has no children because a backend timed out.  Nothing
    partial is returned alongside the marker, so no consumer can read half a
    grouping as a whole one.
    """
    try:
        return await _read_grouped_document(service, project_id, canonical_id)
    except Exception as exc:
        logger.warning(
            'grouped_read: child reads FAILED for canonical_id=%s in project=%s; '
            'reporting children_unavailable rather than a silent "no children"',
            canonical_id,
            project_id,
            exc_info=True,
            extra={'project_id': project_id, 'canonical_id': canonical_id},
        )
        return {CHILDREN_UNAVAILABLE_KEY: True, 'error_type': type(exc).__name__}


async def _read_grouped_document(
    service: Any,
    project_id: str,
    canonical_id: str,
) -> dict[str, Any] | None:
    """The child reads themselves — see :func:`build_grouped_document`.

    Raises whatever the backend raises; the caller owns fault containment.
    """
    total_children = await service.count_memories_by_metadata(
        project_id=project_id,
        filters={PARENT_ID_KEY: canonical_id},
    )
    if not total_children:
        return None
    sighting_count = await service.count_memories_by_metadata(
        project_id=project_id,
        filters={PARENT_ID_KEY: canonical_id, 'kind': SIGHTING_KIND},
    )
    amendment_count = await service.count_memories_by_metadata(
        project_id=project_id,
        filters={PARENT_ID_KEY: canonical_id, 'kind': AMENDMENT_KIND},
    )
    rows = await service.get_memories_by_metadata(
        project_id=project_id,
        filters={PARENT_ID_KEY: canonical_id, 'kind': AMENDMENT_KIND},
        limit=_AMENDMENT_DIGEST_CAP,
    )
    digests = [_digest_entry(row) for row in sorted(rows or [], key=_digest_sort_key)]
    # A non-zero total means children EXIST, so the block is emitted even when
    # none of them is an amendment or a sighting — reporting an empty group for
    # a parent that demonstrably has children would be the silent under-report
    # the total probe exists to prevent.
    block: dict[str, Any] = {
        'amendments': digests,
        'amendment_count': amendment_count,
        'sighting_count': sighting_count,
    }
    # Marked whenever the EXACT count outruns what the scroll returned — which
    # covers both the cap and a write landing between the two reads, exactly as
    # ``DescendantScan.truncated`` does on the delete path.  Omitted (never
    # False) when the listing is complete, matching the search tool's
    # fault-only degraded-key convention.
    if amendment_count > len(digests):
        block['truncated'] = True
    return block


def _parent_id_in_meta(meta: Mapping[str, Any] | None) -> str | None:
    """The parent id *meta* points at, or ``None`` when it is not a child's.

    A child is identified STRICTLY by ``metadata.parent_id`` plus a child
    ``kind`` (D5): a shared ``topic`` is not a grouping key, and two entries
    that merely discuss the same subject stay independent peers.
    """
    meta = meta or {}
    if meta.get('kind') not in CHILD_KINDS:
        return None
    parent_id = meta.get(PARENT_ID_KEY)
    if isinstance(parent_id, str) and parent_id:
        return parent_id
    return None


def _child_parent_id(result: Any) -> str | None:
    """:func:`_parent_id_in_meta` for a search hit."""
    return _parent_id_in_meta(getattr(result, 'metadata', None))


def _is_mem0(result: Any) -> bool:
    """True for a Mem0-sourced hit — the only kind that can BE a parent.

    ``parent_id`` liveness resolves through ``Mem0Backend.get_point_by_id`` at
    the write seam (task 3197), so a Graphiti edge uuid can never be a live
    parent.  Probing one would spend a guaranteed-zero Qdrant count per graph
    hit on every search.
    """
    return getattr(result, 'source_store', None) == SourceStore.mem0


def _promoted_parent(parent_id: str, record: Mapping[str, Any]) -> MemoryResult:
    """Turn a raw ``get_memory_by_id`` record into a search-hit-shaped parent."""
    payload = record.get('metadata') or {}
    raw_category = payload.get('category')
    try:
        category = MemoryCategory(raw_category) if isinstance(raw_category, str) else None
    except ValueError:
        category = None
    created_at = payload.get('created_at')
    return MemoryResult(
        id=parent_id,
        content=record.get('content') or '',
        category=category,
        source_store=SourceStore.mem0,
        metadata=dict(payload),
        created_at=created_at if isinstance(created_at, str) else None,
    )


async def _resolve_absent_parents(
    service: Any,
    project_id: str,
    parent_ids: list[str],
) -> dict[str, MemoryResult]:
    """Fetch each parent NOT already present in the hit list, once, concurrently.

    A parent that does not resolve is simply absent from the returned mapping —
    the caller keeps that child as a top-level hit rather than dropping it, so
    a dangling pointer can never make a child's content unreachable.
    """
    if not parent_ids:
        return {}
    records = await gather_collect(
        service.get_memory_by_id(project_id=project_id, memory_id=parent_id)
        for parent_id in parent_ids
    )
    resolved: dict[str, MemoryResult] = {}
    for parent_id, record in zip(parent_ids, records, strict=True):
        if isinstance(record, Exception):
            logger.warning(
                'grouped_read: parent lookup FAILED for parent_id=%s in project=%s; '
                'keeping the child as a top-level hit',
                parent_id,
                project_id,
                exc_info=record,
                extra={'project_id': project_id, 'parent_id': parent_id},
            )
            continue
        if record:
            resolved[parent_id] = _promoted_parent(parent_id, record)
    return resolved


def _carve_outs_allow_suppression(
    child_meta: Mapping[str, Any] | None,
    parent_base: Any,
    parent_block: Mapping[str, Any] | None,
) -> bool:
    """The carve-outs that do NOT depend on how the block was populated.

    Shared by :func:`_suppress_child` (the decision) and the pinning pass in
    :func:`group_search_results` (which must know the same thing BEFORE the
    block is populated), so the two cannot disagree about who is a candidate:

    * UNRESOLVABLE PARENT — a dangling ``parent_id`` must never make a child's
      content unreachable, so an unresolved parent suppresses nothing;
    * CONTESTED (D6 / esc-5712) — a correction must never be demoted to a
      truncated digest under the entry it contests;
    * FAILED or EMPTY GROUPING — we never learned what that parent's children
      are (a backend fault, or a count that disagrees with the hit in hand), so
      we cannot claim this child is represented there.
    """
    if parent_base is None:
        return False
    if is_contested_child(child_meta):
        return False
    grouping_is_unusable = not parent_block or bool(
        parent_block.get(CHILDREN_UNAVAILABLE_KEY)
    )
    return not grouping_is_unusable


def _pin_matched_child(block: dict[str, Any], hit: Any) -> None:
    """Record a child about to be SWALLOWED inside its parent's grouped block.

    Suppressing a child is only honest when the block that replaces it actually
    carries it, and two live paths break that: a matched amendment beyond
    ``_AMENDMENT_DIGEST_CAP`` (the retained sample is arbitrary — see
    :func:`_digest_sort_key`) and ANY matched sighting, which the
    amendment-only digest scroll can never list.  In both, the matched id and
    its text would vanish from the response entirely.

    The body is taken from the hit ALREADY IN HAND — zero extra backend reads —
    and carried in FULL rather than truncated to ``_DIGEST_CHARS``: that text
    was in the response before grouping, so cutting it here would BE the
    retrieval regression.  Each pinned body appears exactly once, so the
    grouped response can never exceed the ungrouped one it replaces.

    Pins land in a SEPARATE ``matched_children`` list, never in ``amendments``:
    a sighting is not an amendment, and ``amendment_count`` / ``truncated``
    must keep meaning exactly what the count API said.  When the child is
    already listed as a digest (the common in-cap amendment case) that entry is
    marked in place instead of being duplicated.

    Mutates only *block* — a dict this call's :func:`build_grouped_document`
    just built, never a shared or cached structure.
    """
    child_id = getattr(hit, 'id', None)
    if not isinstance(child_id, str) or not child_id:
        # Not pinnable and not verifiable; the caller keeps such a hit.
        return
    for digest in block.get('amendments') or []:
        if digest.get('id') == child_id:
            digest['matched'] = True
            return
    pinned: list[dict[str, Any]] = block.setdefault(MATCHED_CHILDREN_KEY, [])
    if any(entry.get('id') == child_id for entry in pinned):
        return
    meta = getattr(hit, 'metadata', None) or {}
    pinned.append(
        {
            'id': child_id,
            'content': getattr(hit, 'content', '') or '',
            'created_at': getattr(hit, 'created_at', None),
            'kind': meta.get('kind'),
            'matched': True,
        }
    )


def _suppress_child(
    child_meta: Mapping[str, Any] | None,
    parent_base: Any,
    parent_block: Mapping[str, Any] | None,
) -> bool:
    """The SINGLE gate every child-suppression decision passes through.

    Every carve-out lives here, in one place, so a future edit cannot bypass
    one; see :func:`_carve_outs_allow_suppression` for what they are.
    """
    return _carve_outs_allow_suppression(child_meta, parent_base, parent_block)


async def group_search_results(
    service: Any,
    project_id: str,
    results: Sequence[Any],
) -> list[dict[str, Any]]:
    """Collapse child hits into their parents' grouped documents, in rank order.

    * A child whose parent is ALSO a hit folds into it — the esc-5541 failure
      mode where untagged survivors outrank the canonical they amend.
    * A CHILD-ONLY match resolves UPWARD: the parent's grouped document takes
      the child's rank slot (D6), so a child's content is never unreachable.
    * A parent reached both ways is emitted EXACTLY once, keeping the higher
      ``relevance_score``.
    * A child the grouping cannot account for — contested, dangling parent, or
      failed parent grouping — survives untouched; see :func:`_suppress_child`.
    * Every child that IS swallowed is first pinned into its parent's block
      with its full body (:func:`_pin_matched_child`), so a suppressed child's
      text is never lost from the response.

    The list only ever shrinks, and survivors keep their relative order.

    .. warning::
        ``SearchResults.degraded`` / ``failed_stores`` / ``failure_diagnostics``
        do NOT survive a list transform (see ``memory_service.SearchResults``'s
        own warning) — this function returns a plain ``list``.  Read those
        attributes BEFORE calling it.
    """
    hits = list(results)
    hit_by_id: dict[str, Any] = {}
    for hit in hits:
        hit_id = getattr(hit, 'id', None)
        if isinstance(hit_id, str) and hit_id not in hit_by_id:
            hit_by_id[hit_id] = hit

    child_parents = {
        index: parent_id
        for index, hit in enumerate(hits)
        if (parent_id := _child_parent_id(hit)) is not None
    }
    absent = [
        parent_id
        for parent_id in dict.fromkeys(child_parents.values())
        if parent_id not in hit_by_id
    ]
    resolved = await _resolve_absent_parents(service, project_id, absent)

    # Grouped documents are built BEFORE any suppression decision, because
    # "did this parent's grouping actually succeed?" is itself a carve-out:
    # a child must never be dropped on the strength of a grouping we failed to
    # read.  Candidates are the parents children point at, plus every
    # Mem0-sourced non-child hit (a child has no children — the model is flat).
    groupable = list(
        dict.fromkeys(
            [
                hit_id
                for hit in hits
                if _is_mem0(hit)
                and _child_parent_id(hit) is None
                and isinstance(hit_id := getattr(hit, 'id', None), str)
            ]
            + [
                parent_id
                for parent_id in dict.fromkeys(child_parents.values())
                if parent_id in hit_by_id or parent_id in resolved
            ]
        )
    )
    blocks = await gather_collect(
        build_grouped_document(service, project_id, target_id) for target_id in groupable
    )
    block_by_id: dict[str, Any] = {}
    for target_id, block in zip(groupable, blocks, strict=True):
        if isinstance(block, Exception):
            # build_grouped_document contains its own faults; this is
            # belt-and-braces so a future edit there cannot turn a working
            # search into an exception.
            logger.warning(
                'grouped_read: grouping FAILED for canonical_id=%s in project=%s',
                target_id,
                project_id,
                exc_info=block,
                extra={'project_id': project_id, 'canonical_id': target_id},
            )
            block = {CHILDREN_UNAVAILABLE_KEY: True, 'error_type': type(block).__name__}
        block_by_id[target_id] = block

    # PIN every child that is about to be swallowed into its parent's block,
    # BEFORE any suppression decision is taken.  Suppressing a child is only
    # honest when the block that replaces it demonstrably carries it, and the
    # digest list alone does not: a matched amendment beyond the cap or ANY
    # matched sighting is absent from it.  Pinning satisfies both goals at once
    # — the canonical still absorbs its children (esc-5541) and the matched
    # text stays reachable (D6) — at zero extra backend reads.
    for index, hit in enumerate(hits):
        parent_id = child_parents.get(index)
        if parent_id is None:
            continue
        parent_base = hit_by_id.get(parent_id) or resolved.get(parent_id)
        block = block_by_id.get(parent_id)
        if block is None or not _carve_outs_allow_suppression(
            getattr(hit, 'metadata', None), parent_base, block
        ):
            continue
        _pin_matched_child(block, hit)

    ordered: list[str] = []
    entries: dict[str, dict[str, Any]] = {}
    for index, hit in enumerate(hits):
        parent_id = child_parents.get(index)
        parent_base = (hit_by_id.get(parent_id) or resolved.get(parent_id)) if parent_id else None
        child_meta = getattr(hit, 'metadata', None) if parent_id else None
        if parent_id and _suppress_child(child_meta, parent_base, block_by_id.get(parent_id)):
            # _suppress_child's UNRESOLVABLE-PARENT carve-out already returned
            # False for a None parent_base; restated so the checker sees it too.
            assert parent_base is not None
            target_id, base, parent_unresolved = parent_id, parent_base, False
        else:
            # Not a child, or a child a carve-out keeps: it stays a top-level
            # hit in its own right.  A hit with no usable str id can be neither
            # deduped nor grouped (mirroring the isinstance guards above), so it
            # takes a per-position key that cannot collide with a real id —
            # two id-less hits must never collapse into a single entry.
            hit_id = getattr(hit, 'id', None)
            target_id = hit_id if isinstance(hit_id, str) else f'\x00no-id:{index}'
            base = hit
            parent_unresolved = bool(parent_id) and parent_base is None
        score = getattr(hit, 'relevance_score', 0.0)
        existing = entries.get(target_id)
        if existing is not None:
            existing['relevance_score'] = max(existing['relevance_score'], score)
            continue
        entry = base.model_dump()
        entry['relevance_score'] = score
        if parent_unresolved:
            # Loud rather than silent: this hit points at a parent no store
            # could resolve, so its group is genuinely unknown — not empty.
            entry['parent_unresolved'] = True
        block = block_by_id.get(target_id)
        if block:
            entry['grouped'] = block
        entries[target_id] = entry
        ordered.append(target_id)

    return [entries[target_id] for target_id in ordered]


async def group_memory_document(
    service: Any,
    project_id: str,
    memory_id: str,
    record: Mapping[str, Any],
) -> dict[str, Any] | None:
    """The grouped block for an EXACT point-id read, or ``None`` when there is none.

    ADDITIVE ONLY, deliberately asymmetric with :func:`group_search_results`:

    * a CHILD keeps its own ``memory_id`` / ``content`` / ``metadata`` and gains
      only ``{'parent': {...}}``.  ``get_memory_by_id`` is an exact-point-id
      reader whose contract is load-bearing for
      ``reconciliation/citation_verifier.py``, recon stage 1 and
      ``server/recon_report.py::cite_memory`` — answering a child id with its
      parent's text would make a citation silently verify against different
      text, a worse bug than the ungrouped state;
    * anything else is treated as a canonical and gets
      :func:`build_grouped_document`.

    Search hits are ranked CANDIDATES rather than exact-id answers, which is
    why upward *replacement* is safe there and not here.  Both surfaces still
    satisfy D6 — a child's content is never unreachable — one by keeping the
    child, the other by surfacing the group.
    """
    parent_id = _parent_id_in_meta(record.get('metadata'))
    if parent_id is None:
        return await build_grouped_document(service, project_id, memory_id)
    parent: dict[str, Any] = {'id': parent_id}
    block = await build_grouped_document(service, project_id, parent_id)
    if block:
        parent.update(block)
    return {'parent': parent}
