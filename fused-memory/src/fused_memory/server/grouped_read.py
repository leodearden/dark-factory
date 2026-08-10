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

# The canonical 'data' → 'memory' → 'content' payload-key fallback that turns
# a raw Qdrant payload into its text.  IMPORTED rather than re-declared: the
# search path and the point-id path already disagree about which key holds
# the text, and a third private copy would be a fourth place to drift (INV-5).
from fused_memory.services.memory_service import _MEM0_CONTENT_KEYS

if TYPE_CHECKING:
    from collections.abc import Mapping

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
    return {
        'id': row.get('id'),
        'digest': _digest(_child_text(payload)),
        'created_at': row.get('created_at'),
        'kind': payload.get('kind', AMENDMENT_KIND),
    }


def _digest_sort_key(row: Mapping[str, Any]) -> tuple[str, str]:
    """Deterministic (created_at, id) ordering; a missing timestamp sorts first.

    Coerced to str so a ``None`` created_at cannot raise comparing None to str
    — two runs over the same corpus must agree on digest order.
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
