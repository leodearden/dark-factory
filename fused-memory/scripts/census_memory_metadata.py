#!/usr/bin/env python3
"""One-shot READ-ONLY census of the Mem0 metadata population: which metadata
keys exist, which ``kind`` values are in use, and what shape ``supersedes``
pointers actually take -- per (project, category), rolled up per project and
across the whole corpus.

Motivation: PRD ``docs/prds/memory-metadata-vocabulary.md`` §9 leaf alpha.
The PRD's V1 vocabulary (a ``kind`` registry, a rejection rule for scalar
``supersedes``, a grandfather list for pre-existing writers) has to be seeded
from the *measured* live population rather than guesswork -- otherwise the
registry silently orphans an in-use ``kind`` and a validation rule breaks a
writer nobody remembered. Leaf beta cites this census's artifact.

Measured corpus facts that shape this script (live Qdrant, 2026-07-29):

  - ~49.4k entries across two collections, not the ~24k the PRD estimated
    (``fused_dark_factory`` 19,464; ``fused_reify`` 29,951). ``fused_reify``
    alone holds 24,408 ``observations_and_summaries``, so a single capped
    scroll would silently drop most of it -- real pagination is mandatory.
    ``Mem0Backend.scroll_by_metadata`` discards Qdrant's ``next_offset``
    (mem0_client.py:395) and exposes no offset parameter, so it structurally
    cannot page; this script drives the raw async Qdrant client instead
    (the established in-repo script pattern -- clear_malformed_empty_memory.py:334,
    consolidate_namespace_families.py:804) and loops on ``next_offset``.
    Pages are folded into counters as they arrive, so peak memory is one page.

  - Every Mem0 write is stamped with ``category`` (memory_service.py:2185-2190)
    and ``category``-absent measured 0 in both collections -- but the three
    Mem0-primary categories still do NOT cover the corpus: an 80-point residue
    carries a *Graphiti*-primary category from dual-write records. The census
    therefore enumerates all six ``MemoryCategory`` values and reconciles the
    sum against each collection's point count.

Any enumeration shortfall -- scrolled < ``count_by_metadata``, page budget
exhausted with a live ``next_offset``, or sum(per-category) < collection
total -- sets ``coverage.complete=false``, names the deltas in both artifacts
and exits non-zero. A census that under-reports is indistinguishable from a
census of a smaller corpus, and its consumer cannot tell the difference
(INV-2 structured-facts-at-failure / no-silent-fail-soft).

READ-ONLY: this script never writes, updates or deletes a memory. It issues
only Qdrant ``count`` and ``scroll`` calls, and writes two local report files.

Usage
-----
  # Default: census dark_factory + reify, write both artifacts under plans/.
  python scripts/census_memory_metadata.py

  # A single project, custom artifact paths.
  python scripts/census_memory_metadata.py --project-id dark_factory \\
      --json-out /tmp/census.json --md-out /tmp/census.md

  # Larger pages / a longer value tail in the report.
  python scripts/census_memory_metadata.py --page-size 2000 --top-n 100
"""

from __future__ import annotations

import logging
import re
from typing import Any

from fused_memory.models.enums import GRAPHITI_PRIMARY, MEM0_PRIMARY, MemoryCategory

logger = logging.getLogger('census_memory_metadata')

# All six categories, Mem0-primary first, derived from the shared enum --
# never a restated literal list (INV-5 no-lockstep-duplication). The three
# Graphiti-primary categories are included because dual-write records land
# them in the Mem0 collection too (80 points measured live); omitting them
# would leave a silently unenumerated slice of the corpus.
CENSUS_CATEGORIES: tuple[MemoryCategory, ...] = (
    *sorted(MEM0_PRIMARY),
    *sorted(GRAPHITI_PRIMARY),
)

_FULL_UUID_RE = re.compile(
    r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$',
    re.IGNORECASE,
)
_SHORT_HEX_RE = re.compile(r'^[0-9a-f]+$', re.IGNORECASE)


# ---------------------------------------------------------------------------
# Pure-function core (no I/O — fully testable without a live Qdrant)
# ---------------------------------------------------------------------------

def classify_supersedes(payload: dict[str, Any]) -> str:
    """Classify the SHAPE of a payload's ``supersedes`` value.

    Returns one of ``'absent'`` (key missing), ``'null'`` (explicit None),
    ``'scalar'`` (a bare string), ``'list'`` (a list, including the empty
    list) or ``'other'`` (anything else -- int, dict, bool, ...).

    PRD V1 rejects scalar ``supersedes`` in favour of a list, so the scalar
    population is exactly the size of the migration/grandfather problem.
    Does not mutate *payload*.
    """
    if 'supersedes' not in payload:
        return 'absent'
    value = payload['supersedes']
    if value is None:
        return 'null'
    # bool is a subclass of int, and both are 'other' -- check before str/list
    # so a stray True is never mistaken for a pointer.
    if isinstance(value, bool):
        return 'other'
    if isinstance(value, str):
        return 'scalar'
    if isinstance(value, list):
        return 'list'
    return 'other'


def classify_uuid_member(value: Any) -> str:
    """Classify the shape of a single ``supersedes`` member.

    Returns ``'full_uuid'`` for the canonical 36-char hyphenated rendering,
    ``'short_hex'`` for a bare hex string shorter than 32 characters (the
    malformed member shape PRD V1 says beta must reject), and ``'other'`` for
    a non-string or any other string.
    """
    if not isinstance(value, str):
        return 'other'
    if _FULL_UUID_RE.match(value):
        return 'full_uuid'
    if value and len(value) < 32 and _SHORT_HEX_RE.match(value):
        return 'short_hex'
    return 'other'
