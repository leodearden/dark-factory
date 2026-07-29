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
from collections import Counter
from dataclasses import dataclass, field
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


@dataclass
class CategoryCensus:
    """Streaming accumulator for one (project, category) cell.

    Payloads are folded in one at a time by :meth:`add` and never retained,
    so peak memory is independent of corpus size (~49.4k entries measured
    live). :meth:`merge` combines cells into the per-project and grand-total
    rollups.

    VALUE-level counting is confined to the bounded vocabulary axes the PRD's
    V1 table names as load-bearing (``kind``, ``source``, ``topic``,
    ``canonical``). Every other key is presence-counted only -- unbounded
    value counting would put ~24k distinct ``task_id``/``run_id`` values into
    a committed artifact. ``supersedes`` is censused by SHAPE only, never by
    value, so no UUID pointers leak into the report either.
    """

    records: int = 0
    # Presence count per top-level payload key. Deliberately unfiltered:
    # mem0-managed keys (data/hash/created_at/user_id/...) are counted too.
    key_counts: Counter[str] = field(default_factory=Counter)

    kind_counts: Counter[str] = field(default_factory=Counter)
    kind_missing: int = 0

    source_counts: Counter[str] = field(default_factory=Counter)
    # Per-source breakdown of the source-set-but-kind-missing drift
    # documented at server/tools.py:1595-1597.
    source_without_kind: Counter[str] = field(default_factory=Counter)

    topic_present: int = 0
    topic_values: Counter[str] = field(default_factory=Counter)
    parent_id_present: int = 0

    canonical_true: int = 0
    canonical_false: int = 0
    canonical_non_bool: int = 0

    supersedes_shapes: Counter[str] = field(default_factory=Counter)
    supersedes_member_shapes: Counter[str] = field(default_factory=Counter)
    supersedes_list_lengths: Counter[int] = field(default_factory=Counter)

    def add(self, payload: dict[str, Any]) -> None:
        """Fold one Qdrant payload into the counters. Does not mutate *payload*.

        Mem0 stores ``add_memory(metadata=...)`` fields as TOP-LEVEL payload
        keys (mem0_client.py:309-311), so the payload dict *is* the metadata
        namespace -- there is no nested 'metadata' to unwrap.
        """
        self.records += 1
        for key in payload:
            self.key_counts[key] += 1

        kind = payload.get('kind')
        has_kind = 'kind' in payload and kind is not None
        if has_kind:
            self.kind_counts[str(kind)] += 1
        else:
            self.kind_missing += 1

        source = payload.get('source')
        if 'source' in payload and source is not None:
            self.source_counts[str(source)] += 1
            if not has_kind:
                self.source_without_kind[str(source)] += 1

        if 'topic' in payload and payload['topic'] is not None:
            self.topic_present += 1
            self.topic_values[str(payload['topic'])] += 1
        if 'parent_id' in payload and payload['parent_id'] is not None:
            self.parent_id_present += 1

        if 'canonical' in payload:
            canonical = payload['canonical']
            if isinstance(canonical, bool):
                if canonical:
                    self.canonical_true += 1
                else:
                    self.canonical_false += 1
            elif canonical is not None:
                self.canonical_non_bool += 1

        shape = classify_supersedes(payload)
        self.supersedes_shapes[shape] += 1
        if shape == 'scalar':
            # A bare string is one lone member -- counted so the scalar
            # population's member shapes are visible alongside the list ones.
            self.supersedes_member_shapes[classify_uuid_member(payload['supersedes'])] += 1
        elif shape == 'list':
            members = payload['supersedes']
            self.supersedes_list_lengths[len(members)] += 1
            for member in members:
                self.supersedes_member_shapes[classify_uuid_member(member)] += 1

    def merge(self, other: CategoryCensus) -> None:
        """Fold *other* into self. Does not mutate *other*."""
        self.records += other.records
        self.key_counts.update(other.key_counts)
        self.kind_counts.update(other.kind_counts)
        self.kind_missing += other.kind_missing
        self.source_counts.update(other.source_counts)
        self.source_without_kind.update(other.source_without_kind)
        self.topic_present += other.topic_present
        self.topic_values.update(other.topic_values)
        self.parent_id_present += other.parent_id_present
        self.canonical_true += other.canonical_true
        self.canonical_false += other.canonical_false
        self.canonical_non_bool += other.canonical_non_bool
        self.supersedes_shapes.update(other.supersedes_shapes)
        self.supersedes_member_shapes.update(other.supersedes_member_shapes)
        self.supersedes_list_lengths.update(other.supersedes_list_lengths)
