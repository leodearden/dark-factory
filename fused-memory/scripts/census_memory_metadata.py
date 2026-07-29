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


def _value_sort_key(value: Any) -> tuple[int, Any]:
    """Total order over mixed value types so table ordering never depends on
    dict insertion order (numbers before strings, each sorted naturally)."""
    if isinstance(value, int) and not isinstance(value, bool):
        return (0, value)
    return (1, str(value))


def _table(counter: Counter[Any], top_n: int) -> dict[str, Any]:
    """Render a Counter as a capped, deterministically ordered table.

    Sorted by count descending, then by value ascending -- so a re-run over
    an unchanged corpus produces a byte-identical artifact. Truncation
    discloses itself (``distinct_total`` + ``truncated_values``) so a capped
    long tail is never mistaken for a complete one.
    """
    ordered = sorted(counter.items(), key=lambda kv: (-kv[1], _value_sort_key(kv[0])))
    distinct_total = len(ordered)
    return {
        'entries': [{'value': v, 'count': c} for v, c in ordered[:top_n]],
        'distinct_total': distinct_total,
        'truncated_values': distinct_total > top_n,
    }


def _census_to_dict(census: CategoryCensus, top_n: int) -> dict[str, Any]:
    """Render one CategoryCensus (cell or rollup) as a JSON-serialisable dict."""
    return {
        'records': census.records,
        'keys': _table(census.key_counts, top_n),
        'kind': _table(census.kind_counts, top_n),
        'kind_missing': census.kind_missing,
        'source': _table(census.source_counts, top_n),
        'source_without_kind': _table(census.source_without_kind, top_n),
        'topic_present': census.topic_present,
        'topic': _table(census.topic_values, top_n),
        'parent_id_present': census.parent_id_present,
        'canonical_true': census.canonical_true,
        'canonical_false': census.canonical_false,
        'canonical_non_bool': census.canonical_non_bool,
        'supersedes_shapes': _table(census.supersedes_shapes, top_n),
        'supersedes_member_shapes': _table(census.supersedes_member_shapes, top_n),
        'supersedes_list_lengths': _table(census.supersedes_list_lengths, top_n),
    }


def build_report(
    cells: dict[str, dict[str, CategoryCensus]],
    coverage: dict[str, dict[str, Any]],
    top_n: int = 50,
    page_size: int | None = None,
) -> dict[str, Any]:
    """Assemble the JSON census report from per-(project, category) cells.

    Args:
        cells: ``{project_id: {category_value: CategoryCensus}}``.
        coverage: ``{project_id: {'collection', 'collection_points',
            'categories': {category: {'expected', 'scrolled'}}}}`` as
            returned by :func:`census_project`.
        top_n: Cap on entries per value/key table.
        page_size: Scroll page size, recorded in ``params`` so a future
            re-run is comparable.

    Returns:
        A JSON-serialisable dict with per-cell counts, per-project and
        grand-total rollups, and a coverage block. Any enumeration shortfall
        -- a cell scrolling fewer points than ``count_by_metadata`` expected,
        or per-category counts summing short of the collection's point total
        -- sets ``coverage.complete=false`` and appears as a NAMED entry in
        ``coverage.deltas`` (INV-2 no-silent-fail-soft). The caller exits
        non-zero on that flag.
    """
    projects: dict[str, Any] = {}
    grand_total = CategoryCensus()
    category_order: list[str] = []

    for project_id in sorted(cells):
        project_cells = cells[project_id]
        project_total = CategoryCensus()
        rendered: dict[str, Any] = {}
        for category in _ordered_categories(project_cells):
            if category not in category_order:
                category_order.append(category)
            census = project_cells[category]
            rendered[category] = _census_to_dict(census, top_n)
            project_total.merge(census)
        grand_total.merge(project_total)
        projects[project_id] = {
            'total': _census_to_dict(project_total, top_n),
            'categories': rendered,
        }

    return {
        'schema_version': 1,
        'params': {
            'projects': sorted(cells),
            'categories': category_order,
            'page_size': page_size,
            'top_n': top_n,
        },
        'projects': projects,
        'grand_total': _census_to_dict(grand_total, top_n),
        'coverage': _build_coverage(coverage),
    }


def _ordered_categories(project_cells: dict[str, CategoryCensus]) -> list[str]:
    """CENSUS_CATEGORIES order first, then any unexpected category, sorted."""
    known = [c.value for c in CENSUS_CATEGORIES if c.value in project_cells]
    extra = sorted(k for k in project_cells if k not in known)
    return known + extra


def _build_coverage(coverage: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Reconcile expected vs scrolled per cell and per-category sums vs the
    collection point total, naming every shortfall."""
    deltas: list[dict[str, Any]] = []
    projects: dict[str, Any] = {}
    overall_complete = True

    for project_id in sorted(coverage):
        record = coverage[project_id]
        per_category: dict[str, Any] = {}
        counted = 0
        project_complete = True

        for category, counts in record.get('categories', {}).items():
            expected = int(counts.get('expected', 0))
            scrolled = int(counts.get('scrolled', 0))
            delta = scrolled - expected
            counted += expected
            cell_complete = delta == 0
            per_category[category] = {
                'expected': expected,
                'scrolled': scrolled,
                'delta': delta,
                'complete': cell_complete,
            }
            if not cell_complete:
                project_complete = False
                deltas.append({
                    'kind': 'category_shortfall',
                    'project_id': project_id,
                    'category': category,
                    'expected': expected,
                    'scrolled': scrolled,
                    'delta': delta,
                })

        collection_points = int(record.get('collection_points', 0))
        uncovered = collection_points - counted
        if uncovered != 0:
            project_complete = False
            deltas.append({
                'kind': 'uncovered_points',
                'project_id': project_id,
                'collection': record.get('collection'),
                'collection_points': collection_points,
                'counted': counted,
                'delta': uncovered,
            })

        projects[project_id] = {
            'collection': record.get('collection'),
            'collection_points': collection_points,
            'counted': counted,
            'uncovered_points': uncovered,
            'complete': project_complete,
            'categories': per_category,
        }
        overall_complete = overall_complete and project_complete

    return {
        'complete': overall_complete,
        'deltas': deltas,
        'projects': projects,
    }
