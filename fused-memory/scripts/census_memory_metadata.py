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
    Mem0-primary categories still do NOT cover the corpus: a residue of 80
    points ACROSS BOTH collections (79 in ``fused_reify``, 1 in
    ``fused_dark_factory``) carries a *Graphiti*-primary category from
    dual-write records. The census therefore enumerates all six
    ``MemoryCategory`` values and reconciles the sum against each
    collection's point count.

Any enumeration shortfall -- scrolled matching neither the pre- nor the
post-scroll ``count_by_metadata``, page budget exhausted with a live
``next_offset``, or a per-category sum that cannot be reconciled with the
collection total -- sets ``coverage.complete=false``, names the deltas in
both artifacts and exits non-zero. A census that under-reports is
indistinguishable from a census of a smaller corpus, and its consumer
cannot tell the difference (INV-2 structured-facts-at-failure /
no-silent-fail-soft).

Because the corpus is LIVE, every read is bracketed so ordinary drift is
never charged as missing data -- at BOTH levels:

  - Per CATEGORY: counted BEFORE and AFTER its scroll. A scroll matching
    either count saw churn, not truncation.
  - Per COLLECTION: ``count(scope)`` is read before the category loop and
    again after it, giving a collection-points INTERVAL; the counted total
    is likewise the interval (sum expected .. sum recount). The project is
    covered when the two intervals intersect, and the reported residue is
    the signed GAP between them -- the minimum number of points churn
    cannot explain. Movement inside that window lands in ``coverage.churn``
    (kind ``collection_total_moved``), never in ``coverage.deltas``.

A residue that survives is still named: positive (``uncovered_points`` --
points carrying a category outside the censused set, e.g. the measured
79-point dual-write remainder) and negative (``surplus_counted`` --
the collection holds fewer points than the categories counted) get
separate diagnoses, because sending a reader hunting for an unenumerated
category is the wrong answer to a mid-census deletion.

The JSON artifact's value tables are never capped: ``--top-n`` cuts the
MARKDOWN view only. A capped JSON table would hide in-use ``kind`` values in
its tail while ``coverage.complete`` still read ``true`` -- the same silent
under-report in a different coordinate.

READ-ONLY: this script never writes, updates or deletes a memory. It issues
only Qdrant ``count`` and ``scroll`` calls, and writes two local report files.

Usage
-----
  # Default: census dark_factory + reify, write both artifacts under plans/.
  python scripts/census_memory_metadata.py

  # A single project, custom artifact paths.
  python scripts/census_memory_metadata.py --project-id dark_factory \\
      --json-out /tmp/census.json --md-out /tmp/census.md

  # Larger pages / a longer value tail in the MARKDOWN view (the JSON
  # artifact is complete either way).
  python scripts/census_memory_metadata.py --page-size 2000 --top-n 100
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import re
import sys
from collections import Counter
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from fused_memory.models.enums import GRAPHITI_PRIMARY, MEM0_PRIMARY, MemoryCategory

logger = logging.getLogger('census_memory_metadata')

# Default artifact paths, relative to the repo root (this file lives at
# <repo>/fused-memory/scripts/). Committed: leaf beta's grandfather list
# cites them, and a report that exists only on an operator's disk cannot be
# cited.
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_JSON_OUT = str(_REPO_ROOT / 'plans' / 'memory-metadata-census-report.json')
DEFAULT_MD_OUT = str(_REPO_ROOT / 'plans' / 'memory-metadata-census-report.md')

DEFAULT_PROJECT_IDS = ['dark_factory', 'reify']

# Page-budget backstop: 200 pages x the 1000-point default page size covers
# 200k points, an order of magnitude above the largest measured collection
# (fused_reify, 29,951). Hitting it means either the corpus grew hugely or
# the paging loop is not advancing -- either way the scan raises rather than
# returning a silently short stream.
DEFAULT_MAX_PAGES = 200

# All six categories, Mem0-primary first, derived from the shared enum --
# never a restated literal list (INV-5 no-lockstep-duplication). The three
# Graphiti-primary categories are included because dual-write records land
# them in the Mem0 collection too (80 points measured live across both
# collections -- 79 in fused_reify, 1 in fused_dark_factory); omitting them
# would leave a silently unenumerated slice of the corpus.
CENSUS_CATEGORIES: tuple[MemoryCategory, ...] = (
    *sorted(MEM0_PRIMARY),
    *sorted(GRAPHITI_PRIMARY),
)

_FULL_UUID_RE = re.compile(
    r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$',
    re.IGNORECASE,
)
# ``uuid.UUID.hex`` / ``str(u).replace('-', '')`` -- an unhyphenated but
# entirely VALID UUID rendering, kept distinct from genuine garbage.
_HEX32_RE = re.compile(r'^[0-9a-f]{32}$', re.IGNORECASE)
_SHORT_HEX_RE = re.compile(r'^[0-9a-f]+$', re.IGNORECASE)


class CensusScanIncomplete(RuntimeError):
    """A scroll could not enumerate its full result set.

    Raised rather than returning a short stream: a truncated census is
    indistinguishable from a census of a smaller corpus, and its consumer
    (leaf beta's grandfather list) cannot tell the difference (INV-2
    no-silent-fail-soft).
    """


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

    Returns:
        ``'full_uuid'`` for the canonical 36-char hyphenated rendering;
        ``'hex32_uuid'`` for the unhyphenated 32-hex-char rendering
        (``uuid.UUID.hex``) -- non-canonical but a VALID pointer, so beta can
        normalise it rather than reject it; ``'short_hex'`` for a bare hex
        string shorter than 32 characters (the malformed member shape PRD V1
        says beta must reject); ``'other'`` for a non-string or any other
        string.

    ``hex32_uuid`` is separated from ``other`` deliberately: a consumer told
    to reject ``'other'`` must not be forced to guess whether those members
    are malformed pointers or a legitimate alternate rendering.
    """
    if not isinstance(value, str):
        return 'other'
    if _FULL_UUID_RE.match(value):
        return 'full_uuid'
    if _HEX32_RE.match(value):
        return 'hex32_uuid'
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


def _table(counter: Counter[Any]) -> dict[str, Any]:
    """Render a Counter as a COMPLETE, deterministically ordered table.

    Sorted by count descending, then by value ascending -- so a re-run over
    an unchanged corpus produces a byte-identical artifact.

    Never capped. ``--top-n`` bounds the MARKDOWN view only (:func:`_render_table`):
    the JSON artifact is the machine-readable input to leaf beta's ``kind``
    registry and grandfather list, and a capped table there would silently
    orphan every in-use value in the tail -- the exact failure this census
    exists to prevent -- while ``coverage.complete`` still read ``true``.
    Truncation is therefore structurally impossible in the JSON rather than
    merely disclosed, so ``distinct_total`` always equals ``len(entries)``.
    """
    ordered = sorted(counter.items(), key=lambda kv: (-kv[1], _value_sort_key(kv[0])))
    return {
        'entries': [{'value': v, 'count': c} for v, c in ordered],
        'distinct_total': len(ordered),
    }


def _census_to_dict(census: CategoryCensus) -> dict[str, Any]:
    """Render one CategoryCensus (cell or rollup) as a JSON-serialisable dict."""
    return {
        'records': census.records,
        'keys': _table(census.key_counts),
        'kind': _table(census.kind_counts),
        'kind_missing': census.kind_missing,
        'source': _table(census.source_counts),
        'source_without_kind': _table(census.source_without_kind),
        'topic_present': census.topic_present,
        'topic': _table(census.topic_values),
        'parent_id_present': census.parent_id_present,
        'canonical_true': census.canonical_true,
        'canonical_false': census.canonical_false,
        'canonical_non_bool': census.canonical_non_bool,
        'supersedes_shapes': _table(census.supersedes_shapes),
        'supersedes_member_shapes': _table(census.supersedes_member_shapes),
        'supersedes_list_lengths': _table(census.supersedes_list_lengths),
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
            'categories': {category: {'expected', 'recount', 'scrolled'}}}}``
            as returned by :func:`census_project`.
        top_n: Cap on rendered rows per table in the MARKDOWN twin, recorded
            in ``params`` for :func:`render_markdown`. It does NOT cap the
            JSON tables -- those are always the complete population.
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
            rendered[category] = _census_to_dict(census)
            project_total.merge(census)
        grand_total.merge(project_total)
        projects[project_id] = {
            'total': _census_to_dict(project_total),
            'categories': rendered,
        }

    return {
        # v2: value tables are the complete population (v1 capped them at
        # top_n, which could truncate the JSON while coverage read complete);
        # top_n is now the markdown render cap only. Coverage cells gained
        # 'recount'/'churn'.
        # v3: churn tolerance extended from the cell to the COLLECTION level.
        # The per-project residue is now the gap between the counted interval
        # (sum expected .. sum recount) and the collection-points interval
        # (the bracketing count(scope) pair), not a pre-scroll sum compared
        # against a post-scroll total. New: per-project
        # 'collection_points_before'/'collection_points_interval'/
        # 'counted_recount'/'counted_interval'; 'uncovered_points' is the
        # SURVIVING residue; new delta kind 'surplus_counted' and churn kinds
        # 'collection_total_moved'/'category_count_moved'.
        'schema_version': 3,
        'params': {
            'projects': sorted(cells),
            'categories': category_order,
            'page_size': page_size,
            # Markdown render cap only — the tables below are complete.
            'top_n': top_n,
        },
        'projects': projects,
        'grand_total': _census_to_dict(grand_total),
        'coverage': _build_coverage(coverage),
    }


def _ordered_categories(project_cells: dict[str, CategoryCensus]) -> list[str]:
    """CENSUS_CATEGORIES order first, then any unexpected category, sorted."""
    known = [c.value for c in CENSUS_CATEGORIES if c.value in project_cells]
    extra = sorted(k for k in project_cells if k not in known)
    return known + extra


def _build_coverage(coverage: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Reconcile expected vs scrolled per cell and per-category sums vs the
    collection point total, naming every shortfall.

    The corpus is LIVE -- orchestrators write to it while the census runs --
    so every count and the page-scroll are unsynchronised reads. Churn is
    therefore tolerated at BOTH levels, and recorded either way (``churn``
    per cell, ``coverage.churn`` overall) so a reader can see the corpus
    moved rather than infer it from a silent discrepancy:

    * Per CELL: complete when the scroll agrees with EITHER the pre-scroll
      count (``expected``) or the post-scroll re-count (``recount``). A
      memory landing between the two reads is drift, not a truncated scan.

    * Per COLLECTION: the counted total is the interval
      ``[min, max]`` over ``(sum expected, sum recount)`` and the collection
      total is the interval over its bracketing ``count(scope)`` pair
      (``collection_points_before`` / ``collection_points``). The project is
      covered when the two intervals INTERSECT -- the exact statement of
      "these unsynchronised reads are consistent with one another". The
      reported ``uncovered_points`` is the signed GAP between the intervals,
      i.e. the MINIMUM residue that churn cannot explain, not the raw
      ``collection_points - sum(expected)`` difference (which compares a
      pre-scroll sum against a post-scroll total and so charges ordinary
      drift as missing data).

    A record predating the bracket -- no ``collection_points_before`` --
    collapses the points interval to the single post-scroll value, so an
    older caller or fixture still reconciles exactly as before. On a stable
    corpus both intervals collapse to points, intersection reduces to
    equality and the gap to the raw difference, which is what keeps a real
    residue (the measured 79-point dual-write remainder) named.
    """
    deltas: list[dict[str, Any]] = []
    churn: list[dict[str, Any]] = []
    projects: dict[str, Any] = {}
    overall_complete = True

    for project_id in sorted(coverage):
        record = coverage[project_id]
        per_category: dict[str, Any] = {}
        sum_expected = 0
        sum_recount = 0
        project_complete = True

        for category, counts in record.get('categories', {}).items():
            expected = int(counts.get('expected', 0))
            scrolled = int(counts.get('scrolled', 0))
            raw_recount = counts.get('recount')
            recount = None if raw_recount is None else int(raw_recount)
            delta = scrolled - expected
            sum_expected += expected
            # A cell without a recount contributes its pre-scroll count to
            # both bounds: absent evidence of movement, assume none.
            sum_recount += expected if recount is None else recount
            cell_complete = delta == 0 or (recount is not None and scrolled == recount)
            cell_churn = recount is not None and recount != expected
            per_category[category] = {
                'expected': expected,
                'recount': recount,
                'scrolled': scrolled,
                'delta': delta,
                'complete': cell_complete,
                'churn': cell_churn,
            }
            if cell_churn:
                churn.append({
                    'kind': 'category_count_moved',
                    'project_id': project_id,
                    'category': category,
                    'expected': expected,
                    'recount': recount,
                    'scrolled': scrolled,
                })
            if not cell_complete:
                project_complete = False
                # A SURPLUS is a different diagnosis than a shortfall: nothing
                # is missing, the corpus grew or a page double-yielded. Naming
                # them alike sends a reader hunting for data that is not gone.
                deltas.append({
                    'kind': 'category_shortfall' if delta < 0 else 'category_surplus',
                    'project_id': project_id,
                    'category': category,
                    'expected': expected,
                    'recount': recount,
                    'scrolled': scrolled,
                    'delta': delta,
                })

        collection_points = int(record.get('collection_points', 0))
        raw_before = record.get('collection_points_before')
        # No bracket (older caller / fixture): the interval collapses to the
        # single post-scroll read, reconciling exactly as it did before.
        points_before = collection_points if raw_before is None else int(raw_before)

        counted_lo, counted_hi = min(sum_expected, sum_recount), max(sum_expected, sum_recount)
        points_lo, points_hi = min(points_before, collection_points), max(
            points_before, collection_points)

        # Signed gap between the two intervals; 0 when they intersect.
        if points_lo > counted_hi:
            uncovered = points_lo - counted_hi        # points the census never saw
        elif points_hi < counted_lo:
            uncovered = points_hi - counted_lo        # negative: counted more than exist
        else:
            uncovered = 0

        if uncovered > 0:
            project_complete = False
            deltas.append({
                'kind': 'uncovered_points',
                'project_id': project_id,
                'collection': record.get('collection'),
                'collection_points': collection_points,
                'collection_points_interval': [points_lo, points_hi],
                'counted': sum_expected,
                'counted_interval': [counted_lo, counted_hi],
                'delta': uncovered,
            })
        elif uncovered < 0:
            # The collection holds FEWER points than the categories counted:
            # deletions landed mid-census or a category double-counted.
            # Nothing is unenumerated, so this is NOT an uncovered residue.
            project_complete = False
            deltas.append({
                'kind': 'surplus_counted',
                'project_id': project_id,
                'collection': record.get('collection'),
                'collection_points': collection_points,
                'collection_points_interval': [points_lo, points_hi],
                'counted': sum_expected,
                'counted_interval': [counted_lo, counted_hi],
                'delta': uncovered,
            })
        elif (counted_lo, counted_hi) != (points_lo, points_hi):
            # Absorbed by the churn window: the corpus moved under the scan.
            # Recorded, but not a delta -- it drives neither the exit code
            # nor the COVERAGE INCOMPLETE banner.
            churn.append({
                'kind': 'collection_total_moved',
                'project_id': project_id,
                'collection': record.get('collection'),
                'counted_interval': [counted_lo, counted_hi],
                'collection_points_interval': [points_lo, points_hi],
            })

        projects[project_id] = {
            'collection': record.get('collection'),
            'collection_points': collection_points,
            'collection_points_before': points_before,
            'collection_points_interval': [points_lo, points_hi],
            'counted': sum_expected,
            'counted_recount': sum_recount,
            'counted_interval': [counted_lo, counted_hi],
            # The SURVIVING residue -- the number that drove `complete`.
            'uncovered_points': uncovered,
            'complete': project_complete,
            'categories': per_category,
        }
        overall_complete = overall_complete and project_complete

    return {
        'complete': overall_complete,
        'deltas': deltas,
        'churn': churn,
        'projects': projects,
    }


# ---------------------------------------------------------------------------
# Thin I/O band (mock-tested — no live services required)
# ---------------------------------------------------------------------------

async def scroll_all_payloads(
    client: Any,
    collection: str,
    filters: dict[str, Any],
    page_size: int = 1000,
    max_pages: int = DEFAULT_MAX_PAGES,
    read_timeout: float | None = None,
) -> AsyncIterator[dict[str, Any]]:
    """Yield every payload matching *filters*, paging on Qdrant's ``next_offset``.

    This is the paging loop ``Mem0Backend.scroll_by_metadata`` structurally
    cannot perform -- it takes only ``limit`` and discards ``next_offset``
    (mem0_client.py:395), so repeated calls re-read the same head of the
    collection. ``consolidate_namespace_families.py:600-610`` documents that
    same caveat as a known gap; this closes it for the census.

    Payloads are yielded one at a time (never accumulated) so a caller
    folding them into counters holds one page in memory regardless of corpus
    size. The payload filter is built exactly as ``count_by_metadata`` builds
    its own (mem0_client.py:386-394), so the cross-check count and this scroll
    select the same points.

    Each page request is bounded by *read_timeout* (the backend's
    ``_read_timeout``, as every other Qdrant read on this path is --
    mem0_client.py:287, 331, 395), so a wedged connection fails loudly instead
    of hanging a ~30-page scan forever with no artifact and no diagnostic.
    ``None`` disables the bound.

    Raises:
        CensusScanIncomplete: If *max_pages* is consumed while ``next_offset``
            is still live -- the stream is truncated, so it raises instead of
            ending short.
        TimeoutError: If a single page request exceeds *read_timeout*. Not
            swallowed: a census that skipped a page would under-report
            silently (INV-2 no-silent-fail-soft).
    """
    from qdrant_client.http import models as qmodels  # noqa: PLC0415

    must: list[Any] = [
        qmodels.FieldCondition(key=k, match=qmodels.MatchValue(value=v))
        for k, v in filters.items()
    ]
    scroll_filter = qmodels.Filter(must=must)

    offset: Any = None
    pages = 0
    while True:
        page = client.scroll(
            collection_name=collection,
            scroll_filter=scroll_filter,
            with_payload=True,
            limit=page_size,
            offset=offset,
        )
        if read_timeout is not None:
            page = asyncio.wait_for(page, timeout=read_timeout)
        points, next_offset = await page
        pages += 1
        for point in points:
            yield dict(point.payload or {})

        if next_offset is None:
            return
        if pages >= max_pages:
            raise CensusScanIncomplete(
                f'scroll of collection={collection!r} filters={filters!r} exhausted its '
                f'page budget after {pages} page(s) of {page_size} with next_offset still '
                f'live — the scan is truncated. Raise --max-pages or --page-size.',
            )
        offset = next_offset


async def census_project(
    backend: Any,
    project_id: str,
    categories: list[str] | None = None,
    page_size: int = 1000,
    max_pages: int = DEFAULT_MAX_PAGES,
) -> tuple[dict[str, CategoryCensus], dict[str, Any]]:
    """Census one project's Mem0 collection, category by category.

    For each category: COUNT first (``count_by_metadata``), page-scroll and
    fold every payload into a :class:`CategoryCensus`, then COUNT AGAIN.
    Counting before scrolling is what makes an under-enumerated scroll
    detectable at all (INV-3 corroborate-before-acting) -- the two figures
    come from independent Qdrant calls and must agree. Counting again after
    it brackets the scan against a LIVE corpus: orchestrators write while the
    census runs, so a scroll that matches the post-scan count rather than the
    pre-scan one saw corpus churn, not truncation, and ``_build_coverage``
    must be able to tell those apart instead of calling a healthy census
    broken.

    Returns:
        ``(cells, coverage)`` where *cells* maps category value ->
        CategoryCensus, and *coverage* is the per-project record consumed by
        :func:`build_report`: collection name, collection point total, and
        per-category expected/recount/scrolled/delta/complete.
    """
    from fused_memory.models.scope import Scope  # noqa: PLC0415

    category_values = list(categories) if categories is not None else [
        c.value for c in CENSUS_CATEGORIES
    ]
    scope = Scope(project_id=project_id)
    collection = scope.mem0_collection_name(backend.config.mem0.collection_prefix)
    client = await backend._get_async_qdrant()
    # Same per-read bound every other Qdrant call on this path uses, so a
    # wedged socket is never mistaken for a slow one. Guarded by type: a
    # backend without a numeric bound scrolls unbounded rather than blowing
    # up inside asyncio.wait_for.
    raw_timeout = getattr(backend, '_read_timeout', None)
    read_timeout = (
        float(raw_timeout)
        if isinstance(raw_timeout, (int, float)) and not isinstance(raw_timeout, bool)
        else None
    )

    cells: dict[str, CategoryCensus] = {}
    per_category: dict[str, Any] = {}

    for category in category_values:
        filters = {'category': category}
        expected = await backend.count_by_metadata(scope, filters)

        census = CategoryCensus()
        async for payload in scroll_all_payloads(
            client, collection, filters, page_size=page_size, max_pages=max_pages,
            read_timeout=read_timeout,
        ):
            census.add(payload)

        # Re-count AFTER the scroll: the corpus is live, so a delta against
        # the pre-scroll count alone cannot distinguish a truncated scan from
        # a write that landed mid-census.
        recount = await backend.count_by_metadata(scope, filters)

        scrolled = census.records
        delta = scrolled - expected
        cells[category] = census
        per_category[category] = {
            'expected': expected,
            'recount': recount,
            'scrolled': scrolled,
            'delta': delta,
            'complete': delta == 0 or scrolled == recount,
        }
        logger.info(
            'censused project=%s category=%s expected=%d scrolled=%d recount=%d',
            project_id, category, expected, scrolled, recount,
        )
        if delta != 0 and scrolled == recount:
            logger.info(
                'CORPUS CHURN project=%s category=%s: count moved %d → %d while scrolling; '
                'the scroll agrees with the re-count, so the scan is complete.',
                project_id, category, expected, recount,
            )
        elif delta != 0:
            logger.warning(
                'UNDER-ENUMERATED project=%s category=%s: scrolled %d, count said %d before '
                'and %d after (delta %+d) — this census is a LOWER BOUND for that cell.',
                project_id, category, scrolled, expected, recount, delta,
            )

    collection_points = await backend.count(scope)
    counted = sum(c['expected'] for c in per_category.values())
    logger.info(
        'project=%s collection=%s points=%d counted=%d uncovered=%d',
        project_id, collection, collection_points, counted, collection_points - counted,
    )

    return cells, {
        'collection': collection,
        'collection_points': collection_points,
        'categories': per_category,
    }


def render_markdown(report: dict[str, Any]) -> str:
    """Render the markdown twin of the JSON census report.

    Driven ENTIRELY off *report* -- there is no second traversal of the raw
    payloads, so the two artifacts cannot disagree. Deterministic: the same
    report dict renders the identical string every time.

    This is the READABLE twin: long value tables are cut to
    ``report.params.top_n`` rows so a human-facing document stays legible.
    Every cut says so, and the JSON artifact it was rendered from carries the
    full population -- the markdown is never the only copy of a value.
    """
    top_n = report.get('params', {}).get('top_n')
    lines: list[str] = [
        '# Mem0 metadata corpus census',
        '',
        'Measured population of metadata keys, `kind` values and `supersedes` shapes '
        'across the live Mem0 collections — the empirical input to '
        '`docs/prds/memory-metadata-vocabulary.md` (leaf alpha of §9).',
        '',
        'Generated by `fused-memory/scripts/census_memory_metadata.py` (read-only). '
        'Regenerate with `uv run python scripts/census_memory_metadata.py`.',
        '',
    ]

    params = report.get('params', {})
    lines += [
        f'- schema_version: `{report.get("schema_version")}`',
        f'- projects: {_inline_list(params.get("projects", []))}',
        f'- categories: {_inline_list(params.get("categories", []))}',
        f'- page_size: `{params.get("page_size")}` · top_n: `{params.get("top_n")}` '
        '(markdown row cap only — the JSON artifact carries every value)',
        '',
    ]

    lines += _render_coverage(report.get('coverage', {}))

    lines += ['## Record counts', '']
    lines += _render_record_counts(report)

    lines += ['## Grand total', '']
    lines += _render_census_sections(report.get('grand_total', {}), top_n)

    for project_id in sorted(report.get('projects', {})):
        project = report['projects'][project_id]
        lines += [f'## Project `{project_id}`', '', '### All categories', '']
        lines += _render_census_sections(project.get('total', {}), top_n)
        for category, cell in project.get('categories', {}).items():
            lines += [f'### `{project_id}` / `{category}`', '']
            lines += _render_census_sections(cell, top_n)

    return '\n'.join(lines).rstrip() + '\n'


def _inline_list(values: list[Any]) -> str:
    return ', '.join(f'`{v}`' for v in values) if values else '_(none)_'


def _render_coverage(coverage: dict[str, Any]) -> list[str]:
    """Coverage section. An incomplete census says so LOUDLY and names every
    delta (INV-2) -- a silently short census is indistinguishable from a
    census of a smaller corpus."""
    lines = ['## Coverage', '']
    complete = coverage.get('complete', False)
    if complete:
        lines += ['Coverage is **complete**: every category scrolled as many points as '
                  '`count_by_metadata` reported either side of the scan, and the '
                  'per-category counts account for every point in each collection.', '']
    else:
        lines += [
            '> **WARNING — COVERAGE INCOMPLETE.** This census under-enumerates the corpus; '
            'the counts below are a LOWER BOUND, not the population. Do not seed a '
            'registry or grandfather list from it until the deltas below are resolved.',
            '',
        ]

    lines += [
        '| project | collection | collection points | counted | uncovered | complete |',
        '| --- | --- | ---: | ---: | ---: | --- |',
    ]
    for project_id in sorted(coverage.get('projects', {})):
        p = coverage['projects'][project_id]
        lines.append(
            f'| `{project_id}` | `{p.get("collection")}` | {p.get("collection_points", 0):,} | '
            f'{p.get("counted", 0):,} | {p.get("uncovered_points", 0):,} | '
            f'{"yes" if p.get("complete") else "**NO**"} |',
        )
    lines.append('')

    lines += [
        '| project | category | expected | recount | scrolled | delta |',
        '| --- | --- | ---: | ---: | ---: | ---: |',
    ]
    for project_id in sorted(coverage.get('projects', {})):
        for category, c in coverage['projects'][project_id].get('categories', {}).items():
            recount = c.get('recount')
            recount_cell = '—' if recount is None else f'{recount:,}'
            lines.append(
                f'| `{project_id}` | `{category}` | {c.get("expected", 0):,} | '
                f'{recount_cell} | {c.get("scrolled", 0):,} | {c.get("delta", 0):+,} |',
            )
    lines += [
        '',
        'The corpus is LIVE: `expected` is counted before the scroll and `recount` after '
        'it, so a small non-zero `delta` on a cell whose `scrolled` matches `recount` is '
        'corpus churn (a write landed mid-census), NOT a truncated scan — such a cell is '
        'still complete. Only a `scrolled` that matches neither count is a shortfall.',
        '',
    ]

    churn = coverage.get('churn', [])
    if churn:
        lines += ['### Corpus churn during the scan', '']
        lines += [
            '_Not a defect: the count moved while the census ran. Recorded so a reader '
            'can tell drift from truncation._',
            '',
        ]
        for entry in churn:
            lines.append(
                f'- `{entry.get("project_id")}` / `{entry.get("category")}`: count moved '
                f'{entry.get("expected", 0):,} → {entry.get("recount", 0):,} while '
                f'scrolling; scrolled {entry.get("scrolled", 0):,}.',
            )
        lines.append('')

    deltas = coverage.get('deltas', [])
    if deltas:
        lines += ['### Named shortfalls', '']
        for d in deltas:
            kind = d.get('kind')
            if kind == 'category_shortfall':
                lines.append(
                    f'- `category_shortfall` — `{d.get("project_id")}` / '
                    f'`{d.get("category")}`: scrolled {d.get("scrolled", 0):,} of '
                    f'{d.get("expected", 0):,} expected (delta {d.get("delta", 0):+,}) — '
                    f'points are MISSING from this cell.',
                )
            elif kind == 'category_surplus':
                # A surplus is a different diagnosis: nothing is missing.
                lines.append(
                    f'- `category_surplus` — `{d.get("project_id")}` / '
                    f'`{d.get("category")}`: scrolled {d.get("scrolled", 0):,}, '
                    f'{abs(d.get("delta", 0)):,} MORE than either count expected '
                    f'({d.get("expected", 0):,} before, {d.get("recount", 0)} after) — '
                    f'concurrent writes or duplicate pages, not missing data.',
                )
            else:
                lines.append(
                    f'- `uncovered_points` — `{d.get("project_id")}` / '
                    f'`{d.get("collection")}`: {d.get("delta", 0):,} of '
                    f'{d.get("collection_points", 0):,} points carry a category outside '
                    f'the censused set (counted {d.get("counted", 0):,}).',
                )
        lines.append('')
    return lines


def _render_record_counts(report: dict[str, Any]) -> list[str]:
    lines = ['| project | category | records |', '| --- | --- | ---: |']
    for project_id in sorted(report.get('projects', {})):
        project = report['projects'][project_id]
        for category, cell in project.get('categories', {}).items():
            lines.append(f'| `{project_id}` | `{category}` | {cell.get("records", 0):,} |')
        lines.append(
            f'| `{project_id}` | **(all)** | **{project.get("total", {}).get("records", 0):,}** |',
        )
    lines.append(
        f'| **(all)** | **(all)** | **{report.get("grand_total", {}).get("records", 0):,}** |',
    )
    lines.append('')
    return lines


def _render_census_sections(census: dict[str, Any], top_n: int | None = None) -> list[str]:
    """Render one cell/rollup: every table plus the scalar occurrence axes."""
    if not census:
        return ['_(no data)_', '']

    lines = [f'Records: **{census.get("records", 0):,}**', '']

    lines += _render_table(
        'Top-level metadata key population', census.get('keys'), 'key', top_n,
    )
    lines += _render_table('`kind` values', census.get('kind'), 'kind', top_n)
    lines += [f'`kind` missing: **{census.get("kind_missing", 0):,}** record(s).', '']

    lines += _render_table(
        '`supersedes` shapes', census.get('supersedes_shapes'), 'shape', top_n,
    )
    lines += _render_table(
        '`supersedes` member shapes', census.get('supersedes_member_shapes'),
        'member shape', top_n,
    )
    lines += _render_table(
        '`supersedes` list lengths', census.get('supersedes_list_lengths'), 'length', top_n,
    )

    lines += ['#### Occurrence axes', '', '| axis | count |', '| --- | ---: |']
    lines += [
        f'| `topic` present | {census.get("topic_present", 0):,} |',
        f'| `parent_id` present | {census.get("parent_id_present", 0):,} |',
        f'| `canonical` true | {census.get("canonical_true", 0):,} |',
        f'| `canonical` false | {census.get("canonical_false", 0):,} |',
        f'| `canonical` non-bool | {census.get("canonical_non_bool", 0):,} |',
        '',
    ]
    lines += _render_table('`topic` values', census.get('topic'), 'topic', top_n)

    lines += _render_table('`source` values', census.get('source'), 'source', top_n)
    lines += _render_table(
        '`source` set but `kind` missing', census.get('source_without_kind'),
        'source', top_n,
    )
    return lines


def _render_table(
    title: str,
    table: dict[str, Any] | None,
    value_header: str,
    top_n: int | None = None,
) -> list[str]:
    """Render one value table, cut to *top_n* rows for readability.

    The cut is a MARKDOWN-only concern: the JSON table this is rendered from
    is complete, and any row omitted here says so and says where the rest is.
    ``None`` or a non-positive *top_n* renders the whole population.
    """
    resolved = table or {}
    lines = [f'#### {title}', '']
    entries = resolved.get('entries', [])
    if not entries:
        lines += ['_(none)_', '']
        return lines
    shown = entries if not top_n or top_n <= 0 else entries[:top_n]
    distinct_total = resolved.get('distinct_total', len(entries))
    lines += [f'| {value_header} | count |', '| --- | ---: |']
    lines += [f'| `{e["value"]}` | {e["count"]:,} |' for e in shown]
    lines.append('')
    if len(shown) < distinct_total:
        lines += [
            f'_Showing top {len(shown):,} of {distinct_total:,} distinct values — this '
            f'markdown view is **truncated**; the JSON artifact carries the full '
            f'population, and `--top-n` widens this view._',
            '',
        ]
    return lines


# ---------------------------------------------------------------------------
# CLI / main
# ---------------------------------------------------------------------------

def _build_backend(config: Any) -> Any:
    """Construct the Mem0 backend.

    Deliberately NOT ``MemoryService(config)`` + ``initialize()``:
    ``Mem0Backend.__init__`` needs only config and creates its Qdrant client
    lazily, so a read-only census needs neither FalkorDB/Graphiti nor an
    embedder / ``OPENAI_API_KEY``. Fewer live dependencies means the census
    still runs when the graph store or LLM credentials are unavailable.
    Seam kept separate so the CLI band is testable without a live Qdrant.
    """
    from fused_memory.backends.mem0_client import Mem0Backend  # noqa: PLC0415

    return Mem0Backend(config)


async def _run(args: argparse.Namespace) -> int:
    logging.basicConfig(
        level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s',
    )

    import os  # noqa: PLC0415

    from fused_memory.config.schema import FusedMemoryConfig  # noqa: PLC0415

    if args.config:
        os.environ['CONFIG_PATH'] = str(args.config)

    backend = _build_backend(FusedMemoryConfig())
    try:
        cells: dict[str, dict[str, CategoryCensus]] = {}
        coverage: dict[str, dict[str, Any]] = {}
        for project_id in args.project_id:
            logger.info('censusing project=%s', project_id)
            project_cells, project_coverage = await census_project(
                backend,
                project_id,
                page_size=args.page_size,
                max_pages=args.max_pages,
            )
            cells[project_id] = project_cells
            coverage[project_id] = project_coverage

        report = build_report(
            cells, coverage, top_n=args.top_n, page_size=args.page_size,
        )
        _write_artifacts(report, args.json_out, args.md_out)

        complete = report['coverage']['complete']
        logger.info(
            'census complete=%s records=%d json=%s md=%s',
            complete, report['grand_total']['records'], args.json_out, args.md_out,
        )
        if not complete:
            # The artifact still lands — the evidence of the shortfall is IN
            # it — but the exit code refuses to call an under-enumerated
            # census a successful one (INV-2).
            for delta in report['coverage']['deltas']:
                logger.error('COVERAGE SHORTFALL: %s', delta)
            return 1
        return 0
    except CensusScanIncomplete:
        logger.exception('ABORT: a scroll could not enumerate its full result set')
        return 1
    finally:
        await backend.close()


def _write_artifacts(report: dict[str, Any], json_out: str, md_out: str) -> None:
    json_path = Path(json_out)
    md_path = Path(md_out)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(report, indent=2, sort_keys=False) + '\n')
    md_path.write_text(render_markdown(report))


class _AppendReplacingDefault(argparse.Action):
    """``append`` that DISCARDS the default list on first use.

    Plain ``action='append'`` with a list default extends it, so a single
    ``--project-id reify`` would silently census dark_factory too. Replacing
    on first use lets the parser carry the real default (visible in --help,
    and returned by ``parse_args([])``) without that trap.
    """

    def __call__(self, parser, namespace, values, option_string=None):
        current = getattr(namespace, self.dest, None)
        if current is self.default or current is None:
            current = []
            setattr(namespace, self.dest, current)
        current.append(values)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--project-id', dest='project_id', action=_AppendReplacingDefault,
        default=list(DEFAULT_PROJECT_IDS),
        help=(
            'Project id to census; repeatable. '
            f'Default: {" ".join(DEFAULT_PROJECT_IDS)}'
        ),
    )
    parser.add_argument(
        '--page-size', dest='page_size', type=int, default=1000,
        help='Qdrant scroll page size (default: 1000)',
    )
    parser.add_argument(
        '--top-n', dest='top_n', type=int, default=50,
        help=(
            'Max rows per value/key table in the MARKDOWN report (default: 50). '
            'The JSON report always carries every value; 0 or less renders the '
            'full population in the markdown too.'
        ),
    )
    parser.add_argument(
        '--max-pages', dest='max_pages', type=int, default=DEFAULT_MAX_PAGES,
        help=f'Per-category scroll page budget (default: {DEFAULT_MAX_PAGES})',
    )
    parser.add_argument(
        '--json-out', dest='json_out', default=DEFAULT_JSON_OUT,
        help=f'JSON report path (default: {DEFAULT_JSON_OUT})',
    )
    parser.add_argument(
        '--md-out', dest='md_out', default=DEFAULT_MD_OUT,
        help=f'Markdown report path (default: {DEFAULT_MD_OUT})',
    )
    parser.add_argument(
        '--config', default=None,
        help='Path to fused-memory config file (sets CONFIG_PATH env var)',
    )
    return parser


def main() -> int:
    return asyncio.run(_run(_build_parser().parse_args()))


if __name__ == '__main__':
    sys.exit(main())
