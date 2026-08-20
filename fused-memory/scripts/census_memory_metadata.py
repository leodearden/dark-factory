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
    ``Mem0Backend.scroll_by_metadata`` returns one capped page and discards
    Qdrant's ``next_offset``, so this script originally drove the raw async
    Qdrant client and looped on ``next_offset`` itself. Task 3225 moved that
    loop into the backend as ``Mem0Backend.scroll_all_by_metadata``; the
    census now drives it and owns no paging code. Records are folded into
    counters as they arrive, so peak memory is one page.

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
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from fused_memory.backends.mem0_client import (
    DEFAULT_SCROLL_MAX_PAGES,
    ScrollPageBudgetExhausted,
)
from fused_memory.models.enums import GRAPHITI_PRIMARY, MEM0_PRIMARY, MemoryCategory

# THE shared slug predicate, imported and called -- never re-expressed as a
# second regex here (INV-5; tests/test_topic_slug_namespace.py pins the same
# identity by ``is``). topic_slug is a stdlib-only leaf, so this adds no
# heavy import to the census.
from fused_memory.topic_slug import is_valid_topic_slug

logger = logging.getLogger('census_memory_metadata')

# Default artifact paths, relative to the repo root (this file lives at
# <repo>/fused-memory/scripts/). Committed: leaf beta's grandfather list
# cites them, and a report that exists only on an operator's disk cannot be
# cited.
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_JSON_OUT = str(_REPO_ROOT / 'plans' / 'memory-metadata-census-report.json')
DEFAULT_MD_OUT = str(_REPO_ROOT / 'plans' / 'memory-metadata-census-report.md')


def _repo_relative(path: str) -> str:
    """Project *path* to a repo-relative string when it is inside the repo.

    ``_REPO_ROOT`` is derived from this file's location, so it differs
    between the main checkout and every ``.worktrees/<id>`` lane. Recording a
    resolved ABSOLUTE path in the committed artifact's ``params`` therefore
    (a) churns the artifact the first time a run happens in a different
    checkout, and (b) points a reader at a worktree that will be reclaimed.
    Worse, :func:`_regen_command` prints a flag whenever the recorded value
    departs from the default resolved AT READ TIME, so an artifact generated
    in one checkout emits a regen command naming a path that does not exist
    there -- task 3507's failure exactly ("a documented regen command that no
    longer reproduces the artifact is worse than none").

    Paths outside the repo are returned resolved-absolute: they are genuinely
    checkout-independent, and shortening them is not this function's job.

    RECORDING side. *path* is whatever the CLI was handed, so it is resolved
    with the process's CWD semantics -- the same way the run itself
    interpreted it. Reading such a value back is :func:`_resolved_repo_path`,
    which anchors at the repo root instead; the two are not interchangeable.
    """
    resolved = Path(path).resolve()
    try:
        return str(resolved.relative_to(_REPO_ROOT))
    except ValueError:
        return str(resolved)


def _resolved_repo_path(value: str) -> str:
    """Resolve a path RECORDED in ``params`` back to an absolute path.

    READING side of :func:`_repo_relative`. A recorded relative value is
    relative to the REPO ROOT, never to the reader's CWD -- anchoring it at
    the CWD would make an artifact compare unequal to its own default
    depending on which directory the reader happened to run from (measured:
    the census suite runs from ``fused-memory/``, the wrapper from the repo
    root). Absolute values are honoured as-is, which is what lets a
    pre-4006 artifact's absolute default still normalise.
    """
    candidate = Path(value)
    if not candidate.is_absolute():
        candidate = _REPO_ROOT / candidate
    return str(candidate.resolve())

DEFAULT_PROJECT_IDS = ['dark_factory', 'reify']

# CLI defaults for --page-size / --top-n, shared with the argparse setup
# below AND the regen-command emitter in render_markdown() so the two can
# never drift apart (task 3507: they used to be restated literals, which is
# exactly how the documented regen command stopped matching a --top-n 400
# artifact).
DEFAULT_PAGE_SIZE = 1000
DEFAULT_TOP_N = 50

# Page-budget backstop: 200 pages x the 1000-point default page size covers
# 200k points, an order of magnitude above the largest measured collection
# (fused_reify, 29,951). Hitting it means either the corpus grew hugely or
# the paging loop is not advancing -- either way the scan raises rather than
# returning a silently short stream.
#
# ALIASED, never restated: the budget now travels with the paging loop it
# bounds, which lives in Mem0Backend.scroll_collection_pages. 200 has one
# home (INV-5). The script-local spelling survives because --max-pages'
# help text and every caller here already bind this name.
DEFAULT_MAX_PAGES = DEFAULT_SCROLL_MAX_PAGES

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


#: A scroll could not enumerate its full result set.
#:
#: Raised rather than returning a short stream: a truncated census is
#: indistinguishable from a census of a smaller corpus, and its consumer
#: (leaf beta's grandfather list) cannot tell the difference (INV-9
#: no-silent-fail-soft).
#:
#: An ALIAS of the backend's exception, not a subclass. The paging loop that
#: raises it now lives in ``Mem0Backend.scroll_collection_pages``, so
#: ``except CensusScanIncomplete`` in :func:`_run` must name exactly what the
#: backend raises -- a subclass would not catch its parent and would silently
#: convert an abort-with-nonzero-exit into an uncaught traceback. The
#: script-local spelling survives because ``_run``'s handler and
#: ``TestCliExitCodes`` already bind this name.
CensusScanIncomplete = ScrollPageBudgetExhausted


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

    # The topic x canonical CROSS-TAB (task 4006).  ``topic_values`` and
    # ``canonical_true`` above are INDEPENDENT counters, so the coverage
    # partition -- topics with exactly one canonical / zero / more than one --
    # is not derivable from either.  Keyed by topic, this mirrors 3198's
    # write-time uniqueness probe ``count_memories_by_metadata(project_id,
    # {'topic': T, 'canonical': True})``
    # (fused-memory/src/fused_memory/services/memory_service.py::_check_canonical_uniqueness),
    # which is scope-wide and NOT category-filtered -- so the partition must
    # be read off the per-project ``merge()`` rollup, never off a single cell.
    canonical_true_by_topic: Counter[str] = field(default_factory=Counter)
    # A ``canonical: true`` carrying no topic cannot be keyed into the
    # cross-tab, and dropping it would break the identity
    # ``canonical_true == sum(by_topic) + without_topic``.  Reachable today:
    # the ``canonical_without_topic`` shape rule is warn-mode under the
    # shipped ``memory_metadata.enforce=false``, so such records DO land.
    canonical_true_without_topic: int = 0

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

        topic: str | None = None
        if 'topic' in payload and payload['topic'] is not None:
            topic = str(payload['topic'])
            self.topic_present += 1
            self.topic_values[topic] += 1
        if 'parent_id' in payload and payload['parent_id'] is not None:
            self.parent_id_present += 1

        if 'canonical' in payload:
            canonical = payload['canonical']
            if isinstance(canonical, bool):
                if canonical:
                    self.canonical_true += 1
                    # The cross-tab, folded from the topic already parsed
                    # above -- no second payload lookup.  A non-bool
                    # ``canonical`` deliberately reaches neither branch:
                    # a string 'true' is coercion drift, not a canonical,
                    # and scoring it as one would manufacture uniqueness
                    # violations out of the existing canonical_non_bool
                    # population (1 record measured live in reify).
                    if topic is None:
                        self.canonical_true_without_topic += 1
                    else:
                        self.canonical_true_by_topic[topic] += 1
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
        self.canonical_true_by_topic.update(other.canonical_true_by_topic)
        self.canonical_true_without_topic += other.canonical_true_without_topic
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
        # Through _table() like every other value axis: deterministic
        # count-desc/value-asc order, and never capped in the JSON.
        'canonical_true_by_topic': _table(census.canonical_true_by_topic),
        'canonical_true_without_topic': census.canonical_true_without_topic,
        'supersedes_shapes': _table(census.supersedes_shapes),
        'supersedes_member_shapes': _table(census.supersedes_member_shapes),
        'supersedes_list_lengths': _table(census.supersedes_list_lengths),
    }


def build_report(
    cells: dict[str, dict[str, CategoryCensus]],
    coverage: dict[str, dict[str, Any]],
    top_n: int = DEFAULT_TOP_N,
    page_size: int | None = None,
    canonical_uniqueness_enforced: bool | None = None,
    registry: Any = None,
    registry_error: str | None = None,
    history: dict[str, Any] | None = None,
    extra_params: dict[str, Any] | None = None,
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
        canonical_uniqueness_enforced: The live ``memory_metadata.enforce``
            state, INJECTED rather than read here so this stays a pure
            function. Three-valued (``None`` = could not be read); see
            :func:`read_canonical_uniqueness_enforced`, which is what ``_run``
            passes.
        history: the loaded coverage history, used to compute the
            ``coverage_trend`` block. ``None`` (no history supplied) is
            disclosed in that block's reason rather than rendered as a flat
            trend -- see :func:`build_coverage_diff`.

    Returns:
        A JSON-serialisable dict with per-cell counts, per-project and
        grand-total rollups, and a coverage block. Any enumeration shortfall
        -- a cell scrolling fewer points than ``count_by_metadata`` expected,
        or per-category counts summing short of the collection's point total
        -- sets ``coverage.complete=false`` and appears as a NAMED entry in
        ``coverage.deltas`` (INV-9 no-silent-fail-soft). The caller exits
        non-zero on that flag.
    """
    projects: dict[str, Any] = {}
    grand_total = CategoryCensus()
    category_order: list[str] = []
    project_totals: dict[str, CategoryCensus] = {}

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
        project_totals[project_id] = project_total
        projects[project_id] = {
            'total': _census_to_dict(project_total),
            # Off the ROLLUP, so the canonical partition is computed at the
            # project grain 3198's uniqueness invariant is stated at.
            'topic_coverage': _build_topic_coverage(
                project_total, canonical_uniqueness_enforced,
            ),
            'categories': rendered,
        }

    if registry is None:
        registry_coverage = None
        # A load failure the CALLER already diagnosed passes through verbatim;
        # a bare None with no error means the gauge was not requested. Both
        # are DISCLOSED rather than rendered as an all-zero (or perfect) score.
        registry_error = registry_error or 'no topic registry supplied'
    else:
        # SCORING is guarded as well as LOADING. `topic_registry_loader` is
        # lazily memoized and `load_registry_or_error` catches everything, so
        # an unloadable probe already degrades to `registry_error` -- but the
        # scoring below reaches `entry.project_id` / `entry.topic` by attribute
        # on objects owned by ANOTHER script's dataclass. A `RegistryEntry`
        # field rename raised straight out of build_report and took the whole
        # nightly census with it (no report, no markdown, no trend row) from a
        # block the design says is OPTIONAL. Same disclosure, same shape as
        # every other registry failure (INV-2: disclose the shortfall, never
        # escalate it into a total loss).
        try:
            registry_coverage, registry_error = _build_registry_coverage(
                project_totals, registry,
            )
        except Exception as exc:  # noqa: BLE001 - the gauge is never fatal
            registry_coverage = None
            registry_error = f'{type(exc).__name__}: {exc}'

    report = {
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
        # v4 (task 4006): raw counts became COVERAGE. Every census cell gained
        # the topic x canonical cross-tab ('canonical_true_by_topic',
        # 'canonical_true_without_topic'), and a new per-project +
        # grand-total 'topic_coverage' block carries the stamping RATE and
        # the three-way canonical partition (exactly one / zero / more than
        # one per topic), computed at PROJECT grain to match 3198's
        # scope-wide uniqueness probe. New top-level 'registry_coverage'
        # (the accountable 32-topic target) and 'coverage_trend' (this run
        # against the most recent prior run in the committed history).
        'schema_version': 4,
        'params': {
            'projects': sorted(cells),
            'categories': category_order,
            'page_size': page_size,
            # Markdown render cap only — the tables below are complete.
            'top_n': top_n,
            # Run-shaping CLI values recorded so _regen_command can
            # reconstruct them; merged rather than enumerated so the parser
            # and this block cannot drift apart (task 3507).
            **(extra_params or {}),
        },
        'projects': projects,
        'grand_total': _census_to_dict(grand_total),
        'topic_coverage': _regrade_uniqueness_at_project_grain(
            _build_topic_coverage(grand_total, canonical_uniqueness_enforced),
            projects,
        ),
        # Spans projects (each entry carries its own project_id), so it sits
        # at the top level rather than under any one project.
        'registry_coverage': registry_coverage,
        'registry_error': registry_error,
        # Placeholder so the key keeps its position in the artifact; filled
        # below, since the diff is computed FROM the finished report (one
        # distillation path -- see _coverage_run_row).
        'coverage_trend': None,
        'coverage': _build_coverage(coverage),
    }
    report['coverage_trend'] = build_coverage_diff(report, history)
    return report


#: What still stands between here and flipping ``memory_metadata.enforce``.
#: Emitted beside the violation count and the ``enforce`` flag so the report
#: is SELF-CONTAINED: a reader gets the number, the regime that produced it,
#: and what remains, without reconstructing the last part from the PRD
#: (Leo's ruling of 2026-08-12, Option B on esc-4006-3).
#:
#: CITATION-ONLY, deliberately. The ruling's scope note is explicit that this
#: task cites the preconditions and does not chase them: no live probe of
#: 3202/3626's task status is issued here. Single-homed so the JSON and the
#: markdown can never disagree about what blocks the flip.
ENFORCE_FLIP_PRECONDITIONS: tuple[dict[str, str], ...] = (
    {
        'id': 'legacy_topic_spelling_remains',
        'what': (
            'No live topic value may remain in a legacy (non-conforming) '
            'spelling -- such a topic is invisible to every registry-keyed '
            'and exact-match read.'
        ),
        # NOT "open": PRD §159's 2026-08-04 amendment records the sweep
        # applied and the residue normalized -- "that bucket is empty ...
        # This half of the precondition is DISCHARGED". Citing it as open
        # would be as wrong as omitting it. But a status recorded on
        # 2026-08-04 is not a live fact, which is why the emitted entry
        # carries THIS run's conformance count as its re-measurement: the
        # leaf-alpha census's own history (98 -> 103 non-conforming distinct
        # topics, which GREW while in warn mode) proves the bucket regrows.
        'status': 'discharged_2026_08_04',
        'source': 'docs/prds/memory-metadata-vocabulary.md §159 (2026-08-04 amendment)',
    },
    {
        'id': '3202',
        'what': (
            "Leaf iota -- the writer-instruction slug contract -- must land. "
            'PRD §159 names this "the real precondition" after measuring that '
            'normalizing records at rest moved the false-rejection rate only '
            '~20 -> ~19/week: enforce rejects WRITES and never re-validates '
            'the corpus, so "theta WAS NOT THE GATE; iota IS".'
        ),
        'status': 'open',
        'source': 'task 3202 (leaf iota) · docs/prds/memory-metadata-vocabulary.md §159',
    },
    {
        'id': '3626',
        'what': (
            'The milestone gate that re-measures and DECIDES the flip. Its '
            'recipe items 3 and 4 name this census as their instrument: the '
            'non-conforming distinct-topic count, and confirmation that every '
            'live canonical:true topic passes the slug regex in both projects.'
        ),
        'status': 'open',
        'source': 'task 3626 ([MILESTONE] memory_metadata.enforce flip gate)',
    },
)


#: The committed 32-entry registry: the ACCOUNTABLE target set. Every one of
#: its topics should carry exactly one live ``canonical: true`` in its own
#: project. Bounded and named, unlike a corpus-wide stamping percentage over
#: ~49.6k records, most of which is cycle_summary-class with no meaningful
#: topic -- and it is precisely the set E1's retrieval-health run scores
#: 32/32 failing.
DEFAULT_REGISTRY_PATH = str(
    _REPO_ROOT / 'fused-memory' / 'tests' / 'fixtures'
    / 'memory_eval_topic_registry.json'
)

_PROBE_SCRIPT_PATH = _REPO_ROOT / 'fused-memory' / 'scripts' / 'memory_eval_retrieval_probe.py'


def _load_probe_module() -> Any:
    """Load ``memory_eval_retrieval_probe`` by path.

    ``scripts/`` is not a package, so a plain import cannot reach it. This is
    the same importlib idiom
    ``fused-memory/scripts/retro_stamp_topics.py::_load_probe_module`` already
    established for exactly this cross-script reuse, including the
    ``sys.modules``-first lookup: that slot may already hold a module another
    by-path loader executed, and re-executing would hand back a DIFFERENT
    class object for ``RegistryEntry``, silently breaking identity across the
    seam.
    """
    import importlib.util  # noqa: PLC0415

    mod_name = 'memory_eval_retrieval_probe'
    cached = sys.modules.get(mod_name)
    if cached is not None:
        return cached
    spec = importlib.util.spec_from_file_location(mod_name, _PROBE_SCRIPT_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f'Cannot load {_PROBE_SCRIPT_PATH}')
    module = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(mod_name, None)
        raise
    return module


#: Memoized holder for the probe's loader. ``None`` means "not loaded yet",
#: never "unavailable" -- an unavailable probe raises out of the accessor and
#: is disclosed by :func:`load_registry_or_error`.
_TOPIC_REGISTRY_LOADER: Any = None


def topic_registry_loader() -> Any:
    """Return the probe's ``load_topic_registry``, loading it on FIRST USE.

    REUSED, never re-parsed: the registry's schema-version check, its
    zero-entry rejection and its ``RegistryError`` type all stay single-homed
    in the probe, exactly as
    ``fused-memory/scripts/retro_stamp_topics.py::load_topic_registry``
    establishes -- and the reuse stays pinnable by ``is``, through this
    accessor.

    LAZY and memoized, not an import-time module alias. As an alias the load
    ran at ``import census_memory_metadata`` time, which made the most likely
    registry failure -- the probe script renamed, moved, or carrying an import
    error -- the one mode that could NOT be disclosed: it killed the whole
    nightly census (no coverage numbers, no report, no trend row) and this
    module's entire test suite with it, bypassing the ``registry_coverage:
    null`` + ``registry_error`` path built for exactly that failure. The
    registry gauge is ONE optional block of the report; it must not be able to
    take the measurement down with it (INV-2 no-silent-fail-soft cuts both
    ways -- disclose the shortfall, do not escalate it into a total loss).
    """
    global _TOPIC_REGISTRY_LOADER  # noqa: PLW0603 - one-shot memo
    if _TOPIC_REGISTRY_LOADER is None:
        _TOPIC_REGISTRY_LOADER = _load_probe_module().load_topic_registry
    return _TOPIC_REGISTRY_LOADER


def load_registry_or_error(path: str) -> tuple[Any | None, str | None]:
    """Load the topic registry, returning ``(registry, error)``.

    Exactly one of the pair is ever non-``None``. Degrades LOUDLY: the caller
    emits ``registry_coverage: null`` plus the error string rather than a
    silent all-zero gauge, which would read as "measured, and everything is
    fine" -- indistinguishable from a genuinely clean result, when the whole
    value of this gauge is that its zero is meaningful (INV-2).
    """
    try:
        # The loader itself is resolved INSIDE the try: an unloadable probe
        # module is a registry failure like any other, and is disclosed the
        # same way rather than aborting the census.
        registry = topic_registry_loader()(path)
    except Exception as exc:  # noqa: BLE001 - every load failure is disclosed
        return None, f'{type(exc).__name__}: {exc}'
    return registry, None


def _build_registry_coverage(
    project_totals: dict[str, CategoryCensus],
    registry: Any,
) -> tuple[dict[str, Any] | None, str | None]:
    """Score the committed registry against measured corpus state.

    NOT a duplicate of E1's ``METRIC_TOPIC_CANONICAL_PRESENT`` (INV-5
    checked). That tripwire is computed from ``_tripwire_items(observations,
    TRIPWIRE_K)`` -- from SEARCH observations -- so it conflates "the
    canonical does not exist" with "it exists but ranks below K". This
    task's whole thesis is that 32/32 fails because the canonicals are
    ABSENT, and only a deterministic metadata count can establish that. The
    two instruments answer different questions; this one disambiguates the
    probe's.

    Each entry is matched within its OWN ``project_id``. 3198's invariant is
    per-(project, topic), so a canonical stamped in reify must not discharge
    a dark_factory entry -- a global match would report the target met while
    the dark_factory corpus stayed bare.
    """
    entries = list(getattr(registry, 'entries', []) or [])
    if not entries:
        return None, 'topic registry loaded but holds zero entries'

    rows: list[dict[str, Any]] = []
    exactly_one = zero = multiple = 0
    for entry in sorted(entries, key=lambda e: (str(e.project_id), str(e.topic))):
        topic = str(entry.topic)
        project_id = str(entry.project_id)
        # A registry entry naming an uncensused project measures ZERO, not
        # "unknown": the topic demonstrably has no canonical in this run's
        # evidence, and omitting the row would shrink the denominator.
        census = project_totals.get(project_id)
        records = census.topic_values.get(topic, 0) if census else 0
        canonical_count = census.canonical_true_by_topic.get(topic, 0) if census else 0
        if canonical_count == 0:
            zero += 1
        elif canonical_count == 1:
            exactly_one += 1
        else:
            multiple += 1
        rows.append({
            'project_id': project_id,
            'topic': topic,
            'records': records,
            'canonical_count': canonical_count,
        })

    return {
        'registry_topics_total': len(rows),
        'registry_topics_with_exactly_one_canonical': exactly_one,
        'registry_topics_with_zero_canonical': zero,
        'registry_topics_with_multiple_canonical': multiple,
        # The ACTIONABLE worklist -- exactly the set 3201's idempotent retro
        # sweep would stamp, which is why the nightly wrapper runs it (dry-run)
        # right after this census.
        'zero_canonical_topics': [r for r in rows if r['canonical_count'] == 0],
        'topics': rows,
    }, None


def read_canonical_uniqueness_enforced(config: Any | None = None) -> bool | None:
    """Three-valued read of the live ``memory_metadata.enforce`` setting.

    ``True``/``False`` are the measured flag; ``None`` means the config could
    not be read at all -- a DIFFERENT claim from "the guard is off", and one
    the report must be able to make. Defaulting a failed read to ``False``
    would tell a reader the uniqueness violations are expected warn-mode
    backlog when in fact nothing is known about the regime (INV-2
    no-silent-fail-soft).

    Pass *config* to report the regime the run ACTUALLY operated under: the
    census builds its backend from one :class:`FusedMemoryConfig`, and a
    second, independently-constructed one can resolve a different
    ``CONFIG_PATH`` and disclose an ``enforce`` state belonging to a config
    the census never used. Omitting it keeps the standalone contract for
    callers that have none.

    Never raises: losing the whole census to a config fault would be a
    strictly worse outcome than an undisclosed flag, since the measurement
    itself does not depend on the config.
    """
    try:
        if config is None:
            from fused_memory.config.schema import FusedMemoryConfig  # noqa: PLC0415

            config = FusedMemoryConfig()
        return bool(config.memory_metadata.enforce)
    except Exception as exc:  # noqa: BLE001 - any load failure degrades to null
        logger.warning(
            'could not read memory_metadata.enforce (%s: %s) -- reporting null',
            type(exc).__name__,
            exc,
        )
        return None


def _build_topic_coverage(
    census: CategoryCensus,
    canonical_uniqueness_enforced: bool | None = None,
) -> dict[str, Any]:
    """Turn a MERGED census rollup into the topic/canonical COVERAGE block.

    Takes the per-project (or grand-total) rollup produced by
    :meth:`CategoryCensus.merge`, never a single cell.  That is load-bearing,
    not incidental: 3198's write-time uniqueness probe is
    ``count_memories_by_metadata(project_id, {'topic': T, 'canonical': True})``
    (fused-memory/src/fused_memory/services/memory_service.py::_check_canonical_uniqueness)
    -- scope-wide, NOT category-filtered.  Measuring
    the partition at a finer grain than the invariant is stated at would
    score a topic whose canonical sits in a different category from its
    peers as "zero canonical" and split a genuine cross-category duplicate
    into two well-formed cells: false gaps and missed violations at once.

    ``topic_coverage_pct`` is the RATE the existing raw counts cannot
    express.  It is ``None`` -- never ``0.0`` -- for an empty population: a
    0.0 asserts "measured, and nothing is stamped", which is a different
    claim from "there was nothing to measure", and fabricating the former is
    the failure task 3291 hardened ``extract_done_count`` against.

    NAMED CONSUMER. The two slug-conformance partitions below are the
    standing re-measurement of gate **3626**'s recipe items 3 and 4 -- that
    task's description already names this script as their instrument, and
    until now the census emitted neither number, so a ``pending`` milestone
    carried a recipe its own instrument could not execute. They stay two
    columns rather than one because the predicates differ and PRD §159's
    discharge claim is stated over the narrower one only; see the block
    comment at the computation for which is which, and why a non-empty
    canonical-scoped list REGRESSES that discharge.
    """
    by_topic = census.canonical_true_by_topic
    exactly_one = zero = multiple = 0
    violators: list[tuple[str, int]] = []
    for topic in census.topic_values:
        count = by_topic.get(topic, 0)
        if count == 0:
            zero += 1
        elif count == 1:
            exactly_one += 1
        else:
            multiple += 1
            violators.append((topic, count))
    # Same total order _table() uses (count desc, then value asc), so the
    # artifact stays byte-stable across re-runs over an unchanged corpus.
    violators.sort(key=lambda tc: (-tc[1], tc[0]))

    # SLUG CONFORMANCE -- two partitions, because gate 3626's re-measurement
    # recipe asks two different questions and PRD §159's discharge claim is
    # stated over only the second:
    #
    #   item 3: the non-conforming DISTINCT-TOPIC count, corpus-wide. Its
    #           history (leaf-alpha census: 98 -> 103 on 2026-08-04) is a
    #           distinct-VALUE series, so a legacy topic stamped on 8 records
    #           is ONE non-conforming topic, not eight.
    #   item 4: every live ``canonical: true`` record's topic passes the slug
    #           regex, per project. Strictly narrower than item 3.
    #
    # A non-empty canonical-scoped list REGRESSES the discharge PRD §159
    # records for the ``legacy_topic_spelling_remains`` precondition. Merging
    # the two would report the exact state §159 records as of 2026-08-04 --
    # legacy spellings among ordinary records, none among canonicals -- as an
    # undifferentiated failure.
    non_conforming: list[tuple[str, int]] = [
        (topic, count)
        for topic, count in census.topic_values.items()
        if not is_valid_topic_slug(topic)
    ]
    non_conforming.sort(key=lambda tc: (-tc[1], tc[0]))

    canonical_non_conforming: list[tuple[str, int]] = [
        (topic, count)
        for topic, count in by_topic.items()
        if not is_valid_topic_slug(topic)
    ]
    canonical_non_conforming.sort(key=lambda tc: (-tc[1], tc[0]))

    # `dict[str, Any]`, not the declared `dict[str, str]`: the copy is widened
    # below with a nested `live_re_measurement` mapping.
    preconditions: list[dict[str, Any]] = [dict(e) for e in ENFORCE_FLIP_PRECONDITIONS]
    # The legacy-spelling citation carries a RECORDED status (discharged
    # 2026-08-04) plus THIS run's live counts as its standing re-measurement:
    # a bucket recorded empty can regrow, and the 98 -> 103 history proves it
    # does. 3202 and 3626 get no such field -- they are cited, not measured.
    preconditions[0]['live_re_measurement'] = {
        'slug_non_conforming': len(non_conforming),
        'canonical_slug_non_conforming': len(canonical_non_conforming),
    }

    return {
        'records': census.records,
        'topic_present': census.topic_present,
        'topic_coverage_pct': (
            round(census.topic_present * 100.0 / census.records, 4)
            if census.records
            else None
        ),
        'distinct_topics': len(census.topic_values),
        # Sums to distinct_topics by construction -- the partition is a
        # walk over topic_values, so no topic can fall out of it.
        'topics_with_one_canonical': exactly_one,
        'topics_with_zero_canonical': zero,
        'topics_with_multiple_canonical': multiple,
        # NAMED, not merely counted: an operator cannot act on an integer.
        'multiple_canonical_topics': [
            {'topic': t, 'canonical_count': c} for t, c in violators
        ],
        # The regime the count above was produced under. Without it a reader
        # cannot tell "the guard is warn-only, so this is backlog" from "the
        # guard broke". Three-valued -- see read_canonical_uniqueness_enforced.
        'canonical_uniqueness_enforced': canonical_uniqueness_enforced,
        # Deep-copied, never aliased: a consumer mutating the report must not
        # silently rewrite every later run's citations in the same process
        # (the nightly wrapper censuses two projects in one invocation).
        'enforce_flip_preconditions': preconditions,
        # Gate 3626 recipe item 3 -- corpus-wide distinct-topic conformance.
        'slug_conforming': len(census.topic_values) - len(non_conforming),
        'slug_non_conforming': len(non_conforming),
        'non_conforming_topics': [
            {'topic': t, 'count': c} for t, c in non_conforming
        ],
        # Gate 3626 recipe item 4 -- the strictly narrower canonical-scoped
        # predicate PRD §159's discharge claim is actually stated over.
        'canonical_slug_conforming': len(by_topic) - len(canonical_non_conforming),
        'canonical_slug_non_conforming': len(canonical_non_conforming),
        'canonical_slug_non_conforming_topics': [
            {'topic': t, 'count': c} for t, c in canonical_non_conforming
        ],
        'canonical_true': census.canonical_true,
        # Named rather than dropped: these canonicals are invisible to every
        # per-topic tally above, so without this the block cannot be
        # reconciled against the canonical_true axis.
        'canonical_true_without_topic': census.canonical_true_without_topic,
    }


#: The SCALAR canonical-partition columns, re-graded by summing the
#: per-project cells. Iterated directly by
#: :func:`_regrade_uniqueness_at_project_grain` -- this tuple IS the loop's
#: subject, not a parallel description of it, so the two cannot drift.
#: (``column``, not ``field``: the module imports ``dataclasses.field``.)
_UNIQUENESS_GRAIN_SCALARS: tuple[str, ...] = (
    'topics_with_one_canonical',
    'topics_with_zero_canonical',
    'topics_with_multiple_canonical',
)

#: The named-violator list, re-graded by UNION rather than by sum -- a
#: different operation, hence a separate name rather than a fourth scalar.
_UNIQUENESS_GRAIN_NAMED_LIST = 'multiple_canonical_topics'

#: The canonical-partition fields of a ``topic_coverage`` block, i.e. exactly
#: those whose meaning is fixed by 3198's per-(project, topic) invariant.
#: Everything else in the block is a legitimate corpus-wide rollup.
#: DERIVED from the two above, never restated: an earlier spelling listed the
#: columns here AND again as a literal inside the regrade loop, so a field
#: added to this constant would have been documented as grain-fixed while
#: nothing re-graded it -- the corpus-wide block would headline a
#: merged-across-projects number again, the exact defect the regrade exists
#: to fix. Pinned by ``test_the_regrade_touches_exactly_the_fields_the_
#: constant_names``, in both directions.
_UNIQUENESS_GRAIN_FIELDS: tuple[str, ...] = (
    *_UNIQUENESS_GRAIN_SCALARS,
    _UNIQUENESS_GRAIN_NAMED_LIST,
)


def _regrade_uniqueness_at_project_grain(
    grand_block: dict[str, Any],
    projects: dict[str, Any],
) -> dict[str, Any]:
    """Re-grade the corpus-wide canonical partition onto (project, topic) cells.

    WHY THIS EXISTS, measured on the first live regeneration. The corpus-wide
    block is built from a census merged ACROSS projects, which merges the two
    topic NAMESPACES. 3198's uniqueness invariant is per-(project, topic) --
    its probe is ``count_memories_by_metadata(project_id, {'topic': T,
    'canonical': True})``, bound to ONE project -- so a topic correctly
    carrying one canonical in ``dark_factory`` and one in ``reify`` is two
    COMPLIANT cells, and the cross-project merge scored it as a violation.
    The live instance is ``merge-request-bare-task-id-branch-arg``.

    The symptom was worse than a wrong integer: the report headlined "Topics
    carrying more than one live canonical: true: **1**" directly above a
    named-violator table reading "(none)", because the count came from the
    merged block and the table was already iterating per-project. A reader
    could only conclude the instrument was broken.

    SCOPE IS DELIBERATELY NARROW -- only :data:`_UNIQUENESS_GRAIN_FIELDS`. The
    rest of the block is a genuine corpus-wide rollup and stays one. In
    particular the slug-conformance axes are NOT re-graded: gate 3626's recipe
    item 3 is a distinct-VALUE series whose recorded history (98 -> 103) would
    step discontinuously if a topic present in both projects started counting
    twice. ``distinct_topics`` likewise stays a distinct-value count, so it is
    deliberately NOT the sum of the partition below -- the two axes answer
    different questions and the block would rather be honestly non-additive
    than quietly re-grain one of them.

    PURE: neither argument is mutated; a new dict is returned.
    """
    regraded = dict(grand_block)
    blocks = [
        (project_id, projects[project_id].get('topic_coverage') or {})
        for project_id in sorted(projects)
    ]

    for column in _UNIQUENESS_GRAIN_SCALARS:
        regraded[column] = sum(block.get(column) or 0 for _, block in blocks)

    # The union of the per-project named lists, each row carrying the project
    # it was measured in: at corpus grain "topic X is duplicated" is not
    # actionable without saying WHERE. Ordered by count desc then
    # (topic, project) asc, matching _table's existing total order, so two
    # runs over the same corpus produce byte-identical artifacts.
    merged: list[dict[str, Any]] = [
        {'project_id': project_id, **row}
        for project_id, block in blocks
        for row in (block.get(_UNIQUENESS_GRAIN_NAMED_LIST) or [])
    ]
    regraded[_UNIQUENESS_GRAIN_NAMED_LIST] = sorted(
        merged,
        key=lambda row: (-row['canonical_count'], row['topic'], row['project_id']),
    )
    return regraded


#: Schema of the committed, append-only coverage-history file. Bumped only
#: when a persisted RUN's shape changes; an unknown version is refused
#: outright rather than misread (see :func:`load_coverage_history`).
COVERAGE_HISTORY_SCHEMA_VERSION = 1

#: The trend substrate, committed beside the report artifacts. A number
#: measured once is not owned -- the ask names TRENDING as the point -- and
#: diffing consecutive runs needs durable prior state. Precedent:
#: ``docs/legibility/census-state.json``.
DEFAULT_HISTORY_OUT = str(
    _REPO_ROOT / 'plans' / 'memory-metadata-coverage-history.json',
)

#: Retention bound on the committed history. The file is written by a
#: NIGHTLY timer, so it is bounded in runs as well as in per-run size: ~90
#: rows is a quarter of nightly history, enough to see a season's trend.
#: COST, measured on the committed file rather than estimated (10,736 bytes
#: for 2 runs at the current two-project shape): ~5.2 KiB per run, so a full
#: retention window steady-states at **~470 KiB**. An earlier spelling here
#: said "roughly a hundred KB", which was ~4.5x low; OPERATIONS.md §12 quotes
#: the same two figures. Dropping is DISCLOSED in ``retention.dropped_runs``
#: -- a silently truncated history reads as a corpus that only ever had
#: ``max_runs`` runs, the same no-silent-truncation rule the JSON value
#: tables already follow.
DEFAULT_MAX_HISTORY_RUNS = 90


class CoverageHistoryError(RuntimeError):
    """The coverage history exists but cannot be read as a coverage history.

    Raised rather than degrading to an empty history: silently restarting
    from empty loses the trend AND fabricates a "first ever run", which is
    indistinguishable from the real thing. That is the precise failure
    ``census_trigger.extract_done_count`` was hardened against in task 3291,
    where an unusable payload became a done-count of 0, was persisted twice
    as a real baseline, and armed a threshold with it.
    """


#: Scalar columns lifted from a ``topic_coverage`` block into a run row.
#: Scalars ONLY -- see :func:`_coverage_run_row` for why the per-topic
#: tables must never enter a persisted row.
_HISTORY_COVERAGE_COLUMNS: tuple[str, ...] = (
    'records',
    'topic_present',
    'topic_coverage_pct',
    'distinct_topics',
    'topics_with_one_canonical',
    'topics_with_zero_canonical',
    'topics_with_multiple_canonical',
    'canonical_true',
    'canonical_true_without_topic',
    'slug_conforming',
    'slug_non_conforming',
    'canonical_slug_conforming',
    'canonical_slug_non_conforming',
)

#: Registry-gauge scalars, rolled up per project from the top-level gauge.
#: All ``None`` when the gauge did not run -- never 0, which would assert
#: "measured, and no registry topic is stamped".
_HISTORY_REGISTRY_COLUMNS: tuple[str, ...] = (
    'registry_topics_total',
    'registry_topics_with_exactly_one_canonical',
    'registry_topics_with_zero_canonical',
    'registry_topics_with_multiple_canonical',
)


def empty_coverage_history() -> dict[str, Any]:
    """A fresh, empty history -- what an ABSENT file loads as."""
    return {
        'schema_version': COVERAGE_HISTORY_SCHEMA_VERSION,
        'retention': {
            'max_runs': DEFAULT_MAX_HISTORY_RUNS,
            'dropped_runs': 0,
            'oldest_retained_stamp': None,
        },
        'runs': [],
    }


def load_coverage_history(path: str) -> dict[str, Any]:
    """Read the committed coverage history, or an empty one if absent.

    Absent is a real, expected state (the first ever run) and loads as an
    empty history. Present-but-unusable is NOT: a malformed file, a
    non-object, a missing ``runs`` list or an unrecognised
    ``schema_version`` raises :class:`CoverageHistoryError` naming the path
    and the offending shape (INV-2 structured-facts-at-failure). The
    distinction that matters is presence-of-shape, not emptiness-of-content
    -- exactly as ``extract_done_count`` draws it.
    """
    history_path = Path(path)
    if not history_path.exists():
        return empty_coverage_history()
    try:
        raw = json.loads(history_path.read_text(encoding='utf-8'))
    except Exception as exc:  # noqa: BLE001 - every unreadable shape is named
        raise CoverageHistoryError(
            f'{path}: unreadable coverage history ({type(exc).__name__}: {exc})',
        ) from exc
    if not isinstance(raw, dict):
        raise CoverageHistoryError(
            f'{path}: coverage history is a {type(raw).__name__}, not an object',
        )
    version = raw.get('schema_version')
    if version != COVERAGE_HISTORY_SCHEMA_VERSION:
        raise CoverageHistoryError(
            f'{path}: coverage history schema_version {version!r} is not the '
            f'{COVERAGE_HISTORY_SCHEMA_VERSION} this build reads -- refusing to '
            'misread it as one',
        )
    if not isinstance(raw.get('runs'), list):
        raise CoverageHistoryError(
            f'{path}: coverage history has no "runs" list (found '
            f'{type(raw.get("runs")).__name__})',
        )
    return raw


def save_coverage_history(history: dict[str, Any], path: str) -> None:
    """Write the history file, creating its parent directory if needed."""
    history_path = Path(path)
    history_path.parent.mkdir(parents=True, exist_ok=True)
    history_path.write_text(
        json.dumps(history, indent=2, sort_keys=False) + '\n', encoding='utf-8',
    )


def _registry_history_slices(
    registry_coverage: dict[str, Any] | None,
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, int]]]:
    """Split the top-level registry gauge into per-project scalars + a digest.

    The gauge spans projects (each row carries its own ``project_id``), but
    the history is trended per project, so its rows are rolled up here
    rather than persisted whole.
    """
    if not registry_coverage:
        return {}, {}
    per_project: dict[str, dict[str, Any]] = {}
    digest: dict[str, dict[str, int]] = {}
    for row in registry_coverage.get('topics') or []:
        project_id = str(row.get('project_id'))
        canonical = int(row.get('canonical_count') or 0)
        bucket = per_project.setdefault(
            project_id, dict.fromkeys(_HISTORY_REGISTRY_COLUMNS, 0),
        )
        bucket['registry_topics_total'] += 1
        if canonical == 0:
            bucket['registry_topics_with_zero_canonical'] += 1
        elif canonical == 1:
            bucket['registry_topics_with_exactly_one_canonical'] += 1
        else:
            bucket['registry_topics_with_multiple_canonical'] += 1
        digest[f'{project_id}::{row.get("topic")}'] = {
            'records': int(row.get('records') or 0),
            'canonical': canonical,
        }
    return per_project, {k: digest[k] for k in sorted(digest)}


def _coverage_run_row(report: dict[str, Any], *, stamp: str | None = None) -> dict[str, Any]:
    """Distil one census report into a COMPACT persisted run.

    What this row may NOT hold is the load-bearing half. The history is
    written by a nightly timer and committed, so the bound on the repo's
    growth has to be STRUCTURAL rather than a promise: every per-project
    value here is a scalar, and none of ``_table()``'s ``{'entries',
    'distinct_total'}`` structures may enter. reify alone carries 276
    distinct topic values; persisting those tables nightly would grow the
    committed file without bound. The complete tables live in the report
    artifact, which is REWRITTEN rather than appended.

    The single exception is ``registry_topics``, and only because the
    COMMITTED 32-entry registry bounds it. It is what lets the trend name
    the actionable regression -- "this registry topic LOST its canonical"
    -- at ~32 tiny rows per run instead of 353 live topic values. Its scope
    is disclosed by construction: a topic outside the registry cannot
    appear in it.

    *stamp* is INJECTED, never read from the clock here, so a run row is
    deterministic and testable and the artifact stays reproducible. It is
    optional because :func:`build_coverage_diff` distils the CURRENT report
    through this same function purely to compare it -- one distillation
    path, so the persisted row and the trended row can never disagree about
    what a column means.
    """
    registry_by_project, registry_topics = _registry_history_slices(
        report.get('registry_coverage'),
    )
    project_blocks = report.get('projects') or {}
    projects: dict[str, dict[str, Any]] = {}
    # The UNION: a registry entry may name a project this run did not
    # census. Dropping it would shrink the trended target denominator and
    # make the gauge look closer to met than it is. Its coverage columns
    # stay None -- not measured is not zero.
    for project_id in sorted(set(project_blocks) | set(registry_by_project)):
        coverage_block = (project_blocks.get(project_id) or {}).get('topic_coverage') or {}
        row: dict[str, Any] = {
            column: coverage_block.get(column) for column in _HISTORY_COVERAGE_COLUMNS
        }
        row.update(
            registry_by_project.get(project_id)
            or dict.fromkeys(_HISTORY_REGISTRY_COLUMNS),
        )
        projects[project_id] = row

    return {
        'stamp': stamp,
        # The regime the counts above were measured under. A uniqueness
        # violation count trended across a flip of memory_metadata.enforce
        # is two different series; without the flag beside it the trend is
        # unreadable. Three-valued -- see read_canonical_uniqueness_enforced.
        'canonical_uniqueness_enforced': (
            report.get('topic_coverage') or {}
        ).get('canonical_uniqueness_enforced'),
        'registry_error': report.get('registry_error'),
        'projects': projects,
        'registry_topics': registry_topics,
    }


def append_coverage_run(
    history: dict[str, Any],
    report: dict[str, Any],
    *,
    stamp: str,
    max_runs: int = DEFAULT_MAX_HISTORY_RUNS,
) -> dict[str, Any]:
    """Append one run to *history*, oldest-first, returning a NEW history.

    Never mutates its argument: ``_run`` holds the loaded history while it
    writes artifacts, and an in-place append would leave a half-updated
    structure behind on a later failure.

    Retention drops from the OLD end when the run count exceeds *max_runs*,
    and the drop is disclosed in ``retention.dropped_runs`` rather than
    silently applied -- a truncated history is indistinguishable from a
    short one, and the whole point of the file is that its span is legible.
    """
    runs = [*(history.get('runs') or []), _coverage_run_row(report, stamp=stamp)]
    dropped_before = int((history.get('retention') or {}).get('dropped_runs') or 0)
    dropping = max(0, len(runs) - max_runs)
    kept = runs[dropping:]
    return {
        'schema_version': COVERAGE_HISTORY_SCHEMA_VERSION,
        'retention': {
            'max_runs': max_runs,
            'dropped_runs': dropped_before + dropping,
            'oldest_retained_stamp': kept[0]['stamp'] if kept else None,
        },
        'runs': kept,
    }


#: How the regrowth signal's bound is disclosed to a reader. It is scoped
#: to the committed registry because that is the only per-topic state the
#: history may bound-safely carry; saying so is not optional, since a
#: silently narrowed signal reads as a corpus-wide all-clear.
_REGROWTH_SCOPE_NOTE = (
    'Scoped to the committed topic registry (the accountable target set) -- '
    'the only per-topic state the nightly history carries, since the live '
    'distinct-topic population is unbounded. Corpus-wide topic movement is '
    'trended through the distinct_topics / slug_non_conforming columns '
    'instead.'
)


def _regrowth_unavailable(reason: str) -> dict[str, Any]:
    """A regrowth block that measures nothing, and says which nothing."""
    return {
        'available': False,
        'scope': 'registry',
        'scope_note': _REGROWTH_SCOPE_NOTE,
        'reason': reason,
        'new_topics': [],
        'grown_topics': [],
        'lost_canonical_topics': [],
    }


def _topic_regrowth(
    before: dict[str, Any],
    after: dict[str, Any],
) -> dict[str, Any]:
    """Name the per-topic classes that MOVED between two registry digests.

    Both sides must exist. A missing baseline digest would otherwise report
    every registry target as "newly appearing" the first time the gauge
    runs -- a fabricated signal indistinguishable from a genuine 32-topic
    regression -- and a missing current digest would report silence, which
    reads as "nothing moved" rather than "nothing was measured" (INV-2).
    """
    if not before:
        return _regrowth_unavailable(
            'the baseline run carries no registry digest, so nothing can be '
            'compared against it',
        )
    if not after:
        return _regrowth_unavailable(
            'this run produced no registry digest (the gauge did not run)',
        )

    def _split(key: str) -> tuple[str, str]:
        project_id, _, topic = key.partition('::')
        return project_id, topic

    new_topics = []
    grown_topics = []
    lost_canonical_topics = []
    for key in sorted(after):
        project_id, topic = _split(key)
        now = after[key]
        was = before.get(key)
        if was is None:
            new_topics.append({
                'project_id': project_id,
                'topic': topic,
                'records': now.get('records'),
                'canonical': now.get('canonical'),
            })
            continue
        if (now.get('records') or 0) > (was.get('records') or 0):
            grown_topics.append({
                'project_id': project_id,
                'topic': topic,
                'records_before': was.get('records'),
                'records_after': now.get('records'),
            })
        # The actionable REGRESSION: a target that was met and no longer is.
        if (was.get('canonical') or 0) > 0 and (now.get('canonical') or 0) == 0:
            lost_canonical_topics.append({
                'project_id': project_id,
                'topic': topic,
                'canonical_before': was.get('canonical'),
                'canonical_after': now.get('canonical'),
            })
    return {
        'available': True,
        'scope': 'registry',
        'scope_note': _REGROWTH_SCOPE_NOTE,
        'new_topics': new_topics,
        'grown_topics': grown_topics,
        'lost_canonical_topics': lost_canonical_topics,
    }


def _column_delta(before: Any, after: Any) -> dict[str, Any]:
    """One trended column: both endpoints and the movement between them.

    ``delta`` is ``None`` -- never 0 -- when either endpoint is unmeasured.
    "We cannot compare these" is a different claim from "nothing changed",
    and only one of them is a measurement.
    """
    comparable = isinstance(before, (int, float)) and isinstance(after, (int, float))
    delta: Any = None
    if comparable and not isinstance(before, bool) and not isinstance(after, bool):
        delta = after - before
        if isinstance(delta, float):
            delta = round(delta, 4)
    return {'before': before, 'after': after, 'delta': delta}


def _no_baseline_diff(reason: str) -> dict[str, Any]:
    """A trend with nothing behind it -- stated, not rendered as zeros."""
    return {
        'baseline': None,
        'baseline_reason': reason,
        # Deliberately EMPTY rather than a zero-filled column set: a table
        # of zeros reads as "coverage is flat", which is a claim this run
        # cannot make (task 3291's fabricated-baseline failure).
        'projects': {},
        'topic_regrowth': _regrowth_unavailable(reason),
    }


def build_coverage_diff(
    current: dict[str, Any],
    history: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Diff this run's coverage against the most recent prior run.

    PURE: no I/O and no clock, so it is fully unit-testable and a re-run
    over the same inputs produces a byte-identical block. Neither argument
    is mutated.

    Args:
        current: the report being built (its ``projects``,
            ``topic_coverage`` and ``registry_coverage`` blocks).
        history: a history as returned by :func:`load_coverage_history`.
            ``None`` means none was supplied -- disclosed DISTINCTLY from an
            empty one, because "nobody asked for a trend" and "there is no
            prior run yet" are different facts about the same null.

    Returns:
        ``{baseline, baseline_reason, projects, topic_regrowth}``. Every
        shortfall is named rather than defaulted: no baseline yields no
        deltas at all, an unmeasured axis yields a ``null`` delta, a project
        seen for the first time is ``new_project`` rather than a delta from
        zero, and one that vanished is ``absent_from_this_run`` rather than
        silently dropped out of the trend.
    """
    if history is None:
        return _no_baseline_diff(
            'no coverage history was supplied to this build -- the trend was '
            'not requested, so no baseline was sought',
        )
    runs = list(history.get('runs') or [])
    if not runs:
        return _no_baseline_diff(
            'no prior run is recorded in the coverage history: this is the '
            'first recorded run, so there is nothing to compare against',
        )

    baseline = runs[-1]
    row = _coverage_run_row(current)
    before_projects = baseline.get('projects') or {}
    after_projects = row.get('projects') or {}

    projects: dict[str, Any] = {}
    for project_id in sorted(set(before_projects) | set(after_projects)):
        if project_id not in before_projects:
            projects[project_id] = {'status': 'new_project'}
            continue
        if project_id not in after_projects:
            # Named, not dropped: a project silently leaving the trend looks
            # identical to one that never existed.
            projects[project_id] = {'status': 'absent_from_this_run'}
            continue
        before_row = before_projects[project_id]
        after_row = after_projects[project_id]
        projects[project_id] = {
            'status': 'compared',
            'columns': {
                column: _column_delta(before_row.get(column), after_row.get(column))
                for column in (*_HISTORY_COVERAGE_COLUMNS, *_HISTORY_REGISTRY_COLUMNS)
            },
        }

    return {
        'baseline': baseline.get('stamp'),
        'baseline_reason': (
            f'compared against the most recent prior run ({baseline.get("stamp")})'
        ),
        'projects': projects,
        'topic_regrowth': _topic_regrowth(
            baseline.get('registry_topics') or {}, row.get('registry_topics') or {},
        ),
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

    The collection total is bracketed the same way, by a ``count(scope)``
    read before the category loop and another after it. The per-category
    pair cannot stand in for it: a write landing after the last recount but
    before the closing read moves ONLY the collection total, so without its
    own bracket that drift is indistinguishable from a slice of the corpus
    carrying a category outside the censused set.

    Returns:
        ``(cells, coverage)`` where *cells* maps category value ->
        CategoryCensus, and *coverage* is the per-project record consumed by
        :func:`build_report`: collection name, the bracketing
        ``collection_points_before``/``collection_points`` pair, and
        per-category expected/recount/scrolled/delta/complete.
    """
    from fused_memory.models.scope import Scope  # noqa: PLC0415

    category_values = list(categories) if categories is not None else [
        c.value for c in CENSUS_CATEGORIES
    ]
    scope = Scope(project_id=project_id)
    # Still resolved here even though the scroll no longer needs it: the
    # collection name is part of the coverage record.
    collection = scope.mem0_collection_name(backend.config.mem0.collection_prefix)

    # Bracket the WHOLE scan, the same way each category brackets its own
    # scroll. A write landing after the last per-category recount but before
    # the closing read moves only the collection total, which no pair of
    # category sums can bracket -- so the collection total needs its own two
    # reads for _build_coverage to tell that drift from an unenumerated slice.
    collection_points_before = await backend.count(scope)

    cells: dict[str, CategoryCensus] = {}
    per_category: dict[str, Any] = {}

    for category in category_values:
        filters = {'category': category}
        expected = await backend.count_by_metadata(scope, filters)

        census = CategoryCensus()
        # The backend owns the offset/next_offset walk, the per-page read
        # bound and the page budget; the SAME scope and filters go to
        # count_by_metadata above, and the shared _build_payload_filter means
        # the count and this scroll provably select the same points.
        async for record in backend.scroll_all_by_metadata(
            scope, filters, page_size=page_size, max_pages=max_pages,
        ):
            census.add(record['metadata'])

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
    sum_expected = sum(c['expected'] for c in per_category.values())
    sum_recount = sum(
        c['expected'] if c.get('recount') is None else c['recount']
        for c in per_category.values()
    )
    counted_lo, counted_hi = min(sum_expected, sum_recount), max(sum_expected, sum_recount)
    points_lo, points_hi = min(collection_points_before, collection_points), max(
        collection_points_before, collection_points)
    # The SURVIVING residue -- the gap between the two intervals, 0 when they
    # intersect. Logging the raw `collection_points - sum_expected` here would
    # print ordinary churn as an uncovered slice, which is the report-level
    # defect this bracket exists to fix.
    if points_lo > counted_hi:
        residue = points_lo - counted_hi
    elif points_hi < counted_lo:
        residue = points_hi - counted_lo
    else:
        residue = 0
    logger.info(
        'project=%s collection=%s points=%d..%d counted=%d..%d residue=%+d',
        project_id, collection, points_lo, points_hi, counted_lo, counted_hi, residue,
    )
    if residue == 0 and (counted_lo, counted_hi) != (points_lo, points_hi):
        logger.info(
            'CORPUS CHURN project=%s collection=%s: counted %d..%d vs %d..%d points; the '
            'intervals overlap, so every point is accounted for.',
            project_id, collection, counted_lo, counted_hi, points_lo, points_hi,
        )

    return cells, {
        'collection': collection,
        'collection_points_before': collection_points_before,
        'collection_points': collection_points,
        'categories': per_category,
    }


def _regen_command(params: dict[str, Any]) -> str:
    """Build the invocation that reproduces the artifact *params* came from.

    Bare ``census_memory_metadata.py`` only reproduces a run made entirely
    with CLI defaults. Task 3507: the committed artifact was generated with
    a non-default ``--top-n``, but the header still printed the bare command
    -- following it silently truncated the markdown ~4,800 rows shorter
    while the run still exited 0. Reconstructs the three flags actually
    recorded in ``report['params']`` -- ``--project-id``, ``--page-size``
    and ``--top-n`` -- emitting each one that departs from its argparse
    default.

    ``--max-pages``, ``--json-out``, ``--md-out`` and ``--config`` are never
    recorded in ``params`` -- none of the four shape the census's
    *content*, so :func:`build_report` never receives them and this
    function has nothing to reconstruct them from. If a run used a
    non-default value for one of those, reproducing it exactly is the
    caller's responsibility, not this header's.
    """
    # REPO-ROOT relative, and identical to what the nightly wrapper runs and
    # what OPERATIONS.md §12 documents. The earlier spelling named the script
    # relative to `fused-memory/` while `--registry` / `--history-out` carry
    # `_repo_relative` values -- relative to the repo ROOT -- so a run using
    # either non-default path emitted a command whose script and whose flags
    # could not both resolve from any single directory. `--frozen` is not
    # decoration: regenerating a committed artifact under a resolved-fresh
    # dependency set is not a reproduction of the run that wrote it.
    parts = [
        'uv run --frozen --project fused-memory python '
        'fused-memory/scripts/census_memory_metadata.py',
    ]
    projects = params.get('projects') or []
    if sorted(projects) != sorted(DEFAULT_PROJECT_IDS):
        parts.extend(f'--project-id {project_id}' for project_id in projects)
    page_size = params.get('page_size')
    if page_size is not None and page_size != DEFAULT_PAGE_SIZE:
        parts.append(f'--page-size {page_size}')
    top_n = params.get('top_n')
    if top_n is not None and top_n != DEFAULT_TOP_N:
        parts.append(f'--top-n {top_n}')
    # Both path params are compared through _resolved_repo_path, so an
    # artifact reads the same way in every checkout: a run made in a worktree
    # lane and read from the main checkout must not emit a --registry flag
    # naming a path that only ever existed in that lane. This also normalises
    # the absolute spellings written by pre-4006 artifacts.
    registry = params.get('registry')
    if registry is not None and _resolved_repo_path(registry) != _resolved_repo_path(
        DEFAULT_REGISTRY_PATH,
    ):
        parts.append(f'--registry {registry}')
    history_out = params.get('history_out')
    if history_out is not None and _resolved_repo_path(
        history_out,
    ) != _resolved_repo_path(DEFAULT_HISTORY_OUT):
        parts.append(f'--history-out {history_out}')
    if params.get('no_history'):
        parts.append('--no-history')
    return ' '.join(parts)


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
    regen_command = _regen_command(report.get('params', {}))
    lines: list[str] = [
        '# Mem0 metadata corpus census',
        '',
        'Measured population of metadata keys, `kind` values and `supersedes` shapes '
        'across the live Mem0 collections — the empirical input to '
        '`docs/prds/memory-metadata-vocabulary.md` (leaf alpha of §9).',
        '',
        'Generated by `fused-memory/scripts/census_memory_metadata.py` (read-only). '
        f'Regenerate with `{regen_command}`.',
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

    lines += _render_topic_coverage(report, top_n)
    lines += _render_coverage_trend(report)

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
                  'per-category counts account for every point in each collection — up to '
                  'corpus churn, which is listed below rather than charged as missing '
                  'data.', '']
    else:
        lines += [
            '> **WARNING — COVERAGE INCOMPLETE.** This census under-enumerates the corpus; '
            'the counts below are a LOWER BOUND, not the population. Do not seed a '
            'registry or grandfather list from it until the deltas below are resolved.',
            '',
        ]

    lines += [
        '| project | collection | collection points | counted | residue | complete |',
        '| --- | --- | ---: | ---: | ---: | --- |',
    ]
    for project_id in sorted(coverage.get('projects', {})):
        p = coverage['projects'][project_id]
        lines.append(
            f'| `{project_id}` | `{p.get("collection")}` | '
            f'{_interval(p.get("collection_points_interval"), p.get("collection_points", 0))} | '
            f'{_interval(p.get("counted_interval"), p.get("counted", 0))} | '
            f'{p.get("uncovered_points", 0):+,} | '
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
        'The corpus is LIVE, so every figure above is a pair of unsynchronised reads, and '
        'drift between them is not a defect. Per CELL: `expected` is counted before the '
        'scroll and `recount` after it, so a non-zero `delta` on a cell whose `scrolled` '
        'matches `recount` is churn (a write landed mid-census), NOT a truncated scan — '
        'such a cell is still complete, and only a `scrolled` matching neither count is a '
        'shortfall. Per COLLECTION: the total is bracketed by a `count` before the scan '
        'and another after it, and the counted total spans `sum(expected)`..`sum(recount)`; '
        'the project is covered when those two intervals OVERLAP. The `residue` column is '
        'the surviving GAP between them — the minimum number of points churn cannot '
        'explain, `+0` whenever they overlap. Movement inside that window is listed as '
        'churn below, never as a shortfall.',
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
            if entry.get('kind') == 'collection_total_moved':
                lines.append(
                    f'- `{entry.get("project_id")}` / `{entry.get("collection")}`: the '
                    f'collection total moved to '
                    f'{_interval(entry.get("collection_points_interval"), 0)} points while '
                    f'the categories counted '
                    f'{_interval(entry.get("counted_interval"), 0)}; the intervals overlap, '
                    f'so every point is accounted for.',
                )
            else:
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
            # Dispatch on kind EXPLICITLY. An unconditional fallthrough gave
            # every unrecognised kind the uncovered-points sentence, which is
            # a wrong diagnosis rather than a missing one.
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
            elif kind == 'uncovered_points':
                lines.append(
                    f'- `uncovered_points` — `{d.get("project_id")}` / '
                    f'`{d.get("collection")}`: {d.get("delta", 0):,} of '
                    f'{d.get("collection_points", 0):,} points carry a category outside '
                    f'the censused set (counted {d.get("counted", 0):,}).',
                )
            elif kind == 'surplus_counted':
                # The mirror image: the collection holds FEWER points than the
                # categories counted. Nothing is unenumerated, so the
                # uncovered-points sentence would be an outright wrong answer.
                lines.append(
                    f'- `surplus_counted` — `{d.get("project_id")}` / '
                    f'`{d.get("collection")}`: the categories counted '
                    f'{_interval(d.get("counted_interval"), d.get("counted", 0))}, '
                    f'{abs(d.get("delta", 0)):,} MORE than the '
                    f'{_interval(d.get("collection_points_interval"), 0)} points the '
                    f'collection holds — deletions landed mid-scan beyond the bracket, or '
                    f'a category double-counted. Nothing is missing.',
                )
            else:
                # Neutral and labelled: better an unstyled fact than a
                # confident wrong diagnosis.
                lines.append(f'- `{kind}` — {json.dumps(d, sort_keys=True)}')
        lines.append('')
    return lines


def _num(value: Any) -> str:
    """Render a measured number, or ``n/a`` for an UNMEASURED one.

    ``n/a`` and ``0`` must never be confused in this report: one is the
    absence of a measurement, the other is a measurement.
    """
    if value is None:
        return 'n/a'
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, int):
        return f'{value:,}'
    if isinstance(value, float):
        return f'{value:,}'
    return str(value)


def _pct(value: Any) -> str:
    return 'n/a' if value is None else f'{value}%'


def _signed(value: Any) -> str:
    """A delta: signed, or ``n/a`` when the two ends are not comparable."""
    if value is None:
        return 'n/a'
    if value == 0:
        return '0'
    return f'{value:+,}'


def _render_named_rows(
    title: str,
    headers: list[str],
    rows: list[list[str]],
    top_n: int | None = None,
    *,
    aligns: list[str] | None = None,
    cut: bool = True,
) -> list[str]:
    """Render a NAMED row table, cut to *top_n* with the same disclosure
    :func:`_render_table` uses.

    *cut* is opt-out for tables whose length is bounded by construction (the
    registry-scoped ones) and for the ones where a cut would change the
    conclusion rather than merely shorten the view.
    """
    lines = [f'#### {title}', '']
    if not rows:
        lines += ['_(none)_', '']
        return lines
    shown = rows if not cut or not top_n or top_n <= 0 else rows[:top_n]
    resolved_aligns = aligns or ['---'] * len(headers)
    lines += [
        '| ' + ' | '.join(headers) + ' |',
        '| ' + ' | '.join(resolved_aligns) + ' |',
    ]
    lines += ['| ' + ' | '.join(cells) + ' |' for cells in shown]
    lines.append('')
    if len(shown) < len(rows):
        lines += [
            f'_Showing top {len(shown):,} of {len(rows):,} rows — this markdown '
            f'view is **truncated**; the JSON artifact carries the full '
            f'population, and `--top-n` widens this view._',
            '',
        ]
    return lines


def _enforce_caveat(flag: bool | None) -> str:
    """The regime line that makes the violation count above READABLE.

    Three-valued on purpose. Rendering an unread flag as ``false`` would
    tell a reader the duplicates are expected warn-mode backlog when in fact
    nothing is known about the regime.
    """
    if flag is True:
        return (
            '`memory_metadata.enforce`: **true** — a duplicate `canonical: true` '
            'is REJECTED at write time, so a non-zero count above is residue '
            'from before the flip, or a live defect.'
        )
    if flag is False:
        return (
            '`memory_metadata.enforce`: **false** — warn-mode: a duplicate '
            '`canonical: true` is censused and the write still PROCEEDS, so a '
            'non-zero count above is partly backlog, not necessarily a broken '
            'guard.'
        )
    return (
        '`memory_metadata.enforce`: **unknown** — the config could not be read, '
        'so the regime that produced the count above is not known. This is not '
        'the same claim as the guard being off.'
    )


def _render_topic_coverage(report: dict[str, Any], top_n: int | None) -> list[str]:
    """The headline: stamping RATE, the canonical partition, and the target.

    Rendered from the report dict alone, like every other section -- the two
    artifacts cannot disagree because there is only one traversal.
    """
    grand = report.get('topic_coverage') or {}
    projects = report.get('projects') or {}
    lines = [
        '## Topic & canonical coverage',
        '',
        'Stamping **coverage** (a rate, not a raw count) and the per-topic '
        '`canonical` partition. Computed at PROJECT grain across all censused '
        "categories, matching 3198's scope-wide uniqueness probe "
        '(`memory_service._check_canonical_uniqueness`) — a per-category tally '
        'would score a topic whose canonical sits in another category as '
        'having none.',
        '',
        'On the **(all)** row the three canonical columns count '
        '(project, topic) CELLS — they are a sum of the rows above, because '
        'the invariant is per-project and a topic carrying one canonical in '
        'each of two projects is two compliant cells, not a violation. '
        '`distinct topics` on that row stays a distinct-VALUE count '
        '(de-duplicated across projects), which is the axis gate 3626 '
        'recipe item 3 trends. The two therefore need not add up, and are '
        'left honestly non-additive rather than quietly re-grained.',
        '',
        '| scope | records | `topic` present | coverage | distinct topics | '
        '1 canonical | 0 canonical | >1 canonical |',
        '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |',
    ]
    for project_id in sorted(projects):
        block = projects[project_id].get('topic_coverage') or {}
        lines.append(
            f'| `{project_id}` | {_num(block.get("records"))} | '
            f'{_num(block.get("topic_present"))} | '
            f'{_pct(block.get("topic_coverage_pct"))} | '
            f'{_num(block.get("distinct_topics"))} | '
            f'{_num(block.get("topics_with_one_canonical"))} | '
            f'{_num(block.get("topics_with_zero_canonical"))} | '
            f'{_num(block.get("topics_with_multiple_canonical"))} |',
        )
    lines.append(
        f'| **(all)** | {_num(grand.get("records"))} | '
        f'{_num(grand.get("topic_present"))} | '
        f'{_pct(grand.get("topic_coverage_pct"))} | '
        f'{_num(grand.get("distinct_topics"))} | '
        f'{_num(grand.get("topics_with_one_canonical"))} | '
        f'{_num(grand.get("topics_with_zero_canonical"))} | '
        f'{_num(grand.get("topics_with_multiple_canonical"))} |',
    )
    lines.append('')

    # ---- uniqueness: the count, then the REGIME, then what blocks the flip
    lines += [
        '### Canonical uniqueness',
        '',
        'Topics carrying more than one live `canonical: true`: '
        f'**{_num(grand.get("topics_with_multiple_canonical"))}** '
        f'(and **{_num(grand.get("canonical_true_without_topic"))}** '
        '`canonical: true` record(s) carry no topic at all, so they appear in '
        'no per-topic tally above).',
        '',
        _enforce_caveat(grand.get('canonical_uniqueness_enforced')),
        '',
    ]
    # Sourced from the SAME grand list the count above reports, so the two can
    # never disagree. They once did: the count read a cross-project merge while
    # this table iterated per-project, and the artifact shipped "**1**" above a
    # table reading "(none)". _regrade_uniqueness_at_project_grain now puts
    # both on the (project, topic) grain the invariant is stated at, and
    # reading them off one list is what keeps them there.
    violator_rows = [
        [f'`{row.get("project_id")}`', f'`{row["topic"]}`',
         f'{row["canonical_count"]:,}']
        for row in (grand.get('multiple_canonical_topics') or [])
    ]
    lines += _render_named_rows(
        'Topics with more than one `canonical: true`',
        ['project', 'topic', 'canonical count'],
        violator_rows,
        top_n,
        aligns=['---', '---', '---:'],
    )

    # NEVER cut: a truncated precondition list silently under-reports what
    # blocks the flip -- the one list where a cut changes the conclusion
    # rather than merely shortening the view (Leo's 2026-08-12 ruling).
    lines += [
        '#### What still blocks flipping `memory_metadata.enforce`',
        '',
        'Cited, not chased — this census measures, task 3626 decides.',
        '',
    ]
    for entry in grand.get('enforce_flip_preconditions') or []:
        line = (
            f'- **`{entry.get("id")}`** — status `{entry.get("status")}`. '
            f'{entry.get("what")} Source: {entry.get("source")}.'
        )
        live = entry.get('live_re_measurement')
        if live:
            line += (
                ' Re-measured by THIS run: '
                f'**{_num(live.get("slug_non_conforming"))}** non-conforming '
                'distinct topic(s) corpus-wide, '
                f'**{_num(live.get("canonical_slug_non_conforming"))}** among '
                '`canonical: true` topics — a bucket recorded empty can regrow '
                '(this census measured 98 → 103 while in warn mode).'
            )
        lines.append(line)
    lines.append('')

    # ---- slug conformance: two partitions, two questions
    lines += [
        '### Topic slug conformance',
        '',
        "The standing re-measurement of gate **3626**'s recipe: **item 3** is "
        'the non-conforming DISTINCT-TOPIC count corpus-wide, **item 4** is the '
        'strictly narrower check that every live `canonical: true` topic passes '
        'the slug regex in both projects. They are two columns because they are '
        "two questions, and PRD §159's discharge claim is stated over item 4 "
        'only. A topic stamped in a legacy spelling is invisible to every '
        'registry-keyed and exact-match read.',
        '',
        '| scope | distinct topics | conforming (item 3) | non-conforming (item 3) | '
        '`canonical: true` topics | conforming (item 4) | non-conforming (item 4) |',
        '| --- | ---: | ---: | ---: | ---: | ---: | ---: |',
    ]
    for project_id in sorted(projects):
        block = projects[project_id].get('topic_coverage') or {}
        lines.append(
            f'| `{project_id}` | {_num(block.get("distinct_topics"))} | '
            f'{_num(block.get("slug_conforming"))} | '
            f'{_num(block.get("slug_non_conforming"))} | '
            f'{_num((block.get("canonical_slug_conforming") or 0) + (block.get("canonical_slug_non_conforming") or 0))} | '
            f'{_num(block.get("canonical_slug_conforming"))} | '
            f'{_num(block.get("canonical_slug_non_conforming"))} |',
        )
    lines.append('')
    for project_id in sorted(projects):
        block = projects[project_id].get('topic_coverage') or {}
        lines += _render_named_rows(
            f'`{project_id}` — non-conforming topic values (item 3)',
            ['topic', 'records'],
            [
                [f'`{row["topic"]}`', f'{row["count"]:,}']
                for row in block.get('non_conforming_topics') or []
            ],
            top_n,
            aligns=['---', '---:'],
        )
        lines += _render_named_rows(
            f'`{project_id}` — non-conforming `canonical: true` topics (item 4)',
            ['topic', '`canonical: true` records'],
            [
                [f'`{row["topic"]}`', f'{row["count"]:,}']
                for row in block.get('canonical_slug_non_conforming_topics') or []
            ],
            top_n,
            aligns=['---', '---:'],
        )

    lines += _render_registry_coverage(report, top_n)
    return lines


def _render_registry_coverage(report: dict[str, Any], top_n: int | None) -> list[str]:
    """The TARGET: the committed registry scored against measured state."""
    lines = [
        '### Registry coverage — the accountable target',
        '',
        'Every topic in the committed registry '
        '(`fused-memory/tests/fixtures/memory_eval_topic_registry.json`) should '
        'carry exactly one live `canonical: true` **in its own project**. A '
        'bounded, named, checkable set — unlike a corpus-wide stamping '
        'percentage, which has no owner and no worklist.',
        '',
    ]
    gauge = report.get('registry_coverage')
    if not gauge:
        lines += [
            f'_Registry gauge did not run: {report.get("registry_error")}._',
            '',
        ]
        return lines
    lines += [
        f'Registry topics: **{_num(gauge.get("registry_topics_total"))}** · '
        'exactly one canonical: '
        f'**{_num(gauge.get("registry_topics_with_exactly_one_canonical"))}** · '
        'zero: '
        f'**{_num(gauge.get("registry_topics_with_zero_canonical"))}** · '
        'more than one: '
        f'**{_num(gauge.get("registry_topics_with_multiple_canonical"))}**.',
        '',
    ]
    lines += _render_named_rows(
        'Worklist — registry topics with no live `canonical: true`',
        ['project', 'topic', 'records', 'canonical'],
        [
            [
                f'`{row["project_id"]}`', f'`{row["topic"]}`',
                f'{row["records"]:,}', f'{row["canonical_count"]:,}',
            ]
            for row in gauge.get('zero_canonical_topics') or []
        ],
        top_n,
        aligns=['---', '---', '---:', '---:'],
    )
    return lines


def _render_coverage_trend(report: dict[str, Any]) -> list[str]:
    """This run against the most recent prior one -- or a stated null.

    No table is cut here: every table below is bounded by construction (the
    trended column set, and the committed registry).
    """
    trend = report.get('coverage_trend') or {}
    lines = [
        '## Coverage trend',
        '',
        'Movement since the most recent prior run in '
        '`plans/memory-metadata-coverage-history.json`.',
        '',
    ]
    if not trend.get('baseline'):
        # Deliberately no delta table: a table of zeros would read as
        # "coverage is flat", which is not what an absent baseline means.
        lines += [
            f'_No baseline — {trend.get("baseline_reason", "no prior run")}._',
            '',
        ]
        return lines

    lines += [f'Baseline: `{trend.get("baseline")}`.', '']
    projects = trend.get('projects') or {}
    compared = {
        pid: block for pid, block in projects.items() if block.get('status') == 'compared'
    }
    if compared:
        lines += [
            '| project | column | before | after | delta |',
            '| --- | --- | ---: | ---: | ---: |',
        ]
        for project_id in sorted(compared):
            for column, cell in compared[project_id]['columns'].items():
                lines.append(
                    f'| `{project_id}` | `{column}` | {_num(cell.get("before"))} | '
                    f'{_num(cell.get("after"))} | {_signed(cell.get("delta"))} |',
                )
        lines.append('')
    for project_id in sorted(projects):
        status = projects[project_id].get('status')
        if status == 'new_project':
            lines += [
                f'- `{project_id}`: **new_project** — first appearance in the '
                'trend, so no deltas are shown; a delta from zero would '
                'fabricate a baseline it never had.',
            ]
        elif status == 'absent_from_this_run':
            lines += [
                f'- `{project_id}`: **absent_from_this_run** — present in the '
                'baseline but not censused now, so its trend is paused rather '
                'than dropped.',
            ]
    if any(b.get('status') != 'compared' for b in projects.values()):
        lines.append('')

    regrowth = trend.get('topic_regrowth') or {}
    lines += ['### Registry topic regrowth', '', regrowth.get('scope_note', ''), '']
    if not regrowth.get('available'):
        lines += [f'_Not available: {regrowth.get("reason")}._', '']
        return lines
    lines += _render_named_rows(
        'Lost their `canonical: true`',
        ['project', 'topic', 'canonical before', 'canonical after'],
        [
            [
                f'`{r["project_id"]}`', f'`{r["topic"]}`',
                _num(r.get('canonical_before')), _num(r.get('canonical_after')),
            ]
            for r in regrowth.get('lost_canonical_topics') or []
        ],
        None,
        aligns=['---', '---', '---:', '---:'],
        cut=False,
    )
    lines += _render_named_rows(
        'Grew in member records',
        ['project', 'topic', 'records before', 'records after'],
        [
            [
                f'`{r["project_id"]}`', f'`{r["topic"]}`',
                _num(r.get('records_before')), _num(r.get('records_after')),
            ]
            for r in regrowth.get('grown_topics') or []
        ],
        None,
        aligns=['---', '---', '---:', '---:'],
        cut=False,
    )
    lines += _render_named_rows(
        'Newly registered targets',
        ['project', 'topic', 'records', 'canonical'],
        [
            [
                f'`{r["project_id"]}`', f'`{r["topic"]}`',
                _num(r.get('records')), _num(r.get('canonical')),
            ]
            for r in regrowth.get('new_topics') or []
        ],
        None,
        aligns=['---', '---', '---:', '---:'],
        cut=False,
    )
    return lines


def _interval(bounds: Any, fallback: int) -> str:
    """Render an ``[lo, hi]`` interval, collapsing lo == hi to one figure."""
    if not isinstance(bounds, (list, tuple)) or len(bounds) != 2:
        return f'{fallback:,}'
    lo, hi = bounds
    return f'{lo:,}' if lo == hi else f'{lo:,}–{hi:,}'


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

    config = FusedMemoryConfig()
    backend = _build_backend(config)
    try:
        # Read FIRST, before a multi-minute paginated scroll of both live
        # collections. An unusable history is fatal by design (see the
        # handler below), and failing after the scroll would discard a
        # completed measurement over a fault in an unrelated file that costs
        # milliseconds to validate -- the nightly job would then produce
        # nothing at all, every night, until someone noticed.
        history = load_coverage_history(args.history_out)
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

        registry, registry_error = load_registry_or_error(args.registry)
        report = build_report(
            cells, coverage, top_n=args.top_n, page_size=args.page_size,
            # The SAME config the backend was built from, so the disclosed
            # regime provably belongs to the run that measured the count.
            canonical_uniqueness_enforced=read_canonical_uniqueness_enforced(config),
            registry=registry, registry_error=registry_error,
            history=history,
            # Recorded repo-relative: these two land in a COMMITTED artifact,
            # and an absolute path would name the checkout that happened to
            # generate it (see _repo_relative).
            extra_params={
                'registry': _repo_relative(args.registry),
                'history_out': _repo_relative(args.history_out),
                'no_history': bool(args.no_history),
            },
        )
        _write_artifacts(report, args.json_out, args.md_out)
        if args.no_history:
            logger.info(
                'history NOT appended (--no-history): %s left byte-unchanged',
                args.history_out,
            )
        else:
            save_coverage_history(
                append_coverage_run(
                    history, report,
                    # The clock is read HERE, at the process boundary, and
                    # injected -- append_coverage_run stays pure.
                    stamp=datetime.now(UTC).isoformat(timespec='seconds'),
                ),
                args.history_out,
            )
            logger.info('coverage history appended: %s', args.history_out)

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
    except CoverageHistoryError:
        # Loud, per INV-2: the trend is the point of the file, and a run
        # that quietly started a new series would report "first recorded
        # run" forever while the old one sat unread on disk.
        logger.exception('ABORT: the coverage history could not be read')
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
        '--page-size', dest='page_size', type=int, default=DEFAULT_PAGE_SIZE,
        help=f'Qdrant scroll page size (default: {DEFAULT_PAGE_SIZE})',
    )
    parser.add_argument(
        '--top-n', dest='top_n', type=int, default=DEFAULT_TOP_N,
        help=(
            f'Max rows per value/key table in the MARKDOWN report '
            f'(default: {DEFAULT_TOP_N}). '
            'The JSON report always carries every value; 0 or less renders the '
            'full population in the markdown too.'
        ),
    )
    parser.add_argument(
        '--registry', dest='registry', default=DEFAULT_REGISTRY_PATH,
        help=(
            'Committed topic registry scored by the coverage gauge '
            '(default: the 32-entry fixture). A load failure is DISCLOSED as '
            'registry_coverage: null + registry_error, never as an all-zero '
            'gauge.'
        ),
    )
    parser.add_argument(
        '--history-out', dest='history_out', default=DEFAULT_HISTORY_OUT,
        help=(
            'Committed append-only coverage-history file, read for the trend '
            f'and extended by one compact run (default: {DEFAULT_HISTORY_OUT})'
        ),
    )
    parser.add_argument(
        '--no-history', dest='no_history', action='store_true',
        help=(
            'Measure and report the trend WITHOUT appending to the history. '
            'For ad-hoc operator runs: the committed file is the nightly '
            "timer's series, and a one-off look must not enter it."
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
