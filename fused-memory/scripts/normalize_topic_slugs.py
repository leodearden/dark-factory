#!/usr/bin/env python3
"""Corpus-wide ``metadata.topic`` slug normalization (task 4878).

Folds every non-conforming live ``topic`` value into the shape
:mod:`fused_memory.topic_slug` defines — in practice snake_case to
hyphen-case, which is what all of the measured non-conforming values are —
so that exact-match retrieval (``get_memories_by_metadata({'topic': T})``,
which is how consolidation clusters and the canonical-uniqueness probe
address a topic) stops depending on which spelling a writer happened to use.

THIS IS A CORPUS SWEEP — the deliberate contrast with the sibling
-----------------------------------------------------------------
``scripts/retro_stamp_topics.py`` (PRD leaf θ, task 3201) states in its own
docstring that it is *not* a corpus sweep: every one of its targets is
addressed by memory id, drawn from three enumerated bounded sources, and it
reports ``bounded: True``.  That boundedness is its entire safety argument.

This script is the opposite and says so up front: it walks the whole
collection.  They are separate scripts precisely so that claim stays true
where it is made — bolting an unbounded scroll into the bounded sweep would
falsify its docstring, its report field and its rendered header in place.
The jobs differ too: that one STAMPS a topic onto records carrying none;
this one REWRITES the value of one that already exists.

What is shared is the *fold*.  :func:`fused_memory.topic_slug.derive_topic_slug`
has one home (INV-5) and both migrations import it (this one through its
planner), so the two can never disagree about what a legacy value becomes.

Scope boundary
--------------
This script is the DATA migration and nothing else.  Flipping
``memory_metadata.enforce`` from warn to reject belongs to task 3626, and
changing the writer instructions so new records are born conforming belongs to
task 3202.  Normalizing the existing corpus is worth doing regardless of
whether either ever lands: it is what makes an exact-match ``{'topic': T}``
read return the whole cluster instead of whichever spelling the caller
guessed.

Where things live
-----------------
* The operator runbook (dry run, refusal buckets, apply, re-measure, and the
  hand remedy for a split slug):
  ``fused-memory/scripts/README_normalize_topic_slugs.md``.
* The pure planner:
  ``fused-memory/src/fused_memory/maintenance/topic_slug_plan.py``.
* The gate census and gate pairing:
  ``fused-memory/src/fused_memory/maintenance/topic_slug_gates.py``.
* The write boundary and the two-store lockstep, including why every slug
  moves all-or-nothing:
  ``fused-memory/src/fused_memory/maintenance/topic_slug_lockstep.py``.
* Here: the corpus enumeration, the residue probe, the sweep (``run``), the
  report and the CLI.
"""
from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import logging
import sys
from pathlib import Path
from typing import Any

from fused_memory.backends.mem0_client import (
    DEFAULT_SCROLL_MAX_PAGES,
    ScrollPageBudgetExhausted,
)
from fused_memory.canonical_marker import is_canonical
from fused_memory.maintenance.topic_slug_gates import (
    GateCensus,
    census_consolidation_gates,
    gate_topic,
    pair_gate_blocks,
)
from fused_memory.maintenance.topic_slug_lockstep import (
    MOVED_OUTCOMES,
    REHEARSED_OUTCOMES,
    rename_group,
)
from fused_memory.maintenance.topic_slug_plan import (
    Rename,
    in_selection,
    is_non_conforming_topic,
    plan_renames,
    refuse_invalid_selection,
)
from fused_memory.utils.script_loader import load_script
from fused_memory.utils.store_mutation_preflight import assert_store_mutation_allowed

# This script reports through ``print``; stdout carries its machine-read
# markdown/JSON artifact, so the diagnoses that must not land there -- the
# fail-closed store-mutation refusal, and the coverage warnings -- go through
# this logger instead.  Named for the script basename, matching every other
# guarded script (and what the tests filter ``caplog`` on).
logger = logging.getLogger('normalize_topic_slugs')

_REPO_ROOT = Path(__file__).resolve().parents[2]
_CENSUS_SCRIPT_PATH = (
    _REPO_ROOT / 'fused-memory' / 'scripts' / 'census_memory_metadata.py'
)

# Convention inherited from ``retro_stamp_topics.py``: this list carries every
# imported helper the tests reach through the module object, not only the names
# defined locally.  This script is loaded by path via ``importlib`` and cannot
# be imported from, so its ``__all__`` grants the shared rule no second home.
__all__ = [
    'BASELINE_DISTINCT_NON_CONFORMING',
    'BASELINE_HISTORY',
    'DEFAULT_GATE_ROOTS',
    'DEFAULT_JSON_OUT',
    'DEFAULT_MAX_PAGES',
    'DEFAULT_MD_OUT',
    'DEFAULT_PROJECTS',
    'GateCensusIncomplete',
    'load_mcp_client_class',
    'load_mcp_client_module',
    'open_live_memory_service',
    'open_mcp_client',
    'DEFAULT_PAGE_SIZE',
    'ERROR_OUTCOMES',
    'SKIP_BUCKETS',
    'ScrollPageBudgetExhausted',
    'assert_store_mutation_allowed',
    'census_categories',
    'enumerate_topic_bearing',
    'load_census_module',
    'main',
    'REMEASURE_COMMAND',
    'render_json',
    'render_markdown',
    'resolve_exit_code',
    'resolve_gate_roots',
    'resolve_projects',
    'resolve_topics',
    'run',
    'run_live',
    'verify_old_slugs_drained',
]


#: Outcomes that mean the corpus did not get what the plan intended.  Named
#: once, here, so :func:`resolve_exit_code` and the report agree on what
#: "clean" means instead of each keeping its own list.
ERROR_OUTCOMES: frozenset[str] = frozenset({
    'slug_collision',
    'canonical_collision',
    'topic_unfoldable',
    'under_enumerated',
    'scroll_budget_exhausted',
    'memory_not_found',
    'topic_moved_since_plan',
    'update_failed',
    'rename_error',
    'gate_census_incomplete',
    'gate_ambiguous',
    'gate_lockstep_failed',
    'ungated_partial_rename',
    'legacy_slug_residue',
})

#: Every bucket the report carries, PRE-SEEDED TO EMPTY.  Seeding is the
#: point: an absent bucket reads as "nothing was skipped", which is a
#: different claim from "we looked and found nothing".  A hole in coverage
#: that never reaches the artifact is a hole nobody closes.
SKIP_BUCKETS: tuple[str, ...] = (
    # from enumerate_topic_bearing
    'under_enumerated',
    'scroll_budget_exhausted',
    # from plan_renames
    'topic_unfoldable',
    'record_without_id',
    'slug_collision',
    'canonical_collision',
    # from rename_one
    'memory_not_found',
    'topic_moved_since_plan',
    'update_failed',
    'rename_error',
    # from census_consolidation_gates, measured against the swept projects
    'gate_census_incomplete',
    # from pair_gate_blocks / rename_group — the two-store lockstep
    'orphan_gate_topic',
    'gate_ambiguous',
    'gate_lockstep_failed',
    # from rename_group's hold of a split ungated slug
    'ungated_partial_rename',
    # from verify_old_slugs_drained
    'legacy_slug_residue',
)

#: Page size and page budget for each cell's scroll.  ALIASED from the
#: backend, never restated: the budget travels with the paging loop it bounds
#: (``Mem0Backend.scroll_collection_pages``), and ``census_memory_metadata.py``
#: aliases the same object for the same INV-5 reason.  This script must cover
#: the corpus the way the census does or its coverage claim is not comparable
#: to the census baseline it reports against.
DEFAULT_PAGE_SIZE = 1000
DEFAULT_MAX_PAGES = DEFAULT_SCROLL_MAX_PAGES


# ---------------------------------------------------------------------------
# The corpus partition — borrowed, never restated
# ---------------------------------------------------------------------------

def load_census_module() -> Any:
    """Load ``census_memory_metadata`` by path via ``load_script``, sharing its key's module."""
    return load_script(_CENSUS_SCRIPT_PATH, 'census_memory_metadata')


def census_categories() -> tuple[str, ...]:
    """The category partition this sweep covers — the census's, by import.

    An empty filter dict is REJECTED by both ``scroll_by_metadata`` and
    ``scroll_all_by_metadata`` ("to avoid silently enumerating every memory in
    the collection"), so there is no direct all-records scroll at this seam.
    Partitioning on ``{'category': c}`` is how ``census_memory_metadata.py``
    already covers the whole collection with non-empty filters, and this
    script reuses that partition rather than keeping a second list.

    Two reasons it is imported rather than restated.  INV-5: a corpus
    partition kept in two places drifts, and the failure mode here is a
    SILENTLY unenumerated slice — the hardest kind to notice, because the
    report still looks clean.  And this script's coverage claim is only
    comparable to the census baseline it diffs against (item 4) if the two
    provably walk the same cells.
    """
    return tuple(c.value for c in load_census_module().CENSUS_CATEGORIES)


# ---------------------------------------------------------------------------
# The corpus boundary
# ---------------------------------------------------------------------------

async def enumerate_topic_bearing(
    memory_service: Any,
    project_id: str,
    *,
    categories: list[str] | None = None,
    page_size: int = DEFAULT_PAGE_SIZE,
    max_pages: int = DEFAULT_MAX_PAGES,
) -> dict:
    """Walk one project's whole collection, retaining only the residue.

    Per category: ``count_by_metadata`` -> page-scroll every record ->
    ``count_by_metadata`` again.  The bracket is copied from
    ``census_memory_metadata.census_project`` for the reason it has one:
    counting BEFORE is what makes an under-enumerated scroll detectable at
    all, and counting AFTER brackets the scan against a LIVE corpus, so a
    scroll that agrees with the recount saw churn rather than truncation.
    One read cannot tell those apart, and reporting ordinary churn as a
    coverage hole teaches an operator to ignore the bucket that matters.

    **Only NON-conforming records are retained.**  That is what makes a
    corpus-wide sweep affordable: peak memory is bounded by the residue
    (~141 records at the ad707e72 baseline), never by the ~49.4k-record
    corpus.  It is also why ``CategoryCensus`` cannot be reused here — it
    deliberately retains no payloads at all, so it knows WHICH slugs are
    non-conforming but not WHICH RECORDS carry them, and a migration needs
    the ids.  A present topic that is not a string, or is empty, is
    non-conforming too, and is retained so :func:`plan_renames` reports it.

    The backend is reached as ``memory_service.mem0`` (the
    ``consolidate_namespace_families.py`` pattern) so one injected service
    serves both these reads and the later writes — an end-to-end run cannot
    scroll one store and write to another.

    ``ScrollPageBudgetExhausted`` is caught PER CELL and surfaced as an
    explicit incomplete outcome.  Never swallowed: a migration that caught it
    and moved on would report a completed sweep over a corpus it had only
    partly enumerated, and the residue probe would then find the legacy slug
    still populated and blame the writes.  Caught per cell rather than
    per run because one dead cell must not cost the coverage of the other
    five — the artifact is more useful naming which cell failed.

    Returns:
        ``{'project_id', 'records', 'coverage', 'skips', 'complete'}`` —
        *records* are the retained scroll-shaped residue dicts,
        *coverage* maps category -> the expected/scrolled/recount/delta/
        complete cell, and *complete* is the conjunction over all cells.
    """
    from fused_memory.models.scope import Scope

    backend = memory_service.mem0
    category_values = (
        list(categories) if categories is not None else list(census_categories())
    )
    scope = Scope(project_id=project_id)

    records: list[dict] = []
    coverage: dict[str, dict] = {}
    skips: list[dict] = []
    # The conforming half of the corpus is DISCARDED as records but its topic
    # VALUES are kept, because a collision is a statement about the values a
    # fold could land on and most of those live on records this sweep will
    # never rename.  Retaining the values (a few hundred distinct strings)
    # rather than the records keeps peak memory bounded by the namespace, not
    # by the ~49.4k-record corpus, while leaving the collision guard able to
    # see the incumbent it exists to protect.
    occupied_topics: set[str] = set()
    canonical_ids_by_topic: dict[str, list[str]] = {}

    for category in category_values:
        filters = {'category': category}
        expected = await backend.count_by_metadata(scope, filters)

        scrolled = 0
        exhausted: str | None = None
        try:
            async for record in backend.scroll_all_by_metadata(
                scope, filters, page_size=page_size, max_pages=max_pages,
            ):
                scrolled += 1
                metadata = record.get('metadata') or {}
                topic = metadata.get('topic')
                if isinstance(topic, str):
                    occupied_topics.add(topic)
                    if is_canonical(metadata):
                        memory_id = record.get('id')
                        if isinstance(memory_id, str) and memory_id:
                            canonical_ids_by_topic.setdefault(topic, []).append(memory_id)
                if not is_non_conforming_topic(topic):
                    continue
                # The residue triple: the id to write to, the topic to fold,
                # and whether this record claims canonical (which decides
                # whether a collision is severe).  Nothing else is kept.
                records.append({
                    'id': record.get('id'),
                    'created_at': record.get('created_at'),
                    'metadata': {
                        'topic': topic,
                        **(
                            {'canonical': True}
                            if is_canonical(metadata) else {}
                        ),
                    },
                })
        except ScrollPageBudgetExhausted as exc:
            exhausted = str(exc) or exc.__class__.__name__
            logger.warning(
                'SCROLL BUDGET EXHAUSTED project=%s category=%s after %d records: %s',
                project_id, category, scrolled, exhausted,
            )
            skips.append({
                'reason': 'scroll_budget_exhausted',
                'project_id': project_id,
                'category': category,
                'scrolled': scrolled,
                'error': exhausted,
                'note': (
                    'the backend gave up paging this cell, so its enumeration '
                    'is a LOWER BOUND — do not read a clean residue probe as '
                    'proof the legacy slugs drained'
                ),
            })

        recount = await backend.count_by_metadata(scope, filters)
        delta = scrolled - expected
        complete = exhausted is None and (delta == 0 or scrolled == recount)
        coverage[category] = {
            'expected': expected,
            'scrolled': scrolled,
            'recount': recount,
            'delta': delta,
            'complete': complete,
        }
        if exhausted is None and not complete:
            logger.warning(
                'UNDER-ENUMERATED project=%s category=%s: scrolled %d, count said %d '
                'before and %d after (delta %+d) — this sweep is a LOWER BOUND '
                'for that cell.',
                project_id, category, scrolled, expected, recount, delta,
            )
            skips.append({
                'reason': 'under_enumerated',
                'project_id': project_id,
                'category': category,
                'expected': expected,
                'scrolled': scrolled,
                'recount': recount,
                'delta': delta,
                'note': (
                    'the scroll agreed with neither count, so records in this '
                    'cell may carry a non-conforming topic this run never saw'
                ),
            })
        elif delta != 0 and complete:
            logger.info(
                'CORPUS CHURN project=%s category=%s: count moved %d → %d while '
                'scrolling; the scroll agrees with the re-count, so the scan is '
                'complete.',
                project_id, category, expected, recount,
            )

    return {
        'project_id': project_id,
        'records': records,
        'occupied_topics': sorted(occupied_topics),
        'canonical_ids_by_topic': {
            topic: sorted(ids) for topic, ids in sorted(canonical_ids_by_topic.items())
        },
        'coverage': coverage,
        'skips': skips,
        'complete': all(cell['complete'] for cell in coverage.values()),
    }


def load_mcp_client_class() -> type:
    """Reuse ``FusedMemoryClient`` from ``strip_leaked_control_keys.py``.

    Loaded lazily and by path, exactly as
    ``migrate_task_metadata_to_x_namespace._load_sibling_client`` does — that
    script established this route for task-metadata writes, and there is ONE
    JSON-RPC handshake in this repo (INV-5).  A second protocol client here
    would be a second thing to keep in step with the server.

    Lazy so the pure planner, the pairing and the CLI parser stay importable
    with no HTTP dependency, and so the tests — which inject a double — never
    reach this function at all.
    """
    return load_mcp_client_module().FusedMemoryClient


def load_mcp_client_module() -> Any:
    """The by-path-loaded ``strip_leaked_control_keys`` module itself.

    Exposed so :func:`open_mcp_client` reads that module's ``DEFAULT_SERVER``
    beside its client class, rather than restating the URL a third time.
    """
    import importlib.util

    sibling = Path(__file__).parent / 'strip_leaked_control_keys.py'
    mod_name = 'strip_leaked_control_keys'
    existing = sys.modules.get(mod_name)
    if existing is not None and hasattr(existing, 'FusedMemoryClient'):
        return existing
    spec = importlib.util.spec_from_file_location(mod_name, sibling)
    if spec is None or spec.loader is None:
        raise ImportError(f'Cannot load the JSON-RPC client from {sibling}')
    module = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(mod_name, None)
        raise
    return module


# ---------------------------------------------------------------------------
# Scope item 3 — did the old slugs actually drain?
# ---------------------------------------------------------------------------

async def verify_old_slugs_drained(memory_service, results, skips) -> None:
    """Count records still carrying each legacy slug this run moved.

    Scope item 3, folded into the script so the next operator gets the answer
    without re-deriving it.  One ``count_memories_by_metadata`` per distinct
    ``(project_id, old slug)`` — bounded by the number of SLUGS moved, not by
    the number of writes.  Eight records sharing one legacy value is one
    question, asked once.

    The arithmetic is adapted from
    ``retro_stamp_topics._probe_legacy_topic_residue`` and makes the number
    mean one thing in both modes.  In a dry run this sweep's own targets still
    carry the legacy value and so are counted, so the rehearsed rows are
    subtracted (clamped at 0 — a record consolidated away mid-run makes the
    count smaller than the rehearsal, and negative residue is nonsense).  In
    an apply run the writes that landed no longer carry it and the count
    already excludes them, while a write that FAILED still carries it and is
    correctly left in.

    **ONE INVERSION FROM THE SOURCE, and it is not a copy bug.**  That sweep
    is id-bounded, so records outside it can legitimately still carry the
    legacy spelling: there, a non-zero residue is expected-and-merely-reported.
    This sweep is corpus-wide, so after ``--apply`` there should be nothing
    left anywhere — a non-zero residue means a slug was STRANDED, and
    ``legacy_slug_residue`` is therefore in :data:`ERROR_OUTCOMES`.

    A probe that raises is filed in the same bucket carrying its error and NO
    count: unknown residue is not zero residue, and swallowing the failure
    would let the report claim a drained corpus on the strength of a question
    that was never answered.

    Mutates *skips* in place; returns nothing.
    """
    pending: dict[tuple[str, str], dict] = {}
    for result in results:
        if result.get('outcome') not in MOVED_OUTCOMES:
            continue
        legacy = result.get('old_topic')
        if not isinstance(legacy, str) or not legacy:
            continue
        entry = pending.setdefault(
            (result['project_id'], legacy),
            {'new_topic': result.get('new_topic'), 'rehearsed': 0},
        )
        if result.get('outcome') in REHEARSED_OUTCOMES:
            entry['rehearsed'] += 1

    bucket = skips.setdefault('legacy_slug_residue', [])
    for (project_id, legacy), entry in sorted(pending.items()):
        try:
            count = int(await memory_service.count_memories_by_metadata(
                project_id, {'topic': legacy},
            ))
        except Exception as exc:
            bucket.append({
                'reason': 'legacy_slug_residue',
                'project_id': project_id,
                'legacy_topic': legacy,
                'new_topic': entry['new_topic'],
                'error': f'{type(exc).__name__}: {exc}',
                'note': (
                    'residue UNKNOWN — the probe failed, which is not the same '
                    'as zero; do not read this run as a drained corpus'
                ),
            })
            continue
        residue = max(0, count - entry['rehearsed'])
        if residue:
            bucket.append({
                'reason': 'legacy_slug_residue',
                'project_id': project_id,
                'legacy_topic': legacy,
                'new_topic': entry['new_topic'],
                'residue_count': residue,
                'note': (
                    'records still carry the legacy slug after a CORPUS-WIDE '
                    'sweep, so the claim is split across two topic values — '
                    'check the under_enumerated / scroll_budget_exhausted '
                    'buckets and the failed writes before re-running'
                ),
            })


# ---------------------------------------------------------------------------
# The sweep
# ---------------------------------------------------------------------------

#: Distinct NON-CONFORMING ``topic`` values across both corpora at the
#: baseline measurement — dark_factory 49 of 105 distinct, reify 54 of 279.
#:
#: PROVENANCE: memory ad707e72, measured 2026-08-04.  Committed as a constant,
#: with its date, precisely because the live number moves on its own: PRD leaf
#: alpha measured 98 and ad707e72 measured 103 while ``memory_metadata.enforce``
#: sat in warn mode.  A dated measurement is something a reader can evaluate;
#: a live count pinned in a test is a scheduled failure.
#:
#: Re-measure with the standing instrument, never by re-deriving here — see
#: :data:`REMEASURE_COMMAND`.
BASELINE_DISTINCT_NON_CONFORMING = 103

#: The baseline's own history, carried into the report so a reader sees the
#: DIRECTION of travel rather than a bare number to compare against.
BASELINE_HISTORY = (
    'PRD leaf alpha measured 98 distinct non-conforming topic values; memory '
    'ad707e72 measured 103 on 2026-08-04 — the population GREW by 5 while '
    'memory_metadata.enforce sat in warn mode, which is why this baseline is '
    'committed with a date rather than re-derived, and why no test pins a '
    'live count.'
)

#: Where the authoritative re-measurement lives.  ``census_memory_metadata.py``
#: already emits ``slug_non_conforming`` as gate 3626's named standing
#: re-measurement, so this script CITES it rather than growing a second,
#: divergent definition of "non-conforming" (INV-5).
REMEASURE_COMMAND = (
    'uv run --project fused-memory python fused-memory/scripts/'
    'census_memory_metadata.py  # then read '
    'coverage.topic_coverage.slug_non_conforming'
)

#: Both live corpora.  ``--project`` REPLACES this list; see
#: :func:`resolve_projects` for why that matters.
DEFAULT_PROJECTS: tuple[str, ...] = ('dark_factory', 'reify')

#: The task stores the gate census reads when no ``--project-root`` is given:
#: this checkout's own root, which the server normalizes to its main
#: checkout.  ``--project-root`` REPLACES it, as ``--project`` replaces
#: :data:`DEFAULT_PROJECTS` — so a sweep of reify must name reify's checkout,
#: and ``run`` refuses ``--apply`` if it does not.
DEFAULT_GATE_ROOTS: tuple[str, ...] = (str(_REPO_ROOT),)


def _describe_gap(gap: dict) -> str:
    return ', '.join(
        f'{key}={gap[key]!r}'
        for key in ('project_id', 'project_root', 'error')
        if gap.get(key)
    )


class GateCensusIncomplete(RuntimeError):
    """``--apply`` refused: the gate census cannot vouch for every swept project.

    Carries the gap entries, the same structured facts a dry run files under
    ``gate_census_incomplete``.
    """

    def __init__(self, gaps: list[dict]):
        super().__init__(
            'gate census incomplete: ' + '; '.join(_describe_gap(gap) for gap in gaps))
        self.gaps = gaps


def _census_gaps(gates: GateCensus, projects) -> list[dict]:
    """Every reason this sweep cannot trust its gate pairing, as report entries.

    A swept project the census does not cover, plus every root it could not
    read.  A failure is a gap even when every swept project is covered: the
    root that failed may be the one store holding a gate this run would
    otherwise treat as absent.
    """
    uncovered = [
        {
            'reason': 'gate_census_incomplete',
            'project_id': project_id,
            'error': (
                'no task store answered for this project, so its consolidation '
                'gates are unknown'
            ),
        }
        for project_id in projects
        if project_id not in gates.covered_projects
    ]
    return uncovered + [dict(failure) for failure in gates.failures]


async def run(
    memory_service,
    *,
    gates: GateCensus,
    projects: tuple[str, ...] = DEFAULT_PROJECTS,
    apply: bool = False,
    client: Any = None,
    page_size: int = DEFAULT_PAGE_SIZE,
    max_pages: int = DEFAULT_MAX_PAGES,
    topics: frozenset[str] | None = None,
) -> dict:
    """Enumerate, plan, pair, write, verify — and report all of it.

    Sequential awaits throughout.  The target population is bounded in the low
    hundreds (~141 records at the ad707e72 baseline), so concurrency would buy
    little, and it would break the one ordering that matters: the collision
    check is decided against a per-project index built from a COMPLETED scroll,
    and two concurrent renames onto the same target slug could each observe an
    unoccupied namespace and both write.

    Args:
        memory_service: Injected; serves both the scroll (as ``.mem0``) and
            the writes, so a run cannot read one store and write another.
        projects: Which corpora to sweep.
        apply: ``False`` (default) rehearses every read and decision and
            withholds only the writes.
        client: The MCP client for the gate half.  ``None`` is fine for a
            sweep that reaches no gate; a GATED group with no client is
            refused exactly like a refused gate.
        gates: The consolidation-gate census, REQUIRED: an optional gate list
            whose absence read as "no gates" is how the live path once renamed
            gated slugs without their gates.  ``--apply`` refuses
            (:class:`GateCensusIncomplete`) unless it covers every swept
            project with no failure; a dry run files the gap under
            ``gate_census_incomplete`` instead.  Only the swept projects'
            gates are paired — another project's gate is neither this run's
            work nor its orphan.
        topics: The legacy values to migrate.  ``None`` (default) migrates
            every non-conforming value.  A selection narrows the renames, the
            plan-time refusals and the gate pairing to these values, so an
            unselected gate is neither paired nor an orphan.  Enumeration,
            coverage, the collision index and ``baseline_delta`` stay
            corpus-wide, so a selected value contested by an unselected
            spelling is still refused.  An invalid selection raises
            ``ValueError`` before anything is read.

    Returns:
        The report dict — rendered by :func:`render_json` /
        :func:`render_markdown` and graded by :func:`resolve_exit_code`.
    """
    refuse_invalid_selection(topics)
    gaps = _census_gaps(gates, projects)

    # Fail-CLOSED capability preflight: ONE probe per run, ABOVE the scroll.
    #
    # ``run`` is the choke point precisely because
    # ``topic_slug_lockstep.rename_one``'s own ``apply`` gate is PER RECORD:
    # probing there would run the check once per target and -- since
    # ``StoreMutationUnavailable`` subclasses ``RuntimeError`` -- be swallowed
    # by the per-record ``except Exception`` around the write, downgrading a
    # run-wide environment denial into N error rows inside a report that
    # otherwise reads as a completed sweep.
    #
    # Emitted through the logger, never ``print``: stdout is reserved for the
    # machine-read artifact rendered at the end of ``main``.
    if apply:
        assert_store_mutation_allowed(
            operation='normalize_topic_slugs --apply',
            logger=logger,
            extra_detail=(
                'each rename would patch a record and then fail to journal the '
                'change, leaving the corpus HALF-RENAMED: one claim split across '
                'two topic values, which is strictly worse for an exact-match '
                "get_memories_by_metadata({'topic': T}) read -- the way "
                'consolidation clusters and the canonical-uniqueness probe '
                'address a topic -- than either uniform state. Nothing was '
                'scrolled and no record was renamed.'
            ),
        )
        # The same shape and placement as the preflight: one decision per
        # run, above the scroll, through the logger.
        if gaps:
            logger.error(
                'normalize_topic_slugs: --apply NOT started (fail-closed) -- '
                'the consolidation-gate census cannot vouch for every swept '
                'project: %s. Renaming a gated slug without moving its gate '
                "leaves the gate's closure check scrolling a slug no record "
                'carries, so the gate could never close. Nothing was scrolled '
                'and no record was renamed. Pass --project-root <that '
                "project's checkout> for every swept project, and resolve any "
                'root that failed. To see the gap safely, re-run without '
                '--apply.',
                '; '.join(_describe_gap(gap) for gap in gaps),
            )
            raise GateCensusIncomplete(gaps)

    # Pre-seeded, because an ABSENT bucket reads as "nothing was skipped",
    # which is a different claim from "we looked and found nothing".
    skips: dict[str, list[dict]] = {bucket: [] for bucket in SKIP_BUCKETS}

    def _record_skips(entries: list[dict]) -> None:
        """File each entry under its own reason, inventing a bucket if new.

        An unrecognised reason is appended to a bucket created on the spot
        rather than dropped: a reason this function has not heard of is
        exactly the kind of thing that must not vanish between the planner
        that raised it and the artifact.
        """
        for entry in entries:
            skips.setdefault(entry.get('reason', 'unclassified'), []).append(entry)

    _record_skips(gaps)

    coverage: dict[str, dict] = {}
    coverage_complete = True
    all_renames: list[Rename] = []
    candidate_count = 0
    # Distinct non-conforming VALUES per project -- the census's own item-3
    # partition (``_build_topic_coverage``), not a record count.  A legacy
    # slug on eight records is ONE non-conforming value; counting records
    # would be measured against a baseline of distinct values and would
    # manufacture a growth that never happened.
    measured_by_project: dict[str, int] = {}

    for project_id in projects:
        enumerated = await enumerate_topic_bearing(
            memory_service, project_id, page_size=page_size, max_pages=max_pages,
        )
        coverage[project_id] = enumerated['coverage']
        coverage_complete = coverage_complete and enumerated['complete']
        candidate_count += len(enumerated['records'])
        # Keyed by ``str(topic)``, as census_memory_metadata.py::CategoryCensus.add
        # keys topic values: the count stays comparable to the census, and an
        # unhashable list or dict topic cannot raise.
        measured_by_project[project_id] = len({
            str((r.get('metadata') or {}).get('topic')) for r in enumerated['records']
        })
        _record_skips(enumerated['skips'])

        renames, plan_skips = plan_renames(
            enumerated['records'],
            project_id=project_id,
            occupied_topics=enumerated['occupied_topics'],
            canonical_ids_by_topic=enumerated['canonical_ids_by_topic'],
            topics=topics,
        )
        _record_skips(plan_skips)
        all_renames.extend(renames)

    groups, gate_skips = pair_gate_blocks(
        all_renames,
        [
            task for task in gates.tasks
            if task.get('project_id') in projects
            and in_selection(gate_topic(task), topics)
        ],
    )
    _record_skips(gate_skips)

    results: list[dict] = []
    gate_results: list[dict] = []
    for group in groups:
        group_outcome = await rename_group(memory_service, group, apply=apply, client=client)
        results.extend(group_outcome.results)
        if group_outcome.gate_row is not None:
            gate_results.append(group_outcome.gate_row)
        if group_outcome.held_entry is not None:
            _record_skips([group_outcome.held_entry])

    # Scope item 3, in the same run that did the writing.
    await verify_old_slugs_drained(memory_service, results, skips)

    outcomes: dict[str, int] = {}
    renamed_by_topic: dict[str, int] = {}
    would_rename_by_topic: dict[str, int] = {}
    for row in [*results, *gate_results]:
        outcome = str(row.get('outcome'))
        if outcome in ERROR_OUTCOMES:
            skips.setdefault(outcome, []).append(row)
            continue
        outcomes[outcome] = outcomes.get(outcome, 0) + 1
        if outcome == 'renamed':
            topic = str(row.get('new_topic'))
            renamed_by_topic[topic] = renamed_by_topic.get(topic, 0) + 1
        elif outcome == 'would_rename':
            topic = str(row.get('new_topic'))
            would_rename_by_topic[topic] = would_rename_by_topic.get(topic, 0) + 1

    # The buckets are the ONLY source of error counts, so each failure is
    # counted exactly once and every error outcome equals its bucket's length.
    # That also grades a refusal no result row carries: a collision is refused
    # at plan time, and grading result rows alone would exit 0 on a run that
    # declined to migrate the most dangerous slug it found.
    for bucket, entries in skips.items():
        if entries and bucket in ERROR_OUTCOMES:
            outcomes[bucket] = len(entries)

    return {
        'apply': apply,
        # The inverse of the sibling's claim, stated where a reader can check
        # it: `retro_stamp_topics` reports `bounded: True` and means it.
        'bounded': False,
        'scope': (
            'corpus-wide — every record in every census category of every '
            'listed project is enumerated by scroll, not addressed by id'
        ),
        'projects': list(projects),
        'topic_selection': None if topics is None else sorted(topics),
        'coverage': coverage,
        'coverage_complete': coverage_complete,
        'candidate_count': candidate_count,
        'rename_count': len(all_renames),
        'group_count': len(groups),
        'gate_count': sum(1 for g in groups if g.gate_task_id is not None),
        'renamed_total': sum(renamed_by_topic.values()),
        'renamed_by_topic': renamed_by_topic,
        'would_rename_total': sum(would_rename_by_topic.values()),
        'would_rename_by_topic': would_rename_by_topic,
        'outcomes': outcomes,
        'results': results,
        'gate_results': gate_results,
        # Sorted LISTS: a frozenset through render_json's default=str would
        # render in hash order and break the byte-comparable artifact.
        'gate_census': {
            'roots': sorted(gates.roots),
            'covered_projects': sorted(gates.covered_projects),
            'gate_count': len(gates.tasks),
            'complete': not gaps,
        },
        'skips': skips,
        # Scope item 4: the delta, computed here so the next reader inherits
        # it instead of re-deriving it (and re-deriving it differently).
        'baseline_delta': {
            'baseline_distinct_non_conforming': BASELINE_DISTINCT_NON_CONFORMING,
            'baseline_source': (
                'memory ad707e72, measured 2026-08-04 (dark_factory 49 of 105 '
                'distinct topic values, reify 54 of 279)'
            ),
            'baseline_history': BASELINE_HISTORY,
            'measured_distinct_non_conforming': sum(measured_by_project.values()),
            'measured_by_project': measured_by_project,
            'delta': (
                sum(measured_by_project.values())
                - BASELINE_DISTINCT_NON_CONFORMING
            ),
            'remeasure_command': REMEASURE_COMMAND,
            'note': (
                'measured is a distinct-VALUE count over the projects THIS run '
                'swept, matching the census partition; it is comparable to the '
                'baseline only when both corpora were swept and coverage is '
                'complete'
            ),
        },
    }


# ---------------------------------------------------------------------------
# Report rendering, exit code, CLI
# ---------------------------------------------------------------------------

#: Default artifact paths, beside the sibling sweep's and the census's.
#:
#: Deliberately NOT committed: this report snapshots live corpus state that
#: rots immediately, and a committed copy invites trusting a stale count.  The
#: committed number in this migration is :data:`BASELINE_DISTINCT_NON_CONFORMING`
#: — one measurement, dated and attributed — which is what the report diffs
#: against.
DEFAULT_JSON_OUT = str(_REPO_ROOT / 'plans' / 'topic-slug-normalization-report.json')
DEFAULT_MD_OUT = str(_REPO_ROOT / 'plans' / 'topic-slug-normalization-report.md')

#: Keys a skip line leaves out.  The bucket heading already names the reason
#: or outcome, and ``response`` is
#: :func:`topic_slug_lockstep.rename_one`'s whole write envelope, which the
#: JSON keeps.
_SKIP_LINE_OMITTED_KEYS: frozenset[str] = frozenset({'reason', 'outcome', 'response'})


def render_json(report: dict) -> str:
    """The machine-readable artifact.

    ``sort_keys=True`` so two dry runs over an unchanged corpus render
    byte-comparably and a real change is the only thing that shows in a diff.
    ``default=str`` because a report can carry a datetime that wandered in off
    a scroll row, and a renderer that raised on one would lose the whole run's
    evidence over a formatting detail.
    """
    return json.dumps(report, indent=2, default=str, sort_keys=True)


def _render_skip_entry(entry: dict) -> str:
    # The producer's dict is the single point of truth for which fields an
    # entry carries, in reading order.  A drift in the omitted set can only
    # SHOW a field, never hide one.
    parts = [
        f'{key}={value!r}'
        for key, value in entry.items()
        if key not in _SKIP_LINE_OMITTED_KEYS and value is not None
    ]
    # An entry the renderer does not recognise is exactly the one worth
    # showing verbatim, rather than as a bare bullet.
    return '- ' + (', '.join(parts) if parts else repr(entry))


def _render_gate_census(census: dict) -> str:
    verdict = (
        'complete' if census.get('complete')
        else 'INCOMPLETE — see the gate_census_incomplete bucket; --apply '
             'refuses until every swept project is covered'
    )
    return (
        f'**Gate census:** {verdict} '
        f'(roots: {", ".join(census.get("roots") or []) or "none"}; '
        f'covered: {", ".join(census.get("covered_projects") or []) or "none"}; '
        f'gates read: {census.get("gate_count", 0)})'
    )


def _render_topic_selection(selection: list[str] | None) -> str:
    if selection is None:
        return '**Topic selection:** every non-conforming value'
    return (
        f'**Topic selection:** {", ".join(f"`{topic}`" for topic in selection)} '
        '(renames, refusals and gate pairing are narrowed to these; enumeration '
        'and the collision check stay corpus-wide)'
    )


def render_markdown(report: dict) -> str:
    """The human-readable artifact.

    Every EMPTY bucket is stated as an explicit ``: 0`` heading rather than
    omitted.  That is this renderer's most load-bearing rule: an artifact that
    silently drops empty buckets reads identically whether the sweep skipped
    nothing or never computed the bucket at all — and the second case is a
    hole in coverage that, being invisible, nobody ever closes.
    """
    mode = 'APPLY' if report.get('apply') else 'DRY RUN'
    lines: list[str] = [
        '# Topic-slug normalization — corpus-wide migration (task 4878)',
        '',
        f'**Mode:** {mode}',
        f'**Projects:** {", ".join(report.get("projects") or [])}',
        _render_topic_selection(report.get('topic_selection')),
        (
            '**Scope:** corpus-wide — every record in every census category '
            'of every listed project is enumerated by scroll, not addressed '
            'by id. (The sibling `retro_stamp_topics.py` reports '
            '`bounded: True` and means it; this one does not.)'
        ),
        (
            '**Coverage:** '
            + ('complete' if report.get('coverage_complete', False)
               else 'INCOMPLETE — see the under_enumerated / '
                    'scroll_budget_exhausted buckets; this run is a LOWER BOUND')
        ),
        '',
        '## Coverage',
        '',
        '| project | category | expected | scrolled | recount | complete |',
        '| --- | --- | --- | --- | --- | --- |',
    ]
    coverage = report.get('coverage') or {}
    if not coverage:
        lines.append('| _none_ | | | | | |')
    for project_id, cells in sorted(coverage.items()):
        for category, cell in sorted(cells.items()):
            lines.append(
                f'| {project_id} | {category} | {cell.get("expected")} | '
                f'{cell.get("scrolled")} | {cell.get("recount")} | '
                f'{cell.get("complete")} |'
            )

    renamed = report.get('renamed_by_topic') or {}
    would = report.get('would_rename_by_topic') or {}
    lines += [
        '',
        '## Renamed by new slug',
        '',
        '| new topic | renamed | would rename |',
        '| --- | --- | --- |',
    ]
    for topic in sorted(set(renamed) | set(would)):
        lines.append(f'| {topic} | {renamed.get(topic, 0)} | {would.get(topic, 0)} |')
    if not renamed and not would:
        lines.append('| _none_ | 0 | 0 |')
    lines += [
        '',
        f'**Non-conforming records enumerated:** {report.get("candidate_count", 0)}  ',
        f'**Renames planned:** {report.get("rename_count", 0)}  ',
        f'**Slug groups:** {report.get("group_count", 0)} '
        f'(gate-backed: {report.get("gate_count", 0)})  ',
        f'**Total renamed:** {report.get("renamed_total", 0)}  ',
        f'**Total would-rename:** {report.get("would_rename_total", 0)}',
        '',
        '## Outcomes',
        '',
    ]
    outcomes = report.get('outcomes') or {}
    if outcomes:
        lines.extend(f'- `{name}`: {count}' for name, count in sorted(outcomes.items()))
    else:
        lines.append('- _nothing to rename_')

    gate_results = report.get('gate_results') or []
    gate_census = report.get('gate_census')
    lines += ['', '## Consolidation gates', '']
    if gate_census:
        lines += [_render_gate_census(gate_census), '']
    if gate_results:
        for row in gate_results:
            lines.append(
                f'- task {row.get("gate_task_id")}: `{row.get("old_topic")}` -> '
                f'`{row.get("new_topic")}` — {row.get("outcome")}'
                + (f' ({row.get("error")})' if row.get('error') else '')
            )
    elif gate_census and not gate_census.get('complete'):
        lines.append('_gate-backed topics in an uncovered project are UNKNOWN, not absent_')
    else:
        lines.append('_no gate-backed topic in this sweep_')

    baseline = report.get('baseline_delta') or {}
    if baseline:
        lines += [
            '',
            '## Baseline delta (scope item 4)',
            '',
            f'**Baseline (distinct non-conforming topic values):** '
            f'{baseline.get("baseline_distinct_non_conforming")}  ',
            f'**Baseline source:** {baseline.get("baseline_source")}  ',
            f'**Measured this run:** '
            f'{baseline.get("measured_distinct_non_conforming")} '
            f'({", ".join(f"{p}={n}" for p, n in sorted((baseline.get("measured_by_project") or {}).items()))})  ',
            f'**Delta:** {baseline.get("delta")}  ',
            '',
            f'History: {baseline.get("baseline_history")}',
            '',
            f'Re-measure with the standing instrument: `'
            f'{baseline.get("remeasure_command")}`',
        ]

    lines += ['', '## Skips', '']
    skips = report.get('skips') or {}
    # Iterate the CANONICAL bucket order first, so a bucket `run` forgot to
    # emit shows up as a 0 line rather than as an absence, then any extra
    # bucket a planner invented at runtime.
    ordered = list(SKIP_BUCKETS) + sorted(set(skips) - set(SKIP_BUCKETS))
    for bucket in ordered:
        entries = skips.get(bucket) or []
        lines += ['', f'### {bucket}: {len(entries)}', '']
        if not entries:
            lines.append('_none_')
            continue
        lines.extend(_render_skip_entry(entry) for entry in entries)
    return '\n'.join(lines) + '\n'


def resolve_exit_code(report: dict) -> int:
    """0 on a clean run, 1 when anything did not land as planned.

    Graded off :data:`ERROR_OUTCOMES` — the SAME set the writers and the
    report use — so the exit code and the artifact can never disagree about
    whether a run was clean.

    A refusal counts.  This sweep's most valuable outcomes are the ones where
    it declined to act (a collision, an unfoldable value, a held gate pair),
    and every one of them is unfinished work an operator has to resolve by
    hand.  Exiting 0 on those would make the refusals invisible to anything
    that reads only the exit status.
    """
    outcomes = report.get('outcomes') or {}
    return 1 if any(outcomes.get(name) for name in ERROR_OUTCOMES) else 0


def resolve_projects(args) -> tuple[str, ...]:
    """``--project`` REPLACES the default list; it does not extend it.

    ``action='append'`` appends to whatever default argparse holds, so a plain
    ``default=DEFAULT_PROJECTS`` would turn ``--project reify`` into both
    corpora — silently widening a corpus-wide mutation an operator had
    deliberately narrowed.  Hence ``default=None`` here and the fallback in
    one named place.
    """
    return tuple(args.projects) if args.projects else DEFAULT_PROJECTS


def resolve_gate_roots(args) -> tuple[str, ...]:
    """``--project-root`` REPLACES :data:`DEFAULT_GATE_ROOTS`, deduplicated in order.

    The rule :func:`resolve_projects` follows, for the same reason: naming
    the task stores to read narrows the census and never silently widens it.
    """
    if not args.project_roots:
        return DEFAULT_GATE_ROOTS
    return tuple(dict.fromkeys(args.project_roots))


def resolve_topics(args) -> frozenset[str] | None:
    """``--topic`` REPLACES "every value"; absent, it is ``None``, never empty.

    The rule :func:`resolve_projects` follows: naming a legacy value narrows
    the write set and never widens it.  :func:`run` validates the values.
    """
    return None if args.topics is None else frozenset(args.topics)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            'Corpus-wide metadata.topic slug normalization (task 4878). '
            'Dry run by default.'
        ),
    )
    parser.add_argument(
        '--apply', action='store_true',
        help='Commit the renames. Without it the run is a full rehearsal that '
             'reads and decides everything but writes nothing.',
    )
    parser.add_argument(
        '--project', dest='projects', action='append', default=None,
        help=f'Project to sweep; repeatable. Replaces the default '
             f'({", ".join(DEFAULT_PROJECTS)}) rather than extending it.',
    )
    parser.add_argument(
        '--project-root', dest='project_roots', action='append', default=None,
        help=f'Checkout whose task store the consolidation-gate census reads; '
             f'repeatable, one per swept project. Replaces the default '
             f'({", ".join(DEFAULT_GATE_ROOTS)}) rather than extending it.',
    )
    parser.add_argument(
        '--topic', dest='topics', action='append', default=None,
        help='Legacy (non-conforming) topic value to migrate; repeatable. '
             'Replaces the default (every non-conforming value) rather than '
             'extending it. Name the SOURCE value, never its target slug.',
    )
    parser.add_argument(
        '--server-url', default=None,
        help='Fused-memory MCP server URL for the gate census and the gate '
             'patches (default: the JSON-RPC client module\'s DEFAULT_SERVER).',
    )
    parser.add_argument(
        '--json-out', dest='json_out', default=DEFAULT_JSON_OUT,
        help=f'Machine-readable report path (default: {DEFAULT_JSON_OUT}).',
    )
    parser.add_argument(
        '--md-out', dest='md_out', default=DEFAULT_MD_OUT,
        help=f'Human-readable report path (default: {DEFAULT_MD_OUT}).',
    )
    parser.add_argument(
        '--config', default=None,
        help='Path to a fused-memory config file (sets CONFIG_PATH before loading).',
    )
    return parser


@contextlib.asynccontextmanager
async def open_live_memory_service():
    """The production ``MemoryService``, initialized, and closed on the way out.

    Imports deferred so importing this module — which the tests do, by path —
    never constructs a backend.
    """
    from fused_memory.config.schema import FusedMemoryConfig
    from fused_memory.maintenance._utils import maintenance_service

    async with maintenance_service(config=FusedMemoryConfig()) as (_, memory):
        yield memory


def open_mcp_client(server_url: str | None) -> Any:
    """The production task-store client, at *server_url* or the default.

    ``FusedMemoryClient`` is itself an async context manager whose
    ``__aenter__`` performs the initialize handshake, so it is returned as-is.
    The default URL is the loaded client module's own ``DEFAULT_SERVER``.
    """
    url = load_mcp_client_module().DEFAULT_SERVER if server_url is None else server_url
    return load_mcp_client_class()(url)


async def run_live(args, *, open_memory_service, open_client) -> dict:
    """Census the task stores, then sweep — the composition ``main`` runs.

    The census comes FIRST, so an unreachable task service fails before any
    store is touched.  A server that cannot even be opened fails the run in
    both modes: that is an outage of the factory's own task service, not a
    per-project gap, so it is not downgraded into a gap entry.  The one
    client serves the census and the gate patches, so a run cannot pair
    against one task store and patch another.
    """
    async with open_client(args.server_url) as client:
        census = await census_consolidation_gates(client, resolve_gate_roots(args))
        async with open_memory_service() as memory:
            return await run(
                memory,
                projects=resolve_projects(args),
                apply=args.apply,
                client=client,
                gates=census,
                topics=resolve_topics(args),
            )


def main(
    argv: list[str] | None = None,
    *,
    open_memory_service=open_live_memory_service,
    open_client=open_mcp_client,
) -> int:
    """Census, sweep, write both artifacts, exit graded.

    The two I/O openers are parameters, defaulting to the production ones, so
    this composition root is testable through its own public surface.
    """
    # This script is otherwise print-based, so without this the module logger
    # carrying the fail-closed refusal and the coverage warnings would have no
    # handler and would reach the operator only through ``logging.lastResort``
    # -- a bare line with no timestamp, level or logger name. ``stream`` is
    # named explicitly because stdout is reserved for the machine-read report.
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)s %(message)s',
        stream=sys.stderr,
    )

    args = _build_parser().parse_args(argv)

    if args.config:
        import os

        os.environ['CONFIG_PATH'] = str(args.config)

    report = asyncio.run(run_live(
        args, open_memory_service=open_memory_service, open_client=open_client))

    Path(args.json_out).write_text(render_json(report), encoding='utf-8')
    Path(args.md_out).write_text(render_markdown(report), encoding='utf-8')

    print(render_markdown(report))
    print(f'wrote {args.json_out}')
    print(f'wrote {args.md_out}')
    if not args.apply:
        print('DRY RUN — nothing was modified. Re-run with --apply to commit.')
    return resolve_exit_code(report)


if __name__ == '__main__':
    sys.exit(main())
