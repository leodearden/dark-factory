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
has one home (INV-5) and both scripts import it, so the two can never disagree
about what a legacy value becomes.

The migration spans TWO stores
------------------------------
Most of the work patches Mem0 record metadata.  But
``metadata.x_recon_consolidation_gate.topic`` is a Tier-C block on **task**
metadata — read by ``reconciliation/consolidation_gate.py`` and enforced on
the ``done`` transition by ``middleware/task_interceptor.py`` — a genuinely
different store, reached through the MCP server rather than a backend call.

Renaming the memories without moving that block would leave a gate task's
closure check scrolling a slug no record carries any more: the uncloseable-gate
trap re-created at a NEW address, which is strictly worse than leaving the
snake_case slug alone because the operator would then also have to discover
that this migration caused it.  So the two halves are planned as a pair and
refused as a pair.

Dry run is the default — the operator runbook
--------------------------------------------
1. **Dry run**, from anywhere::

       uv run --project fused-memory python \
           fused-memory/scripts/normalize_topic_slugs.py

   Writes ``plans/topic-slug-normalization-report.{json,md}`` and prints the
   markdown.  Nothing is modified.

2. **Read the refusal buckets and resolve them BY HAND.**  ``slug_collision``,
   ``canonical_collision`` and ``topic_unfoldable`` are the buckets this
   script deliberately will not decide for you: the first two would merge two
   topic namespaces (and, in the canonical case, manufacture the second
   canonical ``_check_canonical_uniqueness`` exists to prevent), and the third
   has no honest fold, so any value the script picked would be a guess written
   across the whole corpus.  Also check ``under_enumerated`` /
   ``scroll_budget_exhausted``: a run with either populated is a LOWER BOUND,
   and a clean residue probe over it proves nothing.

3. **Apply**, as an OPERATOR::

       ... normalize_topic_slugs.py --apply

   This is an operator action, not an agent one.  The store-mutation preflight
   fails closed for any process that cannot write mem0's history directory,
   which is the normal posture inside an agent sandbox — so a dry run is what
   an agent can deliver, and the deliverable of task 4878 is THIS INSTRUMENT
   PLUS A VERIFIED DRY RUN, not the live mutation.

4. **Re-measure** with the standing instrument and compare against the
   committed baseline::

       ... census_memory_metadata.py   # read coverage.topic_coverage.slug_non_conforming

   The report's ``baseline_delta`` block already carries the arithmetic; step 4
   is how you confirm it against the corpus rather than against this run's own
   enumeration.

Scope boundary
--------------
This script is the DATA migration and nothing else.  Flipping
``memory_metadata.enforce`` from warn to reject belongs to task 3626, and
changing the writer instructions so new records are born conforming belongs to
task 3202.  Normalizing the existing corpus is worth doing regardless of
whether either ever lands: it is what makes an exact-match ``{'topic': T}``
read return the whole cluster instead of whichever spelling the caller
guessed.
"""
from __future__ import annotations

import argparse
import asyncio
import functools
import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from fused_memory.backends.mem0_client import (
    DEFAULT_SCROLL_MAX_PAGES,
    ScrollPageBudgetExhausted,
)
from fused_memory.reconciliation.consolidation_gate import GATE_METADATA_KEY
from fused_memory.topic_slug import derive_topic_slug, is_valid_topic_slug
from fused_memory.utils.store_mutation_preflight import (
    StoreMutationUnavailable,
    assert_store_mutation_allowed,
)

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
    'DEFAULT_JSON_OUT',
    'DEFAULT_MAX_PAGES',
    'DEFAULT_MD_OUT',
    'DEFAULT_PROJECTS',
    'GATE_METADATA_KEY',
    'GateGroup',
    'WriteRejectedError',
    'assert_write_accepted',
    'load_mcp_client_class',
    'pair_gate_blocks',
    'rename_group',
    'WRITE_REASON',
    'WRITE_SOURCE',
    'DEFAULT_PAGE_SIZE',
    'DEFAULT_SCROLL_MAX_PAGES',
    'ERROR_OUTCOMES',
    'Rename',
    'SKIP_BUCKETS',
    'ScrollPageBudgetExhausted',
    'StoreMutationUnavailable',
    'assert_store_mutation_allowed',
    'census_categories',
    'derive_topic_slug',
    'enumerate_topic_bearing',
    'is_valid_topic_slug',
    'load_census_module',
    'main',
    'plan_renames',
    'REMEASURE_COMMAND',
    'render_json',
    'render_markdown',
    'resolve_exit_code',
    'resolve_projects',
    'run',
    'rename_one',
    'verify_old_slugs_drained',
]


#: Outcomes that mean the corpus did not get what the plan intended.  Named
#: once, here, so :func:`resolve_exit_code` and the report agree on what
#: "clean" means instead of each keeping its own list.
ERROR_OUTCOMES: frozenset[str] = frozenset({
    'slug_collision',
    'canonical_collision',
    'under_enumerated',
    'scroll_budget_exhausted',
    'memory_not_found',
    'topic_moved_since_plan',
    'update_failed',
    'rename_error',
    'gate_lockstep_failed',
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
    # from pair_gate_blocks / rename_group — the two-store lockstep
    'orphan_gate_topic',
    'gate_lockstep_failed',
    # from verify_old_slugs_drained
    'legacy_slug_residue',
)

#: Written to every ``update_memory`` so the write journal attributes each
#: rename to this sweep rather than to a generic ``mcp_tool``.  The amendment
#: storm alarm reads this field; a bulk run under the default source would
#: look exactly like the runaway rewrite that alarm exists to catch.
WRITE_SOURCE = 'normalize_topic_slugs'

#: Recorded on the write journal row beside the patch.
WRITE_REASON = 'corpus-wide topic-slug normalization (task 4878)'

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

@functools.cache
def load_census_module() -> Any:
    """Load ``census_memory_metadata`` by path.

    ``scripts/`` is not a package, so a plain import cannot reach it.  Same
    importlib idiom — including the ``sys.modules``-first lookup — that
    ``census_memory_metadata.py::_load_probe_module`` and
    ``retro_stamp_topics.py::_load_probe_module`` already established for
    exactly this cross-script reuse: that slot may already hold a module
    another by-path loader executed, and re-executing would hand back
    DIFFERENT class objects, silently breaking identity across the seam.

    Memoized because the load is not free and because the identity above is
    only stable if it happens once.
    """
    import importlib.util  # noqa: PLC0415

    mod_name = 'census_memory_metadata'
    cached = sys.modules.get(mod_name)
    if cached is not None:
        return cached
    spec = importlib.util.spec_from_file_location(mod_name, _CENSUS_SCRIPT_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f'Cannot load {_CENSUS_SCRIPT_PATH}')
    module = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(mod_name, None)
        raise
    return module


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
    the ids.

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
    from fused_memory.models.scope import Scope  # noqa: PLC0415

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
                if not isinstance(topic, str) or not topic:
                    continue
                occupied_topics.add(topic)
                # ``is True``, not truthiness: metadata off a live store is
                # not schema-enforced here, and ``canonical: 'false'`` would
                # otherwise read as a canonical claim and escalate a mild
                # collision into the severe bucket.
                if metadata.get('canonical') is True:
                    memory_id = record.get('id')
                    if isinstance(memory_id, str) and memory_id:
                        canonical_ids_by_topic.setdefault(topic, []).append(memory_id)
                if is_valid_topic_slug(topic):
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
                            if metadata.get('canonical') is True else {}
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


# ---------------------------------------------------------------------------
# The plan row
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Rename:
    """One record's topic move, decided before anything is written.

    Frozen because the artifact and the writes must agree: the whole reason
    the corpus is planned before it is touched is so an operator can read the
    rehearsal and know what ``--apply`` will do.  A mutable row lets the two
    diverge between the report and the write loop.

    Attributes:
        project_id: The corpus this record lives in.  Carried on the row
            rather than inferred at the write, because collisions are scoped
            per project and the report interleaves both defaults.
        memory_id: The Mem0 record id — how :func:`rename_one` addresses it.
        old_topic: The raw non-conforming value, verbatim.  Needed to detect
            a record whose topic moved between the plan and the write, and to
            probe the legacy slug for residue afterwards.
        new_topic: The folded, validated slug.  Always a value
            :func:`is_valid_topic_slug` accepts — the planner never emits a
            row it could not stand behind.
    """

    project_id: str
    memory_id: str
    old_topic: str
    new_topic: str


# ---------------------------------------------------------------------------
# Pure core — planning
# ---------------------------------------------------------------------------

def plan_renames(
    records,
    *,
    project_id: str,
    occupied_topics=None,
    canonical_ids_by_topic=None,
) -> tuple[list[Rename], list[dict]]:
    """Decide what to rename in one project's records, and what to refuse.

    Pure: takes scroll-shaped ``{'id', 'created_at', 'metadata'}`` dicts and no
    service, so every verdict is reproducible from the input alone.

    Takes the project's FULL record set rather than one record at a time.  It
    has to: deciding whether ``foo_bar`` may become ``foo-bar`` requires
    knowing whether anything already carries ``foo-bar``, which is not a
    property of the record being examined.

    Four dispositions, and the distinctions between them are load-bearing:

    * **rename** — the topic fails :func:`is_valid_topic_slug` and folds
      cleanly.  This is the work.
    * **ignored, silently** — the topic already conforms, or there is no
      ``topic`` key, or it is ``None``.  Deliberately NOT a skip line: most
      of the ~49k-record corpus conforms, and filing each conforming record
      would bury the ~141 that need attention.  A ``None`` topic is an
      *absent* topic, not a malformed one; stamping an absent topic is
      ``retro_stamp_topics.py``'s job, not this script's.
    * **skipped, reported** — no honest fold exists (``topic_unfoldable``), or
      the record has no id to write to (``record_without_id``).  The entry
      carries the RAW value and never a guessed slug: an operator resolving
      it by hand needs to see what is actually on the record, and the reason
      it is here at all is that this script refused to name a replacement.
    * **refused, per SLUG** — the fold lands on an occupied slug
      (``slug_collision``), possibly severely (``canonical_collision``).  See
      the guard below; the refusal covers every record carrying that legacy
      value, and is filed once for the slug rather than once per record.

    Output is sorted by ``(project_id, new_topic, memory_id)`` — and the skips
    correspondingly — because scroll order is not stable, and two rehearsals
    over an unchanged corpus must produce byte-comparable artifacts for a diff
    to mean anything.

    Args:
        records: Scroll-shaped record dicts for ONE project.
        project_id: The corpus they came from; stamped onto every row.

    Returns:
        ``(renames, skips)`` — frozen :class:`Rename` rows, and report-ready
        skip dicts each carrying a ``reason``.
    """
    renames: list[Rename] = []
    skips: list[dict] = []

    # --- pass 1: what does the corpus already look like? -------------------
    # Built BEFORE any verdict, from the whole record set, so no decision can
    # depend on the order the backend happened to page records back in.
    #
    #   observed_topics       every topic value present, conforming or not
    #   canonical_ids_by_topic  which records under a topic claim canonical
    #   sources_by_target     which legacy values want to fold onto a target
    #
    # *occupied_topics* / *canonical_ids_by_topic* carry the incumbents that
    # are NOT in *records*.  ``enumerate_topic_bearing`` discards conforming
    # records to keep a corpus sweep affordable, so on the live path the
    # incumbent a fold would collide with has already been thrown away by the
    # time this function runs — and a collision guard that cannot see the
    # incumbent silently degrades into a namespace merger.  Tests that pass a
    # complete record set (conforming rows included) need neither argument.
    observed_topics: set[str] = set(occupied_topics or ())
    canonical_ids_by_topic = {
        topic: list(ids) for topic, ids in (canonical_ids_by_topic or {}).items()
    }
    sources_by_target: dict[str, set[str]] = {}
    for record in records:
        metadata = record.get('metadata') or {}
        topic = metadata.get('topic')
        if not isinstance(topic, str):
            continue
        observed_topics.add(topic)
        # ``is True``, not truthiness: metadata comes off a live store and is
        # not schema-enforced at this seam, so ``canonical: 'false'`` — a
        # non-empty string — would otherwise read as a canonical claim and
        # escalate a mild refusal into the severe bucket.
        if metadata.get('canonical') is True:
            memory_id = record.get('id')
            if isinstance(memory_id, str) and memory_id and memory_id not in (
                canonical_ids_by_topic.get(topic) or ()
            ):
                canonical_ids_by_topic.setdefault(topic, []).append(memory_id)
        if not is_valid_topic_slug(topic):
            folded = derive_topic_slug(topic)
            if folded is not None:
                sources_by_target.setdefault(folded, set()).add(topic)

    # --- pass 2: classify, then emit --------------------------------------
    # WHY REFUSE RATHER THAN MERGE.  Folding ``foo_bar`` onto ``foo-bar`` when
    # something already carries ``foo-bar`` is not a normalization: it merges
    # two topic namespaces, changing cluster membership for records this sweep
    # never examined.  When both sides hold a ``canonical: true`` it is worse
    # than that — ``topic`` is what SCOPES canonical uniqueness
    # (``memory_service._check_canonical_uniqueness`` probes
    # ``{'topic': T, 'canonical': True}``), so the merged cluster holds exactly
    # the second canonical that invariant exists to prevent.  Under the shipped
    # ``memory_metadata.enforce = false`` that seam WARN-fails open, so it would
    # land silently with this script as its author.
    #
    # WHY THE SERVICE-SIDE PROBE CANNOT SUBSTITUTE FOR THIS ONE.  The seam
    # adjudicates a SINGLE write against live state, one at a time.  It would
    # let the first member of a colliding group through and refuse only the
    # second — leaving the corpus in a state determined by iteration order, and
    # half-merged.  Deciding here, against the whole record set, makes the
    # verdict a property of the CORPUS instead.  Same posture, same reason, as
    # ``retro_stamp_topics.stamp_one``'s ``stray_canonical_on_member``.
    collisions: dict[str, dict] = {}
    for record in records:
        metadata = record.get('metadata') or {}
        if 'topic' not in metadata:
            continue
        topic = metadata['topic']
        if topic is None:
            continue
        if is_valid_topic_slug(topic):
            continue

        memory_id = record.get('id')
        folded = derive_topic_slug(topic)
        if folded is None:
            skips.append({
                'reason': 'topic_unfoldable',
                'project_id': project_id,
                'memory_id': memory_id,
                'topic': topic,
                'note': (
                    'no honest fold exists for this value (it collapses to '
                    'empty, exceeds TOPIC_SLUG_MAX_LEN once folded, or is not '
                    'a string) — resolve it by hand rather than letting the '
                    'sweep invent a slug no human chose'
                ),
            })
            continue

        # Occupied by a record that already carries the target value, or
        # contested by a SECOND legacy value folding onto the same target.
        # The second case is refused for BOTH sources: letting one through
        # would make the outcome depend on scroll order, and would refuse the
        # other for an occupancy this script had just manufactured.
        incumbent = folded in observed_topics
        contested = len(sources_by_target.get(folded, ())) > 1
        if incumbent or contested:
            entry = collisions.get(topic)
            if entry is None:
                old_canonicals = sorted(canonical_ids_by_topic.get(topic, ()))
                new_canonicals = sorted(
                    memory_id
                    for other in (
                        {folded} | (sources_by_target.get(folded, set()) - {topic})
                    )
                    for memory_id in canonical_ids_by_topic.get(other, ())
                )
                severe = bool(old_canonicals) and bool(new_canonicals)
                entry = {
                    'reason': (
                        'canonical_collision' if severe else 'slug_collision'
                    ),
                    'project_id': project_id,
                    'old_topic': topic,
                    'new_topic': folded,
                    'colliding_count': 0,
                    'memory_ids': [],
                    'old_canonical_ids': old_canonicals,
                    'new_canonical_ids': new_canonicals,
                    'note': (
                        'the fold would merge two topic namespaces AND carry a '
                        'canonical from each side into one cluster, '
                        'manufacturing the second canonical '
                        '_check_canonical_uniqueness exists to prevent — '
                        'resolve by hand before re-running'
                        if severe else
                        'the target slug is already occupied (by conforming '
                        'records, or by a second legacy value folding onto it), '
                        'so the rename would MERGE two topic namespaces rather '
                        'than normalize one — resolve by hand before re-running'
                    ),
                }
                collisions[topic] = entry
            entry['colliding_count'] += 1
            if isinstance(memory_id, str) and memory_id:
                entry['memory_ids'].append(memory_id)
            continue

        if not memory_id or not isinstance(memory_id, str):
            # ``update_memory`` is addressed by memory id.  Refusing here keeps
            # an unaddressable row out of the planned-work count instead of
            # letting it fail one at a time at the write boundary, inside a
            # report that already claimed it.
            skips.append({
                'reason': 'record_without_id',
                'project_id': project_id,
                'memory_id': memory_id,
                'topic': topic,
                'new_topic': folded,
                'note': 'record carries no usable id, so it cannot be written to',
            })
            continue

        renames.append(Rename(
            project_id=project_id,
            memory_id=memory_id,
            old_topic=topic,
            new_topic=folded,
        ))

    for entry in collisions.values():
        entry['memory_ids'].sort()
        skips.append(entry)

    renames.sort(key=lambda r: (r.project_id, r.new_topic, r.memory_id))
    skips.sort(key=lambda s: (
        str(s.get('reason')),
        str(s.get('project_id')),
        str(s.get('old_topic', '')),
        str(s.get('memory_id')),
    ))
    return renames, skips


# ---------------------------------------------------------------------------
# The single write boundary
# ---------------------------------------------------------------------------

async def rename_one(memory_service, rename: Rename, *, apply: bool) -> dict:
    """Move one record's topic, or report precisely why it was not moved.

    The only function here that writes.  Order is copied from
    ``retro_stamp_topics.stamp_one`` and is load-bearing:

    1. **live re-read** (``get_memory_by_id``).  The plan's ids come off a
       scroll that may be minutes old on a corpus orchestrators are writing
       to.  Reading first is what turns a record consolidated away since the
       scroll into a report line instead of a Qdrant ``set_payload`` that
       acknowledges a write to nothing.
    2. **decide** against the LIVE value, not the planned one.  A topic that
       moved since the plan makes this row stale; applying anyway would
       overwrite a live value with a fold of a value that no longer exists —
       a data loss the report would score as a success.  A live topic that
       already equals the target ends the call: a no-op ``update_memory``
       would still journal a write op, still count toward the
       content-amendment storm alarm, and would inflate the renamed count
       with records that gained nothing.
    3. **write**, metadata-only, patch = exactly ``{'topic': new}`` under
       ``metadata_mode='merge'``.  Narrow on purpose: a wider patch would let
       a normalization silently clobber ``canonical``, ``supersedes`` or the
       consolidation bookkeeping across the whole target population at once.
       Never a delete-and-recreate — that would mint a new memory id,
       orphaning every reference to the old one and destroying ``created_at``,
       the field consolidation uses to pick a canonical.

    Every await is individually guarded.  The artifact is written after the
    loop, so an exception escaping here would discard everything the sweep had
    already learned; one backend hiccup must cost one report row instead.

    Args:
        memory_service: The injected ``MemoryService``.
        rename: The frozen plan row.
        apply: ``False`` performs the read and the decision and issues no
            write, so the rehearsal exercises the SAME path ``--apply`` runs
            rather than predicting it.

    Returns:
        A report-ready dict always carrying ``outcome``, plus whatever that
        outcome needs to be actionable without a follow-up query
        (``existing_topic``, ``error``, ``error_type``, ``response``).
    """
    base = {
        'project_id': rename.project_id,
        'memory_id': rename.memory_id,
        'old_topic': rename.old_topic,
        'new_topic': rename.new_topic,
    }

    try:
        record = await memory_service.get_memory_by_id(
            project_id=rename.project_id, memory_id=rename.memory_id,
        )
    except Exception as exc:
        return {
            **base,
            'outcome': 'rename_error',
            'error': f'{type(exc).__name__}: {exc}',
        }
    if record is None:
        return {**base, 'outcome': 'memory_not_found'}

    metadata = dict(record.get('metadata') or {})
    existing_topic = metadata.get('topic')

    if existing_topic == rename.new_topic:
        return {**base, 'outcome': 'already_normalized', 'existing_topic': existing_topic}
    if existing_topic != rename.old_topic:
        return {
            **base,
            'outcome': 'topic_moved_since_plan',
            'existing_topic': existing_topic,
        }

    if not apply:
        return {**base, 'outcome': 'would_rename', 'existing_topic': existing_topic}

    try:
        response = await memory_service.update_memory(
            memory_id=rename.memory_id,
            project_id=rename.project_id,
            metadata_patch={'topic': rename.new_topic},
            metadata_mode='merge',
            reason=WRITE_REASON,
            _source=WRITE_SOURCE,
        )
    except Exception as exc:
        return {
            **base,
            'outcome': 'rename_error',
            'error': f'{type(exc).__name__}: {exc}',
        }
    # ``update_memory`` reports a not-found (and some other rejections) by
    # RETURNING a structured envelope rather than raising, so a caller that
    # only guarded against exceptions would score a refused write as a rename
    # -- and the residue probe would then find the legacy slug still populated
    # with no explanation anywhere in the report.
    if isinstance(response, dict) and response.get('error_type'):
        return {
            **base,
            'outcome': 'update_failed',
            'error_type': response.get('error_type'),
            'error': response.get('error'),
            'response': response,
        }
    return {**base, 'outcome': 'renamed', 'response': response}


# ---------------------------------------------------------------------------
# The two-store lockstep — memories and the consolidation gate move together
# ---------------------------------------------------------------------------

class WriteRejectedError(RuntimeError):
    """An MCP tool call that returned an error envelope inside a success frame."""


def assert_write_accepted(result: object, *, tool: str = 'update_task') -> None:
    """Raise :class:`WriteRejectedError` on a tool-level rejection.

    Restated locally from ``migrate_task_metadata_to_x_namespace.py`` for the
    same reason it exists there: an accepted ``update_task`` returns
    ``{'id', 'message', 'updated', 'updated_task'}`` and a REFUSED one returns
    ``{'success': False, 'error': <code>, ...}`` — inside the SAME successful
    JSON-RPC envelope.  Nothing raises.  A caller guarding only against
    exceptions therefore scores a refused gate move as a completed one, which
    is precisely the state this lockstep exists to make unreachable.

    The rejection most likely to fire here is the ``done_provenance``
    write-authority floor, which refuses any whole-blob metadata replace of a
    done/merged task.  This writer never sends one — see :func:`rename_group`
    — but the guard is what makes that a checked claim rather than a hope.
    """
    if not isinstance(result, dict):
        return
    if result.get('success') is False or result.get('error'):
        raise WriteRejectedError(
            f'{tool} was REJECTED by the server (the JSON-RPC envelope was '
            f'still a success, so this would otherwise be invisible): '
            f'error={result.get("error")!r} '
            f'error_type={result.get("error_type")!r} '
            f'hint={result.get("hint")!r}'
        )


@dataclass(frozen=True)
class GateGroup:
    """Every rename on one slug, plus the gate task that must move with them.

    The unit of atomicity.  Renames are grouped by ``(project_id, old_topic)``
    rather than handled one record at a time because the gate's closure scroll
    addresses a TOPIC, not a record: moving four of a slug's five records
    splits the cluster just as badly as moving none of them.

    Attributes:
        project_id: The corpus this slug lives in.
        old_topic / new_topic: The slug's move, shared by every member.
        renames: The frozen plan rows, in plan order.
        gate_task_id: The consolidation gate filed against *old_topic*, or
            ``None`` — most topics are not gates, and an ungated group needs
            no MCP handshake at all.
        gate_project_root: The gate task's project root (tasks are addressed
            by root, memories by project id — a genuinely different store).
        gate_block: The PATCHED ``x_recon_consolidation_gate`` block: the
            live one with ``topic`` moved and every sibling key carried across
            verbatim.  Built at plan time so the artifact shows exactly what
            ``--apply`` will send.
    """

    project_id: str
    old_topic: str
    new_topic: str
    renames: tuple[Rename, ...]
    gate_task_id: str | None = None
    gate_project_root: str | None = None
    gate_block: dict[str, Any] | None = None


def pair_gate_blocks(renames, gate_tasks) -> tuple[list[GateGroup], list[dict]]:
    """Pair each slug's renames with the consolidation gate filed against it.

    Pure.  Takes the plan rows and the gate tasks as plain dicts (``{'id',
    'project_id', 'project_root', 'metadata'}`` — the shape MCP ``get_task``
    hands over) and returns ``(groups, skips)``.

    ``metadata.x_recon_consolidation_gate`` is a Tier-C block on TASK
    metadata, read by ``reconciliation/consolidation_gate.py`` and enforced on
    the ``done`` transition by ``middleware/task_interceptor.py``.  Its
    ``topic`` is what the closure scroll matches on, so a gate left behind on
    a slug this sweep vacated becomes permanently uncloseable — the trap this
    task exists to remove, re-created at a new address.

    The patch moves ``topic`` and NOTHING else: ``provenance``,
    ``considered_and_kept`` and any future sibling are copied across verbatim.
    A block rebuilt from scratch would silently drop the provenance a curator
    needs to adjudicate the gate.  The source task is never mutated — the
    planner's whole job is to leave the corpus untouched.

    An unmatched gate on a NON-conforming slug is reported as
    ``orphan_gate_topic``: this sweep found no live record carrying it, so
    renaming would not help and staying silent would hide a gate that is
    already uncloseable.  An unmatched gate on a CONFORMING slug is neither
    work nor an orphan and is reported nowhere — nothing to migrate is not
    the same fact as a dangling gate.
    """
    # (project_id, topic) -> the gate tasks carrying it, lowest id first so a
    # pairing is a property of the corpus rather than of read order.
    gates_by_topic: dict[tuple[str, str], list[dict]] = {}
    for task in gate_tasks or ():
        metadata = task.get('metadata') or {}
        block = metadata.get(GATE_METADATA_KEY)
        if not isinstance(block, dict):
            continue
        topic = block.get('topic')
        if not isinstance(topic, str) or not topic:
            continue
        key = (str(task.get('project_id') or ''), topic)
        gates_by_topic.setdefault(key, []).append(task)
    for bucket in gates_by_topic.values():
        bucket.sort(key=lambda t: str(t.get('id')))

    by_slug: dict[tuple[str, str], list[Rename]] = {}
    for rename in renames or ():
        by_slug.setdefault((rename.project_id, rename.old_topic), []).append(rename)

    groups: list[GateGroup] = []
    paired: set[tuple[str, str]] = set()
    for key, members in sorted(by_slug.items(), key=lambda kv: (kv[0][0], kv[1][0].new_topic)):
        project_id, old_topic = key
        gate = (gates_by_topic.get(key) or [None])[0]
        block: dict[str, Any] | None = None
        if gate is not None:
            paired.add((project_id, old_topic, str(gate.get('id'))))
            live_block = gate['metadata'][GATE_METADATA_KEY]
            # Copy, then move exactly one key: the source task stays as read.
            block = {**live_block, 'topic': members[0].new_topic}
        groups.append(GateGroup(
            project_id=project_id,
            old_topic=old_topic,
            new_topic=members[0].new_topic,
            renames=tuple(members),
            gate_task_id=None if gate is None else str(gate.get('id')),
            gate_project_root=None if gate is None else gate.get('project_root'),
            gate_block=block,
        ))
    groups.sort(key=lambda g: (g.project_id, g.new_topic))

    skips: list[dict] = []
    for (project_id, topic), bucket in gates_by_topic.items():
        for task in bucket:
            if (project_id, topic, str(task.get('id'))) in paired:
                continue
            if is_valid_topic_slug(topic):
                continue
            skips.append({
                'reason': 'orphan_gate_topic',
                'project_id': project_id,
                'gate_task_id': str(task.get('id')),
                'gate_project_root': task.get('project_root'),
                'gate_topic': topic,
                'note': (
                    'a consolidation gate carries a non-conforming slug that '
                    'THIS sweep matched to no live record, so normalizing the '
                    'corpus will not close it — the gate needs hand work '
                    '(check the under_enumerated bucket before concluding the '
                    'cluster is genuinely empty)'
                ),
            })
    skips.sort(key=lambda s: (s['project_id'], s['gate_topic'], s['gate_task_id']))
    return groups, skips


def _gate_row(group: GateGroup, outcome: str, **extra) -> dict:
    return {
        'project_id': group.project_id,
        'gate_task_id': group.gate_task_id,
        'gate_project_root': group.gate_project_root,
        'old_topic': group.old_topic,
        'new_topic': group.new_topic,
        'memory_ids': [r.memory_id for r in group.renames],
        'outcome': outcome,
        **extra,
    }


async def _undo_group(memory_service, group: GateGroup, results: list[dict]) -> list[dict]:
    """Put back exactly the writes THIS run made, and report what would not go.

    An undo is just a rename in the opposite direction, so it runs through
    :func:`rename_one` rather than a second write path — same live re-read,
    same narrow metadata-only patch, same guard posture.

    Only rows scored ``renamed`` are reversed.  A record that was
    ``already_normalized`` arrived at the new value before this sweep touched
    it; "undoing" it would be this script inventing a write of its own, and
    would move a record the operator never asked it to move.
    """
    failures: list[dict] = []
    for result in results:
        if result.get('outcome') != 'renamed':
            continue
        undo = await rename_one(
            memory_service,
            Rename(
                project_id=result['project_id'],
                memory_id=result['memory_id'],
                old_topic=result['new_topic'],
                new_topic=result['old_topic'],
            ),
            apply=True,
        )
        if undo.get('outcome') != 'renamed':
            failures.append({
                'memory_id': result['memory_id'],
                'outcome': undo.get('outcome'),
                'error': undo.get('error') or undo.get('error_type'),
            })
        result['undone'] = undo.get('outcome') == 'renamed'
        result['outcome'] = 'gate_lockstep_failed'
    return failures


async def rename_group(
    memory_service,
    group: GateGroup,
    *,
    apply: bool,
    client: Any = None,
) -> tuple[list[dict], dict | None]:
    """Move one slug in both stores, or leave it exactly where it started.

    Order, and why it is this way round:

    1. **the memory renames**, all of them, through :func:`rename_one`;
    2. **the gate patch**, and only once every one of them succeeded.

    The gate is second because it is the single atomic write of the pair: N
    record patches cannot be made atomic, so the one call that can be is the
    commit point.  If it is refused, step 1 is UNDONE, so the pair's net state
    is unchanged and the group is reported as ``gate_lockstep_failed``.  A
    failed half never stands alone in either direction — memories on the new
    slug with the gate on the old, or the gate on the new with the records on
    the old, are the SAME uncloseable-gate failure, just seen from two sides.

    (The undo is why "hold the whole group" is achievable at all with
    memory-first ordering.  It is deliberately narrow: it reverses only the
    writes this run made, and a reversal that itself fails is named on the row
    — an operator must never have to infer that a half-applied state exists.)

    An UNGATED group is the common case and short-circuits after step 1: it
    needs no client, so a sweep that touches no gate never opens a socket.
    A GATED group with no client is the same verdict as a refused gate —
    unable to move it is not permission to leave it behind.

    Returns:
        ``(memory results, gate row or None)``.  Rows whose write was undone
        are re-scored ``gate_lockstep_failed`` so the report never claims a
        rename that no longer stands; rows that never wrote keep their own
        diagnosis, which is the actionable one.
    """
    results = [
        await rename_one(memory_service, rename, apply=apply)
        for rename in group.renames
    ]
    if group.gate_task_id is None:
        return results, None

    stood = {'renamed', 'would_rename', 'already_normalized'}
    failed = [r for r in results if r.get('outcome') not in stood]
    if failed:
        undo_failures = await _undo_group(memory_service, group, results)
        logger.warning(
            'GATE LOCKSTEP HELD project=%s topic=%s -> %s: %d of %d memory writes '
            'did not stand, so gate task %s was NOT patched.',
            group.project_id, group.old_topic, group.new_topic,
            len(failed), len(results), group.gate_task_id,
        )
        return results, _gate_row(
            group, 'gate_lockstep_failed',
            half='memory',
            error=f'{len(failed)} of {len(results)} memory renames did not stand',
            failed_outcomes=sorted({str(r.get('outcome')) for r in failed}),
            undo_failures=undo_failures,
            note=(
                'the gate patch was withheld, so the gate still points at the '
                'legacy slug — resolve the memory-side failures and re-run'
            ),
        )

    if not apply:
        return results, _gate_row(group, 'would_patch_gate', patch=group.gate_block)

    error: str | None = None
    if client is None:
        error = (
            'no MCP client: the gate half of this pair could not be attempted, '
            'and an unmovable gate is refused exactly like a refused one'
        )
    else:
        payload = {
            'id': group.gate_task_id,
            'project_root': group.gate_project_root,
            # A NARROW merge of one Tier-C block, never a whole-blob replace:
            # `update_task` refuses any replace carrying `done_provenance`, and
            # a replace here would also stake the task's entire metadata on
            # this script having read it back correctly.
            'metadata': {GATE_METADATA_KEY: dict(group.gate_block or {})},
            'metadata_mode': 'merge',
        }
        try:
            response = await client.call_tool('update_task', payload)
            assert_write_accepted(response, tool='update_task')
        except Exception as exc:
            error = f'{type(exc).__name__}: {exc}'
        else:
            return results, _gate_row(
                group, 'gate_patched', patch=group.gate_block, response=response)

    undo_failures = await _undo_group(memory_service, group, results)
    logger.warning(
        'GATE LOCKSTEP HELD project=%s topic=%s -> %s: gate task %s was not '
        'patched (%s); %d memory write(s) undone, %d undo failure(s).',
        group.project_id, group.old_topic, group.new_topic, group.gate_task_id,
        error, len(group.renames), len(undo_failures),
    )
    return results, _gate_row(
        group, 'gate_lockstep_failed',
        half='gate', error=error, undo_failures=undo_failures,
        note=(
            'the memory half was undone so the slug is unchanged in BOTH '
            'stores; a gate left pointing at a slug no record carries is '
            'permanently uncloseable, which is worse than not migrating'
        ),
    )


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
    import importlib.util  # noqa: PLC0415

    sibling = Path(__file__).parent / 'strip_leaked_control_keys.py'
    mod_name = 'strip_leaked_control_keys'
    existing = sys.modules.get(mod_name)
    if existing is not None and hasattr(existing, 'FusedMemoryClient'):
        return existing.FusedMemoryClient
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
    return module.FusedMemoryClient


# ---------------------------------------------------------------------------
# Scope item 3 — did the old slugs actually drain?
# ---------------------------------------------------------------------------

#: Outcomes whose rows still carry the LEGACY topic value at probe time and
#: were put there by this run's own rehearsal.  Only these are subtracted.
#: A failed write also still carries it — and that is genuine residue,
#: correctly left in.
_REHEARSED_OUTCOMES = frozenset({'would_rename'})

#: Outcomes that actually moved (or intended to move) a slug.  Anything else
#: never touched one, so there is nothing to drain and nothing to probe.
_MOVED_OUTCOMES = frozenset({'renamed', 'would_rename', 'update_failed', 'rename_error'})


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
        if result.get('outcome') not in _MOVED_OUTCOMES:
            continue
        legacy = result.get('old_topic')
        if not isinstance(legacy, str) or not legacy:
            continue
        entry = pending.setdefault(
            (result['project_id'], legacy),
            {'new_topic': result.get('new_topic'), 'rehearsed': 0},
        )
        if result.get('outcome') in _REHEARSED_OUTCOMES:
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


async def run(
    memory_service,
    *,
    projects: tuple[str, ...] = DEFAULT_PROJECTS,
    apply: bool = False,
    client: Any = None,
    gate_tasks: list[dict] | None = None,
    page_size: int = DEFAULT_PAGE_SIZE,
    max_pages: int = DEFAULT_MAX_PAGES,
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
        gate_tasks: Consolidation-gate tasks, each stamped with the
            ``project_id`` whose corpus its topic lives in.  Injected so the
            tests never open a socket.

    Returns:
        The report dict — rendered by :func:`render_json` /
        :func:`render_markdown` and graded by :func:`resolve_exit_code`.
    """
    # Fail-CLOSED capability preflight: ONE probe per run, ABOVE the scroll.
    #
    # ``run`` is the choke point precisely because ``rename_one``'s own
    # ``apply`` gate is PER RECORD: probing there would run the check once per
    # target and -- since ``StoreMutationUnavailable`` subclasses
    # ``RuntimeError`` -- be swallowed by the per-record ``except Exception``
    # around the write, downgrading a run-wide environment denial into N error
    # rows inside a report that otherwise reads as a completed sweep.
    #
    # Emitted through the logger, never ``print``: stdout is reserved for the
    # machine-read artifact rendered at the end of ``main``.
    if apply:
        try:
            assert_store_mutation_allowed(operation='normalize_topic_slugs --apply')
        except StoreMutationUnavailable:
            logger.error(
                'normalize_topic_slugs: --apply NOT started (fail-closed) -- '
                "this process cannot write mem0's history directory, so each "
                'rename would patch a record and then fail to journal the '
                'change, leaving the corpus HALF-RENAMED: one claim split '
                'across two topic values, which is strictly worse for an '
                "exact-match get_memories_by_metadata({'topic': T}) read -- "
                'the way consolidation clusters and the canonical-uniqueness '
                'probe address a topic -- than either uniform state. Nothing '
                'was scrolled and no record was renamed. Route the migration '
                'through the fused-memory MCP server (the unsandboxed owner '
                'of the store), or re-run from an unsandboxed operator shell. '
                'To obtain the sweep report safely from anywhere, re-run '
                'without --apply.'
            )
            raise

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
        measured_by_project[project_id] = len({
            (r.get('metadata') or {}).get('topic') for r in enumerated['records']
        })
        _record_skips(enumerated['skips'])

        renames, plan_skips = plan_renames(
            enumerated['records'],
            project_id=project_id,
            occupied_topics=enumerated['occupied_topics'],
            canonical_ids_by_topic=enumerated['canonical_ids_by_topic'],
        )
        _record_skips(plan_skips)
        all_renames.extend(renames)

    groups, gate_skips = pair_gate_blocks(all_renames, gate_tasks or [])
    _record_skips(gate_skips)

    results: list[dict] = []
    gate_results: list[dict] = []
    for group in groups:
        group_results, gate_row = await rename_group(
            memory_service, group, apply=apply, client=client)
        results.extend(group_results)
        if gate_row is not None:
            gate_results.append(gate_row)

    # Scope item 3, in the same run that did the writing.
    await verify_old_slugs_drained(memory_service, results, skips)

    outcomes: dict[str, int] = {}
    renamed_by_topic: dict[str, int] = {}
    would_rename_by_topic: dict[str, int] = {}
    for row in results:
        outcome = str(row.get('outcome'))
        outcomes[outcome] = outcomes.get(outcome, 0) + 1
        if outcome == 'renamed':
            topic = str(row.get('new_topic'))
            renamed_by_topic[topic] = renamed_by_topic.get(topic, 0) + 1
        elif outcome == 'would_rename':
            topic = str(row.get('new_topic'))
            would_rename_by_topic[topic] = would_rename_by_topic.get(topic, 0) + 1
    for row in gate_results:
        outcome = str(row.get('outcome'))
        outcomes[outcome] = outcomes.get(outcome, 0) + 1

    # A skip that grades as an error must reach the exit code even when no
    # RESULT row carries it: a collision is refused at plan time, so it never
    # produces a rename row, and grading outcomes alone would exit 0 on a run
    # that declined to migrate the most dangerous slug it found.
    for bucket, entries in skips.items():
        if entries and bucket in ERROR_OUTCOMES:
            outcomes[bucket] = outcomes.get(bucket, 0) + len(entries)

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

#: The skip keys worth printing inline, in reading order.  A whitelist rather
#: than a dump: rows that land in error buckets carry a whole ``response``
#: envelope, which would bury the fields that identify the record.
_SKIP_DETAIL_KEYS: tuple[str, ...] = (
    'project_id', 'category', 'memory_id', 'memory_ids', 'gate_task_id',
    'topic', 'raw_topic', 'old_topic', 'new_topic', 'gate_topic',
    'legacy_topic', 'existing_topic', 'target_topic', 'source_topics',
    'canonical_memory_ids', 'incumbent_ids', 'record_count', 'residue_count',
    'expected', 'scrolled', 'recount', 'delta', 'half', 'error', 'error_type',
    'undo_failures', 'note',
)


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
    parts = [
        f'{key}={entry[key]!r}'
        for key in _SKIP_DETAIL_KEYS
        if entry.get(key) is not None
    ]
    # An entry the renderer does not recognise is exactly the one worth
    # showing verbatim, rather than as a bare bullet.
    return '- ' + (', '.join(parts) if parts else repr(entry))


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
    lines += ['', '## Consolidation gates', '']
    if gate_results:
        for row in gate_results:
            lines.append(
                f'- task {row.get("gate_task_id")}: `{row.get("old_topic")}` -> '
                f'`{row.get("new_topic")}` — {row.get("outcome")}'
                + (f' ({row.get("error")})' if row.get('error') else '')
            )
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


def main(argv: list[str] | None = None) -> int:
    """Build a live service, run the sweep, write both artifacts, exit graded."""
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
        import os  # noqa: PLC0415

        os.environ['CONFIG_PATH'] = str(args.config)

    async def _run_live() -> dict:
        # Deferred so importing this module -- which the tests do, by path --
        # never constructs a backend.
        from fused_memory.config.schema import FusedMemoryConfig  # noqa: PLC0415
        from fused_memory.services.memory_service import MemoryService  # noqa: PLC0415

        config = FusedMemoryConfig()
        memory = MemoryService(config)
        try:
            await memory.initialize()
            return await run(
                memory, projects=resolve_projects(args), apply=args.apply,
            )
        finally:
            if hasattr(memory, 'close'):
                await memory.close()

    report = asyncio.run(_run_live())

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
