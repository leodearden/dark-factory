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

Dry run is the default
----------------------
Nothing is written without ``--apply``.  See the operator runbook below.
"""
from __future__ import annotations

import functools
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from fused_memory.backends.mem0_client import (
    DEFAULT_SCROLL_MAX_PAGES,
    ScrollPageBudgetExhausted,
)
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
    'DEFAULT_MAX_PAGES',
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
    'plan_renames',
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
                if topic is None or is_valid_topic_slug(topic):
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

def plan_renames(records, *, project_id: str) -> tuple[list[Rename], list[dict]]:
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
    observed_topics: set[str] = set()
    canonical_ids_by_topic: dict[str, list[str]] = {}
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
            if isinstance(memory_id, str) and memory_id:
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
