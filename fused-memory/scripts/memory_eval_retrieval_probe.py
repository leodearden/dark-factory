#!/usr/bin/env python3
"""E1 retrieval-health probe — does the memory system return the right thing?

``docs/prds/memory-eval-program.md`` §5 leaf β. The 3111/3112 fix lineage
(canonical pinning, consolidation, curator gates) changed how the corpus is
written; nothing measures whether *retrieval* actually improved. This runner
probes a committed registry of topics with several query phrasings each,
emits an M1 metric series recording what came back, and writes a
human-readable report beside it.

**This script never writes to the live corpus.** There is no ``--apply``
band, no delete/add/update call and no write path anywhere in this module.
D8's read-only runner pattern (``audit_duplicate_memories.py:364-378``) is
copied minus every mutation. The guarantee is asserted as BEHAVIOUR in the
tests — the probe is driven against a service double whose every write method
raises — rather than merely claimed in this docstring. Nor does it evaluate a
limit: it emits measurements, and every threshold lives in leaf α's limits
evaluator (see ``fused_memory/eval/retrieval_probe.py``).

**Schema home (D2).** Every artifact model, validator, path helper, stamp
format and report renderer comes from :mod:`shared.memory_eval_metrics`.
Nothing in that contract is re-declared here.

**What lives where.** The registry data contract (models and loader) is
``fused_memory/eval/topic_registry.py``, and the pure metric math with its M1
series assembly is ``fused_memory/eval/retrieval_probe.py``. This script keeps
the ``--derive-registry`` derivation, the report renderer and the CLI runner.
"""
from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import logging
import os
import re
import sqlite3
import sys
import tempfile
import textwrap
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from types import MappingProxyType
from typing import Any, NamedTuple

from shared.cli_boundary import (
    LoudArgumentParser,
    reset_stdout_failure_state,
    run_cli,
)

from fused_memory.eval.retrieval_probe import (
    DEFAULT_KS,
    EVAL_ID,
    MATCHED_BY_CONTENT_HASH,
    MATCHED_BY_LAST_KNOWN_ID,
    METRIC_CANONICAL_IN_TOP_K_HELD_OUT,
    METRIC_TOPIC_CANONICAL_PRESENT,
    TRIPWIRE_K,
    ClaimObservation,
    ContaminationObservation,
    DegradedQuery,
    InversionObservation,
    ProbeObservations,
    build_series,
    claim_recalled,
    classify_contamination,
    comparable_pairs,
    content_key,
    metric_families_not_measured,
    normalise_ks,
    normalize_whitespace,
    not_measured_topics,
    observe_phrasing,
    rank_index,
    ranks_at_depth,
    superseded_inversions,
    tripwire_items,
)
from fused_memory.eval.topic_registry import (
    REGISTRY_SCHEMA_VERSION,
    RegistryEntry,
    RegistryError,
    TopicRegistry,
    corpus_categories,
    load_topic_registry,
    normalize_topic,
)

logger = logging.getLogger('memory_eval_retrieval_probe')

# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------

class EmptySelectionError(ValueError):
    """No registry entry matched the requested ``--project-id`` selection.

    The same hazard :class:`RegistryError` guards, entered by a different door:
    a mistyped project id selects nothing, every metric family measures zero
    topics, and the run emits an artifact whose ``metrics`` list is empty. That
    artifact is not inert — the evaluator joins runs by ``metric_id`` and simply
    stops trending the seven pinned metrics, and :func:`is_initial_run` counts
    the file, permanently suppressing the D1 initial-state snapshot for the next
    genuine first run. So the selection miss aborts BEFORE emission, exactly as
    a failed registry load does, and the message names both what was asked for
    and what the registry actually carries.
    """


class CensusShapeError(ValueError):
    """The census parsed, but lacks a table this script reads.

    :func:`load_census_coverage` turns it into ``unavailable_reason``, because a
    probe run is never fatal on its census. ``--derive-registry`` refuses it
    with :data:`EXIT_BAD_INPUT`, because there the census IS the input.
    """


class ProbeRunError(RuntimeError):
    """The run could not complete; the message is attributed at the seam that raised it.

    Deliberately not an ``OSError``; see ``shared/src/shared/cli_boundary.py::run_cli``.
    """


# ---------------------------------------------------------------------------
# Offline registry derivation
#
# Every source is COMMITTED, so this band needs no Qdrant, no embedder and no
# OPENAI_API_KEY: the fixture is reproducible in CI and a reviewer can re-run
# the derivation to audit any entry. Derivation lives here rather than in a
# fourth script so the deriver writes exactly what the loader reads — a schema
# drift between them is impossible by construction.
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DerivationResult:
    """Candidate registry entries plus what derivation had to leave out.

    ``disclosures`` is not decoration. Derivation narrows its inputs (the
    count-1 census tail, rows the curator adjudicated as distinct), and a
    narrowing nobody can see reads downstream as "there was nothing there" —
    the silent cap this repo's norms forbid.

    So EVERY drop is counted, and the counters are kept distinct by cause: a
    census row dropped for a malformed ``value`` is a broken census, while
    one dropped for ``count <= 1`` is a healthy census with an uninformative
    tail. Folding them into one number reports a schema break as a corpus
    property, which is the reading that stops anyone looking.

    An absent or misshapen topic table is REFUSED (:class:`CensusShapeError`),
    not counted, because it would read zero in every census counter, the same
    as an empty corpus.
    """

    candidates: tuple[dict[str, Any], ...]
    disclosures: dict[str, int] = field(default_factory=dict)

    def as_registry_payload(self) -> dict[str, Any]:
        """The candidates in the registry's OWN JSON shape.

        Deliberately incomplete: phrasings carry no held-out entry and there
        are no claim_queries, because those are the parts a machine cannot
        regenerate. An operator hand-completes them and commits the result.
        """
        return {
            'schema_version': REGISTRY_SCHEMA_VERSION,
            'entries': [dict(c) for c in self.candidates],
        }


def _candidate(
    topic: str,
    *,
    project_id: str,
    derived_from: str,
    content_hash: str,
    content_prefix: str = '',
    last_known_id: str | None = None,
    phrasings: list[str],
    members: list[str],
    supersedes_pairs: list[dict[str, str]] | None = None,
    provenance: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        'topic': topic,
        'project_id': project_id,
        'derived_from': derived_from,
        'provenance': provenance or {},
        'canonical': {
            'content_hash': content_hash,
            'content_prefix': content_prefix[:160],
            'last_known_id': last_known_id,
        },
        # Never held_out: see DerivationResult.as_registry_payload.
        'phrasings': [{'text': text, 'held_out': False} for text in phrasings],
        'claim_queries': [],
        'members': members,
        'supersedes_pairs': supersedes_pairs or [],
    }


def _derive_curator_gate_candidates(rows: list[dict]) -> tuple[list[dict[str, Any]], int]:
    """Curator-gate candidates, plus the clusters that carried no canonical row.

    A cluster the curator never adjudicated a canonical for cannot become an
    entry — "is the canonical in the top k" has no subject. Skipping it is
    correct; skipping it in silence is not, because a calibration file that
    lost its canonical labels would derive fewer topics while reading, from
    the disclosure block alone, exactly like a smaller corpus.
    """
    groups: dict[str, list[dict]] = {}
    for row in rows:
        groups.setdefault(row['cluster_id'], []).append(row)

    candidates: list[dict[str, Any]] = []
    without_canonical = 0
    for cluster_id, group in sorted(groups.items()):
        canonical_rows = [r for r in group if r.get('label') == 'canonical']
        if not canonical_rows:
            without_canonical += 1
            continue
        canonical = canonical_rows[0]
        # ONLY 'duplicate'. The curator adjudicated 'distinct' and
        # 'pseudo_contradiction' rows as separate claims that merely READ as
        # contradictory; treating them as members would tell the probe that a
        # legitimately different answer is contamination.
        duplicates = [r for r in group if r.get('label') == 'duplicate']
        canonical_hash = content_key(canonical['content'])
        member_hashes = sorted({content_key(r['content']) for r in duplicates})
        gate_ids = sorted({
            r.get('provenance', {}).get('gate_id')
            for r in group
            if isinstance(r.get('provenance'), dict) and r['provenance'].get('gate_id')
        })
        candidates.append(_candidate(
            _slugify(canonical.get('topic') or cluster_id),
            project_id=canonical.get('project_id', 'reify'),
            derived_from='curator_gate',
            content_hash=canonical_hash,
            content_prefix=normalize_whitespace(canonical['content']),
            last_known_id=canonical.get('memory_id'),
            phrasings=[],
            members=member_hashes,
            supersedes_pairs=[
                {'superseded_hash': h, 'successor_hash': canonical_hash}
                for h in member_hashes if h != canonical_hash
            ],
            provenance={
                'cluster_id': cluster_id,
                'gate_ids': gate_ids,
                'labels': {
                    'canonical': len(canonical_rows),
                    'duplicate': len(duplicates),
                    'other': len(group) - len(canonical_rows) - len(duplicates),
                },
            },
        ))
    return candidates, without_canonical


class _CensusDerivation(NamedTuple):
    """``(candidates, skipped_singleton, malformed_value, malformed_count)``.

    Three separate narrowing counters, not one: a row dropped because its
    ``value`` was not a string is a MALFORMED census, while a row dropped for
    ``count <= 1`` is a healthy census whose tail is uninformative. Folding
    them together (as an earlier shape did) reported a schema break as a
    corpus property, which is the one reading that stops anyone looking.
    """

    candidates: list[dict[str, Any]]
    skipped_singleton: int
    malformed_value: int
    malformed_count: int


_CENSUS_TOPIC_TABLE_PATH = ('grand_total', 'topic', 'entries')


def _census_topic_table(census: Any) -> list[Any]:
    """The census's ``grand_total.topic.entries`` rows, which may be empty.

    An empty list is a healthy census of a corpus with no topic values. An
    absent or misshapen table raises :class:`CensusShapeError` naming where
    the walk stopped, because it would otherwise read zero in every census
    counter, exactly like that empty corpus.
    """
    node, location = census, 'the top level'
    for depth, key in enumerate(_CENSUS_TOPIC_TABLE_PATH):
        if not isinstance(node, dict):
            raise CensusShapeError(f'{location} is {type(node).__name__}, not an object')
        node = node.get(key)
        location = '.'.join(_CENSUS_TOPIC_TABLE_PATH[:depth + 1])
        if node is None:
            raise CensusShapeError(f'{location} is null or absent')
    if not isinstance(node, list):
        raise CensusShapeError(f'{location} is {type(node).__name__}, not a list')
    return node


def _derive_census_candidates(census: Any) -> _CensusDerivation:
    """Multi-entry census topics, plus every row the derivation had to drop.

    A count-1 topic has exactly one entry, so "is the canonical in the top k"
    is answered by that entry's mere existence — it measures presence, not
    retrieval. They are skipped, and the count is returned so the skip is
    reported rather than silently applied. Rows whose ``value`` or ``count``
    is the wrong shape are counted separately, for the reason
    :class:`_CensusDerivation` states.

    Candidates come out most-used first, ties broken by value, whatever order
    the census file lists its rows in: that order is what lets the folded
    dedup in :func:`derive_registry_candidates` keep a topic's more-used
    spelling.
    """
    table = _census_topic_table(census)
    multi_entry: list[tuple[str, int]] = []
    skipped = 0
    malformed_value = 0
    malformed_count = 0
    for item in table:
        if not isinstance(item, dict):
            malformed_value += 1
            continue
        value, count = item.get('value'), item.get('count', 0)
        if not isinstance(value, str) or not value:
            malformed_value += 1
            continue
        if not isinstance(count, int) or isinstance(count, bool):
            malformed_count += 1
            continue
        if count <= 1:
            skipped += 1
            continue
        multi_entry.append((value, count))
    multi_entry.sort(key=lambda row: (-row[1], row[0]))
    candidates = [
        _candidate(
            _slugify(value),
            project_id='dark_factory',
            derived_from='census_topic',
            # Unknown offline: the census records topic VALUES and counts, not
            # content. The operator resolves it; until then the entry is
            # incomplete and the loader will say so by name.
            content_hash='',
            phrasings=[value.replace('_', ' ').replace('-', ' ')],
            members=[],
            provenance={'census_topic_value': value, 'census_count': count},
        )
        for value, count in multi_entry
    ]
    return _CensusDerivation(candidates, skipped, malformed_value, malformed_count)


def _derive_guard_cluster_candidates(clusters) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for cluster in clusters:
        candidates.append(_candidate(
            _slugify(cluster.topic_id),
            project_id='dark_factory',
            derived_from='topic_guard_cluster',
            content_hash='',
            phrasings=list(cluster.phrases),
            members=[],
            provenance={'guard_topic_id': cluster.topic_id, 'hint': cluster.hint},
        ))
    return candidates


def _slugify(value: str) -> str:
    slug = re.sub(r'[^a-z0-9_-]+', '-', value.strip().lower()).strip('-_')
    return re.sub(r'-{2,}', '-', slug) or 'unnamed-topic'


def derive_registry_candidates(
    calibration_rows: list[dict],
    census_report: Any,
    guard_clusters,
) -> DerivationResult:
    """Derive candidate registry entries from the three COMMITTED sources.

    Pure: no network, no Qdrant, no ``MemoryService``, and deterministic — two
    runs on the same inputs produce byte-identical output, so a reviewer can
    re-derive and diff.

    Returns candidates in the registry's own shape, minus the two things
    derivation cannot invent: a freshly authored held-out phrasing, and the
    per-facet claim needles. Those are hand-authored, which is exactly why the
    CLI band prints rather than overwrites (see :func:`run_derive_registry`).

    Two candidates collide when their slugs fold together under
    :func:`normalize_topic`, because the corpus spells one topic with ``-``
    and ``_``. The richer one is kept, in its own spelling: curator_gate >
    topic_guard_cluster > census_topic, since a curator gate knows its
    canonical's hash, a guard cluster carries hand-written match phrases, and
    a census topic has only a phrasing synthesised from its own slug. Between
    two census rows, the more-used spelling is kept, and on a tie the
    lexically first. Every dropped candidate is counted in
    ``slug_collisions_dropped``, so the per-source counts plus collisions
    account for every candidate generated.
    """
    curator_candidates, clusters_without_canonical = _derive_curator_gate_candidates(
        calibration_rows,
    )
    census = _derive_census_candidates(census_report)
    candidates = [
        *curator_candidates,
        *_derive_guard_cluster_candidates(guard_clusters),
        *census.candidates,
    ]

    seen: set[str] = set()
    deduped: list[dict[str, Any]] = []
    collisions = 0
    for candidate in candidates:
        folded = normalize_topic(candidate['topic'])
        if folded in seen:
            collisions += 1
            continue
        seen.add(folded)
        deduped.append(candidate)
    deduped.sort(key=lambda c: c['topic'])

    return DerivationResult(
        candidates=tuple(deduped),
        disclosures={
            'curator_gate_clusters': sum(
                1 for c in deduped if c['derived_from'] == 'curator_gate'
            ),
            'census_topics_emitted': sum(
                1 for c in deduped if c['derived_from'] == 'census_topic'
            ),
            'census_topics_skipped_singleton': census.skipped_singleton,
            'topic_guard_clusters': sum(
                1 for c in deduped if c['derived_from'] == 'topic_guard_cluster'
            ),
            'slug_collisions_dropped': collisions,
            # The three narrowings that used to happen in silence. Each one
            # means a DIFFERENT thing has gone wrong upstream, so each gets its
            # own counter rather than being folded into the singleton skip:
            # a calibration file that lost its canonical labels, a census whose
            # topic values changed shape, and a census whose counts did.
            'curator_clusters_without_canonical': clusters_without_canonical,
            'census_rows_malformed_value': census.malformed_value,
            'census_rows_malformed_count': census.malformed_count,
        },
    )


def run_derive_registry(
    calibration_path: str | Path,
    census_path: str | Path,
    *,
    guard_clusters=None,
) -> str:
    """Render derived candidates as registry-shaped JSON, for an operator to complete.

    Deliberately does NOT overwrite the committed fixture. The hand-authored
    phrasings — above all the held-out ones — are the part machines cannot
    regenerate, so clobbering the file in place would destroy the very thing
    the Goodhart guard depends on. Print, diff, merge by hand.
    """
    if guard_clusters is None:
        from fused_memory.config.schema import (  # noqa: PLC0415
            _default_topic_guard_clusters,
        )

        guard_clusters = _default_topic_guard_clusters()

    rows = [
        json.loads(line)
        for line in Path(calibration_path).read_text(encoding='utf-8').splitlines()
        if line.strip()
    ]
    census_text = Path(census_path).read_text(encoding='utf-8')
    try:
        census = json.loads(census_text)
    except ValueError as exc:
        raise CensusShapeError(
            f'the census at {census_path} is not decodable JSON: {exc}'
        ) from exc
    try:
        result = derive_registry_candidates(rows, census, guard_clusters)
    except CensusShapeError as exc:
        raise CensusShapeError(
            f'the census at {census_path} has no usable topic table for '
            f'--derive-registry: {exc}'
        ) from exc
    payload = result.as_registry_payload()
    payload['_disclosures'] = result.disclosures
    return json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + '\n'


# ---------------------------------------------------------------------------
# The per-query probe band
#
# The ONE place this module talks to a search callable. Everything downstream
# is pure, which is what keeps the whole metric family testable in the merge
# lane with no Qdrant and no OPENAI_API_KEY.
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DegradeInfo:
    """A search's results together with the degrade metadata read off it."""

    results: list
    degraded: bool = False
    failed_stores: tuple[str, ...] = ()
    diagnostics: tuple[dict, ...] = ()


def read_degrade_metadata(raw: Any) -> DegradeInfo:
    """Read ``degraded``/``failed_stores``/``failure_diagnostics`` off *raw* FIRST.

    ``MemoryService.search`` returns a ``SearchResults`` (a ``list`` subclass)
    carrying that metadata in-band, and its own docstring
    (``memory_service.py:706-712``) warns that the attributes do not survive a
    slice, a ``sorted()``, a concatenation or a comprehension — those return a
    plain ``list`` and drop them SILENTLY.

    That is why this function exists and why the band calls it before touching
    the list. A probe that sliced to top-k first would, during a Qdrant outage,
    see an empty plain list with no degrade flag on it and report every
    canonical as missing — a fabricated corpus-wide collapse, arriving in leaf
    ε's lap as findings against the 3111 lineage.

    A plain list is read as healthy: no metadata is the ordinary shape for a
    caller that never had a degraded search to report, and inventing
    degradation from its absence would be its own false signal.
    """
    degraded = bool(getattr(raw, 'degraded', False))
    stores = tuple(getattr(raw, 'failed_stores', ()) or ())
    diagnostics = tuple(getattr(raw, 'failure_diagnostics', ()) or ())
    return DegradeInfo(
        results=list(raw),
        degraded=degraded or bool(stores),
        failed_stores=stores,
        diagnostics=diagnostics,
    )


async def probe_topic(
    search,
    entry: RegistryEntry,
    registry: TopicRegistry,
    ks: tuple[int, ...],
    observations: ProbeObservations,
) -> None:
    """Probe one registry topic, appending every observation to *observations*.

    *search* is an awaitable ``search(query, limit)`` returning the store's
    result list — injected rather than reached for, so the whole band is
    exercisable without a live store.

    Each query is searched ONCE, at the widest *k*, and the single returned
    list is then sliced per *k*. Searching per k would double the embedding
    spend and, worse, compare two independently-retrieved lists: k=5 and k=10
    would no longer be the same measurement at two depths.

    Every query that reported a failed store is recorded in
    ``degraded_queries`` and every observation it produced is flagged, so the
    aggregation can exclude it instead of scoring an outage as a regression.

    Claim recall and contamination are defined at :data:`TRIPWIRE_K`, but this
    function is public and callable directly, so it scores them at
    ``min(TRIPWIRE_K, limit)`` — the depth actually FETCHED. Stamping the
    constant instead would let ``probe_topic(ks=(3,))`` label observations
    built from three results as "top 5". :func:`run_probe` normalises ``ks``
    so that never happens from the CLI; this keeps the label honest for the
    direct callers too, and neither guard depends on the other.

    The superseded-inversion family is pinned at the same ``scored_k`` for the
    same reason, one level further down: its ``pairs_comparable`` is not just
    a count but a DENOMINATOR (the metric's ``n``), so if it moved with
    ``--k`` a deeper fetch would silently change what the rate is a rate of,
    on top of relabelling it.
    """
    limit = max(ks) if ks else TRIPWIRE_K
    # min(), not max(): a deeper fetch must not silently widen a metric that
    # is defined at the tripwire's depth.
    scored_k = min(TRIPWIRE_K, limit)

    for phrasing in entry.phrasings:
        info = read_degrade_metadata(await search(phrasing.text, limit))
        if info.degraded:
            observations.degraded_queries.append(DegradedQuery(
                topic=entry.topic,
                query=phrasing.text,
                failed_stores=info.failed_stores,
                diagnostics=info.diagnostics,
            ))
        # Hashed ONCE per search, then read by every consumer below. `ranks`
        # (full depth) feeds canonical presence at each k — canonical_hit and
        # observe_phrasing deliberately scan the whole list so a hit beyond
        # the scored depth is reported at its true rank rather than as
        # absent. `scored_ranks` (derived, not re-hashed — see
        # ranks_at_depth) feeds the inversion family, which must be pinned to
        # scored_k like contamination and claim recall.
        ranks = rank_index(info.results)
        scored_ranks = ranks_at_depth(ranks, scored_k)
        for k in ks:
            observations.phrasings.append(observe_phrasing(
                info.results, entry, phrasing, k, degraded=info.degraded, ranks=ranks,
            ))
        outcome = classify_contamination(info.results, entry, registry, scored_k)
        observations.contamination.append(ContaminationObservation(
            topic=entry.topic,
            phrasing=phrasing.text,
            k=scored_k,
            foreign_count=outcome.foreign_count,
            untopiced_count=outcome.untopiced_count,
            scored_total=outcome.scored_total,
            foreign_records=outcome.foreign_records,
            degraded=info.degraded,
        ))
        # Both at once so `pairs_beyond_scored_depth` is exactly their
        # difference — the population both-present at full fetch depth but
        # not both within the scored depth — rather than a second, possibly
        # divergent computation of the same narrowing.
        comparable_at_full_depth = comparable_pairs(entry, ranks)
        comparable_at_scored_depth = comparable_pairs(entry, scored_ranks)
        observations.inversions.append(InversionObservation(
            topic=entry.topic,
            phrasing=phrasing.text,
            k=scored_k,
            pairs_registered=len(entry.supersedes_pairs),
            pairs_comparable=comparable_at_scored_depth,
            pairs_beyond_scored_depth=comparable_at_full_depth - comparable_at_scored_depth,
            inversions=tuple(
                superseded_inversions(
                    info.results, entry, phrasing=phrasing.text, ranks=scored_ranks,
                ),
            ),
            degraded=info.degraded,
        ))

    for claim in entry.claim_queries:
        info = read_degrade_metadata(await search(claim.query, limit))
        if info.degraded:
            observations.degraded_queries.append(DegradedQuery(
                topic=entry.topic,
                query=claim.query,
                failed_stores=info.failed_stores,
                diagnostics=info.diagnostics,
            ))
        outcome = claim_recalled(info.results, claim, scored_k)
        observations.claims.append(ClaimObservation(
            topic=entry.topic,
            query=claim.query,
            k=scored_k,
            recalled=outcome.recalled,
            missing_needles=outcome.missing_needles,
            scorable=outcome.scorable,
            degraded=info.degraded,
        ))


# ---------------------------------------------------------------------------
# The human-readable report
#
# render_report() (shared) covers the metric table; this wraps it with what
# only this runner knows — which queries degraded, which topics went
# unmeasured, and which canonicals could not be matched at all.
# ---------------------------------------------------------------------------

def is_initial_run(root: str | Path, eval_id: str = EVAL_ID) -> bool:
    """True when no prior metrics artifact for *eval_id* exists under *root*.

    Globs ``metrics-*.json`` in the directory the shared :func:`eval_dir`
    defines, so the layout is not restated here. Scoped to this eval's own
    subdirectory because leaves β/γ/δ share one artifact root — a γ run must
    not make β's first run look like its second.

    Only the metrics artifact counts. A stray report with no series beside it
    is not a prior measurement, and treating it as one would silently suppress
    the initial-state snapshot D1 requires.
    """
    from shared.memory_eval_metrics import eval_dir  # noqa: PLC0415

    directory = eval_dir(root, eval_id)
    return not any(directory.glob('metrics-*.json'))


def _wrap(text: str, width: int = 76) -> list[str]:
    """Wrap *text*, never breaking a hyphenated token across lines.

    ``break_on_hyphens`` defaults True, which split ``canonical-in-top-5-held-out``
    mid-name in the prose. Every identifier this report names — metric ids,
    topic slugs, item keys — is hyphenated, and a name broken across a
    newline is a name an operator's grep will not find.
    """
    return textwrap.wrap(text, width=width, break_on_hyphens=False) or ['']


_KNOWN_BAD_PREAMBLE = (
    'This is the INITIAL STATE of this eval: no prior run exists under this '
    'artifact root. The items below are what the corpus looks like today, '
    'inherited from the 3111/3112 fix lineage (canonical pinning, '
    'consolidation, curator gates) that has been rewriting how memory is '
    'written for months. They are a KNOWN-BAD snapshot, NOT A FINDING and not '
    'a regression anyone introduced in this run. Deciding what to do with '
    'this list is a separate leaf\'s job; this runner only measures.'
)

_KNOWN_BAD_ROUTING_CAVEAT = (
    'Read this list together with "which store served the query" below BEFORE '
    'concluding anything about findability. A phrasing the read router sent to '
    'a store the canonical does not live in cannot hit however healthy '
    'retrieval is, so a failing item here can be a routing fact rather than a '
    'corpus one, and the two are not distinguishable from the rate alone. '
    'Which it is belongs to the retrieval-fix lineage, not to this monitor.'
)


SECTION_METRIC_TABLE = 'metric-table'
SECTION_FAMILIES_NOT_MEASURED = 'metric-families-not-measured'
SECTION_INITIAL_STATE = 'initial-state'
SECTION_KNOWN_BAD_ROUTING_CAVEAT = 'initial-state-routing-caveat'
SECTION_KNOWN_BAD_ITEMS = 'initial-state-known-bad-items'
SECTION_DEGRADED_QUERIES = 'degraded-queries'
SECTION_TOPICS_NOT_MEASURED = 'topics-not-measured'
SECTION_TRIPWIRE_BY_CENSUS = 'tripwire-by-census-canonical-presence'
SECTION_UNMATCHED_CANONICALS = 'canonicals-matched-by-neither-key'
SECTION_HASH_REPAIRS = 'canonicals-matched-by-last-known-id-only'
SECTION_CLAIMS_NOT_RECALLED = 'claims-not-recalled'
SECTION_MATCHED_BY = 'matched-by-breakdown'
SECTION_STORES_SERVED = 'stores-served-breakdown'
SECTION_CONTAMINATION = 'contamination-classification'
SECTION_CORPUS_COUNT_SCOPE = 'corpus-count-scope'
SECTION_MEASUREMENT_DEPTH = 'measurement-depth'
SECTION_TOPICS_NOT_PROBED = 'topics-not-probed'
SECTION_REGISTRY_COMPOSITION = 'registry-composition'
SECTION_SUPERSEDED_INVERSIONS = 'superseded-inversions'


@dataclass(frozen=True)
class ReportSection:
    """One block of the human report, under a STABLE machine key.

    The key is never rendered — it exists so a caller can ask WHICH blocks a
    report carries and in what order without matching on the English inside
    them. Prose is the part of this module expected to be reworded, and a
    check that keys on prose constrains wording rather than behaviour; this
    file already deleted one banned-substring sweep for that reason, and the
    positive form of the same check has no better claim. Keying on the
    structure keeps the disclosure guarantees falsifiable: a section that
    stops being emitted, or is emitted on the wrong run, or lands below the
    thing it qualifies, fails — and a copy edit does not.
    """

    key: str
    lines: tuple[str, ...]

    @property
    def text(self) -> str:
        return '\n'.join(self.lines)


# ---------------------------------------------------------------------------
# The tripwire, split by census canonical presence
#
# Search cannot tell "the canonical does not exist" from "it exists but ranks
# below K"; the metadata census's deterministic count can. Joining the two is a
# diagnosis riding along a measurement, so an unreadable census is said out
# loud in the report and never aborts the metric emission.
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CensusTopicRow:
    """One ``registry_coverage.topics`` row of the metadata census."""

    project_id: str
    topic: str
    records: int
    canonical_count: int
    variant_spelling_records: int | None
    """``None`` on a census older than schema v5, which did not measure it."""


@dataclass(frozen=True)
class CensusCoverage:
    """The census's registry gauge, or why it could not be read."""

    source: str
    rows: Mapping[tuple[str, str], CensusTopicRow] | None
    """Keyed by ``(project_id, topic)``; ``None`` exactly when unavailable."""
    unavailable_reason: str | None
    query_surface_topics: tuple[str, ...] = ()
    """The registry topics the census deliberately did not gauge."""


def _census_count(raw: dict, key: str) -> int:
    value = raw.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise CensusShapeError(f'a registry_coverage row has a non-integer {key!r}: {raw!r}')
    return value


def _parse_census_row(raw: Any) -> CensusTopicRow:
    if not isinstance(raw, dict):
        raise CensusShapeError(f'a registry_coverage row is not an object: {raw!r}')
    project_id, topic = raw.get('project_id'), raw.get('topic')
    if not isinstance(project_id, str) or not isinstance(topic, str):
        raise CensusShapeError(f'a registry_coverage row lacks a project_id or topic: {raw!r}')
    return CensusTopicRow(
        project_id=project_id,
        topic=topic,
        records=_census_count(raw, 'records'),
        canonical_count=_census_count(raw, 'canonical_count'),
        variant_spelling_records=(
            None if raw.get('variant_spelling_records') is None
            else _census_count(raw, 'variant_spelling_records')
        ),
    )


def _parse_query_surface_topics(raw: Any) -> tuple[str, ...]:
    if raw is None:
        return ()
    if not isinstance(raw, list) or not all(
        isinstance(item, dict) and isinstance(item.get('topic'), str) for item in raw
    ):
        raise CensusShapeError(f'query_surface_topics_not_gauged is misshapen: {raw!r}')
    return tuple(item['topic'] for item in raw)


def _parse_registry_gauge(
    payload: Any,
) -> tuple[dict[tuple[str, str], CensusTopicRow], tuple[str, ...]]:
    if not isinstance(payload, dict):
        raise CensusShapeError('the top level is not an object')
    gauge = payload.get('registry_coverage')
    if gauge is None:
        raise CensusShapeError(
            'registry_coverage is null or absent; the census reported '
            f'registry_error={payload.get("registry_error")!r}',
        )
    if not isinstance(gauge, dict) or not isinstance(gauge.get('topics'), list):
        raise CensusShapeError('registry_coverage.topics is not a list')
    rows: dict[tuple[str, str], CensusTopicRow] = {}
    for raw in gauge['topics']:
        row = _parse_census_row(raw)
        key = (row.project_id, row.topic)
        if key in rows:
            raise CensusShapeError(f'registry_coverage.topics repeats {key!r}')
        rows[key] = row
    return rows, _parse_query_surface_topics(gauge.get('query_surface_topics_not_gauged'))


def load_census_coverage(path: str | Path) -> CensusCoverage:
    """Read the census's registry gauge at *path*. Never raises for a bad census.

    Every failure — unreadable, undecodable, or a gauge that is null or
    misshapen — becomes ``rows=None`` plus a reason naming *path*, so the report
    can say why the split is missing while the metrics are emitted regardless.
    """
    source = str(path)

    def unavailable(reason: str) -> CensusCoverage:
        return CensusCoverage(source=source, rows=None, unavailable_reason=reason)

    try:
        payload = json.loads(Path(path).read_text(encoding='utf-8'))
    except OSError as exc:
        return unavailable(f'cannot read the census at {source}: {exc}')
    except ValueError as exc:
        return unavailable(f'the census at {source} is not decodable JSON: {exc}')
    try:
        rows, query_surface = _parse_registry_gauge(payload)
    except CensusShapeError as exc:
        return unavailable(f'the census at {source} carries no usable registry gauge: {exc}')
    return CensusCoverage(
        source=source,
        rows=MappingProxyType(rows),
        unavailable_reason=None,
        query_surface_topics=query_surface,
    )


class CanonicalPresence(Enum):
    """What the census says about a tripwire topic's canonical."""

    CANONICAL_PRESENT = 'canonical-present'
    ABSENT_WITH_RECORDS = 'absent-with-records'
    ABSENT_UNPOPULATED = 'absent-unpopulated'
    QUERY_SURFACE = 'query-surface'
    NOT_CENSUSED = 'not-censused'


_PRESENCE_MEANING = MappingProxyType({
    CanonicalPresence.CANONICAL_PRESENT:
        'the census counts a canonical, so a failure is ranking, routing or fixture decay',
    CanonicalPresence.ABSENT_WITH_RECORDS:
        'records carry the topic but none is canonical: the census stamping_worklist',
    CanonicalPresence.ABSENT_UNPOPULATED:
        'no record under this exact spelling; records under a variant spelling are '
        'fused-memory/scripts/normalize_topic_slugs.py\'s to rewrite, not a stamping job',
    CanonicalPresence.QUERY_SURFACE:
        'keys a briefing query, not a metadata.topic, so the census does not gauge it',
    CanonicalPresence.NOT_CENSUSED:
        'the census has no row for this topic',
})


@dataclass(frozen=True)
class PhrasingRank:
    """Where one phrasing's search put the canonical, at the tripwire depth."""

    phrasing: str
    held_out: bool
    rank: int | None
    """Full-depth rank; ``None`` when the canonical was not in the fetched list."""
    matched_by: str | None
    stores_served: tuple[str, ...]


@dataclass(frozen=True)
class TopicPresence:
    """One tripwire item, classed by the census."""

    topic: str
    project_id: str
    passed: bool
    presence: CanonicalPresence
    census_row: CensusTopicRow | None
    phrasing_ranks: tuple[PhrasingRank, ...]
    """Recorded for a failing topic only: it is what distinguishes its causes."""


@dataclass(frozen=True)
class TripwireCensusSplit:
    """Every tripwire item of one run, each in exactly one :class:`CanonicalPresence`."""

    census_source: str
    unavailable_reason: str | None
    topics: tuple[TopicPresence, ...] = ()
    census_query_surface_topics: tuple[str, ...] = ()

    def by_presence(
        self, presence: CanonicalPresence, *, passed: bool,
    ) -> tuple[TopicPresence, ...]:
        return tuple(
            t for t in self.topics if t.presence is presence and t.passed is passed
        )


def _presence(entry: RegistryEntry, row: CensusTopicRow | None) -> CanonicalPresence:
    if entry.is_query_surface:
        return CanonicalPresence.QUERY_SURFACE
    if row is None:
        return CanonicalPresence.NOT_CENSUSED
    if row.canonical_count > 0:
        return CanonicalPresence.CANONICAL_PRESENT
    if row.records > 0:
        return CanonicalPresence.ABSENT_WITH_RECORDS
    return CanonicalPresence.ABSENT_UNPOPULATED


def split_tripwire_by_census(
    observations: ProbeObservations,
    registry: TopicRegistry,
    coverage: CensusCoverage,
) -> TripwireCensusSplit:
    """Class each tripwire item by what *coverage* says exists. Pure.

    The items and their verdicts are :func:`tripwire_items`' own, so the split
    qualifies the emitted tripwire and cannot disagree with it. An unavailable
    census classifies nothing rather than filing every topic as not censused.
    """
    if coverage.rows is None:
        return TripwireCensusSplit(
            census_source=coverage.source,
            unavailable_reason=coverage.unavailable_reason,
        )
    ranks: dict[str, list[PhrasingRank]] = {}
    for obs in observations.phrasings:
        if obs.degraded or obs.k != TRIPWIRE_K:
            continue
        ranks.setdefault(obs.topic, []).append(PhrasingRank(
            phrasing=obs.phrasing,
            held_out=obs.held_out,
            rank=obs.rank,
            matched_by=obs.matched_by,
            stores_served=obs.stores_served,
        ))
    entries_by_item = {entry.item_key: entry for entry in registry.entries}
    topics = []
    for item_key, passed in tripwire_items(observations, TRIPWIRE_K):
        entry = entries_by_item[item_key]
        row = coverage.rows.get((entry.project_id, entry.topic))
        topics.append(TopicPresence(
            topic=entry.topic,
            project_id=entry.project_id,
            passed=passed,
            presence=_presence(entry, row),
            census_row=row,
            phrasing_ranks=() if passed else tuple(ranks.get(entry.topic, ())),
        ))
    return TripwireCensusSplit(
        census_source=coverage.source,
        unavailable_reason=None,
        topics=tuple(topics),
        census_query_surface_topics=coverage.query_surface_topics,
    )


def _phrasing_rank_line(rank: PhrasingRank, fetched_depth: int) -> str:
    where = f'rank {rank.rank}' if rank.rank is not None else f'not in top-{fetched_depth}'
    return (
        f'{rank.phrasing!r} ({"held-out" if rank.held_out else "tuned"}): {where}; '
        f'matched by {rank.matched_by or "neither key"}; '
        f'served by {", ".join(rank.stores_served) or "nothing"}'
    )


_RANKED_PRESENCES = frozenset({
    CanonicalPresence.CANONICAL_PRESENT, CanonicalPresence.QUERY_SURFACE,
})
"""The classes whose canonical is known to exist — counted by the census, or
hand-adjudicated for a query surface — so a failure's per-phrasing ranks are
the diagnosis."""


def _failing_topic_lines(topic: TopicPresence, fetched_depth: int) -> list[str]:
    row = topic.census_row
    if topic.presence in _RANKED_PRESENCES:
        return [f'    - {topic.topic}'] + [
            f'        {_phrasing_rank_line(rank, fetched_depth)}'
            for rank in topic.phrasing_ranks
        ]
    if topic.presence is CanonicalPresence.ABSENT_WITH_RECORDS and row is not None:
        return [f'    - {topic.topic} (records: {row.records})']
    if topic.presence is CanonicalPresence.ABSENT_UNPOPULATED and row is not None:
        variant = (
            'not measured by this census' if row.variant_spelling_records is None
            else row.variant_spelling_records
        )
        return [f'    - {topic.topic} (variant-spelling records: {variant})']
    return [f'    - {topic.topic}']


def _tripwire_by_census_lines(split: TripwireCensusSplit, fetched_depth: int) -> list[str]:
    """The report block for *split*; *fetched_depth* is the deepest ``k`` searched."""
    lines = ['']
    lines.append(f'tripwire split by census canonical presence (census: {split.census_source}):')
    for chunk in _wrap(
        'The tripwire is measured through search, which cannot '
        'tell a canonical that does not exist from one that exists but ranks below '
        'K. The metadata census counts canonicals deterministically, so each item '
        'is classed here by what the census says exists; the counts behind each '
        'class are the census\'s registry_coverage, read there. For a failing '
        'topic whose canonical the census counts, or was hand-adjudicated for a '
        'briefing query, each phrasing\'s rank, matcher '
        'and serving store tell ranking (a rank below K), routing (served only by '
        'a store the canonical does not live in) and fixture decay (matched by '
        'last_known_id, or by neither key) apart.'
    ):
        lines.append(f'  {chunk}')
    if split.unavailable_reason is not None:
        lines.append(f'  census UNAVAILABLE, so no item is classified: {split.unavailable_reason}')
        return lines
    for presence in CanonicalPresence:
        failed = split.by_presence(presence, passed=False)
        passed = split.by_presence(presence, passed=True)
        lines.append(f'  {presence.value} (failed {len(failed)}, passed {len(passed)}):')
        lines.extend(f'    {chunk}' for chunk in _wrap(_PRESENCE_MEANING[presence], width=72))
        if presence is CanonicalPresence.QUERY_SURFACE:
            lines.append(
                '    census-excluded as query surfaces: '
                f'{", ".join(split.census_query_surface_topics) or "none named by this census"}'
            )
        for topic in failed:
            lines.extend(_failing_topic_lines(topic, fetched_depth))
        if passed:
            chunks = _wrap(', '.join(t.topic for t in passed), width=64)
            lines.append(f'    passed: {chunks[0]}')
            lines.extend(f'            {chunk}' for chunk in chunks[1:])
    return lines


def probe_report_sections(
    series,
    observations: ProbeObservations,
    *,
    is_initial_run: bool = False,
    registry: TopicRegistry | None = None,
    skipped_topics: tuple[str, ...] = (),
    requested_ks: tuple[int, ...] = (),
    measured_ks: tuple[int, ...] = (),
    census_split: TripwireCensusSplit | None = None,
) -> tuple[ReportSection, ...]:
    """The report, decomposed — see :func:`render_probe_report` for the prose.

    THE single source of both: :func:`render_probe_report` joins what this
    returns, so a section present here is present there by construction and
    the two cannot drift into disagreeing about what the run disclosed.

    *census_split* is taken already computed, never derived here, so the report
    renders the very split the caller hands back to its own callers.
    """
    from shared.memory_eval_metrics import render_report  # noqa: PLC0415

    sections: list[ReportSection] = []

    def emit(key: str, lines: list[str]) -> None:
        sections.append(ReportSection(key=key, lines=tuple(lines)))

    emit(SECTION_METRIC_TABLE, [render_report(series).rstrip('\n')])

    # The depths this run actually scored, when the caller did not say.
    # Assuming DEFAULT_KS instead would name canonical-in-top-10 as unmeasured
    # on a run that never asked for k=10 — a false narrowing report is no
    # better than a hidden one.
    scored_ks = tuple(measured_ks) or tuple(dict.fromkeys(o.k for o in observations.phrasings))

    # Directly under the table it qualifies: a family missing from the rows
    # above is the one absence a reader cannot see by reading them.
    absent_families = metric_families_not_measured(series, scored_ks)
    if absent_families:
        lines = ['']
        lines.append(f'metric families NOT MEASURED this run ({len(absent_families)}):')
        for chunk in _wrap(
            'These metrics had no scored trial, so they were not emitted — a '
            'proportion over zero trials and a count over zero exposure are '
            'not measurements, and emitting either would put a fabricated '
            'datapoint into the baseline window the limits evaluator computes '
            'from. Their absence is NOT a pass: the evaluator joins a run to '
            'its baseline BY metric_id and simply stops trending what is '
            'missing, so the gap is said here rather than left to be inferred '
            'from which rows happen to be present.'
        ):
            lines.append(f'  {chunk}')
        lines.extend(f'  - {metric_id}' for metric_id in absent_families)
        emit(SECTION_FAMILIES_NOT_MEASURED, lines)

    if is_initial_run:
        failing = [
            item.item_key
            for metric in series.metrics
            if metric.metric_id == METRIC_TOPIC_CANONICAL_PRESENT
            for item in (metric.items or [])
            if not item.passed
        ]
        header = ['']
        header.append(f'INITIAL STATE — known-bad items ({len(failing)}):')
        for chunk in _wrap(_KNOWN_BAD_PREAMBLE):
            header.append(f'  {chunk}')
        emit(SECTION_INITIAL_STATE, header)

        # The routing caveat is repeated HERE, not left to the store breakdown
        # ~150 lines below, because this is the section an operator reads on the
        # one run it matters for. The first live baseline came back 72/78
        # observations served by Graphiti and 72/78 unmatched; a reader who
        # stops at the headline rate takes a router property for a corpus-wide
        # findability collapse (esc-3208-1). It is its OWN section so that
        # placement — above the item list it qualifies — is a structural fact a
        # caller can check, rather than something only a prose search can see.
        caveat = ['']
        for chunk in _wrap(_KNOWN_BAD_ROUTING_CAVEAT):
            caveat.append(f'  {chunk}')
        emit(SECTION_KNOWN_BAD_ROUTING_CAVEAT, caveat)

        items = ['']
        items.extend(f'  - {key}' for key in failing)
        if not failing:
            items.append('  (no tripwire item is failing in this initial run)')
        emit(SECTION_KNOWN_BAD_ITEMS, items)

    if observations.degraded_queries:
        lines = ['']
        lines.append(f'degraded queries ({len(observations.degraded_queries)}):')
        lines.append(
            '  These searches reported a failed store. Their observations are '
            'EXCLUDED from every denominator above — an outage is not a '
            'retrieval regression.'
        )
        for record in observations.degraded_queries:
            stores = ', '.join(record.failed_stores) or 'unnamed store'
            lines.append(f'  - [{record.topic}] {record.query!r}: failed stores: {stores}')
            for diagnostic in record.diagnostics:
                rendered = ', '.join(
                    f'{k}={v}' for k, v in sorted(diagnostic.items())
                )
                lines.append(f'      {rendered}')
        emit(SECTION_DEGRADED_QUERIES, lines)

    unmeasured = not_measured_topics(observations)
    if unmeasured:
        lines = ['']
        lines.append(f'topics NOT MEASURED this run ({len(unmeasured)}):')
        lines.append(
            '  Every phrasing degraded, so these topics carry no tripwire item. '
            'They are not passing and not failing — they were not measured.'
        )
        lines.extend(f'  - {topic}' for topic in unmeasured)
        emit(SECTION_TOPICS_NOT_MEASURED, lines)

    if census_split is not None:
        emit(SECTION_TRIPWIRE_BY_CENSUS, _tripwire_by_census_lines(
            census_split, max(scored_ks, default=TRIPWIRE_K),
        ))

    unmatched_stores: dict[str, set[str]] = {}
    for obs in observations.phrasings:
        if obs.degraded or not obs.unmatched:
            continue
        unmatched_stores.setdefault(obs.topic, set()).update(obs.stores_served)
    unmatched = sorted(unmatched_stores)
    if unmatched:
        lines = ['']
        lines.append(f'canonicals matched by NEITHER key ({len(unmatched)}):')
        for chunk in _wrap(
            'Neither the content hash nor last_known_id matched anything '
            'returned. Either the entry is genuinely unfindable, the query '
            'was routed to a store the entry does not live in, or the '
            'registry fixture has decayed past both keys — the serving store '
            'is named per topic so an operator can tell those apart rather '
            'than guessing between them.'
        ):
            lines.append(f'  {chunk}')
        lines.extend(
            f'  - {topic} (served by: {", ".join(sorted(unmatched_stores[topic])) or "nothing"})'
            for topic in unmatched
        )
        emit(SECTION_UNMATCHED_CANONICALS, lines)

    repairs = sorted({
        obs.topic for obs in observations.phrasings
        if not obs.degraded and obs.needs_hash_repair
    })
    if repairs:
        lines = ['']
        lines.append(f'canonicals matched by last_known_id only ({len(repairs)}):')
        lines.append(
            '  The content hash missed but the id hit: the stored text changed. '
            'Counted as a hit (the entry IS findable) and reported so the '
            'fixture can be re-hashed — a probe that silently rewrote its own '
            'expectations could never fail.'
        )
        lines.extend(f'  - {topic}' for topic in repairs)
        emit(SECTION_HASH_REPAIRS, lines)

    missing_claims = [
        obs for obs in observations.claims
        if not obs.degraded and obs.scorable and not obs.recalled
    ]
    if missing_claims:
        lines = ['']
        lines.append(f'claims not recalled ({len(missing_claims)}):')
        for obs in missing_claims:
            needles = ', '.join(repr(n) for n in obs.missing_needles)
            lines.append(f'  - [{obs.topic}] {obs.query!r}: missing {needles}')
        emit(SECTION_CLAIMS_NOT_RECALLED, lines)

    scored = [obs for obs in observations.phrasings if not obs.degraded]
    if scored:
        by_matcher: dict[str, int] = {
            MATCHED_BY_CONTENT_HASH: 0, MATCHED_BY_LAST_KNOWN_ID: 0, 'unmatched': 0,
        }
        for obs in scored:
            by_matcher[obs.matched_by or 'unmatched'] += 1
        lines = ['']
        lines.append('how the canonical was matched (observations):')
        for chunk in _wrap(
            'The registry keys a canonical by content hash first because memory '
            'UUIDs rot on re-consolidation, with last_known_id as a disclosed '
            'fallback. Which matcher fired is a fixture-health signal: a run '
            'drifting toward last_known_id needs re-hashing, and unmatched '
            'entries are either genuinely gone or keyed past both.'
        ):
            lines.append(f'  {chunk}')
        for matcher, count in by_matcher.items():
            lines.append(f'  {matcher}: {count}')
        emit(SECTION_MATCHED_BY, lines)

        by_store: dict[str, int] = {}
        for obs in scored:
            for store in obs.stores_served:
                by_store[store] = by_store.get(store, 0) + 1
        lines = ['']
        lines.append('which store served the query (observations):')
        for chunk in _wrap(
            'MemoryService.search routes: the read router picks a store set '
            'per query, and the lists come back homogeneous. A phrasing served '
            'entirely by Graphiti cannot contain a Mem0 entry\'s raw content — '
            'Graphiti returns LLM-extracted edge facts — so its canonical is '
            'unfindable there however healthy retrieval is. The probe pins '
            'stores ONLY for a topic that declares the scope its real caller '
            'searches with (search_scope; today the briefing channels) and '
            'otherwise leaves routing to the router, because an agent\'s own '
            'search is routed too; it reports the routing instead, so a rate '
            'is never read as a corpus finding when it is a routing one.'
        ):
            lines.append(f'  {chunk}')
        for store, count in sorted(by_store.items()):
            lines.append(f'  {store}: {count}')
        scoped_topics = [
            (entry.topic, entry.search_scope)
            for entry in (registry.entries if registry is not None else ())
            if entry.search_scope is not None
        ]
        if scoped_topics:
            lines.append('  topics searched with a declared scope:')
        for topic, scope in scoped_topics:
            lines.append(
                f'  - {topic}: stores={", ".join(scope.stores) or "router"}; '
                f'categories={", ".join(scope.categories) or "any"}'
            )
        emit(SECTION_STORES_SERVED, lines)

    contamination = [c for c in observations.contamination if not c.degraded]
    if contamination:
        untopiced = sum(c.untopiced_count for c in contamination)
        total = sum(c.scored_total for c in contamination)
        foreign = sum(c.foreign_count for c in contamination)
        lines = ['']
        lines.append('contamination classification:')
        lines.append(f'  scored results: {total}')
        lines.append(f'  foreign (a DIFFERENT registered topic): {foreign}')
        lines.append(f'  untopiced (no topic, or one this registry does not know): {untopiced}')
        for chunk in _wrap(
            'Untopiced results are NOT counted as contamination. The census '
            'measured 491 of 49,628 entries carrying a topic at all, so '
            'treating an unstamped result as foreign would measure stamping '
            'coverage while claiming to measure contamination. As 3195/3201 '
            'widen the vocabulary this remainder shrinks and the share reaches '
            'further, with no change to how it is computed.'
        ):
            lines.append(f'  {chunk}')
        emit(SECTION_CONTAMINATION, lines)

    graphiti_primary = ', '.join(graphiti_primary_categories())
    lines = ['']
    lines.append('what the corpus counts cover:')
    for chunk in _wrap(
        'The per-category sizes above come from count_memories_by_metadata, '
        'which is an exact Qdrant payload count — a MEM0-SIDE count. The '
        f'Graphiti-primary categories ({graphiti_primary}) therefore read near '
        'zero here even when the graph holds thousands: the number is honest '
        'about what was counted and would be misleading about what exists, so '
        'it says which it is. The metrics above are unaffected — they are '
        'computed over what search returned, not over these counts.'
    ):
        lines.append(f'  {chunk}')
    emit(SECTION_CORPUS_COUNT_SCOPE, lines)

    if measured_ks and tuple(requested_ks) != tuple(measured_ks):
        added = ', '.join(
            str(k) for k in measured_ks if k not in set(requested_ks)
        )
        lines = ['']
        lines.append(
            'measurement depth '
            f'(requested {", ".join(str(k) for k in requested_ks) or "nothing"}; '
            f'measured {", ".join(str(k) for k in measured_ks)}):'
        )
        for chunk in _wrap(
            f'--k selects the depths canonical-in-top-k is scored at. k={added} '
            'was added to this run because two metrics are DEFINED at it and '
            'cannot be computed without it: the '
            f'{METRIC_TOPIC_CANONICAL_PRESENT} tripwire and '
            f'{METRIC_CANONICAL_IN_TOP_K_HELD_OUT.format(k=TRIPWIRE_K)}. A run '
            'that dropped them would not fail — leaf α joins a run to its '
            'baseline window by metric_id, so they would simply stop being '
            'trended. Measuring a depth nobody asked for is a narrowing like '
            'any other, so it is said here rather than left to be inferred '
            'from which metrics happen to be present.'
        ):
            lines.append(f'  {chunk}')
        emit(SECTION_MEASUREMENT_DEPTH, lines)

    if skipped_topics:
        lines = ['']
        lines.append(f'topics NOT PROBED this run ({len(skipped_topics)}):')
        for chunk in _wrap(
            'These registry entries belong to a project this run did not '
            'select with --project-id. They are absent from every metric '
            'above. Said out loud because a narrowed run and a shrunken '
            'registry produce the same numbers, and only one of them is a '
            'measurement problem.'
        ):
            lines.append(f'  {chunk}')
        lines.extend(f'  - {topic}' for topic in skipped_topics)
        emit(SECTION_TOPICS_NOT_PROBED, lines)

    if registry is not None:
        composition: dict[str, int] = {}
        for entry in registry.entries:
            composition[entry.derived_from] = composition.get(entry.derived_from, 0) + 1
        lines = ['']
        lines.append(f'registry composition ({len(registry.entries)} topics):')
        for source, count in sorted(composition.items()):
            lines.append(f'  {source}: {count}')
        if registry.disclosures:
            # "Left out" alone would mislabel the emitted-per-source counters
            # that share this block; they are what the narrowings are a
            # fraction OF, and both halves are needed to read either.
            lines.append('  derivation ledger — what it produced, and what it left out:')
            for key, value in sorted(registry.disclosures.items()):
                lines.append(f'    {key}: {value}')
        emit(SECTION_REGISTRY_COMPOSITION, lines)

    inversion_records = [
        record
        for obs in observations.inversions if not obs.degraded
        for record in obs.inversions
    ]
    if inversion_records:
        lines = ['']
        lines.append(f'superseded entries outranking their successor ({len(inversion_records)}):')
        for record in inversion_records:
            lines.append(
                f'  - [{record.topic}] {record.phrasing!r}: '
                f'{record.superseded_hash} (rank {record.superseded_rank}) above '
                f'{record.successor_hash} (rank {record.successor_rank})'
            )
        emit(SECTION_SUPERSEDED_INVERSIONS, lines)

    return tuple(sections)


def render_probe_report(
    series,
    observations: ProbeObservations,
    *,
    is_initial_run: bool = False,
    registry: TopicRegistry | None = None,
    skipped_topics: tuple[str, ...] = (),
    requested_ks: tuple[int, ...] = (),
    measured_ks: tuple[int, ...] = (),
    census_split: TripwireCensusSplit | None = None,
) -> str:
    """The prose companion: the shared metric table plus this run's caveats.

    Every section here exists because a number alone would mislead. A
    contamination share without its unclassifiable remainder reads as
    authoritative; a tripwire missing an item reads as a pass; a canonical that
    matched by ``last_known_id`` reads as healthy while the fixture quietly
    rots. None of those are visible in the metrics table, so they are said here
    in words.

    *is_initial_run* defaults to False so a caller that forgot to detect it
    cannot fabricate a first run — a known-bad snapshot printed on run fifty
    would excuse a real regression as inherited state.

    Nothing here adjudicates. No bound, no ratchet and no pass/fail verdict
    appears in this output, because all of that belongs to leaf α's evaluator
    (G6/D1) and a second home for it would drift from the first.

    A pure join of :func:`probe_report_sections` — which sections exist, and
    under what conditions, lives there and only there.
    """
    return join_report_sections(probe_report_sections(
        series,
        observations,
        is_initial_run=is_initial_run,
        registry=registry,
        skipped_topics=skipped_topics,
        requested_ks=requested_ks,
        measured_ks=measured_ks,
        census_split=census_split,
    ))


def join_report_sections(sections: tuple[ReportSection, ...]) -> str:
    """Render *sections* to the report text an operator reads.

    Each section carries its own leading blank line, so the join is a plain
    concatenation — there is no separator policy here that could disagree
    with what a section believes its own shape is.
    """
    return '\n'.join(line for section in sections for line in section.lines) + '\n'


def write_report_text(path: str | Path, text: str) -> None:
    """Replace the report at *path* with *text*, atomically.

    A plain ``write_text`` here would be a truncate-and-write over the file
    :func:`shared.memory_eval_metrics.write_metric_series` had just written
    atomically — and that module's ``_atomic_write_text`` exists precisely
    because the memory-eval leaves (β/γ/δ) share one artifact root and the
    dashboard reads these as plain files. A crash, an ENOSPC or a concurrent
    reader mid-write would leave a truncated report beside a valid metrics
    artifact, which is the one state the shared module's atomicity was
    designed to exclude. Widening the report must not reopen the hole.

    The mechanism is copied rather than imported: ``_atomic_write_text`` is
    module-private in ``shared`` and this leaf holds no lock on that package.
    :func:`tempfile.mkstemp` gives an OS-guaranteed fresh, exclusively-created
    sibling — not a pid-derived name, which concurrent writers under the
    shared root could collide on. A reader sees either the old contents or
    the complete new ones; a failed write unlinks the temp and leaves nothing.
    """
    path = Path(path)
    fd, tmp_name = tempfile.mkstemp(
        suffix='.tmp', prefix=f'{path.name}.', dir=str(path.parent),
    )
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as handle:
            handle.write(text)
        os.replace(tmp_name, path)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(tmp_name)
        raise


def emit_series(series, root: str | Path, *, stamp: str | None = None) -> tuple[Path, Path]:
    """Write *series* under *root*, returning ``(metrics_path, report_path)``.

    A thin pass-through to :func:`shared.memory_eval_metrics.write_metric_series`
    — the layout, the atomic write and the emit-time validation are all the
    shared module's, and re-implementing any of them here would be a second
    home for the artifact contract.
    """
    from shared.memory_eval_metrics import write_metric_series  # noqa: PLC0415

    return write_metric_series(series, root, stamp=stamp)


# ---------------------------------------------------------------------------
# The read-only run band
#
# D8's runner pattern (audit_duplicate_memories.py:364-378) minus every
# mutation: CONFIG_PATH from --config, FusedMemoryConfig(), MemoryService(),
# initialize_for_reads(), try/finally close(); the full initialize() starts the
# durable write queue and runs Graphiti startup maintenance, which repairs
# edges. MemoryService rather than Mem0Backend
# because a retrieval probe MUST embed its queries — unlike the census, which
# skips the embedder deliberately, this one cannot.
#
# There is no --apply band and no write call. The guarantee is asserted as
# behaviour: the tests drive this band against a service double whose every
# write method raises, and a run that completes is a run that never wrote.
# ---------------------------------------------------------------------------

_SCRIPT_DIR = Path(__file__).resolve().parent
_PACKAGE_ROOT = _SCRIPT_DIR.parent
_REPO_ROOT = _PACKAGE_ROOT.parent

DEFAULT_PROJECT_IDS: tuple[str, ...] = ('dark_factory', 'reify')
"""The two projects the metadata census enumerated; the same pair by default here.

Following the census's precedent matters beyond convenience: the registry's
topics were derived from that census, so probing a different set of projects
would measure a corpus the fixture was never built against.
"""

DEFAULT_REGISTRY_PATH = _PACKAGE_ROOT / 'tests' / 'fixtures' / 'memory_eval_topic_registry.json'
"""The committed registry. It lives under tests/fixtures because it IS a fixture —
hand-completed, reviewed, and version-controlled so a run is reproducible."""

DEFAULT_OUT_ROOT = _PACKAGE_ROOT / 'data' / 'memory-evals'
"""Artifact root. ``data/`` is gitignored (fused-memory/.gitignore:9), so a run's
output never lands in a diff by accident.

That cuts BOTH ways, and the second half is the one that bites: because the
tree is ignored, a run here is preserved by nothing — not by merging, not by a
clone, not by anything surviving ``git clean -xdf``. It lives on exactly one
host until that host's disk is wiped. A run worth keeping must be copied OUT to
a tracked path; ``plans/memory-eval-e1-first-live-run/`` is the frozen copy of
the first live run and the precedent for where the next one goes (same
``plans/`` home and same idiom as :data:`DEFAULT_CENSUS_PATH` below, this
repo's existing convention for a committed real-measurement artifact). Task
3694 exists because that copy-out step was skipped for the first live run,
leaving three PRD provenance citations pointing at a path their readers could
not open.

Copy runs OUT; never un-ignore this tree to keep one IN. :func:`is_initial_run`
globs ``metrics-*.json`` under here, so a committed artifact inside this root
would make every fresh clone look like it had already run and would permanently
burn the one-shot initial-state snapshot for the next genuine first run —
destroying evidence rather than merely hiding it. That the frozen copy stays
disjoint from this root is asserted in
``tests/test_memory_eval_e1_first_live_run.py``, so it cannot be undone by a
later tidy-up; read that directory's README before relocating it.
"""

DEFAULT_CALIBRATION_PATH = _PACKAGE_ROOT / 'tests' / 'fixtures' / 'write_triage_calibration.jsonl'
DEFAULT_CENSUS_PATH = _REPO_ROOT / 'plans' / 'memory-metadata-census-report.json'

EXIT_RUN_FAILED = 1
"""The run did not finish: the store could not be opened, the artifacts could
not be written, a derivation source could not be read, or stdout failed.

Agrees with ``shared.cli_boundary.EXIT_STDOUT_FAILED``.
"""

EXIT_BAD_INPUT = 2
"""Nothing can be measured until the input is fixed: the registry did not load,
``--project-id`` selected nothing, or ``--derive-registry``'s census is
undecodable or carries no usable topic table.

A census file that cannot be read at all, such as a wrong path, is
:data:`EXIT_RUN_FAILED` instead, like any unreadable derivation source."""


def graphiti_primary_categories() -> tuple[str, ...]:
    """The categories the Mem0-side corpus count under-reports.

    Named in the report so an operator reading a near-zero count knows it is a
    counting scope, not an empty graph. Derived for the same reason
    ``fused_memory/eval/topic_registry.py::corpus_categories`` is.
    """
    from fused_memory.models.enums import GRAPHITI_PRIMARY  # noqa: PLC0415

    return tuple(sorted(c.value for c in GRAPHITI_PRIMARY))


def corpus_project_id(project_ids: tuple[str, ...]) -> str:
    """The single ``Corpus.project_id`` for a run covering *project_ids*.

    M1's Corpus carries one project id and this runner emits one artifact per
    stamp, so a multi-project run needs a stable joined identifier. Single
    project in, that project out — which is the exemplar's shape and the shape
    an ephemeral (test-collection) run produces, so a seeded run is never
    mistakable for a live one.
    """
    return '+'.join(project_ids)


async def count_corpus(memory, project_ids: tuple[str, ...]) -> dict[str, int]:
    """Corpus size per category, summed over *project_ids*.

    One ``count_memories_by_metadata`` call per (project, category): an exact
    Qdrant count rather than a top-N-bounded search, because this is the
    denominator behind the denominators and a sampled one would quietly
    rescale every proportion in the artifact.
    """
    counts: dict[str, int] = {}
    for project_id in project_ids:
        for category in corpus_categories():
            counts[category] = counts.get(category, 0) + await memory.count_memories_by_metadata(
                project_id, {'category': category},
            )
    return counts


@contextlib.contextmanager
def _artifact_write_seam(out_root: str | Path):
    """Attribute an ``OSError`` from the wrapped artifact write to *out_root*."""
    try:
        yield
    except OSError as exc:
        raise ProbeRunError(f'cannot write the probe artifacts under {out_root}: {exc}') from exc


@dataclass(frozen=True)
class ProbeOutcome:
    """Everything one run produced, for a caller that wants more than an exit code."""

    series: Any
    observations: ProbeObservations
    metrics_path: Path
    report_path: Path
    report: str
    sections: tuple[ReportSection, ...]
    """The report's blocks under their machine keys — ``report`` is their join.

    Carried so a caller can ask what this run DISCLOSED (was the added depth
    said out loud? did the initial-state snapshot fire?) without pattern
    matching English out of the rendered text.
    """
    is_initial_run: bool
    skipped_topics: tuple[str, ...]
    corpus_counts: dict[str, int]
    census_split: TripwireCensusSplit | None
    """The tripwire split by census canonical presence; ``None`` when no census
    was given. The very split the report rendered."""


async def run_probe(
    memory,
    registry: TopicRegistry,
    *,
    project_ids: tuple[str, ...],
    ks: tuple[int, ...],
    out_root: str | Path,
    stamp: str | None = None,
    census: CensusCoverage | None = None,
) -> ProbeOutcome:
    """Measure *registry* against *memory* and emit the run's artifacts.

    *census*, when given, splits the tripwire by census canonical presence in
    the report; it qualifies the metrics and never changes them.

    *memory* is injected rather than constructed here, which is what lets the
    read-only guarantee be tested: the whole band runs against a double whose
    write methods raise.

    Reads only. ``search`` and ``count_memories_by_metadata`` are the only two
    methods touched.

    THE chokepoint for *ks*: normalising here means every caller — ``main``,
    ``_probe``, a test, a notebook — inherits the guarantee that the pinned
    metrics are emitted, so no path can produce an artifact missing them.
    """
    measured_ks = normalise_ks(tuple(ks))
    selected = tuple(dict.fromkeys(project_ids))
    wanted = set(selected)
    probed = tuple(e for e in registry.entries if e.project_id in wanted)
    skipped = tuple(e.topic for e in registry.entries if e.project_id not in wanted)

    # Before any store access, and before any artifact: a selection that matched
    # nothing would otherwise emit `"metrics": []` and exit 0 — silence the
    # evaluator cannot distinguish from a clean run, and a file is_initial_run
    # counts. Same abort as a failed registry load, different door in.
    if not probed:
        available = sorted({e.project_id for e in registry.entries})
        raise EmptySelectionError(
            f'no topic registry entry matches project_id(s) {list(selected)!r}; '
            f'the registry carries {available!r} across {len(registry.entries)} '
            'entries. Emitting nothing: an artifact with zero metrics is '
            'indistinguishable downstream from a healthy run.',
        )

    observations = ProbeObservations()
    for entry in probed:
        # Default-arg binding, so the closure cannot late-bind the loop variable.
        async def search(query: str, limit: int, _entry: RegistryEntry = entry):
            scope = _entry.search_scope.as_search_kwargs() if _entry.search_scope else {}
            return await memory.search(
                query, project_id=_entry.project_id, limit=limit, **scope,
            )

        # The FULL registry, not the probed subset: contamination asks whether
        # a result belongs to a different KNOWN topic, so the widest topic
        # vocabulary available gives the truest answer. Narrowing it here would
        # silently reclassify foreign results as untopiced.
        await probe_topic(search, entry, registry, measured_ks, observations)

    counts = await count_corpus(memory, selected)

    if stamp is None:
        from shared.memory_eval_metrics import run_stamp  # noqa: PLC0415

        stamp = run_stamp()

    # BEFORE emitting: this run's own artifact would otherwise make its first
    # run look like its second and suppress the D1 initial-state snapshot.
    initial = is_initial_run(out_root)

    series = build_series(
        observations, counts, corpus_project_id(selected), stamp, measured_ks,
    )
    with _artifact_write_seam(out_root):
        metrics_path, report_path = emit_series(series, out_root, stamp=stamp)

    # emit_series wrote the shared render_report as the companion; replace it
    # with the extended one — through the same atomic path, so the widening
    # cannot leave a truncated report beside a valid metrics artifact. The
    # shared write still happens first, so emit-time validation continues to
    # gate whether any artifact is created at all — the report is only ever
    # widened over a series that already validated.
    # The PROBED subset here, so "registry composition (N topics)" counts
    # what this run actually measured; the rest is named by skipped_topics.
    probed_registry = TopicRegistry(
        schema_version=registry.schema_version,
        entries=probed,
        disclosures=registry.disclosures,
    )
    census_split = (
        split_tripwire_by_census(observations, probed_registry, census)
        if census is not None else None
    )
    sections = probe_report_sections(
        series,
        observations,
        is_initial_run=initial,
        registry=probed_registry,
        skipped_topics=skipped,
        requested_ks=tuple(ks),
        measured_ks=measured_ks,
        census_split=census_split,
    )
    report = join_report_sections(sections)
    with _artifact_write_seam(out_root):
        write_report_text(report_path, report)

    return ProbeOutcome(
        series=series,
        observations=observations,
        metrics_path=metrics_path,
        report_path=report_path,
        report=report,
        sections=sections,
        is_initial_run=initial,
        skipped_topics=skipped,
        corpus_counts=counts,
        census_split=census_split,
    )


class _ReplacingAppend(argparse.Action):
    """``append``, except the first explicit value REPLACES the default.

    Plain ``action='append'`` with a default appends to it, so
    ``--project-id reify`` would probe dark_factory AND reify — two extra
    projects nobody asked for, and on a seeded ephemeral run that would mean
    reaching into the live corpus. The identity check against ``self.default``
    is what makes the swap exact and stateless across repeated parses.
    """

    def __call__(self, parser, namespace, values, option_string=None):
        current = getattr(namespace, self.dest, None)
        if current is self.default or current is None:
            current = ()
        setattr(namespace, self.dest, (*current, values))


def build_parser() -> LoudArgumentParser:
    """The CLI. Every flag here is a read parameter; none of them mutate anything.

    There is deliberately no ``--apply``, ``--fix``, ``--prune`` or any other
    mutating band, and a test asserts this flag set by EQUALITY so one cannot
    be added without a test saying so out loud.
    """
    # The first line plus the guarantee is what an operator at the terminal
    # actually needs; the rest of the module docstring is for source readers.
    parser = LoudArgumentParser(
        description=(
            f'{(__doc__ or EVAL_ID).splitlines()[0]}\n\n'
            'This script never writes to the live corpus and never evaluates a\n'
            'limit — it emits measurements for the limits evaluator to judge.'
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        '--project-id', dest='project_id', action=_ReplacingAppend,
        default=DEFAULT_PROJECT_IDS, metavar='PROJECT_ID',
        help=(
            'Project to probe; repeatable. Selects which registry entries run. '
            f'Default: {" ".join(DEFAULT_PROJECT_IDS)}'
        ),
    )
    parser.add_argument(
        '--registry', default=str(DEFAULT_REGISTRY_PATH),
        help=f'Topic registry JSON (default: {DEFAULT_REGISTRY_PATH})',
    )
    parser.add_argument(
        '--out-root', dest='out_root', default=str(DEFAULT_OUT_ROOT),
        help=f'Artifact root (default: {DEFAULT_OUT_ROOT})',
    )
    parser.add_argument(
        '--k', dest='k', action=_ReplacingAppend, type=int, default=DEFAULT_KS,
        metavar='K',
        help=(
            'Depth to score canonical-in-top-k at; repeatable. A metric '
            f'parameterisation, not a threshold. Default: {" ".join(str(k) for k in DEFAULT_KS)}'
        ),
    )
    parser.add_argument(
        '--config', default=None,
        help='Path to fused-memory config file (sets CONFIG_PATH env var)',
    )
    parser.add_argument(
        '--census', default=str(DEFAULT_CENSUS_PATH),
        help=(
            'Metadata census JSON, read for the tripwire split by canonical '
            'presence and by --derive-registry. An unreadable census is named '
            'in the report, never fatal to a probe run. --derive-registry '
            'refuses a census that is undecodable or has no usable topic '
            'table (exit 2), and fails on one it cannot read at all, such as '
            f'a wrong path (exit 1) (default: {DEFAULT_CENSUS_PATH})'
        ),
    )
    parser.add_argument(
        '--derive-registry', dest='derive_registry', action='store_true',
        help=(
            'Print registry candidates derived from the committed offline '
            'sources and exit. Never overwrites the committed fixture — the '
            'hand-authored held-out phrasings cannot be regenerated.'
        ),
    )
    return parser


def _derive_registry_text(census_path: str | Path) -> str:
    """``--derive-registry``'s output, with an unreadable source attributed to it."""
    try:
        return run_derive_registry(DEFAULT_CALIBRATION_PATH, census_path)
    except OSError as exc:
        raise ProbeRunError(f'cannot read a registry derivation source: {exc}') from exc


async def _probe(args: argparse.Namespace, registry: TopicRegistry) -> ProbeOutcome:
    """Open the store, measure *registry* against it, and close it again.

    Only the open is converted to :class:`ProbeRunError`; see
    ``shared/src/shared/cli_boundary.py::run_cli``.
    """
    from fused_memory.config.schema import FusedMemoryConfig  # noqa: PLC0415
    from fused_memory.services.memory_service import MemoryService  # noqa: PLC0415

    if args.config:
        os.environ['CONFIG_PATH'] = str(args.config)

    memory = MemoryService(FusedMemoryConfig())
    try:
        try:
            await memory.initialize_for_reads()
        except (OSError, sqlite3.Error) as exc:
            raise ProbeRunError(
                f'the memory store could not be opened ({type(exc).__name__}): {exc}'
            ) from exc
        return await run_probe(
            memory, registry,
            project_ids=tuple(args.project_id),
            ks=tuple(args.k),
            out_root=args.out_root,
            census=load_census_coverage(args.census),
        )
    finally:
        await memory.close()


def main(argv: list[str] | None = None) -> int:
    """Parse, run, print. Every stdout print sits outside every handled failure,
    so a stdout failure always reaches ``shared/src/shared/cli_boundary.py::run_cli``."""
    outcome: ProbeOutcome | None = None

    def _written_artifacts() -> str | None:
        if outcome is None:
            return None
        return (
            f'the metrics were written to {outcome.metrics_path} '
            f'and the report to {outcome.report_path}'
        )

    reset_stdout_failure_state(detail=_written_artifacts)
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s',
    )
    try:
        if args.derive_registry:
            text = _derive_registry_text(args.census)
        else:
            # Before the store, deliberately. A fixture typo must not cost an
            # embedder spin-up to discover, and — the load-bearing half — a
            # registry that failed to load must never reach emission: an
            # artifact reporting zero topics is indistinguishable, downstream,
            # from a healthy corpus that found nothing.
            registry = load_topic_registry(args.registry)
            outcome = asyncio.run(_probe(args, registry))
            text = outcome.report
    except (RegistryError, EmptySelectionError, CensusShapeError) as exc:
        print(str(exc), file=sys.stderr)
        return EXIT_BAD_INPUT
    except ProbeRunError as exc:
        print(f'error: {exc}', file=sys.stderr)
        return EXIT_RUN_FAILED

    print(text, end='')
    if outcome is not None:
        logger.info('metrics: %s', outcome.metrics_path)
        logger.info('report:  %s', outcome.report_path)
    return 0


if __name__ == '__main__':
    sys.exit(run_cli(main))
