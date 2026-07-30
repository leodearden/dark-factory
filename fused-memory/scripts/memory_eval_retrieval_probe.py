#!/usr/bin/env python3
"""E1 retrieval-health probe — does the memory system return the right thing?

``docs/prds/memory-eval-program.md`` §5 leaf β. The 3111/3112 fix lineage
(canonical pinning, consolidation, curator gates) changed how the corpus is
written; nothing measures whether *retrieval* actually improved. This runner
probes a committed registry of topics with several query phrasings each and
emits an M1 metric series recording what came back.

**What it measures** (seven metrics, all owned by this leaf):

======================================  ==========  ==================
metric_id                               kind        direction
======================================  ==========  ==================
``topic-canonical-present``             tripwire    (rule (a) is already
                                                    directional)
``canonical-in-top-5``                  proportion  lower_is_worse
``canonical-in-top-10``                 proportion  lower_is_worse
``canonical-in-top-5-held-out``         proportion  lower_is_worse
``claim-recall``                        proportion  lower_is_worse
``contamination-share``                 proportion  higher_is_worse
``superseded-above-successor``          count       higher_is_worse
======================================  ==========  ==================

``dangling-pointers`` and ``successor-pointer-present`` belong to leaf γ (E4)
and are deliberately NOT emitted here.

**This script never writes to the live corpus and never evaluates a limit.**

Both halves of that sentence are load-bearing:

- *Never writes.* There is no ``--apply`` band, no delete/add/update call and
  no write path anywhere in this module. D8's read-only runner pattern
  (``audit_duplicate_memories.py:364-378``) is copied minus every mutation.
  The guarantee is asserted as BEHAVIOUR in the tests — the probe is driven
  against a service double whose every write method raises — rather than
  merely claimed in this docstring.
- *Never evaluates a limit.* Per D1 the first run is a baseline snapshot, not
  a day-one alarm source, and per G6/M2 every threshold, tolerance,
  grandfather set and alarm lives in leaf α's limits evaluator. No pass rate,
  bound or verdict appears in this script, in the registry fixture, or in any
  of its tests. The only numeric parameter is ``k`` (``--k``, defaulting to 5
  and 10), which is a metric *parameterisation* — "is the canonical in this
  list of five" is set membership, not a tuned bound.

**Schema home (D2).** Every artifact model, validator, path helper, stamp
format and report renderer comes from :mod:`shared.memory_eval_metrics`.
Nothing in that contract is re-declared here.

**Registry keys (D5).** Topics are identified by a stable slug; the expected
canonical ENTRY is identified by a content hash (memory UUIDs rot on
re-consolidation), with ``last_known_id`` as a disclosed fallback. The
registry is a committed fixture derived entirely from committed offline
sources, so it works today — before the ``metadata.topic`` vocabulary of
3195/3201 lands — and strictly widens when that vocabulary does.
"""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, NamedTuple

# ---------------------------------------------------------------------------
# Pinned M1 contract vocabulary
#
# These spellings are COPIED from leaf α's committed consumer-side exemplars
# (shared/tests/fixtures/memory_eval/e1-retrieval-health/metrics-*.json), not
# chosen here. The limits evaluator joins a run to its baseline window BY
# metric_id (shared/src/shared/memory_eval_metrics.py:262-275), so a different
# spelling would not fail loudly — it would make the metric invisible to the
# evaluator and to the dashboard, which is strictly worse than a crash.
# ---------------------------------------------------------------------------

EVAL_ID = 'e1-retrieval-health'
"""Also the artifact directory segment (``<root>/<eval_id>/metrics-<STAMP>.json``)."""

METRIC_TOPIC_CANONICAL_PRESENT = 'topic-canonical-present'
"""Tripwire (M2 rule a). One item per registry topic, keyed ``t-<topic-slug>``."""

METRIC_CANONICAL_IN_TOP_K = 'canonical-in-top-{k}'
"""Proportion over (topic, phrasing) pairs, formatted per ``k``."""

METRIC_CANONICAL_IN_TOP_K_HELD_OUT = 'canonical-in-top-{k}-held-out'
"""The Goodhart guard made visible — held-out phrasings, trended separately."""

METRIC_CLAIM_RECALL = 'claim-recall'
"""Proportion over (topic, claim-query) pairs."""

METRIC_CONTAMINATION_SHARE = 'contamination-share'
"""Proportion of scored results carrying a DIFFERENT registry topic."""

METRIC_SUPERSEDED_ABOVE_SUCCESSOR = 'superseded-above-successor'
"""Count of ranking inversions over registry-recorded (superseded, successor) pairs."""

TRIPWIRE_ITEM_PREFIX = 't-'
"""``TripwireItem.item_key`` shape. This is a STORED key (α's grandfather set
persists it), not a display string."""

TRIPWIRE_K = 5
"""The ``k`` the per-topic tripwire predicate is evaluated at.

Matches the exemplar's ``canonical-in-top-5``. Not a threshold: the predicate
is "is the canonical in this list", and the list has to have some length.
"""

DEFAULT_KS: tuple[int, ...] = (5, 10)
"""Default ``--k`` values. A parameterisation, not a limit (see module docstring)."""

DERIVED_FROM_VALUES: frozenset[str] = frozenset({
    'curator_gate',
    'census_topic',
    'topic_guard_cluster',
    'briefing_query',
    'hand',
})
"""The closed provenance vocabulary for a registry entry.

Lets a future run tell which entries auto-derivation has since taken over from
hand-authoring — the ``derived_from`` half of the additive-tolerant decision.
"""

REGISTRY_SCHEMA_VERSION = 1

_SLUG_RE = re.compile(r'^[a-z0-9]+(?:[-_][a-z0-9]+)*$')
"""Registry topic slugs: lowercase alphanumerics separated by ``-`` or ``_``.

``_`` is accepted because live ``metadata.topic`` values use both spellings
(the census reports ``architect_report_task_already_done_main_reachability``
against the guard cluster's ``architect-report-task-already-done-main-reachability``).
Comparison normalises the two; the slug syntax merely has to admit both.
"""

_WHITESPACE_RE = re.compile(r'\s+')


# ---------------------------------------------------------------------------
# Content identity
# ---------------------------------------------------------------------------

def content_key(text: str) -> str:
    """A stable 16-hex-char identity for a memory's *text*.

    ``sha256(...).hexdigest()[:16]`` — the ``shared/task_metadata.py:310``
    convention, so the digest length matches every other content signature in
    this repo.

    Whitespace is normalised (surrounding stripped, internal runs collapsed to
    one space) BEFORE hashing so that re-indentation, a wrapped line or a
    trailing newline picked up in transit does not read as a different entry.
    Nothing else is normalised: case and punctuation are content, and folding
    them would make two genuinely different claims collide.
    """
    normalized = _WHITESPACE_RE.sub(' ', text).strip()
    return hashlib.sha256(normalized.encode('utf-8')).hexdigest()[:16]


def normalize_topic(topic: str) -> str:
    """Fold a topic slug for comparison: lowercased, ``_`` unified to ``-``.

    The live corpus spells the same topic both ways (see :data:`_SLUG_RE`), so
    an exact string comparison would silently fail to match the very entries
    the registry was built from.
    """
    return topic.strip().lower().replace('_', '-')


# ---------------------------------------------------------------------------
# Registry model
# ---------------------------------------------------------------------------

class RegistryError(ValueError):
    """A topic registry that must not be probed.

    Every message names the offending topic slug. A registry that failed to
    load must abort the run rather than produce an empty artifact: an artifact
    claiming zero topics is indistinguishable from a healthy corpus with no
    registry, and the evaluator would read the silence as a clean run.
    """


@dataclass(frozen=True)
class Canonical:
    """The entry a topic's queries are expected to return.

    Two keys, in priority order. ``content_hash`` is primary because memory
    UUIDs rot on re-consolidation (D5). ``last_known_id`` is the disclosed
    fallback: when the hash misses but the id hits, the fixture needs
    re-hashing; when the id misses but the hash hits, the id needs updating.
    Both are reported facts rather than silent failures.

    ``content_prefix`` is a human anchor for whoever reads the report — it is
    never used for matching.
    """

    content_hash: str
    content_prefix: str = ''
    last_known_id: str | None = None


@dataclass(frozen=True)
class Phrasing:
    """One query string for a topic.

    ``held_out`` marks a phrasing authored fresh for this eval and never used
    to build the entries it retrieves — the Goodhart guard. A held-out
    phrasing is counted in the pooled rate AND in its own separate metric, so
    saturation on tuned phrasings cannot mask held-out rot.
    """

    text: str
    held_out: bool = False


@dataclass(frozen=True)
class ClaimQuery:
    """A query plus the substantive needles whose presence means the claim came back.

    Deliberately weaker than canonical identity: claim recall asks "does the
    claim come back at all", from ANY returned entry. That is the
    Goodhart-resistant question — a consolidation that moved a claim into a
    different entry has not lost it.
    """

    query: str
    needles: tuple[str, ...] = ()


@dataclass(frozen=True)
class SupersedesPair:
    """A registry-recorded (superseded, successor) content-hash pair.

    Recorded OFFLINE at derivation time, deliberately, so the runtime metric
    is a pure ranking comparison and this module never parses raw
    ``metadata.supersedes``. That parser is ``normalize_supersedes()`` (task
    3196), leaf γ's hard dependency — a second one here would be exactly the
    lockstep duplication INV-5 forbids.
    """

    superseded_hash: str
    successor_hash: str


@dataclass(frozen=True)
class RegistryEntry:
    """One probed topic."""

    topic: str
    project_id: str
    derived_from: str
    canonical: Canonical
    phrasings: tuple[Phrasing, ...]
    claim_queries: tuple[ClaimQuery, ...] = ()
    members: tuple[str, ...] = ()
    supersedes_pairs: tuple[SupersedesPair, ...] = ()
    provenance: dict[str, Any] = field(default_factory=dict)
    extra: dict[str, Any] = field(default_factory=dict)
    """Unrecognised keys, preserved verbatim.

    The additive-tolerant half of the loader's contract: when 3201's retro
    stamping widens auto-derivation, a richer entry must load against this
    same loader rather than force a fixture rewrite.
    """

    @property
    def held_out_phrasings(self) -> tuple[Phrasing, ...]:
        return tuple(p for p in self.phrasings if p.held_out)

    @property
    def item_key(self) -> str:
        """This topic's stable tripwire item key."""
        return f'{TRIPWIRE_ITEM_PREFIX}{self.topic}'


@dataclass(frozen=True)
class TopicRegistry:
    """The committed set of probed topics."""

    schema_version: int
    entries: tuple[RegistryEntry, ...]

    @property
    def by_topic(self) -> dict[str, RegistryEntry]:
        return {e.topic: e for e in self.entries}

    @property
    def topics(self) -> set[str]:
        return {e.topic for e in self.entries}

    @property
    def normalized_topics(self) -> set[str]:
        """Topic slugs folded for comparison against live ``metadata.topic`` values."""
        return {normalize_topic(e.topic) for e in self.entries}


# ---------------------------------------------------------------------------
# Registry loading — required-strict, additive-tolerant
# ---------------------------------------------------------------------------

_ENTRY_KNOWN_KEYS = frozenset({
    'topic', 'project_id', 'derived_from', 'canonical', 'phrasings',
    'claim_queries', 'members', 'supersedes_pairs', 'provenance',
})

_REQUIRED_ENTRY_KEYS = ('project_id', 'derived_from', 'canonical', 'phrasings')


def _fail(topic: str, message: str) -> RegistryError:
    """Build a RegistryError that NAMES the offending entry.

    pydantic-style positional locations ("entries.3.canonical") tell an
    operator nothing about which topic they got wrong. Naming the slug is the
    structured-facts-at-failure invariant applied to a fixture error.
    """
    return RegistryError(f'topic registry entry {topic!r}: {message}')


def _parse_canonical(topic: str, raw: Any) -> Canonical:
    if not isinstance(raw, dict):
        raise _fail(topic, f"'canonical' must be an object, got {type(raw).__name__}.")
    content_hash = raw.get('content_hash')
    if not isinstance(content_hash, str) or not content_hash:
        raise _fail(topic, "'canonical' is missing a non-empty 'content_hash'.")
    last_known_id = raw.get('last_known_id')
    if last_known_id is not None and not isinstance(last_known_id, str):
        raise _fail(topic, "'canonical.last_known_id' must be a string or absent.")
    prefix = raw.get('content_prefix', '')
    if not isinstance(prefix, str):
        raise _fail(topic, "'canonical.content_prefix' must be a string.")
    return Canonical(
        content_hash=content_hash, content_prefix=prefix, last_known_id=last_known_id,
    )


def _parse_phrasings(topic: str, raw: Any) -> tuple[Phrasing, ...]:
    if not isinstance(raw, list):
        raise _fail(topic, f"'phrasings' must be a list, got {type(raw).__name__}.")
    if not raw:
        raise _fail(
            topic,
            "'phrasings' is empty. A topic with no query has nothing to measure, and "
            'an unmeasured topic that silently reports no failures is worse than none.',
        )
    phrasings: list[Phrasing] = []
    for item in raw:
        if not isinstance(item, dict):
            raise _fail(topic, f"each phrasing must be an object, got {type(item).__name__}.")
        text = item.get('text')
        if not isinstance(text, str) or not text.strip():
            raise _fail(topic, "a phrasing is missing a non-empty 'text'.")
        held_out = item.get('held_out', False)
        if not isinstance(held_out, bool):
            raise _fail(topic, f"phrasing {text!r} has a non-boolean 'held_out'.")
        phrasings.append(Phrasing(text=text, held_out=held_out))
    if not any(p.held_out for p in phrasings):
        raise _fail(
            topic,
            "no phrasing is marked 'held_out'. At least one freshly authored phrasing "
            'is required per topic: without it a fix that tunes the known phrasings '
            'saturates this topic and the metric stops discriminating (the Goodhart '
            'guard D5 asks for).',
        )
    return tuple(phrasings)


def _parse_claim_queries(topic: str, raw: Any) -> tuple[ClaimQuery, ...]:
    if raw is None:
        return ()
    if not isinstance(raw, list):
        raise _fail(topic, f"'claim_queries' must be a list, got {type(raw).__name__}.")
    queries: list[ClaimQuery] = []
    for item in raw:
        if not isinstance(item, dict):
            raise _fail(topic, f"each claim_query must be an object, got {type(item).__name__}.")
        query = item.get('query')
        if not isinstance(query, str) or not query.strip():
            raise _fail(topic, "a claim_query is missing a non-empty 'query'.")
        needles = item.get('needles', [])
        if not isinstance(needles, list) or not all(isinstance(n, str) for n in needles):
            raise _fail(topic, f"claim_query {query!r} needs 'needles' as a list of strings.")
        queries.append(ClaimQuery(query=query, needles=tuple(needles)))
    return tuple(queries)


def _parse_supersedes_pairs(topic: str, raw: Any) -> tuple[SupersedesPair, ...]:
    if raw is None:
        return ()
    if not isinstance(raw, list):
        raise _fail(topic, f"'supersedes_pairs' must be a list, got {type(raw).__name__}.")
    pairs: list[SupersedesPair] = []
    for item in raw:
        if not isinstance(item, dict):
            raise _fail(topic, 'each supersedes_pair must be an object.')
        superseded = item.get('superseded_hash')
        successor = item.get('successor_hash')
        if not isinstance(superseded, str) or not superseded:
            raise _fail(topic, "a supersedes_pair is missing 'superseded_hash'.")
        if not isinstance(successor, str) or not successor:
            raise _fail(topic, "a supersedes_pair is missing 'successor_hash'.")
        pairs.append(
            SupersedesPair(superseded_hash=superseded, successor_hash=successor),
        )
    return tuple(pairs)


def _parse_entry(raw: Any, index: int) -> RegistryEntry:
    if not isinstance(raw, dict):
        raise RegistryError(
            f'topic registry entry at index {index} must be an object, '
            f'got {type(raw).__name__}.'
        )
    topic = raw.get('topic')
    if not isinstance(topic, str) or not topic.strip():
        raise RegistryError(
            f"topic registry entry at index {index} is missing a non-empty 'topic' slug. "
            'The slug is the tripwire item key, so an entry without one cannot be '
            'reported against.'
        )
    if not _SLUG_RE.match(topic):
        raise _fail(
            topic,
            'topic is not slug-shaped (lowercase alphanumerics separated by - or _).',
        )
    for key in _REQUIRED_ENTRY_KEYS:
        if key not in raw:
            raise _fail(topic, f'required field {key!r} is missing.')

    project_id = raw['project_id']
    if not isinstance(project_id, str) or not project_id.strip():
        raise _fail(topic, "'project_id' must be a non-empty string.")

    derived_from = raw['derived_from']
    if derived_from not in DERIVED_FROM_VALUES:
        raise _fail(
            topic,
            f'derived_from={derived_from!r} is not one of '
            f'{sorted(DERIVED_FROM_VALUES)}.',
        )

    members = raw.get('members', [])
    if not isinstance(members, list) or not all(isinstance(m, str) for m in members):
        raise _fail(topic, "'members' must be a list of content hashes.")

    provenance = raw.get('provenance', {})
    if not isinstance(provenance, dict):
        raise _fail(topic, "'provenance' must be an object.")

    return RegistryEntry(
        topic=topic,
        project_id=project_id,
        derived_from=derived_from,
        canonical=_parse_canonical(topic, raw['canonical']),
        phrasings=_parse_phrasings(topic, raw['phrasings']),
        claim_queries=_parse_claim_queries(topic, raw.get('claim_queries')),
        members=tuple(members),
        supersedes_pairs=_parse_supersedes_pairs(topic, raw.get('supersedes_pairs')),
        provenance=dict(provenance),
        extra={k: v for k, v in raw.items() if k not in _ENTRY_KNOWN_KEYS},
    )


def load_topic_registry(path: str | Path) -> TopicRegistry:
    """Load and validate the committed topic registry at *path*.

    Required-strict, additive-tolerant (see the ``extra`` field of
    :class:`RegistryEntry`): a missing or malformed required field is a hard
    failure naming the offending slug, while an unrecognised key — on an entry
    or at the top level — loads untouched.

    Every failure mode raises :class:`RegistryError`, including a missing file
    and undecodable JSON, so a caller catches one exception type and reports
    one actionable message.
    """
    path = Path(path)
    try:
        text = path.read_text(encoding='utf-8')
    except OSError as exc:
        raise RegistryError(f'cannot read topic registry {str(path)!r}: {exc}') from exc
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise RegistryError(f'topic registry {str(path)!r} is not valid JSON: {exc}') from exc

    if not isinstance(payload, dict):
        raise RegistryError(
            f'topic registry {str(path)!r} must be an object with an "entries" list.'
        )
    raw_entries = payload.get('entries')
    if not isinstance(raw_entries, list):
        raise RegistryError(
            f'topic registry {str(path)!r} is missing its "entries" list.'
        )

    entries = [_parse_entry(raw, i) for i, raw in enumerate(raw_entries)]

    seen: dict[str, int] = {}
    for i, entry in enumerate(entries):
        if entry.topic in seen:
            raise _fail(
                entry.topic,
                f'duplicate topic slug (also at index {seen[entry.topic]}). The slug is '
                'the persisted tripwire item key, so a duplicate would make one of the '
                'two invisible to the grandfather set.',
            )
        seen[entry.topic] = i

    schema_version = payload.get('schema_version', REGISTRY_SCHEMA_VERSION)
    if schema_version != REGISTRY_SCHEMA_VERSION:
        raise RegistryError(
            f'topic registry {str(path)!r} declares schema_version={schema_version!r}, '
            f'but this loader understands {REGISTRY_SCHEMA_VERSION}.'
        )

    return TopicRegistry(schema_version=schema_version, entries=tuple(entries))


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


def _derive_curator_gate_candidates(rows: list[dict]) -> list[dict[str, Any]]:
    groups: dict[str, list[dict]] = {}
    for row in rows:
        groups.setdefault(row['cluster_id'], []).append(row)

    candidates: list[dict[str, Any]] = []
    for cluster_id, group in sorted(groups.items()):
        canonical_rows = [r for r in group if r.get('label') == 'canonical']
        if not canonical_rows:
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
            content_prefix=_WHITESPACE_RE.sub(' ', canonical['content']).strip(),
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
    return candidates


def _derive_census_candidates(census: dict) -> tuple[list[dict[str, Any]], int]:
    """Multi-entry census topics, plus the number of singletons skipped.

    A count-1 topic has exactly one entry, so "is the canonical in the top k"
    is answered by that entry's mere existence — it measures presence, not
    retrieval. They are skipped, and the count is returned so the skip is
    reported rather than silently applied.
    """
    table = ((census.get('grand_total') or {}).get('topic') or {}).get('entries') or []
    candidates: list[dict[str, Any]] = []
    skipped = 0
    for item in table:
        if not isinstance(item, dict):
            continue
        value, count = item.get('value'), item.get('count', 0)
        if not isinstance(value, str) or not value:
            continue
        if not isinstance(count, int) or count <= 1:
            skipped += 1
            continue
        candidates.append(_candidate(
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
        ))
    return candidates, skipped


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
    census_report: dict,
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
    """
    candidates = list(_derive_curator_gate_candidates(calibration_rows))
    census_candidates, skipped = _derive_census_candidates(census_report)
    candidates.extend(census_candidates)
    candidates.extend(_derive_guard_cluster_candidates(guard_clusters))

    seen: set[str] = set()
    deduped: list[dict[str, Any]] = []
    collisions = 0
    for candidate in candidates:
        if candidate['topic'] in seen:
            collisions += 1
            continue
        seen.add(candidate['topic'])
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
            'census_topics_skipped_singleton': skipped,
            'topic_guard_clusters': sum(
                1 for c in deduped if c['derived_from'] == 'topic_guard_cluster'
            ),
            'slug_collisions_dropped': collisions,
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
    census = json.loads(Path(census_path).read_text(encoding='utf-8'))
    result = derive_registry_candidates(rows, census, guard_clusters)
    payload = result.as_registry_payload()
    payload['_disclosures'] = result.disclosures
    return json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + '\n'


# ---------------------------------------------------------------------------
# Canonical-in-top-k — pure over an already-fetched result list
#
# Every function below takes a list of results and returns a record. Nothing
# here touches a store, which is what lets the merge lane test the whole metric
# family with no Qdrant, no embedder and no OPENAI_API_KEY.
# ---------------------------------------------------------------------------

MATCHED_BY_CONTENT_HASH = 'content_hash'
MATCHED_BY_LAST_KNOWN_ID = 'last_known_id'


@dataclass(frozen=True)
class MatchOutcome:
    """Where (and how) a topic's canonical was found in one result list."""

    hit: bool
    rank: int | None
    matched_by: str | None

    @property
    def unmatched(self) -> bool:
        """Neither matcher fired — the entry was not in the list at all.

        Distinct from ``not hit``: an entry found at rank k+1 is present but
        ranked too low, which is a ranking problem. An unmatched entry is
        either absent from the corpus or the fixture has decayed past both
        keys. The report separates the two rather than reporting one number.
        """
        return self.matched_by is None

    @property
    def needs_hash_repair(self) -> bool:
        """The id matched but the hash did not — the fixture needs re-hashing.

        Reported rather than repaired: a probe that silently rewrote its own
        expectations to match what it found could never fail.
        """
        return self.matched_by == MATCHED_BY_LAST_KNOWN_ID


@dataclass(frozen=True)
class PhrasingObservation:
    """One (topic, phrasing, k) probe result."""

    topic: str
    phrasing: str
    held_out: bool
    k: int
    hit: bool
    rank: int | None
    matched_by: str | None
    degraded: bool = False
    """True when the search that produced this list reported a failed store.

    A degraded observation is EXCLUDED from every metric denominator: charging
    a store outage as a canonical-absent failure would manufacture a
    corpus-wide findability collapse out of an infrastructure blip.
    """

    @property
    def unmatched(self) -> bool:
        return self.matched_by is None

    @property
    def needs_hash_repair(self) -> bool:
        return self.matched_by == MATCHED_BY_LAST_KNOWN_ID


def _result_content(result: Any) -> str:
    return getattr(result, 'content', '') or ''


def _result_id(result: Any) -> str:
    return getattr(result, 'id', '') or ''


def canonical_hit(results: list, entry: RegistryEntry, k: int) -> MatchOutcome:
    """Find *entry*'s canonical in *results*, honouring the top-*k* cut.

    Content hash first, ``last_known_id`` second (D5): memory UUIDs rot on
    re-consolidation, so hashing the returned content is the durable key, while
    the id fallback keeps a benignly-reworded canonical from reading as a
    findability regression. Which matcher fired is recorded either way, because
    both divergences are a fixture-repair signal an operator needs to see.

    The whole list is scanned, not just its first *k*, so a canonical that came
    back at rank k+3 is reported at its true rank instead of as "absent" — a
    ranking problem and an absence problem need different fixes.
    """
    canonical = entry.canonical
    hash_rank: int | None = None
    id_rank: int | None = None

    for index, result in enumerate(results, start=1):
        if hash_rank is None and content_key(_result_content(result)) == canonical.content_hash:
            hash_rank = index
        if (
            id_rank is None
            and canonical.last_known_id
            and _result_id(result) == canonical.last_known_id
        ):
            id_rank = index
        if hash_rank is not None and id_rank is not None:
            break

    if hash_rank is not None:
        return MatchOutcome(
            hit=hash_rank <= k, rank=hash_rank, matched_by=MATCHED_BY_CONTENT_HASH,
        )
    if id_rank is not None:
        return MatchOutcome(
            hit=id_rank <= k, rank=id_rank, matched_by=MATCHED_BY_LAST_KNOWN_ID,
        )
    return MatchOutcome(hit=False, rank=None, matched_by=None)


def observe_phrasing(
    results: list,
    entry: RegistryEntry,
    phrasing: Phrasing,
    k: int,
    *,
    degraded: bool = False,
) -> PhrasingObservation:
    """Build the (topic, phrasing, k) observation the metrics aggregate over."""
    outcome = canonical_hit(results, entry, k)
    return PhrasingObservation(
        topic=entry.topic,
        phrasing=phrasing.text,
        held_out=phrasing.held_out,
        k=k,
        hit=outcome.hit,
        rank=outcome.rank,
        matched_by=outcome.matched_by,
        degraded=degraded,
    )


# ---------------------------------------------------------------------------
# superseded-above-successor — a pure ranking comparison
#
# Reads ONLY registry-recorded hash pairs. metadata['supersedes'] is never
# touched: that shape zoo is normalize_supersedes()' problem (task 3196, leaf
# gamma's hard dependency), and a second parser here would be exactly the
# lockstep duplication INV-5 forbids. Because the relation was recorded offline
# at derivation time, this function needs no pointer knowledge at all.
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class InversionRecord:
    """A superseded entry that outranked the entry which replaced it."""

    topic: str
    phrasing: str
    superseded_hash: str
    successor_hash: str
    superseded_rank: int
    successor_rank: int


def superseded_inversions(
    results: list,
    entry: RegistryEntry,
    *,
    phrasing: str = '',
) -> list[InversionRecord]:
    """Registry-recorded pairs where the superseded entry outranks its successor.

    Both-present-only: a pair with just one member in *results* yields nothing.
    An absent successor is a findability question ``canonical-in-top-k`` already
    measures, and counting it here as well would charge one defect against two
    metrics — inflating any downstream trend by double-weighting a single fix.

    Rank is the position in the list the store returned, so equal
    ``relevance_score`` values resolve by that order rather than by an unstable
    re-sort. Two runs over the same list therefore produce the same count; a
    tie that flapped would look to the evaluator like a real regression.

    Records name both hashes and both ranks: a bare count tells an operator
    that something inverted but not which pair to go and look at.
    """
    if not entry.supersedes_pairs:
        return []

    first_rank: dict[str, int] = {}
    for index, result in enumerate(results, start=1):
        key = content_key(_result_content(result))
        first_rank.setdefault(key, index)

    inversions: list[InversionRecord] = []
    for pair in entry.supersedes_pairs:
        superseded_rank = first_rank.get(pair.superseded_hash)
        successor_rank = first_rank.get(pair.successor_hash)
        if superseded_rank is None or successor_rank is None:
            continue
        if superseded_rank < successor_rank:
            inversions.append(InversionRecord(
                topic=entry.topic,
                phrasing=phrasing,
                superseded_hash=pair.superseded_hash,
                successor_hash=pair.successor_hash,
                superseded_rank=superseded_rank,
                successor_rank=successor_rank,
            ))
    return inversions


# ---------------------------------------------------------------------------
# claim-recall — did the claim come back at all?
#
# Deliberately WEAKER than canonical identity. ``canonical-in-top-k`` asks
# whether one specific entry is findable; this asks whether the KNOWLEDGE is,
# from any entry. The distinction is the whole point: consolidation (3111/3112)
# rewrites and merges entries by design, so a metric that demanded the claim
# come back from the same entry would score the fix lineage this eval exists to
# measure as a regression, and the obvious way to make that metric go green
# would be to stop consolidating.
# ---------------------------------------------------------------------------

def _normalize_for_needles(text: str) -> str:
    """Fold *text* for needle matching: whitespace collapsed, then lowercased.

    The whitespace half is exactly :func:`content_key`'s normalisation, so a
    re-wrapped or re-indented stored line does not read as knowledge loss.

    Case folding is the one deliberate DIVERGENCE from ``content_key``. There,
    case is content — two claims differing only in case are two claims, and
    folding them would collide distinct entries under one hash. Here the needle
    is a substring probe against prose, where casing is presentation churn:
    the committed registry carries needles like ``WRITE-SET`` that the corpus
    spells ``write-set``, and a case-sensitive miss would report a knowledge
    loss that did not happen.
    """
    return _WHITESPACE_RE.sub(' ', text).strip().lower()


@dataclass(frozen=True)
class ClaimOutcome:
    """Whether one claim came back, and — when it did not — what was missing."""

    recalled: bool
    missing_needles: tuple[str, ...]
    matched_rank: int | None
    scorable: bool = True
    """False when the claim query carries no needles.

    ``all(needle in text for needle in ())`` is vacuously True, so a needle-less
    claim query would silently score as recalled and inflate the rate for a
    malformed registry entry. Unscorable claims are excluded from the
    denominator and DISCLOSED instead — an unmeasurable claim must not be able
    to masquerade as a healthy one.
    """


def claim_recalled(results: list, claim_query: ClaimQuery, k: int) -> ClaimOutcome:
    """Did *claim_query*'s needles come back in the top-*k* of *results*?

    All needles are required, and they must all come from a SINGLE returned
    entry. Pooling needles across the result set would let an entry saying
    "the merge lane is strictly serial" and an unrelated entry saying "never
    rolls back" jointly satisfy a claim that neither one makes — recall of a
    sentence the corpus never stated.

    When no entry carries every needle, ``missing_needles`` is reported against
    the CLOSEST entry (most needles matched, ties broken by better rank), since
    diffing against the near-miss is what tells an operator which half of the
    claim was lost. With nothing matched at all the full needle list is
    returned, which is the honest answer rather than an empty one.
    """
    needles = tuple(claim_query.needles)
    if not needles:
        return ClaimOutcome(
            recalled=False, missing_needles=(), matched_rank=None, scorable=False,
        )

    normalized_needles = [(needle, _normalize_for_needles(needle)) for needle in needles]

    best_missing: tuple[str, ...] = needles
    best_found = -1
    for rank, result in enumerate(results[:k], start=1):
        haystack = _normalize_for_needles(_result_content(result))
        missing = tuple(
            original for original, folded in normalized_needles if folded not in haystack
        )
        if not missing:
            return ClaimOutcome(
                recalled=True, missing_needles=(), matched_rank=rank, scorable=True,
            )
        found = len(needles) - len(missing)
        if found > best_found:  # strict >: ties keep the better (earlier) rank
            best_found = found
            best_missing = missing

    return ClaimOutcome(
        recalled=False, missing_needles=best_missing, matched_rank=None, scorable=True,
    )


# ---------------------------------------------------------------------------
# contamination-share — how much of what came back belongs to another topic?
#
# This metric has to be WELL-POSED TODAY, before 3195/3201 widen the
# `metadata.topic` vocabulary. The committed census measured 491 of 49,628
# entries carrying a topic at all, so the overwhelming majority of results have
# none, and any definition that treated "no topic" as evidence of anything
# would be measuring stamping coverage while claiming to measure contamination.
#
# So: FOREIGN iff the result carries a topic that is IN the registry and is not
# the probed one. Everything else is UNTOPICED and disclosed alongside the
# count, never folded into the numerator. The definition is monotone in the
# registry — widening it (which is exactly what 3201's retro stamping enables)
# strictly widens the numerator's reach with no code change here.
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ForeignRecord:
    """One returned entry that belongs to a different registered topic."""

    topic: str
    """The topic being probed."""

    foreign_topic: str
    """The registered topic the returned entry actually carries.

    Named, not counted: "3 foreign results" tells an operator that something
    bled in, while "escalation-ladder bled into merge-lane" tells them where to
    look. A bare count of a cross-topic leak is not actionable.
    """

    rank: int
    result_id: str = ''


class ContaminationOutcome(NamedTuple):
    """``(foreign_records, foreign_count, untopiced_count, scored_total)``.

    A NamedTuple so the four fields both unpack positionally (the shape the
    caller's metric assembly wants) and read by name at the use site.

    ``untopiced_count`` and ``scored_total`` are not diagnostics — they are the
    honesty of the share. ``foreign_count / scored_total`` computed without
    them looks authoritative while, on today's corpus, most of the denominator
    is entries that could not have been classified either way.
    """

    foreign_records: tuple[ForeignRecord, ...]
    foreign_count: int
    untopiced_count: int
    scored_total: int


def _result_topic(result: Any) -> str | None:
    """The result's ``metadata['topic']``, or None when it has no usable one.

    Live metadata is not schema-enforced, so a topic that is missing, empty, or
    not a string is treated identically: unusable. Coercing a list-valued topic
    with ``str()`` would invent a topic value nobody wrote.
    """
    metadata = getattr(result, 'metadata', None)
    if not isinstance(metadata, dict):
        return None
    topic = metadata.get('topic')
    if not isinstance(topic, str) or not topic.strip():
        return None
    return topic


def classify_contamination(
    results: list,
    entry: RegistryEntry,
    registry: TopicRegistry,
    k: int,
) -> ContaminationOutcome:
    """Classify the top-*k* of *results* against the topic being probed.

    Foreignness is gated on registry membership in both directions: the
    returned entry's topic must be one the registry knows, AND it must not fold
    onto *entry*'s own slug. Comparison is via :func:`normalize_topic`, because
    the corpus spells the same topic with ``-`` and ``_`` (the guard cluster's
    ``architect-report-...`` against the census's
    ``architect_report_...``) — an exact comparison would report a topic as
    foreign to itself.

    Returns the foreign records, their count, the untopiced count and the
    scored total, so the share's denominator can never be reported without the
    disclosure of how much of it was unclassifiable.
    """
    probed = normalize_topic(entry.topic)
    known = registry.normalized_topics

    foreign: list[ForeignRecord] = []
    untopiced = 0
    scored = 0

    for rank, result in enumerate(results[:k], start=1):
        scored += 1
        raw_topic = _result_topic(result)
        if raw_topic is None:
            untopiced += 1
            continue
        folded = normalize_topic(raw_topic)
        if folded == probed:
            continue
        if folded not in known:
            # An unregistered topic is not evidence of contamination: the
            # census counted 352 distinct live topic values against the ~32
            # this registry adjudicates, so "unknown to us" is the common case
            # and would swamp the numerator with entries nobody has judged.
            untopiced += 1
            continue
        foreign.append(ForeignRecord(
            topic=entry.topic,
            foreign_topic=raw_topic,
            rank=rank,
            result_id=_result_id(result),
        ))

    return ContaminationOutcome(
        foreign_records=tuple(foreign),
        foreign_count=len(foreign),
        untopiced_count=untopiced,
        scored_total=scored,
    )
