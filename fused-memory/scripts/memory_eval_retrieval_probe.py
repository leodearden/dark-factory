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
from typing import Any

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
