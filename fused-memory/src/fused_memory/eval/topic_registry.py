"""The E1 topic-registry data contract: what a probed topic is, and how the committed fixture loads.

``docs/prds/memory-eval-program.md`` §5 leaf β. Each :class:`RegistryEntry`
names a topic, the canonical entry its queries are expected to return, and
the phrasings and claim queries that probe it. :func:`load_topic_registry`
reads the committed fixture required-strict and additive-tolerant: a missing
or malformed required field fails naming the offending topic, while an
unrecognised key loads untouched.

**Registry keys (D5).** Topics are identified by a stable slug; the expected
canonical ENTRY is identified by a content hash (memory UUIDs rot on
re-consolidation), with ``last_known_id`` as a disclosed fallback. The
registry is a committed fixture derived entirely from committed offline
sources, so it works today — before the ``metadata.topic`` vocabulary of
3195/3201 lands — and strictly widens when that vocabulary does.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from fused_memory.models.enums import GRAPHITI_PRIMARY, MEM0_PRIMARY, SourceStore

# ---------------------------------------------------------------------------
# Registry vocabulary
# ---------------------------------------------------------------------------

TRIPWIRE_ITEM_PREFIX = 't-'
"""``TripwireItem.item_key`` shape. This is a STORED key (α's grandfather set
persists it), not a display string."""

QUERY_SURFACE_DERIVATION = 'briefing_query'
"""The ``derived_from`` of a topic that keys a QUERY a caller fires.

See :attr:`RegistryEntry.is_query_surface`.
"""

DERIVED_FROM_VALUES: frozenset[str] = frozenset({
    'curator_gate',
    'census_topic',
    'topic_guard_cluster',
    QUERY_SURFACE_DERIVATION,
    'hand',
})
"""The closed provenance vocabulary for a registry entry.

Lets a future run tell which entries auto-derivation has since taken over from
hand-authoring — the ``derived_from`` half of the additive-tolerant decision.
"""

REGISTRY_SCHEMA_VERSION = 1

REGISTRY_SLUG_RE = re.compile(r'^[a-z0-9]+(?:[-_][a-z0-9]+)*$')
"""Registry topic slugs: lowercase alphanumerics separated by ``-`` or ``_``.

``_`` is accepted because live ``metadata.topic`` values use both spellings
(the census reports ``architect_report_task_already_done_main_reachability``
against the guard cluster's ``architect-report-task-already-done-main-reachability``).
Comparison normalises the two; the slug syntax merely has to admit both. So
this is deliberately wider than ``fused_memory/topic_slug.py::TOPIC_SLUG_RE``,
which rejects ``_``.
"""


def normalize_topic(topic: str) -> str:
    """Fold a topic slug for comparison: lowercased, ``_`` unified to ``-``.

    The live corpus spells the same topic both ways (see :data:`REGISTRY_SLUG_RE`), so
    an exact string comparison would silently fail to match the very entries
    the registry was built from.
    """
    return topic.strip().lower().replace('_', '-')


# ---------------------------------------------------------------------------
# Store vocabulary
# ---------------------------------------------------------------------------

def corpus_categories() -> tuple[str, ...]:
    """The category vocabulary, taken from the store's own enums.

    Derived rather than restated. The bucket vocabulary belongs to the
    memory-metadata PRD, and a copy of it here would keep reporting six
    categories on the day a seventh is added — the artifact would look
    complete while silently under-counting the corpus it claims to describe.
    """
    return tuple(sorted(c.value for c in (GRAPHITI_PRIMARY | MEM0_PRIMARY)))


def search_stores() -> tuple[str, ...]:
    """The store vocabulary a ``search_scope`` may name, from the store's own enum.

    Derived for the same reason :func:`corpus_categories` is.
    """
    return tuple(sorted(s.value for s in SourceStore))


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
class SearchScope:
    """The store/category scope an entry's real caller searches with.

    Declared only for an entry whose caller scopes its own search — today the
    briefing's conventions channel
    (``orchestrator/src/orchestrator/agents/memory_recall.py::MemoryRecall._search``).
    An entry without one is searched unscoped and the read router picks,
    which is what an agent's own ad-hoc search experiences.
    """

    stores: tuple[str, ...] = ()
    categories: tuple[str, ...] = ()

    def as_search_kwargs(self) -> dict[str, list[str]]:
        """``MemoryService.search`` keyword arguments, an empty dimension omitted.

        Omitted rather than sent empty, as
        ``memory_recall.py::MemoryRecall._search`` does, so an undeclared
        dimension is left to the server's own routing instead of becoming a
        filter that matches nothing.
        """
        kwargs: dict[str, list[str]] = {}
        if self.stores:
            kwargs['stores'] = list(self.stores)
        if self.categories:
            kwargs['categories'] = list(self.categories)
        return kwargs


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
    search_scope: SearchScope | None = None
    """The scope this topic is searched with; ``None`` searches unscoped."""
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
    def is_query_surface(self) -> bool:
        """This topic keys a query a caller fires, not a ``metadata.topic`` value.

        No record is ever stamped with such a slug, so a metadata census must
        not gauge it: it would read as a permanently unpopulated topic that no
        stamping sweep can fix. The single definition both the census gauge
        and the tripwire's census split read.
        """
        return self.derived_from == QUERY_SURFACE_DERIVATION

    @property
    def item_key(self) -> str:
        """This topic's stable tripwire item key."""
        return f'{TRIPWIRE_ITEM_PREFIX}{self.topic}'


@dataclass(frozen=True)
class TopicRegistry:
    """The committed set of probed topics."""

    schema_version: int
    entries: tuple[RegistryEntry, ...]
    disclosures: dict[str, int] = field(default_factory=dict)
    """What the probed set does NOT cover, from the payload's ``_disclosures``.

    Two narrowings stack here. Derivation narrows its inputs (the count-1
    census tail above all), and then hand-selection narrows derivation's
    candidates down to the committed fixture. A narrowing recorded only in
    the deriver's stdout is invisible by the time anyone reads a run, so the
    fixture carries both and the report states them in the same place as the
    results.
    """

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
    'claim_queries', 'members', 'supersedes_pairs', 'search_scope', 'provenance',
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


def _parse_scope_dimension(
    topic: str, raw: dict, key: str, vocabulary: tuple[str, ...],
) -> tuple[str, ...]:
    values = raw.get(key, [])
    if not isinstance(values, list):
        raise _fail(topic, f"'search_scope.{key}' must be a list, got {type(values).__name__}.")
    for value in values:
        if not isinstance(value, str) or not value:
            raise _fail(topic, f"'search_scope.{key}' holds {value!r}, not a non-empty string.")
        if value not in vocabulary:
            raise _fail(
                topic, f"'search_scope.{key}' names {value!r}, not one of {list(vocabulary)}.",
            )
    return tuple(values)


def _parse_search_scope(topic: str, raw: Any) -> SearchScope | None:
    if raw is None:
        return None
    if not isinstance(raw, dict):
        raise _fail(topic, f"'search_scope' must be an object, got {type(raw).__name__}.")
    unknown = sorted(set(raw) - {'stores', 'categories'})
    if unknown:
        # Strict, unlike the entry's additive tolerance: a misspelt dimension
        # dropped here would silently widen the search it was meant to narrow.
        raise _fail(topic, f"'search_scope' carries unknown key(s) {unknown}.")
    scope = SearchScope(
        stores=_parse_scope_dimension(topic, raw, 'stores', search_stores()),
        categories=_parse_scope_dimension(topic, raw, 'categories', corpus_categories()),
    )
    if not scope.stores and not scope.categories:
        raise _fail(
            topic,
            "'search_scope' declares neither stores nor categories. An empty scope "
            'searches unscoped; omit the key to say that.',
        )
    return scope


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
    if not REGISTRY_SLUG_RE.match(topic):
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
        search_scope=_parse_search_scope(topic, raw.get('search_scope')),
        provenance=dict(provenance),
        extra={k: v for k, v in raw.items() if k not in _ENTRY_KNOWN_KEYS},
    )


def _parse_disclosures(path: Path, raw: Any) -> dict[str, int]:
    """The payload's ``_disclosures`` block, or a named failure.

    Strict, not tolerant, and deliberately unlike the entry loader's
    additive tolerance. A disclosure is the record of what derivation LEFT
    OUT; dropping a malformed one would delete the very statement that a
    narrowing happened, and the report would then render "what derivation
    left out" without the line — indistinguishable from a derivation that
    left nothing out. So a non-int value (a hand-merged ``"41"``, say) names
    itself here rather than vanishing.

    Absent is legal and means "this registry records no narrowing"; a
    disclosure block is not something every registry has to carry.
    """
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise RegistryError(
            f'topic registry {str(path)!r}: "_disclosures" must be an object mapping '
            f'a narrowing to its count, got {type(raw).__name__}.'
        )
    disclosures: dict[str, int] = {}
    for key, value in raw.items():
        # bool is an int subclass; a True here is a miscount, not a count.
        if not isinstance(value, int) or isinstance(value, bool):
            raise RegistryError(
                f'topic registry {str(path)!r}: disclosure {key!r} is '
                f'{value!r} ({type(value).__name__}), not an integer count. '
                'Dropping it would erase the record that derivation narrowed '
                'its inputs, which reads downstream as "nothing was left out".'
            )
        disclosures[str(key)] = value
    return disclosures


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
    # BEFORE any entry is parsed, deliberately. A future schema_version=2
    # registry whose entry shape changed would otherwise fail on the first
    # entry with "topic 'x': required field 'canonical' is missing" — a
    # message that names the wrong cause and sends an operator to edit an
    # entry that is correct for its own version. Version first, shape second.
    schema_version = payload.get('schema_version', REGISTRY_SCHEMA_VERSION)
    if schema_version != REGISTRY_SCHEMA_VERSION:
        raise RegistryError(
            f'topic registry {str(path)!r} declares schema_version={schema_version!r}, '
            f'but this loader understands {REGISTRY_SCHEMA_VERSION}.'
        )
    # A stale, truncated, or half-written registry decodes as a well-formed
    # object carrying zero entries. Loading it would emit `"metrics": []` and
    # exit 0 — silence leaf alpha cannot distinguish from a healthy run, and an
    # artifact the first-run check counts, which would burn the D1
    # initial-state snapshot on an empty artifact. Fail at the door instead.
    if not raw_entries:
        raise RegistryError(
            f'topic registry {str(path)!r} carries zero entries. An artifact '
            'reporting zero topics is indistinguishable downstream from a '
            'healthy run, and it would consume the one-shot initial-state '
            'snapshot.'
        )

    entries = [_parse_entry(raw, i) for i, raw in enumerate(raw_entries)]

    seen: dict[str, tuple[int, str]] = {}
    for i, entry in enumerate(entries):
        folded = normalize_topic(entry.topic)
        if folded in seen:
            earlier_index, earlier_topic = seen[folded]
            raise _fail(
                entry.topic,
                f'duplicate topic: index {earlier_index} already carries '
                f'{earlier_topic!r}, and both fold to {folded!r} under normalize_topic. '
                'The slug is the persisted tripwire item key, so a duplicate would make '
                'one of the two invisible to the grandfather set, and a spelling-variant '
                'pair would measure one topic twice.',
            )
        seen[folded] = (i, entry.topic)

    return TopicRegistry(
        schema_version=schema_version,
        entries=tuple(entries),
        disclosures=_parse_disclosures(path, payload.get('_disclosures')),
    )
