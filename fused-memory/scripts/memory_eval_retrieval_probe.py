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

**When leaf γ or δ lands, split this file.** Five separable concerns live
here: the registry data contract (models + loader), offline derivation, the
pure metric math, the report renderer, and the CLI runner. Only the last
belongs in ``scripts/``, and the cost is already visible — the tests reach
this module through ``importlib.util.spec_from_file_location`` because it is
not importable, and γ/δ will want the same registry and report vocabulary. At
that point extract the registry model/loader and the pure metric functions
into ``fused_memory/eval/retrieval_probe.py`` (importable, type-checked, no
path loading) and leave this file as the thin argparse/``_run`` band, which is
what D8's runner pattern and D2's precedent (artifact shapes live in
``shared.memory_eval_metrics``, not in a runner) both point at. Deliberately
not done now: one leaf's runner is not yet a shared contract, and a second
home for the vocabulary before there is a second consumer is the drift this
note exists to prevent.
"""
from __future__ import annotations

import argparse
import asyncio
import contextlib
import hashlib
import json
import logging
import os
import re
import sys
import tempfile
import textwrap
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, NamedTuple

logger = logging.getLogger('memory_eval_retrieval_probe')

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
    # exit 0 — silence leaf alpha cannot distinguish from a healthy run, and a
    # file `is_initial_run` counts, which would burn the D1 initial-state
    # snapshot on an empty artifact. Fail at the door instead.
    if not raw_entries:
        raise RegistryError(
            f'topic registry {str(path)!r} carries zero entries. An artifact '
            'reporting zero topics is indistinguishable downstream from a '
            'healthy run, and it would consume the one-shot initial-state '
            'snapshot.'
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

    return TopicRegistry(
        schema_version=schema_version,
        entries=tuple(entries),
        disclosures=_parse_disclosures(path, payload.get('_disclosures')),
    )


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


def _derive_census_candidates(census: dict) -> _CensusDerivation:
    """Multi-entry census topics, plus every row the derivation had to drop.

    A count-1 topic has exactly one entry, so "is the canonical in the top k"
    is answered by that entry's mere existence — it measures presence, not
    retrieval. They are skipped, and the count is returned so the skip is
    reported rather than silently applied. Rows whose ``value`` or ``count``
    is the wrong shape are counted separately, for the reason
    :class:`_CensusDerivation` states.
    """
    table = ((census.get('grand_total') or {}).get('topic') or {}).get('entries') or []
    candidates: list[dict[str, Any]] = []
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
    curator_candidates, clusters_without_canonical = _derive_curator_gate_candidates(
        calibration_rows,
    )
    census = _derive_census_candidates(census_report)
    candidates = [*curator_candidates, *census.candidates]
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
    stores_served: tuple[str, ...] = ()
    """The distinct stores that answered, read off the scored top-k slice.

    ``MemoryService.search`` ROUTES: the read router picks a store set per
    query, and the lists come back homogeneous. A phrasing served entirely by
    Graphiti cannot contain a Mem0 entry's raw content — Graphiti returns
    LLM-extracted edge facts — so its canonical is unfindable there however
    healthy retrieval is.

    The probe deliberately does not pin stores: an agent's search is routed
    too, so "the router sent this query somewhere the canonical does not live"
    is a real fact about what an agent experiences. But a rate dominated by
    routing that does not SAY so is a silent fail-soft — the limits evaluator
    would compute bounds over router coin-flips — so the served set rides
    along with every observation, into the report AND into the artifact.
    """
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


UNKNOWN_STORE = 'unknown'
"""Recorded for a result carrying no ``source_store``.

Dropping it would understate the served set for exactly the shapes this probe
does not recognise — the ones most worth knowing about.
"""


def stores_served(results: list, k: int) -> tuple[str, ...]:
    """The distinct stores that answered, over the top-*k* slice only.

    Sorted so the value is stable across runs: it is compared between runs and
    an incidental ordering change would read as a routing change.
    """
    seen = {
        str(getattr(result, 'source_store', '') or UNKNOWN_STORE)
        for result in results[:k]
    }
    return tuple(sorted(seen))


def rank_index(results: list) -> dict[str, int]:
    """``{content_hash: first rank}`` over the WHOLE of *results*, hashed once.

    One search's list is read by three separate metric families — canonical
    presence at every ``k``, superseded inversions, and the comparable-pair
    exposure that inversion count is per — and each of them wants the same
    question answered: at what rank did this content hash come back. Computing
    it here means the list is sha256'd once per query rather than once per
    consumer.

    ``setdefault`` keeps the FIRST rank: a store that returned the same content
    twice has still returned it at its best rank, and taking the later one
    would report a ranking problem that is really a duplication one.
    """
    first_rank: dict[str, int] = {}
    for index, result in enumerate(results, start=1):
        first_rank.setdefault(content_key(_result_content(result)), index)
    return first_rank


def canonical_hit(
    results: list,
    entry: RegistryEntry,
    k: int,
    *,
    ranks: dict[str, int] | None = None,
) -> MatchOutcome:
    """Find *entry*'s canonical in *results*, honouring the top-*k* cut.

    Content hash first, ``last_known_id`` second (D5): memory UUIDs rot on
    re-consolidation, so hashing the returned content is the durable key, while
    the id fallback keeps a benignly-reworded canonical from reading as a
    findability regression. Which matcher fired is recorded either way, because
    both divergences are a fixture-repair signal an operator needs to see.

    The whole list is scanned, not just its first *k*, so a canonical that came
    back at rank k+3 is reported at its true rank instead of as "absent" — a
    ranking problem and an absence problem need different fixes.

    *ranks* is :func:`rank_index` of the same list, threaded in by a caller
    that already built it (the probe band calls this once per ``k``). Optional
    rather than required so every direct caller and test keeps the one-argument
    shape; computed here when absent, and the answer is identical either way.
    """
    canonical = entry.canonical
    if ranks is None:
        ranks = rank_index(results)
    hash_rank = ranks.get(canonical.content_hash)
    id_rank: int | None = None

    # Only when the primary matcher missed: the id is the FALLBACK, and its
    # rank is never read when the hash has one.
    if hash_rank is None and canonical.last_known_id:
        for index, result in enumerate(results, start=1):
            if _result_id(result) == canonical.last_known_id:
                id_rank = index
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
    ranks: dict[str, int] | None = None,
) -> PhrasingObservation:
    """Build the (topic, phrasing, k) observation the metrics aggregate over.

    *ranks* is passed straight through to :func:`canonical_hit` — see there.
    """
    outcome = canonical_hit(results, entry, k, ranks=ranks)
    return PhrasingObservation(
        topic=entry.topic,
        phrasing=phrasing.text,
        held_out=phrasing.held_out,
        k=k,
        hit=outcome.hit,
        rank=outcome.rank,
        matched_by=outcome.matched_by,
        stores_served=stores_served(results, k),
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


def comparable_pairs(
    entry: RegistryEntry, ranks: dict[str, int] | None = None, results: list | None = None,
) -> int:
    """Registry pairs with BOTH members in the returned list — the real exposure.

    This is the count metric's ``n``, and it is deliberately not
    ``len(entry.supersedes_pairs)``. :func:`superseded_inversions` can only
    ever fire on a both-present pair, so a pair with one member missing
    contributes no possibility of an event. Charging it to the denominator
    anyway makes the rate move for the wrong reason: if retrieval improves so
    that 40 pairs come back both-present instead of 4, the event count can rise
    while a registered-pair ``n`` stays pinned — and leaf α's Poisson tail test
    reads a retrieval IMPROVEMENT as a rate regression.

    Pass *ranks* (:func:`rank_index` of the same list) when the caller already
    has it; *results* is the fallback for a direct caller that does not.
    """
    if ranks is None:
        ranks = rank_index(results or [])
    return sum(
        1 for pair in entry.supersedes_pairs
        if pair.superseded_hash in ranks and pair.successor_hash in ranks
    )


def superseded_inversions(
    results: list,
    entry: RegistryEntry,
    *,
    phrasing: str = '',
    ranks: dict[str, int] | None = None,
) -> list[InversionRecord]:
    """Registry-recorded pairs where the superseded entry outranks its successor.

    Both-present-only: a pair with just one member in *results* yields nothing.
    An absent successor is a findability question ``canonical-in-top-k`` already
    measures, and counting it here as well would charge one defect against two
    metrics — inflating any downstream trend by double-weighting a single fix.
    :func:`comparable_pairs` counts the same both-present population, which is
    why it — and not the registered-pair count — is the metric's exposure.

    Rank is the position in the list the store returned, so equal
    ``relevance_score`` values resolve by that order rather than by an unstable
    re-sort. Two runs over the same list therefore produce the same count; a
    tie that flapped would look to the evaluator like a real regression.

    Records name both hashes and both ranks: a bare count tells an operator
    that something inverted but not which pair to go and look at.
    """
    if not entry.supersedes_pairs:
        return []

    first_rank = rank_index(results) if ranks is None else ranks

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


# ---------------------------------------------------------------------------
# M1 series assembly
#
# Everything above produces per-probe records; this band turns them into the
# ONE artifact leaf alpha's evaluator and the dashboard read. Every model, path
# helper and validator comes from shared.memory_eval_metrics (D2) — none of
# that shape is re-declared here.
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ClaimObservation:
    """One (topic, claim-query, k) probe result."""

    topic: str
    query: str
    k: int
    recalled: bool
    missing_needles: tuple[str, ...] = ()
    scorable: bool = True
    degraded: bool = False


@dataclass(frozen=True)
class ContaminationObservation:
    """One (topic, phrasing, k) contamination sample, disclosures included."""

    topic: str
    phrasing: str
    k: int
    foreign_count: int
    untopiced_count: int
    scored_total: int
    foreign_records: tuple[ForeignRecord, ...] = ()
    degraded: bool = False


@dataclass(frozen=True)
class InversionObservation:
    """The inversions found in one (topic, phrasing) probe, plus its exposure.

    TWO exposure numbers, deliberately. A Poisson tail test needs the exposure
    the rate is per — 2 inversions out of 4 pairs and 2 out of 400 are not the
    same observation — but it needs the exposure that could actually have
    produced an event.

    ``pairs_registered`` is every pair the registry recorded for this topic.
    ``pairs_comparable`` is the subset with BOTH members in the returned list,
    which is the only population :func:`superseded_inversions` can fire on, and
    therefore the one that becomes the metric's ``n``. Using the registered
    count instead makes a retrieval IMPROVEMENT (more pairs coming back
    both-present, so more events observable) read to the evaluator as a rate
    regression against an ``n`` that never moved.

    The registered count rides along because the ratio between the two is
    itself the signal: comparable ≪ registered means most pairs are not being
    returned at all, which is a findability fact ``canonical-in-top-k``
    measures and this metric must not be read as if it had.
    """

    topic: str
    phrasing: str
    pairs_registered: int
    pairs_comparable: int
    inversions: tuple[InversionRecord, ...] = ()
    degraded: bool = False


@dataclass(frozen=True)
class DegradedQuery:
    """One query whose search reported a failed store.

    Carries the diagnostics verbatim rather than a boolean: "the run was
    degraded" tells an operator to distrust the numbers, while "mem0 raised
    TimeoutError" tells them what to restart.
    """

    topic: str
    query: str
    failed_stores: tuple[str, ...]
    diagnostics: tuple[dict, ...] = ()


@dataclass
class ProbeObservations:
    """Everything one probe run measured, before aggregation.

    Mutable and append-only during a run; :func:`build_series` reads it without
    modifying it. Keeping the raw records around (rather than accumulating
    counters) is what lets the report name WHICH topic, phrasing and hash
    failed instead of only how many did.
    """

    phrasings: list[PhrasingObservation] = field(default_factory=list)
    claims: list[ClaimObservation] = field(default_factory=list)
    contamination: list[ContaminationObservation] = field(default_factory=list)
    inversions: list[InversionObservation] = field(default_factory=list)
    degraded_queries: list[DegradedQuery] = field(default_factory=list)


def _proportion(
    metric_id: str,
    successes: int,
    trials: int,
    direction: Literal['higher_is_worse', 'lower_is_worse'],
    *,
    details_path: str | None = None,
):
    """A proportion Metric, or None when nothing was scored.

    ``None`` rather than a 0/0 metric: the shared validator rejects a
    non-positive denominator, and rightly — a proportion over zero trials is
    not a measurement of anything, and emitting one would put a fabricated
    trial into the baseline window the evaluator computes limits from. An
    absent metric is the honest signal, and
    :func:`metric_families_not_measured` puts the absence in the report (a
    metric that vanishes without explanation reads as healthy).
    """
    from shared.memory_eval_metrics import Metric  # noqa: PLC0415

    if trials <= 0:
        return None
    return Metric(
        metric_id=metric_id,
        kind='proportion',
        value=successes / trials,
        n=trials,
        denominator=trials,
        direction=direction,
        details_path=details_path,
    )


def _tripwire_items(observations: ProbeObservations, k: int) -> list[tuple[str, bool]]:
    """(item_key, passed) per topic measured at *k*, in stable slug order.

    A topic passes only when EVERY non-degraded phrasing at *k* — including the
    held-out one — found its canonical. That conjunction is the Goodhart guard
    expressed as a predicate: tuning the known phrasings until they pass cannot
    make the topic pass while the freshly authored phrasing still misses.

    A topic whose every phrasing degraded contributes NO item at all rather
    than a failing one. A store outage must not be able to manufacture a
    corpus-wide canonical-findability collapse for leaf epsilon to file against
    the 3111 lineage.
    """
    passed: dict[str, bool] = {}
    for obs in observations.phrasings:
        if obs.degraded or obs.k != k:
            continue
        passed[obs.topic] = passed.get(obs.topic, True) and obs.hit
    return [(f'{TRIPWIRE_ITEM_PREFIX}{topic}', ok) for topic, ok in sorted(passed.items())]


def _disclosure_counts(observations: ProbeObservations) -> dict[str, int]:
    """The narrowings that must ride along INSIDE the machine-readable artifact.

    Reporting them only in prose would be a silent cap for every consumer that
    reads the JSON — which is all of them. ``corpus.counts`` is free-form
    category -> size by design (its docstring: the bucket vocabulary is not
    this schema's to own), so it is where a per-run disclosure belongs.

    **Every per-observation key names its depth.** Each query produces one
    observation per ``k``, so on the default ``(5, 10)`` run an unsuffixed
    tally counted every query twice — against a ``canonical-in-top-5`` whose
    ``n`` counts it once. Worse, ``stores_served`` is computed over
    ``results[:k]``, so a store appearing only at ranks 6-10 was credited in
    the k=10 tally and not the k=5 one: the two numbers under one key came
    from different slice depths. A consumer comparing them to a rate's
    denominator got a 2x mismatch with nothing in the artifact saying why,
    which is the uninterpretable-rate hazard this disclosure exists to
    prevent. Suffixing makes each number comparable to exactly one metric.
    """
    counts = {
        'contamination_scored_results': sum(
            c.scored_total for c in observations.contamination if not c.degraded
        ),
        'contamination_untopiced_results': sum(
            c.untopiced_count for c in observations.contamination if not c.degraded
        ),
        'claim_queries_unscorable': sum(
            1 for c in observations.claims if not c.degraded and not c.scorable
        ),
        # Per QUERY, not per observation: a degraded search is one event
        # whatever depth it is later scored at, and this is the number the
        # degraded-queries report section enumerates.
        'degraded_queries': len(observations.degraded_queries),
    }
    # Which store answered, and how many observations degraded, at each depth.
    # An observation served by a store the canonical does not live in cannot
    # hit however healthy retrieval is, so a canonical-in-top-k rate is
    # uninterpretable without this — and a consumer reading only the JSON
    # would never see it said in prose.
    for observation in observations.phrasings:
        if observation.degraded:
            key = f'degraded_observations_at_k{observation.k}'
            counts[key] = counts.get(key, 0) + 1
            continue
        for store in observation.stores_served:
            key = f'observations_served_by_{store}_at_k{observation.k}'
            counts[key] = counts.get(key, 0) + 1
    return counts


def build_series(
    observations: ProbeObservations,
    corpus_counts: dict[str, int],
    project_id: str,
    stamp: str,
    ks: tuple[int, ...] = DEFAULT_KS,
):
    """Assemble the M1 metric series for one probe run.

    Emits at most the seven metrics this leaf owns, in the pinned vocabulary.
    ``dangling-pointers`` / ``successor-pointer-present`` are leaf γ's (E4) and
    never appear here.

    The result is validated before it is returned, so an aggregation bug
    surfaces in this runner rather than in leaf α's evaluator — the M1
    "rejected at emit time, not read time" guarantee applied to the producer's
    own arithmetic.
    """
    from shared.memory_eval_metrics import (  # noqa: PLC0415
        SCHEMA_VERSION,
        Corpus,
        Metric,
        MetricSeries,
        TripwireItem,
        report_artifact_path,
        validate_metric_series,
    )

    # The report filename, not its absolute path: the artifact directory gets
    # copied and served (the dashboard reads it as plain files), and an
    # absolute path from this machine would be a dangling pointer there.
    details_path = report_artifact_path('.', EVAL_ID, stamp).name

    metrics: list[Any] = []

    items = _tripwire_items(observations, TRIPWIRE_K)
    if items:
        metrics.append(Metric(
            metric_id=METRIC_TOPIC_CANONICAL_PRESENT,
            kind='tripwire',
            value=float(sum(1 for _, ok in items if not ok)),
            n=len(items),
            items=[TripwireItem(item_key=key, passed=ok) for key, ok in items],
            details_path=details_path,
        ))

    scored_phrasings = [o for o in observations.phrasings if not o.degraded]
    for k in ks:
        at_k = [o for o in scored_phrasings if o.k == k]
        metric = _proportion(
            METRIC_CANONICAL_IN_TOP_K.format(k=k),
            sum(1 for o in at_k if o.hit),
            len(at_k),
            'lower_is_worse',
            details_path=details_path,
        )
        if metric is not None:
            metrics.append(metric)

    # Held-out only at the tripwire's k, deliberately: the two answer the same
    # question (is the canonical in this list of five) over different phrasing
    # populations, so trending them at one k keeps them comparable. Emitting a
    # held-out variant per k would also grow the metric vocabulary every time
    # someone passed another --k.
    held_out = [o for o in scored_phrasings if o.k == TRIPWIRE_K and o.held_out]
    metric = _proportion(
        METRIC_CANONICAL_IN_TOP_K_HELD_OUT.format(k=TRIPWIRE_K),
        sum(1 for o in held_out if o.hit),
        len(held_out),
        'lower_is_worse',
        details_path=details_path,
    )
    if metric is not None:
        metrics.append(metric)

    claims = [c for c in observations.claims if not c.degraded and c.scorable]
    metric = _proportion(
        METRIC_CLAIM_RECALL,
        sum(1 for c in claims if c.recalled),
        len(claims),
        'lower_is_worse',
        details_path=details_path,
    )
    if metric is not None:
        metrics.append(metric)

    contamination = [c for c in observations.contamination if not c.degraded]
    metric = _proportion(
        METRIC_CONTAMINATION_SHARE,
        sum(c.foreign_count for c in contamination),
        sum(c.scored_total for c in contamination),
        'higher_is_worse',
        details_path=details_path,
    )
    if metric is not None:
        metrics.append(metric)

    # Exposure, not observation count. `if inversions:` emitted a value=0 / n=0
    # count metric for any selection whose entries record no supersedes pair —
    # a fabricated "no inversions here" datapoint entering leaf α's baseline
    # window. That is the same hazard `_proportion` refuses a 0/0 proportion
    # for, and a count kind has no more claim to a zero-trial measurement than
    # a proportion does. Absent is the honest signal, and the report names the
    # family so the absence cannot read as health.
    inversions = [i for i in observations.inversions if not i.degraded]
    exposure = sum(i.pairs_comparable for i in inversions)
    if exposure > 0:
        metrics.append(Metric(
            metric_id=METRIC_SUPERSEDED_ABOVE_SUCCESSOR,
            kind='count',
            value=float(sum(len(i.inversions) for i in inversions)),
            n=exposure,
            direction='higher_is_worse',
            details_path=details_path,
        ))

    counts = dict(corpus_counts)
    for key, value in _disclosure_counts(observations).items():
        if key in counts:
            raise ValueError(
                f'corpus_counts key {key!r} collides with a run disclosure this '
                'runner computes. Rename the caller-supplied key: silently '
                'overwriting either one would hide a narrowing.'
            )
        counts[key] = value

    series = MetricSeries(
        schema_version=SCHEMA_VERSION,
        eval_id=EVAL_ID,
        run_stamp=stamp,
        corpus=Corpus(project_id=project_id, counts=counts),
        metrics=metrics,
    )
    validate_metric_series(series)
    return series


def pinned_metric_ids(ks: tuple[int, ...]) -> tuple[str, ...]:
    """Every metric_id a run at *ks* is expected to emit.

    THE list this leaf owns, in one place, so "which family went unmeasured"
    is answerable by comparing against it rather than by remembering what
    :func:`build_series` can emit. ``ks`` goes through :func:`normalise_ks`
    for the same reason the run does: the tripwire's depth is always measured.
    """
    return (
        METRIC_TOPIC_CANONICAL_PRESENT,
        *(METRIC_CANONICAL_IN_TOP_K.format(k=k) for k in normalise_ks(tuple(ks))),
        METRIC_CANONICAL_IN_TOP_K_HELD_OUT.format(k=TRIPWIRE_K),
        METRIC_CLAIM_RECALL,
        METRIC_CONTAMINATION_SHARE,
        METRIC_SUPERSEDED_ABOVE_SUCCESSOR,
    )


def metric_families_not_measured(series, ks: tuple[int, ...]) -> list[str]:
    """Pinned metric ids *series* does not carry, in the pinned order.

    Every family in this runner declines to emit rather than emit a
    zero-trial datapoint: ``_proportion`` refuses a 0/0 proportion, and the
    count metric refuses a zero-exposure count, because a fabricated trial in
    leaf α's baseline window is worse than a gap in it. But an absent metric
    is not an error to the evaluator either — it joins by metric_id and simply
    stops trending what is missing. So the absence is named HERE, in the run's
    own report, where the alternative is a metric that silently stops existing
    and reads to a human as one that had nothing to report.
    """
    present = {metric.metric_id for metric in series.metrics}
    return [metric_id for metric_id in pinned_metric_ids(ks) if metric_id not in present]


def not_measured_topics(
    observations: ProbeObservations, k: int = TRIPWIRE_K,
) -> list[str]:
    """Topics that were probed but produced no usable observation at *k*.

    Distinct from a failing topic and from an absent one. The report has to say
    "we could not measure this" out loud: a topic that silently vanishes from
    the tripwire's items is indistinguishable, in the artifact alone, from a
    topic that passed.
    """
    attempted: set[str] = set()
    measured: set[str] = set()
    for obs in observations.phrasings:
        if obs.k != k:
            continue
        attempted.add(obs.topic)
        if not obs.degraded:
            measured.add(obs.topic)
    return sorted(attempted - measured)


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
        # Hashed ONCE per search, then read by every consumer below: canonical
        # presence at each k, the inversions, and their comparable exposure.
        ranks = rank_index(info.results)
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
        observations.inversions.append(InversionObservation(
            topic=entry.topic,
            phrasing=phrasing.text,
            pairs_registered=len(entry.supersedes_pairs),
            pairs_comparable=comparable_pairs(entry, ranks),
            inversions=tuple(
                superseded_inversions(
                    info.results, entry, phrasing=phrasing.text, ranks=ranks,
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


def probe_report_sections(
    series,
    observations: ProbeObservations,
    *,
    is_initial_run: bool = False,
    registry: TopicRegistry | None = None,
    skipped_topics: tuple[str, ...] = (),
    requested_ks: tuple[int, ...] = (),
    measured_ks: tuple[int, ...] = (),
) -> tuple[ReportSection, ...]:
    """The report, decomposed — see :func:`render_probe_report` for the prose.

    THE single source of both: :func:`render_probe_report` joins what this
    returns, so a section present here is present there by construction and
    the two cannot drift into disagreeing about what the run disclosed.
    """
    from shared.memory_eval_metrics import render_report  # noqa: PLC0415

    sections: list[ReportSection] = []

    def emit(key: str, lines: list[str]) -> None:
        sections.append(ReportSection(key=key, lines=tuple(lines)))

    emit(SECTION_METRIC_TABLE, [render_report(series).rstrip('\n')])

    # Directly under the table it qualifies: a family missing from the rows
    # above is the one absence a reader cannot see by reading them.
    absent_families = metric_families_not_measured(
        series,
        # The depths this run actually scored, when the caller did not say.
        # Assuming DEFAULT_KS instead would name canonical-in-top-10 as
        # unmeasured on a run that never asked for k=10 — a false narrowing
        # report is no better than a hidden one.
        tuple(measured_ks) or tuple(dict.fromkeys(o.k for o in observations.phrasings)),
    )
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
            'unfindable there however healthy retrieval is. The probe does not '
            'pin stores, because an agent\'s search is routed too; it reports '
            'the routing instead, so a rate is never read as a corpus finding '
            'when it is a routing one.'
        ):
            lines.append(f'  {chunk}')
        for store, count in sorted(by_store.items()):
            lines.append(f'  {store}: {count}')
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
# initialize(), try/finally close(). MemoryService rather than Mem0Backend
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
output never lands in a diff by accident."""

DEFAULT_CALIBRATION_PATH = _PACKAGE_ROOT / 'tests' / 'fixtures' / 'write_triage_calibration.jsonl'
DEFAULT_CENSUS_PATH = _REPO_ROOT / 'plans' / 'memory-metadata-census-report.json'


def corpus_categories() -> tuple[str, ...]:
    """The category vocabulary, taken from the store's own enums.

    Derived rather than restated. The bucket vocabulary belongs to the
    memory-metadata PRD, and a copy of it here would keep reporting six
    categories on the day a seventh is added — the artifact would look
    complete while silently under-counting the corpus it claims to describe.
    """
    from fused_memory.models.enums import (  # noqa: PLC0415
        GRAPHITI_PRIMARY,
        MEM0_PRIMARY,
    )

    return tuple(sorted(c.value for c in (GRAPHITI_PRIMARY | MEM0_PRIMARY)))


def graphiti_primary_categories() -> tuple[str, ...]:
    """The categories the Mem0-side corpus count under-reports.

    Named in the report so an operator reading a near-zero count knows it is a
    counting scope, not an empty graph. Derived for the same reason
    :func:`corpus_categories` is.
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


def normalise_ks(ks: tuple[int, ...]) -> tuple[int, ...]:
    """*ks* with :data:`TRIPWIRE_K` guaranteed present, caller's order kept.

    ``--k`` is a repeatable parameterisation, so ``--k 7`` is a legitimate
    request — but two metrics are DEFINED at k=5 and cannot be computed
    without it: the ``topic-canonical-present`` tripwire and the held-out
    proportion. Without this, ``--k 7`` emitted neither, and did so silently:
    leaf α's evaluator joins a run to its baseline window BY metric_id, so an
    absent metric is not an error there, it simply stops being trended. A
    metric that quietly stops existing is worse than one that crashes.

    Normalising rather than rejecting keeps ``--k 7`` legal — the extra depth
    is genuinely useful — while making the pinned metric set a guaranteed
    superset of every run's, which is what keeps α's join keys stable. The
    added depth is disclosed in the report; it is not left to be inferred
    from the metric list.

    Deliberately the idiom already one line below in :func:`run_probe`, which
    dedups ``project_ids`` the same way.
    """
    return tuple(dict.fromkeys((*ks, TRIPWIRE_K)))


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


async def run_probe(
    memory,
    registry: TopicRegistry,
    *,
    project_ids: tuple[str, ...],
    ks: tuple[int, ...],
    out_root: str | Path,
    stamp: str | None = None,
) -> ProbeOutcome:
    """Measure *registry* against *memory* and emit the run's artifacts.

    *memory* is injected rather than constructed here, which is what lets the
    read-only guarantee be tested: the whole band runs against a double whose
    write methods raise.

    Reads only. ``search`` and ``count_memories_by_metadata`` are the only two
    methods touched.

    THE chokepoint for *ks*: normalising here means every caller — ``main``,
    ``_run``, a test, a notebook — inherits the guarantee that the pinned
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
        async def search(query: str, limit: int, _project_id: str = entry.project_id):
            return await memory.search(query, project_id=_project_id, limit=limit)

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
    metrics_path, report_path = emit_series(series, out_root, stamp=stamp)

    # emit_series wrote the shared render_report as the companion; replace it
    # with the extended one — through the same atomic path, so the widening
    # cannot leave a truncated report beside a valid metrics artifact. The
    # shared write still happens first, so emit-time validation continues to
    # gate whether any artifact is created at all — the report is only ever
    # widened over a series that already validated.
    sections = probe_report_sections(
        series,
        observations,
        is_initial_run=initial,
        # The PROBED subset here, so "registry composition (N topics)" counts
        # what this run actually measured; the rest is named by skipped_topics.
        registry=TopicRegistry(
            schema_version=registry.schema_version,
            entries=probed,
            disclosures=registry.disclosures,
        ),
        skipped_topics=skipped,
        requested_ks=tuple(ks),
        measured_ks=measured_ks,
    )
    report = join_report_sections(sections)
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


def build_parser() -> argparse.ArgumentParser:
    """The CLI. Every flag here is a read parameter; none of them mutate anything.

    There is deliberately no ``--apply``, ``--fix``, ``--prune`` or any other
    mutating band, and a test asserts this flag set by EQUALITY so one cannot
    be added without a test saying so out loud.
    """
    # The module docstring carries an RST table that argparse's default
    # formatter reflows into rubble; the first line plus the guarantee is what
    # an operator at the terminal actually needs.
    parser = argparse.ArgumentParser(
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
        '--derive-registry', dest='derive_registry', action='store_true',
        help=(
            'Print registry candidates derived from the committed offline '
            'sources and exit. Never overwrites the committed fixture — the '
            'hand-authored held-out phrasings cannot be regenerated.'
        ),
    )
    return parser


async def _run(args: argparse.Namespace) -> int:
    logging.basicConfig(
        level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s',
    )

    if args.derive_registry:
        print(run_derive_registry(DEFAULT_CALIBRATION_PATH, DEFAULT_CENSUS_PATH), end='')
        return 0

    # Before the store, deliberately. A fixture typo must not cost an embedder
    # spin-up to discover, and — the load-bearing half — a registry that failed
    # to load must never reach emission: an artifact reporting zero topics is
    # indistinguishable, downstream, from a healthy corpus that found nothing.
    try:
        registry = load_topic_registry(args.registry)
    except RegistryError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    from fused_memory.config.schema import FusedMemoryConfig  # noqa: PLC0415
    from fused_memory.services.memory_service import MemoryService  # noqa: PLC0415

    if args.config:
        os.environ['CONFIG_PATH'] = str(args.config)

    config = FusedMemoryConfig()
    memory = MemoryService(config)
    await memory.initialize()
    try:
        outcome = await run_probe(
            memory, registry,
            project_ids=tuple(args.project_id),
            ks=tuple(args.k),
            out_root=args.out_root,
        )
    except EmptySelectionError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    finally:
        await memory.close()

    print(outcome.report, end='')
    logger.info('metrics: %s', outcome.metrics_path)
    logger.info('report:  %s', outcome.report_path)
    return 0


def main(argv: list[str] | None = None) -> int:
    return asyncio.run(_run(build_parser().parse_args(argv)))


if __name__ == '__main__':
    sys.exit(main())
