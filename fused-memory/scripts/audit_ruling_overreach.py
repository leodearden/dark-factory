#!/usr/bin/env python3
"""Re-measure and adjudicate ruling-scope overreach in extracted edges.

Task 4716 / esc-4639-1. Overreach is an edge whose fact asserts more than the
source episode decided: a holding generalised past its scope, a rejected
option stated as adopted, a dropped qualifier. This instrument makes every
number in docs/ruling-overreach-guard-design-2026-10-01/design.md
re-derivable. It does four things:

* census the population under each candidate ruling-shape classifier;
* census what the existing write-time detectors would catch;
* emit a deterministic, stratified, out-of-sample adjudication worksheet;
* validate a hand-adjudicated verdict file and report rates with Wilson CIs.

Read-only by construction: the graph is read over GRAPH.RO_QUERY only, through
fused_memory.backends.graphiti_client._paged_ro_query, never through
MemoryService, GraphitiBackend or graphiti_core's FalkorDriver (see
scripts/audit_wrong_binding_edges.py::RO_COMMAND for why each is excluded).
There is no mutation flag.

Usage (from fused-memory/)::

    uv run python scripts/audit_ruling_overreach.py \\
        --graph dark_factory --graph reify \\
        --emit-worksheet /tmp/ruling-overreach-worksheet-4716.jsonl

    uv run python scripts/audit_ruling_overreach.py \\
        --graph dark_factory --graph reify \\
        --verdicts ../docs/ruling-overreach-guard-design-2026-10-01/verdicts.json \\
        --out-dir ../docs/ruling-overreach-guard-design-2026-10-01

Exit codes: 0 ran; 1 a read was incomplete or failed (NOTHING is written: a
truncated report that looks complete is worse than none); 2 the verdict file
failed validation (nothing is written).
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import logging
import math
import os
import re
import sys
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable, Iterator, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from types import MappingProxyType
from typing import Any

from fused_memory.backends.graphiti_client import (
    _DEFAULT_READ_PAGE_SIZE,
    _RESULTSET_SIZE,
    PagedRead,
    _paged_ro_query,
)
from fused_memory.reconciliation.task_filter import (
    is_batch_plan_framing,
    is_proposed_resolution_framing,
)
from fused_memory.services.completion_claim_gate import (
    UNVERIFIED_CLAIM_TAG,
    extract_completion_claims,
)

# --------------------------------------------------------------------------- #
# Records
# --------------------------------------------------------------------------- #

ADD_MEMORY_PREFIX = 'add_memory:'
_TAG_GROUP = re.compile(r'\[([^\]]+)\]\s*')


@dataclass(frozen=True)
class SourceDescription:
    """An episode's ``source_description``, parsed once at the read boundary.

    The writers compose it as a string:
    services/memory_service.py::MemoryService.add_memory writes
    ``add_memory:<category>``, and
    backends/graphiti_client.py::GraphitiBackend.add_episode prefixes
    ``[temporal:X] `` and then ``[unverified_claim] `` outermost. The stored
    artefact is that string, so it is parsed here; the build follow-up carries
    the verdict structurally on the write payload instead of re-parsing.
    """

    category: str | None
    tags: frozenset[str]


def parse_source_description(raw: str | None) -> SourceDescription:
    """Split leading ``[tag] `` groups off *raw*, then read ``add_memory:<category>``."""
    rest = raw or ''
    tags: list[str] = []
    while match := _TAG_GROUP.match(rest):
        tags.append(match.group(1))
        rest = rest[match.end():]
    category = rest[len(ADD_MEMORY_PREFIX):] if rest.startswith(ADD_MEMORY_PREFIX) else ''
    return SourceDescription(category=category or None, tags=frozenset(tags))


@dataclass(frozen=True)
class Episode:
    graph: str
    uuid: str
    created_at: str
    source: SourceDescription
    content: str


@dataclass(frozen=True)
class Edge:
    graph: str
    uuid: str
    fact: str
    source_name: str
    target_name: str
    episodes: tuple[str, ...]
    invalid_at: str | None
    expired_at: str | None

    @property
    def served(self) -> bool:
        """Every read path on main filters ``invalid_at IS NULL`` and nothing else."""
        return self.invalid_at is None

    @property
    def live_strict(self) -> bool:
        """Both stamps null: comparable with the esc-4639-1 '92.3% still live'."""
        return self.invalid_at is None and self.expired_at is None


# --------------------------------------------------------------------------- #
# Edge attribution: minted versus corroborated
# --------------------------------------------------------------------------- #

EpisodeKey = tuple[str, str]
"""``(graph, episode_uuid)``."""


@dataclass(frozen=True)
class EpisodeEdges:
    minted: tuple[Edge, ...]
    corroborated: tuple[Edge, ...]


_NO_EDGES = EpisodeEdges(minted=(), corroborated=())


@dataclass(frozen=True)
class Attribution:
    by_episode: Mapping[EpisodeKey, EpisodeEdges]
    unattributed: int = 0

    def edges_of(self, graph: str, episode_uuid: str) -> EpisodeEdges:
        return self.by_episode.get((graph, episode_uuid), _NO_EDGES)


def attribute_edges(edges: Iterable[Edge]) -> Attribution:
    """Group *edges* by the episode that minted them and the ones that corroborate.

    ``episodes[0]`` is the minting episode because graphiti_core's
    utils/maintenance/edge_operations.py::resolve_extracted_edge APPENDS the
    current episode to an existing edge on a dedupe hit.
    """
    minted: defaultdict[EpisodeKey, list[Edge]] = defaultdict(list)
    corroborated: defaultdict[EpisodeKey, list[Edge]] = defaultdict(list)
    unattributed = 0
    for edge in edges:
        if not edge.episodes:
            unattributed += 1
            continue
        minter = edge.episodes[0]
        minted[(edge.graph, minter)].append(edge)
        for later in dict.fromkeys(edge.episodes[1:]):
            if later != minter:
                corroborated[(edge.graph, later)].append(edge)
    by_episode = {
        key: EpisodeEdges(tuple(minted.get(key, ())), tuple(corroborated.get(key, ())))
        for key in sorted(minted.keys() | corroborated.keys())
    }
    return Attribution(by_episode=MappingProxyType(by_episode), unattributed=unattributed)


# --------------------------------------------------------------------------- #
# Candidate ruling-shape classifiers, under evaluation
# --------------------------------------------------------------------------- #
# They live here, not in fused_memory src, until design.md picks one; the
# build promotes the winner and this script imports it (one copy, SPOT).

HEAD_CHARS = 200
DECISIONS_CATEGORY = 'decisions_and_rationale'
_RULING_LEXEME = re.compile(r'\b(ruling|ruled)\b', re.IGNORECASE)
_DECISION_LEXEME = re.compile(r'\b(ruling|ruled|decision|decisions|decided)\b', re.IGNORECASE)
_DECISION_ANCHOR = re.compile(r'\besc-\d+-\d+\b|\([^)]*\b\d{4}-\d{2}-\d{2}')
_RULING_HEADER = re.compile(r'\s*RULING\b', re.IGNORECASE)


def _head(episode: Episode) -> str:
    return (episode.content or '')[:HEAD_CHARS]


def _category_decisions(episode: Episode) -> bool:
    return episode.source.category == DECISIONS_CATEGORY


def _header_ruling_paren(episode: Episode) -> bool:
    return (episode.content or '').lstrip().startswith('RULING (')


def _header_ruling(episode: Episode) -> bool:
    return _RULING_HEADER.match(episode.content or '') is not None


def _ruling_lexeme_head(episode: Episode) -> bool:
    return _RULING_LEXEME.search(_head(episode)) is not None


def _decision_anchor_head(episode: Episode) -> bool:
    head = _head(episode)
    return bool(_DECISION_LEXEME.search(head) and _DECISION_ANCHOR.search(head))


CLASSIFIERS: Mapping[str, Callable[[Episode], bool]] = MappingProxyType({
    'category_decisions': _category_decisions,
    'header_ruling_paren': _header_ruling_paren,
    'header_ruling': _header_ruling,
    'ruling_lexeme_head': _ruling_lexeme_head,
    'decision_anchor_head': _decision_anchor_head,
})

STRATA: tuple[str, ...] = ('ruling_lexeme', 'decision_anchor', 'other_decisions')


def stratum_of(episode: Episode) -> str | None:
    """The first stratum in ``STRATA`` order whose classifiers hold."""
    if _header_ruling(episode) or _ruling_lexeme_head(episode):
        return 'ruling_lexeme'
    if _decision_anchor_head(episode):
        return 'decision_anchor'
    if _category_decisions(episode):
        return 'other_decisions'
    return None


def matching_classifiers(episode: Episode) -> tuple[str, ...]:
    return tuple(name for name, holds in CLASSIFIERS.items() if holds(episode))


@dataclass(frozen=True)
class Specimen:
    graph: str
    episode_uuid: str
    edge_uuid: str
    note: str


SPECIMENS: tuple[Specimen, ...] = (
    Specimen(
        'reify', '59d2d750-4042-4e58-a893-798f5c4fd2c1',
        '4f99fbf2-3608-4437-aba5-eeeb99ddf991',
        'Implication not adopted: the ruling narrows orient_exp/transform_exp; '
        'the edge extends it to sin, which the record weighs as a counter-signal '
        'and leaves accepting both.',
    ),
    Specimen(
        'reify', '5c0884a3-1572-4afe-b7bc-b4786b1095cd',
        'c6ac6d99-a98f-4f52-a59d-0bbbbfadd0e1',
        'AUTHORED overreach: the episode itself states the over-assertion '
        '(task 4639 details), so the edge is a faithful extraction and no '
        'extraction-side lever or episode-only rubric can see it.',
    ),
    Specimen(
        'dark_factory', '9b33077f-03a9-49e1-ac80-09de623c3d1b',
        'b2267a98-49db-49ae-862c-75d42439ddd1',
        'Dropped qualifiers: "generalizes the rule to live unblock sessions" '
        'loses the three qualifiers its sibling edge 92a1bbda keeps.',
    ),
    Specimen(
        'dark_factory', 'cf03f276-1351-4861-be67-c7799098f4fb',
        'df7b7746-066e-4380-b4ea-f684cac1c6d0',
        'Rejected option stated as adopted: the record reframes recovery as '
        'rebase-first; the edge keeps "discard and re-dispatch" as the strategy.',
    ),
)


def specimen_recall(
    classifier_name: str, episodes: Mapping[EpisodeKey, Episode]
) -> float | None:
    """Fraction of the SPECIMENS present in *episodes* that the classifier matches."""
    present = [
        episodes[(s.graph, s.episode_uuid)]
        for s in SPECIMENS if (s.graph, s.episode_uuid) in episodes
    ]
    if not present:
        return None
    holds = CLASSIFIERS[classifier_name]
    return round(sum(1 for episode in present if holds(episode)) / len(present), 4)


# --------------------------------------------------------------------------- #
# Counterfactual census of the write-time detectors — IMPORTED, never re-derived
# --------------------------------------------------------------------------- #

DETECTORS: tuple[str, ...] = (
    'unverified_claim_tag', 'completion_claim', 'proposed_resolution', 'batch_plan',
)

WIRED_ON_ADD_MEMORY: frozenset[str] = frozenset({'unverified_claim_tag', 'completion_claim'})
"""What actually runs on the add_memory path after task 4715: the completion-claim
gate and the tag it writes. The two framing detectors were deliberately left
unwired (esc-4715-5), so their columns are counterfactual."""

KNOWN_PROJECT_IDS: frozenset[str] = frozenset({'dark_factory', 'reify'})


def detector_hits(
    episode: Episode, known_project_ids: frozenset[str] = KNOWN_PROJECT_IDS
) -> frozenset[str]:
    content = episode.content or ''
    fired = {
        'unverified_claim_tag': UNVERIFIED_CLAIM_TAG in episode.source.tags,
        'completion_claim': bool(extract_completion_claims(
            content, default_project_id=episode.graph,
            known_project_ids=known_project_ids,
        )),
        'proposed_resolution': is_proposed_resolution_framing(content),
        'batch_plan': is_batch_plan_framing(content),
    }
    return frozenset(name for name in DETECTORS if fired[name])


def detector_census(
    episodes_by_stratum: Mapping[str, Iterable[Episode]],
) -> dict[str, dict[str, int]]:
    """Per stratum: how many episodes, and how many each detector fires on."""
    census: dict[str, dict[str, int]] = {}
    for stratum, episodes in episodes_by_stratum.items():
        counts = dict.fromkeys(('episodes', *DETECTORS), 0)
        for episode in episodes:
            counts['episodes'] += 1
            for name in detector_hits(episode):
                counts[name] += 1
        census[stratum] = counts
    return census


# --------------------------------------------------------------------------- #
# The deterministic, stratified, out-of-sample adjudication sample
# --------------------------------------------------------------------------- #

DEFAULT_WINDOW_START = '2026-08-25T00:00:00+00:00'
"""The day after the esc-4639-1 ruling: disjoint from the 30 episodes adjudicated before it."""

DEFAULT_WINDOW_END = '2026-10-01T00:00:00+00:00'
"""Frozen, so later writes cannot shift the sample."""

DEFAULT_CAP = 8
"""Per graph per stratum: at most 48 episodes, roughly 300 minted edges."""

HASH_RULE = 'sha256(graph:uuid)'


def sample_key(graph: str, uuid: str) -> str:
    return hashlib.sha256(f'{graph}:{uuid}'.encode()).hexdigest()


def _instant(iso: str) -> datetime:
    """Parse an ISO timestamp; a naive one is UTC, which is how graphiti writes them."""
    parsed = datetime.fromisoformat(iso)
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)


def _created_instant(episode: Episode) -> datetime | None:
    try:
        return _instant(episode.created_at)
    except (TypeError, ValueError):
        return None


def in_window(episode: Episode, window_start: str, window_end: str) -> bool:
    at = _created_instant(episode)
    return at is not None and _instant(window_start) <= at < _instant(window_end)


@dataclass(frozen=True)
class SampledEpisode:
    episode: Episode
    stratum: str
    sample_key: str


def select_sample(
    episodes: Iterable[Episode], *, window_start: str, window_end: str, cap: int
) -> tuple[SampledEpisode, ...]:
    """The first *cap* by ``sample_key`` of each (graph, stratum) group in the window."""
    groups: defaultdict[tuple[str, int], list[SampledEpisode]] = defaultdict(list)
    for episode in episodes:
        stratum = stratum_of(episode)
        if stratum is None or not in_window(episode, window_start, window_end):
            continue
        groups[(episode.graph, STRATA.index(stratum))].append(
            SampledEpisode(episode, stratum, sample_key(episode.graph, episode.uuid))
        )
    return tuple(
        member
        for group in sorted(groups)
        for member in sorted(groups[group], key=lambda m: m.sample_key)[:cap]
    )


@dataclass(frozen=True)
class SampleDefinition:
    """Everything that fixes the sample; the verdict file carries it."""

    window_start: str = DEFAULT_WINDOW_START
    window_end: str = DEFAULT_WINDOW_END
    cap: int = DEFAULT_CAP
    strata: tuple[str, ...] = STRATA
    hash_rule: str = HASH_RULE

    def to_dict(self) -> dict[str, Any]:
        return {
            'window_start': self.window_start, 'window_end': self.window_end,
            'cap': self.cap, 'strata': list(self.strata), 'hash_rule': self.hash_rule,
        }

    @classmethod
    def from_dict(cls, obj: Mapping[str, Any]) -> SampleDefinition:
        return cls(
            window_start=str(obj['window_start']), window_end=str(obj['window_end']),
            cap=int(obj['cap']), strata=tuple(obj['strata']), hash_rule=str(obj['hash_rule']),
        )

    def select(self, episodes: Iterable[Episode]) -> tuple[SampledEpisode, ...]:
        return select_sample(
            episodes, window_start=self.window_start, window_end=self.window_end, cap=self.cap,
        )


def _edge_row(edge: Edge) -> dict[str, Any]:
    return {
        'edge_uuid': edge.uuid, 'fact': edge.fact, 'source_name': edge.source_name,
        'target_name': edge.target_name, 'served': edge.served, 'live_strict': edge.live_strict,
    }


def worksheet_rows(
    sample: Iterable[SampledEpisode], attribution: Attribution
) -> Iterator[dict[str, Any]]:
    """One adjudication row per sampled episode. Only MINTED edges are offered."""
    for member in sample:
        episode = member.episode
        edges = attribution.edges_of(episode.graph, episode.uuid)
        yield {
            'graph': episode.graph, 'uuid': episode.uuid, 'stratum': member.stratum,
            'classifiers': list(matching_classifiers(episode)),
            'created_at': episode.created_at, 'category': episode.source.category,
            'content': episode.content,
            'minted': [_edge_row(e) for e in sorted(edges.minted, key=lambda e: e.uuid)],
            'corroborated_count': len(edges.corroborated),
        }


# --------------------------------------------------------------------------- #
# The adjudicated verdict file, validated fail-closed
# --------------------------------------------------------------------------- #

LABELS: tuple[str, ...] = (
    'holding', 'bookkeeping', 'context', 'overreach', 'misbound', 'unjudgeable',
)
"""The pre-registered vocabulary; definitions in design.md §2."""

EdgeKey = tuple[str, str]
"""``(graph, edge_uuid)``."""


class VerdictError(ValueError):
    """A verdict file the report must not be built from."""

    def __init__(self, message: str, edge_uuids: Iterable[str] = ()) -> None:
        super().__init__(message)
        self.edge_uuids: tuple[str, ...] = tuple(sorted(set(edge_uuids)))


class MalformedVerdicts(VerdictError):
    pass


class SampleMismatch(VerdictError):
    pass


class UnknownLabel(VerdictError):
    pass


class MissingRationale(VerdictError):
    pass


class DuplicateVerdict(VerdictError):
    pass


class OutOfSampleVerdict(VerdictError):
    pass


class MissingVerdicts(VerdictError):
    pass


@dataclass(frozen=True)
class Verdict:
    graph: str
    episode_uuid: str
    edge_uuid: str
    fact: str
    label: str
    rationale: str

    @property
    def key(self) -> EdgeKey:
        return (self.graph, self.edge_uuid)

    def sort_key(self) -> tuple[str, str, str]:
        return (self.graph, self.episode_uuid, self.edge_uuid)

    def to_dict(self) -> dict[str, str]:
        return {name: getattr(self, name) for name in _VERDICT_FIELDS}


_VERDICT_FIELDS = ('graph', 'episode_uuid', 'edge_uuid', 'fact', 'label', 'rationale')


@dataclass(frozen=True)
class VerdictSet:
    definition: SampleDefinition
    verdicts: tuple[Verdict, ...]
    stale: tuple[Verdict, ...]
    """Verdicts whose edge no longer exists (merged or deleted): counted, never rated."""

    @property
    def by_edge(self) -> Mapping[EdgeKey, str]:
        return MappingProxyType({v.key: v.label for v in self.verdicts})

    def to_dict(self) -> dict[str, Any]:
        every = sorted(self.verdicts + self.stale, key=Verdict.sort_key)
        return {'sample': self.definition.to_dict(), 'verdicts': [v.to_dict() for v in every]}


def _parse_definition(raw: Any) -> SampleDefinition:
    try:
        return SampleDefinition.from_dict(raw)
    except (KeyError, TypeError, ValueError) as exc:
        raise MalformedVerdicts(f'the sample block is not a SampleDefinition: {exc}') from exc


def _parse_verdict(raw: Any) -> Verdict:
    if not isinstance(raw, Mapping) or not all(
        isinstance(raw.get(name), str) for name in _VERDICT_FIELDS
    ):
        raise MalformedVerdicts(f'a verdict must carry string fields {_VERDICT_FIELDS}: {raw!r}')
    return Verdict(**{name: raw[name] for name in _VERDICT_FIELDS})


def _reject(error: type[VerdictError], offenders: list[Verdict], problem: str) -> None:
    if offenders:
        uuids = [v.edge_uuid for v in offenders]
        raise error(f'{len(set(uuids))} verdict(s) {problem}: {sorted(set(uuids))}', uuids)


def load_verdicts(
    obj: Any,
    *,
    expected_edges: Mapping[EdgeKey, str],
    existing_edges: Iterable[EdgeKey],
    definition: SampleDefinition,
) -> VerdictSet:
    """Validate a parsed verdict file against the re-derived sample.

    *expected_edges* maps each sampled minted edge to its episode uuid;
    *existing_edges* is every edge the graph still holds.
    """
    if not isinstance(obj, Mapping) or not isinstance(obj.get('verdicts'), list):
        raise MalformedVerdicts("a verdict file is {'sample': {...}, 'verdicts': [...]}")
    if _parse_definition(obj.get('sample')) != definition:
        raise SampleMismatch(
            f'the file was adjudicated under {obj["sample"]!r}, not {definition.to_dict()!r}'
        )
    verdicts = [_parse_verdict(raw) for raw in obj['verdicts']]
    _reject(UnknownLabel, [v for v in verdicts if v.label not in LABELS], f'outside {LABELS}')
    _reject(MissingRationale, [v for v in verdicts if not v.rationale.strip()], 'lack a rationale')
    judgements = Counter(v.key for v in verdicts)
    _reject(DuplicateVerdict, [v for v in verdicts if judgements[v.key] > 1], 'judge one edge twice')

    existing = set(existing_edges)
    stale = [v for v in verdicts if v.key not in expected_edges and v.key not in existing]
    in_sample = [v for v in verdicts if expected_edges.get(v.key) == v.episode_uuid]
    stray = [v for v in verdicts if v not in stale and v not in in_sample]
    _reject(OutOfSampleVerdict, stray, 'judge an edge the sample did not offer')
    judged = {v.key for v in in_sample}
    missing = [edge_uuid for graph, edge_uuid in expected_edges if (graph, edge_uuid) not in judged]
    if missing:
        raise MissingVerdicts(f'{len(missing)} sampled edge(s) have no verdict', missing)
    return VerdictSet(
        definition=definition,
        verdicts=tuple(sorted(in_sample, key=Verdict.sort_key)),
        stale=tuple(sorted(stale, key=Verdict.sort_key)),
    )


# --------------------------------------------------------------------------- #
# Rates
# --------------------------------------------------------------------------- #

WILSON_Z = 1.959963984540054
"""Two-sided 95%; reproduces esc-4639-1's published intervals exactly."""

Row = Mapping[str, Any]
"""A worksheet row, as :func:`worksheet_rows` yields it."""


def wilson_interval(k: int, n: int) -> tuple[float, float] | None:
    if n <= 0:
        return None
    p, z2 = k / n, WILSON_Z ** 2
    denominator = 1 + z2 / n
    centre = (p + z2 / (2 * n)) / denominator
    half = WILSON_Z * math.sqrt(p * (1 - p) / n + z2 / (4 * n * n)) / denominator
    return (round(max(0.0, centre - half), 4), round(min(1.0, centre + half), 4))


def rate(k: int, n: int) -> dict[str, Any]:
    """``k/n`` with its Wilson interval; both None when ``n`` is 0, never 0%."""
    interval = wilson_interval(k, n)
    return {
        'k': k, 'n': n, 'rate': round(k / n, 4) if n else None,
        'ci': list(interval) if interval else None,
    }


def _judged(rows: Iterable[Row], labels: Mapping[EdgeKey, str]) -> list[tuple[Row, Row, str]]:
    """``(row, minted_edge, label)`` for every minted edge that carries a verdict."""
    return [
        (row, minted, labels[(row['graph'], minted['edge_uuid'])])
        for row in rows for minted in row['minted']
        if (row['graph'], minted['edge_uuid']) in labels
    ]


def _fraction_by_label(judged: list[tuple[Row, Row, str]], flag: str) -> dict[str, Any]:
    return {
        label: rate(
            sum(1 for _, minted, lab in judged if lab == label and minted[flag]),
            sum(1 for _, _, lab in judged if lab == label),
        )
        for label in LABELS
    }


def _rates_over(rows: list[Row], labels: Mapping[EdgeKey, str]) -> dict[str, Any]:
    judged = _judged(rows, labels)
    counts = Counter(label for _, _, label in judged)
    minted = len(judged)
    hit_keys = {(row['graph'], row['uuid']) for row, _, lab in judged if lab == 'overreach'}
    hit_rows = [r for r in rows if (r['graph'], r['uuid']) in hit_keys]
    in_hit = _judged(hit_rows, labels)
    substantive = minted - counts['bookkeeping'] - counts['unjudgeable']
    return {
        'episodes': len(rows),
        'minted': minted,
        'label_counts': {label: counts[label] for label in LABELS},
        'overreach_rate_minted': rate(counts['overreach'], minted),
        'overreach_rate_substantive': rate(counts['overreach'], substantive),
        'episode_hit_rate': rate(len(hit_rows), len(rows)),
        'holding_share': rate(counts['holding'], minted),
        'holding_share_in_hit_episodes': rate(
            sum(1 for _, _, lab in in_hit if lab == 'holding'), len(in_hit),
        ),
        'served_fraction_by_label': _fraction_by_label(judged, 'served'),
        'live_strict_fraction_by_label': _fraction_by_label(judged, 'live_strict'),
        'misbound_count': counts['misbound'],
    }


def adjudicated_rates(rows: Iterable[Row], verdicts: VerdictSet) -> dict[str, Any]:
    """The esc-4639-1 measures, per stratum and rolled up under ``'all'``."""
    rows, labels = list(rows), verdicts.by_edge
    rates = {s: _rates_over([r for r in rows if r['stratum'] == s], labels) for s in STRATA}
    rates['all'] = _rates_over(rows, labels)
    return rates


_PER_CLASSIFIER_KEYS = (
    'episodes', 'minted', 'overreach_rate_minted', 'overreach_rate_substantive',
    'episode_hit_rate',
)


def per_classifier_rates(rows: Iterable[Row], verdicts: VerdictSet) -> dict[str, Any]:
    """Overreach over the sampled episodes each classifier matches (they overlap)."""
    rows, labels = list(rows), verdicts.by_edge
    rates = {}
    for name in CLASSIFIERS:
        full = _rates_over([r for r in rows if name in r['classifiers']], labels)
        rates[name] = {key: full[key] for key in _PER_CLASSIFIER_KEYS}
    return rates


def detector_catch(
    rows: Iterable[Row],
    verdicts: VerdictSet,
    hits_by_episode: Mapping[EpisodeKey, frozenset[str]],
) -> dict[str, Any]:
    """Of the adjudicated overreach edges, how many a detector fires on the episode of."""
    sources = [
        hits_by_episode.get((row['graph'], row['uuid']), frozenset())
        for row, _, label in _judged(rows, verdicts.by_edge) if label == 'overreach'
    ]
    return {
        'overreach_edges': len(sources),
        'by_detector': {d: sum(1 for hits in sources if d in hits) for d in DETECTORS},
        'any_detector': sum(1 for hits in sources if hits),
        'any_wired': sum(1 for hits in sources if hits & WIRED_ON_ADD_MEMORY),
    }


# --------------------------------------------------------------------------- #
# The read seam: GRAPH.RO_QUERY through the shared paged primitive
# --------------------------------------------------------------------------- #

DEFAULT_GRAPHS: tuple[str, ...] = ('dark_factory', 'reify')

_EPISODE_MATCH = 'MATCH (e:Episodic) '
EPISODE_PAGE_CYPHER = (
    _EPISODE_MATCH + 'RETURN e.uuid, e.source_description, e.created_at, e.content '
    'ORDER BY e.uuid SKIP {skip} LIMIT {limit}'
)
EPISODE_CENSUS_CYPHER = _EPISODE_MATCH + 'RETURN count(*)'

_EDGE_MATCH = 'MATCH (a)-[r:RELATES_TO]->(b) '
EDGE_PAGE_CYPHER = (
    _EDGE_MATCH + 'RETURN r.uuid, r.fact, a.name, b.name, r.episodes, r.invalid_at, '
    'r.expired_at ORDER BY r.uuid SKIP {skip} LIMIT {limit}'
)
"""ALL edges, live or not: durability is a measured output. No embedding projected.
The directed pattern yields one row per edge, so ``r.uuid`` is a total order (see
graphiti_client.py::_paged_ro_query on why that is load-bearing)."""
EDGE_CENSUS_CYPHER = _EDGE_MATCH + 'RETURN count(*)'


def _text(value: Any) -> str | None:
    return None if value is None else str(value)


def _episode_from_row(graph: str, row: list) -> Episode:
    uuid, source, created_at, content = (list(row) + [None] * 4)[:4]
    return Episode(
        graph=graph, uuid=str(uuid), created_at=_text(created_at) or '',
        source=parse_source_description(_text(source)), content=_text(content) or '',
    )


def _edge_from_row(graph: str, row: list) -> Edge:
    uuid, fact, source, target, episodes, invalid_at, expired_at = (list(row) + [None] * 7)[:7]
    return Edge(
        graph=graph, uuid=str(uuid), fact=_text(fact) or '', source_name=_text(source) or '',
        target_name=_text(target) or '', episodes=tuple(str(e) for e in episodes or ()),
        invalid_at=_text(invalid_at), expired_at=_text(expired_at),
    )


class GraphReader:
    """One graph's episodes and edges, each with its completeness proof.

    Read-only rests on ``_paged_ro_query`` issuing ``ro_query`` (GRAPH.RO_QUERY is
    server-enforced read-only); the tests' double raises on ``query``. The
    RO-proxy seam copied into three other scripts is deliberately not copied a
    fourth time: consolidating it is a filed follow-up.
    """

    def __init__(
        self, *, graph: Any | None = None, graph_name: str = DEFAULT_GRAPHS[0],
        uri: str | None = None, page_size: int = _DEFAULT_READ_PAGE_SIZE,
        resultset_size: int = _RESULTSET_SIZE,
    ) -> None:
        self._graph = graph
        self.graph_name = graph_name
        self.uri = uri
        self.page_size = page_size
        self.resultset_size = resultset_size

    def _resolve_graph(self) -> Any:
        """Open a ``falkordb.asyncio`` client lazily, never graphiti's FalkorDriver."""
        if self._graph is None:
            from falkordb.asyncio import FalkorDB  # noqa: PLC0415

            client = FalkorDB.from_url(self.uri) if self.uri else FalkorDB()
            self._graph = client.select_graph(self.graph_name)
        return self._graph

    async def _read(self, page: str, census: str) -> PagedRead:
        return await _paged_ro_query(
            self._resolve_graph(), page, census,
            page_size=self.page_size, resultset_size=self.resultset_size,
        )

    async def fetch_episodes(self) -> tuple[list[Episode], PagedRead]:
        read = await self._read(EPISODE_PAGE_CYPHER, EPISODE_CENSUS_CYPHER)
        return [_episode_from_row(self.graph_name, row) for row in read.rows], read

    async def fetch_edges(self) -> tuple[list[Edge], PagedRead]:
        read = await self._read(EDGE_PAGE_CYPHER, EDGE_CENSUS_CYPHER)
        return [_edge_from_row(self.graph_name, row) for row in read.rows], read


@dataclass(frozen=True)
class Corpus:
    graphs: tuple[str, ...]
    episodes: tuple[Episode, ...]
    edges: tuple[Edge, ...]
    reads: Mapping[str, Mapping[str, PagedRead]]
    """graph -> {'episodes' | 'edges': PagedRead}."""

    def incomplete_reads(self) -> list[str]:
        return [
            f'{graph}/{kind}: {read.reason}'
            for graph, by_kind in self.reads.items()
            for kind, read in by_kind.items() if not read.complete
        ]


async def read_corpus(graphs: Iterable[str], make_reader: Callable[[str], Any]) -> Corpus:
    episodes: list[Episode] = []
    edges: list[Edge] = []
    reads: dict[str, dict[str, PagedRead]] = {}
    for graph in graphs:
        reader = make_reader(graph)
        graph_episodes, episode_read = await reader.fetch_episodes()
        graph_edges, edge_read = await reader.fetch_edges()
        episodes.extend(graph_episodes)
        edges.extend(graph_edges)
        reads[graph] = {'episodes': episode_read, 'edges': edge_read}
    return Corpus(tuple(reads), tuple(episodes), tuple(edges), MappingProxyType(reads))


# --------------------------------------------------------------------------- #
# The report
# --------------------------------------------------------------------------- #

PRIOR_MEASUREMENT: Mapping[str, Any] = MappingProxyType({
    'source': 'task 4639 details (esc-4639-1, ruled 2026-08-24): two complete-coverage '
              'passes over 30 ruling episodes',
    'episodes': 30,
    'overreach_minted': rate(13, 174),
    'overreach_live_ruling_subject': rate(10, 111),
    'overreach_rate_substantive': 0.143,
    'episode_hit_rate': 0.30,
    'holding_share_upper_bound': 0.40,
    'live_strict_fraction_by_label': {'overreach': 0.923, 'holding': 0.725},
})

CAVEATS: tuple[str, ...] = (
    'SINGLE ADJUDICATOR: one pass under the pre-registered rubric; the prior figure was '
    'two independent passes.',
    'EPISODE-ONLY RUBRIC: edges are judged against their source episode text alone, so '
    'AUTHORED overreach (specimen c6ac6d99, whose episode states the over-assertion) '
    'reads as faithful and is not counted.',
    'MINTED = episodes[0]: graphiti_core appends corroborating episodes on dedupe; only '
    'edges an episode minted are adjudicated.',
    'SERVED = invalid_at IS NULL, which is all every read path filters. expired_at alone '
    'is the restored shape task 4714 measured, and is reported as live_strict only.',
    'OUT OF SAMPLE, NOT RANDOM OVER TIME: the window starts the day after the ruling; '
    'within it, selection is by sha256(graph:uuid), stratified and capped, so strata are '
    'not population-weighted and the "all" roll-up is a sample mean, not a corpus rate.',
    'CANDIDATE CLASSIFIERS: the five regex/category predicates are under evaluation, not '
    'adopted; per-classifier rates overlap because one episode matches several.',
    'LIVE CORPUS: both graphs are written continuously; counts are a snapshot at swept_at.',
)


@dataclass(frozen=True)
class Measurement:
    corpus: Corpus
    definition: SampleDefinition
    attribution: Attribution
    rows: tuple[Mapping[str, Any], ...]

    @classmethod
    def of(cls, corpus: Corpus, definition: SampleDefinition) -> Measurement:
        attribution = attribute_edges(corpus.edges)
        rows = tuple(worksheet_rows(definition.select(corpus.episodes), attribution))
        return cls(corpus, definition, attribution, rows)

    def expected_edges(self) -> dict[EdgeKey, str]:
        return {(r['graph'], m['edge_uuid']): r['uuid'] for r in self.rows for m in r['minted']}

    def window_episodes(self) -> list[Episode]:
        d = self.definition
        return [e for e in self.corpus.episodes if in_window(e, d.window_start, d.window_end)]


def _window_days(definition: SampleDefinition) -> float:
    span = _instant(definition.window_end) - _instant(definition.window_start)
    return span.total_seconds() / 86400


def _classifier_block(m: Measurement) -> dict[str, Any]:
    window, days = m.window_episodes(), _window_days(m.definition)
    block: dict[str, Any] = {}
    for name, holds in CLASSIFIERS.items():
        block[name] = {}
        for graph in m.corpus.graphs:
            in_window_n = sum(1 for e in window if e.graph == graph and holds(e))
            block[name][graph] = {
                'population': sum(1 for e in m.corpus.episodes if e.graph == graph and holds(e)),
                'window_population': in_window_n,
                'window_per_day': round(in_window_n / days, 2) if days else None,
            }
    return block


def _specimen_block(m: Measurement) -> dict[str, Any]:
    episodes = {(e.graph, e.uuid): e for e in m.corpus.episodes}
    edges = {(e.graph, e.uuid): e for e in m.corpus.edges}
    members = []
    for s in SPECIMENS:
        episode, edge = episodes.get((s.graph, s.episode_uuid)), edges.get((s.graph, s.edge_uuid))
        members.append({
            'graph': s.graph, 'episode_uuid': s.episode_uuid, 'edge_uuid': s.edge_uuid,
            'note': s.note, 'episode_read': episode is not None,
            'classifiers': list(matching_classifiers(episode)) if episode else [],
            'stratum': stratum_of(episode) if episode else None,
            'detectors': sorted(detector_hits(episode)) if episode else [],
            'edge_read': edge is not None,
            'edge_served': edge.served if edge else None,
            'edge_live_strict': edge.live_strict if edge else None,
            'minted_by_episode': edge.episodes[:1] == (s.episode_uuid,) if edge else None,
        })
    return {'recall': {n: specimen_recall(n, episodes) for n in CLASSIFIERS}, 'members': members}


def _by_graph_and_stratum(m: Measurement, episodes: Iterable[Episode]) -> dict[str, dict]:
    grouped: dict[str, dict[str, list[Episode]]] = {
        g: {s: [] for s in STRATA} for g in m.corpus.graphs
    }
    for episode in episodes:
        stratum = stratum_of(episode)
        if stratum is not None:
            grouped[episode.graph][stratum].append(episode)
    return grouped


def _strata_block(m: Measurement) -> dict[str, Any]:
    everything = _by_graph_and_stratum(m, m.corpus.episodes)
    window = _by_graph_and_stratum(m, m.window_episodes())
    return {
        g: {s: {'population': len(everything[g][s]), 'window_population': len(window[g][s])}
            for s in STRATA}
        for g in m.corpus.graphs
    }


def _detector_block(m: Measurement) -> dict[str, Any]:
    block: dict[str, Any] = {}
    for graph, by_stratum in _by_graph_and_stratum(m, m.corpus.episodes).items():
        census = detector_census(by_stratum)
        for stratum, episodes in by_stratum.items():
            hits = [detector_hits(e) for e in episodes]
            census[stratum]['any_detector'] = sum(1 for h in hits if h)
            census[stratum]['any_wired'] = sum(1 for h in hits if h & WIRED_ON_ADD_MEMORY)
        block[graph] = census
    return block


def _sample_block(m: Measurement) -> dict[str, Any]:
    sizes = {g: {s: {'episodes': 0, 'minted': 0, 'corroborated': 0} for s in STRATA}
             for g in m.corpus.graphs}
    for row in m.rows:
        cell = sizes[row['graph']][row['stratum']]
        cell['episodes'] += 1
        cell['minted'] += len(row['minted'])
        cell['corroborated'] += row['corroborated_count']
    return {
        'definition': m.definition.to_dict(), 'sizes': sizes,
        'window_days': _window_days(m.definition),
        'unattributed_edges': m.attribution.unattributed,
    }


def _adjudicated_block(m: Measurement, verdicts: VerdictSet) -> dict[str, Any]:
    sampled = {(e.graph, e.uuid): e for e in m.corpus.episodes}
    hits = {(r['graph'], r['uuid']): detector_hits(sampled[(r['graph'], r['uuid'])])
            for r in m.rows}
    return {
        'rates': adjudicated_rates(m.rows, verdicts),
        'per_classifier': per_classifier_rates(m.rows, verdicts),
        'detector_catch': detector_catch(m.rows, verdicts, hits),
        'verdicts': len(verdicts.verdicts),
        'stale_verdicts': len(verdicts.stale),
    }


def _read_block(corpus: Corpus) -> dict[str, Any]:
    return {
        graph: {kind: {'rows_seen': r.rows_seen, 'expected_rows': r.expected_rows,
                       'complete': r.complete} for kind, r in by_kind.items()}
        for graph, by_kind in corpus.reads.items()
    }


def build_report(m: Measurement, verdicts: VerdictSet | None, *, swept_at: str) -> dict[str, Any]:
    return {
        'swept_at': swept_at,
        'graphs': list(m.corpus.graphs),
        'read_population': _read_block(m.corpus),
        'classifiers': _classifier_block(m),
        'specimens': _specimen_block(m),
        'strata': _strata_block(m),
        'detector_census': _detector_block(m),
        'wired_on_add_memory': sorted(WIRED_ON_ADD_MEMORY),
        'sample': _sample_block(m),
        'adjudicated': _adjudicated_block(m, verdicts) if verdicts else None,
        'prior_measurement': dict(PRIOR_MEASUREMENT),
        'caveats': list(CAVEATS),
    }


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

EXIT_OK, EXIT_READ_FAILED, EXIT_BAD_VERDICTS = 0, 1, 2

logger = logging.getLogger('audit_ruling_overreach')


def _build_parser() -> argparse.ArgumentParser:
    """Read-only: there is no mutation flag of any kind."""
    parser = argparse.ArgumentParser(
        prog='audit_ruling_overreach',
        description='Read-only re-measurement of ruling-scope overreach (task 4716).',
    )
    parser.add_argument('--graph', action='append', metavar='GRAPH',
                        help=f'Repeatable. Default: {" and ".join(DEFAULT_GRAPHS)}.')
    parser.add_argument('--graph-uri', default=None,
                        help='FalkorDB URI. Default: env FALKORDB_URI, then config.')
    parser.add_argument('--window-start', default=DEFAULT_WINDOW_START)
    parser.add_argument('--window-end', default=DEFAULT_WINDOW_END)
    parser.add_argument('--cap', type=int, default=DEFAULT_CAP,
                        help='Episodes per graph per stratum.')
    parser.add_argument('--emit-worksheet', metavar='PATH',
                        help='Write the adjudication worksheet as JSONL.')
    parser.add_argument('--verdicts', metavar='PATH', help='Validate and rate this verdict file.')
    parser.add_argument('--out-dir', metavar='DIR', help='Write DIR/report.json.')
    parser.add_argument('--json', action='store_true', help='Print the report on stdout.')
    return parser


def _resolve_uri(args: argparse.Namespace) -> str | None:
    """--graph-uri, then env FALKORDB_URI, then config (as audit_wrong_binding_edges.py)."""
    if args.graph_uri:
        return str(args.graph_uri)
    if env_uri := os.environ.get('FALKORDB_URI'):
        return env_uri
    try:
        from fused_memory.config.schema import FusedMemoryConfig  # noqa: PLC0415

        return getattr(getattr(FusedMemoryConfig().graphiti, 'falkordb', None), 'uri', None)
    except Exception:
        logger.warning('no FalkorDB uri from config; using the client default', exc_info=True)
        return None


def _load_verdict_file(path: str, m: Measurement) -> VerdictSet:
    try:
        obj = json.loads(Path(path).read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise MalformedVerdicts(f'cannot read {path}: {exc}') from exc
    return load_verdicts(
        obj, expected_edges=m.expected_edges(),
        existing_edges={(e.graph, e.uuid) for e in m.corpus.edges}, definition=m.definition,
    )


def _emit(args: argparse.Namespace, m: Measurement, report: dict[str, Any]) -> None:
    if args.emit_worksheet:
        path = Path(args.emit_worksheet)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(''.join(json.dumps(row, sort_keys=True) + '\n' for row in m.rows))
    blob = json.dumps(report, indent=2, sort_keys=True)
    if args.out_dir:
        out = Path(args.out_dir)
        out.mkdir(parents=True, exist_ok=True)
        (out / 'report.json').write_text(blob + '\n')
    if args.json:
        print(blob)
    else:
        sizes = report['sample']['sizes']
        sampled = sum(c['episodes'] for g in sizes.values() for c in g.values())
        print(f'episodes={len(m.corpus.episodes)} edges={len(m.corpus.edges)} '
              f'sampled={sampled} minted={len(m.expected_edges())} '
              f'adjudicated={"yes" if report["adjudicated"] else "no"}')


def _live_reader_factory(args: argparse.Namespace) -> Callable[[str], GraphReader]:
    logging.basicConfig(level=logging.INFO, stream=sys.stderr,
                        format='%(levelname)s %(name)s: %(message)s')
    uri = _resolve_uri(args)
    return lambda name: GraphReader(graph_name=name, uri=uri)


async def _run(
    args: argparse.Namespace, *, reader_factory: Callable[[str], Any] | None = None
) -> int:
    """Read, validate, then write — nothing is written unless every check passed."""
    make_reader = reader_factory or _live_reader_factory(args)
    graphs = tuple(args.graph or DEFAULT_GRAPHS)
    try:
        corpus = await read_corpus(graphs, make_reader)
    except Exception:
        logger.error('the graph read failed; nothing is written', exc_info=True)
        return EXIT_READ_FAILED
    if incomplete := corpus.incomplete_reads():
        logger.error('incomplete read(s), nothing is written: %s', incomplete)
        return EXIT_READ_FAILED

    definition = SampleDefinition(
        window_start=args.window_start, window_end=args.window_end, cap=args.cap,
    )
    measurement = Measurement.of(corpus, definition)
    verdicts = None
    if args.verdicts:
        try:
            verdicts = _load_verdict_file(args.verdicts, measurement)
        except VerdictError as exc:
            logger.error('verdict file rejected (%s): %s; edges=%s',
                         type(exc).__name__, exc, list(exc.edge_uuids))
            return EXIT_BAD_VERDICTS
    report = build_report(measurement, verdicts, swept_at=datetime.now(UTC).isoformat())
    _emit(args, measurement, report)
    return EXIT_OK


def main() -> int:
    return asyncio.run(_run(_build_parser().parse_args()))


if __name__ == '__main__':
    sys.exit(main())
