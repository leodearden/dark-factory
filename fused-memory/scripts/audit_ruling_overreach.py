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

import hashlib
import re
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable, Iterator, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from types import MappingProxyType
from typing import Any

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
