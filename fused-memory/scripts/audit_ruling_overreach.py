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

import re
from collections import defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from types import MappingProxyType

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
