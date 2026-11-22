#!/usr/bin/env python3
"""Draw the frozen, read-only sample frame and fixtures that task 9945's gpt-4o-mini vs gpt-6-luna eval runs over.

The extraction frame stratifies every non-decision add_memory write by its
category, with the 4716 audit's own hash rule and window semantics, so
replay_overreach_extraction.py replays and scores it unchanged. Graphs are read
over GRAPH.RO_QUERY only, through fused_memory/backends/falkor_read_only.py's
ReadOnlyGraphReader; episode bodies are read only for sampled episodes, and only
the worksheet carries them, never the frame. The classifier and router
fixtures come from the write journal, opened read-only, and keep only inputs
the production WriteClassifier and ReadRouter send to their LLM.
"""
from __future__ import annotations

import contextlib
import hashlib
import json
import os
import sqlite3
import sys
from collections import defaultdict
from collections.abc import Awaitable, Iterable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx
from graphiti_core.utils.maintenance.dedup_helpers import _normalize_string_exact
from graphiti_core.utils.text_utils import MAX_SUMMARY_CHARS
from openai import AsyncOpenAI

from fused_memory.backends.falkor_read_only import PagedQuery, ReadOnlyGraphReader
from fused_memory.config.schema import FusedMemoryConfig, LLMConfig, RoutingConfig
from fused_memory.routing.classifier import WriteClassifier
from fused_memory.routing.router import ReadRouter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from audit_ruling_overreach import (
    Episode,
    SampleDefinition,
    in_window,
    instant_in_window,
    parse_source_description,
    sample_key,
    stored_instant,
)

# --------------------------------------------------------------------------- #
# The extraction frame
# --------------------------------------------------------------------------- #

TEMPORAL, RELATIONS, OTHER_CATEGORIES = 'temporal_facts', 'entities_and_relations', 'other_categories'
EXTRACTION_STRATA: tuple[str, ...] = (TEMPORAL, RELATIONS, OTHER_CATEGORIES)

EXCLUDED_CATEGORY = 'decisions_and_rationale'
"""The ruling class the 4716 audit and levers O1/O6 measured; this frame is every other write."""

EPISODE_METADATA = PagedQuery(
    match='MATCH (e:Episodic)', row='e',
    returns=('e.uuid', 'e.created_at', 'e.source_description', 'e.valid_at'),
)
"""Every episode, paged and census-proven, without its body."""

EPISODE_CONTENT_CYPHER = 'MATCH (e:Episodic) WHERE e.uuid IN $uuids RETURN e.uuid, e.content'
"""The bodies of the sampled episodes only: at most cap x strata rows per graph."""


class FrameReadIncomplete(RuntimeError):
    """A graph read that cannot prove it saw every episode, so no frame is drawn from it."""


@dataclass(frozen=True)
class ExtractionFrame:
    frame: dict[str, Any]
    """frame.json: the keys replay_overreach_extraction.py::Baseline.from_report reads, and no body."""
    worksheet: tuple[dict[str, Any], ...]
    """One row per sampled episode, body included, in replay_overreach_extraction's worksheet shape."""


def extraction_stratum(category: str | None) -> str | None:
    """The frame stratum of a write's category; None for the excluded decision class."""
    if category == EXCLUDED_CATEGORY:
        return None
    return category if category in (TEMPORAL, RELATIONS) else OTHER_CATEGORIES


async def _read_complete(graph: str, reader: ReadOnlyGraphReader, query: PagedQuery) -> list[list[Any]]:
    """Every row of *query*, or FrameReadIncomplete when the read cannot prove it saw them all."""
    read = await reader.read_paged(query)
    if not read.complete:
        raise FrameReadIncomplete(f'{graph}: {read.reason}')
    return read.rows


def _metadata_episode(graph: str, row: list[Any]) -> Episode:
    uuid, created_at, source_description = (list(row) + [None] * 3)[:3]
    return Episode(
        graph=graph, uuid=str(uuid), created_at='' if created_at is None else str(created_at),
        source=parse_source_description(None if source_description is None else str(source_description)),
        content='',
    )


async def _window_cells(
    graph: str, reader: ReadOnlyGraphReader, definition: SampleDefinition,
) -> dict[str, list[Episode]]:
    """stratum -> the graph's in-window episodes of that stratum, in sample_key order."""
    cells: dict[str, list[Episode]] = {stratum: [] for stratum in definition.strata}
    for row in await _read_complete(graph, reader, EPISODE_METADATA):
        episode = _metadata_episode(graph, row)
        stratum = extraction_stratum(episode.source.category)
        if stratum is not None and in_window(episode, definition.window_start, definition.window_end):
            cells[stratum].append(episode)
    for episodes in cells.values():
        episodes.sort(key=lambda episode: sample_key(graph, episode.uuid))
    return cells


async def _contents(graph: str, reader: ReadOnlyGraphReader, uuids: list[str]) -> dict[str, str]:
    """Each sampled episode's body; one the graph no longer holds refuses the frame."""
    if not uuids:
        return {}
    rows = await reader.read_unpaged(EPISODE_CONTENT_CYPHER, {'uuids': uuids})
    contents = {str(uuid): '' if content is None else str(content) for uuid, content in rows}
    if missing := sorted(set(uuids) - contents.keys()):
        raise FrameReadIncomplete(f'{graph}: sampled episodes vanished before their bodies were read: {missing}')
    return contents


def _worksheet_row(episode: Episode, stratum: str, content: str) -> dict[str, Any]:
    return {
        'graph': episode.graph, 'uuid': episode.uuid, 'stratum': stratum,
        'category': episode.source.category, 'created_at': episode.created_at, 'content': content,
    }


async def extraction_frame(
    readers: Mapping[str, ReadOnlyGraphReader], definition: SampleDefinition,
) -> ExtractionFrame:
    """The first ``definition.cap`` in-window episodes of each (graph, stratum) cell by sample_key.

    *definition* must carry EXTRACTION_STRATA: the frame's verdict file carries
    it, and replay_overreach_extraction's score validates against it.
    """
    if tuple(definition.strata) != EXTRACTION_STRATA:
        raise ValueError(f'an extraction frame has strata {EXTRACTION_STRATA}, not {definition.strata}')
    population: dict[str, dict[str, dict[str, int]]] = {}
    sizes: dict[str, dict[str, dict[str, int]]] = {}
    worksheet: list[dict[str, Any]] = []
    for graph, reader in readers.items():
        cells = await _window_cells(graph, reader, definition)
        sampled = {stratum: episodes[: definition.cap] for stratum, episodes in cells.items()}
        contents = await _contents(graph, reader, [e.uuid for episodes in sampled.values() for e in episodes])
        population[graph] = {stratum: {'window_population': len(cells[stratum])} for stratum in definition.strata}
        sizes[graph] = {stratum: {'episodes': len(sampled[stratum])} for stratum in definition.strata}
        worksheet.extend(
            _worksheet_row(episode, stratum, contents[episode.uuid])
            for stratum in definition.strata for episode in sampled[stratum]
        )
    frame = {
        'graphs': list(readers),
        'sample': {'definition': definition.to_dict(), 'sizes': sizes, 'window_days': definition.window_days},
        'strata': population,
    }
    return ExtractionFrame(frame=frame, worksheet=tuple(worksheet))


# --------------------------------------------------------------------------- #
# Graph-derived fixtures: resolve_edge and summaries
# --------------------------------------------------------------------------- #

Window = tuple[str, str]
"""``[start, end)`` as ISO instants, compared on each record's created_at."""

RELATES_TO_READ = PagedQuery(
    match='MATCH (a:Entity)-[r:RELATES_TO]->(b:Entity)', row='r',
    returns=('r.uuid', 'a.uuid', 'b.uuid', 'r.fact', 'r.created_at', 'r.valid_at', 'r.invalid_at',
             'r.expired_at', 'r.episodes'),
)
"""Every entity edge, without its embedding."""

MENTIONS_READ = PagedQuery(
    match='MATCH (e:Episodic)-[m:MENTIONS]->(n:Entity)', row='m', returns=('m.uuid', 'e.uuid', 'n.uuid'),
)

ENTITY_READ = PagedQuery(match='MATCH (n:Entity)', row='n', returns=('n.uuid', 'n.name', 'n.summary'))

DEFAULT_SIBLING_CAP = 10
"""Duplicate candidates per fixture: the older edges on the candidate's own directed node pair,
newest first. graphiti_core 0.28.2's edge_operations.py::resolve_extracted_edges takes them from
EntityEdge.get_between_nodes, filtered and ranked by EDGE_HYBRID_SEARCH_RRF."""

DEFAULT_CANDIDATE_CAP = 10
"""Invalidation candidates per fixture: older edges incident to either endpoint and off the pair,
in sha256(graph:edge_uuid) order, approximating resolve_extracted_edges' unfiltered
EDGE_HYBRID_SEARCH_RRF search. The hybrid-search ranking is not reproduced."""

SUMMARY_LLM_THRESHOLD = 4 * MAX_SUMMARY_CHARS
"""graphiti_core 0.28.2's node_operations.py::_extract_entity_summaries_batch summarises a node by
LLM only once its summary with the episode's edge facts appended is longer than this."""


@dataclass(frozen=True)
class GraphEdge:
    uuid: str
    source: str
    target: str
    fact: str
    created_at: str | None
    valid_at: str | None
    invalid_at: str | None
    expired_at: str | None
    episodes: tuple[str, ...]

    @classmethod
    def from_row(cls, row: list[Any]) -> GraphEdge:
        uuid, source, target, fact, created, valid, invalid, expired, episodes = (list(row) + [None] * 9)[:9]
        return cls(
            uuid=str(uuid), source=str(source), target=str(target), fact='' if fact is None else str(fact),
            created_at=_text(created), valid_at=_text(valid), invalid_at=_text(invalid),
            expired_at=_text(expired), episodes=tuple(str(e) for e in episodes or ()),
        )

    @property
    def created(self) -> datetime | None:
        return stored_instant(self.created_at)

    def older_than(self, other: GraphEdge) -> bool:
        mine, theirs = self.created, other.created
        return mine is not None and theirs is not None and mine < theirs

    def to_fixture(self) -> dict[str, Any]:
        return {
            'uuid': self.uuid, 'fact': self.fact, 'source_node_uuid': self.source,
            'target_node_uuid': self.target, 'created_at': self.created_at, 'valid_at': self.valid_at,
            'invalid_at': self.invalid_at, 'expired_at': self.expired_at,
        }


def _text(value: Any) -> str | None:
    return None if value is None else str(value)


async def _graph_edges(graph: str, reader: ReadOnlyGraphReader) -> list[GraphEdge]:
    return [GraphEdge.from_row(row) for row in await _read_complete(graph, reader, RELATES_TO_READ)]


async def _episode_valid_at(graph: str, reader: ReadOnlyGraphReader) -> dict[str, tuple[str | None, str | None]]:
    """episode uuid -> (created_at, valid_at)."""
    return {
        str(row[0]): (_text(row[1]), _text(row[3]) if len(row) > 3 else None)
        for row in await _read_complete(graph, reader, EPISODE_METADATA)
    }


def _incident(edges: Iterable[GraphEdge]) -> dict[str, list[GraphEdge]]:
    by_node: defaultdict[str, list[GraphEdge]] = defaultdict(list)
    for edge in edges:
        by_node[edge.source].append(edge)
        if edge.target != edge.source:
            by_node[edge.target].append(edge)
    return by_node


def _resolve_fixture(
    graph: str, candidate: GraphEdge, pair: list[GraphEdge], incident: Mapping[str, list[GraphEdge]],
    episode_valid_at: str | None, sibling_cap: int, candidate_cap: int,
) -> dict[str, Any] | None:
    """The fixture for *candidate*, or None when graphiti would resolve it with no LLM call."""
    siblings = sorted(
        (edge for edge in pair if edge.uuid != candidate.uuid and edge.older_than(candidate)),
        key=lambda edge: edge.created or datetime.min.replace(tzinfo=UTC), reverse=True,
    )
    fact = _normalize_string_exact(candidate.fact)
    if not siblings or any(_normalize_string_exact(edge.fact) == fact for edge in siblings):
        return None
    on_pair = {edge.uuid for edge in pair}
    invalidation = sorted(
        {edge.uuid: edge for node in (candidate.source, candidate.target) for edge in incident[node]
         if edge.uuid not in on_pair and edge.older_than(candidate)}.values(),
        key=lambda edge: sample_key(graph, edge.uuid),
    )
    return {
        'graph': graph, 'edge_uuid': candidate.uuid, 'fact': candidate.fact,
        'source_node_uuid': candidate.source, 'target_node_uuid': candidate.target,
        'created_at': candidate.created_at, 'valid_at': candidate.valid_at,
        'invalid_at': candidate.invalid_at, 'expired_at': candidate.expired_at,
        'episode_uuid': candidate.episodes[0], 'episode_valid_at': episode_valid_at,
        'related': [edge.to_fixture() for edge in siblings[:sibling_cap]],
        'invalidation_candidates': [edge.to_fixture() for edge in invalidation[:candidate_cap]],
    }


async def resolve_edge_fixtures(
    readers: Mapping[str, ReadOnlyGraphReader], window: Window, n: int,
    sibling_cap: int = DEFAULT_SIBLING_CAP, candidate_cap: int = DEFAULT_CANDIDATE_CAP,
) -> list[dict[str, Any]]:
    """The first *n* window edges, by sha256(graph:edge_uuid) over every graph, that graphiti would
    resolve by LLM: each with its duplicate and invalidation candidates, and no episode body."""
    pending: list[tuple[str, str, Any]] = []
    for graph, reader in readers.items():
        edges = await _graph_edges(graph, reader)
        episodes = await _episode_valid_at(graph, reader)
        by_pair: defaultdict[tuple[str, str], list[GraphEdge]] = defaultdict(list)
        for edge in edges:
            by_pair[(edge.source, edge.target)].append(edge)
        incident = _incident(edges)
        for edge in edges:
            if edge.episodes and instant_in_window(edge.created_at, *window):
                pending.append((sample_key(graph, edge.uuid), graph, (
                    edge, by_pair[(edge.source, edge.target)], incident,
                    episodes.get(edge.episodes[0], (None, None))[1],
                )))
    fixtures: list[dict[str, Any]] = []
    for _, graph, (edge, pair, incident, valid_at) in sorted(pending, key=lambda item: item[0]):
        if len(fixtures) == n:
            break
        if fixture := _resolve_fixture(graph, edge, pair, incident, valid_at, sibling_cap, candidate_cap):
            fixtures.append(fixture)
    return fixtures


def summary_with_edges(summary: str, facts: list[str]) -> str:
    """The text _extract_entity_summaries_batch measures: the summary, then each incident edge's fact."""
    if not facts:
        return summary
    joined = '\n'.join(fact for fact in facts if fact)
    return f'{summary}\n{joined}'.strip()


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


@dataclass(frozen=True)
class _SummaryGraph:
    """One graph's episodes, mentions, entities and minted edges, as the summary items need them."""

    episodes: Mapping[str, tuple[str | None, str | None]]
    mentioned: Mapping[str, list[str]]
    entities: Mapping[str, tuple[str, str]]
    minted: Mapping[str, list[GraphEdge]]

    @classmethod
    async def read(cls, graph: str, reader: ReadOnlyGraphReader) -> _SummaryGraph:
        mentioned: defaultdict[str, list[str]] = defaultdict(list)
        for _, episode, node in await _read_complete(graph, reader, MENTIONS_READ):
            mentioned[str(episode)].append(str(node))
        minted: defaultdict[str, list[GraphEdge]] = defaultdict(list)
        for edge in await _graph_edges(graph, reader):
            if edge.episodes:
                minted[edge.episodes[0]].append(edge)
        return cls(
            episodes=await _episode_valid_at(graph, reader), mentioned=mentioned,
            entities={
                str(uuid): ('' if name is None else str(name), '' if summary is None else str(summary))
                for uuid, name, summary in await _read_complete(graph, reader, ENTITY_READ)
            },
            minted=minted,
        )

    def items(self, graph: str, episode: str, nodes_per_episode: int) -> list[dict[str, Any]]:
        items = []
        for node in sorted(set(self.mentioned.get(episode, ())), key=lambda uuid: sample_key(graph, uuid)):
            name, summary = self.entities.get(node, ('', ''))
            edges = [edge for edge in self.minted.get(episode, ()) if node in (edge.source, edge.target)]
            measured = summary_with_edges(summary, [edge.fact for edge in edges])
            if len(measured) > SUMMARY_LLM_THRESHOLD:
                items.append({
                    'graph': graph, 'episode_uuid': episode, 'episode_valid_at': self.episodes[episode][1],
                    'node_uuid': node, 'node_name': name, 'summary_sha256': _sha256(summary),
                    'edge_uuids': [edge.uuid for edge in edges], 'summary_with_edges_chars': len(measured),
                })
            if len(items) == nodes_per_episode:
                break
        return items


async def summary_fixtures(
    readers: Mapping[str, ReadOnlyGraphReader], window: Window, n: int, nodes_per_episode: int = 2,
) -> list[dict[str, Any]]:
    """The first *n* (episode, mentioned Entity) items graphiti would summarise by LLM, over window
    episodes by sha256(graph:uuid), at most *nodes_per_episode* each; ids and hashes, never text."""
    graphs = {graph: await _SummaryGraph.read(graph, reader) for graph, reader in readers.items()}
    episodes = sorted(
        ((sample_key(graph, uuid), graph, uuid) for graph, read in graphs.items()
         for uuid, (created_at, _) in read.episodes.items() if instant_in_window(created_at, *window)),
    )
    items: list[dict[str, Any]] = []
    for _, graph, episode in episodes:
        items.extend(graphs[graph].items(graph, episode, nodes_per_episode)[: n - len(items)])
        if len(items) == n:
            break
    for graph, reader in readers.items():
        chosen = list(dict.fromkeys(item['episode_uuid'] for item in items if item['graph'] == graph))
        contents = await _contents(graph, reader, chosen)
        for item in items:
            if item['graph'] == graph:
                item['episode_content_sha256'] = _sha256(contents[item['episode_uuid']])
    return items


# --------------------------------------------------------------------------- #
# Journal fixtures: the classifier and the router
# --------------------------------------------------------------------------- #

JOURNAL_ROWS_SQL = 'SELECT id, created_at, params FROM write_ops WHERE operation = ?'

_PROBE_ANSWER = {
    'primary': 'observations_and_summaries', 'confidence': 0.5,
    'query_type': 'broad', 'stores': ['graphiti', 'mem0'], 'primary_store': 'graphiti',
}
"""One answer both the classifier and the router parse, so the probe logs no parse failure."""


def open_journal(path: str | Path) -> sqlite3.Connection:
    """The write journal, read-only at the SQLite level; a missing file is refused, never created."""
    resolved = Path(path)
    if not resolved.exists():
        raise FileNotFoundError(f'write journal not found: {resolved}')
    return sqlite3.connect(f'file:{resolved}?mode=ro', uri=True)


def _row_key(row_id: str) -> str:
    return hashlib.sha256(row_id.encode()).hexdigest()


def _journal_rows(path: str | Path, operation: str, window: Window) -> list[tuple[str, str, dict[str, Any]]]:
    """``(id, created_at, params)`` of *operation*'s rows created in *window*, in sha256(id) order."""
    with contextlib.closing(open_journal(path)) as db:
        rows = db.execute(JOURNAL_ROWS_SQL, (operation,)).fetchall()
    kept = []
    for row_id, created_at, raw in rows:
        try:
            params = json.loads(raw or '{}')
        except json.JSONDecodeError:
            continue
        if isinstance(params, dict) and instant_in_window(created_at, *window):
            kept.append((str(row_id), str(created_at), params))
    return sorted(kept, key=lambda row: _row_key(row[0]))


class LlmReachProbe:
    """A real AsyncOpenAI whose transport answers locally and counts requests, so the production
    routing code is asked, through its public openai_client= seam, whether an input reaches its LLM."""

    def __init__(self) -> None:
        self.requests = 0
        self.config = FusedMemoryConfig.model_construct(routing=RoutingConfig(), llm=LLMConfig())
        """Routing defaults: heuristics first, the LLM below the confidence threshold."""

    def client(self) -> AsyncOpenAI:
        return AsyncOpenAI(
            api_key='probe', base_url='http://llm-reach-probe.invalid/v1', max_retries=0,
            http_client=httpx.AsyncClient(transport=httpx.MockTransport(self._answer)),
        )

    def _answer(self, request: httpx.Request) -> httpx.Response:
        self.requests += 1
        return httpx.Response(200, json={
            'id': 'probe', 'object': 'chat.completion', 'created': 0, 'model': 'probe',
            'choices': [{'index': 0, 'finish_reason': 'stop',
                         'message': {'role': 'assistant', 'content': json.dumps(_PROBE_ANSWER)}}],
        })

    async def reaches(self, call: Awaitable[Any]) -> bool:
        before = self.requests
        await call
        return self.requests > before


async def classifier_fixtures(journal_path: str | Path, window: Window, per_category: int) -> list[dict[str, Any]]:
    """The first *per_category* window add_memory inputs per journalled category, by sha256(row id),
    that the production WriteClassifier sends to its LLM. The input is the journal's 200-char prefix."""
    probe = LlmReachProbe()
    classifier = WriteClassifier(probe.config, openai_client=probe.client())
    by_category: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    try:
        for row_id, created_at, params in _journal_rows(journal_path, 'add_memory', window):
            content, category = params.get('content'), params.get('category')
            if not isinstance(content, str) or not isinstance(category, str):
                continue
            if len(by_category[category]) < per_category and await probe.reaches(classifier.classify(content)):
                by_category[category].append(
                    {'id': row_id, 'created_at': created_at, 'input': content, 'reference_category': category},
                )
    finally:
        await classifier.close()
    return [fixture for category in sorted(by_category) for fixture in by_category[category]]


async def router_fixtures(journal_path: str | Path, window: Window, n: int) -> list[dict[str, Any]]:
    """The first *n* distinct, untruncated window search queries, by sha256(row id), that the
    production ReadRouter sends to its LLM."""
    probe = LlmReachProbe()
    router = ReadRouter(probe.config, openai_client=probe.client())
    seen: set[str] = set()
    fixtures: list[dict[str, Any]] = []
    try:
        for row_id, created_at, params in _journal_rows(journal_path, 'search', window):
            query = params.get('query')
            if len(fixtures) == n:
                break
            if params.get('query_truncated') is not False or not isinstance(query, str) or query in seen:
                continue
            seen.add(query)
            if await probe.reaches(router.route(query)):
                fixtures.append({'id': row_id, 'created_at': created_at, 'query': query})
    finally:
        await router.close()
    return fixtures
