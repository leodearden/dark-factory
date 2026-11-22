#!/usr/bin/env python3
"""Run one fixture of one call site under one arm through graphiti_core's or production's own function, for task 9945.

Each runner first calls replay_overreach_extraction.check_arm_client, so a
client on another model, reasoning effort or endpoint than its arm is refused
before any call, and measures the call's tokens with measure_llm_tokens. A
record is plain JSON keyed (site, arm, fixture_id), as the replay's records are
keyed (arm, graph, uuid), so a run resumes the same way.
"""
from __future__ import annotations

import dataclasses
import os
import sys
from collections.abc import Iterable, Mapping, Sequence
from datetime import datetime
from enum import StrEnum
from types import MappingProxyType
from typing import Any

from graphiti_core.edges import EntityEdge
from graphiti_core.embedder.client import EmbedderClient
from graphiti_core.graphiti_types import GraphitiClients
from graphiti_core.llm_client.client import LLMClient
from graphiti_core.nodes import EntityNode, EpisodeType, EpisodicNode
from graphiti_core.utils.maintenance.edge_operations import resolve_extracted_edge
from graphiti_core.utils.maintenance.node_operations import extract_attributes_from_nodes

from fused_memory.backends.llm_token_usage import LlmTokenUsage, measure_llm_tokens

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from audit_ruling_overreach import stored_instant
from replay_overreach_extraction import (
    ARMS,
    CONTROL,
    RecordKind,
    ReplayArm,
    check_arm_client,
    client_endpoint,
    client_reasoning_effort,
    measured_usage,
)

LUNA_ARM = 'luna-low'

SITE_ARMS: Mapping[str, ReplayArm] = MappingProxyType({name: ARMS[name] for name in (CONTROL, LUNA_ARM)})
"""The replay's own arms: replay_overreach_extraction.ARMS is the one definition of each arm's
model, effort and endpoint."""


class Site(StrEnum):
    resolve_edge = 'resolve_edge'
    summaries = 'summaries'


EDGE_NAME = 'RELATES_TO'
"""The relation name a fixture edge carries. resolve_extracted_edge reads it only to pick a custom
edge type, and fused-memory passes none."""

NULL_EMBEDDING_DIM = 1

SiteRecordKey = tuple[str, str, str]
"""``(site, arm, fixture_id)``."""


class NullEmbedder(EmbedderClient):
    """A graphiti embedder that returns zero vectors and calls nothing; no eval measure reads an embedding."""

    async def create(self, input_data: str | list[str] | Iterable[int] | Iterable[Iterable[int]]) -> list[float]:
        return [0.0] * NULL_EMBEDDING_DIM

    async def create_batch(self, input_data_list: list[str]) -> list[list[float]]:
        return [[0.0] * NULL_EMBEDDING_DIM for _ in input_data_list]


def record_key(record: Mapping[str, Any]) -> SiteRecordKey:
    return (record['site'], record['arm'], record['fixture_id'])


def _at(value: object) -> datetime | None:
    return stored_instant(value)


def _required_at(value: object, what: str) -> datetime:
    if (at := _at(value)) is None:
        raise ValueError(f'{what} has no parseable timestamp: {value!r}')
    return at


def _entity_edge(graph: str, edge: Mapping[str, Any], *, uuid_key: str = 'uuid') -> EntityEdge:
    return EntityEdge(
        uuid=edge[uuid_key], group_id=graph, source_node_uuid=edge['source_node_uuid'],
        target_node_uuid=edge['target_node_uuid'], name=EDGE_NAME, fact=edge['fact'], episodes=[],
        created_at=_required_at(edge['created_at'], f"edge {edge[uuid_key]}"),
        valid_at=_at(edge.get('valid_at')), invalid_at=_at(edge.get('invalid_at')),
        expired_at=_at(edge.get('expired_at')),
    )


def _episode(graph: str, uuid: str, valid_at: datetime, content: str) -> EpisodicNode:
    return EpisodicNode(
        uuid=uuid, name=uuid, group_id=graph, labels=[], source=EpisodeType.text, content=content,
        source_description='', created_at=valid_at, valid_at=valid_at,
    )


def _record(
    site: Site, arm: ReplayArm, llm: LLMClient, fixture_id: str, usage: LlmTokenUsage, **measured: Any,
) -> dict[str, Any]:
    return {
        'kind': RecordKind.replayed, 'site': site, 'arm': arm.name, 'model': llm.model,
        'reasoning_effort': client_reasoning_effort(llm), 'endpoint': client_endpoint(llm),
        'fixture_id': fixture_id, **measured, 'tokens': {site.value: dataclasses.asdict(usage)},
    }


async def run_resolve_edge(llm: LLMClient, fixture: Mapping[str, Any], arm: ReplayArm) -> dict[str, Any]:
    """graphiti's resolve_extracted_edge over one luna_eval_frame.resolve_edge_fixtures fixture."""
    await check_arm_client(llm, arm)
    graph = fixture['graph']
    new_edge = _entity_edge(graph, fixture, uuid_key='edge_uuid')
    episode = _episode(
        graph, fixture['episode_uuid'], _at(fixture.get('episode_valid_at')) or new_edge.created_at, '',
    )
    related = [_entity_edge(graph, edge) for edge in fixture['related']]
    existing = [_entity_edge(graph, edge) for edge in fixture['invalidation_candidates']]
    async with measure_llm_tokens(llm) as tokens:
        resolved, invalidated, duplicates = await resolve_extracted_edge(llm, new_edge, related, existing, episode)
    return _record(
        Site.resolve_edge, arm, llm, f"{graph}/{fixture['edge_uuid']}", measured_usage(tokens),
        duplicate_uuids=[edge.uuid for edge in duplicates],
        merged_into=resolved.uuid if resolved.uuid != new_edge.uuid else None,
        invalidated_uuids=[edge.uuid for edge in invalidated],
    )


async def run_summaries(
    llm: LLMClient, fixture: Mapping[str, Any], previous: Sequence[str], arm: ReplayArm,
) -> dict[str, Any]:
    """graphiti's extract_attributes_from_nodes over one episode's summary items.

    *fixture* is an episode's luna_eval_frame.summary_fixtures items with their text restored:
    ``episode_content``, ``nodes`` (node_uuid, node_name, summary) and ``edges`` (uuid,
    source_node_uuid, target_node_uuid, fact). *previous* is the previous episodes' contents,
    oldest first, as replay_overreach_extraction.EpisodeContextReader reads them.
    """
    await check_arm_client(llm, arm)
    graph = fixture['graph']
    valid_at = _required_at(fixture['episode_valid_at'], f"episode {fixture['episode_uuid']}")
    episode = _episode(graph, fixture['episode_uuid'], valid_at, fixture['episode_content'])
    previous_nodes = [
        _episode(graph, f"{fixture['episode_uuid']}:previous:{index}", valid_at, content)
        for index, content in enumerate(previous)
    ]
    nodes = [
        EntityNode(uuid=node['node_uuid'], name=node['node_name'], group_id=graph, labels=['Entity'],
                   summary=node['summary'], created_at=valid_at)
        for node in fixture['nodes']
    ]
    edges = [
        _entity_edge(graph, {**edge, 'created_at': fixture['episode_valid_at']}) for edge in fixture['edges']
    ]
    clients = GraphitiClients.model_construct(llm_client=llm, embedder=NullEmbedder())
    async with measure_llm_tokens(llm) as tokens:
        await extract_attributes_from_nodes(clients, nodes, episode, previous_nodes, entity_types=None, edges=edges)
    return _record(
        Site.summaries, arm, llm, f"{graph}/{fixture['episode_uuid']}", measured_usage(tokens),
        summaries={node.uuid: node.summary for node in nodes},
    )
