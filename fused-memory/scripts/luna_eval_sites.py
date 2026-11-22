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
import logging
import os
import sys
from collections.abc import Awaitable, Callable, Iterable, Mapping, Sequence
from datetime import datetime
from enum import StrEnum
from pathlib import Path
from types import MappingProxyType
from typing import Any

import httpx
from graphiti_core.edges import EntityEdge
from graphiti_core.embedder.client import EmbedderClient
from graphiti_core.graphiti_types import GraphitiClients
from graphiti_core.llm_client.client import LLMClient
from graphiti_core.nodes import EntityNode, EpisodeType, EpisodicNode
from graphiti_core.utils.maintenance.edge_operations import resolve_extracted_edge
from graphiti_core.utils.maintenance.node_operations import extract_attributes_from_nodes
from openai import AsyncOpenAI

from fused_memory.backends.llm_token_usage import LlmTokenUsage, measure_llm_tokens
from fused_memory.config.schema import FusedMemoryConfig, RoutingConfig
from fused_memory.routing.classifier import WriteClassifier
from fused_memory.routing.openai_client import routing_chat_params, routing_endpoint
from fused_memory.routing.router import ReadRouter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _graph_script_cli import write_files_atomically
from audit_ruling_overreach import stored_instant
from replay_overreach_extraction import (
    ARMS,
    CONTROL,
    ArmEndpoint,
    RecordKind,
    ReplayArm,
    check_arm_client,
    client_endpoint,
    client_reasoning_effort,
    failure_fields,
    jsonl_text,
    measured_usage,
    plan_proxy_endpoint,
    read_jsonl,
)

logger = logging.getLogger('luna_eval_sites')

LUNA_ARM = 'luna-low'

SITE_ARMS: Mapping[str, ReplayArm] = MappingProxyType({name: ARMS[name] for name in (CONTROL, LUNA_ARM)})
"""The replay's own arms: replay_overreach_extraction.ARMS is the one definition of each arm's
model, effort and endpoint."""


class Site(StrEnum):
    resolve_edge = 'resolve_edge'
    summaries = 'summaries'
    classifier = 'classifier'
    router = 'router'


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


def fixture_id(site: Site, fixture: Mapping[str, Any]) -> str:
    """A fixture's id within its site: graph/edge for resolve_edge, graph/episode for summaries,
    the journal row id for the classifier and the router."""
    if site is Site.resolve_edge:
        return f"{fixture['graph']}/{fixture['edge_uuid']}"
    if site is Site.summaries:
        return f"{fixture['graph']}/{fixture['episode_uuid']}"
    return str(fixture['id'])


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
    site: Site, arm: ReplayArm, client: Mapping[str, Any], fixture: Mapping[str, Any], usage: LlmTokenUsage,
    **measured: Any,
) -> dict[str, Any]:
    """A run line; *client* is the model, reasoning_effort and endpoint the call was sent on."""
    return {
        'kind': RecordKind.replayed, 'site': site, 'arm': arm.name, **client,
        'fixture_id': fixture_id(site, fixture), **measured, 'tokens': {site.value: dataclasses.asdict(usage)},
    }


def _graphiti_client(llm: LLMClient) -> dict[str, Any]:
    return {'model': llm.model, 'reasoning_effort': client_reasoning_effort(llm), 'endpoint': client_endpoint(llm)}


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
        Site.resolve_edge, arm, _graphiti_client(llm), fixture, measured_usage(tokens),
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
        Site.summaries, arm, _graphiti_client(llm), fixture, measured_usage(tokens),
        summaries={node.uuid: node.summary for node in nodes},
    )


# --------------------------------------------------------------------------- #
# The routing sites: the production WriteClassifier and ReadRouter per arm
# --------------------------------------------------------------------------- #


def usage_hook(sink: list[LlmTokenUsage]) -> Callable[[httpx.Response], Awaitable[None]]:
    """An httpx response hook appending each chat completion's usage to *sink*."""

    async def record(response: httpx.Response) -> None:
        await response.aread()
        try:
            usage = response.json().get('usage')
        except ValueError:
            return
        if isinstance(usage, dict):
            sink.append(LlmTokenUsage(
                input_tokens=int(usage.get('prompt_tokens') or 0),
                output_tokens=int(usage.get('completion_tokens') or 0), llm_calls=1,
            ))

    return record


class UsageRecordingOpenAI(AsyncOpenAI):
    """An AsyncOpenAI that records every response's usage and declares the endpoint it was built on."""

    def __init__(
        self, *, api_key: str | None, base_url: str | None, endpoint: ArmEndpoint,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self.usages: list[LlmTokenUsage] = []
        self._endpoint = endpoint
        super().__init__(
            api_key=api_key, base_url=base_url,
            http_client=httpx.AsyncClient(transport=transport, event_hooks={'response': [usage_hook(self.usages)]}),
        )

    @property
    def endpoint(self) -> ArmEndpoint:
        return self._endpoint

    def usage_since(self, mark: int) -> LlmTokenUsage:
        """The usage recorded after ``len(self.usages)`` was *mark*."""
        recorded = self.usages[mark:]
        return LlmTokenUsage(
            input_tokens=sum(u.input_tokens for u in recorded),
            output_tokens=sum(u.output_tokens for u in recorded), llm_calls=sum(u.llm_calls for u in recorded),
        )


def routing_arm(
    cfg: FusedMemoryConfig, arm: ReplayArm, environ: Mapping[str, str],
    *, transport: httpx.AsyncBaseTransport | None = None,
) -> tuple[FusedMemoryConfig, UsageRecordingOpenAI]:
    """*arm* as routing config, and a client on the endpoint production's routing_endpoint resolves for it.

    The arm's model, effort and endpoint become exactly the routing leaves a flip
    ships: routing.model, routing.reasoning_effort, and for a plan_proxy arm
    routing.openai_api_url/openai_api_key from plan_proxy_endpoint(*environ*).
    *cfg* is never modified.
    """
    leaves: dict[str, Any] = {}
    if arm.llm.endpoint is ArmEndpoint.plan_proxy:
        leaves['openai_api_url'], leaves['openai_api_key'] = plan_proxy_endpoint(environ)
    if arm.llm.model is not None:
        leaves['model'] = arm.llm.model
    if arm.llm.reasoning_effort is not None:
        leaves['reasoning_effort'] = arm.llm.reasoning_effort
    arm_cfg = cfg.model_copy(deep=True)
    arm_cfg.routing = RoutingConfig.model_validate(cfg.routing.model_dump() | leaves)
    api_key, base_url = routing_endpoint(arm_cfg)
    return arm_cfg, UsageRecordingOpenAI(
        api_key=api_key, base_url=base_url, endpoint=arm.llm.endpoint, transport=transport,
    )


def _routing_client(arm_cfg: FusedMemoryConfig, client: UsageRecordingOpenAI) -> dict[str, Any]:
    return {
        'model': routing_chat_params(arm_cfg)['model'], 'reasoning_effort': arm_cfg.routing.reasoning_effort,
        'endpoint': client.endpoint,
    }


async def run_classifier(
    arm_cfg: FusedMemoryConfig, client: UsageRecordingOpenAI, fixture: Mapping[str, Any], arm: ReplayArm,
) -> dict[str, Any]:
    """The production WriteClassifier.classify over one journal input. *client* stays the caller's
    to close: the classifier holding it is never closed."""
    mark = len(client.usages)
    result = await WriteClassifier(arm_cfg, openai_client=client).classify(fixture['input'])
    return _record(
        Site.classifier, arm, _routing_client(arm_cfg, client), fixture, client.usage_since(mark),
        label=result.primary.value, secondary=result.secondary.value if result.secondary else None,
        fallback=result.fallback.value if result.fallback else None, stores=None, primary_store=None,
    )


async def run_router(
    arm_cfg: FusedMemoryConfig, client: UsageRecordingOpenAI, fixture: Mapping[str, Any], arm: ReplayArm,
) -> dict[str, Any]:
    """The production ReadRouter.route over one journal query. *client* stays the caller's to close."""
    mark = len(client.usages)
    result = await ReadRouter(arm_cfg, openai_client=client).route(fixture['query'])
    return _record(
        Site.router, arm, _routing_client(arm_cfg, client), fixture, client.usage_since(mark),
        label=result.query_type.value, secondary=None, fallback=None,
        stores=[store.value for store in result.stores], primary_store=result.primary_store.value,
    )


# --------------------------------------------------------------------------- #
# The resumable run over one site's fixtures
# --------------------------------------------------------------------------- #

SiteRunner = Callable[[ReplayArm, Mapping[str, Any]], Awaitable[dict[str, Any]]]
"""One arm over one fixture, returning its run line."""


async def run_site_fixtures(
    site: Site, fixtures: Sequence[Mapping[str, Any]], arms: Sequence[ReplayArm], runner: SiteRunner, out: Path,
) -> int:
    """Run every (arm, fixture) *out* does not already hold, appending one line each; returns the failures.

    *out* is resumed as replay_overreach_extraction's replay file is: its run
    lines are kept and *site*'s failure lines dropped, so a rerun retries exactly
    them. Every site shares one *out*, so another site's failure lines are kept.
    """
    kept = [
        r for r in read_jsonl(out) if r.get('kind') == RecordKind.replayed or r.get('site') != site
    ] if out.exists() else []
    write_files_atomically({out: jsonl_text(kept)})
    done = {record_key(record) for record in kept}
    failures = 0
    with out.open('a') as sink:
        for fixture in fixtures:
            identity = fixture_id(site, fixture)
            records = []
            for arm in arms:
                if (site, arm.name, identity) in done:
                    continue
                try:
                    records.append(await runner(arm, fixture))
                except Exception as exc:
                    logger.warning('%s %s under %s failed', site, identity, arm.name, exc_info=True)
                    records.append({
                        'kind': RecordKind.failed, 'site': site, 'arm': arm.name, 'fixture_id': identity,
                        **failure_fields(exc),
                    })
                    failures += 1
            sink.write(jsonl_text(records))
            sink.flush()
    return failures
