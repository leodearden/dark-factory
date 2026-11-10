#!/usr/bin/env python3
"""Replay ruling-class extraction under an arm, to measure lever O1 (task 9040).

docs/ruling-overreach-guard-design-2026-10-01/design.md §6 names lever O1:
``custom_extraction_instructions`` for the ``decisions_and_rationale`` class.
§10.3 re-opened the lever choice with O1 to be measured by replaying the 48
frozen-window episodes. This script is that replay. Each episode is
re-extracted under each arm by graphiti_core 0.28.2's own
utils/maintenance/node_operations.py::extract_nodes and
utils/maintenance/edge_operations.py::extract_edges, unmodified, followed by
graphiti's in-batch exact-fact dedupe (the driver-free fast path of
edge_operations.py::resolve_extracted_edges). The arms differ only in the
instructions they splice into those prompts; the model is a separate axis.

Read-only and driver-free: graphiti is handed a GraphitiClients that holds
only an LLM client, so no FalkorDriver is ever built (see
fused_memory/backends/falkor_read_only.py for why that matters). Episode
inputs are read over GRAPH.RO_QUERY only. Nothing is written to any graph.
"""
from __future__ import annotations

import hashlib
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import datetime
from types import MappingProxyType

from graphiti_core.edges import EntityEdge
from graphiti_core.graphiti_types import GraphitiClients
from graphiti_core.llm_client.client import LLMClient
from graphiti_core.nodes import EpisodeType, EpisodicNode
from graphiti_core.utils.maintenance.dedup_helpers import _normalize_string_exact
from graphiti_core.utils.maintenance.edge_operations import extract_edges
from graphiti_core.utils.maintenance.node_operations import extract_nodes

from fused_memory.backends.graphiti_client import build_llm_client
from fused_memory.backends.llm_token_usage import (
    LlmTokenUsage,
    TokenMeasurement,
    measure_llm_tokens,
)
from fused_memory.config.schema import FusedMemoryConfig

# --------------------------------------------------------------------------- #
# The arms, pre-registered (design §2.1)
# --------------------------------------------------------------------------- #

O1_INSTRUCTIONS = """\
This is a DECISION RECORD. Extract only what it states, at the scope it states it.
1. Keep the record's scope and qualifiers. Never generalise a decision past the cases, components, actors or conditions the record names, and never drop a qualifier (who, which, when, under what condition) that limits it.
2. Add no implication or consequence the record does not state. Do not link two things with a vague verb such as "impacts", "affects", "relates to" or "implies" unless the record itself links them.
3. An option the record weighed, rejected, deferred or only proposed is never adopted. State it only together with its status (for example "X was rejected in favour of Y"), never as current fact.
4. Preserve direction exactly: who or what acts on what, what calls, imports or depends on what, what is above or below, before or after.
5. Place each symbol, file, function, value or measurement with the entity the record places it with. Never move it to a neighbouring module, component or task.
6. PREVIOUS_MESSAGES only help resolve references. Never attribute their content to this record.
"""

O1_INSTRUCTIONS_VERSION = 'o1-v1-' + hashlib.sha256(O1_INSTRUCTIONS.encode()).hexdigest()[:12]

NO_INSTRUCTIONS = 'none'


@dataclass(frozen=True)
class ReplayArm:
    """What one arm splices into graphiti's extraction prompts; ``model`` None means the configured one."""

    name: str
    instructions: str | None
    instructions_version: str
    model: str | None = None


ARMS: Mapping[str, ReplayArm] = MappingProxyType({
    'control': ReplayArm('control', None, NO_INSTRUCTIONS),
    'o1': ReplayArm('o1', O1_INSTRUCTIONS, O1_INSTRUCTIONS_VERSION),
})

EDGE_TYPE_MAP: Mapping[tuple[str, str], list[str]] = MappingProxyType({('Entity', 'Entity'): []})
"""graphiti_core's graphiti.py::Graphiti.add_episode default when no edge_types are passed, as fused-memory passes none."""

# --------------------------------------------------------------------------- #
# One episode, replayed
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class ReplayInput:
    """An episode as production extracted it: its own fields and its previous-episode contents, oldest first."""

    graph: str
    uuid: str
    stratum: str
    content: str
    source: str
    source_description: str
    valid_at: datetime
    previous: tuple[str, ...]


@dataclass(frozen=True)
class MintedEdge:
    edge_uuid: str
    fact: str
    source_name: str
    target_name: str


@dataclass(frozen=True)
class ReplayResult:
    arm: str
    model: str | None
    instructions_version: str
    graph: str
    uuid: str
    stratum: str
    entities: tuple[str, ...]
    minted: tuple[MintedEdge, ...]
    tokens: Mapping[str, LlmTokenUsage]
    """``extract_nodes`` and ``extract_edges``: each stage's measured usage, retries included."""


class UnmeasurableClient(TypeError):
    """The LLM client carries no AttributingTokenUsageTracker, so its tokens cannot be measured."""


def build_arm_llm(cfg: FusedMemoryConfig, model: str | None) -> LLMClient | None:
    """The configured extraction client, on *model* when given; *cfg* itself is never modified."""
    arm_cfg = cfg.model_copy(deep=True)
    if model is not None:
        arm_cfg.llm.model = model
    return build_llm_client(arm_cfg)


def in_batch_dedupe(edges: Iterable[EntityEdge]) -> list[EntityEdge]:
    """The first edge per (source, target, exactly-normalised fact), as resolve_extracted_edges' fast path keeps it."""
    kept: dict[tuple[str, str, str], EntityEdge] = {}
    for edge in edges:
        kept.setdefault(
            (edge.source_node_uuid, edge.target_node_uuid, _normalize_string_exact(edge.fact)), edge,
        )
    return list(kept.values())


def _episodic_node(episode: ReplayInput, *, uuid: str, content: str) -> EpisodicNode:
    return EpisodicNode(
        uuid=uuid, name=uuid, group_id=episode.graph, labels=[],
        source=EpisodeType(episode.source), content=content,
        source_description=episode.source_description,
        created_at=episode.valid_at, valid_at=episode.valid_at,
    )


def _usage(measurement: TokenMeasurement) -> LlmTokenUsage:
    if measurement.usage is None:
        raise UnmeasurableClient('the replay needs a client built by build_arm_llm, whose tokens are attributable')
    return measurement.usage


async def replay_episode(llm: LLMClient, episode: ReplayInput, arm: ReplayArm) -> ReplayResult:
    """Extract *episode* under *arm* through graphiti's own functions; no driver exists to reach."""
    clients = GraphitiClients.model_construct(llm_client=llm)
    current = _episodic_node(episode, uuid=episode.uuid, content=episode.content)
    previous = [
        _episodic_node(episode, uuid=f'{episode.uuid}:previous:{index}', content=content)
        for index, content in enumerate(episode.previous)
    ]
    async with measure_llm_tokens(llm) as node_tokens:
        nodes = await extract_nodes(
            clients, current, previous, entity_types=None,
            custom_extraction_instructions=arm.instructions,
        )
    async with measure_llm_tokens(llm) as edge_tokens:
        edges = await extract_edges(
            clients, current, nodes, previous, dict(EDGE_TYPE_MAP), group_id=episode.graph,
            custom_extraction_instructions=arm.instructions,
        )
    names = {node.uuid: node.name for node in nodes}
    return ReplayResult(
        arm=arm.name, model=llm.model, instructions_version=arm.instructions_version,
        graph=episode.graph, uuid=episode.uuid, stratum=episode.stratum,
        entities=tuple(node.name for node in nodes),
        minted=tuple(
            MintedEdge(edge.uuid, edge.fact, names[edge.source_node_uuid], names[edge.target_node_uuid])
            for edge in in_batch_dedupe(edges)
        ),
        tokens=MappingProxyType({'extract_nodes': _usage(node_tokens), 'extract_edges': _usage(edge_tokens)}),
    )
