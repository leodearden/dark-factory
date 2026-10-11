"""A scratch graph's topology, the part a re-embed must leave untouched, read and hashed.

The queries project identity and content only: node uuid, labels and name, and edge
uuid, type, endpoints, name and fact. They use explicit map projections, so an
embedding vector can never enter a record or the hash. Every read goes through
``ro_query`` (PRD §Hazards).
"""

import hashlib
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any, NamedTuple, Protocol, TypeVar

from shared.memory_eval_metrics import canonical_json_text

from fused_memory.arm_harness.scratch_guard import GuardCheckpoint, require_scratch_name

NODE_CYPHER = 'MATCH (n) RETURN {uuid: n.uuid, labels: labels(n), name: n.name}'
EDGE_CYPHER = (
    'MATCH (s)-[r]->(t) RETURN {uuid: r.uuid, rel_type: type(r), source_uuid: s.uuid, '
    'target_uuid: t.uuid, name: r.name, fact: r.fact}'
)


class ReadOnlyGraph(Protocol):
    """The slice of falkordb's AsyncGraph a topology read uses."""

    async def ro_query(self, cypher: str, /) -> Any: ...


@dataclass(frozen=True)
class TopologyNode:
    uuid: str
    labels: tuple[str, ...]
    name: str | None


@dataclass(frozen=True)
class TopologyEdge:
    uuid: str
    rel_type: str
    source_uuid: str
    target_uuid: str
    name: str | None
    fact: str | None


class Topology(NamedTuple):
    nodes: tuple[TopologyNode, ...]
    edges: tuple[TopologyEdge, ...]


@dataclass(frozen=True)
class IntegrityVerdict:
    identical: bool
    node_count: int
    edge_count: int
    missing_node_uuids: tuple[str, ...]
    extra_node_uuids: tuple[str, ...]
    changed_node_uuids: tuple[str, ...]
    changed_edge_uuids: tuple[str, ...]


async def read_topology(graph: ReadOnlyGraph, graph_name: str) -> Topology:
    require_scratch_name(graph_name, checkpoint=GuardCheckpoint.TOPOLOGY_READ)
    node_rows = await _projected_rows(graph, NODE_CYPHER)
    edge_rows = await _projected_rows(graph, EDGE_CYPHER)
    return Topology(
        nodes=tuple(_node(row) for row in node_rows),
        edges=tuple(_edge(row) for row in edge_rows),
    )


async def _projected_rows(graph: ReadOnlyGraph, cypher: str) -> list[Mapping[str, Any]]:
    response = await graph.ro_query(cypher)
    return [row[0] for row in response.result_set]


def _node(row: Mapping[str, Any]) -> TopologyNode:
    return TopologyNode(uuid=row['uuid'], labels=tuple(sorted(row['labels'])), name=row['name'])


def _edge(row: Mapping[str, Any]) -> TopologyEdge:
    return TopologyEdge(
        uuid=row['uuid'],
        rel_type=row['rel_type'],
        source_uuid=row['source_uuid'],
        target_uuid=row['target_uuid'],
        name=row['name'],
        fact=row['fact'],
    )


def topology_hash(nodes: Sequence[TopologyNode], edges: Sequence[TopologyEdge]) -> str:
    payload = {'nodes': _sorted_records(nodes), 'edges': _sorted_records(edges)}
    return hashlib.sha256(canonical_json_text(payload).encode()).hexdigest()


def _sorted_records(records: Sequence[TopologyNode] | Sequence[TopologyEdge]) -> list[object]:
    return sorted((asdict(record) for record in records), key=canonical_json_text)


_Record = TypeVar('_Record', TopologyNode, TopologyEdge)


def _by_uuid(records: Sequence[_Record]) -> dict[str, _Record]:
    counts = Counter(record.uuid for record in records)
    repeated = sorted(uuid for uuid, count in counts.items() if count > 1)
    if repeated:
        raise ValueError(f'a topology repeats uuids {repeated}')
    return {record.uuid: record for record in records}


def _changed(reference: Mapping[str, object], candidate: Mapping[str, object]) -> set[str]:
    shared = reference.keys() & candidate.keys()
    return {uuid for uuid in shared if reference[uuid] != candidate[uuid]}


def check_reembed_integrity(reference: Topology, candidate: Topology) -> IntegrityVerdict:
    """Whether a re-embedded copy kept the reference topology, naming each differing uuid."""
    ref_nodes, cand_nodes = _by_uuid(reference.nodes), _by_uuid(candidate.nodes)
    ref_edges, cand_edges = _by_uuid(reference.edges), _by_uuid(candidate.edges)
    added_or_removed_edges = ref_edges.keys() ^ cand_edges.keys()
    return IntegrityVerdict(
        identical=topology_hash(*reference) == topology_hash(*candidate),
        node_count=len(reference.nodes),
        edge_count=len(reference.edges),
        missing_node_uuids=tuple(sorted(ref_nodes.keys() - cand_nodes.keys())),
        extra_node_uuids=tuple(sorted(cand_nodes.keys() - ref_nodes.keys())),
        changed_node_uuids=tuple(sorted(_changed(ref_nodes, cand_nodes))),
        changed_edge_uuids=tuple(
            sorted(added_or_removed_edges | _changed(ref_edges, cand_edges))
        ),
    )
