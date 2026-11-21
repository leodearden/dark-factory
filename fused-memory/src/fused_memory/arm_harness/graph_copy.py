"""An arm's re-embedded copy of the frozen reference graph: copied, moved onto its own group, re-embedded.

The texts re-embedded are exactly the ones graphiti embeds, Entity.name and
RELATES_TO.fact. Every incumbent vector is nulled before any arm vector is
written, so "no stale incumbent vector survives" is checkable by census for
every arm, the 1536-dimension incumbent included.
"""

import asyncio
import logging
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol

from redis.exceptions import ResponseError

from fused_memory.arm_harness.arm_embedder import DocumentEmbedder, DocumentEmbeddings
from fused_memory.arm_harness.fork_slot import ForkHolder, ServerInfo, read_fork_holders
from fused_memory.arm_harness.frozen_model import FrozenModel
from fused_memory.arm_harness.normalization import NormStats, norm_stats
from fused_memory.arm_harness.scratch_guard import GuardCheckpoint, require_scratch_name
from fused_memory.arm_harness.sleep_seam import Sleep

logger = logging.getLogger(__name__)

ADOPT_NODE_GROUP_CYPHER = 'MATCH (n) SET n.group_id = $group_id RETURN count(n)'
ADOPT_EDGE_GROUP_CYPHER = 'MATCH ()-[r]->() SET r.group_id = $group_id RETURN count(r)'
READ_ENTITY_NAMES_CYPHER = 'MATCH (n:Entity) RETURN n.uuid, n.name'
READ_EDGE_FACTS_CYPHER = 'MATCH ()-[r:RELATES_TO]->() RETURN r.uuid, r.fact'
NULL_NAME_EMBEDDINGS_CYPHER = 'MATCH (n:Entity) SET n.name_embedding = NULL RETURN count(n)'
NULL_FACT_EMBEDDINGS_CYPHER = (
    'MATCH ()-[r:RELATES_TO]->() SET r.fact_embedding = NULL RETURN count(r)'
)
WRITE_NAME_EMBEDDINGS_CYPHER = (
    'UNWIND $rows AS row MATCH (n:Entity {uuid: row.uuid}) '
    'SET n.name_embedding = vecf32(row.v) RETURN count(n)'
)
WRITE_FACT_EMBEDDINGS_CYPHER = (
    'UNWIND $rows AS row MATCH ()-[r:RELATES_TO {uuid: row.uuid}]->() '
    'SET r.fact_embedding = vecf32(row.v) RETURN count(r)'
)
COUNT_NAME_EMBEDDINGS_CYPHER = (
    'MATCH (n:Entity) WHERE n.name_embedding IS NOT NULL RETURN count(n)'
)
COUNT_FACT_EMBEDDINGS_CYPHER = (
    'MATCH ()-[r:RELATES_TO]->() WHERE r.fact_embedding IS NOT NULL RETURN count(r)'
)

DEFAULT_WRITE_BATCH_SIZE = 64

COPY_FORK_REFUSAL = 'GRAPH.COPY failed, could not fork'
"""How FalkorDB's reply begins while another child process (e.g. a BGSAVE) holds the fork; graph module 4.18.0."""

COPY_ATTEMPTS = 60
FORK_POLL_INTERVAL_S = 2.0
"""After a refused copy: one interval, then INFO persistence every interval while a child holds the fork.

So COPY_ATTEMPTS bounds refusals, not seconds: however long a BGSAVE runs, it costs
one refused copy, and the copy is reissued once the child has exited.
"""
FORK_WAIT_REPORT_EVERY_POLLS = 30
"""A wait on a held fork is logged when its holders change, and again every this many polls."""


class ScratchGraph(Protocol):
    """The slice of falkordb's AsyncGraph a re-embed uses."""

    async def query(self, cypher: str, params: dict[str, object] | None = None, /) -> Any: ...

    async def ro_query(self, cypher: str, params: dict[str, object] | None = None, /) -> Any: ...


class CopyableGraph(Protocol):
    async def copy(self, clone: str, /) -> object: ...


class GraphCopyClient(Protocol):
    """The slice of falkordb's async FalkorDB client a reference copy uses."""

    @property
    def connection(self) -> ServerInfo: ...

    async def list_graphs(self) -> list[str]: ...

    def select_graph(self, graph_id: str, /) -> CopyableGraph: ...


class GraphCopyRefused(RuntimeError):
    """FalkorDB refused every GRAPH.COPY attempt because it could not fork; no copy exists."""

    def __init__(
        self,
        reference: str,
        target: str,
        attempts: int,
        waited_s: float,
        holders: frozenset[ForkHolder],
        reply: str,
    ) -> None:
        self.reference = reference
        self.target = target
        self.attempts = attempts
        self.waited_s = waited_s
        self.holders = holders
        self.reply = reply
        super().__init__(
            f'GRAPH.COPY {reference!r} -> {target!r} was refused {attempts} times over '
            f'{waited_s:.0f}s: {reply}; {_fork_account(holders)}, so no copy was made'
        )


def _fork_account(holders: frozenset[ForkHolder]) -> str:
    if holders:
        return f'it waited out {", ".join(sorted(holders))} and the server still could not fork'
    return 'no child held the fork, so the server itself could not fork'


class ReembedCensusError(RuntimeError):
    """The vectors read back from a re-embedded graph are not the vectors written."""


@dataclass(frozen=True)
class GroupAdoption:
    node_count: int
    edge_count: int


class GraphReembed(FrozenModel):
    entity_count: int
    edge_count: int
    written: int
    failures: tuple[tuple[str, str], ...]
    raw_norms: NormStats | None
    embed_seconds: float
    write_seconds: float


@dataclass(frozen=True)
class _VectorProperty:
    """One embedded text property graphiti writes, and the statements that touch it."""

    property_name: str
    read: str
    null: str
    write: str
    count: str


_NAME_EMBEDDING = _VectorProperty(
    'name_embedding',
    READ_ENTITY_NAMES_CYPHER,
    NULL_NAME_EMBEDDINGS_CYPHER,
    WRITE_NAME_EMBEDDINGS_CYPHER,
    COUNT_NAME_EMBEDDINGS_CYPHER,
)
_FACT_EMBEDDING = _VectorProperty(
    'fact_embedding',
    READ_EDGE_FACTS_CYPHER,
    NULL_FACT_EMBEDDINGS_CYPHER,
    WRITE_FACT_EMBEDDINGS_CYPHER,
    COUNT_FACT_EMBEDDINGS_CYPHER,
)
_VECTOR_PROPERTIES = (_NAME_EMBEDDING, _FACT_EMBEDDING)


async def copy_reference_graph(
    client: GraphCopyClient,
    reference: str,
    target: str,
    *,
    sleep: Sleep = asyncio.sleep,
) -> None:
    require_scratch_name(reference, checkpoint=GuardCheckpoint.GRAPH_COPY)
    require_scratch_name(target, checkpoint=GuardCheckpoint.GRAPH_COPY)
    if target == reference:
        raise ValueError(f'copy target {target!r} is the reference graph itself')
    if target in await client.list_graphs():
        raise ValueError(
            f'copy target {target!r} already exists; tear it down first, because a stale '
            'copy is never reused'
        )
    await _copy_through_fork_refusals(
        client.select_graph(reference), client.connection, reference, target, sleep
    )


@dataclass(frozen=True)
class _ForkWait:
    slept_s: float
    holders: frozenset[ForkHolder]


async def _copy_through_fork_refusals(
    graph: CopyableGraph,
    server: ServerInfo,
    reference: str,
    target: str,
    sleep: Sleep,
) -> None:
    """A refused copy leaves no target behind (measured), so the same copy is simply reissued."""
    refusal: ResponseError | None = None
    waited_s = 0.0
    holders: frozenset[ForkHolder] = frozenset()
    for attempt in range(COPY_ATTEMPTS):
        if attempt:
            wait = await _wait_for_free_fork(server, sleep, reference, target)
            waited_s += wait.slept_s
            holders |= wait.holders
        try:
            await graph.copy(target)
            return
        except ResponseError as error:
            if not _is_fork_refusal(error):
                raise
            refusal = error
    raise GraphCopyRefused(
        reference, target, COPY_ATTEMPTS, waited_s, holders, str(refusal)
    ) from refusal


async def _wait_for_free_fork(
    server: ServerInfo, sleep: Sleep, reference: str, target: str
) -> _ForkWait:
    seen: frozenset[ForkHolder] = frozenset()
    previous: frozenset[ForkHolder] = frozenset()
    polls = 0
    while True:
        await sleep(FORK_POLL_INTERVAL_S)
        polls += 1
        holding = await read_fork_holders(server)
        if not holding:
            return _ForkWait(polls * FORK_POLL_INTERVAL_S, seen)
        if holding != previous or polls % FORK_WAIT_REPORT_EVERY_POLLS == 0:
            logger.info(
                'GRAPH.COPY %r -> %r has waited %.0fs since its refusal; %s holds the fork',
                reference,
                target,
                polls * FORK_POLL_INTERVAL_S,
                ', '.join(sorted(holding)),
            )
        previous = holding
        seen |= holding


def _is_fork_refusal(error: ResponseError) -> bool:
    return str(error).startswith(COPY_FORK_REFUSAL)


async def adopt_group_id(graph: ScratchGraph, name: str) -> GroupAdoption:
    require_scratch_name(name, checkpoint=GuardCheckpoint.REEMBED)
    params: dict[str, object] = {'group_id': name}
    nodes = await graph.query(ADOPT_NODE_GROUP_CYPHER, params)
    edges = await graph.query(ADOPT_EDGE_GROUP_CYPHER, params)
    return GroupAdoption(node_count=_count(nodes), edge_count=_count(edges))


async def reembed_graph(
    graph: ScratchGraph,
    name: str,
    embedder: DocumentEmbedder,
    *,
    write_batch_size: int = DEFAULT_WRITE_BATCH_SIZE,
    clock: Callable[[], float] = time.perf_counter,
) -> GraphReembed:
    require_scratch_name(name, checkpoint=GuardCheckpoint.REEMBED)
    texts = {prop: await _texts(graph, prop) for prop in _VECTOR_PROPERTIES}
    for prop in _VECTOR_PROPERTIES:
        await graph.query(prop.null)
    embedded = {prop: await embedder.embed_documents(texts[prop]) for prop in _VECTOR_PROPERTIES}
    started = clock()
    for prop in _VECTOR_PROPERTIES:
        await _write_vectors(graph, prop, embedded[prop], write_batch_size)
    write_seconds = clock() - started
    for prop in _VECTOR_PROPERTIES:
        await _require_census(graph, name, prop, len(embedded[prop].vectors))
    norms = [norm for result in embedded.values() for norm in result.raw_norms.values()]
    return GraphReembed(
        entity_count=len(texts[_NAME_EMBEDDING]),
        edge_count=len(texts[_FACT_EMBEDDING]),
        written=sum(len(result.vectors) for result in embedded.values()),
        failures=tuple(failure for result in embedded.values() for failure in result.failures),
        raw_norms=norm_stats(norms) if norms else None,
        embed_seconds=sum(result.embed_seconds for result in embedded.values()),
        write_seconds=write_seconds,
    )


async def _texts(graph: ScratchGraph, prop: _VectorProperty) -> list[tuple[str, str]]:
    response = await graph.ro_query(prop.read)
    rows = [(uuid, text) for uuid, text in response.result_set]
    untexted = [uuid for uuid, text in rows if not isinstance(text, str)]
    if untexted:
        raise ValueError(f'{prop.property_name}: records {untexted} carry no text to embed')
    return rows


async def _write_vectors(
    graph: ScratchGraph, prop: _VectorProperty, embedded: DocumentEmbeddings, batch_size: int
) -> None:
    rows: list[dict[str, object]] = [
        {'uuid': key, 'v': list(vector)} for key, vector in embedded.vectors.items()
    ]
    for start in range(0, len(rows), batch_size):
        await graph.query(prop.write, {'rows': rows[start : start + batch_size]})


async def _require_census(
    graph: ScratchGraph, name: str, prop: _VectorProperty, expected: int
) -> None:
    actual = _count(await graph.ro_query(prop.count))
    if actual != expected:
        raise ReembedCensusError(
            f'{name}: {prop.property_name} census read back {actual} non-null vectors, '
            f'expected {expected} (items embedded minus failures)'
        )


def _count(response: Any) -> int:
    return int(response.result_set[0][0])
