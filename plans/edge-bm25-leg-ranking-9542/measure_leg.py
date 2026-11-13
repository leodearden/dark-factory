"""Task 9542 gates M1 and M2: the edge BM25 leg's rankings on production, read-only.

Every query reaches FalkorDB through GRAPH.RO_QUERY (``falkordb.Graph.ro_query``),
which the server refuses to run if it would write.  The shipped leg is
``search_utils.edge_fulltext_search`` on an unconnected
``_MultiTenantFalkorDriver`` whose ``execute_query`` replays each issued query
with ``ro_query``, as ``tests/_query_recorder.py`` records them.

Variants, all on the leg's own union Cypher:

* A, pre-change: the query ``build_query(sanitize(q), [graph], 128)``, no promotion;
* B, union only: the shipped union query, no promotion;
* C, the shipped leg, identifier tier included.

Usage: ``.venv/bin/python plans/edge-bm25-leg-ranking-9542/measure_leg.py OUT.json``
"""

from __future__ import annotations

import asyncio
import datetime
import json
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from falkordb import FalkorDB
from fused_memory.backends.falkor_fulltext import build_query
from fused_memory.backends.graphiti_client import _MultiTenantFalkorDriver
from graphiti_core.search import search_utils
from graphiti_core.search.search_filters import SearchFilters

HOST, PORT, GRAPH = 'localhost', 6379, 'dark_factory'
WRITE_PATH_LIMIT = 20
READ_PATH_LIMIT = 16
M1_SAMPLE = 200
QUERY_SAMPLE = Path('fused-memory/tests/fixtures/production_query_sample.jsonl')
TICKET_877_QUERY = 'task 877 context and related decisions'
TICKET_877_EDGES = ('f0465663', 'dbf74612', '38241d9e', 'bba24b78')

LIVE_EDGES = (
    'MATCH ()-[e:RELATES_TO]->() WHERE e.invalid_at IS NULL AND e.group_id = $group'
    ' WITH e, startNode(e) AS n, endNode(e) AS m WHERE n:Entity AND m:Entity'
    ' RETURN e.uuid AS uuid, e.fact AS fact ORDER BY uuid SKIP $skip LIMIT $page'
)
# FalkorDB caps a result set at 10000 rows, so the live edges are read in pages.
LIVE_EDGE_PAGE = 5000


@dataclass
class ReadOnlyGraph:
    graph: Any
    issued: list[tuple[str, dict[str, Any], int]] = field(default_factory=list)

    def rows(self, cypher: str, params: dict[str, Any]) -> list[dict[str, Any]]:
        server_params = {key: value for key, value in params.items() if key != 'routing_'}
        result = self.graph.ro_query(cypher, server_params)
        header = [column[1] for column in result.header]
        return [dict(zip(header, row, strict=False)) for row in result.result_set]

    def driver(self) -> _MultiTenantFalkorDriver:
        driver = object.__new__(_MultiTenantFalkorDriver)

        async def execute(cypher: str, **params: Any):
            records = self.rows(cypher, params)
            self.issued.append((cypher, params, len(records)))
            return records, [], None

        driver.execute_query = execute  # pyright: ignore[reportAttributeAccessIssue]
        return driver


@dataclass(frozen=True)
class Variants:
    a: list[str]
    b: list[str]
    c: list[str]
    tier_rows: int | None
    promoted: list[str]
    facts: dict[str, str]
    seconds_c: float


async def variants(ro: ReadOnlyGraph, query: str, limit: int) -> Variants:
    ro.issued.clear()
    driver = ro.driver()
    started = time.perf_counter()
    edges = await search_utils.edge_fulltext_search(driver, query, SearchFilters(), [GRAPH], limit)
    seconds_c = time.perf_counter() - started
    issued = list(ro.issued)
    if not issued:
        return Variants([], [], [], None, [], {}, seconds_c)
    union_cypher, union_params, _ = issued[-1]
    tier_rows = issued[0][2] if len(issued) == 2 else None
    b_rows = ro.rows(union_cypher, {**union_params, 'promoted_uuids': []})
    a_query = build_query(driver.sanitize(query), [GRAPH], search_utils.MAX_QUERY_LENGTH)
    a_rows = (
        []
        if a_query == ''
        else ro.rows(union_cypher, {**union_params, 'query': a_query, 'promoted_uuids': []})
    )
    facts = {row['uuid']: row['fact'] for row in [*a_rows, *b_rows]}
    facts.update({edge.uuid: edge.fact for edge in edges})
    return Variants(
        a=[row['uuid'] for row in a_rows],
        b=[row['uuid'] for row in b_rows],
        c=[edge.uuid for edge in edges],
        tier_rows=tier_rows,
        promoted=list(union_params['promoted_uuids']),
        facts=facts,
        seconds_c=seconds_c,
    )


def overlap(x: list[str], y: list[str]) -> float:
    """|X & Y| / max(|X|, |Y|); 1.0 when both are empty."""
    if not x and not y:
        return 1.0
    return len(set(x) & set(y)) / max(len(x), len(y))


def rank_of(uuid: str, ranking: list[str]) -> int | None:
    return ranking.index(uuid) + 1 if uuid in ranking else None


def mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def live_edges(ro: ReadOnlyGraph) -> list[dict[str, Any]]:
    edges: list[dict[str, Any]] = []
    while True:
        page = ro.rows(LIVE_EDGES, {'group': GRAPH, 'skip': len(edges), 'page': LIVE_EDGE_PAGE})
        edges.extend(page)
        if len(page) < LIVE_EDGE_PAGE:
            return edges


async def measure_write_path(ro: ReadOnlyGraph) -> dict[str, Any]:
    live = live_edges(ro)
    stride = len(live) // M1_SAMPLE
    sample = live[::stride][:M1_SAMPLE]
    per_fact = []
    for edge in sample:
        result = await variants(ro, edge['fact'], WRITE_PATH_LIMIT)
        per_fact.append(
            {
                'uuid': edge['uuid'],
                'rank': {
                    name: rank_of(edge['uuid'], ranking)
                    for name, ranking in (('A', result.a), ('B', result.b), ('C', result.c))
                },
                'overlap_ab': overlap(result.a, result.b),
                'overlap_bc': overlap(result.b, result.c),
                'tier_rows': result.tier_rows,
                'tier_fired': bool(result.promoted),
                'seconds_c': result.seconds_c,
            }
        )
    summary = {
        'live_edges': len(live),
        'stride': stride,
        'sample': len(sample),
        'rule': f'every {stride}-th live edge in uuid order, starting at the first, first {M1_SAMPLE}',
    }
    for name in ('A', 'B', 'C'):
        ranks = [row['rank'][name] for row in per_fact if row['rank'][name] is not None]
        summary[f'self_recall_at_20_{name}'] = len(ranks) / len(per_fact)
        summary[f'mean_self_rank_{name}'] = mean([float(rank) for rank in ranks])
    summary['mean_overlap_ab'] = mean([row['overlap_ab'] for row in per_fact])
    summary['mean_overlap_bc'] = mean([row['overlap_bc'] for row in per_fact])
    summary['tier_issued_share'] = sum(row['tier_rows'] is not None for row in per_fact) / len(per_fact)
    summary['tier_fired_share'] = sum(row['tier_fired'] for row in per_fact) / len(per_fact)
    summary['max_seconds_c'] = max(row['seconds_c'] for row in per_fact)
    return {'summary': summary, 'per_fact': per_fact}


async def measure_read_path(ro: ReadOnlyGraph) -> dict[str, Any]:
    rows = [json.loads(line) for line in QUERY_SAMPLE.read_text().splitlines() if line.strip()]
    per_query = []
    for row in rows:
        result = await variants(ro, row['text'], READ_PATH_LIMIT)
        per_query.append(
            {
                'query_id': row['query_id'],
                'text': row['text'],
                'observed_count': row['observed_count'],
                'overlap_ab': overlap(result.a, result.b),
                'overlap_bc': overlap(result.b, result.c),
                'changed_b': result.b != result.a,
                'changed_c': result.c != result.a,
                'tier_rows': result.tier_rows,
                'promoted_facts': [result.facts[uuid] for uuid in result.promoted],
            }
        )
    total = sum(row['observed_count'] for row in per_query)
    summary = {
        'queries': len(per_query),
        'weighted_share_changed_b': sum(r['observed_count'] for r in per_query if r['changed_b']) / total,
        'weighted_share_changed_c': sum(r['observed_count'] for r in per_query if r['changed_c']) / total,
    }
    ticket = await variants(ro, TICKET_877_QUERY, READ_PATH_LIMIT)
    top4 = ticket.c[:4]
    ticket_877 = {
        'query': TICKET_877_QUERY,
        'tier_rows': ticket.tier_rows,
        'promoted': ticket.promoted,
        'ranking_c': [{'uuid': uuid, 'fact': ticket.facts[uuid]} for uuid in ticket.c],
        'ranks_a': {prefix: rank_of_prefix(prefix, ticket.a) for prefix in TICKET_877_EDGES},
        'ranks_b': {prefix: rank_of_prefix(prefix, ticket.b) for prefix in TICKET_877_EDGES},
        'ranks_c': {prefix: rank_of_prefix(prefix, ticket.c) for prefix in TICKET_877_EDGES},
        'gate_top4_are_the_877_edges': sorted(uuid[:8] for uuid in top4) == sorted(TICKET_877_EDGES),
    }
    return {'summary': summary, 'per_query': per_query, 'ticket_877': ticket_877}


def rank_of_prefix(prefix: str, ranking: list[str]) -> int | None:
    for index, uuid in enumerate(ranking):
        if uuid.startswith(prefix):
            return index + 1
    return None


def now() -> str:
    return datetime.datetime.now(datetime.UTC).strftime('%Y-%m-%dT%H:%M:%SZ')


async def main(out: Path) -> None:
    client = FalkorDB(host=HOST, port=PORT)
    ro = ReadOnlyGraph(client.select_graph(GRAPH))
    head = subprocess.run(['git', 'rev-parse', 'HEAD'], capture_output=True, text=True, check=True)
    context = {'host': f'{HOST}:{PORT}', 'graph': GRAPH, 'head': head.stdout.strip(), 'start': now()}
    report = {'context': context, 'm1': await measure_write_path(ro), 'm2': await measure_read_path(ro)}
    context['end'] = now()
    out.write_text(json.dumps(report, indent=1, ensure_ascii=False) + '\n')


if __name__ == '__main__':
    asyncio.run(main(Path(sys.argv[1])))
