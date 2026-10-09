"""Read-only activation measurement for FalkorDB index provisioning (PRD ζ).

The integration gate of ``docs/prds/falkordb-index-provisioning.md``. It reads
every registered graph's index catalog, selects the re-baselined briefing-probe
ids from fresh fulltext counts, fires the briefing probe through the live
fused-memory MCP ``search`` tool, and records the case-fold tripwire, emitting
one JSON record. ``plans/falkordb-index-activation-run/README.md`` is the record
of the run and of why each piece is shaped as it is.
"""
from __future__ import annotations

import asyncio
import contextlib
import dataclasses
import json
import re
import sys
from collections.abc import (
    AsyncIterator,
    Awaitable,
    Callable,
    Collection,
    Iterator,
    Mapping,
    Sequence,
)
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from functools import partial
from pathlib import Path
from typing import Protocol
from urllib.parse import urlparse

from falkordb import FalkorDB
from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client
from shared.cli_boundary import LoudArgumentParser, run_cli

from fused_memory.backends.falkor_indices import (
    IndexSpec,
    expected_index_set,
    normalize_index_records,
    resolve_header_positions,
    unsettled_index_statuses,
)
from fused_memory.backends.graphiti_client import GraphitiBackend
from fused_memory.config.schema import FusedMemoryConfig
from fused_memory.models.scope import KNOWN_PROJECT_ROOTS_ENV, known_project_roots_from_env
from fused_memory.reconciliation.index_health import summarize_index_health

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = Path('plans') / 'falkordb-index-activation-run'

ORIGINAL_PROBE_IDS = ('877', '2286', '3127', '3600', '1157')
DRIFTED_IDS = ('877', '3600')
RETIRED_TASK_TEMPLATE = 'task {task_id} context and related decisions'
PROBE_LIMIT = 5
PROBE_FLOOR = 4
REPLACEMENT_FLOOR = 2
REPLACEMENT_MAX_DISTANCE = 50

GRAPHITI_STORE = 'graphiti'
PROBE_PROJECT_ID = 'dark_factory'
FULLTEXT_GRAPH = 'dark_factory'

MCP_URL = 'http://127.0.0.1:8002/mcp'
MCP_CALL_TIMEOUT = timedelta(seconds=120)
CALLER_AGENT_ID = 'claude-task-3711-activation-probe'
CALLER_TASK_ID = '3711'

READ_TIMEOUT_MS = 60000

CASE_FOLD_SPLIT_SINCE = '2026-10-02T11:42:23+00:00'
CASE_FOLD_SPLIT_SINCE_COMMIT = '13a9caaca2'

INDEXES = 'CALL db.indexes()'
FULLTEXT = (
    "CALL db.idx.fulltext.queryRelationships('RELATES_TO', $q) YIELD relationship AS rel "
    'RETURN rel.uuid, rel.fact, rel.invalid_at, rel.expired_at'
)
DUP_UUID = 'MATCH ()-[e:RELATES_TO]->() WITH e.uuid AS u, count(e) AS c WHERE c > 1 RETURN count(u)'
CASE_FOLD_POPULATION = (
    'MATCH (n:Entity) WITH toLower(n.name) AS k, count(n) AS c, '
    'count(DISTINCT n.name) AS names WHERE c > 1 AND names > 1 RETURN count(k), sum(c)'
)
CASE_FOLD_SPLIT = (
    'MATCH ()-[e:RELATES_TO]->() WHERE e.created_at > $since '
    'WITH startNode(e) AS a, endNode(e) AS b UNWIND [a, b] AS n WITH DISTINCT n '
    'WHERE n:Entity WITH toLower(n.name) AS k, collect(DISTINCT n.name) AS spellings '
    'WHERE size(spellings) > 1 RETURN k, spellings'
)

EXIT_BAD_INPUT = 2
EXIT_RUN_FAILED = 1

INDEX_COLUMNS = {
    'label': 'label',
    'field': 'properties',
    'type': 'types',
    'entity_type': 'entitytype',
    'status': 'status',
}

_JOINER = r'(?:\s*(?:,|&|/|\band\b|\bor\b))+\s*'
_TASK_REFERENCE_RE = re.compile(
    rf'\btasks?\b[\s#:/-]*(?P<ids>\d+\b(?:{_JOINER}#?\d+\b)*)',
    re.IGNORECASE,
)
_NUMBER_RE = re.compile(r'\d+')


@dataclass(frozen=True)
class FulltextEdge:
    uuid: str
    fact: str | None
    invalid_at: str | None
    expired_at: str | None


@dataclass(frozen=True)
class CandidateEdges:
    task_id: str
    edges: tuple[FulltextEdge, ...]


@dataclass(frozen=True)
class ReplacementSelection:
    anchor: str
    chosen: str
    examined: tuple[CandidateEdges, ...]


@dataclass(frozen=True)
class SearchHit:
    id: str
    source_store: str
    content: str
    invalid_at: str | None


@dataclass(frozen=True)
class BriefingVerdict:
    hits: int
    total: int
    floor: int
    passed: bool
    missed: tuple[str, ...]


@dataclass(frozen=True)
class GraphIndexStatus:
    group_id: str
    present: bool
    expected_total: int
    actual: tuple[IndexSpec, ...]
    missing: tuple[IndexSpec, ...]
    unexpected: tuple[IndexSpec, ...]
    unsettled: tuple[tuple[object, object], ...]
    complete: bool


@dataclass(frozen=True)
class QueryRows:
    header: list
    rows: list


@dataclass(frozen=True)
class SearchOutcome:
    results: tuple[SearchHit, ...]
    degraded: bool


@dataclass(frozen=True)
class DupUuidCount:
    group_id: str
    groups: int


@dataclass(frozen=True)
class E1Preflight:
    dup_uuid_groups: tuple[DupUuidCount, ...]
    incomplete_graphs: tuple[str, ...]
    maintenance_noop: bool


@dataclass(frozen=True)
class ProbeQuery:
    task_id: str
    query: str
    degraded: bool
    results: tuple[SearchHit, ...]
    hit: bool


@dataclass(frozen=True)
class ProbeRun:
    template: str
    stores: tuple[str, ...] | None
    limit: int
    queries: tuple[ProbeQuery, ...]
    verdict: BriefingVerdict | None


@dataclass(frozen=True)
class SplitGroup:
    key: str
    spellings: tuple[str, ...]


@dataclass(frozen=True)
class CaseFoldRecord:
    group_id: str
    keys: object
    nodes: object
    split_since: str
    split_since_commit: str
    split_groups: tuple[SplitGroup, ...]


@dataclass(frozen=True)
class ActivationRecord:
    measured_at: str
    mcp_url: str
    registry: tuple[str, ...]
    listed_graphs: tuple[str, ...]
    expected_index_set: tuple[IndexSpec, ...]
    index_statuses: tuple[GraphIndexStatus, ...]
    e1_preflight: E1Preflight
    replacements: tuple[ReplacementSelection, ...]
    original_id_counts: tuple[CandidateEdges, ...]
    asserted_probe: ProbeRun
    original_ids_probe: ProbeRun
    unscoped_probe: ProbeRun
    case_fold: tuple[CaseFoldRecord, ...]


class GraphReader(Protocol):
    def list_graphs(self) -> list[str]: ...

    def ro_query(
        self, graph: str, cypher: str, params: Mapping[str, object] | None = None,
    ) -> QueryRows: ...


class SearchFn(Protocol):
    def __call__(
        self, query: str, *, stores: list[str] | None, limit: int,
    ) -> Awaitable[SearchOutcome]: ...


class NoReplacementError(LookupError):
    """No candidate id within the walk qualified as a replacement."""


class SearchPayloadShapeError(ValueError):
    """The MCP search tool answered in a shape this probe does not recognise."""


class SearchToolError(RuntimeError):
    """The MCP search tool reported an error."""


class RegistryUnavailableError(RuntimeError):
    """The project registry input is unset, so the sweep would silently narrow."""


def mentions_task(text: str | None, task_id: str) -> bool:
    """Whether *text* names *task_id* as a task, e.g. 'Task 3127' or 'tasks 2293 and 2286'."""
    return any(
        task_id in _NUMBER_RE.findall(match.group('ids'))
        for match in _TASK_REFERENCE_RE.finditer(text or '')
    )


def is_live(edge: FulltextEdge) -> bool:
    return edge.invalid_at is None and edge.expired_at is None


def genuine_live_count(edges: Sequence[FulltextEdge], task_id: str) -> int:
    return sum(1 for edge in edges if is_live(edge) and mentions_task(edge.fact, task_id))


def candidate_ids(anchor: str, exclude: Collection[str], max_distance: int) -> Iterator[str]:
    """Ids around *anchor*, nearest first and below before above, skipping *exclude*."""
    centre = int(anchor)
    for distance in range(1, max_distance + 1):
        for candidate in (centre - distance, centre + distance):
            if candidate >= 1 and str(candidate) not in exclude:
                yield str(candidate)


def select_replacement(
    anchor: str,
    measure: Callable[[str], Sequence[FulltextEdge]],
    *,
    exclude: Collection[str],
    floor: int = REPLACEMENT_FLOOR,
    max_distance: int = REPLACEMENT_MAX_DISTANCE,
) -> ReplacementSelection:
    """The first outward candidate with at least *floor* live genuine edges."""
    examined: list[CandidateEdges] = []
    for task_id in candidate_ids(anchor, exclude, max_distance):
        edges = tuple(measure(task_id))
        examined.append(CandidateEdges(task_id=task_id, edges=edges))
        if genuine_live_count(edges, task_id) >= floor:
            return ReplacementSelection(anchor=anchor, chosen=task_id, examined=tuple(examined))
    raise NoReplacementError(
        f'no replacement for task {anchor}: no id within distance {max_distance} '
        f'has {floor} or more live edges that mention it as a task '
        f'({len(examined)} candidates examined)'
    )


def select_replacements(
    measure: Callable[[str], Sequence[FulltextEdge]],
    *,
    anchors: Sequence[str] = DRIFTED_IDS,
    exclude: Collection[str] = ORIGINAL_PROBE_IDS,
) -> tuple[ReplacementSelection, ...]:
    """One replacement per anchor, in order; an earlier choice is excluded from later walks."""
    selections: list[ReplacementSelection] = []
    for anchor in anchors:
        taken = {*exclude, *(selection.chosen for selection in selections)}
        selections.append(select_replacement(anchor, measure, exclude=taken))
    return tuple(selections)


def rebaselined_ids(replacements: Sequence[ReplacementSelection]) -> tuple[str, ...]:
    """The original probe ids with each drifted id swapped for its replacement, in place."""
    chosen = {selection.anchor: selection.chosen for selection in replacements}
    return tuple(chosen.get(task_id, task_id) for task_id in ORIGINAL_PROBE_IDS)


def briefing_hits(results: Sequence[SearchHit], task_id: str) -> list[SearchHit]:
    """The Graphiti results that mention *task_id* as a task."""
    return [
        result for result in results
        if result.source_store == GRAPHITI_STORE and mentions_task(result.content, task_id)
    ]


def query_hit(results: Sequence[SearchHit], task_id: str) -> bool:
    return bool(briefing_hits(results, task_id))


def briefing_verdict(hits_by_id: Mapping[str, bool], floor: int = PROBE_FLOOR) -> BriefingVerdict:
    hits = sum(1 for hit in hits_by_id.values() if hit)
    return BriefingVerdict(
        hits=hits,
        total=len(hits_by_id),
        floor=floor,
        passed=hits >= floor,
        missed=tuple(task_id for task_id, hit in hits_by_id.items() if not hit),
    )


def index_records(header, rows) -> list[dict]:
    """``CALL db.indexes()`` rows as the records ``GraphitiBackend.list_indices`` returns."""
    positions = resolve_header_positions(header, INDEX_COLUMNS)
    return [{key: row[position] for key, position in positions.items()} for row in rows]


def graph_index_status(
    group_id: str,
    records: Sequence[Mapping] | None,
    expected: Collection[IndexSpec],
) -> GraphIndexStatus:
    """One graph's catalog against *expected*; ``records`` is None when the graph key is absent."""
    if records is None:
        return GraphIndexStatus(
            group_id=group_id, present=False, expected_total=len(expected),
            actual=(), missing=(), unexpected=(), unsettled=(), complete=False,
        )
    actual = normalize_index_records(records)
    health = summarize_index_health(actual, set(expected))
    unsettled = tuple(unsettled_index_statuses(records))
    return GraphIndexStatus(
        group_id=group_id,
        present=True,
        expected_total=health['expected_total'],
        actual=tuple(sorted(actual)),
        missing=tuple(health['missing']),
        unexpected=tuple(health['unexpected']),
        unsettled=unsettled,
        complete=health['healthy'] and not unsettled,
    )


def incomplete_graph_ids(statuses: Sequence[GraphIndexStatus]) -> list[str]:
    """Present graphs whose catalog is not complete; an absent graph is not incomplete."""
    return [status.group_id for status in statuses if status.present and not status.complete]


def require_known_project_roots(roots: list[str]) -> list[str]:
    if not roots:
        raise RegistryUnavailableError(
            f'{KNOWN_PROJECT_ROOTS_ENV} is unset or empty, so the registry would '
            'narrow to the primary project alone and the sweep would certify one '
            'graph as "every registered graph". It lives in the fused-memory unit, '
            'not in .env; read it with '
            '`systemctl --user show fused-memory.service -p Environment` and export it.'
        )
    return roots


def e1_preflight(
    statuses: Sequence[GraphIndexStatus], dup_counts: Sequence[DupUuidCount],
) -> E1Preflight:
    """Whether E1's startup maintenance (index sweep, dup-uuid repair) would write nothing."""
    incomplete = tuple(incomplete_graph_ids(statuses))
    return E1Preflight(
        dup_uuid_groups=tuple(dup_counts),
        incomplete_graphs=incomplete,
        maintenance_noop=not incomplete and all(count.groups == 0 for count in dup_counts),
    )


def _search_hit(result: object) -> SearchHit:
    if not isinstance(result, Mapping) or not all(
        isinstance(result.get(key), str) for key in ('id', 'source_store', 'content')
    ):
        raise SearchPayloadShapeError(f'search result lacks id/source_store/content: {result!r}')
    temporal = result.get('temporal') or {}
    return SearchHit(
        id=result['id'],
        source_store=result['source_store'],
        content=result['content'],
        invalid_at=temporal.get('invalid_at'),
    )


def parse_search_payload(payload: object) -> SearchOutcome:
    """The MCP ``search`` tool's structured result as a :class:`SearchOutcome`."""
    if isinstance(payload, Mapping) and set(payload) == {'result'}:
        payload = payload['result']
    if not isinstance(payload, Mapping) or not isinstance(payload.get('results'), list):
        raise SearchPayloadShapeError(f'search returned no results list: {payload!r}')
    return SearchOutcome(
        results=tuple(_search_hit(result) for result in payload['results']),
        degraded=bool(payload.get('degraded') or payload.get('failed_stores')),
    )


class FalkorReadOnlyReader:
    """FalkorDB through ``GRAPH.RO_QUERY`` only, so the server itself rejects any write."""

    def __init__(self, client) -> None:
        self._client = client

    @classmethod
    def connect(cls, uri: str, password: str | None) -> FalkorReadOnlyReader:
        parsed = urlparse(uri)
        return cls(FalkorDB(host=parsed.hostname or 'localhost', port=parsed.port or 6379, password=password))

    def list_graphs(self) -> list[str]:
        return list(self._client.list_graphs())

    def ro_query(
        self, graph: str, cypher: str, params: Mapping[str, object] | None = None,
    ) -> QueryRows:
        result = self._client.select_graph(graph).ro_query(
            cypher, dict(params) if params is not None else None, timeout=READ_TIMEOUT_MS,
        )
        return QueryRows(header=result.header, rows=result.result_set)


@contextlib.asynccontextmanager
async def open_mcp_search(url: str = MCP_URL) -> AsyncIterator[SearchFn]:
    """A :class:`SearchFn` over the live fused-memory MCP ``search`` tool.

    An exception from the ``async with`` body is re-raised once the session has
    closed, so the caller receives it as itself rather than inside the
    ExceptionGroup the MCP client's anyio task groups would wrap it in.
    """
    body_error: Exception | None = None
    async with streamablehttp_client(url) as (read, write, _session_id), ClientSession(read, write) as session:
        await session.initialize()

        async def search(query: str, *, stores: list[str] | None, limit: int) -> SearchOutcome:
            arguments: dict[str, object] = {
                'query': query,
                'project_id': PROBE_PROJECT_ID,
                'limit': limit,
                'caller_agent_id': CALLER_AGENT_ID,
                'caller_task_id': CALLER_TASK_ID,
            }
            if stores is not None:
                arguments['stores'] = stores
            result = await session.call_tool('search', arguments, read_timeout_seconds=MCP_CALL_TIMEOUT)
            if result.isError:
                raise SearchToolError(f'search {query!r} failed: {result.content!r}')
            return parse_search_payload(result.structuredContent)

        try:
            yield search
        except Exception as exc:
            body_error = exc
    if body_error is not None:
        raise body_error


def _single_row(rows: QueryRows) -> list:
    (row,) = rows.rows
    return row


def _read_index_records(reader: GraphReader, group_id: str) -> list[dict]:
    rows = reader.ro_query(group_id, INDEXES)
    return index_records(rows.header, rows.rows)


def _fulltext_edges(reader: GraphReader, task_id: str) -> tuple[FulltextEdge, ...]:
    rows = reader.ro_query(FULLTEXT_GRAPH, FULLTEXT, {'q': task_id}).rows
    return tuple(
        FulltextEdge(uuid=uuid, fact=fact, invalid_at=invalid_at, expired_at=expired_at)
        for uuid, fact, invalid_at, expired_at in rows
    )


def _case_fold(reader: GraphReader, group_id: str) -> CaseFoldRecord:
    keys, nodes = _single_row(reader.ro_query(group_id, CASE_FOLD_POPULATION))
    split = reader.ro_query(group_id, CASE_FOLD_SPLIT, {'since': CASE_FOLD_SPLIT_SINCE}).rows
    return CaseFoldRecord(
        group_id=group_id,
        keys=keys,
        nodes=nodes,
        split_since=CASE_FOLD_SPLIT_SINCE,
        split_since_commit=CASE_FOLD_SPLIT_SINCE_COMMIT,
        split_groups=tuple(sorted(
            (SplitGroup(key=key, spellings=tuple(sorted(spellings))) for key, spellings in split),
            key=lambda group: group.key,
        )),
    )


async def run_probe(
    search: SearchFn, task_ids: Sequence[str], *, stores: tuple[str, ...] | None,
) -> ProbeRun:
    queries: list[ProbeQuery] = []
    for task_id in task_ids:
        query = RETIRED_TASK_TEMPLATE.format(task_id=task_id)
        outcome = await search(
            query, stores=list(stores) if stores is not None else None, limit=PROBE_LIMIT,
        )
        queries.append(ProbeQuery(
            task_id=task_id,
            query=query,
            degraded=outcome.degraded,
            results=outcome.results,
            hit=query_hit(outcome.results, task_id),
        ))
    return ProbeRun(
        template=RETIRED_TASK_TEMPLATE, stores=stores, limit=PROBE_LIMIT,
        queries=tuple(queries), verdict=None,
    )


def with_verdict(probe: ProbeRun) -> ProbeRun:
    hits_by_id = {query.task_id: query_hit(query.results, query.task_id) for query in probe.queries}
    return dataclasses.replace(probe, verdict=briefing_verdict(hits_by_id, floor=PROBE_FLOOR))


async def measure(
    reader: GraphReader, search: SearchFn, *, registry: Collection[str], measured_at: str,
) -> ActivationRecord:
    expected = expected_index_set()
    listed = sorted(reader.list_graphs())
    statuses = tuple(
        graph_index_status(
            group_id, _read_index_records(reader, group_id) if group_id in listed else None, expected,
        )
        for group_id in sorted(registry)
    )
    preflight = e1_preflight(statuses, [
        DupUuidCount(group_id=group_id, groups=_single_row(reader.ro_query(group_id, DUP_UUID))[0])
        for group_id in listed
    ])

    fulltext = partial(_fulltext_edges, reader)
    replacements = select_replacements(fulltext)
    original_id_counts = tuple(
        CandidateEdges(task_id=task_id, edges=fulltext(task_id)) for task_id in ORIGINAL_PROBE_IDS
    )

    probe_ids = rebaselined_ids(replacements)
    asserted = with_verdict(await run_probe(search, probe_ids, stores=(GRAPHITI_STORE,)))
    original = await run_probe(search, ORIGINAL_PROBE_IDS, stores=(GRAPHITI_STORE,))
    unscoped = await run_probe(search, probe_ids, stores=None)

    return ActivationRecord(
        measured_at=measured_at,
        mcp_url=MCP_URL,
        registry=tuple(sorted(registry)),
        listed_graphs=tuple(listed),
        expected_index_set=tuple(sorted(expected)),
        index_statuses=statuses,
        e1_preflight=preflight,
        replacements=replacements,
        original_id_counts=original_id_counts,
        asserted_probe=asserted,
        original_ids_probe=original,
        unscoped_probe=unscoped,
        case_fold=tuple(_case_fold(reader, status.group_id) for status in statuses if status.present),
    )


def record_to_json(record: ActivationRecord) -> str:
    return json.dumps(dataclasses.asdict(record), indent=2, ensure_ascii=False) + '\n'


def summary_text(record: ActivationRecord, path: Path) -> str:
    lines = [f'activation record: {path}', 'index catalog per registered graph:']
    for status in record.index_statuses:
        state = 'absent' if not status.present else 'complete' if status.complete else 'INCOMPLETE'
        lines.append(
            f'  {status.group_id}: {state} (actual {len(status.actual)}/{status.expected_total}, '
            f'missing {len(status.missing)}, unsettled {len(status.unsettled)})'
        )
    lines.append(f'E1 maintenance_noop: {record.e1_preflight.maintenance_noop}')
    for selection in record.replacements:
        chosen_edges = selection.examined[-1].edges
        lines.append(
            f'replacement {selection.anchor} -> {selection.chosen} '
            f'({genuine_live_count(chosen_edges, selection.chosen)} live genuine edges, '
            f'{len(selection.examined)} examined)'
        )
    for name, probe in (
        ('asserted', record.asserted_probe),
        ('original ids', record.original_ids_probe),
        ('unscoped', record.unscoped_probe),
    ):
        hits = sum(1 for query in probe.queries if query.hit)
        degraded = sum(1 for query in probe.queries if query.degraded)
        lines.append(
            f'probe {name}: {hits}/{len(probe.queries)} hits, {degraded} degraded '
            f'({" ".join(q.task_id + ("+" if q.hit else "-") for q in probe.queries)})'
        )
    for case_fold in record.case_fold:
        lines.append(
            f'case-fold {case_fold.group_id}: keys {case_fold.keys}, nodes {case_fold.nodes}, '
            f'split groups {len(case_fold.split_groups)}'
        )
    return '\n'.join(lines)


def build_parser() -> LoudArgumentParser:
    parser = LoudArgumentParser(description=__doc__)
    parser.add_argument(
        '--out-dir', type=Path, default=DEFAULT_OUT_DIR,
        help=f'Directory for activation-<stamp>.json, relative to the repo root (default: {DEFAULT_OUT_DIR})',
    )
    return parser


async def _run(config: FusedMemoryConfig, registry: Collection[str], measured_at: str) -> ActivationRecord:
    falkordb = config.graphiti.falkordb
    reader = FalkorReadOnlyReader.connect(falkordb.uri, falkordb.password)
    async with open_mcp_search() as search:
        return await measure(reader, search, registry=registry, measured_at=measured_at)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        require_known_project_roots(known_project_roots_from_env())
    except RegistryUnavailableError as exc:
        print(f'error: {exc}', file=sys.stderr)
        return EXIT_BAD_INPUT
    config = FusedMemoryConfig()
    registry = GraphitiBackend(config).registered_graph_ids
    now = datetime.now(UTC).replace(microsecond=0)
    try:
        record = asyncio.run(_run(config, registry, now.isoformat().replace('+00:00', 'Z')))
    except (NoReplacementError, SearchPayloadShapeError, SearchToolError) as exc:
        print(f'error: {exc}', file=sys.stderr)
        return EXIT_RUN_FAILED
    out_dir = REPO_ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f'activation-{now.strftime("%Y%m%dT%H%M%SZ")}.json'
    path.write_text(record_to_json(record), encoding='utf-8')
    print(summary_text(record, path))
    return 0


if __name__ == '__main__':
    sys.exit(run_cli(main))
