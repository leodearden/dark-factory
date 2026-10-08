"""Read-only activation measurement for FalkorDB index provisioning (PRD ζ).

The integration gate of ``docs/prds/falkordb-index-provisioning.md``. It reads
every registered graph's index catalog, selects the re-baselined briefing-probe
ids from fresh fulltext counts, fires the briefing probe through the live
fused-memory MCP ``search`` tool, and records the case-fold tripwire, emitting
one JSON record. ``plans/falkordb-index-activation-run/README.md`` is the record
of the run and of why each piece is shaped as it is.
"""
from __future__ import annotations

import re
from collections.abc import Callable, Collection, Iterator, Mapping, Sequence
from dataclasses import dataclass

from fused_memory.backends.falkor_indices import (
    IndexSpec,
    expected_index_set,
    normalize_index_records,
    resolve_header_positions,
    unsettled_index_statuses,
)
from fused_memory.models.scope import KNOWN_PROJECT_ROOTS_ENV
from fused_memory.reconciliation.index_health import summarize_index_health

ORIGINAL_PROBE_IDS = ('877', '2286', '3127', '3600', '1157')
DRIFTED_IDS = ('877', '3600')
RETIRED_TASK_TEMPLATE = 'task {task_id} context and related decisions'
PROBE_LIMIT = 5
PROBE_FLOOR = 4
REPLACEMENT_FLOOR = 2
REPLACEMENT_MAX_DISTANCE = 50

GRAPHITI_STORE = 'graphiti'

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


class NoReplacementError(LookupError):
    """No candidate id within the walk qualified as a replacement."""


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


def sorted_expected_specs() -> tuple[IndexSpec, ...]:
    return tuple(sorted(expected_index_set()))


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
