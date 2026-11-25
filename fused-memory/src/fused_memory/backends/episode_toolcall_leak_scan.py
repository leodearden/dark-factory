"""Find which of one group's Graphiti episodes carry a leaked tool-call fragment.

The server narrows the transfer with a CONTAINS prefilter over the bound
``PREFILTER_NEEDLES``, and the shared detector re-checks every candidate for
the verdict. The prefilter cannot miss a leak: every ``LEAK_TAIL`` match begins
with one of those literals, and FalkorDB's CONTAINS is an exact, case-sensitive
substring test. Rationale and runbook: docs/mcp-toolcall-xml-leak.md.
"""

from __future__ import annotations

from dataclasses import dataclass

from fused_memory.backends.graph_access import GraphHandle, OpBoundReader, paged_ro_query
from fused_memory.backends.paged_read import DEFAULT_READ_PAGE_SIZE, PagedRead, census_count
from fused_memory.utils.toolcall_xml_leak import PREFILTER_NEEDLES, find_toolcall_xml_leak

__all__ = [
    'EpisodeToolcallLeak',
    'EpisodeToolcallLeakScan',
    'read_episode_toolcall_leaks',
]

_OP = 'scan_episode_toolcall_leaks'

_CANDIDATES_MATCH = (
    'MATCH (e:Episodic) WHERE e.group_id = $group_id '
    'AND any(needle IN $needles WHERE e.content CONTAINS needle) '
)
_CANDIDATES_PAGE = (
    _CANDIDATES_MATCH
    + 'RETURN e.uuid, e.name, e.created_at, e.content ORDER BY e.uuid SKIP {skip} LIMIT {limit}'
)
_CANDIDATES_CENSUS = _CANDIDATES_MATCH + 'RETURN count(*)'
_POPULATION_CENSUS = 'MATCH (e:Episodic) WHERE e.group_id = $group_id RETURN count(*)'


@dataclass(frozen=True, slots=True)
class EpisodeToolcallLeak:
    """One episode whose content the shared detector confirms as leaked."""

    uuid: str
    name: str | None
    created_at: str | None
    content: str
    fragment: str
    index: int


@dataclass(frozen=True, slots=True)
class EpisodeToolcallLeakScan:
    """A group's confirmed leaks, the candidate read they came from, and the denominator.

    ``episodes_in_group`` is None when the population census gave no usable
    count. ``candidates`` carries the prefiltered read's completeness, so a
    caller can tell a short read from a clean group.
    """

    group_id: str
    episodes_in_group: int | None
    candidates: PagedRead
    leaks: tuple[EpisodeToolcallLeak, ...]


async def read_episode_toolcall_leaks(
    graph: GraphHandle, *, group_id: str, page_size: int = DEFAULT_READ_PAGE_SIZE,
) -> EpisodeToolcallLeakScan:
    """Scan *group_id*'s episodes on *graph* for leaks. A read past its budget raises TimeoutError."""
    episodes_in_group = await census_count(
        OpBoundReader(graph, f'{_OP}.population'), _POPULATION_CENSUS, {'group_id': group_id},
    )
    candidates = await paged_ro_query(
        graph,
        _CANDIDATES_PAGE,
        _CANDIDATES_CENSUS,
        op=_OP,
        params={'group_id': group_id, 'needles': list(PREFILTER_NEEDLES)},
        page_size=page_size,
    )
    return EpisodeToolcallLeakScan(
        group_id=group_id,
        episodes_in_group=episodes_in_group,
        candidates=candidates,
        leaks=_confirmed_leaks(candidates.rows),
    )


def _confirmed_leaks(rows: list[list]) -> tuple[EpisodeToolcallLeak, ...]:
    """The rows the shared detector confirms, first occurrence of each uuid kept."""
    seen: set[str] = set()
    leaks: list[EpisodeToolcallLeak] = []
    for uuid, name, created_at, content in rows:
        if uuid in seen:
            continue
        seen.add(uuid)
        hits = find_toolcall_xml_leak(content)
        if hits:
            leaks.append(EpisodeToolcallLeak(
                uuid=uuid,
                name=name,
                created_at=created_at,
                content=content,
                fragment=hits[0].fragment,
                index=hits[0].index,
            ))
    return tuple(leaks)
