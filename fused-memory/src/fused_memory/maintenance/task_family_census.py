"""Count the task-node families that are still fragmented — read-only.

Task families split when graphiti_core's LLM extraction mints a task node under
whichever spelling the source text used: 'Task 605', 'task 605', 'tasks 605'
and 'task #605' become four nodes for one task, splitting its edges four ways.
``MemoryService._normalize_task_node_names`` repairs the families an episode
TOUCHES, on the write path, as they are touched.

This module reports the RESIDUE that fix can never reach: families belonging to
tasks no future episode will mention again. No amount of correct write-path
behaviour collapses those, because nothing will ever trigger the write path for
them — so the only way to know how many there are is to go and look.

Scope, deliberately narrow. This is option (a) of task 5264's three: REPORT the
residue. Wiring the census into ``GraphitiBackend._scan_duplicate_entity_names``
(option b) and running an unattended collapse sweep (option c) are NOT
authorized by that task and are not implemented here. Nothing in this module
mutates anything: it calls ``enumerate_entity_nodes`` and
``find_entity_nodes_by_name_substring``, both ``ro_query``-backed, and nothing
else. That read-only guarantee is structural rather than a promise in prose —
tests/test_task_family_census.py asserts it against the backend's mutating
methods and against the driver's writable query channel.

Family membership is decided by ``utils.task_naming.group_task_node_families``,
the same function the write-path normalizer uses, which is what keeps the two
from ever disagreeing about what belongs together.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

from fused_memory.utils.task_naming import group_task_node_families

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class VariantSpelling:
    """One node in a fragmented family: how it is spelled, and what it holds.

    ``edge_count`` is the number of still-VALID edges hanging off the node. It
    is the field that turns a count into a decision: it says which node a
    collapse would keep and how many edges the others would have to move.
    """

    name: str
    uuid: str
    edge_count: int


@dataclass(frozen=True)
class FragmentedFamily:
    """One task whose nodes are still split across more than one node.

    ``variants`` is ordered survivor-first — most valid edges, then oldest,
    then uuid — which is the order the backend returns and the order a collapse
    would use, so ``variants[0]`` reads as "the node that would survive".
    """

    canonical_name: str
    variants: tuple[VariantSpelling, ...]


@dataclass(frozen=True)
class GraphCensus:
    """What one graph's census found, and how much of that graph it saw.

    The completeness fields are not decoration. This module's entire output is
    a number a human will act on, and a number derived from a silently
    truncated enumeration is worse than no number at all — so the
    ``PagedRead`` that ``enumerate_entity_nodes`` already returns is carried
    here verbatim rather than being collapsed into a bare count.
    """

    group_id: str
    families: tuple[FragmentedFamily, ...]
    nodes_scanned: int
    complete: bool
    incomplete_kind: str | None
    rows_seen: int
    expected_rows: int | None


class TaskFamilyCensus:
    """Counts fragmented task families in a project graph. Never writes.

    Args:
        backend: A GraphitiBackend instance (must be initialized before use).
    """

    def __init__(self, *, backend):
        self.backend = backend

    async def run(self, group_id: str) -> GraphCensus:
        """Census one graph.

        Enumerates every Entity node, partitions them into task families, keeps
        only the families holding MORE THAN ONE node, and asks the backend for
        each survivor family's per-spelling edge counts.

        The >1 threshold is the honest one: a single node whose NAME is
        non-canonical ('task 800' alone) is not a fragmented family — it splits
        nothing, and the write-path normalizer renames it the next time an
        episode touches that task.

        Args:
            group_id: Project graph to census.

        Returns:
            A :class:`GraphCensus`, including the completeness of the
            underlying enumeration.
        """
        nodes, paged = await self.backend.enumerate_entity_nodes(group_id=group_id)

        families = []
        for referent, members in group_task_node_families(nodes).items():
            if len(members) <= 1:
                continue
            candidates = await self.backend.find_entity_nodes_by_name_substring(
                referent.number, group_id=group_id,
            )
            variants = tuple(
                VariantSpelling(
                    name=row['name'], uuid=row['uuid'], edge_count=row['edge_count'],
                )
                for row in group_task_node_families(candidates).get(referent, [])
            )
            family = FragmentedFamily(
                canonical_name=referent.node_name, variants=variants,
            )
            families.append(family)
            logger.warning(
                'fragmented task family %s in graph %r: %d node(s) — %s',
                family.canonical_name,
                group_id,
                len(variants),
                ', '.join(f'{v.name!r}={v.edge_count} edge(s)' for v in variants),
            )

        if not paged.complete:
            logger.error(
                'COVERAGE SHORTFALL: graph %r enumerated only %d of %s node(s) (%s); '
                'the %d fragmented family/families reported below are a LOWER BOUND, '
                'not a count',
                group_id, paged.rows_seen, paged.expected_rows, paged.incomplete_kind,
                len(families),
            )

        return GraphCensus(
            group_id=group_id,
            families=tuple(families),
            nodes_scanned=len(nodes),
            complete=paged.complete,
            incomplete_kind=paged.incomplete_kind,
            rows_seen=paged.rows_seen,
            expected_rows=paged.expected_rows,
        )
