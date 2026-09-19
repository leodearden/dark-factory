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

import argparse
import asyncio
import json
import logging
import sys
import time
from dataclasses import asdict, dataclass

from fused_memory.maintenance._utils import maintenance_service
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

    ``variants`` holds at least TWO spellings, always: it is what the family
    was measured to be, so a family the probe fails to confirm is recorded as
    an :class:`UnconfirmedFamily` instead of as a one- or zero-variant instance
    of this type. :meth:`TaskFamilyCensus.run` is the only constructor, which
    is what lets that be an invariant rather than a hope.

    It is ordered survivor-first — most valid edges, then oldest, then uuid —
    which is the order the backend returns and the order a collapse would use,
    so ``variants[0]`` reads as "the node that would survive".
    """

    canonical_name: str
    variants: tuple[VariantSpelling, ...]


@dataclass(frozen=True)
class UnconfirmedFamily:
    """A family the enumeration flagged and the per-family probe did not confirm.

    The census makes two reads, and they are neither identically scoped nor a
    single snapshot. ``enumerate_entity_nodes`` matches every ``:Entity`` node
    under a graph KEY, while ``find_entity_nodes_by_name_substring`` also
    requires ``n.group_id = $group_id`` — the predicate that keeps task-2115's
    cross-graph leak (a node whose group_id names ANOTHER project while sitting
    physically inside this key) out of any collapse. A leaked node therefore
    counts toward the enumeration's >1 threshold and then vanishes from the
    probe. The same shortfall arises with no leak at all: the store is live, so
    the write-path normalizer can collapse the family between the two reads.

    Either way the family is not residue by the time it was measured, and
    reporting it as a :class:`FragmentedFamily` of fewer than two variants
    would contradict that type's meaning and inflate the very number this
    module exists to produce. Recording it here keeps the count honest without
    discarding the signal: an occasional one is a race, while a persistent
    non-zero count is a leak worth chasing, and the two fields are what tell
    them apart.
    """

    canonical_name: str
    enumerated_members: int
    probed_members: int


@dataclass(frozen=True)
class GraphCensus:
    """What one graph's census found, and how much of that graph it saw.

    The completeness fields are not decoration. This module's entire output is
    a number a human will act on, and a number derived from a silently
    truncated enumeration is worse than no number at all — so the
    ``PagedRead`` that ``enumerate_entity_nodes`` already returns is carried
    here verbatim rather than being collapsed into a bare count.

    ``complete`` answers only "was the whole graph read", which is why
    ``unconfirmed`` sits beside it rather than flipping it: a family the probe
    did not confirm is a fully-read graph reporting a divergence between its
    two reads, not a coverage shortfall (see :class:`UnconfirmedFamily`).
    """

    group_id: str
    families: tuple[FragmentedFamily, ...]
    unconfirmed: tuple[UnconfirmedFamily, ...]
    nodes_scanned: int
    complete: bool
    incomplete_kind: str | None
    rows_seen: int
    expected_rows: int | None


@dataclass(frozen=True)
class GraphFailure:
    """A graph the sweep tried to census and could not read.

    Recorded as structure rather than folded into a log line, because the two
    fields answer different operator questions: ``group_id`` is what a re-run
    is scoped to, and ``error`` names the exception TYPE as well as its message,
    which is what says whether to retry or to go and fix something.
    """

    group_id: str
    error: str


@dataclass(frozen=True)
class CensusSweep:
    """What a sweep over one or more graphs found.

    ``total_families`` and ``complete`` are summaries of ``graphs`` and
    ``failures`` rather than independent facts. They are stored rather than
    derived on access so that ``--json`` emits them (``dataclasses.asdict``
    sees fields, not properties) — and they cannot drift from what they
    summarise because :meth:`TaskFamilyCensus.sweep` is this type's only
    constructor. ``GraphCensus.nodes_scanned`` is stored for the same reason.

    ``complete`` is the flag to read before acting on ``total_families``: it is
    False if ANY graph was truncated OR failed outright. A failed graph is an
    unknown, not a zero, and the per-graph flags cannot say that on their own —
    every graph the sweep DID read can be complete while a graph it never read
    holds the residue that matters.
    """

    graphs: tuple[GraphCensus, ...]
    failures: tuple[GraphFailure, ...]
    total_families: int
    total_unconfirmed: int
    complete: bool
    elapsed_ms: float


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

        The threshold is applied TWICE, because the enumeration decides
        membership and the probe supplies the edge counts, and the two are
        neither identically scoped nor a single snapshot. A family that thins
        below two spellings by the time it is probed is recorded as an
        :class:`UnconfirmedFamily` rather than reported as residue — see that
        class for the two ways it happens.

        The per-family probe is a second read per fragmented family rather than
        one join, because edge counts are what turn a count into a decision and
        ``enumerate_entity_nodes`` does not return them. That cost scales with
        the residue, not with the graph, so it stays proportionate while the
        residue is small.

        Args:
            group_id: Project graph to census.

        Returns:
            A :class:`GraphCensus`, including the completeness of the
            underlying enumeration.
        """
        nodes, paged = await self.backend.enumerate_entity_nodes(group_id=group_id)

        families = []
        unconfirmed = []
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
            if len(variants) < 2:
                unconfirmed.append(UnconfirmedFamily(
                    canonical_name=referent.node_name,
                    enumerated_members=len(members),
                    probed_members=len(variants),
                ))
                logger.warning(
                    'unconfirmed task family %s in graph %r: enumerated %d node(s), '
                    'probe returned %d — a group_id-leaked node (task 2115) or a '
                    'concurrent collapse; NOT counted as residue',
                    referent.node_name, group_id, len(members), len(variants),
                )
                continue
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
            unconfirmed=tuple(unconfirmed),
            nodes_scanned=len(nodes),
            complete=paged.complete,
            incomplete_kind=paged.incomplete_kind,
            rows_seen=paged.rows_seen,
            expected_rows=paged.expected_rows,
        )

    async def sweep(self, group_id: str | None = None) -> CensusSweep:
        """Census every project graph, or just one.

        Each graph is censused inside its own ``try/except Exception``, copying
        ``GraphitiBackend._run_startup_identity_scan``'s shape: one unreachable
        graph costs the sweep that graph and nothing else. The failure is
        recorded in the result, not merely logged — a count that quietly omits
        an unreachable graph reads as "this graph has no residue" when the truth
        is "this graph was never looked at".

        ``except Exception`` deliberately does not catch ``CancelledError``,
        ``KeyboardInterrupt`` or ``SystemExit``: all three are ``BaseException``,
        so an operator's Ctrl-C aborts the sweep instead of being recorded as
        one graph's data problem.

        Args:
            group_id: A single graph to census. When None, every graph
                ``backend.list_graphs()`` reports is censused. Supplying one
                skips that call entirely — scoping a census to one project
                should not turn into a read of the whole store.

        Returns:
            A :class:`CensusSweep`. Read its ``complete`` flag before acting on
            ``total_families``.
        """
        graphs = [group_id] if group_id is not None else await self.backend.list_graphs()
        start = time.monotonic()

        censused: list[GraphCensus] = []
        failures: list[GraphFailure] = []
        for graph_name in graphs:
            try:
                censused.append(await self.run(graph_name))
            except Exception as exc:
                logger.error(
                    'task-family census failed for graph %r', graph_name, exc_info=True,
                )
                failures.append(
                    GraphFailure(group_id=graph_name, error=f'{type(exc).__name__}: {exc}')
                )

        elapsed_ms = (time.monotonic() - start) * 1000
        total_families = sum(len(graph.families) for graph in censused)
        total_unconfirmed = sum(len(graph.unconfirmed) for graph in censused)
        complete = not failures and all(graph.complete for graph in censused)
        logger.info(
            'task-family census complete: graphs_censused=%d graphs_failed=%d '
            'fragmented_families=%d unconfirmed=%d complete=%s elapsed_ms=%.1f',
            len(censused), len(failures), total_families, total_unconfirmed,
            complete, elapsed_ms,
        )
        return CensusSweep(
            graphs=tuple(censused),
            failures=tuple(failures),
            total_families=total_families,
            total_unconfirmed=total_unconfirmed,
            complete=complete,
            elapsed_ms=elapsed_ms,
        )


async def run_task_family_census(
    config_path: str | None = None,
    group_id: str | None = None,
) -> CensusSweep:
    """Load config, initialize the service, sweep, close resources.

    The CLI-callable entrypoint. Lifecycle is delegated entirely to
    ``maintenance_service``, exactly as ``verify_zombie_edges.py`` does, so this
    module owns no setup or teardown of its own.

    Args:
        config_path: Optional path to the YAML config file. When given it is
            set as CONFIG_PATH before constructing FusedMemoryConfig.
        group_id: A single graph to census; None sweeps every graph.

    Returns:
        A :class:`CensusSweep`.
    """
    async with maintenance_service(config_path) as (_config, service):
        census = TaskFamilyCensus(backend=service.graphiti)
        return await census.sweep(group_id=group_id)


def _build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser.

    A factory rather than inline construction so the flag surface can be
    asserted structurally in tests. That matters more here than it usually
    would: this module's central promise is that counting the residue cannot
    change it, and a flag offering to fix what it found would break that
    promise from the outside no matter how read-only the code stayed. There is
    deliberately no ``--dry-run`` either — ``verify_zombie_edges.py`` needs one
    because it can delete, whereas every census run is already a dry run and
    offering the flag would imply an unsafe mode exists.
    """
    parser = argparse.ArgumentParser(
        description=(
            'Report task-node families that are still fragmented across several '
            'Entity nodes (task 5264). Read-only: this never changes the graph.'
        ),
    )
    parser.add_argument(
        '--config', default=None,
        help='Path to the YAML config file (overrides CONFIG_PATH env var).',
    )
    parser.add_argument(
        '--group-id', default=None,
        help='Census only this project graph. Default: every graph in the store.',
    )
    parser.add_argument(
        '--json', action='store_true',
        help='Emit the full census as JSON instead of the one-line summary.',
    )
    return parser


def _format_summary(sweep: CensusSweep) -> str:
    """The one-line human summary, with the per-graph breakdown inline."""
    breakdown = ' '.join(
        f'{graph.group_id}={len(graph.families)}' for graph in sweep.graphs
    ) or '(no graphs censused)'
    line = (
        f'task-family census: {sweep.total_families} fragmented family/families '
        f'across {len(sweep.graphs)} graph(s) [{breakdown}] '
        f'unconfirmed={sweep.total_unconfirmed} '
        f'complete={sweep.complete} elapsed_ms={sweep.elapsed_ms:.1f}'
    )
    if sweep.failures:
        failed = ', '.join(f'{f.group_id} ({f.error})' for f in sweep.failures)
        line += f'\nFAILED to census {len(sweep.failures)} graph(s): {failed}'
    return line


def main(argv: list[str] | None = None) -> int:
    """CLI entry point.

    Run as::

        python -m fused_memory.maintenance.task_family_census --json

    ``logging.basicConfig`` is called here rather than at import, so importing
    this module — from a test or from another tool — never reconfigures the
    importer's logging.

    Returns:
        0 only when every graph was read in full; 1 when any graph was
        truncated or could not be read at all. A truncated census must never
        pass for a clean one to a script reading the exit status — the number it
        reports is then a lower bound, and acting on it as a total would
        under-count the residue.

        An unconfirmed family deliberately does NOT move the exit code: the
        graph was read in full, and the count it produced is honest precisely
        because that family was excluded. It is reported in the summary and in
        the JSON, where a persistent non-zero value is the thing to chase.
    """
    args = _build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO)

    sweep = asyncio.run(
        run_task_family_census(config_path=args.config, group_id=args.group_id)
    )

    if args.json:
        print(json.dumps(asdict(sweep), indent=2, default=str))
    else:
        print(_format_summary(sweep))
    return 0 if sweep.complete else 1


if __name__ == '__main__':
    sys.exit(main())
