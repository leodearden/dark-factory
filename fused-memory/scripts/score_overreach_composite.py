#!/usr/bin/env python3
"""Sample and score the O3+O4 composite: Stage 1's decision on the judge's overreach flags (task 9040).

docs/ruling-overreach-guard-design-2026-10-01/design.md §6 pairs lever O3, the
shadow judge (services/overreach_shadow.py), with O4, Stage 1 retiring what the
judge flagged (reconciliation/overreach_flags.py). Stage 1 retires an edge with
update_edge(invalid_at=now), so a retired edge carries invalid_at and no
expired_at; graphiti's own contradiction handling stamps both. ``sample`` reads
every overreach flag recon handed over in a frozen window, classifies each by
its live edge state, counts all of them, and freezes a capped hash sample of
the retired and kept cells for blind adjudication.

Read-only: the sidecar is opened with OverreachShadowStore.open_read_only, and
every graph read is GRAPH.RO_QUERY through
fused_memory/backends/falkor_read_only.py::ReadOnlyGraphReader.

Usage (from fused-memory/)::

    uv run python scripts/score_overreach_composite.py sample \\
        --sidecar <queue.data_dir>/overreach_shadow.db \\
        --graph dark_factory --graph reify \\
        --window-start 2026-10-20T00:00:00+00:00 --window-end 2026-11-10T00:00:00+00:00 \\
        --out-sample ../docs/ruling-overreach-lever-reopen-2026-11-10/composite-sample.json \\
        --out-worksheet /tmp/9040-composite-worksheet.jsonl

The sample file carries neither episode bodies nor judge rationales; the
worksheet carries bodies and facts but neither the decision nor the rationale,
so it is not committed.

Exit codes: 0 ran; 1 the sidecar or a graph is unavailable; 2 an input is
invalid. Nothing is written on error.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sqlite3
import sys
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from fused_memory.backends.falkor_read_only import ReadOnlyGraphReader
from fused_memory.services.overreach_shadow import HandedFlag, OverreachShadowStore

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from audit_ruling_overreach import SampleDefinition, _resolve_uri, sample_key

logger = logging.getLogger('score_overreach_composite')

# --------------------------------------------------------------------------- #
# What Stage 1 did with a flag, read from the edge as it stands
# --------------------------------------------------------------------------- #

RETIRED, KEPT, SUPERSEDED, GONE = 'retired', 'kept', 'superseded', 'gone'
DECISIONS: tuple[str, ...] = (RETIRED, KEPT, SUPERSEDED, GONE)

SAMPLED_DECISIONS: tuple[str, ...] = (RETIRED, KEPT)
"""The cells, per graph. A superseded or gone flag is counted and never sampled."""

DEFAULT_CAP = 20
HASH_RULE = 'sha256(graph:edge_uuid)'

UUID_BATCH = 500
"""Uuids per read, so no answer comes near the server's silent result-set cap."""

EDGE_STATE_CYPHER = (
    'MATCH (s)-[e:RELATES_TO]->(t) WHERE e.uuid IN $uuids '
    'RETURN e.uuid, e.invalid_at, e.expired_at, s.name, t.name'
)

EPISODE_CONTENT_CYPHER = 'MATCH (e:Episodic) WHERE e.uuid IN $uuids RETURN e.uuid, e.content'


@dataclass(frozen=True)
class EdgeState:
    invalid_at: str | None
    expired_at: str | None
    source_name: str
    target_name: str


def decision_of(state: EdgeState | None) -> str:
    if state is None:
        return GONE
    if state.invalid_at is None:
        return KEPT
    return RETIRED if state.expired_at is None else SUPERSEDED


def _text(value: Any) -> str | None:
    return None if value is None else str(value)


async def _read_batched(reader: ReadOnlyGraphReader, cypher: str, uuids: Iterable[str]) -> list[list]:
    unique = sorted(set(uuids))
    rows: list[list] = []
    for start in range(0, len(unique), UUID_BATCH):
        rows += await reader.read_unpaged(cypher, {'uuids': unique[start:start + UUID_BATCH]})
    return rows


async def read_edge_states(reader: ReadOnlyGraphReader, uuids: Iterable[str]) -> dict[str, EdgeState]:
    """The state of each edge the graph still holds; an absent uuid is a gone edge."""
    return {
        str(uuid): EdgeState(_text(invalid_at), _text(expired_at), _text(source) or '', _text(target) or '')
        for uuid, invalid_at, expired_at, source, target
        in await _read_batched(reader, EDGE_STATE_CYPHER, uuids)
    }


async def read_episode_contents(reader: ReadOnlyGraphReader, uuids: Iterable[str]) -> dict[str, str | None]:
    return {
        str(uuid): _text(content)
        for uuid, content in await _read_batched(reader, EPISODE_CONTENT_CYPHER, uuids)
    }


# --------------------------------------------------------------------------- #
# The sample: every handed flag counted, the retired and kept cells capped
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class ClassifiedFlag:
    graph: str
    handed: HandedFlag
    state: EdgeState | None

    @property
    def edge_uuid(self) -> str:
        return self.handed.flag.edge_uuid

    @property
    def episode_uuid(self) -> str:
        return self.handed.flag.episode_uuid

    @property
    def decision(self) -> str:
        return decision_of(self.state)

    def hash_key(self) -> str:
        return sample_key(self.graph, self.edge_uuid)

    def sample_entry(self) -> dict[str, str]:
        """What the frozen sample keeps of a flag: never the judge's rationale."""
        return {
            'graph': self.graph, 'edge_uuid': self.edge_uuid, 'episode_uuid': self.episode_uuid,
            'fact': self.handed.flag.fact, 'decision': self.decision,
            'handed_at': self.handed.handed_at.isoformat(),
        }

    def worksheet_edge(self) -> dict[str, str]:
        """What the adjudicator sees of a flag: never its decision or the judge's rationale."""
        source, target = (self.state.source_name, self.state.target_name) if self.state else ('', '')
        return {'edge_uuid': self.edge_uuid, 'fact': self.handed.flag.fact,
                'source_name': source, 'target_name': target}


def select_flags(flags: Iterable[ClassifiedFlag], cap: int) -> tuple[ClassifiedFlag, ...]:
    """One graph's sample: the first *cap* by hash of each sampled decision, cell by cell."""
    cells: defaultdict[str, list[ClassifiedFlag]] = defaultdict(list)
    for flag in flags:
        cells[flag.decision].append(flag)
    return tuple(
        member for decision in SAMPLED_DECISIONS
        for member in sorted(cells[decision], key=ClassifiedFlag.hash_key)[:cap]
    )


@dataclass(frozen=True)
class GraphSample:
    graph: str
    flags: tuple[ClassifiedFlag, ...]
    """Every overreach flag handed over in the window."""
    sampled: tuple[ClassifiedFlag, ...]
    contents: Mapping[str, str | None]
    """The sampled flags' episode bodies, by episode uuid; None if the graph no longer holds one."""


async def sample_graph(
    store: OverreachShadowStore, reader: ReadOnlyGraphReader, graph: str,
    definition: SampleDefinition, window: tuple[datetime, datetime],
) -> GraphSample:
    handed = await store.handed_overreach_flags(graph, handed_from=window[0], handed_until=window[1])
    states = await read_edge_states(reader, (h.flag.edge_uuid for h in handed))
    flags = tuple(ClassifiedFlag(graph, h, states.get(h.flag.edge_uuid)) for h in handed)
    sampled = select_flags(flags, definition.cap)
    contents = await read_episode_contents(reader, (flag.episode_uuid for flag in sampled))
    return GraphSample(graph=graph, flags=flags, sampled=sampled, contents=contents)


def window_days(definition: SampleDefinition) -> float:
    span = datetime.fromisoformat(definition.window_end) - datetime.fromisoformat(definition.window_start)
    return span.total_seconds() / 86400


def census(samples: Sequence[GraphSample], days: float) -> dict[str, Any]:
    """Per graph, every handed flag by decision, as a count and per day over the window."""
    by_graph: dict[str, Any] = {}
    for sample in samples:
        counts = Counter(flag.decision for flag in sample.flags)
        by_graph[sample.graph] = {'handed': len(sample.flags)} | {
            decision: {'flags': counts[decision], 'per_day': round(counts[decision] / days, 2)}
            for decision in DECISIONS
        }
    return {
        'window_days': days,
        'by_graph': by_graph,
        'prompt_versions': sorted({f.handed.flag.prompt_version for s in samples for f in s.flags}),
    }


def sample_document(samples: Sequence[GraphSample], definition: SampleDefinition, sampled_at: str) -> dict[str, Any]:
    return {
        'sample': definition.to_dict(),
        'graphs': [sample.graph for sample in samples],
        'sampled_at': sampled_at,
        'census': census(samples, window_days(definition)),
        'flags': [flag.sample_entry() for sample in samples for flag in sample.sampled],
    }


def worksheet_rows(samples: Sequence[GraphSample]) -> list[dict[str, Any]]:
    """One row per sampled episode, rows and edges in hash order, so no cell shows through."""
    by_episode: defaultdict[tuple[str, str], list[ClassifiedFlag]] = defaultdict(list)
    contents: dict[tuple[str, str], str | None] = {}
    for sample in samples:
        for flag in sample.sampled:
            by_episode[(sample.graph, flag.episode_uuid)].append(flag)
            contents[(sample.graph, flag.episode_uuid)] = sample.contents.get(flag.episode_uuid)
    return [
        {'graph': graph, 'uuid': uuid, 'content': contents[(graph, uuid)],
         'edges': [flag.worksheet_edge() for flag in sorted(by_episode[(graph, uuid)], key=ClassifiedFlag.hash_key)]}
        for graph, uuid in sorted(by_episode, key=lambda episode: sample_key(*episode))
    ]


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

EXIT_OK, EXIT_UNAVAILABLE, EXIT_INVALID = 0, 1, 2

ReaderFactory = Callable[[str], ReadOnlyGraphReader]


class InvalidInput(ValueError):
    """An argument or input file the composite must not be sampled or scored from."""


def _aware_instant(text: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError as exc:
        raise InvalidInput(f'{text!r} is not an ISO instant') from exc
    if parsed.tzinfo is None:
        raise InvalidInput(f'{text!r} names no UTC offset')
    return parsed


def handoff_window(start: str, end: str, cap: int) -> tuple[SampleDefinition, tuple[datetime, datetime]]:
    window = (_aware_instant(start), _aware_instant(end))
    if window[0] >= window[1]:
        raise InvalidInput(f'the window [{start}, {end}) is empty')
    if cap < 1:
        raise InvalidInput(f'a cap of {cap} samples nothing')
    definition = SampleDefinition(
        window_start=start, window_end=end, cap=cap, strata=SAMPLED_DECISIONS, hash_rule=HASH_RULE,
    )
    return definition, window


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog='score_overreach_composite',
        description="Read-only sample and score of Stage 1's decision on handed overreach flags (task 9040).",
    )
    commands = parser.add_subparsers(dest='command', required=True)
    sample = commands.add_parser('sample', help='Census the handed flags and freeze the capped sample.')
    sample.add_argument('--sidecar', required=True, help='<queue.data_dir>/overreach_shadow.db, opened read-only.')
    sample.add_argument('--graph', action='append', required=True, metavar='GRAPH', help='Repeatable.')
    sample.add_argument('--window-start', required=True, help='Handoff window start, inclusive, with offset.')
    sample.add_argument('--window-end', required=True, help='Handoff window end, exclusive, with offset.')
    sample.add_argument('--cap', type=int, default=DEFAULT_CAP, help='Flags per graph per sampled decision.')
    sample.add_argument('--out-sample', required=True, help='The frozen composite-sample.json.')
    sample.add_argument('--out-worksheet', required=True, help='The blind adjudication worksheet (JSONL).')
    sample.add_argument('--graph-uri', default=None, help='FalkorDB URI. Default: env FALKORDB_URI, then config.')
    sample.add_argument('--graph-host', default=None, help='FalkorDB host, instead of a URI.')
    sample.add_argument('--graph-port', type=int, default=None, help='FalkorDB port, with --graph-host.')
    return parser


def _live_reader_factory(args: argparse.Namespace) -> ReaderFactory:
    if args.graph_host is not None or args.graph_port is not None:
        return lambda graph: ReadOnlyGraphReader(graph_name=graph, host=args.graph_host, port=args.graph_port)
    uri = _resolve_uri(args)
    return lambda graph: ReadOnlyGraphReader(graph_name=graph, uri=uri)


def _write_together(texts: Mapping[Path, str]) -> None:
    """Stage every file before any replaces its target, so a failed write leaves none half-done."""
    staged = {}
    for path, text in texts.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        staged[path] = path.with_name(path.name + '.tmp')
        staged[path].write_text(text)
    for path, tmp in staged.items():
        tmp.replace(path)


async def _read_samples(
    args: argparse.Namespace, definition: SampleDefinition, window: tuple[datetime, datetime],
    reader_factory: ReaderFactory,
) -> list[GraphSample]:
    store = await OverreachShadowStore.open_read_only(args.sidecar)
    readers = {graph: reader_factory(graph) for graph in dict.fromkeys(args.graph)}
    try:
        return [await sample_graph(store, reader, graph, definition, window) for graph, reader in readers.items()]
    finally:
        await store.close()
        for reader in readers.values():
            await reader.aclose()


async def _sample_command(args: argparse.Namespace, reader_factory: ReaderFactory) -> int:
    try:
        definition, window = handoff_window(args.window_start, args.window_end, args.cap)
    except InvalidInput as exc:
        logger.error('invalid inputs, nothing is written: %s', exc)
        return EXIT_INVALID
    try:
        samples = await _read_samples(args, definition, window, reader_factory)
    except (OSError, sqlite3.Error) as exc:
        logger.error('the sidecar is unavailable, nothing is written: %s', exc)
        return EXIT_UNAVAILABLE
    except Exception:
        logger.error('a graph read failed, nothing is written', exc_info=True)
        return EXIT_UNAVAILABLE
    document = sample_document(samples, definition, datetime.now(UTC).isoformat())
    _write_together({
        Path(args.out_sample): json.dumps(document, indent=2, sort_keys=True) + '\n',
        Path(args.out_worksheet): ''.join(json.dumps(row, sort_keys=True) + '\n' for row in worksheet_rows(samples)),
    })
    return EXIT_OK


async def _run(args: argparse.Namespace, reader_factory: ReaderFactory | None) -> int:
    return await _sample_command(args, reader_factory or _live_reader_factory(args))


def run(argv: Sequence[str] | None = None, *, reader_factory: ReaderFactory | None = None) -> int:
    """Parse *argv* (default ``sys.argv[1:]``) and run one subcommand.

    *reader_factory* maps a graph name to a ReadOnlyGraphReader (default:
    FalkorDB over GRAPH.RO_QUERY).
    """
    return asyncio.run(_run(_build_parser().parse_args(argv), reader_factory))


def main() -> int:
    logging.basicConfig(level=logging.INFO, stream=sys.stderr, format='%(levelname)s %(name)s: %(message)s')
    return run()


if __name__ == '__main__':
    sys.exit(main())
