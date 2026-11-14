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

``score`` validates the verdicts on every sampled flag with the audit's own
load_verdicts, weights each (graph, decision) cell by its census population
over its sampled flags, and rates Stage 1's retirements with the audit's
weighted_rate against the §6.1 precision bars imported from
eval_overreach_judge.py::

    uv run python scripts/score_overreach_composite.py score \\
        --sample ../docs/ruling-overreach-lever-reopen-2026-11-10/composite-sample.json \\
        --verdicts ../docs/ruling-overreach-lever-reopen-2026-11-10/composite-verdicts.json \\
        --judge-eval ../docs/ruling-overreach-guard-design-2026-10-01/judge-eval-offline-2026-11-09.json \\
        --out ../docs/ruling-overreach-lever-reopen-2026-11-10/composite-report.json

Exit codes: 0 ran; 1 the sidecar or a graph is unavailable; 2 an input is
invalid. Nothing is written on error.
"""
from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import logging
import os
import sqlite3
import sys
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any

from fused_memory.backends.falkor_read_only import ReadOnlyGraphReader
from fused_memory.services.overreach_guard import LABELS, OVERREACH
from fused_memory.services.overreach_shadow import HandedFlag, OverreachShadowStore

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _graph_script_cli import (
    GRAPH_UNAVAILABLE,
    ReaderFactory,
    UnreadableInput,
    add_graph_arguments,
    live_reader_factory,
    read_json,
    write_files_atomically,
)
from audit_ruling_overreach import (
    SampleDefinition,
    VerdictError,
    VerdictSet,
    load_verdicts,
    rate,
    sample_key,
    weighted_rate,
)
from eval_overreach_judge import PRECISION_LOWER_MIN, PRECISION_MIN

logger = logging.getLogger('score_overreach_composite')

# --------------------------------------------------------------------------- #
# What Stage 1 did with a flag, read from the edge as it stands
# --------------------------------------------------------------------------- #


class Decision(StrEnum):
    retired = 'retired'
    kept = 'kept'
    superseded = 'superseded'
    gone = 'gone'


SAMPLED_DECISIONS: tuple[Decision, ...] = (Decision.retired, Decision.kept)
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


def decision_of(state: EdgeState | None) -> Decision:
    if state is None:
        return Decision.gone
    if state.invalid_at is None:
        return Decision.kept
    return Decision.retired if state.expired_at is None else Decision.superseded


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
    def decision(self) -> Decision:
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
    cells: defaultdict[Decision, list[ClassifiedFlag]] = defaultdict(list)
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
            for decision in Decision
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
# The score: Stage 1's decision against the adjudicated sample
# --------------------------------------------------------------------------- #

HOLDING = 'holding'

COMPOSITE_RECALL_LABEL = (
    'product of two frames: the offline judge recall on the 2026-08-25..2026-10-01 adjudicated '
    "window times Stage-1 recall on the flags handed in this sample's window; the range multiplies "
    'their bounds and is not the interval of any one estimator'
)

Cell = tuple[str, Decision]
"""``(graph, decision)``."""


def _cell(flag: Mapping[str, str]) -> Cell:
    """A sampled flag's cell; ValueError names a decision that is not one."""
    return flag['graph'], Decision(flag['decision'])


class InvalidInput(ValueError):
    """An argument or input file the composite must not be sampled or scored from."""


@dataclass(frozen=True)
class CompositeSample:
    """A composite-sample.json, as the score reads it."""

    definition: SampleDefinition
    window_days: float
    population: Mapping[Cell, int]
    """Every handed flag in each sampled cell, from the census."""
    flags: tuple[Mapping[str, str], ...]

    @classmethod
    def from_document(cls, document: Mapping[str, Any]) -> CompositeSample:
        try:
            by_graph = document['census']['by_graph']
            sample = cls(
                definition=SampleDefinition.from_dict(document['sample']),
                window_days=float(document['census']['window_days']),
                population={(graph, decision): int(by_graph[graph][decision]['flags'])
                            for graph in by_graph for decision in SAMPLED_DECISIONS},
                flags=tuple(document['flags']),
            )
            sampled = Counter(_cell(flag) for flag in sample.flags)
        except (KeyError, TypeError, ValueError) as exc:
            raise InvalidInput(f'not a composite sample: {exc!r}') from exc
        if over := sorted(cell for cell, n in sampled.items() if n > sample.population.get(cell, 0)):
            raise InvalidInput(f'cells sampled beyond their census population: {over}')
        return sample


def validate_verdicts(obj: Any, sample: CompositeSample) -> VerdictSet:
    """load_verdicts against the frozen sample: every sampled flag judged once, and nothing else.
    No sampled episode is offered for staleness, so any other verdict is out of sample."""
    return load_verdicts(
        obj, expected_edges={(flag['graph'], flag['edge_uuid']): flag['episode_uuid'] for flag in sample.flags},
        sampled_episodes=(), existing_edges=(), definition=sample.definition,
    )


@dataclass(frozen=True)
class WeightedCells:
    """Each sampled cell's adjudicated labels, and the weight each of its flags carries:
    ``census population / sampled``."""

    counts: Mapping[Cell, Counter[str]]
    weights: Mapping[Cell, float]
    window_days: float

    @classmethod
    def of(cls, sample: CompositeSample, verdicts: VerdictSet) -> WeightedCells:
        labels = verdicts.by_edge
        counts: defaultdict[Cell, Counter[str]] = defaultdict(Counter)
        for flag in sample.flags:
            counts[_cell(flag)][labels[(flag['graph'], flag['edge_uuid'])]] += 1
        return cls(
            counts=dict(counts),
            weights={cell: sample.population[cell] / counter.total() for cell, counter in counts.items()},
            window_days=sample.window_days,
        )

    def cells(self, decision: Decision | None = None) -> list[Cell]:
        return [cell for cell in self.weights if decision in (None, cell[1])]

    def share(self, label: str, decision: Decision | None = None) -> dict[str, Any]:
        """The weighted share of *decision*'s flags (every flag, by default) adjudicated *label*."""
        return weighted_rate(
            (self.weights[cell], self.counts[cell][label], self.counts[cell].total()) for cell in self.cells(decision)
        )

    def retired_share(self, label: str) -> dict[str, Any]:
        """Of the flags adjudicated *label*, the weighted share Stage 1 retired."""
        return weighted_rate(
            (self.weights[cell], self.counts[cell][label] if cell[1] is Decision.retired else 0,
             self.counts[cell][label])
            for cell in self.cells()
        )

    def retired_per_day(self, label: str) -> float:
        weighted = sum(self.weights[cell] * self.counts[cell][label] for cell in self.cells(Decision.retired))
        return round(weighted / self.window_days, 2)

    def unweighted_share(self, label: str, decision: Decision | None = None) -> dict[str, Any]:
        cells = self.cells(decision)
        return rate(sum(self.counts[cell][label] for cell in cells), sum(self.counts[cell].total() for cell in cells))


def _by_graph(values: Mapping[Cell, Any]) -> dict[str, dict[str, Any]]:
    nested: dict[str, dict[str, Any]] = {}
    for (graph, decision), value in values.items():
        nested.setdefault(graph, {})[decision] = value
    return nested


def _at_least(value: float | None, floor: float) -> bool:
    return value is not None and value >= floor


def stage1_decision(retire_precision: Mapping[str, Any]) -> dict[str, Any]:
    """The 2026-10-01 design.md §6.1 precision bars, applied to what Stage 1 retired."""
    checks = {
        'retire_precision': _at_least(retire_precision['rate'], PRECISION_MIN),
        'retire_precision_lower_bound': _at_least((retire_precision['ci'] or [None])[0], PRECISION_LOWER_MIN),
    }
    return {
        'bars': {'precision_min': PRECISION_MIN, 'precision_lower_min': PRECISION_LOWER_MIN},
        'checks': checks, 'passes': all(checks.values()),
    }


def _product(one: float | None, other: float | None) -> float | None:
    return None if one is None or other is None else round(one * other, 4)


def composite_recall(judge_recall: Mapping[str, Any], stage1_recall: Mapping[str, Any]) -> dict[str, Any]:
    judge_ci, stage1_ci = judge_recall['ci'] or (None, None), stage1_recall['ci'] or (None, None)
    return {
        'point': _product(judge_recall['rate'], stage1_recall['rate']),
        'range': [_product(judge_ci[0], stage1_ci[0]), _product(judge_ci[1], stage1_ci[1])],
        'judge_recall': {'rate': judge_recall['rate'], 'ci': judge_recall['ci']},
        'label': COMPOSITE_RECALL_LABEL,
    }


def composite_score(
    sample: CompositeSample, verdicts: VerdictSet, judge_recall: Mapping[str, Any] | None,
) -> dict[str, Any]:
    cells = WeightedCells.of(sample, verdicts)
    retire_precision = cells.share(OVERREACH, Decision.retired)
    stage1_recall = cells.retired_share(OVERREACH)
    return {
        'sample': sample.definition.to_dict(),
        'verdicts': len(verdicts.verdicts),
        'label_counts': _by_graph({
            cell: {label: counter[label] for label in LABELS} for cell, counter in cells.counts.items()
        }),
        'weights': _by_graph({cell: round(weight, 4) for cell, weight in cells.weights.items()}),
        'retire_precision': retire_precision,
        'flag_precision': cells.share(OVERREACH),
        'stage1_recall_on_flags': stage1_recall,
        'holding_retired_share': cells.share(HOLDING, Decision.retired),
        'holdings_retired_per_day': cells.retired_per_day(HOLDING),
        'overreach_retired_per_day': cells.retired_per_day(OVERREACH),
        'unweighted': {
            'retire_precision': cells.unweighted_share(OVERREACH, Decision.retired),
            'flag_precision': cells.unweighted_share(OVERREACH),
        },
        'composite_recall': composite_recall(judge_recall, stage1_recall) if judge_recall else None,
        'decision': stage1_decision(retire_precision),
    }


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

EXIT_OK, EXIT_UNAVAILABLE, EXIT_INVALID = 0, 1, 2


class SidecarUnavailable(Exception):
    """The sidecar cannot be opened read-only."""


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
    add_graph_arguments(sample)
    score = commands.add_parser('score', help="Rate Stage 1's decision against the adjudicated sample.")
    score.add_argument('--sample', required=True, help='The frozen composite-sample.json.')
    score.add_argument('--verdicts', required=True, help='Verdicts on every sampled flag, in load_verdicts format.')
    score.add_argument('--judge-eval', default=None,
                       help="An eval_overreach_judge.py offline result; its recall makes the composite's.")
    score.add_argument('--out', required=True, help='composite-report.json to write.')
    return parser


async def _read_samples(
    args: argparse.Namespace, definition: SampleDefinition, window: tuple[datetime, datetime],
    reader_factory: ReaderFactory,
) -> list[GraphSample]:
    async with contextlib.AsyncExitStack() as opened:
        try:
            store = await OverreachShadowStore.open_read_only(args.sidecar)
        except (OSError, sqlite3.Error) as exc:
            raise SidecarUnavailable(str(exc)) from exc
        opened.push_async_callback(store.close)
        readers: dict[str, ReadOnlyGraphReader] = {}
        for graph in dict.fromkeys(args.graph):
            readers[graph] = reader_factory(graph)
            opened.push_async_callback(readers[graph].aclose)
        return [await sample_graph(store, reader, graph, definition, window) for graph, reader in readers.items()]


async def _sample_command(args: argparse.Namespace, reader_factory: ReaderFactory) -> int:
    try:
        definition, window = handoff_window(args.window_start, args.window_end, args.cap)
    except InvalidInput as exc:
        logger.error('invalid inputs, nothing is written: %s', exc)
        return EXIT_INVALID
    try:
        samples = await _read_samples(args, definition, window, reader_factory)
    except (SidecarUnavailable, sqlite3.Error) as exc:
        logger.error('the sidecar is unavailable, nothing is written: %s', exc)
        return EXIT_UNAVAILABLE
    except GRAPH_UNAVAILABLE:
        logger.error('a graph read failed, nothing is written', exc_info=True)
        return EXIT_UNAVAILABLE
    document = sample_document(samples, definition, datetime.now(UTC).isoformat())
    write_files_atomically({
        Path(args.out_sample): json.dumps(document, indent=2, sort_keys=True) + '\n',
        Path(args.out_worksheet): ''.join(json.dumps(row, sort_keys=True) + '\n' for row in worksheet_rows(samples)),
    })
    return EXIT_OK


def _judge_recall(path: str | None) -> Mapping[str, Any] | None:
    if path is None:
        return None
    try:
        recall = read_json(path)['recall']
        return {'rate': recall['rate'], 'ci': recall['ci']}
    except (KeyError, TypeError) as exc:
        raise InvalidInput(f'{path} carries no judge recall: {exc!r}') from exc


def _score_command(args: argparse.Namespace) -> int:
    try:
        sample = CompositeSample.from_document(read_json(args.sample))
        report = composite_score(
            sample, validate_verdicts(read_json(args.verdicts), sample), _judge_recall(args.judge_eval),
        )
    except (InvalidInput, UnreadableInput, VerdictError) as exc:
        logger.error('invalid inputs, nothing is written (%s): %s', type(exc).__name__, exc)
        return EXIT_INVALID
    write_files_atomically({Path(args.out): json.dumps(report, indent=2, sort_keys=True) + '\n'})
    return EXIT_OK


async def _run(args: argparse.Namespace, reader_factory: ReaderFactory | None) -> int:
    if args.command == 'score':
        return _score_command(args)
    return await _sample_command(args, reader_factory or live_reader_factory(args))


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
