#!/usr/bin/env python3
"""Replay ruling-class extraction under an arm, to measure lever O1 (task 9040).

docs/ruling-overreach-guard-design-2026-10-01/design.md §6 names lever O1:
``custom_extraction_instructions`` for the ``decisions_and_rationale`` class.
§10.3 re-opened the lever choice with O1 to be measured by replaying the 48
frozen-window episodes. This script is that replay. Each episode is
re-extracted under each arm by graphiti_core 0.28.2's own
utils/maintenance/node_operations.py::extract_nodes and
utils/maintenance/edge_operations.py::extract_edges, unmodified, followed by
graphiti's in-batch exact-fact dedupe (the driver-free fast path of
edge_operations.py::resolve_extracted_edges). The arms differ only in the
instructions they splice into those prompts; the model is a separate axis.

Read-only and driver-free: graphiti is handed a GraphitiClients that holds
only an LLM client, so no FalkorDriver is ever built (see
fused_memory/backends/falkor_read_only.py for why that matters). Episode
inputs are read over GRAPH.RO_QUERY only. Nothing is written to any graph;
the replay file records edges, entity names and tokens, never episode text.

Usage (from fused-memory/)::

    uv run python scripts/replay_overreach_extraction.py replay \\
        --worksheet /tmp/9040-worksheet-4716.jsonl \\
        --baseline-report ../docs/ruling-overreach-guard-design-2026-10-01/report.json \\
        --arm control --arm o1 --out /tmp/9040-replay.jsonl

    uv run python scripts/replay_overreach_extraction.py worksheet \\
        --replay /tmp/9040-replay.jsonl --worksheet /tmp/9040-worksheet-4716.jsonl \\
        --out /tmp/9040-blind-worksheet.jsonl

``replay`` resumes an existing --out: recorded (arm, episode) pairs are
skipped and failure lines are retried. ``--model`` replays the same arms on
another extraction model; give it its own --out. ``worksheet`` interleaves
every arm's edges by sha256(edge_uuid) and names no arm, for blind
adjudication; it carries episode bodies, so it is not committed.

Exit codes: 0 ran; 1 the LLM or graph is unavailable, or an episode failed
(re-run to resume); 2 an input is invalid (nothing is written).
"""
from __future__ import annotations

import argparse
import asyncio
import dataclasses
import hashlib
import json
import logging
import os
import sys
from collections import Counter
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from types import MappingProxyType
from typing import Any

from graphiti_core.edges import EntityEdge
from graphiti_core.graphiti_types import GraphitiClients
from graphiti_core.llm_client.client import LLMClient
from graphiti_core.nodes import EpisodeType, EpisodicNode
from graphiti_core.search.search_utils import RELEVANT_SCHEMA_LIMIT
from graphiti_core.utils.maintenance.dedup_helpers import _normalize_string_exact
from graphiti_core.utils.maintenance.edge_operations import extract_edges
from graphiti_core.utils.maintenance.node_operations import extract_nodes

from fused_memory.backends.falkor_read_only import ReadOnlyGraphReader
from fused_memory.backends.falkor_string_bounds import exact_string_upper_bound
from fused_memory.backends.graphiti_client import build_llm_client
from fused_memory.backends.llm_token_usage import (
    LlmTokenUsage,
    TokenMeasurement,
    measure_llm_tokens,
)
from fused_memory.config.schema import FusedMemoryConfig

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from audit_ruling_overreach import _resolve_uri

logger = logging.getLogger('replay_overreach_extraction')

# --------------------------------------------------------------------------- #
# The arms, pre-registered (design §2.1)
# --------------------------------------------------------------------------- #

O1_INSTRUCTIONS = """\
This is a DECISION RECORD. Extract only what it states, at the scope it states it.
1. Keep the record's scope and qualifiers. Never generalise a decision past the cases, components, actors or conditions the record names, and never drop a qualifier (who, which, when, under what condition) that limits it.
2. Add no implication or consequence the record does not state. Do not link two things with a vague verb such as "impacts", "affects", "relates to" or "implies" unless the record itself links them.
3. An option the record weighed, rejected, deferred or only proposed is never adopted. State it only together with its status (for example "X was rejected in favour of Y"), never as current fact.
4. Preserve direction exactly: who or what acts on what, what calls, imports or depends on what, what is above or below, before or after.
5. Place each symbol, file, function, value or measurement with the entity the record places it with. Never move it to a neighbouring module, component or task.
6. PREVIOUS_MESSAGES only help resolve references. Never attribute their content to this record.
"""

O1_INSTRUCTIONS_VERSION = 'o1-v1-' + hashlib.sha256(O1_INSTRUCTIONS.encode()).hexdigest()[:12]

NO_INSTRUCTIONS = 'none'


@dataclass(frozen=True)
class ReplayArm:
    """What one arm splices into graphiti's extraction prompts; ``model`` None means the configured one."""

    name: str
    instructions: str | None
    instructions_version: str
    model: str | None = None


ARMS: Mapping[str, ReplayArm] = MappingProxyType({
    'control': ReplayArm('control', None, NO_INSTRUCTIONS),
    'o1': ReplayArm('o1', O1_INSTRUCTIONS, O1_INSTRUCTIONS_VERSION),
})

EDGE_TYPE_MAP: Mapping[tuple[str, str], list[str]] = MappingProxyType({('Entity', 'Entity'): []})
"""graphiti_core's graphiti.py::Graphiti.add_episode default when no edge_types are passed, as fused-memory passes none."""

# --------------------------------------------------------------------------- #
# One episode, replayed
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class ReplayInput:
    """An episode as production extracted it: its own fields and its previous-episode contents, oldest first."""

    graph: str
    uuid: str
    stratum: str
    content: str
    source: str
    source_description: str
    valid_at: datetime
    previous: tuple[str, ...]


@dataclass(frozen=True)
class MintedEdge:
    edge_uuid: str
    fact: str
    source_name: str
    target_name: str


@dataclass(frozen=True)
class ReplayResult:
    arm: str
    model: str | None
    instructions_version: str
    graph: str
    uuid: str
    stratum: str
    entities: tuple[str, ...]
    minted: tuple[MintedEdge, ...]
    tokens: Mapping[str, LlmTokenUsage]
    """``extract_nodes`` and ``extract_edges``: each stage's measured usage, retries included."""


class UnmeasurableClient(TypeError):
    """The LLM client carries no AttributingTokenUsageTracker, so its tokens cannot be measured."""


def build_arm_llm(cfg: FusedMemoryConfig, model: str | None) -> LLMClient | None:
    """The configured extraction client, on *model* when given; *cfg* itself is never modified."""
    arm_cfg = cfg.model_copy(deep=True)
    if model is not None:
        arm_cfg.llm.model = model
    return build_llm_client(arm_cfg)


def in_batch_dedupe(edges: Iterable[EntityEdge]) -> list[EntityEdge]:
    """The first edge per (source, target, exactly-normalised fact), as resolve_extracted_edges' fast path keeps it."""
    kept: dict[tuple[str, str, str], EntityEdge] = {}
    for edge in edges:
        kept.setdefault(
            (edge.source_node_uuid, edge.target_node_uuid, _normalize_string_exact(edge.fact)), edge,
        )
    return list(kept.values())


def _episodic_node(episode: ReplayInput, *, uuid: str, content: str) -> EpisodicNode:
    return EpisodicNode(
        uuid=uuid, name=uuid, group_id=episode.graph, labels=[],
        source=EpisodeType(episode.source), content=content,
        source_description=episode.source_description,
        created_at=episode.valid_at, valid_at=episode.valid_at,
    )


def _usage(measurement: TokenMeasurement) -> LlmTokenUsage:
    if measurement.usage is None:
        raise UnmeasurableClient('the replay needs a client built by build_arm_llm, whose tokens are attributable')
    return measurement.usage


async def replay_episode(llm: LLMClient, episode: ReplayInput, arm: ReplayArm) -> ReplayResult:
    """Extract *episode* under *arm* through graphiti's own functions; no driver exists to reach."""
    clients = GraphitiClients.model_construct(llm_client=llm)
    current = _episodic_node(episode, uuid=episode.uuid, content=episode.content)
    previous = [
        _episodic_node(episode, uuid=f'{episode.uuid}:previous:{index}', content=content)
        for index, content in enumerate(episode.previous)
    ]
    async with measure_llm_tokens(llm) as node_tokens:
        nodes = await extract_nodes(
            clients, current, previous, entity_types=None,
            custom_extraction_instructions=arm.instructions,
        )
    async with measure_llm_tokens(llm) as edge_tokens:
        edges = await extract_edges(
            clients, current, nodes, previous, dict(EDGE_TYPE_MAP), group_id=episode.graph,
            custom_extraction_instructions=arm.instructions,
        )
    names = {node.uuid: node.name for node in nodes}
    return ReplayResult(
        arm=arm.name, model=llm.model, instructions_version=arm.instructions_version,
        graph=episode.graph, uuid=episode.uuid, stratum=episode.stratum,
        entities=tuple(node.name for node in nodes),
        minted=tuple(
            MintedEdge(edge.uuid, edge.fact, names[edge.source_node_uuid], names[edge.target_node_uuid])
            for edge in in_batch_dedupe(edges)
        ),
        tokens=MappingProxyType({'extract_nodes': _usage(node_tokens), 'extract_edges': _usage(edge_tokens)}),
    )


# --------------------------------------------------------------------------- #
# The read side: each episode's production inputs, over GRAPH.RO_QUERY only
# --------------------------------------------------------------------------- #

EPISODE_CYPHER = (
    'MATCH (e:Episodic {uuid: $uuid}) '
    'RETURN e.valid_at, e.source, e.source_description, e.group_id'
)

PREVIOUS_EPISODES_CYPHER = (
    'MATCH (e:Episodic) '
    f"WHERE {exact_string_upper_bound('e.valid_at', 'reference_time', inclusive=True)} "
    'AND e.group_id IN $group_ids AND e.source = $source AND e.uuid <> $uuid '
    'RETURN e.content, e.valid_at ORDER BY e.valid_at DESC LIMIT $num_episodes'
)
"""graphiti_core 0.28.2's utils/maintenance/graph_data_operations.py::retrieve_episodes, as
Graphiti.add_episode calls it, less the episode itself. Its bare ``e.valid_at <= $reference_time``
is served by FalkorDB's RANGE index on Episodic.valid_at, which over-returns on a string ``<=``:
measured 2026-11-10 it matched all 11042 dark_factory episodes for a 2026-09-17 bound. That is
harmless at write time, when nothing newer exists, and wrong for a replay, so the bound here is
exact_string_upper_bound's. On the 48 frozen-window episodes it selects the same 10 episodes as
'created before this one' does."""


class EpisodeMissing(LookupError):
    """A worksheet episode the graph no longer holds."""


class WorksheetMismatch(ValueError):
    """A worksheet whose (graph, stratum) episode counts differ from the baseline report's sample."""


def _stored_instant(value: object) -> datetime:
    parsed = datetime.fromisoformat(str(value))
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)


class EpisodeContextReader:
    """Reads one graph's episodes as production extracted them, through the read-only seam."""

    def __init__(self, reader: ReadOnlyGraphReader) -> None:
        self._reader = reader

    async def read(self, row: Mapping[str, Any]) -> ReplayInput:
        """*row* is a worksheet row (audit_ruling_overreach.py --emit-worksheet); it supplies the body."""
        own = await self._reader.read_unpaged(EPISODE_CYPHER, {'uuid': row['uuid']})
        if not own:
            raise EpisodeMissing(f"{row['graph']}/{row['uuid']} is no longer in the graph")
        valid_at, source, source_description, group_id = own[0]
        previous_desc = await self._reader.read_unpaged(PREVIOUS_EPISODES_CYPHER, {
            'reference_time': valid_at, 'group_ids': [group_id], 'source': source,
            'uuid': row['uuid'], 'num_episodes': RELEVANT_SCHEMA_LIMIT,
        })
        return ReplayInput(
            graph=row['graph'], uuid=row['uuid'], stratum=row['stratum'], content=row['content'],
            source=str(source), source_description=str(source_description or ''),
            valid_at=_stored_instant(valid_at),
            previous=tuple(str(content) for content, _ in reversed(previous_desc)),
        )


def _cell_counts(rows: Iterable[Mapping[str, Any]]) -> Counter[str]:
    return Counter(f"{row['graph']}/{row['stratum']}" for row in rows)


def check_worksheet(rows: Iterable[Mapping[str, Any]], baseline_report: Mapping[str, Any]) -> None:
    """Raise WorksheetMismatch unless every cell holds the episodes ``sample.sizes`` says it sampled."""
    found = _cell_counts(rows)
    expected = Counter({
        f'{graph}/{stratum}': cell['episodes']
        for graph, by_stratum in baseline_report['sample']['sizes'].items()
        for stratum, cell in by_stratum.items()
    })
    differing = sorted(cell for cell in found.keys() | expected.keys() if found[cell] != expected[cell])
    if differing:
        raise WorksheetMismatch('worksheet episodes differ from the baseline sample in '
                                + ', '.join(f'{c} ({found[c]} != {expected[c]})' for c in differing))


# --------------------------------------------------------------------------- #
# The replay file: one JSONL record per (arm, episode), never episode text
# --------------------------------------------------------------------------- #

REPLAYED, FAILED = 'replayed', 'failed'
_ERROR_CHARS = 500

RecordKey = tuple[str, str, str]
"""``(arm, graph, episode_uuid)``."""


class ReplayFileError(ValueError):
    """A replay file a worksheet or a score must not be built from."""


def record_key(record: Mapping[str, Any]) -> RecordKey:
    return (record['arm'], record['graph'], record['uuid'])


def replayed_record(result: ReplayResult, replay_input: ReplayInput) -> dict[str, Any]:
    return {
        'kind': REPLAYED, 'arm': result.arm, 'model': result.model,
        'instructions_version': result.instructions_version,
        'graph': result.graph, 'uuid': result.uuid, 'stratum': result.stratum,
        'previous_count': len(replay_input.previous),
        'entities': list(result.entities),
        'minted': [dataclasses.asdict(edge) for edge in result.minted],
        'tokens': {stage: dataclasses.asdict(usage) for stage, usage in result.tokens.items()},
    }


def failed_record(arm: ReplayArm, model: str | None, row: Mapping[str, Any], exc: BaseException) -> dict[str, Any]:
    return {
        'kind': FAILED, 'arm': arm.name, 'model': model,
        'graph': row['graph'], 'uuid': row['uuid'], 'stratum': row['stratum'],
        'error_type': type(exc).__name__, 'error': str(exc)[:_ERROR_CHARS],
    }


def _read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    try:
        return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    except (OSError, json.JSONDecodeError) as exc:
        raise ReplayFileError(f'cannot read {path}: {exc}') from exc


def _jsonl(records: Iterable[Mapping[str, Any]]) -> str:
    return ''.join(json.dumps(record, sort_keys=True) + '\n' for record in records)


def _write_atomically(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    staged = path.with_name(path.name + '.tmp')
    staged.write_text(text)
    staged.replace(path)


def complete_replay(
    records: Iterable[Mapping[str, Any]], episodes: Iterable[tuple[str, str]],
) -> dict[str, dict[tuple[str, str], Mapping[str, Any]]]:
    """arm -> (graph, uuid) -> record, for a replay holding every episode under every arm and no failure."""
    records, wanted = list(records), set(episodes)
    if failed := sorted(f'{r["arm"]}:{r["graph"]}/{r["uuid"]}' for r in records if r.get('kind') != REPLAYED):
        raise ReplayFileError(f'the replay holds failure lines: {failed}')
    by_arm: dict[str, dict[tuple[str, str], Mapping[str, Any]]] = {}
    for record in records:
        episode = (record['graph'], record['uuid'])
        if episode not in wanted:
            raise ReplayFileError(f'{record["arm"]} replayed {episode}, which the worksheet does not hold')
        if episode in by_arm.setdefault(record['arm'], {}):
            raise ReplayFileError(f'{record["arm"]} replayed {episode} twice')
        by_arm[record['arm']][episode] = record
    for arm, replayed in sorted(by_arm.items()):
        if missing := sorted(wanted - replayed.keys()):
            raise ReplayFileError(f'arm {arm} is missing {len(missing)} episode(s): {missing}')
    if not by_arm:
        raise ReplayFileError('the replay holds no records')
    return by_arm


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

EXIT_OK, EXIT_UNAVAILABLE, EXIT_INVALID = 0, 1, 2

LlmFactory = Callable[[str | None], LLMClient | None]
ReaderFactory = Callable[[str], ReadOnlyGraphReader]


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog='replay_overreach_extraction',
        description='Read-only replay of ruling-class extraction under lever arms (task 9040).',
    )
    commands = parser.add_subparsers(dest='command', required=True)
    replay = commands.add_parser('replay', help='Re-extract each worksheet episode under each arm.')
    replay.add_argument('--worksheet', required=True,
                        help='audit_ruling_overreach.py --emit-worksheet output (JSONL).')
    replay.add_argument('--baseline-report', required=True,
                        help="The 4716 report.json; the worksheet must match its sample.sizes.")
    replay.add_argument('--arm', action='append', required=True, choices=sorted(ARMS))
    replay.add_argument('--model', default=None, help='Extraction model. Default: llm.model in config.')
    replay.add_argument('--graph-uri', default=None,
                        help='FalkorDB URI. Default: env FALKORDB_URI, then config.')
    replay.add_argument('--graph-host', default=None, help='FalkorDB host, instead of a URI.')
    replay.add_argument('--graph-port', type=int, default=None, help='FalkorDB port, with --graph-host.')
    replay.add_argument('--out', required=True, help='Replay JSONL; an existing file is resumed.')
    worksheet = commands.add_parser('worksheet', help='Emit the blinded mixed-arm adjudication worksheet.')
    worksheet.add_argument('--replay', required=True)
    worksheet.add_argument('--worksheet', required=True, help='The worksheet the replay ran over.')
    worksheet.add_argument('--out', required=True)
    return parser


def _live_reader_factory(args: argparse.Namespace) -> ReaderFactory:
    if args.graph_host is not None or args.graph_port is not None:
        return lambda graph: ReadOnlyGraphReader(graph_name=graph, host=args.graph_host, port=args.graph_port)
    uri = _resolve_uri(args)
    return lambda graph: ReadOnlyGraphReader(graph_name=graph, uri=uri)


def _live_llm_factory(model: str | None) -> LLMClient | None:
    return build_arm_llm(FusedMemoryConfig(), model)


def _load_replay_inputs(args: argparse.Namespace) -> list[dict[str, Any]]:
    rows = _read_jsonl(args.worksheet)
    try:
        check_worksheet(rows, json.loads(Path(args.baseline_report).read_text()))
    except (OSError, json.JSONDecodeError, KeyError) as exc:
        raise ReplayFileError(f'cannot read the baseline sample from {args.baseline_report}: {exc}') from exc
    return rows


def _model_conflicts(kept: Iterable[Mapping[str, Any]], arms: Iterable[ReplayArm], model: str | None) -> list[str]:
    names = {arm.name for arm in arms}
    return sorted({
        f'{r["arm"]} was replayed on {r["model"]}, not {model}'
        for r in kept if r['arm'] in names and r['model'] != model
    })


async def _replay_rows(
    rows: Sequence[Mapping[str, Any]], arms: Sequence[ReplayArm], llm: LLMClient,
    readers: Mapping[str, EpisodeContextReader], done: set[RecordKey], out: Path,
) -> int:
    failures = 0
    with out.open('a') as sink:
        for row in rows:
            pending = [arm for arm in arms if (arm.name, row['graph'], row['uuid']) not in done]
            if not pending:
                continue
            records: list[dict[str, Any]] = []
            try:
                replay_input = await readers[row['graph']].read(row)
            except Exception as exc:
                logger.warning('reading %s/%s failed', row['graph'], row['uuid'], exc_info=True)
                records = [failed_record(arm, llm.model, row, exc) for arm in pending]
            else:
                for arm in pending:
                    try:
                        records.append(replayed_record(await replay_episode(llm, replay_input, arm), replay_input))
                    except Exception as exc:
                        logger.warning('replaying %s/%s under %s failed', row['graph'], row['uuid'],
                                       arm.name, exc_info=True)
                        records.append(failed_record(arm, llm.model, row, exc))
            failures += sum(1 for record in records if record['kind'] == FAILED)
            sink.write(_jsonl(records))
            sink.flush()
    return failures


async def _replay_command(args: argparse.Namespace, llm_factory: LlmFactory, reader_factory: ReaderFactory) -> int:
    """Validate, then replay what is not yet recorded; failure lines from an earlier run are retried."""
    try:
        rows = _load_replay_inputs(args)
        out = Path(args.out)
        kept = [r for r in _read_jsonl(out) if r.get('kind') == REPLAYED] if out.exists() else []
    except (ReplayFileError, WorksheetMismatch) as exc:
        logger.error('invalid inputs, nothing is written: %s', exc)
        return EXIT_INVALID
    llm = llm_factory(args.model)
    if llm is None:
        logger.error('no extraction LLM is configured; nothing is written')
        return EXIT_UNAVAILABLE
    arms = [dataclasses.replace(ARMS[name], model=args.model) for name in dict.fromkeys(args.arm)]
    if conflicts := _model_conflicts(kept, arms, llm.model):
        logger.error('this --out already holds other models, nothing is written: %s', conflicts)
        return EXIT_INVALID
    _write_atomically(out, _jsonl(kept))
    graph_readers = {graph: reader_factory(graph) for graph in dict.fromkeys(row['graph'] for row in rows)}
    try:
        failures = await _replay_rows(
            rows, arms, llm, {g: EpisodeContextReader(r) for g, r in graph_readers.items()},
            {record_key(r) for r in kept}, out,
        )
    finally:
        for reader in graph_readers.values():
            await reader.aclose()
    if failures:
        logger.error('%d (arm, episode) replay(s) failed; re-run to resume them', failures)
        return EXIT_UNAVAILABLE
    return EXIT_OK


def _hash_order(edge: Mapping[str, Any]) -> str:
    return hashlib.sha256(edge['edge_uuid'].encode()).hexdigest()


def blinded_rows(
    rows: Iterable[Mapping[str, Any]], by_arm: Mapping[str, Mapping[tuple[str, str], Mapping[str, Any]]],
) -> list[dict[str, Any]]:
    """Each episode's body with every arm's minted edges interleaved by sha256(edge_uuid); no arm is named."""
    return [
        {
            'graph': row['graph'], 'uuid': row['uuid'], 'stratum': row['stratum'], 'content': row['content'],
            'minted': sorted(
                (dict(edge) for replayed in by_arm.values()
                 for edge in replayed[(row['graph'], row['uuid'])]['minted']),
                key=_hash_order,
            ),
        }
        for row in rows
    ]


def _worksheet_command(args: argparse.Namespace) -> int:
    try:
        rows = _read_jsonl(args.worksheet)
        by_arm = complete_replay(_read_jsonl(args.replay), ((r['graph'], r['uuid']) for r in rows))
    except ReplayFileError as exc:
        logger.error('invalid inputs, nothing is written: %s', exc)
        return EXIT_INVALID
    _write_atomically(Path(args.out), _jsonl(blinded_rows(rows, by_arm)))
    return EXIT_OK


async def _run(
    args: argparse.Namespace, llm_factory: LlmFactory | None, reader_factory: ReaderFactory | None,
) -> int:
    if args.command == 'replay':
        return await _replay_command(
            args, llm_factory or _live_llm_factory, reader_factory or _live_reader_factory(args),
        )
    return _worksheet_command(args)


def run(
    argv: Sequence[str] | None = None, *,
    llm_factory: LlmFactory | None = None, reader_factory: ReaderFactory | None = None,
) -> int:
    """Parse *argv* (default ``sys.argv[1:]``) and run one subcommand.

    *llm_factory* maps the --model value to an extraction client (default:
    build_arm_llm over the loaded config); *reader_factory* maps a graph name to
    a ReadOnlyGraphReader (default: FalkorDB over GRAPH.RO_QUERY).
    """
    return asyncio.run(_run(_build_parser().parse_args(argv), llm_factory, reader_factory))


def main() -> int:
    logging.basicConfig(level=logging.INFO, stream=sys.stderr, format='%(levelname)s %(name)s: %(message)s')
    return run()


if __name__ == '__main__':
    sys.exit(main())
