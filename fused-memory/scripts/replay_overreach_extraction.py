#!/usr/bin/env python3
"""Replay ruling-class extraction under an arm, to measure levers O1 (task 9040) and O6 (task 9065).

docs/ruling-overreach-guard-design-2026-10-01/design.md §6 names lever O1:
``custom_extraction_instructions`` for the ``decisions_and_rationale`` class.
§10.3 re-opened the lever choice with O1 to be measured by replaying the 48
frozen-window episodes. This script is that replay. Each episode is
re-extracted under each arm by graphiti_core 0.28.2's own
utils/maintenance/node_operations.py::extract_nodes and
utils/maintenance/edge_operations.py::extract_edges, unmodified, followed by
graphiti's in-batch exact-fact dedupe (the driver-free fast path of
edge_operations.py::resolve_extracted_edges). An arm is (instructions, client):
control and o1 differ only in the instructions they splice into those prompts,
on the configured client, and o6-low and o6-medium are the control's
instructions on O6_MODEL at reasoning effort low and medium (esc-9065-2). So
every arm shares one replay file, one blinded worksheet and one score against
the control.

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

    uv run python scripts/replay_overreach_extraction.py replay \\
        --worksheet /tmp/9040-worksheet-4716.jsonl \\
        --baseline-report ../docs/ruling-overreach-guard-design-2026-10-01/report.json \\
        --arm control --arm o6-low --arm o6-medium --out /tmp/9065-replay.jsonl

    uv run python scripts/replay_overreach_extraction.py worksheet \\
        --replay /tmp/9040-replay.jsonl --worksheet /tmp/9040-worksheet-4716.jsonl \\
        --out /tmp/9040-blind-worksheet.jsonl

    uv run python scripts/replay_overreach_extraction.py score \\
        --replay ../docs/ruling-overreach-lever-reopen-2026-11-10/replay.jsonl \\
        --verdicts ../docs/ruling-overreach-lever-reopen-2026-11-10/verdicts.json \\
        --baseline-report ../docs/ruling-overreach-guard-design-2026-10-01/report.json \\
        --out ../docs/ruling-overreach-lever-reopen-2026-11-10/report.json

``replay`` resumes an existing --out: recorded (arm, episode) pairs are
skipped and failure lines are retried. Each arm runs on the client its
ExtractionLlm names, one client per distinct model and reasoning effort, so
arms on different clients share one --out; an --out recording an arm on
another model or effort than that arm's client is refused. A record written
before task 9065 carries no reasoning_effort and reads as None, the configured
client's. ``worksheet`` interleaves
every arm's edges by sha256(edge_uuid) and names no arm, for blind
adjudication; it carries episode bodies, so it is not committed. ``score``
validates one verdict file over every arm's edges with the audit's own
load_verdicts, rates each arm at the 4716 cell weights with the audit's own
estimators, and judges each lever arm against the control by the
pre-registered rule beside OVERREACH_RATIO_MAX.

Exit codes: 0 ran; 1 the LLM or graph is unavailable, or an episode failed
(re-run to resume); 2 an input is invalid, or an arm's client cannot be built
or cannot replay it (nothing is written).
"""
from __future__ import annotations

import argparse
import asyncio
import contextlib
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
from enum import StrEnum
from pathlib import Path
from types import MappingProxyType
from typing import Any

from graphiti_core.edges import EntityEdge
from graphiti_core.graphiti_types import GraphitiClients
from graphiti_core.llm_client.client import LLMClient
from graphiti_core.llm_client.openai_client import OpenAIClient
from graphiti_core.nodes import EpisodeType, EpisodicNode
from graphiti_core.search.search_utils import RELEVANT_SCHEMA_LIMIT
from graphiti_core.utils.maintenance.dedup_helpers import _normalize_string_exact
from graphiti_core.utils.maintenance.edge_operations import extract_edges
from graphiti_core.utils.maintenance.node_operations import extract_nodes
from openai.types.chat import ChatCompletionMessageParam
from openai.types.responses import EasyInputMessageParam, ResponseInputParam
from openai.types.shared import ReasoningEffort
from openai.types.shared_params import Reasoning
from pydantic import BaseModel

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
    OutOfSampleVerdict,
    SampleDefinition,
    VerdictError,
    VerdictSet,
    WeightedStratum,
    adjudicated_rates,
    load_verdicts,
    population_weighted_rates,
    weighted_rate,
)

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

CONTROL = 'control'


@dataclass(frozen=True)
class ExtractionLlm:
    """The client an arm extracts on.

    ``model`` None is config ``llm.model``, production's extraction model.
    ``reasoning_effort`` None is the client build_llm_client builds, as
    configured, temperature included. A set effort is sent as
    ``reasoning.effort``, with no temperature.
    """

    model: str | None = None
    reasoning_effort: ReasoningEffort = None


@dataclass(frozen=True)
class ReplayArm:
    """What one arm splices into graphiti's extraction prompts, and the client it extracts on."""

    name: str
    instructions: str | None
    instructions_version: str
    llm: ExtractionLlm = ExtractionLlm()


O6_MODEL = 'gpt-6.1-sol'
"""Lever O6's stronger extraction model, priced in the 2026-11-10 design.md §5 from
score_write_triage_pairs.py::LIST_PRICES. Effort low is what the production write-triage
judge runs it at; medium is the API's default."""

ARMS: Mapping[str, ReplayArm] = MappingProxyType({
    CONTROL: ReplayArm(CONTROL, None, NO_INSTRUCTIONS),
    'o1': ReplayArm('o1', O1_INSTRUCTIONS, O1_INSTRUCTIONS_VERSION),
    'o6-low': ReplayArm('o6-low', None, NO_INSTRUCTIONS, ExtractionLlm(O6_MODEL, 'low')),
    'o6-medium': ReplayArm('o6-medium', None, NO_INSTRUCTIONS, ExtractionLlm(O6_MODEL, 'medium')),
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
    reasoning_effort: str | None
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


class ArmClientMismatch(ValueError):
    """An arm handed a client on another model or reasoning effort than its own, or an effort
    that the configured client class cannot send."""


def _responses_input(messages: list[ChatCompletionMessageParam]) -> ResponseInputParam:
    """graphiti's converted messages as Responses input. graphiti emits only system and user
    messages with text content, so anything else is a defect, refused before it is sent."""
    converted: ResponseInputParam = []
    for message in messages:
        role, content = message['role'], message.get('content')
        if role not in ('system', 'user') or not isinstance(content, str):
            raise TypeError(f'a {role} message with {type(content).__name__} content is not Responses input')
        converted.append(EasyInputMessageParam(role=role, content=content))
    return converted


class ExplicitEffortOpenAIClient(OpenAIClient):
    """graphiti's OpenAIClient for a reasoning model graphiti_core 0.28.2 does not recognise by name.

    graphiti treats only gpt-5, o1 and o3 names as reasoning models: any other
    model is sent temperature and never sent reasoning. This client sends no
    temperature, and exactly the arm's reasoning effort (esc-9065-2). It wraps
    the client build_llm_client built, reusing its SDK client, config,
    max_tokens and token tracker. graphiti's own ``reasoning`` stays at its
    default and is never read: ``reasoning_effort`` is the one copy.
    """

    def __init__(self, built: OpenAIClient, effort: ReasoningEffort) -> None:
        if effort is None:
            raise ValueError('an effort client sends a chosen effort; build the configured client instead')
        super().__init__(config=built.config, client=built.client, max_tokens=built.max_tokens)
        self.token_tracker = built.token_tracker
        self._reasoning_effort: ReasoningEffort = effort

    @property
    def reasoning_effort(self) -> ReasoningEffort:
        """The effort every structured call sends."""
        return self._reasoning_effort

    async def _create_structured_completion(
        self, model: str, messages: list[ChatCompletionMessageParam], temperature: float | None,
        max_tokens: int, response_model: type[BaseModel], reasoning: str | None = None,
        verbosity: str | None = None,
    ) -> Any:
        return await self.client.responses.parse(
            model=model, input=_responses_input(messages), max_output_tokens=max_tokens,
            text_format=response_model, reasoning=Reasoning(effort=self._reasoning_effort),
        )

    async def _create_completion(
        self, model: str, messages: list[ChatCompletionMessageParam], temperature: float | None,
        max_tokens: int, response_model: type[BaseModel] | None = None, reasoning: str | None = None,
        verbosity: str | None = None,
    ) -> Any:
        raise NotImplementedError(
            'a JSON-mode chat completion cannot carry reasoning.effort, so it is never sent without it',
        )


def client_reasoning_effort(llm: LLMClient) -> str | None:
    """The reasoning effort *llm* declares it sends: its ``reasoning_effort``, or None when it has
    none, as graphiti's own clients have none. The one reader of that attribute."""
    return getattr(llm, 'reasoning_effort', None)


def build_arm_llm(cfg: FusedMemoryConfig, llm: ExtractionLlm) -> LLMClient | None:
    """The configured extraction client on *llm*'s model, sending *llm*'s effort when it sets one.

    *cfg* itself is never modified. An effort is sent on the Responses API, so
    a configured client class that serves chat completions only is refused
    rather than handed an effort it would drop.
    """
    arm_cfg = cfg.model_copy(deep=True)
    if llm.model is not None:
        arm_cfg.llm.model = llm.model
    client = build_llm_client(arm_cfg)
    if client is None or llm.reasoning_effort is None:
        return client
    if not isinstance(client, OpenAIClient):
        raise ArmClientMismatch(
            f'reasoning effort {llm.reasoning_effort} is sent on the Responses API, which the '
            f'{cfg.llm.provider}/{cfg.llm.client_class} client ({type(client).__name__}) does not serve',
        )
    return ExplicitEffortOpenAIClient(client, llm.reasoning_effort)


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


def _client_axis(model: str | None, effort: str | None) -> str:
    return f"(model {model or 'as configured'}, effort {effort!r})"


async def check_arm_client(llm: LLMClient, arm: ReplayArm) -> None:
    """Raise unless *llm* can replay *arm*: on the arm's model, sending exactly the arm's effort
    (none for an arm that names none), with attributable tokens. Calls nothing."""
    effort = client_reasoning_effort(llm)
    if (arm.llm.model is not None and llm.model != arm.llm.model) or effort != arm.llm.reasoning_effort:
        raise ArmClientMismatch(
            f'arm {arm.name} extracts on {_client_axis(arm.llm.model, arm.llm.reasoning_effort)}, '
            f'but the client is on {_client_axis(llm.model, effort)}',
        )
    async with measure_llm_tokens(llm) as probe:
        pass
    _usage(probe)


async def replay_episode(llm: LLMClient, episode: ReplayInput, arm: ReplayArm) -> ReplayResult:
    """Extract *episode* under *arm* through graphiti's own functions; no driver exists to reach."""
    await check_arm_client(llm, arm)
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
        arm=arm.name, model=llm.model, reasoning_effort=client_reasoning_effort(llm),
        instructions_version=arm.instructions_version,
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


def baseline_cells(baseline_report: Mapping[str, Any]) -> Counter[str]:
    """``graph/stratum`` -> the episodes the baseline report's ``sample.sizes`` says it sampled."""
    try:
        return Counter({
            f'{graph}/{stratum}': int(cell['episodes'])
            for graph, by_stratum in baseline_report['sample']['sizes'].items()
            for stratum, cell in by_stratum.items()
        })
    except (AttributeError, KeyError, TypeError, ValueError) as exc:
        raise ReplayFileError(f'the baseline report has no sample.sizes: {exc!r}') from exc


def check_worksheet(rows: Iterable[Mapping[str, Any]], sampled: Mapping[str, int]) -> None:
    """Raise WorksheetMismatch unless every cell holds the episodes *sampled* (baseline_cells) names."""
    found, expected = _cell_counts(rows), Counter(sampled)
    differing = sorted(cell for cell in found.keys() | expected.keys() if found[cell] != expected[cell])
    if differing:
        raise WorksheetMismatch('worksheet episodes differ from the baseline sample in '
                                + ', '.join(f'{c} ({found[c]} != {expected[c]})' for c in differing))


# --------------------------------------------------------------------------- #
# The replay file: one JSONL record per (arm, episode), never episode text
# --------------------------------------------------------------------------- #


class RecordKind(StrEnum):
    replayed = 'replayed'
    failed = 'failed'


_ERROR_CHARS = 500

RecordKey = tuple[str, str, str]
"""``(arm, graph, episode_uuid)``. An arm's name fixes its client, model and effort, so the key needs neither."""


class ReplayFileError(ValueError):
    """A replay file a worksheet or a score must not be built from."""


def record_key(record: Mapping[str, Any]) -> RecordKey:
    return (record['arm'], record['graph'], record['uuid'])


def _record_effort(record: Mapping[str, Any]) -> str | None:
    """A record's reasoning effort. A record written before task 9065 has no such key: its
    client sent none, so it reads as None."""
    return record.get('reasoning_effort')


def replayed_record(result: ReplayResult, replay_input: ReplayInput) -> dict[str, Any]:
    return {
        'kind': RecordKind.replayed, 'arm': result.arm, 'model': result.model,
        'reasoning_effort': result.reasoning_effort, 'instructions_version': result.instructions_version,
        'graph': result.graph, 'uuid': result.uuid, 'stratum': result.stratum,
        'previous_count': len(replay_input.previous),
        'entities': list(result.entities),
        'minted': [dataclasses.asdict(edge) for edge in result.minted],
        'tokens': {stage: dataclasses.asdict(usage) for stage, usage in result.tokens.items()},
    }


def failed_record(arm: ReplayArm, llm: LLMClient, row: Mapping[str, Any], exc: BaseException) -> dict[str, Any]:
    return {
        'kind': RecordKind.failed, 'arm': arm.name, 'model': llm.model,
        'reasoning_effort': client_reasoning_effort(llm),
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


def complete_replay(
    records: Iterable[Mapping[str, Any]], episodes: Iterable[tuple[str, str]],
) -> dict[str, dict[tuple[str, str], Mapping[str, Any]]]:
    """arm -> (graph, uuid) -> record, for a replay holding every episode under every arm and no failure."""
    records, wanted = list(records), set(episodes)
    if failed := sorted(
        f'{r["arm"]}:{r["graph"]}/{r["uuid"]}' for r in records if r.get('kind') != RecordKind.replayed
    ):
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
# The score: each arm at the 4716 cell weights, and the O1 decision (design §2.3-§2.4)
# --------------------------------------------------------------------------- #

OVERREACH_RATIO_MAX = 0.5
"""A lever passes only if its overreach, per minted edge and per day, is at most this share of
the control's: the 2026-10-01 design.md §6.1 revisit clause's 'at least halved'."""

HOLDING_RETENTION_MIN = 0.8
"""A lever passes only if its weighted holdings/day are at least this share of the control's."""

REPORTED_RATES = (
    'label_counts', 'overreach_rate_minted', 'overreach_rate_substantive',
    'episode_hit_rate', 'holding_share', 'misbound_count',
)
"""The adjudicated_rates measures that mean something for an edge no graph ever held."""

EpisodeKey = tuple[str, str]
"""``(graph, episode_uuid)``."""


@dataclass(frozen=True)
class Baseline:
    """What the score takes from the 4716 report.json: its sample, cell populations and own result."""

    sample_cells: Mapping[str, int]
    """baseline_cells of the report."""
    window_population: Mapping[str, Mapping[str, int]]
    window_days: float
    overreach_rate_minted: Mapping[str, Any]

    @classmethod
    def from_report(cls, report: Mapping[str, Any]) -> Baseline:
        sample_cells = baseline_cells(report)
        try:
            return cls(
                sample_cells=sample_cells,
                window_population={
                    graph: {stratum: int(cell['window_population']) for stratum, cell in by_stratum.items()}
                    for graph, by_stratum in report['strata'].items()
                },
                window_days=float(report['sample']['window_days']),
                overreach_rate_minted=report['adjudicated']['population_weighted']['overreach_rate_minted'],
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ReplayFileError(f'the baseline report lacks {exc}') from exc


def _audit_row(record: Mapping[str, Any]) -> dict[str, Any]:
    """An audit worksheet row for one replayed episode; no graph ever served or kept its edges."""
    return {
        'graph': record['graph'], 'uuid': record['uuid'], 'stratum': record['stratum'],
        'minted': [dict(edge, served=None, live_strict=None) for edge in record['minted']],
    }


ReplayByArm = Mapping[str, Mapping[EpisodeKey, Mapping[str, Any]]]
"""arm -> episode -> replayed record, as complete_replay returns it."""


def _minted_edges(by_arm: ReplayByArm) -> dict[tuple[str, str], str]:
    """``(graph, edge_uuid)`` -> episode uuid, over every arm; one uuid minted twice is ambiguous."""
    edges: dict[tuple[str, str], str] = {}
    for replayed in by_arm.values():
        for record in replayed.values():
            for edge in record['minted']:
                key = (record['graph'], edge['edge_uuid'])
                if key in edges:
                    raise ReplayFileError(f'edge {key} was minted twice, so no verdict can say which')
                edges[key] = record['uuid']
    return edges


def validate_verdicts(obj: Any, by_arm: ReplayByArm, episodes: Iterable[EpisodeKey]) -> VerdictSet:
    """load_verdicts once over every arm's edges. A replayed edge was never in a graph, so the
    verdict load_verdicts would call stale names an edge no arm minted."""
    verdicts = load_verdicts(
        obj, expected_edges=_minted_edges(by_arm), sampled_episodes=episodes,
        existing_edges=(), definition=SampleDefinition(),
    )
    if verdicts.stale:
        raise OutOfSampleVerdict(
            f'{len(verdicts.stale)} verdict(s) judge an edge no arm minted',
            [v.edge_uuid for v in verdicts.stale],
        )
    return verdicts


def _per_day(weighted_count: float, window_days: float) -> float | None:
    return round(weighted_count / window_days, 1) if window_days else None


def _weighted_cells(
    rows: Sequence[Mapping[str, Any]], verdicts: VerdictSet,
    weights: Mapping[str, Mapping[str, float]], measure: str,
) -> list[WeightedStratum]:
    """``(weight, k, n)`` per sampled cell for one adjudicated_rates *measure* over minted edges."""
    cells = []
    for graph, by_stratum in weights.items():
        for stratum, weight in by_stratum.items():
            cell = [row for row in rows if (row['graph'], row['stratum']) == (graph, stratum)]
            measured = adjudicated_rates(cell, verdicts)['all'][measure]
            cells.append((weight, measured['k'], measured['n']))
    return cells


def _window_count(cells: Iterable[WeightedStratum]) -> float:
    return sum(weight * k for weight, k, _ in cells)


def window_measures(
    rows: Sequence[Mapping[str, Any]], verdicts: VerdictSet,
    weights: Mapping[str, Mapping[str, float]], window_days: float,
) -> dict[str, Any]:
    """Holding share and holdings/day at the cell *weights* population_weighted_rates derived, and
    the unrounded window counts of minted, overreach and holding edges that lever_decision compares."""
    overreach = _weighted_cells(rows, verdicts, weights, 'overreach_rate_minted')
    holding = _weighted_cells(rows, verdicts, weights, 'holding_share')
    return {
        'holding_share': weighted_rate(holding),
        'holdings_per_day': _per_day(_window_count(holding), window_days),
        'window_counts': {
            'minted': sum(weight * n for weight, _, n in overreach),
            'overreach': _window_count(overreach),
            'holding': _window_count(holding),
        },
    }


def _entity_names(record: Mapping[str, Any]) -> frozenset[str]:
    return frozenset(_normalize_string_exact(name) for name in record['entities'])


def entity_overlap(
    replayed: Mapping[EpisodeKey, Mapping[str, Any]], control: Mapping[EpisodeKey, Mapping[str, Any]],
) -> dict[str, float | None]:
    """Entities per episode, and the mean per-episode share of the control's names this arm kept."""
    shares = [
        len(_entity_names(replayed[episode]) & names) / len(names)
        for episode, record in control.items() if (names := _entity_names(record))
    ]
    counts = [len(record['entities']) for record in replayed.values()]
    return {
        'mean_per_episode': round(sum(counts) / len(counts), 4),
        'mean_share_of_control': round(sum(shares) / len(shares), 4) if shares else None,
    }


def token_totals(records: Iterable[Mapping[str, Any]]) -> dict[str, dict[str, int]]:
    totals: dict[str, Counter[str]] = {}
    for record in records:
        for stage, usage in record['tokens'].items():
            totals.setdefault(stage, Counter()).update(usage)
    return {stage: dict(total) for stage, total in sorted(totals.items())}


def _only(field: str, values: Iterable[Any]) -> Any:
    distinct = set(values)
    if len(distinct) != 1:
        raise ReplayFileError(f'one arm holds records with {field} {sorted(map(str, distinct))}')
    return distinct.pop()


def arm_score(
    replayed: Mapping[EpisodeKey, Mapping[str, Any]], control: Mapping[EpisodeKey, Mapping[str, Any]],
    verdicts: VerdictSet, baseline: Baseline,
) -> dict[str, Any]:
    rows = [_audit_row(record) for record in replayed.values()]
    weighted = population_weighted_rates(rows, verdicts, baseline.window_population, baseline.window_days)
    return {
        'model': _only('model', (record['model'] for record in replayed.values())),
        'reasoning_effort': _only('reasoning_effort', map(_record_effort, replayed.values())),
        'instructions_version': _only(
            'instructions_version', (record['instructions_version'] for record in replayed.values()),
        ),
        'episodes': len(rows),
        'minted': sum(len(row['minted']) for row in rows),
        'rates': {
            scope: {key: measured[key] for key in REPORTED_RATES}
            for scope, measured in adjudicated_rates(rows, verdicts).items()
        },
        'population_weighted': weighted | window_measures(
            rows, verdicts, weighted['weights'], baseline.window_days,
        ),
        'entities': entity_overlap(replayed, control),
        'tokens': token_totals(replayed.values()),
    }


def _ratio(candidate: float | None, control: float | None) -> float | None:
    return candidate / control if candidate is not None and control else None


def _overreach_share(counts: Mapping[str, float]) -> float | None:
    return _ratio(counts['overreach'], counts['minted'])


def _shown(ratio: float | None) -> float | None:
    return None if ratio is None else round(ratio, 4)


def lever_decision(control: Mapping[str, Any], candidate: Mapping[str, Any]) -> dict[str, Any]:
    """design §2.4, over two arms' ``population_weighted`` blocks.

    Every ratio is taken over the unrounded ``window_counts`` and rounded only
    for display. Both arms share the baseline's window, so a ratio of window
    counts is the ratio of their per-day rates.
    """
    control_counts, candidate_counts = control['window_counts'], candidate['window_counts']
    control_point = control['overreach_rate_minted']['rate']
    candidate_upper = (candidate['overreach_rate_minted']['ci'] or [None, None])[1]
    per_minted = _ratio(_overreach_share(candidate_counts), _overreach_share(control_counts))
    per_day = _ratio(candidate_counts['overreach'], control_counts['overreach'])
    retention = _ratio(candidate_counts['holding'], control_counts['holding'])
    checks = {
        'halves_overreach_per_minted': per_minted is not None and per_minted <= OVERREACH_RATIO_MAX,
        'halves_overreach_per_day': per_day is not None and per_day <= OVERREACH_RATIO_MAX,
        'upper_bound_below_control_point': (
            candidate_upper is not None and control_point is not None and candidate_upper < control_point
        ),
        'retains_holdings': retention is not None and retention >= HOLDING_RETENTION_MIN,
    }
    return {
        'overreach_ratio_minted': _shown(per_minted), 'overreach_ratio_per_day': _shown(per_day),
        'holding_retention': _shown(retention),
        'bars': {'overreach_ratio_max': OVERREACH_RATIO_MAX, 'holding_retention_min': HOLDING_RETENTION_MIN},
        'checks': checks, 'passes': all(checks.values()),
    }


def _overlap(one: Sequence[float] | None, other: Sequence[float] | None) -> bool | None:
    if not one or not other:
        return None
    return one[0] <= other[1] and other[0] <= one[1]


def fidelity(baseline: Mapping[str, Any], control: Mapping[str, Any]) -> dict[str, Any]:
    """The control against the production rate it should reproduce, each with its interval."""
    return {
        'baseline': {'rate': baseline['rate'], 'ci': baseline['ci']},
        'control': {'rate': control['rate'], 'ci': control['ci']},
        'intervals_overlap': _overlap(baseline['ci'], control['ci']),
    }


def score_replay(
    records: Sequence[Mapping[str, Any]], verdict_obj: Any, baseline_report: Mapping[str, Any],
) -> dict[str, Any]:
    """Every arm scored against one verdict file, then each lever arm judged against the control."""
    baseline = Baseline.from_report(baseline_report)
    episodes = list(dict.fromkeys((record['graph'], record['uuid']) for record in records))
    by_arm = complete_replay(records, episodes)
    if CONTROL not in by_arm:
        raise ReplayFileError(f'every arm is judged against {CONTROL!r}, which this replay lacks')
    check_worksheet(by_arm[CONTROL].values(), baseline.sample_cells)
    verdicts = validate_verdicts(verdict_obj, by_arm, episodes)
    arms = {arm: arm_score(replayed, by_arm[CONTROL], verdicts, baseline) for arm, replayed in sorted(by_arm.items())}
    control = arms[CONTROL]['population_weighted']
    return {
        'sample': verdicts.definition.to_dict(),
        'verdicts': len(verdicts.verdicts),
        'arms': arms,
        'comparison': {
            arm: lever_decision(control, scored['population_weighted'])
            for arm, scored in arms.items() if arm != CONTROL
        },
        'fidelity': fidelity(baseline.overreach_rate_minted, control['overreach_rate_minted']),
    }


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

EXIT_OK, EXIT_UNAVAILABLE, EXIT_INVALID = 0, 1, 2

LlmFactory = Callable[[ExtractionLlm], LLMClient | None]
ArmClients = tuple[tuple[ReplayArm, LLMClient], ...]
"""Each selected arm with the client its ExtractionLlm names."""


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog='replay_overreach_extraction',
        description='Read-only replay of ruling-class extraction under lever arms (tasks 9040, 9065).',
    )
    commands = parser.add_subparsers(dest='command', required=True)
    replay = commands.add_parser('replay', help='Re-extract each worksheet episode under each arm.')
    replay.add_argument('--worksheet', required=True,
                        help='audit_ruling_overreach.py --emit-worksheet output (JSONL).')
    replay.add_argument('--baseline-report', required=True,
                        help="The 4716 report.json; the worksheet must match its sample.sizes.")
    replay.add_argument('--arm', action='append', required=True, choices=sorted(ARMS))
    add_graph_arguments(replay)
    replay.add_argument('--out', required=True, help='Replay JSONL; an existing file is resumed.')
    worksheet = commands.add_parser('worksheet', help='Emit the blinded mixed-arm adjudication worksheet.')
    worksheet.add_argument('--replay', required=True)
    worksheet.add_argument('--worksheet', required=True, help='The worksheet the replay ran over.')
    worksheet.add_argument('--out', required=True)
    score = commands.add_parser('score', help='Rate every arm against the adjudicated verdicts.')
    score.add_argument('--replay', required=True)
    score.add_argument('--verdicts', required=True,
                       help='Verdicts on every replayed edge, in audit_ruling_overreach.py format.')
    score.add_argument('--baseline-report', required=True,
                       help='The 4716 report.json, for its cell populations and its own rate.')
    score.add_argument('--out', required=True, help='report.json to write.')
    return parser


def _live_llm_factory(llm: ExtractionLlm) -> LLMClient | None:
    return build_arm_llm(FusedMemoryConfig(), llm)


def _load_replay_inputs(args: argparse.Namespace) -> list[dict[str, Any]]:
    rows = _read_jsonl(args.worksheet)
    check_worksheet(rows, baseline_cells(read_json(args.baseline_report)))
    return rows


def _arm_clients(arms: Sequence[ReplayArm], llm_factory: LlmFactory) -> ArmClients | None:
    """One client per distinct arm ExtractionLlm, in first-seen order; None if any has none.
    The factory raises ArmClientMismatch for a client it cannot build."""
    clients: dict[ExtractionLlm, LLMClient] = {}
    for llm in dict.fromkeys(arm.llm for arm in arms):
        client = llm_factory(llm)
        if client is None:
            logger.error('no extraction LLM for %s at reasoning effort %s; nothing is written',
                         llm.model or 'the configured llm.model', llm.reasoning_effort)
            return None
        clients[llm] = client
    return tuple((arm, clients[arm.llm]) for arm in arms)


def _client_conflicts(kept: Iterable[Mapping[str, Any]], arm_clients: ArmClients) -> list[str]:
    client_of = {arm.name: (llm.model, client_reasoning_effort(llm)) for arm, llm in arm_clients}
    return sorted({
        f'{r["arm"]} was replayed on {_client_axis(r["model"], _record_effort(r))}, '
        f'not {_client_axis(*client_of[r["arm"]])}'
        for r in kept if r['arm'] in client_of and (r['model'], _record_effort(r)) != client_of[r['arm']]
    })


async def _replay_rows(
    rows: Sequence[Mapping[str, Any]], arm_clients: ArmClients,
    readers: Mapping[str, EpisodeContextReader], done: set[RecordKey], out: Path,
) -> int:
    failures = 0
    with out.open('a') as sink:
        for row in rows:
            pending = [(arm, llm) for arm, llm in arm_clients if (arm.name, row['graph'], row['uuid']) not in done]
            if not pending:
                continue
            records: list[dict[str, Any]] = []
            try:
                replay_input = await readers[row['graph']].read(row)
            except (EpisodeMissing, *GRAPH_UNAVAILABLE) as exc:
                logger.warning('reading %s/%s failed', row['graph'], row['uuid'], exc_info=True)
                records = [failed_record(arm, llm, row, exc) for arm, llm in pending]
            else:
                for arm, llm in pending:
                    try:
                        records.append(replayed_record(await replay_episode(llm, replay_input, arm), replay_input))
                    except Exception as exc:
                        logger.warning('replaying %s/%s under %s failed', row['graph'], row['uuid'],
                                       arm.name, exc_info=True)
                        records.append(failed_record(arm, llm, row, exc))
            failures += sum(1 for record in records if record['kind'] == RecordKind.failed)
            sink.write(_jsonl(records))
            sink.flush()
    return failures


async def _replay_command(args: argparse.Namespace, llm_factory: LlmFactory, reader_factory: ReaderFactory) -> int:
    """Validate, then replay what is not yet recorded; failure lines from an earlier run are retried."""
    try:
        rows = _load_replay_inputs(args)
        out = Path(args.out)
        kept = [r for r in _read_jsonl(out) if r.get('kind') == RecordKind.replayed] if out.exists() else []
    except (ReplayFileError, UnreadableInput, WorksheetMismatch) as exc:
        logger.error('invalid inputs, nothing is written: %s', exc)
        return EXIT_INVALID
    try:
        arm_clients = _arm_clients([ARMS[name] for name in dict.fromkeys(args.arm)], llm_factory)
        for arm, llm in arm_clients or ():
            await check_arm_client(llm, arm)
    except (ArmClientMismatch, UnmeasurableClient) as exc:
        logger.error('an arm cannot be replayed on its client, nothing is written: %s', exc)
        return EXIT_INVALID
    if arm_clients is None:
        return EXIT_UNAVAILABLE
    if conflicts := _client_conflicts(kept, arm_clients):
        logger.error('this --out already holds other clients, nothing is written: %s', conflicts)
        return EXIT_INVALID
    write_files_atomically({out: _jsonl(kept)})
    async with contextlib.AsyncExitStack() as opened:
        readers: dict[str, EpisodeContextReader] = {}
        for graph in dict.fromkeys(row['graph'] for row in rows):
            reader = reader_factory(graph)
            opened.push_async_callback(reader.aclose)
            readers[graph] = EpisodeContextReader(reader)
        failures = await _replay_rows(rows, arm_clients, readers, {record_key(r) for r in kept}, out)
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
    write_files_atomically({Path(args.out): _jsonl(blinded_rows(rows, by_arm))})
    return EXIT_OK


def _score_command(args: argparse.Namespace) -> int:
    try:
        report = score_replay(
            _read_jsonl(args.replay), read_json(args.verdicts), read_json(args.baseline_report),
        )
    except (ReplayFileError, UnreadableInput, WorksheetMismatch, VerdictError) as exc:
        logger.error('invalid inputs, nothing is written (%s): %s', type(exc).__name__, exc)
        return EXIT_INVALID
    write_files_atomically({Path(args.out): json.dumps(report, indent=2, sort_keys=True) + '\n'})
    return EXIT_OK


async def _run(
    args: argparse.Namespace, llm_factory: LlmFactory | None, reader_factory: ReaderFactory | None,
) -> int:
    if args.command == 'replay':
        return await _replay_command(
            args, llm_factory or _live_llm_factory, reader_factory or live_reader_factory(args),
        )
    if args.command == 'score':
        return _score_command(args)
    return _worksheet_command(args)


def run(
    argv: Sequence[str] | None = None, *,
    llm_factory: LlmFactory | None = None, reader_factory: ReaderFactory | None = None,
) -> int:
    """Parse *argv* (default ``sys.argv[1:]``) and run one subcommand.

    *llm_factory* maps each selected arm's ExtractionLlm (model None = the
    configured llm.model, effort None = the configured client) to an
    extraction client (default: build_arm_llm over the loaded config);
    *reader_factory* maps a graph name to a ReadOnlyGraphReader (default:
    FalkorDB over GRAPH.RO_QUERY).
    """
    return asyncio.run(_run(_build_parser().parse_args(argv), llm_factory, reader_factory))


def main() -> int:
    logging.basicConfig(level=logging.INFO, stream=sys.stderr, format='%(levelname)s %(name)s: %(message)s')
    return run()


if __name__ == '__main__':
    sys.exit(main())
