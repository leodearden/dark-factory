"""The replay engine: an arm's corpus episodes onto its scratch graph, aborting per INV-4.

Each item is one ``add_episode`` on the arm's ``evalmem_`` graph. It is measured for
duration and LLM tokens and journaled through the write journal. Failures are counted
in completion order. ``MAX_CONSECUTIVE_FAILURES`` in a row abort the run: dispatch
stops, in-flight work is cancelled and listed, and the result is incomplete
(PRD §Contract, failure/storm rule).
"""

import asyncio
import uuid
from collections import Counter
from collections.abc import Awaitable, Callable, Sequence
from contextlib import AbstractAsyncContextManager
from datetime import datetime
from typing import Any, NamedTuple, Protocol

from graphiti_core.nodes import EpisodeType

from fused_memory.arm_harness.arm_spec import LlmArmSpec
from fused_memory.arm_harness.replay_types import (
    ArmAbort,
    ArmRunResult,
    EpisodeOutcome,
    ReplayItem,
    ReplaySettings,
    normalize_entity_name,
)
from fused_memory.arm_harness.scratch_guard import GuardCheckpoint, require_scratch_name
from fused_memory.backends.llm_token_usage import LlmTokenUsage, TokenMeasurement

MAX_CONSECUTIVE_FAILURES = 5
OVER_BUDGET_ERROR_CLASS = 'EpisodeOverBudget'
JOURNAL_SOURCE = 'arm_harness'
JOURNAL_OPERATION = 'arm_replay_episode'


class ArmGraph(Protocol):
    """The slice of GraphitiBackend a replay and its retrieval probe use."""

    async def add_episode(
        self,
        *,
        name: str,
        content: str,
        source: EpisodeType,
        group_id: str,
        source_description: str,
        reference_time: datetime | None,
    ) -> Any: ...

    async def search(
        self, query: str, group_ids: list[str] | None = None, num_results: int = 10
    ) -> list[Any]: ...

    def token_probe(self) -> AbstractAsyncContextManager[TokenMeasurement]: ...


class ReplayJournal(Protocol):
    """The slice of WriteJournal a replay writes through."""

    async def log_write_op(
        self,
        *,
        write_op_id: str,
        source: str,
        operation: str,
        project_id: str | None,
        params: dict | None,
        success: bool,
        error: str | None,
    ) -> None: ...

    async def log_backend_op(
        self,
        *,
        write_op_id: str | None,
        backend: str,
        operation: str,
        payload: dict | None,
        success: bool,
        error: str | None,
        duration_ms: float | None,
        llm_tokens: LlmTokenUsage | None,
    ) -> None: ...


class ConsecutiveFailureBreaker:
    """Trips once ``limit`` failures arrive in a row; any success resets the streak."""

    def __init__(self, *, arm_id: str, limit: int = MAX_CONSECUTIVE_FAILURES) -> None:
        if limit < 1:
            raise ValueError(f'breaker limit must be >= 1, got {limit}')
        self._arm_id = arm_id
        self._limit = limit
        self._streak: list[tuple[str, str]] = []

    def observe(self, episode_id: str, error_class: str | None) -> ArmAbort | None:
        if error_class is None:
            self._streak.clear()
            return None
        self._streak.append((episode_id, error_class))
        if len(self._streak) < self._limit:
            return None
        return ArmAbort(
            arm_id=self._arm_id,
            item_ids=tuple(episode_id for episode_id, _ in self._streak),
            error_classes=tuple(error_class for _, error_class in self._streak),
        )


async def replay_arm(
    spec: LlmArmSpec,
    items: Sequence[ReplayItem],
    *,
    graph: ArmGraph,
    journal: ReplayJournal,
    settings: ReplaySettings,
) -> ArmRunResult:
    require_scratch_name(spec.scratch_group_id, checkpoint=GuardCheckpoint.REPLAY)
    _require_unique_episode_ids(items)

    async def run_one(item: ReplayItem) -> EpisodeOutcome:
        outcome = await _replay_one(item, spec.scratch_group_id, graph, settings)
        await _journal(journal, spec, item, outcome)
        return outcome

    breaker = ConsecutiveFailureBreaker(arm_id=spec.arm_id)
    outcomes, cancelled_ids, abort = await _dispatch(items, run_one, breaker, settings.concurrency)
    return ArmRunResult(
        arm_id=spec.arm_id, outcomes=outcomes, cancelled_ids=cancelled_ids, abort=abort
    )


def _require_unique_episode_ids(items: Sequence[ReplayItem]) -> None:
    counts = Counter(item.episode_id for item in items)
    duplicates = sorted(episode_id for episode_id, count in counts.items() if count > 1)
    if duplicates:
        raise ValueError(f'replay items repeat episode ids: {duplicates}')


_Finished = asyncio.Queue[EpisodeOutcome | Exception]


async def _dispatch(
    items: Sequence[ReplayItem],
    run_one: Callable[[ReplayItem], Awaitable[EpisodeOutcome]],
    breaker: ConsecutiveFailureBreaker,
    concurrency: int,
) -> tuple[tuple[EpisodeOutcome, ...], tuple[str, ...], ArmAbort | None]:
    """Run items at most ``concurrency`` at a time, feeding the breaker in completion order."""
    pending = iter(items)
    finished: _Finished = asyncio.Queue()
    in_flight: dict[str, asyncio.Task[None]] = {}

    def launch_next() -> None:
        item = next(pending, None)
        if item is not None:
            in_flight[item.episode_id] = asyncio.create_task(_report(run_one, item, finished))

    for _ in range(concurrency):
        launch_next()
    outcomes: list[EpisodeOutcome] = []
    abort: ArmAbort | None = None
    try:
        while in_flight and abort is None:
            outcome = _raise_if_error(await finished.get())
            del in_flight[outcome.episode_id]
            outcomes.append(outcome)
            abort = breaker.observe(outcome.episode_id, outcome.error_class)
            if abort is None:
                launch_next()
    finally:
        cancelled_ids = await _cancel(in_flight)
    outcomes.extend(_drain(finished))
    return tuple(outcomes), cancelled_ids, abort


async def _report(
    run_one: Callable[[ReplayItem], Awaitable[EpisodeOutcome]],
    item: ReplayItem,
    finished: _Finished,
) -> None:
    try:
        finished.put_nowait(await run_one(item))
    except Exception as error:  # a harness defect, not an episode failure: surface it
        finished.put_nowait(error)


def _raise_if_error(result: EpisodeOutcome | Exception) -> EpisodeOutcome:
    if isinstance(result, Exception):
        raise result
    return result


def _drain(finished: _Finished) -> list[EpisodeOutcome]:
    """Outcomes of items that completed between the abort and their cancellation."""
    return [_raise_if_error(finished.get_nowait()) for _ in range(finished.qsize())]


async def _cancel(in_flight: dict[str, asyncio.Task[None]]) -> tuple[str, ...]:
    for task in in_flight.values():
        task.cancel()
    if in_flight:
        await asyncio.wait(in_flight.values())
    return tuple(episode_id for episode_id, task in in_flight.items() if task.cancelled())


class _GraphFacts(NamedTuple):
    replay_episode_uuid: str
    entity_names: tuple[str, ...]
    edge_triples: tuple[tuple[str, str, str], ...]


async def _replay_one(
    item: ReplayItem, group_id: str, graph: ArmGraph, settings: ReplaySettings
) -> EpisodeOutcome:
    measurement = TokenMeasurement()
    facts: _GraphFacts | None = None
    error_class: str | None = None
    started = settings.clock()
    try:
        async with graph.token_probe() as measurement:
            result = await graph.add_episode(
                name=item.name,
                content=item.content,
                source=EpisodeType.text,
                group_id=group_id,
                source_description=item.source_description,
                reference_time=item.reference_time,
            )
        facts = _graph_facts(result)
    except Exception as error:
        error_class = type(error).__name__
    duration_ms = (settings.clock() - started) * 1000
    if error_class is None and duration_ms > settings.episode_timeout_s * 1000:
        error_class = OVER_BUDGET_ERROR_CLASS
    return _outcome(item, error_class, duration_ms, measurement.usage, facts)


def _graph_facts(result: Any) -> _GraphFacts:
    names_by_uuid = {node.uuid: node.name for node in result.nodes}

    def endpoint(node_uuid: str) -> str:
        return normalize_entity_name(names_by_uuid.get(node_uuid, node_uuid))

    return _GraphFacts(
        replay_episode_uuid=result.episode.uuid,
        entity_names=tuple(sorted({normalize_entity_name(node.name) for node in result.nodes})),
        edge_triples=tuple(sorted({
            (endpoint(edge.source_node_uuid), edge.name, endpoint(edge.target_node_uuid))
            for edge in result.edges
        })),
    )


def _outcome(
    item: ReplayItem,
    error_class: str | None,
    duration_ms: float,
    tokens: LlmTokenUsage | None,
    facts: _GraphFacts | None,
) -> EpisodeOutcome:
    kept = facts if error_class is None else None
    return EpisodeOutcome(
        episode_id=item.episode_id,
        ok=error_class is None,
        error_class=error_class,
        duration_ms=duration_ms,
        tokens=tokens,
        replay_episode_uuid=kept.replay_episode_uuid if kept else None,
        entity_names=kept.entity_names if kept else (),
        edge_triples=kept.edge_triples if kept else (),
    )


async def _journal(
    journal: ReplayJournal, spec: LlmArmSpec, item: ReplayItem, outcome: EpisodeOutcome
) -> None:
    write_op_id = str(uuid.uuid4())
    await journal.log_write_op(
        write_op_id=write_op_id,
        source=JOURNAL_SOURCE,
        operation=JOURNAL_OPERATION,
        project_id=spec.scratch_group_id,
        params={'arm_id': spec.arm_id, 'episode_id': item.episode_id},
        success=outcome.ok,
        error=outcome.error_class,
    )
    await journal.log_backend_op(
        write_op_id=write_op_id,
        backend='graphiti',
        operation='add_episode',
        payload={'episode_id': item.episode_id},
        success=outcome.ok,
        error=outcome.error_class,
        duration_ms=outcome.duration_ms,
        llm_tokens=outcome.tokens,
    )
