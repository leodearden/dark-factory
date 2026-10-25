"""The ruling-overreach guard's Stage S shadow (design.md §6.1, task 6137).

After a ``decisions_and_rationale`` write releases its identity lock, a judge
labels the edges the episode minted and records every label to a SQLite
sidecar, ``<queue.data_dir>/overreach_shadow.db``. It writes nothing to the
graph and filters nothing. Recon reads the overreach flags back as Stage 1's
O4 list, acknowledging each page once its run completes, and
scripts/eval_overreach_judge.py reads the labels for the online gate.
"""
from __future__ import annotations

import asyncio
import dataclasses
import json
import logging
from collections import defaultdict
from collections.abc import Awaitable, Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any, Protocol

from shared.async_sqlite_base import (
    AtomicConnection,
    CheckpointResult,
    apply_full_durability_pragmas,
    connect_daemon,
)

from fused_memory.backends.graphiti_client import build_llm_client
from fused_memory.config.schema import FusedMemoryConfig, OverreachGuardConfig
from fused_memory.models.enums import MemoryCategory
from fused_memory.services.overreach_guard import (
    OVERREACH,
    PROMPT_VERSION,
    EdgeJudgement,
    JudgeEdge,
    StructuredLLM,
    judge_edges,
    matching_classifiers,
    minted_edges,
)

logger = logging.getLogger(__name__)

SIDECAR_FILENAME = 'overreach_shadow.db'
_ERROR_CHARS = 500
_CONCURRENT_JUDGES = 2

_DDL = (
    """CREATE TABLE IF NOT EXISTS judged_episodes (
        graph TEXT NOT NULL,
        episode_uuid TEXT NOT NULL,
        judged_at TEXT NOT NULL,
        judge_model TEXT NOT NULL,
        prompt_version TEXT NOT NULL,
        classifiers TEXT NOT NULL,
        minted_count INTEGER NOT NULL,
        outcome TEXT NOT NULL,
        error TEXT,
        PRIMARY KEY (graph, episode_uuid)
    )""",
    """CREATE TABLE IF NOT EXISTS edge_judgements (
        graph TEXT NOT NULL,
        edge_uuid TEXT NOT NULL,
        episode_uuid TEXT NOT NULL,
        fact TEXT NOT NULL,
        label TEXT NOT NULL,
        rationale TEXT NOT NULL,
        judged_at TEXT NOT NULL,
        PRIMARY KEY (graph, edge_uuid)
    )""",
    'CREATE INDEX IF NOT EXISTS idx_edge_judgements_flags ON edge_judgements (graph, label, judged_at)',
    """CREATE TABLE IF NOT EXISTS flag_handoffs (
        graph TEXT NOT NULL,
        edge_uuid TEXT NOT NULL,
        handed_at TEXT NOT NULL,
        PRIMARY KEY (graph, edge_uuid)
    )""",
)


class ShadowOutcome(StrEnum):
    judged = 'judged'
    no_minted = 'no_minted'
    failed = 'failed'


@dataclass(frozen=True)
class ShadowJob:
    graph: str
    episode_uuid: str
    content: str
    edge_uuids: tuple[str, ...]
    classifiers: tuple[str, ...]


@dataclass(frozen=True)
class JudgedEpisode:
    graph: str
    episode_uuid: str
    judge_model: str
    prompt_version: str
    classifiers: tuple[str, ...]
    minted_count: int
    outcome: ShadowOutcome
    error: str | None = None


@dataclass(frozen=True)
class StoredJudgement:
    episode_uuid: str
    label: str
    rationale: str
    judge_model: str
    prompt_version: str
    judged_at: datetime


@dataclass(frozen=True)
class OverreachFlag:
    episode_uuid: str
    edge_uuid: str
    fact: str
    rationale: str
    judged_at: datetime
    prompt_version: str


@dataclass(frozen=True)
class OverreachFlagPage:
    """One cycle's slice of *graph*'s flags not yet handed over.

    ``flags`` go to Stage 1. ``dropped`` names the edges read for this page
    that are gone or retired. ``omitted`` counts the flags left for a later page.
    """

    graph: str
    flags: tuple[OverreachFlag, ...]
    omitted: int
    dropped: tuple[str, ...] = ()

    def handed_over(self) -> tuple[str, ...]:
        """The edge uuids this page disposes of once its run completes."""
        return (*(flag.edge_uuid for flag in self.flags), *self.dropped)


def _to_stored_instant(instant: datetime) -> str:
    """Fixed-width UTC ISO text, so a stored instant compares lexically as an instant."""
    aware = instant if instant.tzinfo else instant.replace(tzinfo=UTC)
    return aware.astimezone(UTC).isoformat(timespec='microseconds')


def _from_stored_instant(text: str) -> datetime:
    return datetime.fromisoformat(text)


def _utc_now() -> datetime:
    return datetime.now(UTC)


# --------------------------------------------------------------------------- #
# The sidecar store
# --------------------------------------------------------------------------- #

_PENDING_FLAGS_SQL = """
SELECT j.episode_uuid, j.edge_uuid, j.fact, j.rationale, j.judged_at, e.prompt_version,
       COUNT(*) OVER () AS total
FROM edge_judgements j
JOIN judged_episodes e ON e.graph = j.graph AND e.episode_uuid = j.episode_uuid
LEFT JOIN flag_handoffs h ON h.graph = j.graph AND h.edge_uuid = j.edge_uuid
WHERE j.graph = ? AND j.label = ? AND h.edge_uuid IS NULL
ORDER BY j.judged_at, j.edge_uuid
LIMIT ?
"""

_JUDGEMENTS_SQL = """
SELECT j.edge_uuid, j.episode_uuid, j.label, j.rationale, e.judge_model, e.prompt_version,
       j.judged_at
FROM edge_judgements j
JOIN judged_episodes e ON e.graph = j.graph AND e.episode_uuid = j.episode_uuid
WHERE j.graph = ? AND j.edge_uuid IN ({marks})
"""


class OverreachShadowStore:
    """The sidecar: one episode row per judged write, one edge row per judged edge.

    Every access goes through AtomicConnection, because concurrent judge jobs
    and recon's flag reads share one connection (task 5562).
    """

    def __init__(self, access: AtomicConnection, *, clock: Callable[[], datetime] = _utc_now) -> None:
        self._access: AtomicConnection | None = access
        self._clock = clock

    @classmethod
    async def open(
        cls, data_dir: str | Path, *, clock: Callable[[], datetime] = _utc_now,
    ) -> OverreachShadowStore:
        directory = Path(data_dir)
        await asyncio.to_thread(directory.mkdir, parents=True, exist_ok=True)
        conn = await connect_daemon(str(directory / SIDECAR_FILENAME))
        await apply_full_durability_pragmas(conn, busy_timeout_ms=5000)
        access = AtomicConnection(conn)
        async with access.write() as db:
            for statement in _DDL:
                await db.execute(statement)
        return cls(access, clock=clock)

    @classmethod
    async def open_read_only(cls, db_path: str | Path) -> OverreachShadowStore:
        """A read-only view for the eval script; it never creates the file."""
        path = Path(db_path)
        if not await asyncio.to_thread(path.is_file):
            raise FileNotFoundError(f'no overreach shadow sidecar at {path}')
        conn = await connect_daemon(f'file:{path}?mode=ro', uri=True)
        return cls(AtomicConnection(conn))

    def _require_access(self) -> AtomicConnection:
        if self._access is None:
            raise RuntimeError('OverreachShadowStore is closed')
        return self._access

    async def close(self) -> None:
        if self._access is not None:
            access, self._access = self._access, None
            await access.close()

    async def checkpoint(self) -> CheckpointResult:
        return await AtomicConnection.checkpoint_or_unavailable(self._access)

    async def record(
        self, *, episode: JudgedEpisode, judgements: Sequence[tuple[JudgeEdge, EdgeJudgement]],
    ) -> None:
        """One write unit; INSERT OR IGNORE, so a re-judged row keeps its first judgement."""
        judged_at = _to_stored_instant(self._clock())
        async with self._require_access().write() as db:
            await db.execute(
                'INSERT OR IGNORE INTO judged_episodes VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)',
                (episode.graph, episode.episode_uuid, judged_at, episode.judge_model,
                 episode.prompt_version, json.dumps(list(episode.classifiers)),
                 episode.minted_count, episode.outcome.value, episode.error),
            )
            await db.executemany(
                'INSERT OR IGNORE INTO edge_judgements VALUES (?, ?, ?, ?, ?, ?, ?)',
                [(episode.graph, edge.uuid, episode.episode_uuid, edge.fact, judgement.label,
                  judgement.rationale, judged_at) for edge, judgement in judgements],
            )

    async def judged_episode(self, graph: str, episode_uuid: str) -> JudgedEpisode | None:
        row = await self._require_access().read_one(
            'SELECT judge_model, prompt_version, classifiers, minted_count, outcome, error '
            'FROM judged_episodes WHERE graph = ? AND episode_uuid = ?',
            (graph, episode_uuid),
        )
        if row is None:
            return None
        judge_model, prompt_version, classifiers, minted_count, outcome, error = row
        return JudgedEpisode(
            graph=graph, episode_uuid=episode_uuid, judge_model=judge_model,
            prompt_version=prompt_version, classifiers=tuple(json.loads(classifiers)),
            minted_count=minted_count, outcome=ShadowOutcome(outcome), error=error,
        )

    async def judgements_for(
        self, keys: Iterable[tuple[str, str]],
    ) -> dict[tuple[str, str], StoredJudgement]:
        """The stored judgement of each ``(graph, edge_uuid)``; an unjudged key is absent."""
        by_graph: defaultdict[str, list[str]] = defaultdict(list)
        for graph, edge_uuid in keys:
            by_graph[graph].append(edge_uuid)
        found: dict[tuple[str, str], StoredJudgement] = {}
        for graph, edge_uuids in by_graph.items():
            sql = _JUDGEMENTS_SQL.format(marks=', '.join('?' * len(edge_uuids)))
            for row in await self._require_access().read_all(sql, (graph, *edge_uuids)):
                edge_uuid, episode_uuid, label, rationale, judge_model, prompt_version, at = row
                found[(graph, edge_uuid)] = StoredJudgement(
                    episode_uuid=episode_uuid, label=label, rationale=rationale,
                    judge_model=judge_model, prompt_version=prompt_version,
                    judged_at=_from_stored_instant(at),
                )
        return found

    async def pending_overreach_flags(self, graph: str, limit: int) -> OverreachFlagPage:
        """*graph*'s overreach labels not yet handed over, oldest first, at most *limit*."""
        rows = await self._require_access().read_all(_PENDING_FLAGS_SQL, (graph, OVERREACH, limit))
        flags = tuple(
            OverreachFlag(
                episode_uuid=episode_uuid, edge_uuid=edge_uuid, fact=fact, rationale=rationale,
                judged_at=_from_stored_instant(at), prompt_version=prompt_version,
            )
            for episode_uuid, edge_uuid, fact, rationale, at, prompt_version, _ in rows
        )
        total = rows[0][-1] if rows else 0
        return OverreachFlagPage(graph=graph, flags=flags, omitted=total - len(flags))

    async def record_handoffs(self, graph: str, edge_uuids: Sequence[str]) -> None:
        """Take *edge_uuids* out of *graph*'s pending flags; the first handoff of an edge is kept."""
        if not edge_uuids:
            return
        handed_at = _to_stored_instant(self._clock())
        async with self._require_access().write() as db:
            await db.executemany(
                'INSERT OR IGNORE INTO flag_handoffs VALUES (?, ?, ?)',
                [(graph, edge_uuid, handed_at) for edge_uuid in edge_uuids],
            )


# --------------------------------------------------------------------------- #
# The shadow runner
# --------------------------------------------------------------------------- #


class EdgeReader(Protocol):
    """The surviving-edge read, GraphitiBackend.get_edges_by_uuids."""

    def __call__(
        self, uuids: Sequence[str], *, group_id: str,
    ) -> Awaitable[Sequence[Mapping[str, Any]]]: ...


def shadow_job_from_write(
    result: Any, *, group_id: str, content: str, category: MemoryCategory | None,
) -> ShadowJob | None:
    """The job for a landed write, or None when the result names no episode.

    Read defensively, as memory_service.py::_episode_uuid_of does: the write is
    already durable, so an unreadable result must cost the judgement, never
    the write.
    """
    episode_uuid = getattr(getattr(result, 'episode', None), 'uuid', None)
    if not isinstance(episode_uuid, str) or not episode_uuid:
        return None
    edge_uuids = tuple(
        uuid for edge in (getattr(result, 'edges', None) or ())
        if isinstance(uuid := getattr(edge, 'uuid', None), str)
    )
    return ShadowJob(
        graph=group_id, episode_uuid=episode_uuid, content=content, edge_uuids=edge_uuids,
        classifiers=matching_classifiers(category, content),
    )


class OverreachShadow:
    """Judges submitted writes in the background and owns the sidecar they land in."""

    def __init__(
        self,
        store: OverreachShadowStore,
        llm: StructuredLLM,
        judge_model: str,
        read_edges: EdgeReader,
        settings: Callable[[], OverreachGuardConfig],
        *,
        max_pending: int = 64,
    ) -> None:
        self.judge_model = judge_model
        self._store = store
        self._llm = llm
        self._read_edges = read_edges
        self._settings = settings
        self._max_pending = max_pending
        self._tasks: set[asyncio.Task[None]] = set()
        self._judge_slots = asyncio.Semaphore(_CONCURRENT_JUDGES)
        self._closed = False

    def submit(self, job: ShadowJob) -> bool:
        """Schedule *job*; False when it was refused. Never raises."""
        try:
            return self._admit(job)
        except Exception:
            logger.warning('overreach shadow: submit failed for %s/%s',
                           job.graph, job.episode_uuid, exc_info=True)
            return False

    def _admit(self, job: ShadowJob) -> bool:
        if self._closed or not self._settings().shadow_enabled:
            return False
        if len(self._tasks) >= self._max_pending:
            logger.warning('overreach shadow: %d judgements pending, dropping %s/%s',
                           len(self._tasks), job.graph, job.episode_uuid)
            return False
        task = asyncio.create_task(self._run(job), name=f'overreach-shadow:{job.episode_uuid}')
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)
        return True

    async def drain(self) -> None:
        """Wait for every job submitted so far."""
        if self._tasks:
            await asyncio.wait(tuple(self._tasks))

    async def overreach_flag_page(self, graph: str, limit: int) -> OverreachFlagPage:
        """The oldest *limit* flags not yet handed over, less those whose edge is gone or retired."""
        pending = await self._store.pending_overreach_flags(graph, limit)
        if not pending.flags:
            return pending
        rows = await self._read_edges([flag.edge_uuid for flag in pending.flags], group_id=graph)
        active = {row['uuid']: bool(row['active']) for row in rows}
        return dataclasses.replace(
            pending,
            flags=tuple(flag for flag in pending.flags if active.get(flag.edge_uuid)),
            dropped=tuple(flag.edge_uuid for flag in pending.flags if not active.get(flag.edge_uuid)),
        )

    async def acknowledge(self, page: OverreachFlagPage) -> None:
        """No later page offers what *page* handed over."""
        await self._store.record_handoffs(page.graph, page.handed_over())

    async def checkpoint(self) -> CheckpointResult:
        """The periodic WAL checkpoint of the sidecar this shadow owns."""
        return await self._store.checkpoint()

    async def close(self) -> None:
        """Cancel in-flight judgements, then close the store; a second call is a no-op.

        Cancel rather than drain: a judge call takes seconds and the close
        budget is half a second. A lost shadow judgement is visible to the
        online gate as an unjudged edge.
        """
        if self._closed:
            return
        self._closed = True
        pending = tuple(self._tasks)
        for task in pending:
            task.cancel()
        if pending:
            await asyncio.wait(pending)
        await self._store.close()

    async def _run(self, job: ShadowJob) -> None:
        edges: tuple[JudgeEdge, ...] = ()
        try:
            edges = minted_edges(job.episode_uuid, await self._read_edges(job.edge_uuids, group_id=job.graph))
            judged = await self._judge(job, edges) if edges else ()
            outcome, error = (ShadowOutcome.judged if edges else ShadowOutcome.no_minted), None
        except Exception as exc:
            logger.warning('overreach shadow: judging %s/%s failed',
                           job.graph, job.episode_uuid, exc_info=True)
            judged, outcome, error = (), ShadowOutcome.failed, f'{type(exc).__name__}: {exc}'[:_ERROR_CHARS]
        await self._record(JudgedEpisode(
            graph=job.graph, episode_uuid=job.episode_uuid, judge_model=self.judge_model,
            prompt_version=PROMPT_VERSION, classifiers=job.classifiers,
            minted_count=len(edges), outcome=outcome, error=error,
        ), judged)

    async def _judge(
        self, job: ShadowJob, edges: tuple[JudgeEdge, ...],
    ) -> tuple[tuple[JudgeEdge, EdgeJudgement], ...]:
        async with self._judge_slots:
            judgements = await asyncio.wait_for(
                judge_edges(self._llm, job.content, edges), self._settings().judge_timeout_seconds,
            )
        return tuple(zip(edges, judgements, strict=True))

    async def _record(
        self, episode: JudgedEpisode, judged: Sequence[tuple[JudgeEdge, EdgeJudgement]],
    ) -> None:
        try:
            await self._store.record(episode=episode, judgements=judged)
        except Exception:
            logger.warning('overreach shadow: recording %s/%s failed',
                           episode.graph, episode.episode_uuid, exc_info=True)


async def open_overreach_shadow(
    config: FusedMemoryConfig, *, read_edges: EdgeReader,
) -> OverreachShadow | None:
    """The shadow for *config*, or None when no judge LLM is configured.

    No LLM means no sidecar is opened: a shadow that cannot judge has nothing
    to serve recon. ``settings`` re-reads the shared config object per job, so
    a hot reload of ``overreach_guard`` is observed without a restart.
    """
    llm = build_llm_client(config)
    if llm is None:
        logger.warning('overreach shadow disabled: no judge LLM is configured '
                       '(the llm provider block or its api_key is missing)')
        return None
    store = await OverreachShadowStore.open(config.queue.data_dir)
    return OverreachShadow(store, llm, config.llm.model, read_edges, lambda: config.overreach_guard)
