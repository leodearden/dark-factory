"""LLM-judged task curator gate.

At task-creation time, the curator decides whether a candidate task should:

- **drop** — already covered by an existing non-cancelled task; return that task's id
  instead of creating a new one.
- **combine** — rewrite a pending task to subsume the candidate's work, producing a
  single coherent task instead of two fragments.
- **create** — genuinely new work; proceed with task creation.

The decision is made by an LLM against a layered corpus of existing tasks (review-chain
anchor + module-lock pool + embedding neighbors + dependency neighbors). See
docs/reify-task-fragmentation-report-2026-04-11.txt for the motivating analysis.

The curator is best-effort: any failure (embedder, Qdrant, LLM, taskmaster) degrades
to ``action="create"`` so task creation is never blocked.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import time
import uuid as uuid_mod
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, Literal

from shared.cli_invoke import AgentResult, invoke_with_cap_retry
from shared.locking import files_to_modules

if TYPE_CHECKING:
    from pathlib import Path

    from shared.usage_gate import UsageGate

    from fused_memory.backends.taskmaster_client import TaskmasterBackend
    from fused_memory.config.schema import FusedMemoryConfig
    from fused_memory.middleware.curator_escalator import CuratorEscalator

logger = logging.getLogger(__name__)


class CuratorFailureError(RuntimeError):
    """Raised when the curator's LLM call fails and cannot be salvaged.

    The curator no longer degrades silently to ``action='create'`` on LLM
    errors; instead the interceptor translates this into an L1 escalation or
    a hard failure at the MCP boundary so operators notice breakage.

    Attaches ``timed_out`` and ``duration_ms`` from the underlying
    ``AgentResult`` so :class:`CuratorEscalator` can surface them in the L1
    escalation detail. Defaults make the attributes safe to read even when
    the failure originates outside :meth:`TaskCurator._call_llm`.
    """

    def __init__(
        self,
        message: str,
        *,
        timed_out: bool | None = None,
        duration_ms: int | None = None,
    ) -> None:
        super().__init__(message)
        self.timed_out = timed_out
        self.duration_ms = duration_ms


Action = Literal['drop', 'combine', 'create']

_STATUS_RANK = {
    'pending': 0,
    'in-progress': 1,
    'in_progress': 1,
    'blocked': 2,
    'deferred': 3,
    'done': 4,
}
_PRIORITY_RANK = {'critical': 0, 'high': 1, 'medium': 2, 'low': 3, 'polish': 4}
DEFAULT_PRIORITY = 'medium'  # canonical fallback used by both record_task and backfill_corpus
if DEFAULT_PRIORITY not in _PRIORITY_RANK:
    raise ValueError(f'{DEFAULT_PRIORITY!r} not in _PRIORITY_RANK')

# JSON schema for the curator's structured output — used by invoke_with_cap_retry
# to constrain the LLM's response.
CURATOR_OUTPUT_SCHEMA: dict[str, Any] = {
    'type': 'object',
    'required': ['action', 'justification'],
    'properties': {
        'action': {'type': 'string', 'enum': ['drop', 'combine', 'create']},
        'target_id': {'type': ['string', 'null']},
        # Verbatim echo of the target task's title (required for combine).
        # Verified against the live target before rewriting to prevent
        # silently clobbering an unrelated task when target_id is wrong.
        'target_fingerprint': {'type': ['string', 'null']},
        'justification': {'type': 'string'},
        'rewritten_task': {
            'type': ['object', 'null'],
            'properties': {
                'title': {'type': 'string'},
                'description': {'type': 'string'},
                'details': {'type': 'string'},
                'files_to_modify': {
                    'type': 'array',
                    'items': {'type': 'string'},
                },
                'priority': {'type': 'string'},
            },
        },
    },
}


@dataclass
class CandidateTask:
    """A task about to be created, fed to the curator for a drop/combine/create decision."""

    title: str
    description: str = ''
    details: str = ''
    files_to_modify: list[str] = field(default_factory=list)
    priority: str = DEFAULT_PRIORITY
    spawned_from: str | None = None  # task id of the review-chain anchor
    spawn_context: str = 'manual'  # review | steward-triage | expand | parse_prd | manual

    def payload_hash(self) -> str:
        """Stable hash over fields the curator actually reads.

        Two candidates with the same hash get the same cached decision — this
        prevents combine-on-combine drift when a creation is retried.
        """
        h = hashlib.sha256()
        h.update(self.title.encode())
        h.update(b'\x00')
        h.update(self.description.encode())
        h.update(b'\x00')
        h.update(self.details.encode())
        h.update(b'\x00')
        h.update('\n'.join(sorted(self.files_to_modify)).encode())
        h.update(b'\x00')
        h.update((self.spawned_from or '').encode())
        return h.hexdigest()[:16]


@dataclass
class RewrittenTask:
    """A task description produced by the combine action."""

    title: str
    description: str
    details: str
    files_to_modify: list[str]
    priority: str


@dataclass
class BackfillResult:
    """Result of a backfill_corpus() call."""

    upserted: int = 0
    skipped: int = 0
    errors: int = 0


@dataclass
class CuratorDecision:
    """Result of a curator call. Always returned — never raised."""

    action: Action
    target_id: str | None = None
    target_fingerprint: str | None = None
    rewritten_task: RewrittenTask | None = None
    justification: str = ''
    pool_sizes: dict[str, int] = field(default_factory=dict)
    latency_ms: int = 0
    cost_usd: float = 0.0

    def to_log_fields(self) -> dict[str, Any]:
        return {
            'action': self.action,
            'target_id': self.target_id,
            'target_fingerprint': self.target_fingerprint,
            'justification': self.justification[:200],
            'pool_sizes': self.pool_sizes,
            'latency_ms': self.latency_ms,
            'cost_usd': self.cost_usd,
        }


@dataclass
class _PoolEntry:
    """One task as seen by the curator — extracted from raw taskmaster data."""

    task_id: str
    title: str
    description: str
    details: str
    files_to_modify: list[str]
    module_keys: list[str]
    status: str
    priority: str
    source: str  # anchor | module | embedding | dependency
    combine_eligible: bool

    def render(self, desc_cap: int, details_cap: int) -> str:
        """Render this entry as a human-readable block for the LLM prompt."""
        desc = self.description[:desc_cap]
        if len(self.description) > desc_cap:
            desc += '…'
        det = self.details[:details_cap]
        if len(self.details) > details_cap:
            det += '…'
        files = ', '.join(self.files_to_modify) if self.files_to_modify else '(none)'
        return (
            f'[Task {self.task_id}] status={self.status} priority={self.priority} '
            f'source={self.source} combine_eligible={self.combine_eligible}\n'
            f'  title: {self.title}\n'
            f'  description: {desc}\n'
            f'  details: {det}\n'
            f'  files_to_modify: {files}'
        )


_SYSTEM_PROMPT = """\
You are the task curator for the dark-factory orchestrator. For each candidate \
task a reviewer or agent wants to create, you decide ONE of three actions:

- "drop": the candidate is already covered by an existing task in the pool. The \
  existing task may be pending, in-progress, blocked, or done — any of these mean \
  the work will happen or has happened, so creating a duplicate wastes a full \
  architect→implement→review cycle.

- "combine": the candidate should be folded into an existing PENDING task (marked \
  combine_eligible=true in the pool). You MUST rewrite the target task to coherently \
  subsume the candidate's work, producing a single unified task. Combining into a \
  non-pending task would silently drop the candidate's work because the workflow \
  has already moved past planning for that task.

- "create": genuinely new work that cannot be merged without loss. Default to \
  "create" when in doubt — a duplicated task is cheaper than a lost or garbled one.

## Combine rules (ALL must hold)

1. Coherence. The rewritten task can be described in one sentence without the word \
   "and also". If you need "plus unrelated X", it's not one task.
2. Narrow lock. The union of files_to_modify must stay within the same 1–2 \
   module-lock keys the target already touches. If combining adds a new module \
   to the lock set, refuse and "create" instead.
3. Planning budget. The rewritten task can be planned by one architect in one \
   session: ≤ 15 files_to_modify, ≤ 5 distinct concerns at bullet level, combined \
   details under ~8000 chars.
4. No critical-path drag. If the target is high-priority and on a milestone critical \
   path, the candidate must itself be critical or tightly coupled to the target's \
   correctness. Do NOT weigh critical work down with adjacent nice-to-haves.

## Positive signals (bias toward combining when the above hold)

- Either task alone is tiny (< 300 chars of real detail, or < 3 files). Tiny tasks \
  have orchestrator overhead disproportionate to the work delivered.
- The two tasks share a root cause or subsystem concept — fixing both together \
  means the implementer loads context once.
- They will serialize on the same module lock anyway (scheduler already serializes \
  tasks touching the same module key). Combining loses zero parallelism.
- They touch the same function body or adjacent lines — splitting invites a merge \
  conflict.

## Rewriting, not concatenating

When you combine, the rewritten task must be a single coherent rewrite:

- NEW title that names the unified concern (not "T and also C").
- NEW details structured as coherent bullets under that unified concern.
- PRESERVE every concrete code reference from both tasks VERBATIM: file paths, \
  line numbers, test names, function/class/symbol names. Do not paraphrase specifics \
  away — the implementer needs them.
- Union files_to_modify (verbatim, deduplicated).
- Carry the higher priority of the two.

If you cannot produce a coherent rewrite that satisfies the hard constraints above, \
the answer is "create", not a half-hearted combine.

## Output format

Return a single JSON object matching the provided schema. The `justification` field \
should briefly explain which rule triggered the decision. For "drop" and "combine", \
set `target_id` to the id of the matched pool task. For "combine", set \
`rewritten_task` to the full rewritten task fields. For "create", leave `target_id` \
and `rewritten_task` as null.

## Combine safety: target_fingerprint (required for combine)

When you choose "combine", you MUST also populate `target_fingerprint` with the \
VERBATIM title of the pool task you are targeting — copy the exact string \
immediately following `title:` from the pool entry whose id matches `target_id`. \
This fingerprint is verified against the live task before the rewrite is applied; \
if it does not match, the combine is refused and the candidate is created fresh \
instead of silently clobbering an unrelated task. Copy the title character-for-\
character — do not shorten, summarize, or reformat it. For "drop" and "create", \
leave `target_fingerprint` null.
"""


class TaskCurator:
    """LLM-judged drop/combine/create gate plus the Qdrant corpus backing it."""

    def __init__(
        self,
        config: FusedMemoryConfig,
        taskmaster: TaskmasterBackend | None = None,
        usage_gate: UsageGate | None = None,
        cwd: Path | None = None,
        escalator: 'CuratorEscalator | None' = None,
    ) -> None:
        self._config = config
        self._taskmaster = taskmaster
        self._usage_gate = usage_gate
        self._cwd = cwd
        self._escalator = escalator
        self._qdrant_client = None  # AsyncQdrantClient, lazy
        self._embedder = None  # OpenAIEmbedder, lazy
        self._initialized_collections: set[str] = set()
        # Idempotency cache: payload_hash -> (decision, monotonic_time)
        self._decision_cache: dict[str, tuple[CuratorDecision, float]] = {}
        # Pre-LLM exact-match cache: project_id -> {normalized_hash: (task_id, ts)}
        # Catches the common "two triages race to create identical tasks"
        # case cheaply — skips embedding + LLM entirely on the second call.
        self._recent_creates: dict[str, dict[str, tuple[str, float]]] = {}

    # ------------------------------------------------------------------
    # Lazy init (mirrors task_dedup.py pattern)
    # ------------------------------------------------------------------

    async def _get_qdrant(self):
        if self._qdrant_client is None:
            from qdrant_client import AsyncQdrantClient

            self._qdrant_client = AsyncQdrantClient(
                url=self._config.mem0.qdrant_url,
                timeout=30,
            )
        return self._qdrant_client

    async def _get_embedder(self):
        if self._embedder is None:
            cfg = self._config.embedder
            if cfg.provider == 'openai' and cfg.providers.openai:
                api_key = cfg.providers.openai.api_key
                if api_key:
                    from graphiti_core.embedder import OpenAIEmbedder
                    from graphiti_core.embedder.openai import OpenAIEmbedderConfig

                    self._embedder = OpenAIEmbedder(
                        config=OpenAIEmbedderConfig(
                            api_key=api_key,
                            embedding_model=cfg.model,
                            base_url=cfg.providers.openai.api_url,
                            embedding_dim=cfg.dimensions,
                        ),
                    )
            if self._embedder is None:
                raise RuntimeError(
                    'TaskCurator requires an OpenAI embedder — check config.embedder',
                )
        return self._embedder

    def _collection_name(self, project_id: str) -> str:
        return f'task_curator_{project_id}'

    async def _ensure_collection(self, project_id: str) -> str:
        name = self._collection_name(project_id)
        if name in self._initialized_collections:
            return name
        client = await self._get_qdrant()
        if not await client.collection_exists(name):
            from qdrant_client.models import Distance, VectorParams

            await client.create_collection(
                collection_name=name,
                vectors_config=VectorParams(
                    size=self._config.embedder.dimensions,
                    distance=Distance.COSINE,
                ),
            )
            logger.info('Created task curator collection: %s', name)
        self._initialized_collections.add(name)
        return name

    @staticmethod
    def _embedding_text(
        title: str, description: str, files_to_modify: list[str],
    ) -> str:
        """Text used for embedding — title + description + file list."""
        parts = [title]
        if description:
            parts.append(description)
        if files_to_modify:
            parts.append('\n'.join(files_to_modify))
        return '\n\n'.join(parts)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Pre-LLM normalized-key cache (R3)
    # ------------------------------------------------------------------

    _RECENT_CREATES_TTL_SECS = 600.0  # 10 min — long enough to catch
    # concurrent triages but short enough that stale entries don't pin
    # memory after a task is removed.

    @staticmethod
    def _normalize_key(candidate: CandidateTask) -> str:
        """Stable 16-char hash over normalised (title, files_to_modify).

        Case + whitespace insensitive on the title, file-order insensitive.
        Designed to catch "two reviewers raced with the same suggestion"
        without depending on embeddings. See plan R3 — pre-LLM
        short-circuit.
        """
        title = (candidate.title or '').strip().lower()
        files = '\n'.join(sorted(candidate.files_to_modify or []))
        h = hashlib.sha256()
        h.update(title.encode())
        h.update(b'|')
        h.update(files.encode())
        return h.hexdigest()[:16]

    def _evict_stale_recent_creates(self, project_id: str, now: float) -> None:
        bucket = self._recent_creates.get(project_id)
        if not bucket:
            return
        stale = [
            k for k, (_, ts) in bucket.items()
            if now - ts > self._RECENT_CREATES_TTL_SECS
        ]
        for k in stale:
            del bucket[k]

    async def _pre_llm_exact_match(
        self, candidate: CandidateTask, project_id: str, project_root: str,
    ) -> CuratorDecision | None:
        """Return a drop decision if a recently-created task matches
        verbatim on normalised title + file list. Otherwise return None.

        Verifies the cached task still exists (and isn't cancelled) so a
        stale cache entry doesn't redirect to a gone task.
        """
        bucket = self._recent_creates.get(project_id)
        if not bucket:
            return None
        now = time.monotonic()
        self._evict_stale_recent_creates(project_id, now)
        key = self._normalize_key(candidate)
        entry = bucket.get(key)
        if entry is None:
            return None
        cached_id, _ts = entry
        if self._taskmaster is not None:
            try:
                task = await self._taskmaster.get_task(cached_id, project_root)
            except Exception:
                task = None
            data = task.get('data') if isinstance(task, dict) else None
            if isinstance(data, dict):
                task = data
            status = (
                str(task.get('status', '')) if isinstance(task, dict) else ''
            )
            if not isinstance(task, dict) or status == 'cancelled':
                # Stale entry — drop it and fall through to full curate.
                bucket.pop(key, None)
                return None
        return CuratorDecision(
            action='drop',
            target_id=cached_id,
            justification='pre-llm-exact-match',
            pool_sizes={'anchor': 0, 'module': 0, 'embedding': 0, 'dependency': 0},
            latency_ms=0,
        )

    def note_created(
        self, project_id: str, candidate: CandidateTask, task_id: str,
    ) -> None:
        """Record a just-created task so identical concurrent candidates
        short-circuit on the pre-LLM exact-match cache.

        Call this *after* ``tm.add_task`` returns and within the
        project's add_task critical section so the next waiter sees it.
        """
        bucket = self._recent_creates.setdefault(project_id, {})
        bucket[self._normalize_key(candidate)] = (task_id, time.monotonic())

    async def curate(
        self,
        candidate: CandidateTask,
        project_id: str,
        project_root: str,
    ) -> CuratorDecision:
        """Render a drop/combine/create decision for a candidate task.

        Best-effort: any internal failure returns a ``create`` decision with the
        failure reason in ``justification``. Never raises.
        """
        start = time.monotonic()
        payload_hash = candidate.payload_hash()

        # Pre-LLM exact-match short-circuit comes FIRST. The payload_hash
        # idempotency cache below can only return drop/combine safely — a
        # cached 'create' decision is stale if the first caller has since
        # landed the task and ``note_created`` it. Running the exact-match
        # check first means the second concurrent creator sees the
        # first's recorded task even when their payload_hashes collide.
        exact = await self._pre_llm_exact_match(
            candidate, project_id, project_root,
        )
        if exact is not None:
            self._store_cache(payload_hash, exact)
            return exact

        # Idempotency check — same payload within the TTL returns the cached decision.
        cached = self._check_cache(payload_hash)
        if cached is not None:
            return cached

        try:
            pool, pool_sizes = await self._build_corpus(
                candidate, project_id, project_root,
            )
        except Exception as exc:
            logger.warning(
                'task_curator: corpus assembly failed, falling through to create: %s',
                exc,
                exc_info=True,
            )
            decision = CuratorDecision(
                action='create',
                justification=f'corpus-failed: {exc}',
                pool_sizes={'anchor': 0, 'module': 0, 'embedding': 0, 'dependency': 0},
                latency_ms=int((time.monotonic() - start) * 1000),
            )
            self._store_cache(payload_hash, decision)
            return decision

        # Render the LLM call. A genuine LLM failure raises
        # CuratorFailureError; route it through the escalator if one was
        # configured (see curator_escalator.py). If the escalator re-raises
        # (no orchestrator → interactive path) let it propagate so the MCP
        # caller sees the outage loudly instead of silently falling back.
        try:
            decision = await self._call_llm(
                candidate, pool, pool_sizes, start, project_id, project_root,
            )
        except CuratorFailureError as exc:
            if self._escalator is not None:
                # May re-raise CuratorFailureError on the interactive path.
                await self._escalator.report_failure(
                    project_root=project_root,
                    project_id=project_id,
                    justification=str(exc),
                    candidate_title=candidate.title,
                    timed_out=exc.timed_out,
                    duration_ms=exc.duration_ms,
                )
            else:
                logger.warning(
                    'task_curator: LLM failure with no escalator wired — '
                    'falling through to create: %s',
                    exc,
                )
            decision = CuratorDecision(
                action='create',
                justification='llm-error-escalated',
                pool_sizes=pool_sizes,
                latency_ms=int((time.monotonic() - start) * 1000),
            )
        except Exception as exc:
            logger.warning(
                'task_curator: LLM call failed, falling through to create: %s', exc,
            )
            decision = CuratorDecision(
                action='create',
                justification=f'llm-failed: {exc}',
                pool_sizes=pool_sizes,
                latency_ms=int((time.monotonic() - start) * 1000),
            )

        self._store_cache(payload_hash, decision)
        logger.info(
            'task_curator: decision=%s target=%s pool_sizes=%s latency_ms=%d cost_usd=%.4f',
            decision.action,
            decision.target_id,
            decision.pool_sizes,
            decision.latency_ms,
            decision.cost_usd,
        )
        return decision

    async def record_task(
        self,
        task_id: str,
        candidate: CandidateTask,
        project_id: str,
    ) -> None:
        """Upsert a task's embedding into the curator corpus.

        Called after a successful create. Fire-and-forget from the interceptor.
        """
        try:
            collection = await self._ensure_collection(project_id)
            embedder = await self._get_embedder()
            text = self._embedding_text(
                candidate.title, candidate.description, candidate.files_to_modify,
            )
            embedding = await embedder.create(text)

            from qdrant_client.models import PointStruct

            point_id = self._point_id(project_id, task_id)
            client = await self._get_qdrant()
            await client.upsert(
                collection_name=collection,
                points=[
                    PointStruct(
                        id=point_id,
                        vector=embedding,
                        payload={
                            'task_id': task_id,
                            'title': candidate.title,
                            'description': candidate.description[:1000],
                            'files_to_modify': candidate.files_to_modify or [],
                            'priority': candidate.priority or DEFAULT_PRIORITY,
                            'project_id': project_id,
                            'updated_at': datetime.now(UTC).isoformat(),
                        },
                    ),
                ],
            )
        except Exception:
            logger.warning(
                'task_curator: record_task failed for %s', task_id, exc_info=True,
            )

    async def reembed_task(
        self,
        task_id: str,
        candidate: CandidateTask,
        project_id: str,
    ) -> None:
        """Re-upsert a task's embedding after its description/details changed.

        Same deterministic point id as record_task, so this is idempotent.
        """
        await self.record_task(task_id, candidate, project_id)

    @staticmethod
    def _point_id(project_id: str, task_id: str) -> str:
        """Deterministic UUID5 point ID for a task in a project.

        Shared by record_task() and backfill_corpus() to ensure idempotent overlap.
        """
        return str(uuid_mod.uuid5(uuid_mod.NAMESPACE_URL, f'{project_id}/{task_id}'))

    async def corpus_count(self, project_id: str) -> int:
        """Return the number of points in the curator corpus for a project.

        Returns 0 if the collection does not exist or an error occurs.
        This public method encapsulates collection-existence + count logic so
        callers (e.g. TaskInterceptor) don't need to access private methods.
        """
        try:
            client = await self._get_qdrant()
            collection_name = self._collection_name(project_id)
            if not await client.collection_exists(collection_name):
                return 0
            result = await client.count(collection_name=collection_name)
            return result.count
        except Exception:
            logger.warning(
                'task_curator: corpus_count failed for project %s', project_id, exc_info=True,
            )
            return 0

    async def backfill_corpus(
        self,
        tasks: list[dict],
        project_id: str,
    ) -> BackfillResult:
        """Upsert embeddings for a flat list of existing task dicts.

        Idempotent: uses the same deterministic UUID5 point-ID scheme as
        record_task(). Re-running just re-upserts the same points.

        Args:
            tasks: Flat list of task dicts (as returned by flatten_task_tree).
            project_id: Project identifier used for the collection name and point IDs.

        Returns:
            BackfillResult with counts of upserted, skipped, and error tasks.
        """
        if not tasks:
            return BackfillResult()

        result = BackfillResult()
        collection = await self._ensure_collection(project_id)
        embedder = await self._get_embedder()
        client = await self._get_qdrant()

        from qdrant_client.models import PointStruct

        # Embed all tasks with bounded concurrency.
        sem = asyncio.Semaphore(10)
        points: list[PointStruct] = []

        async def _embed_one(task: dict) -> None:
            task_id = str(task.get('id', '') or '')
            title = str(task.get('title', '') or '').strip()
            if not title:
                result.skipped += 1
                return

            description = str(task.get('description', '') or '')
            files = _task_files(task)
            text = self._embedding_text(title, description, files)

            try:
                async with sem:
                    embedding = await embedder.create(text)
            except Exception:
                logger.warning(
                    'task_curator: backfill embed failed for task %s', task_id, exc_info=True,
                )
                result.errors += 1
                return

            point_id = self._point_id(project_id, task_id)
            points.append(
                PointStruct(
                    id=point_id,
                    vector=embedding,
                    payload={
                        'task_id': task_id,
                        'title': title,
                        'description': description[:1000],
                        'files_to_modify': files,
                        'priority': task.get('priority') or DEFAULT_PRIORITY,
                        'project_id': project_id,
                        'updated_at': datetime.now(UTC).isoformat(),
                    },
                )
            )

        await asyncio.gather(*[_embed_one(t) for t in tasks])

        if points:
            await client.upsert(collection_name=collection, points=points)
            result.upserted = len(points)

        logger.info(
            'task_curator: backfill_corpus complete project=%s upserted=%d skipped=%d errors=%d',
            project_id, result.upserted, result.skipped, result.errors,
        )
        return result

    async def close(self) -> None:
        if self._qdrant_client is not None:
            import contextlib as _ctx

            with _ctx.suppress(Exception):
                await self._qdrant_client.close()
            self._qdrant_client = None

    # ------------------------------------------------------------------
    # Idempotency cache
    # ------------------------------------------------------------------

    def _check_cache(self, payload_hash: str) -> CuratorDecision | None:
        ttl = self._config.curator.idempotency_ttl_seconds
        now = time.monotonic()
        # Lazy eviction
        stale = [
            k for k, (_, ts) in self._decision_cache.items() if now - ts > ttl
        ]
        for k in stale:
            del self._decision_cache[k]
        entry = self._decision_cache.get(payload_hash)
        if entry is None:
            return None
        decision, ts = entry
        if now - ts > ttl:
            del self._decision_cache[payload_hash]
            return None
        return decision

    def _store_cache(self, payload_hash: str, decision: CuratorDecision) -> None:
        self._decision_cache[payload_hash] = (decision, time.monotonic())

    # ------------------------------------------------------------------
    # Corpus assembly
    # ------------------------------------------------------------------

    async def _build_corpus(
        self,
        candidate: CandidateTask,
        project_id: str,
        project_root: str,
    ) -> tuple[list[_PoolEntry], dict[str, int]]:
        """Assemble the four-stream pool for the LLM prompt."""
        lock_depth = self._config.curator.lock_depth
        pool: list[_PoolEntry] = []
        seen_ids: set[str] = set()

        # Stream 1: review-chain anchor
        anchor_entry: _PoolEntry | None = None
        if candidate.spawned_from and self._taskmaster is not None:
            try:
                anchor_task = await self._taskmaster.get_task(
                    candidate.spawned_from, project_root,
                )
                anchor_entry = _to_pool_entry(
                    anchor_task, source='anchor', lock_depth=lock_depth,
                )
                if anchor_entry is not None:
                    pool.append(anchor_entry)
                    seen_ids.add(anchor_entry.task_id)
            except Exception as exc:
                logger.debug('task_curator: anchor fetch failed: %s', exc)

        # Stream 2: module-lock pool
        candidate_modules = set(
            files_to_modules(candidate.files_to_modify, depth=lock_depth),
        )
        if not candidate_modules and anchor_entry is not None:
            candidate_modules = set(anchor_entry.module_keys)

        all_tasks_flat: list[dict] = []
        if self._taskmaster is not None and candidate_modules:
            try:
                tasks_result = await self._taskmaster.get_tasks(project_root)
                all_tasks_flat = flatten_task_tree(tasks_result)
            except Exception as exc:
                logger.debug('task_curator: get_tasks failed: %s', exc)

        module_matches: list[_PoolEntry] = []
        for t in all_tasks_flat:
            tid = str(t.get('id', ''))
            if not tid or tid in seen_ids:
                continue
            status = str(t.get('status', 'unknown'))
            if status == 'cancelled':
                continue
            task_modules = set(files_to_modules(_task_files(t), depth=lock_depth))
            if not task_modules or not (task_modules & candidate_modules):
                continue
            entry = _to_pool_entry(t, source='module', lock_depth=lock_depth)
            if entry is not None:
                module_matches.append(entry)

        module_matches.sort(key=_module_sort_key)
        for entry in module_matches[: self._config.curator.pool_module_cap]:
            pool.append(entry)
            seen_ids.add(entry.task_id)

        # Stream 3: embedding neighbors
        embedding_matches: list[_PoolEntry] = []
        try:
            collection = await self._ensure_collection(project_id)
            embedder = await self._get_embedder()
            text = self._embedding_text(
                candidate.title, candidate.description, candidate.files_to_modify,
            )
            embedding = await embedder.create(text)
            client = await self._get_qdrant()
            # Over-fetch so filtering out stream 1/2 overlaps still leaves K.
            overfetch = self._config.curator.pool_embedding_cap + 20
            results = await client.query_points(
                collection_name=collection,
                query=embedding,
                limit=overfetch,
                with_payload=True,
            )
            for point in results.points:
                payload = point.payload or {}
                tid = str(payload.get('task_id', ''))
                if not tid or tid in seen_ids:
                    continue
                # We need full task data — try to fetch it, fall back to the
                # qdrant payload if taskmaster is unavailable.
                entry = await self._fetch_entry_for_neighbor(
                    tid, payload, source='embedding',
                    lock_depth=lock_depth, project_root=project_root,
                )
                if entry is None:
                    continue
                embedding_matches.append(entry)
                if len(embedding_matches) >= self._config.curator.pool_embedding_cap:
                    break
        except Exception as exc:
            logger.debug('task_curator: embedding neighbors failed: %s', exc)

        for entry in embedding_matches:
            pool.append(entry)
            seen_ids.add(entry.task_id)

        # Stream 4: dependency neighbors of the anchor
        dep_matches: list[_PoolEntry] = []
        if anchor_entry is not None and all_tasks_flat:
            anchor_deps = _task_dependencies(
                next(
                    (t for t in all_tasks_flat if str(t.get('id', '')) == anchor_entry.task_id),
                    {},
                ),
            )
            dependents = [
                str(t.get('id', '')) for t in all_tasks_flat
                if anchor_entry.task_id in _task_dependencies(t)
            ]
            candidate_ids = set(anchor_deps) | set(dependents)
            for t in all_tasks_flat:
                tid = str(t.get('id', ''))
                if tid not in candidate_ids or tid in seen_ids:
                    continue
                if str(t.get('status', '')) == 'cancelled':
                    continue
                entry = _to_pool_entry(t, source='dependency', lock_depth=lock_depth)
                if entry is not None:
                    dep_matches.append(entry)

        for entry in dep_matches[: self._config.curator.pool_dependency_cap]:
            pool.append(entry)
            seen_ids.add(entry.task_id)

        # Final cap — trim weakest entries first (embedding, then module, then dep).
        pool = _trim_pool(pool, self._config.curator.pool_total_cap)

        pool_sizes = {
            'anchor': sum(1 for e in pool if e.source == 'anchor'),
            'module': sum(1 for e in pool if e.source == 'module'),
            'embedding': sum(1 for e in pool if e.source == 'embedding'),
            'dependency': sum(1 for e in pool if e.source == 'dependency'),
        }
        return pool, pool_sizes

    async def _fetch_entry_for_neighbor(
        self,
        tid: str,
        payload: dict,
        *,
        source: str,
        lock_depth: int,
        project_root: str,
    ) -> _PoolEntry | None:
        """Fetch a full task for an embedding neighbor, fall back to payload."""
        if self._taskmaster is not None:
            try:
                task = await self._taskmaster.get_task(tid, project_root)
                entry = _to_pool_entry(task, source=source, lock_depth=lock_depth)
                if entry is not None:
                    return entry
            except Exception:
                pass
        # Fallback: build a thin entry from the qdrant payload
        return _PoolEntry(
            task_id=tid,
            title=str(payload.get('title', '')),
            description=str(payload.get('description', '')),
            details='',
            files_to_modify=list(payload.get('files_to_modify', []) or []),
            module_keys=files_to_modules(
                payload.get('files_to_modify', []) or [], depth=lock_depth,
            ),
            status='unknown',
            priority=str(payload.get('priority', DEFAULT_PRIORITY)),
            source=source,
            combine_eligible=False,  # unknown status → treat as drop-only
        )

    # ------------------------------------------------------------------
    # LLM call
    # ------------------------------------------------------------------

    async def _call_llm(
        self,
        candidate: CandidateTask,
        pool: list[_PoolEntry],
        pool_sizes: dict[str, int],
        start: float,
        project_id: str,
        project_root: str,
    ) -> CuratorDecision:
        from pathlib import Path as _Path

        cwd = self._cwd or _Path(project_root)
        user_prompt = self._build_user_prompt(candidate, pool)

        agent_result: AgentResult = await invoke_with_cap_retry(
            usage_gate=self._usage_gate,
            label=f'task-curator[{project_id}]',
            project_id=project_id,
            prompt=user_prompt,
            system_prompt=_SYSTEM_PROMPT,
            cwd=cwd,
            model=self._config.curator.model,
            # ``max_turns=1`` is incompatible with ``--json-schema`` because
            # the schema mechanism burns a tool-use turn; the CLI returns
            # ``error_max_turns`` after the schema turn, even when the
            # structured payload is already attached. Three leaves room for
            # an optional reasoning turn, the schema tool-use, and the final
            # assistant response. Schema salvage in cli_invoke.py covers
            # the boundary case.
            max_turns=3,
            max_budget_usd=self._config.curator.max_budget_usd,
            disallowed_tools=['*'],  # no tool access — this is a pure classifier
            output_schema=CURATOR_OUTPUT_SCHEMA,
            permission_mode='bypassPermissions',
            timeout_seconds=self._config.curator.timeout_seconds,
            max_cap_retries=3,
        )

        latency_ms = int((time.monotonic() - start) * 1000)
        if not agent_result.success:
            raise CuratorFailureError(
                f'curator LLM call failed: output={agent_result.output[:200]!r} '
                f'subtype={agent_result.subtype!r} turns={agent_result.turns} '
                f'timed_out={agent_result.timed_out} '
                f'duration_ms={agent_result.duration_ms} '
                f'configured_timeout_secs={self._config.curator.timeout_seconds}',
                timed_out=agent_result.timed_out,
                duration_ms=agent_result.duration_ms,
            )

        return _parse_decision(
            agent_result, pool_sizes=pool_sizes,
            latency_ms=latency_ms, pool=pool,
        )

    def _build_user_prompt(
        self, candidate: CandidateTask, pool: list[_PoolEntry],
    ) -> str:
        desc_cap = self._config.curator.entry_description_chars
        details_cap = self._config.curator.entry_details_chars

        lines: list[str] = []
        lines.append('# Candidate task (not yet created)')
        lines.append(f'  title: {candidate.title}')
        lines.append(f'  priority: {candidate.priority}')
        lines.append(f'  spawn_context: {candidate.spawn_context}')
        if candidate.spawned_from:
            lines.append(f'  spawned_from: {candidate.spawned_from}')
        if candidate.description:
            lines.append(f'  description: {candidate.description[:desc_cap]}')
        if candidate.details:
            lines.append(f'  details: {candidate.details[:details_cap]}')
        if candidate.files_to_modify:
            lines.append(
                '  files_to_modify: '
                + ', '.join(candidate.files_to_modify),
            )
        lines.append('')

        lines.append(f'# Pool ({len(pool)} tasks)')
        if not pool:
            lines.append('  (empty — no candidates; answer "create")')
        else:
            for entry in pool:
                lines.append(entry.render(desc_cap, details_cap))
                lines.append('')

        lines.append(
            'Decide drop / combine / create per the system-prompt rules. '
            'Return only the JSON object required by the schema.',
        )
        return '\n'.join(lines)


# ----------------------------------------------------------------------
# Pure helpers (module-level — easier to unit-test)
# ----------------------------------------------------------------------


def _task_files(task: dict) -> list[str]:
    """Extract files_to_modify from a raw taskmaster task dict."""
    files = task.get('files_to_modify')
    if files is None:
        # Fall back to metadata.modules, which is how reviewer/steward tasks
        # currently carry locking hints (see roles.py prompts).
        meta = task.get('metadata') or {}
        if isinstance(meta, dict):
            files = meta.get('modules') or meta.get('files_to_modify')
    if not files:
        return []
    if isinstance(files, str):
        return [files]
    return [str(f) for f in files if f]


def _task_dependencies(task: dict) -> list[str]:
    deps = task.get('dependencies') or []
    if isinstance(deps, str):
        # CSV fallback
        return [d.strip() for d in deps.split(',') if d.strip()]
    return [str(d) for d in deps if d]


def _task_metadata_spawned_from(task: dict) -> str | None:
    meta = task.get('metadata') or {}
    if isinstance(meta, dict):
        v = meta.get('spawned_from')
        if isinstance(v, str) and v:
            return v
    return None


def _to_pool_entry(
    task: dict | None, *, source: str, lock_depth: int,
) -> _PoolEntry | None:
    if not task or not isinstance(task, dict):
        return None
    tid = str(task.get('id', ''))
    if not tid:
        return None
    status = str(task.get('status', 'unknown'))
    files = _task_files(task)
    return _PoolEntry(
        task_id=tid,
        title=str(task.get('title', '')),
        description=str(task.get('description', '') or ''),
        details=str(task.get('details', '') or ''),
        files_to_modify=files,
        module_keys=files_to_modules(files, depth=lock_depth),
        status=status,
        priority=str(task.get('priority', DEFAULT_PRIORITY)),
        source=source,
        combine_eligible=(status == 'pending'),
    )


def flatten_task_tree(tasks_result: dict) -> list[dict]:
    """Walk a get_tasks response and return a flat list of task dicts."""
    out: list[dict] = []

    def _walk(items: Any) -> None:
        if not isinstance(items, list):
            return
        for t in items:
            if isinstance(t, dict):
                out.append(t)
                _walk(t.get('subtasks') or [])

    # Taskmaster get_tasks shapes vary; try common keys.
    if isinstance(tasks_result, dict):
        _walk(tasks_result.get('tasks'))
        data = tasks_result.get('data')
        if isinstance(data, dict):
            _walk(data.get('tasks'))
    return out


def _module_sort_key(entry: _PoolEntry) -> tuple[int, int, str]:
    status_rank = _STATUS_RANK.get(entry.status, 99)
    priority_rank = _PRIORITY_RANK.get(entry.priority, 99)
    return (status_rank, priority_rank, entry.task_id)


def _trim_pool(pool: list[_PoolEntry], total_cap: int) -> list[_PoolEntry]:
    """Trim a pool that exceeds the total cap, dropping the weakest first.

    Weak = dependency > embedding > module > anchor, preserving anchor always.
    """
    if len(pool) <= total_cap:
        return pool
    weakest_order = ['dependency', 'embedding', 'module', 'anchor']
    result = list(pool)
    for source in weakest_order:
        if len(result) <= total_cap:
            break
        # drop tail-most entries of this source class first
        indices = [i for i, e in enumerate(result) if e.source == source]
        # keep at least one anchor
        if source == 'anchor':
            indices = indices[1:]
        # drop from the end so earlier (higher-priority) keeps survive
        for i in reversed(indices):
            if len(result) <= total_cap:
                break
            result.pop(i)
    return result[:total_cap]


def _parse_decision(
    agent_result: AgentResult,
    *,
    pool_sizes: dict[str, int],
    latency_ms: int,
    pool: list[_PoolEntry],
) -> CuratorDecision:
    raw = agent_result.structured_output
    if raw is None:
        # Try to parse from raw output
        try:
            raw = json.loads(agent_result.output)
        except Exception as exc:
            return CuratorDecision(
                action='create',
                justification=f'parse-failed: {exc}; output={agent_result.output[:200]}',
                pool_sizes=pool_sizes,
                latency_ms=latency_ms,
                cost_usd=agent_result.cost_usd,
            )

    if not isinstance(raw, dict):
        return CuratorDecision(
            action='create',
            justification=f'parse-failed: not-a-dict ({type(raw).__name__})',
            pool_sizes=pool_sizes,
            latency_ms=latency_ms,
            cost_usd=agent_result.cost_usd,
        )

    action = str(raw.get('action', '')).lower()
    if action not in ('drop', 'combine', 'create'):
        return CuratorDecision(
            action='create',
            justification=f'invalid-action: {action!r}',
            pool_sizes=pool_sizes,
            latency_ms=latency_ms,
            cost_usd=agent_result.cost_usd,
        )

    justification = str(raw.get('justification', ''))
    target_id = raw.get('target_id')
    if target_id is not None:
        target_id = str(target_id)
    target_fingerprint = raw.get('target_fingerprint')
    if target_fingerprint is not None:
        target_fingerprint = str(target_fingerprint)

    # Safety: drop/combine must reference a pool task id that actually exists.
    valid_ids = {e.task_id for e in pool}
    if action in ('drop', 'combine'):
        if not target_id or target_id not in valid_ids:
            return CuratorDecision(
                action='create',
                justification=(
                    f'invalid-target: action={action} target_id={target_id!r}; '
                    f'not in pool'
                ),
                pool_sizes=pool_sizes,
                latency_ms=latency_ms,
                cost_usd=agent_result.cost_usd,
            )
        # combine-only tasks must also be combine_eligible (pending status).
        if action == 'combine':
            target_entry = next(e for e in pool if e.task_id == target_id)
            if not target_entry.combine_eligible:
                return CuratorDecision(
                    action='create',
                    justification=(
                        f'invalid-combine-target: {target_id} has status '
                        f'{target_entry.status}, not pending'
                    ),
                    pool_sizes=pool_sizes,
                    latency_ms=latency_ms,
                    cost_usd=agent_result.cost_usd,
                )

    rewritten: RewrittenTask | None = None
    if action == 'combine':
        rt = raw.get('rewritten_task')
        if not isinstance(rt, dict):
            return CuratorDecision(
                action='create',
                justification='combine-missing-rewrite',
                pool_sizes=pool_sizes,
                latency_ms=latency_ms,
                cost_usd=agent_result.cost_usd,
            )
        try:
            rewritten = RewrittenTask(
                title=str(rt.get('title', '')),
                description=str(rt.get('description', '')),
                details=str(rt.get('details', '')),
                files_to_modify=[str(f) for f in (rt.get('files_to_modify') or [])],
                priority=str(rt.get('priority', DEFAULT_PRIORITY)),
            )
        except Exception as exc:
            return CuratorDecision(
                action='create',
                justification=f'rewrite-parse-failed: {exc}',
                pool_sizes=pool_sizes,
                latency_ms=latency_ms,
                cost_usd=agent_result.cost_usd,
            )
        if not rewritten.title or not rewritten.details:
            return CuratorDecision(
                action='create',
                justification='rewrite-empty-title-or-details',
                pool_sizes=pool_sizes,
                latency_ms=latency_ms,
                cost_usd=agent_result.cost_usd,
            )

    return CuratorDecision(
        action=action,  # type: ignore[arg-type]
        target_id=target_id,
        target_fingerprint=target_fingerprint,
        rewritten_task=rewritten,
        justification=justification,
        pool_sizes=pool_sizes,
        latency_ms=latency_ms,
        cost_usd=agent_result.cost_usd,
    )
