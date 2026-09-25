"""SQLite-backed durable write queue for Graphiti operations.

Replaces the in-memory QueueService with crash-safe persistence, retry with
exponential backoff, dead-lettering, and per-group worker pools.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import random
import re
import time
from collections.abc import Callable, Coroutine, Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

import aiosqlite
from shared.async_sqlite_base import apply_full_durability_pragmas, connect_daemon

logger = logging.getLogger(__name__)

# Maximum number of ids per DELETE…RETURNING batch.  SQLite's legacy
# SQLITE_MAX_VARIABLE_NUMBER default is 999 (modern builds allow 32766);
# keeping this well below 999 leaves headroom for the trailing group_id
# placeholder and ensures correctness across all SQLite versions.
_DELETE_DEAD_BATCH_SIZE = 500

# Exception class names (matched by walking type(exc).__mro__, so subclasses
# match too) treated as transient and granted the extended
# transient_max_attempts retry budget instead of the plain max_attempts.
# Name-based matching avoids a hard dependency on graphiti_core from this
# generic SQLite write-queue component. Keep in sync with
# config.schema.QueueConfig.transient_error_names' default.
#
# THE NOT-FOUND FAMILY IS DELIBERATELY ABSENT (task 3585, esc-3561-3).
# 'NodeNotFoundError' / 'EdgeNotFoundError' / 'EdgesNotFoundError' were added
# here by task 1936 on the hypothesis of a "node-visibility race". Task 3585
# removed all three on two independent lines of evidence, which do NOT age the
# same way — keep them distinct:
#
#   * Empirical, and topology-independent — PERMANENT: across all 28 recorded
#     add_episode calls there were 304 execution attempts and 0 successes; one
#     was retried 55 times across replays and never converged. A visibility
#     race converges. 1936's own motivating incident (dead-letter id 8533) was
#     in fact the self-referential add_episode uuid bug that task 3561 fixes —
#     not a race at all. EdgesNotFoundError additionally has zero raise sites
#     anywhere in graphiti_core, so it could never have matched.
#
#   * Topology-dependent, AND THIS HALF EXPIRES: graphiti_core's by-uuid
#     lookups pass routing_='r', which can land on a lagging follower only on
#     a CLUSTERED, read-routed backend. We run a single FalkorDB container
#     with no replicas, and every Graphiti write for a group_id serialises
#     under MemoryService's per-group identity lock, so that window is
#     ~nil here.
#
# REINSTATEMENT CONDITION. If Graphiti ever moves to a clustered or
# read-routed backend, the second argument lapses and an extended budget for
# the not-found family becomes defensible again — re-derive it from the
# empirical half rather than restoring the names by reflex. The removal is a
# judgement about the CURRENT deployment, not a fact settled for all time.
#
# REINSTATEMENT NOW TAKES TWO EDITS, NOT ONE (task 3586). Re-adding names here
# is no longer sufficient, because _classify_failure checks its payload-derived
# permanent rule BEFORE the transient one: a self-referential not-found dies at
# attempt 1 whatever this set says. That rule could itself mis-fire under the
# very topology that would justify reinstatement — attempt 1 creates the
# episodic node and then fails downstream; attempt 2's by-uuid lookup
# (routing_='r') lands on a lagging follower and raises NodeNotFoundError naming
# the item's OWN uuid, which is genuinely transient but reads as the permanent
# proof. Disabling it means passing identity_payload_keys={} to
# DurableWriteQueue, and that is a CODE CHANGE today, not a config edit: the
# kwarg is deliberately not a QueueConfig field (see
# DEFAULT_IDENTITY_PAYLOAD_KEYS below) and MemoryService's construction site
# (memory_service.py:1190) does not pass it. Whoever reinstates the budget must
# wire that kwarg through in the same change, or the reinstatement is a no-op
# for exactly the failures it was meant to cover.
#
# ABSENT HERE MEANS "NOT EXTENDED", NOT "FAIL FAST" — and the empirical half
# above argues the stronger claim. Dropping these names only returns them to
# the plain max_attempts (5) ceiling, so a permanently-unsatisfiable write
# still spends five backend round-trips plus exponential backoff proving what
# attempt 1 already showed. 3585 deliberately changed only WHICH errors get
# the extended budget.
#
# MAKING IT TERMINAL ON ATTEMPT 1 LANDED SEPARATELY, as task 3586: see
# _classify_failure below, which took the payload-aware route rather than the
# blunter "terminal error names" counterpart 3585 offered as a fallback. It
# does NOT act on this set or on any name list — it fires only when the
# not-found names the very uuid the operation exists to create, a proof derived
# from the item's own payload. So the two mechanisms stay independent: editing
# the names here cannot switch the permanent rule on or off, and 3586's rule is
# checked FIRST precisely so re-adding a name here cannot hand a
# provably-doomed write the extended budget back.
DEFAULT_TRANSIENT_ERROR_NAMES = frozenset({
    'TimeoutError',
    'ConnectionError',
    'ConnectionResetError',
    'ServerDisconnectedError',
    'OperationalError',
})

# graphiti_core's not-found message format, pinned to 0.28.2:
#   errors.py:54-59  NodeNotFoundError(uuid) -> f'node {uuid} not found'
#   errors.py:22-27  EdgeNotFoundError(uuid) -> f'edge {uuid} not found'
#
# The uuid is recovered by PARSING because those exceptions expose no .uuid
# attribute, no group_id and no node label — the message is the only carrier.
# This module deliberately does not import graphiti_core (the same independence
# rationale as DEFAULT_TRANSIENT_ERROR_NAMES' name-based matching above), so the
# format is encoded here as a regex and pinned against the real class in
# tests/test_durable_queue_selfref_classifier.py, which builds its expectation
# from str(NodeNotFoundError(u)) rather than a copied literal.
#
# FAIL-OPEN INVARIANT. The pattern is fully anchored, so any message that is not
# EXACTLY this shape returns None, which routes the caller back to the ordinary
# retry policy. An upstream reword therefore degrades to today's retry
# behaviour and NEVER to permanent failure — the asymmetry that matters, since a
# missed permanent failure costs a few extra retries while a false one discards
# a write. fused-memory's own unrelated NodeNotFoundError
# (backends/graphiti_client.py:219) is a live instance of this: none of its
# messages match, so all of them fail open.
#
# ANCHORED WITH \Z, NOT $. Python's $ also matches immediately before a single
# trailing newline, so 'node <uuid> not found\n' would match and be read as
# PROOF of permanent failure — the fail-CLOSED direction the invariant above
# forbids. \Z matches only at the true end of the string. graphiti_core's
# f-strings emit no trailing newline today, so this costs nothing and closes the
# one hole in the "fully anchored" claim rather than leaving it merely asserted.
_NOT_FOUND_MESSAGE_RE = re.compile(r'^(?:node|edge) (\S+) not found\Z')


def _parse_not_found_uuid(message: str) -> str | None:
    """Return the uuid named by a graphiti_core not-found message, else None.

    None means "this is not a message I recognise" — the caller must fall back
    to its ordinary policy rather than draw any conclusion from the absence.
    """
    match = _NOT_FOUND_MESSAGE_RE.match(message)
    return match.group(1) if match else None


# operation -> the payload key holding THE UUID THAT OPERATION CREATES.
#
# Deliberately NOT "any uuid appearing in the payload". A payload may
# legitimately REFERENCE other nodes' uuids, and a not-found naming one of those
# is genuinely retryable: the node may live in a different graph (for FalkorDB
# the group_id IS the database, and graphiti's by-uuid lookup Cypher carries no
# group_id predicate), or a concurrent write creating it may still be in flight.
# Those keep their full retry budget. Only the uuid this operation exists to
# CREATE licenses the conclusion that retrying cannot possibly help.
#
# Verified vocabulary (the queue has exactly four operation strings, dispatched
# by MemoryService._execute_durable_write at memory_service.py:1402-1410):
#   add_episode           — 'uuid' is minted by the producer as its OWN fresh
#                           identity (memory_service.py:2516), stamped onto the
#                           payload (:2534) and forwarded to the backend
#                           (:2248). The self-referential case.
#   add_memory_graphiti   — no 'uuid' key at all.
#   mem0_classify_and_add — no uuid key.
#   mem0_add              — legacy, no live producer.
# ('_write_op_id' / '_causation_id' are write-journal ids, not graph uuids.)
#
# COUPLING TO TASK 3561, which is in flight and changes exactly this key: its
# fix stops handing graphiti a caller-minted uuid (passing None so the library
# takes its CREATE branch rather than its LOAD branch). If it lands and the key
# disappears, this rule simply falls open and every add_episode failure returns
# to the ordinary policy — which is correct. This classifier is the generic
# safety net for the NEXT operation that references its own not-yet-created
# uuid, not a fix for that one instance.
#
# Not a QueueConfig field: the config surface exists for operator tuning,
# whereas this is a fact about the payload vocabulary that an operator has no
# basis to retune. It is overridable via the constructor only — the test seam,
# and the property that keeps this a generic component.
DEFAULT_IDENTITY_PAYLOAD_KEYS: Mapping[str, str] = MappingProxyType({
    'add_episode': 'uuid',
})

# -- Schema ------------------------------------------------------------------

_CREATE_TABLE = """\
CREATE TABLE IF NOT EXISTS write_queue (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    group_id    TEXT    NOT NULL,
    operation   TEXT    NOT NULL,
    payload     TEXT    NOT NULL,
    callback_type TEXT,
    status      TEXT    NOT NULL DEFAULT 'pending',
    attempts    INTEGER NOT NULL DEFAULT 0,
    max_attempts INTEGER NOT NULL DEFAULT 5,
    next_retry_at REAL  NOT NULL DEFAULT 0,
    created_at  REAL    NOT NULL,
    completed_at REAL,
    error       TEXT,
    -- executed (task 4116): did a backend write for this item land, in any
    -- attempt; domain on DurableWriteQueue.get_dead_items. Bare NULLable and
    -- LAST on purpose: ALTER can only append (see QueueItem), and a DEFAULT
    -- would backfill legacy rows with a false "never landed".
    executed    INTEGER
);
"""

_CREATE_INDEX = """\
CREATE INDEX IF NOT EXISTS idx_wq_status_group
    ON write_queue (status, group_id, next_retry_at);
"""


# -- Data class ---------------------------------------------------------------

class QueueItem:
    """Lightweight representation of a row."""

    # Order must match write_queue's column order: __init__ unpacks a
    # `SELECT *` row positionally, so a mismatch shifts every field silently.
    # TestExecutedColumnSchema pins the two together.
    __slots__ = (
        'id', 'group_id', 'operation', 'payload', 'callback_type',
        'status', 'attempts', 'max_attempts', 'next_retry_at',
        'created_at', 'completed_at', 'error', 'executed',
    )

    def __init__(self, row: aiosqlite.Row | tuple):
        (
            self.id, self.group_id, self.operation, self.payload,
            self.callback_type, self.status, self.attempts, self.max_attempts,
            self.next_retry_at, self.created_at, self.completed_at, self.error,
            self.executed,
        ) = row

    def parsed_payload(self) -> dict[str, Any]:
        return json.loads(self.payload)


# -- Queue --------------------------------------------------------------------

CallbackFn = Callable[[str, Any, dict[str, Any]], Coroutine[Any, Any, None]]

# (write_op_id, terminal_status, error) -> None. Invoked once per item when it
# reaches a TERMINAL state ('completed' or 'dead'), never on an intermediate
# retry. An injected callback rather than a WriteJournal reference, preserving
# this module's deliberate independence from fused-memory-specific components.
TerminalHookFn = Callable[[str, str, str | None], Coroutine[Any, Any, None]]


@dataclass(frozen=True)
class DeadLetterEvent:
    """Everything an operator alarm needs about one permanently-abandoned write.

    A frozen dataclass rather than positional hook arguments: the event has
    eight fields with no natural order, and a consumer that had to remember
    which position held ``operation`` versus ``group_id`` would be a meaningful
    string in disguise (structured data, not positional convention). Frozen
    because the hook runs after the item's state is already committed — there
    is nothing a consumer could usefully mutate, and a mutation would only
    diverge the alarm from the row it describes.

    ``attempts`` is the COMMITTED count (the value now on the row), not the
    pre-increment count the claimed item carried, so it matches both the
    ``write_queue`` row and the dead-letter WARN line.

    ``post_execute`` is the structured form of ``POST_EXECUTE_DEAD_PREFIX``:
    True means the backend write LANDED and only the post-execute work kept
    failing, so a blind replay DUPLICATES it. The prefix is still applied to
    ``error`` — nothing about the journal contract changes — but a consumer
    branching on remediation should read this flag rather than re-parse it.
    """

    item_id: int
    group_id: str
    operation: str
    attempts: int
    error: str | None
    write_op_id: str | None
    payload: dict[str, Any] | None
    post_execute: bool


# (event) -> None. Invoked once per item that reaches 'dead', and never on
# 'completed' or an intermediate retry. Injected exactly as ``on_terminal`` is,
# keeping this module free of any fused-memory-specific import.
DeadLetterHookFn = Callable[[DeadLetterEvent], Coroutine[Any, Any, None]]

# Prefix applied to the reported error when an item dead-letters after a
# backend write for it LANDED, in this attempt or an earlier one — i.e. the
# registered callback (or the completion commit) is what kept failing, not the
# backend write. 'dead' alone means only "the queue exhausted its attempts"; it
# does NOT imply the write never happened, and blind-replaying such an item
# DUPLICATES it. Reported so the two cases are separable in whatever the hook
# writes them to. The fact itself is the row's `executed` column, reported by
# DurableWriteQueue.get_dead_items.
POST_EXECUTE_DEAD_PREFIX = (
    'post-execute failure (the backend write LANDED; do not blind-replay): '
)


class DurableWriteQueue:
    """SQLite WAL-backed write queue with per-group workers and global semaphore."""

    def __init__(
        self,
        *,
        data_dir: str | Path,
        execute_write: Callable[..., Coroutine[Any, Any, Any]],
        workers_per_group: int = 3,
        semaphore_limit: int = 20,
        max_attempts: int = 5,
        retry_base_seconds: float = 5.0,
        retry_max_delay_seconds: float = 300.0,
        write_timeout_seconds: float = 120.0,
        transient_max_attempts: int | None = None,
        transient_error_names: Iterable[str] | None = None,
        identity_payload_keys: Mapping[str, str] | None = None,
        on_terminal: TerminalHookFn | None = None,
        on_dead_letter: DeadLetterHookFn | None = None,
    ):
        self._data_dir = Path(data_dir)
        self._execute_write = execute_write
        self._on_terminal = on_terminal
        self._on_dead_letter = on_dead_letter
        self._workers_per_group = workers_per_group
        self._max_attempts = max_attempts
        self._retry_base_seconds = retry_base_seconds
        self._retry_max_delay_seconds = retry_max_delay_seconds
        self._write_timeout_seconds = write_timeout_seconds
        self._transient_max_attempts = (
            transient_max_attempts if transient_max_attempts is not None
            else max(max_attempts, 12)
        )
        self._transient_error_names = (
            frozenset(transient_error_names) if transient_error_names is not None
            else DEFAULT_TRANSIENT_ERROR_NAMES
        )
        # An explicit map REPLACES the default rather than extending it, so a
        # caller can express "no operation has a self-identity" as {}.
        self._identity_payload_keys: Mapping[str, str] = (
            dict(identity_payload_keys) if identity_payload_keys is not None
            else DEFAULT_IDENTITY_PAYLOAD_KEYS
        )

        self._semaphore = asyncio.Semaphore(semaphore_limit)
        self._db: aiosqlite.Connection | None = None
        self._callbacks: dict[str, CallbackFn] = {}
        self._group_events: dict[str, asyncio.Event] = {}
        self._group_locks: dict[str, asyncio.Lock] = {}
        self._worker_tasks: dict[str, list[asyncio.Task]] = {}
        self._closed = False

    # -- lifecycle ------------------------------------------------------------

    async def initialize(self) -> None:
        self._data_dir.mkdir(parents=True, exist_ok=True)
        db_path = self._data_dir / 'write_queue.db'
        self._db = await connect_daemon(str(db_path))
        self._db.row_factory = aiosqlite.Row
        await apply_full_durability_pragmas(self._db, busy_timeout_ms=5000)
        # Table, then in-place migrate, then indexes — matching ticket_store's
        # ordering, so a migration-added column is present before any index
        # that might reference it.
        await self._db.execute(_CREATE_TABLE)
        await self._migrate()
        await self._db.execute(_CREATE_INDEX)
        await self._db.commit()
        # Recover any items left in_flight from a previous crash
        await self._recover_in_flight()
        # Spin up workers for groups that have pending work
        await self._start_workers_for_pending_groups()
        logger.info('DurableWriteQueue initialized at %s', db_path)

    async def _migrate(self) -> None:
        """Add columns that post-date a DB's creation.

        ``CREATE TABLE IF NOT EXISTS`` is a no-op against an existing table, so
        additive changes need an explicit ALTER for DBs already on disk.
        Idempotent: probes ``PRAGMA table_info`` first.

        No ``PRAGMA user_version`` ladder — the project's rule (stated in
        orchestrator/run_store.py) is that purely additive changes use this
        light feature-detect idiom.
        """
        assert self._db is not None
        cursor = await self._db.execute('PRAGMA table_info(write_queue)')
        cols = {row[1] for row in await cursor.fetchall()}
        if 'executed' not in cols:
            try:
                await self._db.execute(
                    'ALTER TABLE write_queue ADD COLUMN executed INTEGER'
                )
                logger.info(
                    'DurableWriteQueue: migrated write_queue — added executed column'
                )
            except Exception as exc:
                # Multiple processes can open this same DB file, so two
                # initialize() calls can race between the probe and the ALTER.
                # Losing that race is benign; anything else is not.
                if 'duplicate column name' not in str(exc).lower():
                    raise
                logger.debug(
                    'DurableWriteQueue: executed column already exists (concurrent init)'
                )

    async def close(self) -> None:
        self._closed = True
        # Cancel all workers
        for tasks in self._worker_tasks.values():
            for t in tasks:
                t.cancel()
        # Wait for them to finish
        all_tasks = [t for tasks in self._worker_tasks.values() for t in tasks]
        if all_tasks:
            await asyncio.gather(*all_tasks, return_exceptions=True)
        self._worker_tasks.clear()
        if self._db:
            # Final TRUNCATE checkpoint so the next open sees an empty WAL.
            with contextlib.suppress(Exception):
                await self._db.execute('PRAGMA wal_checkpoint(TRUNCATE)')
            await self._db.close()
            self._db = None
        logger.info('DurableWriteQueue closed')

    async def checkpoint(self) -> tuple[int, int, int]:
        """``PRAGMA wal_checkpoint(TRUNCATE)`` → ``(busy, log, checkpointed)``.
        Called by the periodic loop in ``server/main.py``."""
        if self._db is None:
            return (-1, -1, -1)
        cursor = await self._db.execute('PRAGMA wal_checkpoint(TRUNCATE)')
        row = await cursor.fetchone()
        if row is None:
            return (-1, -1, -1)
        return int(row[0]), int(row[1]), int(row[2])

    # -- callbacks ------------------------------------------------------------

    def register_callback(self, name: str, fn: CallbackFn) -> None:
        self._callbacks[name] = fn

    # -- enqueue --------------------------------------------------------------

    async def enqueue(
        self,
        group_id: str,
        operation: str,
        payload: dict[str, Any],
        callback_type: str | None = None,
    ) -> int:
        """Persist a write item and signal workers. Returns item id."""
        assert self._db is not None
        now = time.time()
        cursor = await self._db.execute(
            # `executed` = 0 is a RECORDED negative, distinct from a legacy
            # row's NULL (unknown); see get_dead_items.
            'INSERT INTO write_queue '
            '(group_id, operation, payload, callback_type, status, attempts, '
            ' max_attempts, next_retry_at, created_at, executed) '
            'VALUES (?, ?, ?, ?, ?, 0, ?, 0, ?, 0)',
            (group_id, operation, json.dumps(payload), callback_type,
             'pending', self._max_attempts, now),
        )
        await self._db.commit()
        item_id = cursor.lastrowid
        self._ensure_workers(group_id)
        self._signal_group(group_id)
        return item_id  # type: ignore[return-value]

    async def enqueue_batch(
        self, items: list[dict[str, Any]]
    ) -> list[int]:
        """Bulk insert in a single transaction. Each dict needs group_id,
        operation, payload, and optionally callback_type."""
        assert self._db is not None
        now = time.time()
        ids: list[int] = []
        groups_seen: set[str] = set()
        await self._db.execute('BEGIN')
        try:
            for item in items:
                cursor = await self._db.execute(
                    # Explicit `executed = 0` for the same reason as enqueue().
                    'INSERT INTO write_queue '
                    '(group_id, operation, payload, callback_type, status, attempts, '
                    ' max_attempts, next_retry_at, created_at, executed) '
                    'VALUES (?, ?, ?, ?, ?, 0, ?, 0, ?, 0)',
                    (item['group_id'], item['operation'],
                     json.dumps(item['payload']),
                     item.get('callback_type'),
                     'pending', self._max_attempts, now),
                )
                ids.append(cursor.lastrowid)  # type: ignore[arg-type]
                groups_seen.add(item['group_id'])
            await self._db.commit()
        except Exception:
            await self._db.rollback()
            raise
        for g in groups_seen:
            self._ensure_workers(g)
            self._signal_group(g)
        return ids

    # -- worker pool ----------------------------------------------------------

    def _ensure_workers(self, group_id: str) -> None:
        """Spawn workers for group_id if not already running."""
        if self._closed:
            return
        if group_id not in self._group_events:
            self._group_events[group_id] = asyncio.Event()
        if group_id not in self._group_locks:
            self._group_locks[group_id] = asyncio.Lock()
        existing = self._worker_tasks.get(group_id, [])
        # Clean up completed tasks
        alive = [t for t in existing if not t.done()]
        needed = self._workers_per_group - len(alive)
        for _ in range(needed):
            task = asyncio.create_task(
                self._worker_loop(group_id), name=f'dq-worker-{group_id}'
            )
            alive.append(task)
        self._worker_tasks[group_id] = alive

    def _signal_group(self, group_id: str) -> None:
        ev = self._group_events.get(group_id)
        if ev:
            ev.set()

    async def _worker_loop(self, group_id: str) -> None:
        event = self._group_events[group_id]
        while not self._closed:
            item = await self._claim_next(group_id)
            if item is None:
                event.clear()
                # Short poll interval so retries with backoff are picked up promptly
                with contextlib.suppress(TimeoutError):
                    await asyncio.wait_for(event.wait(), timeout=0.5)
                continue
            await self._process_item(item)

    async def _claim_next(self, group_id: str) -> QueueItem | None:
        """Claim the next pending/retry item for this group.

        Uses a per-group asyncio.Lock so only one worker claims at a time.
        """
        assert self._db is not None
        lock = self._group_locks[group_id]
        async with lock:
            now = time.time()
            cursor = await self._db.execute(
                "SELECT * FROM write_queue "
                "WHERE group_id = ? AND status IN ('pending', 'retry') "
                "  AND next_retry_at <= ? "
                "ORDER BY id ASC LIMIT 1",
                (group_id, now),
            )
            row = await cursor.fetchone()
            if row is None:
                return None
            item = QueueItem(tuple(row))
            await self._db.execute(
                "UPDATE write_queue SET status = 'in_flight' WHERE id = ?",
                (item.id,),
            )
            await self._db.commit()
            return item

    async def _process_item(self, item: QueueItem) -> None:
        """Execute the write, handle success/failure.

        Callbacks run *before* marking completed so that a callback
        failure triggers retry instead of being silently lost.

        Both post-commit hooks, by contrast, run AFTER the queue's own commit
        and OUTSIDE the semaphore: ``_notify_terminal`` (the durable write-back
        onto the ``write_ops`` row) and then, for a dead item only,
        ``_notify_dead_letter`` (the operator alarm).
        """
        terminal: tuple[str, str | None] | None = None
        write_op_id: str | None = None
        # Seeded from the row, since an earlier attempt may already have
        # landed. A NULL (unknown) row seeds False: the prefix asserts a
        # landing, so it needs positive evidence.
        executed = bool(item.executed)
        async with self._semaphore:
            try:
                # Parsed ONCE, and the join key captured BEFORE dispatch:
                # _execute_write pops '_write_op_id' (and '_causation_id' /
                # 'temporal_context') off the dict it is handed. Parsing inside
                # the try keeps a malformed payload routed to _handle_failure
                # exactly as before, with write_op_id left None so the hook is
                # skipped.
                payload = item.parsed_payload()
                write_op_id = payload.get('_write_op_id')
                result = await asyncio.wait_for(
                    self._execute_write(item.operation, payload),
                    timeout=self._write_timeout_seconds,
                )
                # The backend write LANDED. Anything that fails below is a
                # post-execute failure, which is a materially different fact
                # for anyone deciding whether a dead item is safe to replay.
                # Made durable BEFORE the callback, because the callback is
                # what can fail and schedule a retry.
                executed = True
                await self._mark_executed(item)
                # Fire callback before marking completed — failure retries item.
                # A FRESH parse, deliberately not `payload`: _execute_write pops
                # journal metadata that the callbacks read back out.
                if item.callback_type and item.callback_type in self._callbacks:
                    await self._callbacks[item.callback_type](
                        item.callback_type, result, item.parsed_payload()
                    )
                await self._mark_completed(item)
                terminal = ('completed', None)
            except Exception as exc:
                terminal = await self._handle_failure(item, exc, executed=executed)

        if terminal is not None:
            status, error = terminal
            if status == 'dead' and executed:
                error = f'{POST_EXECUTE_DEAD_PREFIX}{error}'
            await self._notify_terminal(item.id, write_op_id, status, error)
            if status == 'dead':
                # AFTER the journal write-back, deliberately: the durable audit
                # trail must land before the best-effort alarm gets a chance to
                # misbehave.
                await self._notify_dead_letter(
                    item, write_op_id, error, post_execute=executed
                )

    async def _notify_terminal(
        self,
        item_id: int,
        write_op_id: str | None,
        status: str,
        error: str | None,
    ) -> None:
        """Report a terminal outcome to the injected ``on_terminal`` hook.

        Deliberately runs after the item's own durable state is committed and
        outside ``self._semaphore``: the queue's correctness must NEVER depend
        on the hook succeeding (a raising hook would otherwise flip a landed
        write back to retry, turning an audit-trail improvement into a
        correctness regression), and the hook's own work must not hold a slot
        in a pool shared across every group.

        A raising hook is logged and swallowed. A payload with no
        ``_write_op_id`` (``replay_from_store``, ``mem0_classify_and_add``) or
        one that would not parse is skipped silently: there is nothing to join
        back to.
        """
        if self._on_terminal is None or not write_op_id:
            return
        try:
            await self._on_terminal(write_op_id, status, error)
        except Exception:
            logger.warning(
                'Item %d: on_terminal hook failed for write_op %s (%s)',
                item_id, write_op_id, status, exc_info=True,
            )

    async def _mark_executed(self, item: QueueItem) -> None:
        """Durably record that this item's backend write LANDED — best-effort.

        A failure is logged and swallowed, never raised. Raising would
        reschedule the item, and the retry would re-execute a write that has
        already landed: the duplicate this flag exists to prevent. The fact is
        not lost with it: ``_handle_failure`` records it again in the commit
        that settles a failed attempt, and only a failed attempt's row is ever
        read for it. It goes unrecorded only if the process dies before that
        commit.
        """
        if item.executed:
            return
        assert self._db is not None
        try:
            await self._db.execute(
                'UPDATE write_queue SET executed = 1 WHERE id = ?',
                (item.id,),
            )
            await self._db.commit()
        except Exception:
            logger.warning(
                'Item %d (%s, group_id=%s): backend write landed but the '
                'executed flag could not be persisted; continuing without a '
                'retry — if this attempt fails, its failure commit records it',
                item.id, item.operation, item.group_id, exc_info=True,
            )

    async def _notify_dead_letter(
        self,
        item: QueueItem,
        write_op_id: str | None,
        error: str | None,
        *,
        post_execute: bool,
    ) -> None:
        """Report a permanently-abandoned write to the ``on_dead_letter`` hook.

        Shares ``_notify_terminal``'s post-commit, outside-the-semaphore
        discipline for the same two reasons — the queue's correctness must
        never depend on a hook, and a hook's own work must not hold a slot in a
        pool shared across every group — and DIVERGES from it on exactly one
        point, deliberately: there is no ``or not write_op_id`` guard here.

        That guard is right for the journal write-back, which has nothing to
        join an outcome back to without a key. It is wrong for an alarm, which
        needs no join key at all — and inheriting it would silently exempt
        every ``mem0_classify_and_add`` (one per extracted fact per episode)
        and every ``replay_from_store`` from the only push signal they have.
        Those deaths currently reach nothing but a WARNING log.

        A raising hook is logged and swallowed. Unlike the journal write-back,
        a raise here would not merely lose one record: it escapes
        ``_process_item`` into ``_worker_loop``, which has no handler, so the
        worker task dies and the group stops draining. A failed alarm must cost
        the operator a heads-up, never the queue.
        """
        if self._on_dead_letter is None:
            return
        try:
            payload: dict[str, Any] | None = item.parsed_payload()
        except (ValueError, TypeError):
            # A payload that will not parse still has to raise the alarm — the
            # unparseable payload is itself part of what went wrong.
            payload = None
        event = DeadLetterEvent(
            item_id=item.id,
            group_id=item.group_id,
            operation=item.operation,
            # The committed count: _handle_failure wrote item.attempts + 1.
            attempts=item.attempts + 1,
            error=error,
            write_op_id=write_op_id,
            payload=payload,
            # Structured, so the consumer never re-parses POST_EXECUTE_DEAD_PREFIX.
            post_execute=post_execute,
        )
        try:
            await self._on_dead_letter(event)
        except Exception:
            logger.warning(
                'Item %d (%s, group_id=%s): on_dead_letter hook failed; the '
                'dead-letter is committed but was NOT escalated',
                item.id, item.operation, item.group_id, exc_info=True,
            )

    async def _mark_completed(self, item: QueueItem) -> None:
        assert self._db is not None
        await self._db.execute(
            "UPDATE write_queue SET status = 'completed', completed_at = ?, "
            "attempts = attempts + 1 WHERE id = ?",
            (time.time(), item.id),
        )
        await self._db.commit()

    def _is_transient(self, exc: BaseException) -> bool:
        """Whether *exc* (or any class in its MRO) is a known-transient error
        that should receive the extended ``transient_max_attempts`` retry
        budget instead of the plain ``max_attempts`` ceiling.
        """
        return any(c.__name__ in self._transient_error_names for c in type(exc).__mro__)

    def _identity_uuid(self, item: QueueItem) -> str | None:
        """The uuid *item*'s operation exists to CREATE, or None if it has none.

        Fails open at every step — an unmapped operation, an unparseable or
        non-dict payload, or a missing/None/non-str/empty value all yield None,
        which returns the caller to its ordinary retry policy. None means "no
        identity could be established", never "no identity exists".
        """
        key = self._identity_payload_keys.get(item.operation)
        if key is None:
            return None
        try:
            payload = item.parsed_payload()
        except (ValueError, TypeError):
            return None
        if not isinstance(payload, dict):
            return None
        value = payload.get(key)
        if isinstance(value, str) and value:
            return value
        return None

    def _classify_failure(self, item: QueueItem, exc: BaseException) -> tuple[str, int]:
        """Classify *exc* for *item* and return (classification, attempts_limit).

        The three outcomes:

        * ``('permanent', 1)`` — a not-found naming THIS item's own identity
          uuid. When the missing node is the node this operation exists to
          create, no number of retries can succeed; that is derived from local
          state, not guessed from an error type. A limit of 1 (rather than a
          separate boolean) keeps ``died = new_attempts >= limit`` a single
          uniform expression at the call site, and also correctly re-kills an
          item arriving with ``attempts > 0``. That case is produced by
          ``_recover_in_flight``: crash recovery flips ``in_flight`` back to
          ``pending`` while PRESERVING the attempt count. (``replay_dead`` is
          not that path — it explicitly zeroes ``attempts``. What it does supply
          is re-execution of the identical poison payload, which is how the
          esc-3561-3 item accumulated 55 attempts across successive replays:
          five per replay, from a counter reset each time.)
        * ``('transient', transient_max_attempts)`` — the existing name-based
          policy, unchanged.
        * ``('normal', item.max_attempts)`` — everything else, today's default.

        PERMANENT IS CHECKED FIRST, DELIBERATELY. ``transient_error_names`` is
        operator-tunable and task 3585 removed the not-found family only from
        the DEFAULT set — config validation still permits an operator to add
        names back. Were transient checked first, re-adding 'NodeNotFoundError'
        would hand a provably-doomed self-referential write the extended budget,
        resurrecting the very loop this classifier exists to kill. A proof
        derived from the payload outranks a guess about a class name.

        Both gates fail open independently: an unrecognised message shape and an
        unresolvable identity each fall through to the ordinary policy.
        """
        identity = self._identity_uuid(item)
        if identity is not None and _parse_not_found_uuid(str(exc)) == identity:
            return ('permanent', 1)
        if self._is_transient(exc):
            return ('transient', self._transient_max_attempts)
        return ('normal', item.max_attempts)

    async def _handle_failure(
        self, item: QueueItem, exc: Exception, *, executed: bool
    ) -> tuple[str, str | None] | None:
        """Dead-letter or schedule a retry.

        Returns ``('dead', error_msg)`` when the item reached its terminal
        dead state, or ``None`` when it was merely rescheduled — the caller
        uses that to decide whether to fire the terminal hook.

        *executed* (a backend write for this item has landed) is recorded in
        the same commit, because ``_mark_executed`` is best-effort. The CASE
        only ever sets the flag: an attempt that did not land leaves the
        column as it was, so a legacy NULL stays unknown.
        """
        assert self._db is not None
        new_attempts = item.attempts + 1
        error_msg = f'{type(exc).__name__}: {exc}'
        classification, limit = self._classify_failure(item, exc)
        died = new_attempts >= limit
        if died:
            await self._db.execute(
                "UPDATE write_queue SET status = 'dead', attempts = ?, error = ?, "
                "executed = CASE WHEN ? THEN 1 ELSE executed END WHERE id = ?",
                (new_attempts, error_msg, executed, item.id),
            )
            # operation / group_id / classification are what a triager needs
            # first, and the log line is all that survives once the queue row is
            # deleted — after that, get_dead_items can no longer supply them.
            # The classification separates "doomed from attempt 1" from
            # "exhausted its budget", which the attempt count alone cannot when
            # max_attempts is 1.
            logger.warning(
                'Item %d (%s, group_id=%s) dead-lettered after %d attempts [%s]: %s',
                item.id, item.operation, item.group_id, new_attempts,
                classification, error_msg,
            )
        else:
            delay = min(
                self._retry_base_seconds * (2 ** (new_attempts - 1))
                + random.uniform(0, self._retry_base_seconds),
                self._retry_max_delay_seconds,
            )
            next_retry = time.time() + delay
            await self._db.execute(
                "UPDATE write_queue SET status = 'retry', attempts = ?, "
                "next_retry_at = ?, error = ?, "
                "executed = CASE WHEN ? THEN 1 ELSE executed END WHERE id = ?",
                (new_attempts, next_retry, error_msg, executed, item.id),
            )
            # Kept symmetrical with the dead-letter line above so a retry storm
            # is attributable to an operation and a project without a second
            # lookup.
            logger.info(
                'Item %d (%s, group_id=%s) retry %d/%d in %.1fs [%s]: %s',
                item.id, item.operation, item.group_id, new_attempts, limit,
                delay, classification, error_msg,
            )
        await self._db.commit()
        return ('dead', error_msg) if died else None

    # -- recovery -------------------------------------------------------------

    async def _recover_in_flight(self) -> None:
        """Reset items left in_flight (crashed mid-write) back to pending."""
        assert self._db is not None
        cursor = await self._db.execute(
            "UPDATE write_queue SET status = 'pending' WHERE status = 'in_flight'"
        )
        await self._db.commit()
        if cursor.rowcount:
            logger.info('Recovered %d in-flight items to pending', cursor.rowcount)

    async def _start_workers_for_pending_groups(self) -> None:
        """Start workers for any groups that have pending/retry items."""
        assert self._db is not None
        cursor = await self._db.execute(
            "SELECT DISTINCT group_id FROM write_queue "
            "WHERE status IN ('pending', 'retry')"
        )
        rows = await cursor.fetchall()
        for row in rows:
            group_id = row[0] if isinstance(row, tuple) else row['group_id']
            self._ensure_workers(group_id)
            self._signal_group(group_id)

    # -- management -----------------------------------------------------------

    async def replay_dead(self, group_id: str | None = None) -> int:
        """Reset dead items to pending for retry. Returns count reset.

        Resets the retry BUDGET (attempts, next_retry_at) and clears the last
        error. Deliberately does NOT clear ``executed``: that flag is sticky
        for the life of the row. "A backend write for this item landed at some
        point" does not stop being true because an operator pressed replay —
        and it is exactly the fact that makes a SECOND blind replay dangerous.
        Read it before replaying; the rule per value is on get_dead_items.
        """
        assert self._db is not None
        if group_id:
            cursor = await self._db.execute(
                "UPDATE write_queue SET status = 'pending', attempts = 0, "
                "next_retry_at = 0, error = NULL "
                "WHERE status = 'dead' AND group_id = ?",
                (group_id,),
            )
        else:
            cursor = await self._db.execute(
                "UPDATE write_queue SET status = 'pending', attempts = 0, "
                "next_retry_at = 0, error = NULL "
                "WHERE status = 'dead'",
            )
        await self._db.commit()
        count = cursor.rowcount or 0
        if count:
            if group_id:
                self._ensure_workers(group_id)
                self._signal_group(group_id)
            else:
                await self._start_workers_for_pending_groups()
        return count

    async def get_stats(self, group_id: str | None = None) -> dict[str, Any]:
        """Return counts by status, oldest pending age, and dead-by-operation.

        ``dead_by_operation`` maps operation name -> count over ``status='dead'``
        rows only.  A nonzero entry means writes of that operation have been
        PERMANENTLY abandoned: the queue exhausted their attempts and gave up,
        after the caller was already told the write had been accepted.  It is
        always present, and ``{}`` when nothing is dead — a probe must never
        have to distinguish "no deaths" from "an older server".

        This counter is the health-probe CONFIRMATION, not the primary alarm.
        The push signal is the ``durable_write_dead_letter`` escalation
        (``middleware/dead_letter_escalator.py::emit_dead_letter_escalation``),
        which survives cleanup; this reads the live ``write_queue`` table, so
        it returns to zero once :mcp-tool:`delete_dead_letters` sweeps the rows.
        ``counts['dead']`` is the same population without the attribution, so
        the two always sum consistently.

        Args:
            group_id: When given, restrict counts, oldest-pending age and the
                dead-by-operation breakdown to rows whose ``group_id``
                matches.  Default ``None`` returns unscoped (global)
                statistics — preserving the behaviour required by
                :mcp-tool:`get_queue_stats` and the dashboard.
        """
        assert self._db is not None

        if group_id is not None:
            cursor = await self._db.execute(
                'SELECT status, COUNT(*) as cnt FROM write_queue '
                'WHERE group_id = ? GROUP BY status',
                (group_id,),
            )
        else:
            cursor = await self._db.execute(
                'SELECT status, COUNT(*) as cnt FROM write_queue GROUP BY status'
            )
        rows = await cursor.fetchall()
        counts = {
            row[0] if isinstance(row, tuple) else row['status']:
            row[1] if isinstance(row, tuple) else row['cnt']
            for row in rows
        }

        oldest_pending_age = None
        if group_id is not None:
            cursor = await self._db.execute(
                "SELECT MIN(created_at) FROM write_queue "
                "WHERE status IN ('pending', 'retry') AND group_id = ?",
                (group_id,),
            )
        else:
            cursor = await self._db.execute(
                "SELECT MIN(created_at) FROM write_queue "
                "WHERE status IN ('pending', 'retry')"
            )
        row = await cursor.fetchone()
        if row:
            min_created = row[0] if isinstance(row, tuple) else row[0]
            if min_created is not None:
                oldest_pending_age = time.time() - min_created

        # No new index: idx_wq_status_group is on (status, group_id,
        # next_retry_at), so both spellings below seek the status='dead'
        # prefix — and the scoped one seeks (status, group_id). An index is
        # not free on a live DB; see write_journal.py's idx_wo_created note,
        # where adding one measured ~47 s of one-time startup DDL.
        if group_id is not None:
            cursor = await self._db.execute(
                'SELECT operation, COUNT(*) as cnt FROM write_queue '
                "WHERE status = 'dead' AND group_id = ? GROUP BY operation",
                (group_id,),
            )
        else:
            cursor = await self._db.execute(
                'SELECT operation, COUNT(*) as cnt FROM write_queue '
                "WHERE status = 'dead' GROUP BY operation"
            )
        rows = await cursor.fetchall()
        dead_by_operation = {
            row[0] if isinstance(row, tuple) else row['operation']:
            row[1] if isinstance(row, tuple) else row['cnt']
            for row in rows
        }

        return {
            'counts': counts,
            'oldest_pending_age_seconds': oldest_pending_age,
            'dead_by_operation': dead_by_operation,
        }

    async def delete_dead(
        self,
        group_id: str,
        ids: list[int],
    ) -> dict[str, Any]:
        """Delete specific dead-lettered items by id.

        Only rows with ``status='dead'`` that belong to ``group_id`` are
        eligible.  Cross-project ids, non-existent ids, and non-dead-status
        ids all land in ``not_found`` without leaking information.

        Large id lists are processed internally in chunks of
        :data:`_DELETE_DEAD_BATCH_SIZE` so callers can pass arbitrarily many
        ids without hitting SQLite's SQLITE_MAX_VARIABLE_NUMBER limit.

        **Success envelope**::

            {'deleted': [<sorted ids removed>], 'not_found': [<sorted ids missed>]}

        **Transient-error envelope** (returned, not raised, on
        ``aiosqlite.OperationalError`` — e.g. database is locked, disk full)::

            {
                'error':      '<original exception message>',
                'error_type': 'TransientSqliteError',
                'retriable':  True,
                'deleted':    [<ids durably deleted in prior chunks>],
                'not_found':  [<ineligible ids discovered in prior chunks>],
                'remaining':  [<ids in the failing chunk and all later chunks>],
            }

        ``remaining`` contains only ids that were *never attempted* — it
        excludes ineligible ids already classified in prior chunks.  Retrying
        with ``ids=remaining`` is therefore safe and non-redundant.

        If the recovery ``COMMIT`` after the error also fails (e.g. disk still
        full), ``deleted`` and ``not_found`` are set to ``[]`` and ``remaining``
        covers all input ids, so a full retry is safe.

        Programmer-bug exceptions (``ProgrammingError``, ``IntegrityError``)
        are NOT caught and propagate to the caller.

        Args:
            group_id: Project scope — only dead rows in this group are deleted.
            ids: Integer row ids to delete.  Any size list is accepted.

        Returns:
            Success or transient-error envelope as described above.
        """
        if not ids:
            return {'deleted': [], 'not_found': []}

        assert self._db is not None
        deleted: set[int] = set()
        not_found_completed: set[int] = set()

        for i in range(0, len(ids), _DELETE_DEAD_BATCH_SIZE):
            chunk = ids[i : i + _DELETE_DEAD_BATCH_SIZE]
            placeholders = ','.join('?' * len(chunk))
            # Single atomic statement — SELECT + DELETE in one round-trip (SQLite >=3.35).
            # The WHERE guards (status='dead', group_id=?) are enforced atomically so
            # a concurrent replay or worker cannot slip a row past the eligibility check
            # between a separate SELECT and DELETE.
            try:
                cursor = await self._db.execute(
                    f"DELETE FROM write_queue "
                    f"WHERE id IN ({placeholders}) AND status='dead' AND group_id=? "
                    f"RETURNING id",
                    (*chunk, group_id),
                )
                rows = await cursor.fetchall()
            except aiosqlite.OperationalError as exc:
                # Commit prior-chunk deletions durably before returning.
                # The commit itself can fail (e.g. disk still full), in which case
                # we cannot guarantee prior deletions landed — return a conservative
                # envelope covering all inputs so a full retry is safe.
                try:
                    await self._db.commit()
                except aiosqlite.OperationalError:
                    logger.exception(
                        'delete_dead: recovery commit failed after OperationalError;'
                        ' reporting full input as remaining',
                    )
                    return {
                        'error': str(exc),
                        'error_type': 'TransientSqliteError',
                        'retriable': True,
                        'deleted': [],
                        'not_found': [],
                        'remaining': sorted(ids),
                    }
                return {
                    'error': str(exc),
                    'error_type': 'TransientSqliteError',
                    'retriable': True,
                    'deleted': sorted(deleted),
                    'not_found': sorted(not_found_completed),
                    'remaining': sorted(ids[i:]),
                }
            chunk_deleted = {
                (row[0] if isinstance(row, tuple) else row['id']) for row in rows
            }
            deleted |= chunk_deleted
            # Subtract `deleted` at point-of-update so the invariant holds
            # throughout the loop — any future early-return path is safe.
            not_found_completed |= (set(chunk) - chunk_deleted) - deleted

        await self._db.commit()

        return {
            'deleted': sorted(deleted),
            'not_found': sorted(not_found_completed),
        }

    async def get_dead_items(
        self,
        group_id: str | None = None,
        *,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        """Return dead-lettered items, newest-first.

        Each item carries ``executed``, the fact a replay decision turns on,
        over a THREE-valued domain:

        * ``True`` — a backend write for this item landed; replaying it
          DUPLICATES that write.
        * ``False`` — the queue recorded that no backend write landed, so the
          item is safe to replay.
        * ``None`` — unknown: the row predates the column (task 4116). Check
          ``backend_ops`` (joined on the payload's ``_write_op_id``) before
          replaying. POST_EXECUTE_DEAD_PREFIX on the error reported to
          ``on_terminal`` can confirm a landing (``error`` here never carries
          it), but its absence proves nothing: before 4116 the prefix was
          recomputed per attempt and was lost whenever a landed item retried.

        ``None`` is FALSY, so ``if not row['executed']`` is the wrong test —
        it reads "unknown" as "safe". Only an explicit ``is False`` licenses a
        replay.

        Args:
            group_id: Optional filter by group_id.
            limit: Optional maximum number of items to return.  When *None*
                all dead items are returned (backward-compatible default).
        """
        assert self._db is not None
        if group_id:
            sql = (
                "SELECT * FROM write_queue WHERE status = 'dead' AND group_id = ?"
                " ORDER BY id DESC"
            )
            params: tuple = (group_id,)
        else:
            sql = "SELECT * FROM write_queue WHERE status = 'dead' ORDER BY id DESC"
            params = ()

        if limit is not None:
            sql += ' LIMIT ?'
            params = (*params, limit)

        cursor = await self._db.execute(sql, params)
        rows = await cursor.fetchall()
        results = []
        for row in rows:
            item = QueueItem(tuple(row))
            results.append({
                'id': item.id,
                'group_id': item.group_id,
                'operation': item.operation,
                'payload': item.parsed_payload(),
                'attempts': item.attempts,
                'error': item.error,
                'created_at': item.created_at,
                'executed': None if item.executed is None else bool(item.executed),
            })
        return results
