"""In-memory event queue with background drainer.

Decouples MCP-tool write paths from the SQLite-backed :class:`EventBuffer`.
Callers enqueue events via a non-blocking ``put_nowait`` and return to the
MCP boundary immediately. A single background drainer task pulls events out
of the queue and pushes them into the buffer, retrying transient SQLite
failures (``aiosqlite.OperationalError``) with exponential backoff. Events
that hit non-retriable errors or overflow the queue are appended to a
dead-letter JSONL file for manual replay.

Design rationale: before WP-B, a locked ``reconciliation.db`` surfaced as
an MCP error on ``submit_task`` / ``set_task_status`` / etc., even after the
canonical tasks.json / Graphiti / Mem0 write succeeded. Agents retried and
created duplicates (2026-04-17 reify incident). The queue lets the canonical
write return success immediately; journal persistence becomes eventually
consistent.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import os
import random
import time
from collections import deque
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

import aiosqlite

if TYPE_CHECKING:
    from fused_memory.models.reconciliation import ReconciliationEvent
    from fused_memory.reconciliation.event_buffer import EventBuffer

logger = logging.getLogger(__name__)


def _iter_lines_reversed(
    path: Path,
    chunk_size: int = 8192,
    max_line_bytes: int = 1_048_576,
) -> Iterator[str]:
    """Yield lines from *path* newest-first without materialising the whole file.

    Opens *path* in binary mode, seeks to the end, and walks backward in
    *chunk_size* chunks, stitching partial lines across chunk boundaries.
    Decodes each line as UTF-8 with ``errors='replace'`` so malformed bytes
    pass through rather than raising — the ``json.JSONDecodeError`` path in
    :meth:`EventQueue.read_dead_letters` handles them gracefully.

    Empty lines are yielded as-is; callers should strip and skip blank lines.

    Args:
        path: Path to the JSONL file.
        chunk_size: Read granularity in bytes (default 8 KiB).
        max_line_bytes: Safety cap on the internal carry buffer.  When a
            partial-line fragment exceeds this threshold (e.g. from a
            corrupted file or a rogue writer), the accumulated bytes are
            yielded immediately with a warning so the downstream
            ``json.loads`` call treats them as a malformed line and skips
            them.  Defaults to 1 MiB.

    Raises:
        OSError: If the file cannot be opened or read.  Callers should catch
            and log.
    """
    with path.open('rb') as f:
        remaining = f.seek(0, 2)
        carry = b''
        while remaining > 0:
            to_read = min(chunk_size, remaining)
            remaining -= to_read
            f.seek(remaining)
            chunk = f.read(to_read)
            # Prepend older bytes to the fragment accumulated from the right.
            data = chunk + carry
            lines = data.split(b'\n')
            # lines[0]  — leftmost fragment; may extend further left in the file.
            # lines[1:] — complete lines (each starts after a '\n' boundary).
            carry = lines[0]
            if len(carry) > max_line_bytes:
                logger.warning(
                    '_iter_lines_reversed: carry buffer exceeded %d bytes in %s; '
                    'yielding truncated fragment as malformed — downstream '
                    'json.loads will skip it',
                    max_line_bytes, path,
                )
                yield carry.decode('utf-8', errors='replace')
                carry = b''
            for line in reversed(lines[1:]):
                yield line.decode('utf-8', errors='replace')
        # Yield the oldest fragment (the first line in the file, which has no
        # preceding '\n' so it stays in carry after the loop).
        if carry:
            yield carry.decode('utf-8', errors='replace')


class EventQueue:
    """Non-blocking in-memory hand-off to :class:`EventBuffer`.

    Drainer ownership: exactly one coroutine consumes the queue. Events are
    committed to SQLite one-by-one (no batching — keeps recovery simple and
    matches the existing ``EventBuffer.push`` contract). On retriable errors
    the same event is re-tried with exponential backoff; on non-retriable
    errors the event is diverted to the dead-letter file.
    """

    def __init__(
        self,
        event_buffer: EventBuffer,
        *,
        dead_letter_path: Path | str,
        maxsize: int = 10_000,
        retry_initial_seconds: float = 0.1,
        retry_max_seconds: float = 30.0,
        shutdown_flush_seconds: float = 10.0,
        overflow_warn_interval_seconds: float = 60.0,
        max_bytes: int | None = None,
        keep_rotations: int = 3,
    ):
        self._buffer = event_buffer
        self._dead_letter_path = Path(dead_letter_path)
        self._queue: asyncio.Queue[ReconciliationEvent] = asyncio.Queue(maxsize=maxsize)
        self._retry_initial = retry_initial_seconds
        self._retry_max = retry_max_seconds
        self._shutdown_flush = shutdown_flush_seconds
        self._overflow_warn_interval = overflow_warn_interval_seconds
        self._max_bytes = max_bytes
        self._keep_rotations = keep_rotations
        self._drainer_task: asyncio.Task | None = None
        self._closed = False
        # Stats surface for WP-C watchdog / WP-D policy.
        self._last_commit_ts: float | None = None
        self._overflow_drops = 0
        self._dead_letters = 0
        self._retry_in_flight = 0
        self._events_committed = 0
        self._last_overflow_warn_ts: float = 0.0
        # Ring buffer of recent drainer attempts — feeds the watchdog's
        # diagnostic payload so operators can see what wedged the writer.
        # Each entry: (monotonic_ts, event_id, event_type, status, attempts)
        self._recent_ops: deque[tuple[float, str, str, str, int]] = deque(maxlen=20)

    # ── lifecycle ──────────────────────────────────────────────────────

    async def start(self) -> None:
        """Spawn the background drainer coroutine."""
        if self._drainer_task is not None:
            raise RuntimeError('EventQueue already started')
        # Ensure dead-letter directory exists.
        self._dead_letter_path.parent.mkdir(parents=True, exist_ok=True)
        self._drainer_task = asyncio.create_task(
            self._drain_loop(), name='event-queue-drainer',
        )
        logger.info(
            'EventQueue started (maxsize=%d, dead_letter=%s)',
            self._queue.maxsize, self._dead_letter_path,
        )

    async def close(self) -> None:
        """Flush the queue within ``shutdown_flush_seconds`` then stop the drainer.

        Any events that cannot be flushed in time are dumped to the
        dead-letter file with ``reason="shutdown_timeout"``.
        """
        if self._closed:
            return
        self._closed = True

        if self._drainer_task is None:
            return

        # Bounded flush: let the drainer finish what it can.
        try:
            await asyncio.wait_for(
                self._queue.join(), timeout=self._shutdown_flush,
            )
        except TimeoutError:
            logger.warning(
                'EventQueue shutdown flush window (%.1fs) elapsed with '
                '%d events remaining — diverting to dead-letter',
                self._shutdown_flush, self._queue.qsize(),
            )

        # Cancel drainer — remaining events go to dead-letter below.
        self._drainer_task.cancel()
        with contextlib.suppress(asyncio.CancelledError, Exception):
            await self._drainer_task
        self._drainer_task = None

        # Drain residue synchronously to the dead-letter file. Using
        # get_nowait avoids awaiting after cancellation.
        residue: list[ReconciliationEvent] = []
        while True:
            try:
                event = self._queue.get_nowait()
            except asyncio.QueueEmpty:
                break
            residue.append(event)
            # Don't task_done() here — the drainer already abandoned the
            # contract; the queue is going away.

        if residue:
            for event in residue:
                self._write_dead_letter(event, reason='shutdown_timeout', attempts=0)
            logger.warning(
                'EventQueue: dead-lettered %d events on shutdown', len(residue),
            )

    # ── enqueue (sync, non-blocking) ───────────────────────────────────

    def enqueue(self, event: ReconciliationEvent) -> bool:
        """Try to enqueue an event. Returns True on accept, False on overflow.

        On overflow the event is written to the dead-letter file immediately
        and an overflow warning is logged (rate-limited to once per
        ``overflow_warn_interval_seconds``).
        """
        if self._closed:
            # Shutting down — divert straight to dead-letter so nothing is lost.
            self._write_dead_letter(event, reason='post_close', attempts=0)
            return False
        try:
            self._queue.put_nowait(event)
            return True
        except asyncio.QueueFull:
            self._overflow_drops += 1
            self._write_dead_letter(event, reason='overflow_drop', attempts=0)
            now = time.monotonic()
            if (now - self._last_overflow_warn_ts) >= self._overflow_warn_interval:
                self._last_overflow_warn_ts = now
                logger.warning(
                    'EventQueue overflow — capacity=%d, total_drops=%d '
                    '(event diverted to dead-letter)',
                    self._queue.maxsize, self._overflow_drops,
                )
            return False

    # ── stats surface ──────────────────────────────────────────────────

    def stats(self) -> dict:
        """Snapshot of queue health. Feeds WP-C watchdog and WP-D policy."""
        return {
            'queue_depth': self._queue.qsize(),
            'queue_capacity': self._queue.maxsize,
            'last_commit_ts': self._last_commit_ts,
            'events_committed': self._events_committed,
            'overflow_drops': self._overflow_drops,
            'dead_letters': self._dead_letters,
            'retry_in_flight': self._retry_in_flight,
            'drainer_running': (
                self._drainer_task is not None and not self._drainer_task.done()
            ),
        }

    def recent_ops(self) -> list[dict]:
        """Snapshot of the last ~20 drainer attempts (oldest first).

        Feeds the watchdog's wedge-diagnostic payload. ``status`` is one of
        ``committed``, ``retrying``, or ``dead_letter``.
        """
        return [
            {
                'monotonic_ts': ts,
                'event_id': event_id,
                'event_type': event_type,
                'status': status,
                'attempts': attempts,
            }
            for (ts, event_id, event_type, status, attempts) in self._recent_ops
        ]

    # ── drainer loop ───────────────────────────────────────────────────

    async def _drain_loop(self) -> None:
        """Pull events from the queue and commit to the buffer, with retry."""
        while True:
            event = await self._queue.get()
            try:
                await self._commit_with_retry(event)
            finally:
                # Always mark task_done so queue.join() unblocks even on
                # drainer errors — otherwise shutdown hangs.
                self._queue.task_done()

    async def _commit_with_retry(self, event: ReconciliationEvent) -> None:
        """Attempt to persist ``event``, retrying transient SQLite failures.

        On retriable errors (``aiosqlite.OperationalError``): exponential
        backoff with jitter, unlimited retries. The drainer can be
        cancelled at any await point and the event is then routed to the
        dead-letter file by the close() residue path (it's already been
        ``queue.get``-ed so it's no longer in the queue).

        On non-retriable errors: log and dead-letter immediately.
        """
        delay = self._retry_initial
        attempts = 0
        event_type = event.type.value if hasattr(event.type, 'value') else str(event.type)
        while True:
            attempts += 1
            try:
                await self._buffer.push(event)
                self._last_commit_ts = time.time()
                self._events_committed += 1
                if self._retry_in_flight > 0 and attempts > 1:
                    self._retry_in_flight = max(0, self._retry_in_flight - 1)
                self._recent_ops.append(
                    (time.monotonic(), event.id, event_type, 'committed', attempts),
                )
                return
            except aiosqlite.OperationalError as exc:
                # Retriable — the buffer's SQLite connection is locked or
                # transiently unavailable. Keep trying.
                if attempts == 1:
                    self._retry_in_flight += 1
                self._recent_ops.append(
                    (time.monotonic(), event.id, event_type, 'retrying', attempts),
                )
                logger.warning(
                    'EventQueue.drain: transient SQLite error (event=%s, '
                    'attempt=%d, sleep=%.2fs): %s',
                    event.id, attempts, delay, exc,
                )
                await asyncio.sleep(delay + random.uniform(0, delay * 0.2))
                delay = min(delay * 2.0, self._retry_max)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                # Non-retriable — schema / serialization / programmer error.
                # Dead-letter and move on; don't let one bad event wedge the
                # drainer.
                logger.error(
                    'EventQueue.drain: non-retriable error for event=%s: %s',
                    event.id, exc, exc_info=True,
                )
                self._write_dead_letter(event, reason='non_retriable', attempts=attempts)
                if self._retry_in_flight > 0 and attempts > 1:
                    self._retry_in_flight = max(0, self._retry_in_flight - 1)
                self._recent_ops.append(
                    (time.monotonic(), event.id, event_type, 'dead_letter', attempts),
                )
                return

    # ── dead-letter ────────────────────────────────────────────────────

    def _write_dead_letter(
        self,
        event: ReconciliationEvent,
        *,
        reason: str,
        attempts: int,
    ) -> None:
        """Append an event to the dead-letter JSONL file.

        Best-effort. A failure here is logged but must not raise — we'd
        rather lose a single event than crash the drainer or the caller.
        """
        record = {
            'event': event.model_dump(mode='json'),
            'reason': reason,
            'attempts': attempts,
            'failed_at': datetime.now(UTC).isoformat(),
        }
        try:
            self._dead_letter_path.parent.mkdir(parents=True, exist_ok=True)
            # Rotate before appending when the file has grown past the byte cap.
            if (
                self._max_bytes is not None
                and self._dead_letter_path.exists()
                and self._dead_letter_path.stat().st_size >= self._max_bytes
            ):
                self._rotate_dead_letter()
            with self._dead_letter_path.open('a', encoding='utf-8') as fh:
                fh.write(json.dumps(record) + '\n')
            self._dead_letters += 1
        except Exception as exc:
            logger.error(
                'EventQueue: failed to append dead-letter record for event=%s: %s',
                event.id, exc,
            )

    def _rotate_dead_letter(self) -> None:
        """Cascade-rotate dead-letter files.

        ``dead_letter.jsonl``   → ``dead_letter.jsonl.1``
        ``dead_letter.jsonl.1`` → ``dead_letter.jsonl.2``
        …
        ``dead_letter.jsonl.{keep}`` → overwritten/dropped by the cascade

        When ``keep_rotations == 0``, the current file is simply unlinked so
        the byte cap is still honoured (nothing is archived).

        After the cascade (or the zero-keep discard), :meth:`_purge_orphan_rotations`
        runs unconditionally via a ``finally`` block.  This makes retention
        self-repairing in all branches: if an operator lowers ``keep_rotations``
        between runs (e.g. 5 → 0 or 5 → 2), previously-created ``.3``, ``.4``,
        ``.5`` files are cleaned up the next time a rotation fires — regardless
        of which branch executes or whether the cascade loop itself raises
        mid-iteration.

        **Safety note for the finally-on-exception case**: the cascade loop
        only writes to indices ``1`` … ``keep_rotations`` (i.e. ``dl.1``,
        ``dl.2``, …, ``dl.{keep}``), while :meth:`_purge_orphan_rotations`
        only removes siblings with index ``> keep_rotations``.  These ranges
        are disjoint, so invoking the purge in the ``finally`` block after a
        mid-cascade failure cannot remove any file the partial cascade just
        placed.  The only data that could be lost is the oldest rotation slot
        (``dl.{keep}``) that the cascade would have overwritten anyway when it
        ran successfully — that slot is subject to the best-effort contract.

        Uses ``os.replace`` for atomicity — it already overwrites the
        destination on POSIX and Windows, so no pre-unlink is needed.
        Any error is caught and logged to preserve the best-effort contract.
        """
        try:
            try:
                if self._keep_rotations == 0:
                    # No archival rotations: just discard the current file.
                    # The finally clause below still runs, purging any orphan
                    # siblings left over from a prior higher-keep_rotations run.
                    self._dead_letter_path.unlink(missing_ok=True)
                    return
                # Work from oldest → newest so we never overwrite unsaved data.
                # os.replace atomically overwrites the destination; the oldest
                # file (index keep_rotations) is simply replaced/dropped by the
                # cascade without a separate unlink step.
                for i in range(self._keep_rotations, 0, -1):
                    src = Path(f'{self._dead_letter_path}.{i - 1}') if i > 1 else self._dead_letter_path
                    dst = Path(f'{self._dead_letter_path}.{i}')
                    if src.exists():
                        os.replace(src, dst)
            finally:
                # Remove any siblings whose numeric suffix exceeds the current
                # keep_rotations bound.  Runs unconditionally — including the
                # zero-keep branch above and when the cascade raises mid-loop.
                self._purge_orphan_rotations()
        except Exception as exc:
            logger.error(
                'EventQueue: dead-letter rotation failed: %s', exc,
            )

    def _purge_orphan_rotations(self) -> None:
        """Remove rotation siblings whose index exceeds ``keep_rotations``.

        Scans the parent directory for files whose name matches
        ``{dead_letter_path.name}.{N}`` where N is an integer greater than
        ``keep_rotations``, and unlinks them.  Non-numeric suffixes (e.g.
        ``.bak``, ``.swp``) are left untouched.

        This is a best-effort cleanup: any OSError on an individual sibling
        (permission denied, race-deleted, etc.) is silently suppressed so
        the rotation as a whole succeeds.  A missing or unreadable directory
        is also silently ignored.
        """
        parent = self._dead_letter_path.parent
        prefix = f'{self._dead_letter_path.name}.'
        try:
            siblings = list(parent.iterdir())
        except (FileNotFoundError, OSError):
            return
        for sibling in siblings:
            if not sibling.name.startswith(prefix):
                continue
            suffix = sibling.name[len(prefix):]
            if not suffix.isdigit():
                continue  # non-numeric suffix (or '-1', '+3', '01', …) — leave it alone
            index = int(suffix)
            if index > self._keep_rotations:
                with contextlib.suppress(OSError):
                    sibling.unlink(missing_ok=True)

    # ── read dead-letters ──────────────────────────────────────────────

    def read_dead_letters(
        self,
        *,
        limit: int | None = None,
        project_id: str | None = None,
    ) -> list[dict]:
        """Return dead-letter records from the JSONL file(s), newest-first.

        Enumerates :attr:`_dead_letter_path`, then ``.1``, ``.2`` … up to
        ``keep_rotations`` siblings.  Within each file lines are reversed so
        the newest record appears first.  Records from the current file
        precede those from rotated siblings.

        Args:
            limit: Maximum number of records to return.  *None* means all.
            project_id: When given, only records whose
                ``event['project_id']`` matches are included.

        Returns:
            A list of dicts with keys ``event``, ``reason``, ``attempts``,
            ``failed_at``.  Never raises — I/O errors yield an empty list
            plus a logged warning.
        """
        # Build the ordered list of paths: current first, then .1, .2, …
        paths: list[Path] = [self._dead_letter_path]
        for i in range(1, self._keep_rotations + 1):
            paths.append(Path(f'{self._dead_letter_path}.{i}'))

        results: list[dict] = []
        try:
            for path in paths:
                if not path.exists():
                    continue
                try:
                    # Stream lines newest-first without materialising the file.
                    for line in _iter_lines_reversed(path):
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            rec = json.loads(line)
                        except json.JSONDecodeError as exc:
                            logger.warning(
                                'EventQueue.read_dead_letters: malformed line in %s: %s',
                                path, exc,
                            )
                            continue

                        # Filter by project_id if requested.
                        if project_id is not None:
                            event = rec.get('event') or {}
                            if event.get('project_id') != project_id:
                                continue

                        results.append(rec)
                        if limit is not None and len(results) >= limit:
                            return results
                except OSError as exc:
                    logger.warning(
                        'EventQueue.read_dead_letters: cannot read %s: %s', path, exc,
                    )
                    continue

        except Exception as exc:
            logger.warning(
                'EventQueue.read_dead_letters: unexpected error: %s', exc,
            )
            return []

        return results
