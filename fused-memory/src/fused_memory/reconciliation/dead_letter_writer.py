"""The sole appender of one rotating dead-letter JSONL file.

Every append, and every rotation an append triggers, runs on one dedicated
worker thread, so no event-loop coroutine performs that file I/O (INV-8).
One worker also means one in-process appender: records land in submission
order, and a rotation can never interleave with an append.
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import threading
from collections.abc import Mapping
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def _running_future() -> Future[Any]:
    """A future already marked running, so ``cancel()`` can never discard it."""
    future: Future[Any] = Future()
    future.set_running_or_notify_cancel()
    return future


class DeadLetterWriter:
    """Appends JSON records to ``path`` from one worker thread, rotating by size.

    :meth:`append` never blocks and never raises: it hands the record to the
    worker and returns a future that resolves ``True`` once the line is on
    disk, or ``False`` if the write failed (the failure is logged). The
    future cannot be cancelled, so an awaiter that is itself cancelled cannot
    drop an accepted record.
    """

    def __init__(
        self,
        path: Path | str,
        *,
        max_bytes: int | None = None,
        keep_rotations: int = 3,
    ) -> None:
        self._path = Path(path)
        self._max_bytes = max_bytes
        self._keep_rotations = keep_rotations
        self._executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix='event-dead-letter-writer',
        )
        self._pending_lock = threading.Lock()
        self._last: Future[bool] | None = None
        self._records_pending = 0
        self._records_written = 0

    @property
    def records_pending(self) -> int:
        """Records accepted by :meth:`append` whose write has not finished yet."""
        return self._records_pending

    @property
    def records_written(self) -> int:
        """Records actually appended so far. Only the worker thread increments it."""
        return self._records_written

    def paths(self) -> list[Path]:
        """The current file, then each rotation ``.1`` … ``.keep_rotations``."""
        return [
            self._path,
            *(self._rotation_path(i) for i in range(1, self._keep_rotations + 1)),
        ]

    def append(self, record: Mapping[str, Any]) -> Future[bool]:
        """Hand ``record`` to the worker thread; resolve once it is written or has failed."""
        written: Future[bool] = _running_future()
        try:
            line = json.dumps(record) + '\n'
            with self._pending_lock:
                self._executor.submit(self._write, line, written)
                self._last = written
                self._records_pending += 1
        except Exception as exc:
            logger.error(
                'DeadLetterWriter: cannot hand a dead-letter record for %s to '
                'its writer thread: %s',
                self._path, exc,
            )
            written.set_result(False)
        return written

    def flushed(self) -> Future[None]:
        """Resolve once every append submitted before this call has finished."""
        flushed: Future[None] = _running_future()
        with self._pending_lock:
            last = self._last
        if last is None:
            flushed.set_result(None)
        else:
            last.add_done_callback(lambda _: flushed.set_result(None))
        return flushed

    def _rotation_path(self, index: int) -> Path:
        return Path(f'{self._path}.{index}')

    def _write(self, line: str, written: Future[bool]) -> None:
        ok = False
        try:
            self._append_line(line)
            ok = True
        except Exception as exc:
            logger.error(
                'DeadLetterWriter: failed to append a dead-letter record to %s: %s',
                self._path, exc,
            )
        finally:
            with self._pending_lock:
                self._records_pending -= 1
            written.set_result(ok)

    def _append_line(self, line: str) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        # Rotate before appending when the file has grown past the byte cap.
        if (
            self._max_bytes is not None
            and self._path.exists()
            and self._path.stat().st_size >= self._max_bytes
        ):
            self._rotate()
        with self._path.open('a', encoding='utf-8') as fh:
            fh.write(line)
        self._records_written += 1

    def _rotate(self) -> None:
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
                    self._path.unlink(missing_ok=True)
                    return
                # Work from oldest → newest so we never overwrite unsaved data.
                # os.replace atomically overwrites the destination; the oldest
                # file (index keep_rotations) is simply replaced/dropped by the
                # cascade without a separate unlink step.
                for i in range(self._keep_rotations, 0, -1):
                    src = self._rotation_path(i - 1) if i > 1 else self._path
                    dst = self._rotation_path(i)
                    if src.exists():
                        os.replace(src, dst)
            finally:
                # Remove any siblings whose numeric suffix exceeds the current
                # keep_rotations bound.  Runs unconditionally — including the
                # zero-keep branch above and when the cascade raises mid-loop.
                self._purge_orphan_rotations()
        except Exception as exc:
            logger.error(
                'DeadLetterWriter: dead-letter rotation failed: %s', exc,
            )

    def _purge_orphan_rotations(self) -> None:
        """Remove rotation siblings whose index exceeds ``keep_rotations``.

        Scans the parent directory for files whose name matches
        ``{path.name}.{N}`` where N is an integer greater than
        ``keep_rotations``, and unlinks them.  Non-numeric suffixes (e.g.
        ``.bak``, ``.swp``) are left untouched.

        This is a best-effort cleanup: any OSError on an individual sibling
        (permission denied, race-deleted, etc.) is silently suppressed so
        the rotation as a whole succeeds.  A missing or unreadable directory
        is also silently ignored.
        """
        parent = self._path.parent
        prefix = f'{self._path.name}.'
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
