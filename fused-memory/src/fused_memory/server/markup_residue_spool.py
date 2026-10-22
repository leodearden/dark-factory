"""The durable FLOOR under the markup-residue escalation queue (task 4809).

When the MCP boundary guard refuses a call it cannot repair, the caller's
payload is filed into a project's escalation queue. When no queue takes it,
this spool holds it instead: one JSON file per payload, read only when someone
looks. Nothing sits below the spool, so a write that fails here is a LOST
payload. Each instance counts those, and ``get_status`` reports the count
(INV-4) beside how many files are waiting for an operator.

The spool lives in the server's own data dir, outside every task worktree, so
it survives restarts and worktree teardown. It is deliberately not the
worktree-local orchestrator floor task 4584 removed.
"""

from __future__ import annotations

import json
import logging
import os
import secrets
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, BinaryIO

from shared.mcp_markup_middleware import residue_payload_length

logger = logging.getLogger(__name__)

#: Fixed-width UTC, so locators sort lexically in the order they were written.
_LOCATOR_TIME_FORMAT = '%Y%m%dT%H%M%S.%fZ'

#: Fresh names tried when a claimed name already exists.
_CLAIM_ATTEMPTS = 4

#: Past this many waiting files the spool still preserves every payload, but
#: logs one ERROR line asking an operator to drain it.
DEFAULT_SOFT_CAP_FILES = 1000


class MarkupResidueSpool:
    """One spool directory, and the count of payloads it failed to hold.

    The count is process-local and reset by a restart, the same contract
    ``services/write_journal.py::WriteJournal.journal_drop_stats`` states: it
    makes a live loss visible rather than keeping durable statistics.
    """

    def __init__(
        self, spool_dir: Path, *, soft_cap_files: int = DEFAULT_SOFT_CAP_FILES
    ) -> None:
        # Absolute at construction, so a later chdir cannot move the spool and
        # every path handed to a caller can be opened as it stands.
        self._spool_dir = Path(spool_dir).absolute()
        self._soft_cap_files = soft_cap_files
        self._over_soft_cap = False
        self._last_preserved: str | None = None
        self._failures = 0
        self._last_failure: dict[str, Any] | None = None

    def preserve(self, record: Mapping[str, Any]) -> str | None:
        """Write *record* to a new file and return its absolute path, else ``None``.

        The path is reported to the refused caller, so it is returned only for
        a file that exists and is non-empty. A value JSON cannot encode is
        written as its ``repr``. Never raises.
        """
        try:
            body = json.dumps(dict(record), default=repr, ensure_ascii=False)
            path = self._write_new_file(body.encode('utf-8'))
        except Exception as exc:
            failures = self._count_loss(record)
            logger.error(
                'markup_residue_lost: %s.%s (%d chars) could not be preserved in '
                '%s (%s: %s); nothing durable holds it (preservation_failures=%d)',
                record.get('tool'), record.get('field'), residue_payload_length(record),
                self._spool_dir, type(exc).__name__, exc, failures,
            )
            return None
        self._last_preserved = path
        self._check_soft_cap()
        return path

    def status(self) -> dict[str, Any]:
        """The spool's dir, waiting files and losses, as a fresh dict.

        ``spooled_files`` is ``None`` when the dir cannot be listed.
        ``last_preserved`` and the loss fields cover this process only.
        """
        return {
            'spool_dir': str(self._spool_dir),
            'spooled_files': self._count_spooled_files(),
            'last_preserved': self._last_preserved,
            'preservation_failures': self._failures,
            'last_failure': dict(self._last_failure) if self._last_failure else None,
        }

    def _write_new_file(self, body: bytes) -> str:
        self._spool_dir.mkdir(parents=True, exist_ok=True)
        for _ in range(_CLAIM_ATTEMPTS):
            name = f'{datetime.now(UTC):{_LOCATOR_TIME_FORMAT}}-{secrets.token_hex(4)}.json'
            path = self._spool_dir / name
            try:
                fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
            except FileExistsError:
                continue
            _write_claimed(fd, path, body)
            return str(path)
        raise FileExistsError(f'no unclaimed name after {_CLAIM_ATTEMPTS} attempts')

    def _count_spooled_files(self) -> int | None:
        try:
            with os.scandir(self._spool_dir) as entries:
                return sum(
                    1 for entry in entries
                    if entry.name.endswith('.json') and entry.is_file()
                )
        except FileNotFoundError:
            return 0
        except OSError as exc:
            logger.warning(
                'markup_residue_spool: could not count the files in %s (%s: %s)',
                self._spool_dir, type(exc).__name__, exc,
            )
            return None

    def _check_soft_cap(self) -> None:
        """Log one ERROR line when the spool crosses its soft cap, not one per write."""
        spooled = self._count_spooled_files()
        if spooled is None:
            return
        over = spooled > self._soft_cap_files
        if over and not self._over_soft_cap:
            logger.error(
                'markup_residue_spool_over_soft_cap: %d files wait in %s, past the '
                'soft cap of %d. Every payload is still preserved; an operator '
                'should recover and drain them.',
                spooled, self._spool_dir, self._soft_cap_files,
            )
        self._over_soft_cap = over

    def _count_loss(self, record: Mapping[str, Any]) -> int:
        self._failures += 1
        self._last_failure = {
            'tool': record.get('tool'),
            'field': record.get('field'),
            'at': datetime.now(UTC).isoformat(),
        }
        return self._failures


def _write_claimed(fd: int, path: Path, body: bytes) -> None:
    """Fill the file *fd* claimed at *path*, or unlink it: never leave a stub."""
    complete = False
    try:
        with _adopt(fd) as handle:
            handle.write(body)
            handle.flush()
            os.fsync(handle.fileno())
        complete = path.stat().st_size > 0
    finally:
        if not complete:
            path.unlink(missing_ok=True)
    if not complete:
        raise OSError(f'{path} was empty after writing')


def _adopt(fd: int) -> BinaryIO:
    """``os.fdopen(fd, 'wb')``, closing *fd* itself if the wrap fails."""
    try:
        return os.fdopen(fd, 'wb')
    except BaseException:
        os.close(fd)
        raise
