"""The durable FLOOR under the markup-residue escalation queue (task 4809).

When the MCP boundary guard refuses a call it cannot repair, the caller's
payload is filed into a project's escalation queue. When no queue takes it,
this spool holds it instead: one JSON file per payload, read only when someone
looks. Nothing sits below the spool, so a write that fails here is a LOST
payload. Each instance counts those, and ``get_status`` reports the count
(INV-4).

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
from typing import Any

logger = logging.getLogger(__name__)

#: Fixed-width UTC, so locators sort lexically in the order they were written.
_LOCATOR_TIME_FORMAT = '%Y%m%dT%H%M%S.%fZ'

#: Fresh names tried when a claimed name already exists.
_CLAIM_ATTEMPTS = 4


class MarkupResidueSpool:
    """One spool directory, and the count of payloads it failed to hold.

    The count is process-local and reset by a restart, the same contract
    ``services/write_journal.py::WriteJournal.journal_drop_stats`` states: it
    makes a live loss visible rather than keeping durable statistics.
    """

    def __init__(self, spool_dir: Path) -> None:
        self._spool_dir = Path(spool_dir)
        self._failures = 0
        self._last_failure: dict[str, Any] | None = None

    def preserve(self, record: Mapping[str, Any]) -> str | None:
        """Write *record* to a new file and return its name, else ``None``.

        The name is the locator reported to the refused caller, so it is
        returned only for a file that exists and is non-empty. A value JSON
        cannot encode is written as its ``repr``. Never raises.
        """
        try:
            body = json.dumps(dict(record), default=repr, ensure_ascii=False)
            return self._write_new_file(body.encode('utf-8'))
        except Exception as exc:
            failures = self._count_loss(record)
            logger.error(
                'markup_residue_lost: %s.%s (%d chars) could not be preserved in '
                '%s (%s: %s); nothing durable holds it (preservation_failures=%d)',
                record.get('tool'), record.get('field'), _raw_length(record),
                self._spool_dir, type(exc).__name__, exc, failures,
            )
            return None

    def status(self) -> dict[str, Any]:
        """``spool_dir``, ``preservation_failures`` and ``last_failure``, copied."""
        return {
            'spool_dir': str(self._spool_dir),
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
            return name
        raise FileExistsError(f'no unclaimed name after {_CLAIM_ATTEMPTS} attempts')

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
        with os.fdopen(fd, 'wb') as handle:
            handle.write(body)
            handle.flush()
            os.fsync(handle.fileno())
        complete = path.stat().st_size > 0
    finally:
        if not complete:
            path.unlink(missing_ok=True)
    if not complete:
        raise OSError(f'{path} was empty after writing')


def _raw_length(record: Mapping[str, Any]) -> int:
    raw_value = record.get('raw_value')
    return len(raw_value) if isinstance(raw_value, str) else 0
