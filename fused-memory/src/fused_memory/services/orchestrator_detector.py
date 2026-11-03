"""Detect whether an orchestrator instance is live for a given project root.

At startup the orchestrator takes ``LOCK_EX | LOCK_NB`` on
``<project_root>/data/orchestrator/orchestrator.lock`` and writes
``PID <N> started <timestamp>`` as its first line. This module reads two
liveness signals from that one file. Pick one by what a wrong answer costs:

* :func:`is_orchestrator_lock_held` asks the kernel whether some process
  holds a flock on the file, so it cannot go stale. Use it to decide whether
  to file work that only a running orchestrator drains, where a false "live"
  loses the work silently. Its momentary shared lock refuses an orchestrator
  start that lands in the same instant, and
  ``orchestrator/src/orchestrator/harness.py::_acquire_project_lock`` exits
  rather than retrying.
* :func:`is_orchestrator_live_for` sends signal 0 to the PID in the file. It
  never touches the lock, but a PID that outlived its orchestrator (a crash,
  then PID reuse) reads as live.
"""

from __future__ import annotations

import fcntl
import logging
import os
from datetime import UTC, datetime
from pathlib import Path

logger = logging.getLogger(__name__)


def is_orchestrator_live_for(project_root: str | Path) -> bool:
    """Return True iff the PID recorded in the project's orchestrator.lock answers signal 0."""
    lock_path = _lock_path(project_root)
    try:
        text = lock_path.read_text(encoding='utf-8')
    except FileNotFoundError:
        return False
    except OSError as exc:
        logger.debug('orchestrator_detector: cannot read %s: %s', lock_path, exc)
        return False

    pid = _parse_pid(text)
    if pid is None:
        return False
    return _pid_alive(pid)


def is_orchestrator_lock_held(project_root: str | Path) -> bool:
    """Return True iff some process holds a flock on the project's orchestrator.lock.

    Probes with ``LOCK_SH | LOCK_NB`` on its own handle, so it never waits;
    closing that handle releases the probe's lock. A missing or unopenable
    lock file reads as not held; any other flock error propagates.

    This is synchronous file I/O, so a coroutine calls it through
    ``asyncio.to_thread``.
    """
    try:
        handle = _lock_path(project_root).open('rb')
    except OSError:
        return False
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_SH | fcntl.LOCK_NB)
    except BlockingIOError:
        return True
    finally:
        handle.close()
    return False


def orchestrator_started_at(project_root: str | Path) -> datetime | None:
    """Return the current orchestrator's start time from its lock file.

    Reads the SAME ``<project_root>/data/orchestrator/orchestrator.lock`` file
    as :func:`is_orchestrator_live_for` and parses the ``started <ISO>`` token
    from its first line (format ``PID <N> started <ISO-8601>``, written on every
    orchestrator (re)start). The parsed timestamp is normalized to a UTC-aware
    ``datetime`` (a trailing ``Z`` is accepted by ``fromisoformat`` on Py3.11+;
    a naive timestamp is assumed UTC).

    This is the orchestrator-restart boundary used by the live-workflow
    corroboration gate (task 2963): a routing decision that predates this start
    time cannot correspond to a currently-live workflow. Deriving the boundary
    from the lock's ``started`` timestamp avoids any fused-memory→orchestrator
    or systemd coupling — the lock file is already fused-memory's on-disk
    orchestrator-liveness source.

    Fail-safe: returns ``None`` (never raises) on a missing lock file
    (``FileNotFoundError``), any other read error (``OSError`` — e.g. the lock
    path is a directory), an absent/short first line, a missing ``started``
    token, or an unparseable timestamp (``ValueError``). All failure modes are
    logged at debug, mirroring the other helpers in this module.
    """
    lock_path = _lock_path(project_root)
    try:
        text = lock_path.read_text(encoding='utf-8')
    except FileNotFoundError:
        return None
    except OSError as exc:
        logger.debug('orchestrator_detector: cannot read %s: %s', lock_path, exc)
        return None

    return _parse_started(text)


def _lock_path(project_root: str | Path) -> Path:
    return Path(project_root) / 'data' / 'orchestrator' / 'orchestrator.lock'


def _parse_started(content: str) -> datetime | None:
    """Parse the ``started <ISO>`` timestamp from a lock file's first line.

    Tokenizes the first line (mirroring :func:`_parse_pid`), locates the
    ``started`` token, and parses the FOLLOWING token as an ISO-8601 timestamp
    via ``datetime.fromisoformat``. Returns a UTC-aware datetime, or ``None`` on
    any structural or parse failure (fail-safe).
    """
    first = content.splitlines()[0] if content else ''
    tokens = first.strip().split()
    try:
        idx = tokens.index('started')
    except ValueError:
        return None
    if idx + 1 >= len(tokens):
        return None
    try:
        dt = datetime.fromisoformat(tokens[idx + 1])
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return dt


def _parse_pid(content: str) -> int | None:
    """Parse the PID from the first line of an orchestrator.lock file.

    Accepts ``PID 12345 started ...`` and a bare integer as fallback.
    """
    first = content.splitlines()[0] if content else ''
    tokens = first.strip().split()
    if len(tokens) >= 2 and tokens[0].upper() == 'PID':
        try:
            return int(tokens[1])
        except ValueError:
            return None
    try:
        return int(first.strip())
    except ValueError:
        return None


def _pid_alive(pid: int) -> bool:
    """Return True iff signal 0 can be sent to ``pid``."""
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        # Process exists but we can't signal it — still counts as live.
        return True
    except OSError as exc:
        logger.debug('orchestrator_detector: os.kill(%d, 0) failed: %s', pid, exc)
        return False
    return True
