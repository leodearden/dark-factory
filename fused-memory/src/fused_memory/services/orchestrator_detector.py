"""Detect whether an orchestrator instance is live for a given project root.

The orchestrator writes its PID to ``<project_root>/data/orchestrator/orchestrator.lock``
when it starts. The first line format is ``PID <N> started <timestamp>``. We treat
the lock as live only when (a) the file exists, (b) we can parse a PID from it, and
(c) sending signal 0 to that PID succeeds (process exists and we have permission to
signal it). Anything else — missing file, unparseable content, dead PID — is a stale
or absent lock, so the orchestrator is not live.

Used by :class:`BacklogPolicy` to decide whether to escalate (orchestrator can see
the escalation file) or reject the MCP call (no one's watching).
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)


def is_orchestrator_live_for(project_root: str | Path) -> bool:
    """Return True iff a running orchestrator process holds the project's lock."""
    lock_path = Path(project_root) / 'data' / 'orchestrator' / 'orchestrator.lock'
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
