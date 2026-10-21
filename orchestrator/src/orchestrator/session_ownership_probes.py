"""Process-tree probes behind the session hooks' ownership decisions.

Answers, from one observation of the process tree, which ``claude`` process
fired a hook event, whether it is the one ``spawn-claude.sh`` launched, and
which pid a new session record's liveness should track.

Stdlib-only: ``session_hooks.py`` imports this at module scope and runs from a
bare shell with only ``PYTHONPATH=orchestrator/src``. Pinned by
``orchestrator/tests/test_session_registry.py::TestStdlibOnlySelfContainment``.

See ``docs/session-hook-ownership.md`` §Process-tree probes.
"""

from __future__ import annotations

import enum
import logging
import os
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

MAX_CLAUDE_ANCESTOR_HOPS = 6
"""How many ancestors above the hook's parent the walk inspects for the firing ``claude``."""


def _parent_pid_of(pid: int) -> int | None:
    """The parent pid of *pid* per ``/proc/<pid>/status``, or None.

    ``status`` (not ``stat``) is parsed deliberately: its ``PPid:`` line is
    plainly whitespace-delimited, immune to the ``(comm)`` field's embedded
    spaces and parens. Returns None -- never raises -- on a platform with no
    ``/proc``, on a race where *pid* exits mid-read, and on any unparseable
    body, so every caller can treat the answer as a best-effort hint.
    """
    try:
        for line in Path(f'/proc/{pid}/status').read_text().splitlines():
            if line.startswith('PPid:'):
                return int(line.split()[1])
    except (OSError, ValueError, IndexError):
        return None
    return None


def _process_comm(pid: int) -> str | None:
    """``/proc/<pid>/comm`` for *pid*, stripped, or None.

    Returns None -- never raises -- on a platform with no ``/proc``, on a
    race where *pid* exits mid-read, and on any unreadable body, so callers
    can treat the answer as a best-effort hint.
    """
    try:
        return Path(f'/proc/{pid}/comm').read_text().strip()
    except OSError:
        return None


def _session_leader_pid() -> int:
    """This process's POSIX session leader, or its parent where ``getsid`` raises."""
    try:
        return os.getsid(0)
    except OSError:
        return os.getppid()


@dataclass(frozen=True)
class ProcessTree:
    """The process tree one hook event's probes read.

    ``ProcessTree()`` reads the live tree: this process's parent and session
    leader, then ``/proc``. The hook entry points accept one as
    ``process_tree=``, so a caller names the tree to be judged rather than
    patching these readers.
    """

    hook_parent_pid: Callable[[], int] = os.getppid
    parent_pid_of: Callable[[int], int | None] = _parent_pid_of
    comm_of: Callable[[int], str | None] = _process_comm
    session_leader_pid: Callable[[], int] = _session_leader_pid


def _owning_claude_pid(tree: ProcessTree) -> int | None:
    """The first ``claude`` ancestor of the hook within ``MAX_CLAUDE_ANCESTOR_HOPS``, or None.

    Resolved by identity, not by a fixed depth, because Claude Code runs the
    hook through ``sh -c``; the measured tree is
    ``orchestrator/tests/test_session_ownership_probes.py::_REAL_HOOK_CHAIN``.
    Never raises; None means "cannot prove ownership".
    """
    pid: int | None = tree.hook_parent_pid()
    for _ in range(MAX_CLAUDE_ANCESTOR_HOPS):
        if pid is None or pid <= 1:
            return None
        if tree.comm_of(pid) == 'claude':
            return pid
        pid = tree.parent_pid_of(pid)
    return None


class _Unresolved(enum.Enum):
    """"Not computed yet", distinct from None, which is a resolved answer."""

    TOKEN = enum.auto()


class EventProbes:
    """One hook event's ownership probes, each computed at most once.

    Construct one per event and pass it down, so every decision the event
    makes reads one walk of one ``ProcessTree``. Never share one across
    events: the process tree is live state. *process_tree* None reads the
    live tree. See ``docs/session-hook-ownership.md`` §Process-tree probes.
    """

    def __init__(self, env: Mapping[str, str], process_tree: ProcessTree | None) -> None:
        self._env = env
        self._tree = ProcessTree() if process_tree is None else process_tree
        self._pid: int | None | _Unresolved = _Unresolved.TOKEN
        self._verdict: bool | None | _Unresolved = _Unresolved.TOKEN

    def owning_claude_pid(self) -> int | None:
        """The pid of the ``claude`` that fired this event, or None when unresolvable."""
        if self._pid is _Unresolved.TOKEN:
            self._pid = _owning_claude_pid(self._tree)
        return self._pid

    def owner_ppid_verdict(self) -> bool | None:
        """Is this event's ``claude`` the one ``spawn-claude.sh`` launched?

        True when its direct parent equals ``CLAUDE_SPAWN_OWNER_PPID``, False
        when it differs, and None when the variable or either pid cannot be
        read, so callers fail soft. See ``docs/session-hook-ownership.md``
        §Owner or inheritor.
        """
        if self._verdict is _Unresolved.TOKEN:
            self._verdict = self._compute_owner_ppid_verdict()
        return self._verdict

    def hand_launched_liveness_pid(self) -> int:
        """The terminal's session leader, which outlives the hook's own shell."""
        return self._tree.session_leader_pid()

    def nested_claude_liveness_pid(self) -> int:
        """The nested ``claude`` itself, so a forked record dies with it; else the session leader."""
        return self.owning_claude_pid() or self.hand_launched_liveness_pid()

    def _compute_owner_ppid_verdict(self) -> bool | None:
        raw = (self._env.get('CLAUDE_SPAWN_OWNER_PPID') or '').strip()
        if not raw:
            return None
        try:
            expected = int(raw)
        except ValueError:
            logger.warning('unparseable CLAUDE_SPAWN_OWNER_PPID %r; no ownership verdict', raw)
            return None
        if expected <= 1:
            return None
        owner_pid = self.owning_claude_pid()
        if owner_pid is None:
            return None
        actual = self._tree.parent_pid_of(owner_pid)
        if actual is None:
            return None
        return actual == expected
