"""Find the systemd scopes a spawned command's session has moved into.

A verify command is spawned as the leader of its own session. A payload that
runs ``systemd-run --user --scope`` itself (reify's cpu-governed-exec.sh on
the merge leg) moves its processes into a new scope, outside the one the
spawner kills. Moved processes keep their session id, so the session finds
those scopes again (task 6078).
"""

from __future__ import annotations

import asyncio
import os
from collections.abc import Iterator
from pathlib import Path

from shared.proc_group import read_session_id, read_stat_fields

_PROC_ROOT = Path('/proc')


async def relocated_scopes(
    leader: int, *, launched_in: str | None, proc_root: Path = _PROC_ROOT
) -> frozenset[str]:
    """The scope units holding members of the session *leader* leads, other than
    *launched_in* and the caller's own scope.

    Empty unless *leader* is the caller's own child leading its own session;
    only then is /proc scanned, on a worker thread. Never raises.
    """
    if not _is_own_session_leader(proc_root / str(leader)):
        return frozenset()
    return await asyncio.to_thread(_scopes_of_session, leader, launched_in, proc_root)


def _scopes_of_session(leader: int, launched_in: str | None, proc_root: Path) -> frozenset[str]:
    excluded = {launched_in, _scope_of(proc_root / 'self')}
    scopes = (_scope_of(member) for member in _session_members(proc_root, leader))
    return frozenset(scope for scope in scopes if scope is not None and scope not in excluded)


def _is_own_session_leader(entry: Path) -> bool:
    fields = read_stat_fields(entry, skip_malformed=True)
    return (
        fields is not None
        and fields.ppid == os.getpid()
        and read_session_id(entry) == int(entry.name)
    )


def _scope_of(entry: Path) -> str | None:
    """The deepest ``*.scope`` segment of *entry*'s unified (``0::``) cgroup path."""
    try:
        text = os.fsdecode((entry / 'cgroup').read_bytes())
    except OSError:
        return None
    unified = next((line[3:] for line in text.splitlines() if line.startswith('0::')), None)
    if unified is None:
        return None
    return next(
        (segment for segment in reversed(unified.split('/')) if segment.endswith('.scope')), None
    )


def _session_members(proc_root: Path, leader: int) -> Iterator[Path]:
    try:
        entries = [entry for entry in proc_root.iterdir() if entry.name.isdigit()]
    except OSError:
        return
    for entry in entries:
        if read_session_id(entry) == leader:
            yield entry
