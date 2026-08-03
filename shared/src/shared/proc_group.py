"""Process-group signal helpers for clean subprocess tree termination.

When the orchestrator spawns a bash shell that in turn spawns cargo, which
spawns rustc worker threads, a plain ``proc.terminate()`` / ``proc.kill()``
only signals the immediate bash PID.  The cargo and rustc grandchildren
remain alive, keeping files open and consuming CPU/memory after the parent
has nominally "cleaned up".

``terminate_process_group`` fixes this by using ``os.killpg`` to deliver
the signal to every process in the group at once, provided the child was
spawned with ``start_new_session=True`` (which makes it the leader of a
fresh process group).

Usage
-----
The pgid MUST be captured by the caller immediately after spawn::

    proc = await asyncio.create_subprocess_exec(
        *cmd, ..., start_new_session=True,
    )
    pgid = proc.pid  # start_new_session guarantees pgid == pid at spawn

    try:
        await proc.communicate(...)
    except TimeoutError:
        await terminate_process_group(proc, pgid, grace_secs=5.0)

Why the caller captures pgid instead of the helper calling
``os.getpgid(proc.pid)``:
- Once ``proc`` has been reaped (``proc.returncode is not None``), the kernel
  is free to reuse that PID for an unrelated process.  ``os.getpgid`` on a
  reused PID returns the *new* owner's group — which has, in practice,
  ended up being the user ``systemd --user`` manager's group and killed the
  user's entire login session (see root cause for task 845).
- Capturing ``pgid`` at spawn eliminates that TOCTOU entirely.  After spawn,
  ``start_new_session=True`` guarantees the child is the leader of its own
  group with ``pgid == pid``.  The captured int is frozen; it is never
  refreshed from a possibly-reaped PID.

Safety checks
-------------
Even with a correctly-captured pgid, ``terminate_process_group`` refuses to
``killpg`` any of the following as defence-in-depth:

- ``pgid <= 1`` (init or invalid)
- ``pgid == os.getpid()`` (ourselves)
- ``pgid == os.getppid()`` (our parent)
- ``pgid == os.getpgrp()`` (our own process group — hitting this would kill
  our own orchestrator/tests)
- ``pgid != proc.pid`` (mismatch — a caller corrupted the capture, or the
  ``proc`` object was somehow swapped)

If any check fires, the helper logs an error and returns without signalling.

Limitations
-----------
If a grandchild calls ``setsid()`` on its own it will escape into a new
session and process group, making it invisible to ``killpg``.  This is
acceptable for the current codebase because cargo/rustc do not call
``setsid``.  Git sub-processes spawned by short-lived helpers are out of
scope (they are bounded and never appear in stuck-process incidents).
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import signal
import time
from collections.abc import Iterable
from pathlib import Path

logger = logging.getLogger(__name__)


def snapshot_process_group(pgid: int) -> str:
    """Return a human-readable snapshot of all processes in process group *pgid*.

    Walks ``/proc`` to find processes whose process-group id matches *pgid*,
    then formats one line per process with pid, ppid, state, wchan (the kernel
    function the process is currently blocked in), comm (executable name, capped
    at 15 chars by the kernel), and cmdline (full argv from ``/proc/<pid>/cmdline``,
    truncated to ~200 chars — identifies the culprit process by name and arguments
    rather than just the short comm).

    Designed to be called from ``_run_subprocess``'s ``TimeoutError`` handler
    **before** ``proc.terminate()`` / ``terminate_process_group`` — at that
    point the wedged child processes are still alive and their ``/proc`` entries
    are readable.  After the kill the group may vanish mid-iteration; all I/O
    is wrapped in try/except so this function *never* raises.

    Returns a non-empty diagnostic string (with a header row) when at least one
    process belongs to *pgid*, or a benign "no processes found" note otherwise.

    Linux-specific: depends on ``/proc/<pid>/stat``, ``/proc/<pid>/wchan``,
    and ``/proc/<pid>/comm``.  The whole ``proc_group`` module already relies on
    Linux semantics (``os.killpg`` / ``os.getpgrp``), so this is acceptable.
    """
    try:
        return _snapshot_process_group_unsafe(pgid)
    except Exception:
        # Belt-and-suspenders: if the outer try fails (unusual /proc layout,
        # permissions, or unexpected exception), return a diagnostic string
        # rather than propagating.
        return f'snapshot_process_group({pgid}): unexpected error — see logs'


def _snapshot_process_group_unsafe(pgid: int) -> str:
    """Implement snapshot_process_group; may raise — caller wraps in try/except."""
    if pgid <= 0:
        return f'snapshot_process_group({pgid}): pgid <= 0 — no snapshot taken'

    proc_dir = Path('/proc')
    if not proc_dir.exists():
        return f'snapshot_process_group({pgid}): /proc not available'

    rows: list[str] = []
    try:
        entries = list(proc_dir.iterdir())
    except OSError:
        return f'snapshot_process_group({pgid}): could not list /proc'

    for entry in entries:
        if not entry.name.isdigit():
            continue
        pid = int(entry.name)

        # Read /proc/<pid>/stat to get pgrp (field 5, 1-indexed), ppid (4), state (3).
        # stat format: "pid (comm) state ppid pgrp ..."
        try:
            stat_text = (entry / 'stat').read_text()
        except OSError:
            continue

        # Parse: find the closing ')' of the comm field to handle spaces/parens in names.
        try:
            rparen = stat_text.rfind(')')
            if rparen < 0:
                continue
            tail = stat_text[rparen + 2 :]  # skip ') '
            fields = tail.split()
            # fields[0]=state, [1]=ppid, [2]=pgrp, [3]=session, ...
            state = fields[0]
            ppid = int(fields[1])
            pgrp = int(fields[2])
        except (IndexError, ValueError):
            continue

        if pgrp != pgid:
            continue

        # Read comm (short executable name, capped at 15 chars by the kernel).
        try:
            comm = (entry / 'comm').read_text().strip()
        except OSError:
            comm = '?'

        # Read wchan (kernel function the task is blocked in, or '0' when running).
        try:
            wchan = (entry / 'wchan').read_text().strip()
        except OSError:
            wchan = '?'

        # Read /proc/<pid>/cmdline (NUL-separated argv → spaces) for the full
        # command-line.  Kernel threads have an empty cmdline; fall back to
        # comm so the field is always populated.  Truncate to ~200 chars for
        # log friendliness.  Mirrors the comm/wchan try/except idiom so
        # snapshot_process_group never raises (module invariant).
        try:
            raw = (entry / 'cmdline').read_bytes()
            cmdline = raw.replace(b'\x00', b' ').decode('utf-8', 'replace').strip()
            if not cmdline:
                cmdline = comm  # kernel thread — fall back to short comm
            if len(cmdline) > 200:
                cmdline = cmdline[:200] + '…'
        except OSError:
            cmdline = '?'

        rows.append(
            f'  pid={pid} ppid={ppid} state={state} wchan={wchan} comm={comm} cmdline={cmdline}'
        )

    if not rows:
        return f'snapshot_process_group({pgid}): no processes found in group'

    header = f'snapshot_process_group({pgid}): {len(rows)} process(es) in group:'
    return '\n'.join([header] + rows)


def _unsafe_pgid_reason(pgid: int, proc_pid: int | None) -> str | None:
    """Return a reason string if *pgid* is unsafe to killpg, else ``None``.

    Applied as defence-in-depth: even if a caller (or a PID-reuse race) hands
    us a pgid that targets the user session or ourselves, we refuse.
    """
    if pgid <= 1:
        return f'pgid <= 1 ({pgid!r})'
    if pgid == os.getpid():
        return f'pgid == os.getpid() ({pgid})'
    try:
        ppid = os.getppid()
    except OSError:
        ppid = None
    if ppid is not None and pgid == ppid:
        return f'pgid == os.getppid() ({pgid})'
    try:
        own_pgrp = os.getpgrp()
    except OSError:
        own_pgrp = None
    if own_pgrp is not None and pgid == own_pgrp:
        return f'pgid == os.getpgrp() ({pgid})'
    if proc_pid is not None and pgid != proc_pid:
        return f'pgid ({pgid}) != proc.pid ({proc_pid}) — captured value corrupted'
    return None


async def terminate_process_group(
    proc: asyncio.subprocess.Process,
    pgid: int,
    *,
    grace_secs: float = 5.0,
) -> None:
    """Send SIGTERM to *pgid*, then SIGKILL if the group outlives *grace_secs*.

    *pgid* must be the process-group id captured immediately after spawning
    *proc* with ``start_new_session=True`` (at which point ``pgid == proc.pid``
    by POSIX guarantee).  Passing a value fetched via ``os.getpgid(proc.pid)``
    after *proc* may have been reaped is unsafe — see module docstring.

    Behaviour:
    1. If *proc* has already been reaped (``returncode is not None``), return
       immediately.  The group is already gone with the leader.
    2. Sanity-check *pgid* via :func:`_unsafe_pgid_reason`.  If unsafe, log
       and return without signalling.
    3. ``os.killpg(pgid, SIGTERM)``.  Wait up to *grace_secs* for *proc* to
       exit.
    4. If *proc* is still alive, ``os.killpg(pgid, SIGKILL)`` and wait
       another *grace_secs*.

    All ``killpg`` calls are wrapped in ``contextlib.suppress(ProcessLookupError,
    OSError)`` because the group may vanish between our liveness check and
    the signal dispatch.
    """
    if proc.returncode is not None:
        # Already reaped — the entire group has exited along with the leader.
        return

    reason = _unsafe_pgid_reason(pgid, proc.pid)
    if reason is not None:
        logger.error(
            'terminate_process_group: refusing to killpg — %s. '
            'This indicates a bug in the caller; proc will NOT be signalled.',
            reason,
        )
        return

    with contextlib.suppress(ProcessLookupError, OSError):
        os.killpg(pgid, signal.SIGTERM)

    try:
        await asyncio.wait_for(proc.wait(), grace_secs)
    except TimeoutError:
        # Re-check liveness before escalating.
        if proc.returncode is not None:
            return
        with contextlib.suppress(ProcessLookupError, OSError):
            os.killpg(pgid, signal.SIGKILL)
        with contextlib.suppress(TimeoutError):
            await asyncio.wait_for(proc.wait(), grace_secs)


# ---------------------------------------------------------------------------
# At-or-under /proc scan + foreign-pgid reap (task 2828 startup survivor
# barrier). These generalize the two idioms above — the snapshot /proc-walk
# and the terminate_process_group SIGTERM→wait→SIGKILL escalation — from a
# single owned proc handle to "find every process group touching a directory
# subtree, and reap them". reify ships these primitives; the orchestrator's
# GitOps wires the path + knob and invokes them off the event loop.
#
# Linux-specific: depends on /proc/<pid>/{stat,cwd,fd/*,maps}. The whole
# module already assumes Linux (os.killpg / os.getpgrp), so this is fine.
# ---------------------------------------------------------------------------


def _path_at_or_under(candidate: str, root: str) -> bool:
    """Return True iff *candidate* is *root* exactly or a descendant of it.

    Boundary-safe: exact equality OR ``startswith(root + os.sep)``, so a
    sibling like ``/x/_merge-verifyXYZ`` never matches root ``/x/_merge-verify``.
    """
    return candidate == root or candidate.startswith(root + os.sep)


def _pid_references_path_at_or_under(entry: Path, root: str) -> bool:
    """True if the pid at *entry* (=/proc/<pid>) touches a path at-or-under root.

    Short-circuits cheapest-first: cwd, then open fds, then mmap'd pathnames
    from ``maps``.  Every per-pid I/O is wrapped so a vanished or
    permission-denied pid is skipped rather than raising (module invariant).
    """
    # 1. cwd — the most common and cheapest signal (cargo/rustc run in the tree).
    with contextlib.suppress(OSError):
        if _path_at_or_under(os.readlink(entry / 'cwd'), root):
            return True

    # 2. open fds — an open build artifact / lockfile under the tree.
    with contextlib.suppress(OSError):
        for fd_entry in (entry / 'fd').iterdir():
            try:
                target = os.readlink(fd_entry)
            except OSError:
                continue
            if _path_at_or_under(target, root):
                return True

    # 3. mmap'd pathnames — an mmap'd .rlib / .so living under the tree.
    try:
        maps_text = (entry / 'maps').read_text()
    except OSError:
        return False
    for line in maps_text.splitlines():
        # maps line: "addr perms offset dev inode  pathname" — pathname is the
        # 6th field (may contain spaces; keep it whole with maxsplit=5). Skip
        # anonymous/special regions ([heap], [stack], anon → no leading '/').
        parts = line.split(maxsplit=5)
        if len(parts) < 6:
            continue
        pathname = parts[5]
        if not pathname.startswith('/'):
            continue
        if _path_at_or_under(pathname, root):
            return True
    return False


def _scan_process_groups_under_path_unsafe(root: str, exclude_pgids: Iterable[int]) -> set[int]:
    """Implement scan_process_groups_under_path; may raise — caller wraps it."""
    result: set[int] = set()
    exclude = frozenset(exclude_pgids)

    proc_dir = Path('/proc')
    if not proc_dir.exists():
        return result
    try:
        entries = list(proc_dir.iterdir())
    except OSError:
        return result

    for entry in entries:
        if not entry.name.isdigit():
            continue

        # Parse pgrp from /proc/<pid>/stat (field 5, after the parenthesized
        # comm). Reuses _snapshot_process_group_unsafe's rfind(')') idiom so a
        # comm containing spaces/parens is handled correctly.
        try:
            stat_text = (entry / 'stat').read_text()
        except OSError:
            continue
        try:
            rparen = stat_text.rfind(')')
            if rparen < 0:
                continue
            fields = stat_text[rparen + 2 :].split()
            pgrp = int(fields[2])  # fields: state, ppid, pgrp, ...
        except (IndexError, ValueError):
            continue

        if pgrp in exclude or pgrp in result:
            # Excluded, or already recorded via another pid in the same group —
            # skip the expensive fd/maps inspection.
            continue

        if _pid_references_path_at_or_under(entry, root):
            result.add(pgrp)

    return result


def scan_process_groups_under_path(
    root: str | os.PathLike[str],
    *,
    exclude_pgids: Iterable[int] = frozenset(),
) -> set[int]:
    """Return pgids of processes whose cwd / an open fd / an mmap'd path is
    at-or-under *root*.

    Walks ``/proc``; for each pid parses its process-group id and checks
    whether it references *root* (or any descendant).  *exclude_pgids* drops
    the caller's own group so the scan never targets itself.

    Never raises — every per-pid I/O is wrapped, and the whole walk is
    belt-and-suspenders-wrapped so a vanishing pid, a permission-denied read,
    or an unexpected /proc layout yields an empty/partial set rather than
    propagating (mirrors snapshot_process_group's invariant).
    """
    try:
        return _scan_process_groups_under_path_unsafe(str(root), exclude_pgids)
    except Exception:
        logger.warning(
            'scan_process_groups_under_path(%s): unexpected error — returning empty set',
            root,
            exc_info=True,
        )
        return set()


def _pgid_alive(pgid: int) -> bool:
    """True while *pgid* still exists (``killpg(pgid, 0)`` succeeds).

    ProcessLookupError (ESRCH) → gone.  PermissionError (EPERM) → the pgid was
    recycled to another user's group, so the group we were reaping is gone;
    treat as gone.  Any other OSError → treat as gone (fail-safe: never report
    a group as alive on an ambiguous error, which would falsely mark it
    'survived').
    """
    try:
        os.killpg(pgid, 0)
        return True
    except (ProcessLookupError, PermissionError, OSError):
        return False


def _drop_dead_pgids(pgids: list[int], grace_secs: float, poll_step: float) -> list[int]:
    """Bounded-poll *pgids*, returning those still alive after *grace_secs*.

    Checks liveness immediately (so an already-dead group needs no sleep),
    then polls every *poll_step* until the deadline.
    """
    remaining = list(pgids)
    deadline = time.monotonic() + max(0.0, grace_secs)
    while True:
        remaining = [p for p in remaining if _pgid_alive(p)]
        if not remaining or time.monotonic() >= deadline:
            return remaining
        time.sleep(poll_step)


def reap_process_groups(
    pgids: Iterable[int],
    *,
    grace_secs: float = 5.0,
    poll_step: float = 0.1,
) -> dict[int, str]:
    """SIGTERM→wait→SIGKILL every pgid in *pgids*; return a per-pgid outcome.

    Outcomes:
    - ``'reaped'``       — the group is gone.
    - ``'survived'``     — still alive after SIGKILL + *grace_secs* (rare;
      only same-user-uncooperative or unsignalable groups).
    - ``'refused:<reason>'`` — an unsafe pgid (``pgid <= 1`` / self / parent /
      own group per :func:`_unsafe_pgid_reason`); never signalled at all.

    Generalizes :func:`terminate_process_group`'s escalation from one owned
    proc handle to a set of foreign pgids reaped by number.  All ``killpg``
    calls suppress ProcessLookupError/OSError because a group may vanish
    between the liveness probe and the signal.  This is a *synchronous*
    blocking routine (it ``time.sleep``s) — call it via ``asyncio.to_thread``
    off the event loop.
    """
    outcomes: dict[int, str] = {}
    safe: list[int] = []
    for pgid in pgids:
        reason = _unsafe_pgid_reason(pgid, None)
        if reason is not None:
            outcomes[pgid] = f'refused:{reason}'
            logger.error(
                'reap_process_groups: refusing to killpg pgid %s — %s',
                pgid,
                reason,
            )
        else:
            safe.append(pgid)

    if not safe:
        return outcomes

    for pgid in safe:
        with contextlib.suppress(ProcessLookupError, OSError):
            os.killpg(pgid, signal.SIGTERM)

    remaining = _drop_dead_pgids(safe, grace_secs, poll_step)

    if remaining:
        for pgid in remaining:
            with contextlib.suppress(ProcessLookupError, OSError):
                os.killpg(pgid, signal.SIGKILL)
        remaining = _drop_dead_pgids(remaining, grace_secs, poll_step)

    remaining_set = set(remaining)
    for pgid in safe:
        outcomes[pgid] = 'survived' if pgid in remaining_set else 'reaped'
    return outcomes
