"""Host-side cancellation contract for ``orchestrator verify-merge``.

Overview
--------
When ``orchestrator verify-merge --request-id X`` starts, it:

1. Calls :func:`start_own_process_group` to join a new session/process group
   (``os.setsid``), isolating itself from the SSH session that launched it.
   After ``setsid`` the process's pgid equals its pid, so one integer serves
   as both the tree-walk root and the ``killpg`` backstop target.

2. Writes that pgid to a *pgid file* via :func:`write_pgid_file` so a
   concurrent ``orchestrator cancel-verify --request-id X`` can find it.

3. Removes the file on exit (normal or exceptional) via :func:`remove_pgid_file`.

The verify bundle spawns every build/test command with
``start_new_session=True`` (``verify.py`` ``_run_cmd``), making each command
the leader of its own session and process group — it **escapes** the
``verify-merge`` process group.  A plain ``killpg(recorded_group)`` therefore
strands those zombie builds.  :func:`cancel_request` fixes this by:

* Snapshotting the full ``/proc`` PPID map *before* sending any signals
  (killing the root reparents survivors to init, losing tree linkage).
* Walking the descendant tree of the recorded pgid to collect every
  ``start_new_session`` escape.
* ``SIGKILL``-ing every collected descendant **and** firing a
  ``killpg(pgid, SIGKILL)`` backstop for any same-group stragglers.

pgid-file directory
-------------------
Files live at::

    <worktree_base>/.merge_verify_pgids/<sanitized_request_id>

This mirrors the existing host-state precedent
``<worktree_base>/.merge_verify_host_attempts`` (``git_ops.py``:1864):
host-side, per-project, survives stateless CLI invocations, and never
inside a registered worktree so it is never pruned or ``git``-cleaned.
``worktree_base`` is derived via
``GitOps(config.git, config.project_root).worktree_base`` (single source
of truth, same as ``git_ops.py``:480).

Cross-host rollout
------------------
Both the server and the laptop host run the ``df`` checkout, so landing this
module on ``main`` ships the cancellation contract to the laptop via its
normal checkout sync — no separate deploy step required.
"""

from __future__ import annotations

import os
import re
import signal
from collections import deque
from pathlib import Path

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Name of the directory (under ``worktree_base``) where pgid files are kept.
#: Mirrors the ``.merge_verify_host_attempts`` file precedent: host-side,
#: per-project, never pruned/git-cleaned.
PGID_DIR_NAME: str = '.merge_verify_pgids'

# Allowlist for request_id characters: alphanumeric, dash, underscore, dot.
# Anything outside this set is stripped so the sanitized result is a safe
# filename that cannot path-traverse out of PGID_DIR_NAME.
_SAFE_RE = re.compile(r'[^A-Za-z0-9\-_.]')


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------


def pgid_dir(worktree_base: Path) -> Path:
    """Return the pgid-file directory (not necessarily created yet)."""
    return worktree_base / PGID_DIR_NAME


def pgid_file(worktree_base: Path, request_id: str) -> Path:
    """Return the path for the pgid file for *request_id*.

    *request_id* is sanitized: only ``[A-Za-z0-9-_.]`` chars are kept, so a
    hostile id such as ``'../escape'`` or ``'a/b'`` cannot path-traverse
    outside :data:`PGID_DIR_NAME`.  If the result is empty after sanitization
    a fallback of ``'_unknown'`` is used.
    """
    safe = _SAFE_RE.sub('', request_id)
    if not safe:
        safe = '_unknown'
    return pgid_dir(worktree_base) / safe


# ---------------------------------------------------------------------------
# File lifecycle
# ---------------------------------------------------------------------------


def write_pgid_file(path: Path, pgid: int) -> None:
    """Atomically write *pgid* to *path*, creating parent directories first.

    Uses a sibling tmp file + ``os.replace`` for atomicity so a concurrent
    reader never sees a partial write.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp')
    tmp.write_text(str(pgid))
    os.replace(tmp, path)


def remove_pgid_file(path: Path) -> None:
    """Remove *path* if it exists; silently a no-op when absent (idempotent)."""
    try:
        path.unlink()
    except FileNotFoundError:
        pass


# ---------------------------------------------------------------------------
# Process-group management
# ---------------------------------------------------------------------------


def start_own_process_group() -> int:
    """Call ``os.setsid()`` to start a new session/process group and return the pgid.

    After ``setsid`` the process's ``pgid`` equals its ``pid``, so the single
    returned integer serves as both the ``/proc`` tree-walk root and the
    ``os.killpg`` backstop target in :func:`cancel_request`.

    The purpose of ``setsid`` is to isolate ``verify-merge`` from the SSH
    session that launched it (``sshd`` is an ancestor, not a descendant, so
    the descendant-tree walk already prevents touching it).  The ``killpg``
    backstop adds a second layer of protection for same-group stragglers.

    If ``setsid`` raises ``OSError`` (``EPERM`` — the process is already a
    session leader, which can happen in a bare ``ssh`` invocation without a
    controlling terminal) we fall back to returning ``os.getpgrp()`` so that a
    valid kill target is always recorded.
    """
    try:
        os.setsid()
        return os.getpgrp()
    except OSError:
        # Already a session leader; return the current group as a fallback.
        return os.getpgrp()


# ---------------------------------------------------------------------------
# /proc PPID map & descendant walk
# ---------------------------------------------------------------------------


def read_ppid_map() -> dict[int, int]:
    """Parse ``/proc/*/stat`` and return ``{pid: ppid}`` for every live process.

    The ``comm`` field (field 2) may contain spaces and parentheses, so we
    parse by finding the *last* ``') '`` separator (``rsplit``) rather than
    splitting on whitespace naïvely.  Vanished or unreadable entries are
    silently skipped (the process exited between the ``glob`` and the
    ``read``).
    """
    ppid_map: dict[int, int] = {}
    proc = Path('/proc')
    for entry in proc.iterdir():
        if not entry.name.isdigit():
            continue
        stat_path = entry / 'stat'
        try:
            raw = stat_path.read_text()
        except OSError:
            continue
        # Format: "pid (comm) state ppid ..."
        # rsplit on ') ' to skip over the comm field safely.
        try:
            right = raw.rsplit(') ', 1)[1]
            fields = right.split()
            pid = int(entry.name)
            ppid = int(fields[1])  # field index after stripping pid+comm+state
            ppid_map[pid] = ppid
        except (IndexError, ValueError):
            continue
    return ppid_map


def collect_descendants(root: int, ppid_map: dict[int, int]) -> set[int]:
    """Return the set of all transitive descendant pids of *root* (BFS, cycle-safe).

    *root* itself is NOT included in the result.  Processes not reachable from
    *root* (e.g. unrelated pid 1 and its subtree) are excluded.

    The walk is iterative with a ``visited`` set so a cyclic ``ppid_map``
    (e.g. from a test fixture) cannot loop forever.
    """
    # Build a children map for efficient BFS
    children: dict[int, list[int]] = {}
    for pid, ppid in ppid_map.items():
        children.setdefault(ppid, []).append(pid)

    visited: set[int] = set()
    queue: deque[int] = deque([root])
    while queue:
        current = queue.popleft()
        for child in children.get(current, []):
            if child not in visited:
                visited.add(child)
                queue.append(child)
    # Exclude root itself (it wasn't added to visited on entry)
    visited.discard(root)
    return visited


# ---------------------------------------------------------------------------
# Cancellation
# ---------------------------------------------------------------------------


def cancel_request(
    path: Path,
    *,
    ppid_map_provider=read_ppid_map,
    kill=os.kill,
    killpg=os.killpg,
) -> int:
    """Cancel the verify-merge identified by the pgid file at *path*.

    Algorithm
    ---------
    1. Read and parse the pgid file.

       * ``FileNotFoundError`` → nothing to cancel; return 0.
       * Non-integer content → corrupt file; remove it and return 0.

    2. **Snapshot first**: call *ppid_map_provider* to get the full ``/proc``
       PPID map *before* sending any signal.  Killing the root reparents
       survivors to init and severs the tree linkage — the snapshot must
       precede all kills.

    3. ``collect_descendants(pgid, ppid_map)`` to find every
       ``start_new_session`` escape (they remain descendants even after
       escaping the process group).

    4. ``SIGKILL`` every target (descendants ∪ {pgid}):

       * ``ProcessLookupError`` → already dead, counts as success.
       * ``PermissionError`` → live process we cannot kill; record failure.

    5. ``killpg(pgid, SIGKILL)`` backstop — suppresses ``OSError`` /
       ``ProcessLookupError`` (group may already be gone).

    6. If any live process could not be killed → return 1 and **retain** the
       pgid file (lets a retry or β's quarantine probe act on it).
       Otherwise remove the file and return 0.

    Return value
    ------------
    * ``0`` — all processes killed (or already gone), file removed.
    * ``1`` — at least one live process raised ``PermissionError``; file kept.
    """
    # Step 1: read pgid file
    try:
        raw = path.read_text().strip()
    except FileNotFoundError:
        return 0

    try:
        pgid = int(raw)
    except ValueError:
        # Corrupt / empty file — clean up and treat as nothing to do
        remove_pgid_file(path)
        return 0

    # Step 2: snapshot PPID map BEFORE any kills (invariant: snapshot precedes kill)
    ppid_map = ppid_map_provider()

    # Step 3: collect every descendant (start_new_session escapes included)
    descendants = collect_descendants(pgid, ppid_map)
    targets = descendants | {pgid}

    # Step 4: SIGKILL every target
    failed = False
    for pid in targets:
        try:
            kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass  # already dead — success
        except PermissionError:
            failed = True  # live but unkillable

    # Step 5: killpg backstop (suppresses errors — group may already be gone)
    try:
        killpg(pgid, signal.SIGKILL)
    except (ProcessLookupError, OSError):
        pass

    # Step 6: decide outcome
    if failed:
        # Retain the file so a retry can act on it
        return 1
    remove_pgid_file(path)
    return 0
