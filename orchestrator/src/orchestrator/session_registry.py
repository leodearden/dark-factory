"""orchestrator.session_registry — durable session-registry substrate (Attention Rail T3).

PRD: plans/session-attention-rail-prd.md T3 (spine root; §4.1-4.2, §4.8, §6 G5).

This module runs the global, cross-project session registry at
``~/.claude/fleet/sessions/<slug>/record.json``: the reaps, the T7 leases and
the fleet registry CLI. The record schema and its store live in
``session_records``, THIS session's identity in ``session_identity``, and the
decision verbs in ``decision_cli``.

It imports only the stdlib, this checkout's stdlib-only ``shared`` leaves and
its stdlib-only siblings, layered so each imports only downward:
``fleet_files``, ``session_identity`` <- ``session_records`` <-
``session_registry`` and ``fleet_files`` <- ``decision_registry`` <-
``decision_cli`` <- ``session_registry``, which also imports ``fleet_files``
and ``session_identity`` directly. Nothing else from the orchestrator, so it
can be:

- invoked directly by ``skills/spawn/spawn-claude.sh`` and the SKILL.md verb
  lines via an absolute path (no venv/PYTHONPATH/install required: the
  bootstrap below reaches ``shared/src`` from ``__file__``), and
- imported as ``orchestrator.session_registry`` by downstream Python
  consumers (T4 verify, T7 leases).
"""

from __future__ import annotations

import argparse
import contextlib
import dataclasses
import json
import logging
import os
import re
import shutil
import subprocess
import sys
import time
from collections.abc import Callable, Iterator, Mapping, Sized
from collections.abc import Set as AbstractSet
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING, Any

# NOTE: this module is executed by absolute path from
# skills/spawn/spawn-claude.sh and the SKILL.md verb lines with no venv,
# install or PYTHONPATH, so the only non-stdlib imports allowed at module
# scope are this checkout's stdlib-only `shared` leaves, reached through the
# bootstrap below. It APPENDS, so an installed or PYTHONPATH `shared` wins.
# No third-party import, and no unguarded `from orchestrator import <sibling>`:
# running a script by path puts only the script's OWN directory on sys.path,
# so intra-orchestrator imports fail there even though they work for
# consumers that import this module with PYTHONPATH set. Stdlib-only
# siblings are imported through the load-mode block below the shared imports.
# Pinned by the tier-1 (bare shell, NO PYTHONPATH) row of
# _BARE_SHELL_ENTRYPOINTS in
# test_session_registry.py::TestStdlibOnlySelfContainment, and mutation-tested
# there via _MUST_BE_REJECTED.
if (
    _CHECKOUT_SHARED_SRC := str(Path(__file__).resolve().parents[3] / 'shared' / 'src')
) not in sys.path and Path(_CHECKOUT_SHARED_SRC).is_dir():
    sys.path.append(_CHECKOUT_SHARED_SRC)

from shared.proc_liveness import pid_alive
from shared.process_identity import Liveness, ProcessIdentity
from shared.session_liveness import capture_for_record, judge

# Siblings are imported relative to orchestrator.session_registry, or else from this
# file's own directory: loaded by path, as tier-1 __main__ or under a foreign dotless
# name, that directory is on sys.path only if Python put it there (not under -P, -I or
# PYTHONSAFEPATH, nor for a foreign-name load), so it is borrowed for these imports
# alone. Each mode is pinned in test_session_registry.py::TestStdlibOnlySelfContainment.
# TYPE_CHECKING points the type checker at the package arm, the only one it can resolve.
if TYPE_CHECKING or __package__:
    from . import decision_cli, session_identity, session_records
    from .fleet_files import (
        ATOMIC_WRITE_TEMP_SUFFIX,
        atomic_write_text,
        exclusive_sidecar_lock,
        fleet_root,
    )
else:
    _OWN_DIR = str(Path(__file__).resolve().parent)
    _OWN_DIR_BORROWED = _OWN_DIR not in sys.path
    if _OWN_DIR_BORROWED:
        sys.path.append(_OWN_DIR)
    try:
        import decision_cli
        import session_identity
        import session_records
        from fleet_files import (
            ATOMIC_WRITE_TEMP_SUFFIX,
            atomic_write_text,
            exclusive_sidecar_lock,
            fleet_root,
        )
    finally:
        if _OWN_DIR_BORROWED:
            sys.path.remove(_OWN_DIR)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# TTL / pid stale-record reaper (PRD §4.8)
# ---------------------------------------------------------------------------

TERMINAL_TTL = timedelta(hours=24)
"""How long a terminal (EXITED/FAILED_TO_START) record survives after its
last write before the reaper reclaims it, regardless of launcher_pid."""

NON_TERMINAL_HEARTBEAT_TTL = timedelta(hours=1)
"""How long a non-terminal record survives with its launcher judged DEAD and
no fresh write (heartbeat) before the reaper reclaims it. A launcher not
judged DEAD retains the record regardless of age."""


def _swept_as_dead(
    identity: ProcessIdentity | None, pid: int, *, sweep: str, subject: str
) -> bool:
    """Whether a liveness sweep may treat *pid* as dead: ``judge``'s DEAD, with an OTHER_HOST skip logged."""
    verdict = judge(identity, pid, pid_alive=pid_alive)
    if verdict is Liveness.OTHER_HOST and identity is not None:
        logger.info('%s: skipped %s, reason=other_host host=%s', sweep, subject, identity.host)
    return verdict is Liveness.DEAD


@dataclass(frozen=True)
class ReapedSessionRecord:
    """One session-registry record directory removed by reap_stale_records.

    path: the removed ``<slug>/`` directory (not the record.json file).
    session_slug: the session's identity/directory key, derived from the
        directory name itself (never from the record body).
    reason: why it was reaped -- one of ``'terminal_ttl'`` (a terminal
        record older than TERMINAL_TTL), ``'stale_pid'`` (a non-terminal
        record whose launcher is judged DEAD and older than
        NON_TERMINAL_HEARTBEAT_TTL), or ``'corrupt'`` (record.json missing
        or unparseable and older than NON_TERMINAL_HEARTBEAT_TTL).
    """

    path: Path
    session_slug: str
    reason: str


def reap_stale_records(
    root: Path | str | None = None,
    *,
    now: datetime | None = None,
) -> list[ReapedSessionRecord]:
    """Sweep ``<root>/sessions/*/`` and remove stale session-record directories.

    Mirrors the ``reap_interactive_worktrees``/``interactive.json`` idiom
    (git_ops.py): identity (which directory to remove) is derived from the
    *path* -- the slug directory name -- never from the record body, so a
    missing or corrupt record.json is still reapable. Age is measured from
    record.json's mtime (bumped by every session_records.write_record/session_records.refresh_record call,
    i.e. its heartbeat); when record.json itself is absent, the slug
    directory's own mtime is used instead so identity-only reap still works.

    Rules (first match wins):
    - record body unreadable (missing or corrupt) and age >
      NON_TERMINAL_HEARTBEAT_TTL -> reaped, reason='corrupt'.
    - status is terminal (EXITED/FAILED_TO_START) and age > TERMINAL_TTL ->
      reaped, reason='terminal_ttl' (independent of launcher_pid liveness).
    - status is non-terminal, age > NON_TERMINAL_HEARTBEAT_TTL, and the
      launcher is judged DEAD by
      ``shared/src/shared/session_liveness.py::judge`` -> reaped,
      reason='stale_pid'. A T3-hosted
      record (``display.kind == 't3'``) is exempt while
      ``_t3_thread_sweep_owns`` holds, since its liveness is its thread.
    - otherwise -> kept.

    Directories are visited in ``sorted(iterdir())`` order (directory-name
    order, not age order).

    *now* is injectable for deterministic tests; defaults to the real UTC
    clock.
    """
    if now is None:
        now = datetime.now(UTC)
    base = session_records.sessions_dir(root)
    reaped: list[ReapedSessionRecord] = []
    if not base.is_dir():
        return reaped

    for slug_dir in sorted(base.iterdir()):
        if not slug_dir.is_dir():
            continue
        slug = slug_dir.name
        record_path = slug_dir / 'record.json'
        age_source = record_path if record_path.is_file() else slug_dir
        try:
            mtime = age_source.stat().st_mtime
        except OSError:
            continue  # vanished mid-sweep (e.g. concurrent reap) -- skip it
        age = now - datetime.fromtimestamp(mtime, tz=UTC)

        try:
            record = session_records.read_record(slug, root=root)
        except (FileNotFoundError, session_records.CorruptSessionRecord):
            reason = 'corrupt' if age > NON_TERMINAL_HEARTBEAT_TTL else None
        else:
            if record.status in session_records.TERMINAL_STATUSES:
                reason = 'terminal_ttl' if age > TERMINAL_TTL else None
            else:
                stale = (
                    not _t3_thread_sweep_owns(record, slug, root)
                    and age > NON_TERMINAL_HEARTBEAT_TTL
                    and _swept_as_dead(
                        record.launcher_identity,
                        record.launcher_pid,
                        sweep='reap_stale_records',
                        subject=slug,
                    )
                )
                reason = 'stale_pid' if stale else None

        if reason is not None:
            try:
                shutil.rmtree(slug_dir)
            except OSError:
                # A single unreapable directory (permission error, a file held
                # open, an ENOTEMPTY race with a concurrent writer) must not
                # abort the sweep -- log and move on to the next candidate.
                logger.error('reap_stale_records: failed to remove %s', slug_dir, exc_info=True)
                continue
            reaped.append(ReapedSessionRecord(path=slug_dir, session_slug=slug, reason=reason))

    return reaped


_POINTER_TEMP_FILE_GRACE = timedelta(minutes=5)
"""How long a ``sessions-by-pid/`` temp file is presumed to belong to a writer
still between ``mkstemp`` and ``os.replace``. Past it, the writer was killed."""


def _is_abandoned_temp_file(path: Path) -> bool:
    try:
        mtime = path.stat().st_mtime
    except OSError:
        return False
    age = datetime.now(UTC) - datetime.fromtimestamp(mtime, tz=UTC)
    return age > _POINTER_TEMP_FILE_GRACE


def _is_stale_session_pointer(path: Path, root: Path | str | None) -> bool:
    if path.name.endswith(ATOMIC_WRITE_TEMP_SUFFIX):
        return _is_abandoned_temp_file(path)
    try:
        pid = int(path.name)
    except ValueError:
        return True
    if pid <= 0 or path.name != str(pid):
        return True
    slug = session_records.read_session_pointer(path)
    if (
        slug is None
        or not session_records.is_record_key(slug)
        or not session_records.record_path_for_slug(slug, root=root).is_file()
    ):
        return True
    return _swept_as_dead(
        _pointed_owner_identity(slug, pid, root),
        pid,
        sweep='reap_stale_session_pointers',
        subject=path.name,
    )


def _pointed_owner_identity(
    slug: str, pid: int, root: Path | str | None
) -> ProcessIdentity | None:
    """The owner identity *slug*'s record holds for *pid*; None when unreadable or naming another pid."""
    try:
        identity = session_records.read_record(slug, root=root).claude_owner_identity
    except (session_records.CorruptSessionRecord, OSError):
        return None
    return identity if identity is not None and identity.pid == pid else None


def _is_stale_claude_session_pointer(path: Path, root: Path | str | None) -> bool:
    if path.name.endswith(ATOMIC_WRITE_TEMP_SUFFIX):
        return _is_abandoned_temp_file(path)
    if not session_records.is_claude_session_pointer_name(path.name):
        return True
    return session_records.resolve_session_slug_for_claude_session_id(path.name, root=root) is None


def _is_stale_t3_thread_pointer(path: Path, root: Path | str | None) -> bool:
    if path.name.endswith(ATOMIC_WRITE_TEMP_SUFFIX):
        return _is_abandoned_temp_file(path)
    if not session_records.is_t3_thread_pointer_name(path.name):
        return True
    return session_records.resolve_session_slug_for_t3_thread(path.name, root=root) is None


def _drop_t3_thread_pointer_if_stale(record: session_records.SessionRecord, root: Path | str | None) -> None:
    """After a terminal write of *record*, unlink its thread's pointer unless a live record holds it.

    A failed unlink is logged, never raised: the reap retries it.
    """
    thread_id = session_records.t3_thread_of(record)
    if thread_id is None:
        return
    path = session_records.t3_thread_pointer_path(thread_id, root=root)
    if not _is_stale_t3_thread_pointer(path, root):
        return
    try:
        path.unlink(missing_ok=True)
    except OSError as exc:
        logger.warning('t3 thread pointer drop failed: path=%s: %s', path, exc)


def _reap_pointer_dir(
    base: Path, is_stale: Callable[[Path, Path | str | None], bool], root: Path | str | None
) -> list[Path]:
    if not base.is_dir():
        return []
    entries = sorted(base.iterdir())
    removed: list[Path] = []
    for path in entries:
        if not is_stale(path, root):
            continue
        try:
            path.unlink()
        except OSError as exc:
            logger.warning('reap_stale_session_pointers: failed to remove %s: %s', path, exc)
            continue
        removed.append(path)
    logger.info(
        'reap_stale_session_pointers: removed %d of %d pointer(s) under %s',
        len(removed),
        len(entries),
        base,
    )
    return removed


def reap_stale_session_pointers(root: Path | str | None = None) -> list[Path]:
    """Remove every pointer entry that can no longer serve a reader.

    A ``sessions-by-pid/`` entry is stale when its name is not a canonical
    positive pid, its content is not a record key, the record it names has no
    ``record.json``, or its pid is judged DEAD by
    ``shared/src/shared/session_liveness.py::judge`` against that record's
    ``claude_owner_identity`` when it names the same pid, so each such entry
    costs one record read.
    A dead pid's pointer goes even while its record still exists: its owner
    can never query it again, and a process that reuses the pid must not
    inherit it.

    A ``sessions-by-claude-id/`` entry is stale when its name is not a
    session-id token or the record it names no longer vouches for it (gone,
    unreadable, or bound to another session id), so each such entry costs one
    record read.

    A ``sessions-by-t3-thread/`` entry is stale when its name is not a token
    or ``session_records.resolve_session_slug_for_t3_thread`` finds no live t3 record of that
    thread behind it, so each such entry costs one record read.

    In every dir a temp file of ``fleet_files.py::atomic_write_text`` is left to its writer
    until ``_POINTER_TEMP_FILE_GRACE`` has passed, so a sweep never unlinks it
    out from under a concurrent ``os.replace``.

    This is a separate sweep from ``reap_stale_records`` because that one
    derives identity from each record's PATH and never reads the body, so it
    cannot know which pointers name the records it removes. Returns the
    removed paths; an entry that cannot be removed is logged and skipped.
    """
    return [
        *_reap_pointer_dir(session_records.session_pointers_dir(root), _is_stale_session_pointer, root),
        *_reap_pointer_dir(
            session_records.claude_session_pointers_dir(root), _is_stale_claude_session_pointer, root
        ),
        *_reap_pointer_dir(session_records.t3_thread_pointers_dir(root), _is_stale_t3_thread_pointer, root),
    ]


ORPHAN_EXIT_CODE: int = 200
"""Synthetic exit code stamped by ``mark_orphaned_sessions_exited`` when a
non-terminal session's ``launcher_pid`` is provably dead but no real exit
code was ever recorded -- ``finish()`` never ran (SIGKILL, force-closed
terminal, crash, reboot). Distinct from every real/sentinel code
spawn-claude.sh can record (0-125 claude exit codes; 126/127/129/143/144
spawn-claude.sh sentinels; 145-199 called out as reserved-free by
spawn-claude.sh's own exit-codes comment) and from ``None`` (no exit_code
recorded yet), so a downstream consumer can tell an orphan-reaped record
from a cleanly-exited one. Deliberately a POSITIVE value at/past 200, not
-1: by POSIX/subprocess convention a NEGATIVE returncode means "killed by
signal N" (``returncode == -N``), so -1 could be misread by a consumer
applying that convention as "killed by SIGHUP". A positive sentinel clear
of every reserved range avoids that ambiguity."""


def _mark_exited_if_still_non_terminal(
    slug: str,
    root: Path | str | None,
    *,
    exit_code: int,
    still_applies: Callable[[session_records.SessionRecord], bool] | None = None,
) -> session_records.SessionRecord | None:
    """TOCTOU-safe terminal stamp used by the liveness sweeps and ``mark_session_folded``.

    Re-reads *slug* itself -- a second, fresher read than the sweep's own
    caller-side liveness check -- and, ONLY if status is still non-terminal
    on THIS read and *still_applies* (when given) accepts THIS read, stamps
    status=EXITED/exit_code=*exit_code* and writes. Returns the marked
    record, or ``None`` if a concurrent writer (most plausibly
    spawn-claude.sh's own ``finish()``, or a hook re-binding the record)
    already changed it in the window between the sweep's own liveness check
    and this call -- in which case nothing is written, so that writer's
    real state is never clobbered by the orphan sentinel.

    Deliberately a narrow variant rather than a change to ``session_records.update_status``
    itself: ``session_records.update_status``'s own ``exit`` CLI subcommand caller is an
    authoritative, unconditional final report from the exiting process and
    must stay unconditional; only this sweep's best-effort guess needs to
    defer to a fresher real write.
    """
    record = session_records.read_record(slug, root=root)
    if record.status in session_records.TERMINAL_STATUSES:
        return None
    if still_applies is not None and not still_applies(record):
        return None
    record.status = session_records.Status.EXITED
    record.exit_code = exit_code
    session_records.write_record(record, root=root)
    _drop_t3_thread_pointer_if_stale(record, root)
    return record


def mark_orphaned_sessions_exited(
    root: Path | str | None = None,
    *,
    now: datetime | None = None,
) -> list[session_records.SessionRecord]:
    """Sweep ``<root>/sessions/*/`` and mark provably-orphaned records EXITED.

    This is the liveness sweep that fixes the F2 staleness backlog: record
    CONVERGENCE (session_hooks' CLAUDE_SPAWN_SESSION_ID adoption) only
    advances CLEANLY-exiting spawned sessions through their hook-driven
    launching -> running -> idle/awaiting-input -> exited lifecycle. The
    backlog is overwhelmingly UNCLEAN deaths (SIGKILL, a force-closed
    terminal, a crash, a reboot) where spawn-claude.sh's own ``finish()``
    never runs, so ``exited`` is never written -- only this sweep ever marks
    those records terminal.

    Mirrors ``reap_stale_records``'s ``stale_pid`` predicate EXACTLY --
    status is non-terminal (not in session_records.TERMINAL_STATUSES), the launcher is judged
    DEAD (``shared/src/shared/session_liveness.py::judge``), and age
    (record.json's mtime vs *now*) exceeds
    ``NON_TERMINAL_HEARTBEAT_TTL`` -- the same false-positive guard that
    rule already trusts: a genuinely-live session (including a
    detached/tmux one) keeps bumping its record's mtime via its own hooks,
    so it is never swept regardless of how long it has been non-terminal.

    Known trade-off for DETACHED spawns (reviewer-flagged): spawn-claude.sh
    itself exits once it hands off (its ``launcher_pid`` dies) while claude
    keeps running standalone, so for the entire remaining life of a
    detached session the launcher is already judged DEAD -- the heartbeat
    (record.json's mtime, bumped only by that session's own
    Notification/Stop hooks after convergence) is the SOLE thing keeping it
    unswept. A detached session that goes quieter than
    ``NON_TERMINAL_HEARTBEAT_TTL`` (1h) between hook events -- e.g. one long
    tool-heavy turn with no intervening Notification/Stop -- can therefore
    be marked EXITED while genuinely still running, by the reap every other
    spawn's ``_run_launching`` starts, not just by an explicit CLI ``reap``.
    This is self-correcting
    -- its next hook event calls ``session_records.refresh_record`` and flips status back --
    but there is a real window of an incorrectly-EXITED cockpit row. Reusing
    the pre-existing ``NON_TERMINAL_HEARTBEAT_TTL`` (rather than introducing
    a second, sweep-specific threshold) keeps this sweep's semantics
    identical to the reaper's already-trusted ``stale_pid`` rule; widening
    the TTL or tracking a detached session's actual claude pid instead of
    its exited launcher would be a separate, broader design change (touching
    session_hooks.py/spawn-claude.sh's pid bookkeeping) and is left as a
    follow-up rather than folded into this sweep.

    Unlike ``reap_stale_records``, this does NOT delete anything -- it marks
    a matching record EXITED (exit_code=``ORPHAN_EXIT_CODE``), preserving
    every other field, and leaves its directory in place.
    ``reap_stale_records``'s existing ``terminal_ttl`` rule reclaims it
    later, unchanged. An already-terminal record is left fully untouched
    (idempotent -- never re-stamped with the sentinel), and a record whose
    body is missing or corrupt is skipped here -- that is
    ``reap_stale_records``'s own ``'corrupt'`` rule's concern, not this
    sweep's.

    The final stamp itself goes through ``_mark_exited_if_still_non_terminal``,
    which re-reads the record ONE more time immediately before writing: if a
    concurrent writer (most plausibly ``finish()``) already made it terminal
    in the window since this sweep's own liveness check above, the sweep
    backs off instead of clobbering that writer's real exit_code with the
    orphan sentinel.

    A per-record try/except means one bad record (a corrupt body, a
    concurrent-reap race vanishing the file mid-sweep, an unwritable
    record) never aborts the sweep -- it is logged and skipped, exactly
    like ``reap_stale_records``'s own fault handling.

    A T3-hosted record (``display.kind == 't3'``) is not swept here while
    ``_t3_thread_sweep_owns`` holds: its ``launcher_pid`` is the spawn
    script, which exits in sibling mode, and its hooks fall silent whenever
    the thread waits on a human. ``mark_t3_sessions_exited`` sweeps it by
    thread instead.

    *now* is injectable for deterministic tests; defaults to the real UTC
    clock. Returns the list of records this call marked (their post-mark,
    now-EXITED state).
    """
    if now is None:
        now = datetime.now(UTC)
    base = session_records.sessions_dir(root)
    marked: list[session_records.SessionRecord] = []
    if not base.is_dir():
        return marked

    for slug_dir in sorted(base.iterdir()):
        if not slug_dir.is_dir():
            continue
        slug = slug_dir.name

        try:
            record = session_records.read_record(slug, root=root)
        except (FileNotFoundError, session_records.CorruptSessionRecord):
            continue  # missing/corrupt body -- reap_stale_records' 'corrupt' rule's job
        if record.status in session_records.TERMINAL_STATUSES:
            continue  # already terminal -- idempotent, never re-stamped
        if _t3_thread_sweep_owns(record, slug, root):
            continue
        if not _swept_as_dead(
            record.launcher_identity,
            record.launcher_pid,
            sweep='mark_orphaned_sessions_exited',
            subject=slug,
        ):
            continue  # a launcher not judged dead is never swept, regardless of age

        record_path = slug_dir / 'record.json'
        try:
            mtime = record_path.stat().st_mtime
        except OSError:
            continue  # vanished mid-sweep (e.g. concurrent reap) -- skip it
        age = now - datetime.fromtimestamp(mtime, tz=UTC)
        if age <= NON_TERMINAL_HEARTBEAT_TTL:
            continue  # heartbeat grace -- not stale enough yet

        try:
            marked_record = _mark_exited_if_still_non_terminal(
                slug, root, exit_code=ORPHAN_EXIT_CODE
            )
        except (FileNotFoundError, session_records.CorruptSessionRecord, OSError):
            # A single unmarkable record (a concurrent-reap race, an
            # unwritable record) must not abort the sweep -- log and move on
            # to the next candidate.
            logger.error(
                'mark_orphaned_sessions_exited: failed to mark %s', slug, exc_info=True
            )
            continue
        if marked_record is None:
            continue  # lost the race: a concurrent writer already made this terminal
        marked.append(marked_record)

    return marked


def index_unpointed_t3_records(root: Path | str | None = None) -> list[str]:
    """Pass every record under ``<root>/sessions/`` to ``session_records.index_t3_record_if_unheld``; returns the slugs indexed.

    ``run_reap``'s first phase. A live t3 record whose thread no pointer
    holds -- one written before the index existed, or whose pointer write
    failed -- joins the index ``mark_t3_sessions_exited`` iterates only here.
    Until then ``_t3_thread_sweep_owns`` keeps it out of the pid sweeps. An
    unreadable record is skipped, never fatal.
    """
    base = session_records.sessions_dir(root)
    if not base.is_dir():
        return []
    indexed: list[str] = []
    for slug_dir in sorted(base.iterdir()):
        if not slug_dir.is_dir():
            continue
        try:
            record = session_records.read_record(slug_dir.name, root=root)
        except (OSError, session_records.CorruptSessionRecord):
            continue
        if session_records.index_t3_record_if_unheld(record, slug_dir.name, root=root):
            indexed.append(slug_dir.name)
    return indexed


def _t3_thread_sweep_owns(record: session_records.SessionRecord, slug: str, root: Path | str | None) -> bool:
    """Whether ``mark_t3_sessions_exited``, not a pid sweep, owns non-terminal *record*. Read-only.

    True for a t3 record whose thread pointer names *slug* or no live record.
    An unheld thread stays owned, so a record ``index_unpointed_t3_records``
    has not indexed yet is never handed to the pid sweeps. Only displacement
    by another live record, or a thread id that cannot name a pointer, does.
    """
    thread_id = session_records.t3_thread_of(record)
    if thread_id is None:
        return False
    held_by = session_records.resolve_session_slug_for_t3_thread(thread_id, root=root)
    return held_by is None or held_by == slug


def mark_t3_sessions_exited(
    environment_id: str,
    listed_threads: AbstractSet[str],
    root: Path | str | None = None,
    *,
    listed_at: datetime | None = None,
    thread_ended: Callable[[str], bool],
    routine: bool = False,
) -> list[str]:
    """Mark EXITED every indexed T3-hosted record of *environment_id* whose thread has gone.

    *listed_threads* is one complete listing of the environment's active
    threads. Only ``sessions-by-t3-thread/`` is read: a record behind a
    vouched entry whose thread is unlisted and whose ``t3_environment_id`` is
    *environment_id* or unset is marked (exit_code ``ORPHAN_EXIT_CODE``),
    which drops its pointer, once *thread_ended* confirms that thread ended:
    a listing omits live side chats. With *listed_at*, a record written after
    that instant is skipped: the listing predates it and cannot vouch for its
    thread's absence.

    Returns the marked slugs. One unreadable or unwritable record is logged
    and skipped, never fatal. One line reports the index size, at INFO
    unless *routine* -- one pass of a wait loop -- marked nothing, then at
    DEBUG.
    """
    base = session_records.t3_thread_pointers_dir(root)
    entries = sorted(base.iterdir()) if base.is_dir() else []
    thread_ids = [path.name for path in entries if not path.name.endswith(ATOMIC_WRITE_TEMP_SUFFIX)]
    marked: list[str] = []
    for thread_id in thread_ids:
        if thread_id in listed_threads:
            continue
        slug = _mark_t3_session_gone(thread_id, environment_id, root, listed_at, thread_ended)
        if slug is not None:
            marked.append(slug)
    logger.log(
        logging.DEBUG if routine and not marked else logging.INFO,
        'mark_t3_sessions_exited: marked %d of %d indexed session(s) under %s',
        len(marked),
        len(thread_ids),
        base,
    )
    return marked


def _mark_t3_session_gone(
    thread_id: str,
    environment_id: str,
    root: Path | str | None,
    listed_at: datetime | None,
    thread_ended: Callable[[str], bool],
) -> str | None:
    """Mark the live record of unlisted thread *thread_id* EXITED once *thread_ended* confirms it; its slug, or None."""
    slug = session_records.resolve_session_slug_for_t3_thread(thread_id, root=root)
    if slug is None:
        return None
    if listed_at is not None and _written_after(session_records.record_path_for_slug(slug, root=root), listed_at):
        return None

    def still_this_environments(record: session_records.SessionRecord) -> bool:
        return (
            session_records.t3_thread_of(record) == thread_id
            and record.display is not None
            and record.display.t3_environment_id in (None, environment_id)
        )

    try:
        record = session_records.read_record(slug, root=root)
    except (FileNotFoundError, session_records.CorruptSessionRecord, OSError):
        logger.error('mark_t3_sessions_exited: failed to read %s', slug, exc_info=True)
        return None
    if record.status in session_records.TERMINAL_STATUSES or not still_this_environments(record):
        return None
    if not thread_ended(thread_id):
        return None
    try:
        marked = _mark_exited_if_still_non_terminal(
            slug, root, exit_code=ORPHAN_EXIT_CODE, still_applies=still_this_environments
        )
    except (FileNotFoundError, session_records.CorruptSessionRecord, OSError):
        logger.error('mark_t3_sessions_exited: failed to mark %s', slug, exc_info=True)
        return None
    return slug if marked is not None else None


def _written_after(path: Path, instant: datetime) -> bool:
    try:
        return datetime.fromtimestamp(path.stat().st_mtime, tz=UTC) > instant
    except OSError:
        return True


FOLDED_EXIT_CODE: int = 201
"""Synthetic exit code ``mark_session_folded`` stamps on a hand-launched
record whose session turned out to be a spawn's: the spawn record now carries
that session. Clear of every code ``ORPHAN_EXIT_CODE``'s docstring lists, and
of that sentinel itself."""


def mark_session_folded(slug: str, root: Path | str | None = None) -> session_records.SessionRecord | None:
    """Mark the record *slug* EXITED with ``FOLDED_EXIT_CODE`` unless already terminal.

    Returns the marked record, or None when it was already terminal. Raises
    like ``session_records.read_record`` when it is missing or unreadable.
    """
    return _mark_exited_if_still_non_terminal(slug, root, exit_code=FOLDED_EXIT_CODE)


def _normalize_wm_window_id(raw: str | None) -> str | None:
    """Canonicalize a window id to unpadded ``0x<hex>``, or ``None`` if unprovable.

    ``_resolve_display``'s WINDOWID branch (session_hooks.py) can persist a
    DECIMAL window id (e.g. ``'26'``), while ``wmctrl -l`` always prints
    zero-padded hex (e.g. ``'0x0000001a'``). Comparing the two raw strings
    would treat a live decimal-captured window as "gone" and falsely reap
    it -- avoiding exactly that is ``mark_windowless_wm_sessions_exited``'s
    whole reason for calling this. This parses *raw* as base-16 when it
    starts with ``'0x'``/``'0X'``, else base-10, and re-renders the parsed
    value as unpadded ``f'0x{value:x}'`` -- so ``'26'`` and ``'0x0000001a'``
    both normalize to ``'0x1a'`` and compare equal; an already-canonical
    ``'0x1a'`` round-trips unchanged.

    Fails toward "keep, don't reap": returns ``None`` for ``None``/empty/
    whitespace-only input, a value that doesn't parse as an int in the
    selected base, or a negative value. Callers must treat ``None`` as
    "cannot prove this window is gone" -- never as a wildcard match against
    another ``None``.
    """
    if raw is None:
        return None
    text = raw.strip()
    if not text:
        return None
    try:
        value = int(text, 16) if text.startswith(('0x', '0X')) else int(text, 10)
    except ValueError:
        return None
    if value < 0:
        return None
    return f'0x{value:x}'


_WMCTRL_TIMEOUT_SECS = 2
"""Per-probe ``wmctrl -l`` timeout for ``_wmctrl_list``, seconds. Mirrors
``session_hooks._WMCTRL_TIMEOUT_SECS`` -- kept as a separate module-level
constant (not imported) per this module's stdlib-only/import-free
constraint; see ``_wmctrl_list``'s docstring."""


def _wmctrl_list(argv: list[str]) -> subprocess.CompletedProcess[str]:
    """Fail-soft ``wmctrl -l`` runner -- default for ``_wmctrl_live_window_ids``.

    DUPLICATED (not imported) from ``session_hooks._wmctrl_list``: this
    module is deliberately stdlib-only and import-free of the rest of the
    orchestrator (see module docstring), and ``session_hooks`` already
    imports ``session_registry``, so importing back would be circular. Keep
    this in sync with ``session_hooks._wmctrl_list`` if either changes.

    Never raises: a genuinely-missing ``wmctrl`` binary (``FileNotFoundError``)
    yields a permanent-failure rc=127 sentinel; any other ``OSError``/
    ``subprocess.SubprocessError`` (notably a timeout) yields a distinct,
    transient rc=124 sentinel. Both sentinels -- and every other nonzero
    return code -- are treated identically (fail-soft: no window evidence)
    by ``_wmctrl_live_window_ids``; they are only distinguished here in case
    a future caller ever needs to react to them differently.
    """
    try:
        return subprocess.run(argv, capture_output=True, text=True, timeout=_WMCTRL_TIMEOUT_SECS)
    except FileNotFoundError:
        return subprocess.CompletedProcess(argv, returncode=127, stdout='')
    except (OSError, subprocess.SubprocessError):
        return subprocess.CompletedProcess(argv, returncode=124, stdout='')


def _wmctrl_live_window_ids(
    run: Callable[[list[str]], subprocess.CompletedProcess[str]] | None = None,
) -> set[str] | None:
    """Return the set of normalized window ids ``wmctrl -l`` currently reports.

    HARD fail-soft: returns ``None`` -- never an empty set -- for a missing
    ``wmctrl`` binary, ANY nonzero return code, or any exception raised by
    *run* itself. Callers MUST treat ``None`` ("no window evidence") as
    distinct from an empty set (a successful probe that legitimately found
    zero windows) -- only a successful run (rc=0, however many windows)
    authorizes ``mark_windowless_wm_sessions_exited`` to reap anything.

    Each ``wmctrl -l`` line is parsed via ``split(None, 3)`` -- the same
    idiom as ``WmBackend.is_alive`` (cockpit/src/cockpit/backends/wm.py) and
    ``session_hooks._resolve_wm_window_id`` -- but, unlike both of those,
    this takes column 0 (the window id) of ANY non-empty line, even a
    titleless line with fewer than 4 columns. A reaper must fail toward
    keeping live sessions: the stricter >=4-column gate those two use would
    silently drop a live-but-titleless window from the live set, risking a
    false reap. ``wmctrl -l`` has no header and every line begins with a
    window id, so this can't admit a phantom id. Each id is normalized via
    ``_normalize_wm_window_id`` before being added to the set (a column 0
    that fails to parse is simply skipped, not an error).

    NOTE (reviewer-flagged): this parse is now duplicated across THREE call
    sites -- ``WmBackend.is_alive``, ``session_hooks._resolve_wm_window_id``,
    and this function -- keep all three in sync if the ``wmctrl -l`` column
    layout ever changes. The three already diverge in one respect (this
    function's deliberate any-non-empty-line/column-0 leniency vs. the other
    two's stricter >=4-column gate, documented above), which raises the odds
    of further drift the next time any one of them changes. No change is
    needed now given this module's stdlib-only/import-free constraint (see
    module docstring); if/when that constraint is ever relaxed, extracting
    this parse and ``_wmctrl_list`` into one shared stdlib-only helper all
    three call sites import would remove the drift risk entirely.

    *run* defaults to ``None``, which resolves to the real ``_wmctrl_list``
    at CALL time -- a plain name lookup in this function's body, not a bound
    default-parameter value -- so a test's
    ``monkeypatch.setattr(session_registry, '_wmctrl_list', fake)`` takes
    effect even when reached indirectly via
    ``mark_windowless_wm_sessions_exited()``'s own default ``run``
    passthrough.
    """
    if run is None:
        run = _wmctrl_list
    try:
        result = run(['wmctrl', '-l'])
    except Exception:
        logger.warning('_wmctrl_live_window_ids: wmctrl -l invocation raised', exc_info=True)
        return None
    if result.returncode != 0:
        return None
    live_ids: set[str] = set()
    for line in result.stdout.splitlines():
        columns = line.split(None, 3)
        if not columns:
            continue
        normalized = _normalize_wm_window_id(columns[0])
        if normalized is not None:
            live_ids.add(normalized)
    return live_ids


def _wm_window_id_of(record: session_records.SessionRecord) -> str | None:
    """Normalized window id of a wm-display *record*; ``None`` for other displays or a bad id."""
    if record.display is None or record.display.kind != session_records.DisplayKind.WM.value:
        return None
    return _normalize_wm_window_id(record.display.wm_window_id)


def _names_window(window_id: str) -> Callable[[session_records.SessionRecord], bool]:
    return lambda record: _wm_window_id_of(record) == window_id


def mark_windowless_wm_sessions_exited(
    root: Path | str | None = None,
    *,
    run: Callable[[list[str]], subprocess.CompletedProcess[str]] | None = None,
) -> list[session_records.SessionRecord]:
    """Sweep ``<root>/sessions/*/`` and mark EXITED any wm-display session whose window is gone.

    A SEPARATE liveness sweep from ``mark_orphaned_sessions_exited`` above,
    not a folded-in extension of it: a closed terminal window is DEFINITIVE
    evidence the session is dead/un-interactable, so this sweep is
    deliberately PID- and AGE-INDEPENDENT -- it reaps exactly the zombies
    the pid/TTL sweep is forced to keep (a live ``launcher_pid``, or a
    record younger than ``NON_TERMINAL_HEARTBEAT_TTL``). Folding the two
    predicates together would muddy ``mark_orphaned_sessions_exited``'s
    "mirrors reap_stale_records' stale_pid predicate EXACTLY" contract, and
    would run ``wmctrl -l`` on every liveness check instead of at most twice
    per sweep.

    Scope: only records with ``display is not None and display.kind ==
    DisplayKind.WM.value`` are ever considered. A ``display=None`` (headless
    fleet-agent) or ``display.kind == 'tmux'`` record is untouched here --
    both stay on the existing pid/TTL path (a closed tmux pane is not
    "window gone" in the X11 sense this sweep checks). A wm record whose
    captured ``wm_window_id`` is missing or fails to parse
    (``_normalize_wm_window_id`` returns ``None``) is ALSO left untouched --
    we cannot prove the window is gone, so we fail toward keep and defer to
    the pid/TTL sweep.

    HARD fail-soft: if the first ``_wmctrl_live_window_ids(run)`` probe is
    ``None`` (missing ``wmctrl`` binary, a nonzero return code, or any
    exception -- see its docstring), this returns ``[]`` immediately and
    marks NOTHING. "No window evidence" (a headless host, an X11 hiccup)
    must never be treated as "every window is gone" -- that would mass-mark
    every wm session on the fleet EXITED.

    Two-probe rule: a record is marked only on a probe taken AFTER it was
    read. ``_run_launching``'s detached reap runs concurrently with the
    spawned session's SessionStart hook, which writes its ``wm_window_id``
    (``orchestrator/src/orchestrator/session_hooks.py::_resolve_display``),
    so a window id can land after the first probe; the pass therefore only
    collects candidates absent from the first probe, and one confirming
    probe after the pass decides (``None`` again marks nothing). The mark
    itself re-reads the record and goes ahead only if it still names the
    candidate's window, so a hook that re-bound it to a new window since
    the pass read it is left alone.

    Self-correction trade-off (mirrors ``mark_orphaned_sessions_exited``): a
    window that both probes miss (e.g. racing a window remap) can mark a
    still-live session EXITED. This is deliberately accepted -- the
    session's next hook event calls ``session_records.refresh_record`` and flips status back
    -- rather than adding retries beyond the confirming probe.

    Known trade-off for a multi-DISPLAY fleet (reviewer-flagged): ``wmctrl
    -l`` only reports windows on the single X ``DISPLAY`` the invoking
    process is connected to -- whatever ``_run_launching``/``run_reap`` (or
    a directly-invoked CLI) happens to inherit as ``$DISPLAY`` -- never
    every display a fleet's sessions might be spread across. A session
    whose window genuinely lives on a DIFFERENT display than the one this
    sweep's probe ran under is indistinguishable here from one that is
    actually gone -- its id is simply absent from ``live_ids`` -- so a
    multi-display fleet risks a false reap of a still-live session on
    another display. This is accepted for the same reason as the
    self-correction trade-off above (a live session's next hook event flips
    it back), and single-DISPLAY is this sweep's assumed common case. To
    keep a suspected cross-display false reap diagnosable rather than
    silent, any sweep that marks at least one record logs the ``DISPLAY``
    value it probed under.

    Unlike ``reap_stale_records``, this does NOT delete anything -- it marks
    a matching record EXITED (exit_code=``ORPHAN_EXIT_CODE``), preserving
    every other field, and leaves its directory in place;
    ``reap_stale_records``'s existing ``terminal_ttl`` rule reclaims it
    later, unchanged. An already-terminal record is left fully untouched
    (idempotent -- never re-stamped with the sentinel).

    A per-record try/except means one bad record (a corrupt body, a
    concurrent-reap race vanishing the file mid-sweep, an unwritable record)
    never aborts the sweep -- it is logged and skipped, exactly like
    ``mark_orphaned_sessions_exited``'s own fault handling.

    *run* is the injectable ``wmctrl -l`` runner passed through to
    ``_wmctrl_live_window_ids`` (``None``, the default, resolves to the real
    ``_wmctrl_list`` at call time).
    """
    live_ids = _wmctrl_live_window_ids(run)
    if live_ids is None:
        return []  # HARD fail-soft: no window evidence -- mark nothing.

    base = session_records.sessions_dir(root)
    marked: list[session_records.SessionRecord] = []
    if not base.is_dir():
        return marked

    candidates: list[tuple[str, str]] = []
    for slug_dir in sorted(base.iterdir()):
        if not slug_dir.is_dir():
            continue
        slug = slug_dir.name

        try:
            record = session_records.read_record(slug, root=root)
        except (FileNotFoundError, session_records.CorruptSessionRecord):
            continue  # missing/corrupt body -- reap_stale_records' 'corrupt' rule's job
        if record.status in session_records.TERMINAL_STATUSES:
            continue  # already terminal -- idempotent, never re-stamped
        window_id = _wm_window_id_of(record)
        if window_id is None or window_id in live_ids:
            continue  # tmux / headless / unprovable id, or the window is still live
        candidates.append((slug, window_id))

    if not candidates:
        return marked
    confirmed_live_ids = _wmctrl_live_window_ids(run)
    if confirmed_live_ids is None:
        return marked  # the confirming probe yielded no evidence -- mark nothing.

    for slug, window_id in candidates:
        if window_id in confirmed_live_ids:
            continue  # the window appeared after the first probe

        try:
            marked_record = _mark_exited_if_still_non_terminal(
                slug, root, exit_code=ORPHAN_EXIT_CODE, still_applies=_names_window(window_id)
            )
        except (FileNotFoundError, session_records.CorruptSessionRecord, OSError):
            # A single unmarkable record (a concurrent-reap race, an
            # unwritable record) must not abort the sweep -- log and move on
            # to the next candidate.
            logger.error(
                'mark_windowless_wm_sessions_exited: failed to mark %s', slug, exc_info=True
            )
            continue
        if marked_record is None:
            continue  # lost the race: a concurrent writer already made this terminal
        marked.append(marked_record)

    if marked:
        # Diagnostic breadcrumb for the multi-DISPLAY trade-off documented
        # above: ties every mark to the DISPLAY the probe actually ran
        # under, so a suspected cross-display false reap is diagnosable
        # after the fact instead of silent. Only logged when this sweep
        # actually marked something, to stay quiet on the common no-op call.
        logger.info(
            'mark_windowless_wm_sessions_exited: marked %d session(s) EXITED (probed under DISPLAY=%r)',
            len(marked),
            os.environ.get('DISPLAY'),
        )
    return marked


# ---------------------------------------------------------------------------
# Reap pipeline: one serialised, timed, fail-soft drain (task 7944)
# ---------------------------------------------------------------------------

REAP_LOCK_FILENAME = 'reap.lock'
REAP_REPORT_FILENAME = 'last-reap.json'


def reap_lock_path(root: Path | str | None = None) -> Path:
    return fleet_root(root) / REAP_LOCK_FILENAME


def reap_report_path(root: Path | str | None = None) -> Path:
    return fleet_root(root) / REAP_REPORT_FILENAME


@contextlib.contextmanager
def reap_lock(root: Path | str | None = None) -> Iterator[None]:
    """Blocking exclusive flock on ``<fleet_root>/reap.lock``, serialising reaps.

    The sidecar is a stable lock target, created once and never renamed or
    deleted, for the same reason as ``decision_registry.py::decision_id_lock``'s.
    """
    with exclusive_sidecar_lock(reap_lock_path(root)):
        yield


@dataclass(frozen=True)
class ReapPhase:
    """One phase of ``run_reap``: its wall time, and either its count or its error."""

    name: str
    seconds: float
    count: int | None
    error: str | None


@dataclass(frozen=True)
class ReapReport:
    """What one ``run_reap`` did, persisted at ``reap_report_path``.

    ``requested_at`` to ``started_at`` is the time it queued on ``reap_lock``;
    ``remaining_record_dirs`` is the registry's size after the drain, or
    ``None`` if it could not be counted.
    """

    requested_at: str
    started_at: str
    finished_at: str
    phases: tuple[ReapPhase, ...]
    remaining_record_dirs: int | None

    def to_json(self) -> str:
        return json.dumps(dataclasses.asdict(self), indent=2) + '\n'


def _run_reap_phase(
    name: str, phase: Callable[[Path | str | None], Sized], root: Path | str | None
) -> ReapPhase:
    started = time.monotonic()
    try:
        count = len(phase(root))
    except Exception as exc:
        logger.error('run_reap: phase %s failed', name, exc_info=True)
        return ReapPhase(
            name=name,
            seconds=time.monotonic() - started,
            count=None,
            error=f'{type(exc).__name__}: {exc}',
        )
    return ReapPhase(name=name, seconds=time.monotonic() - started, count=count, error=None)


def _count_record_dirs(root: Path | str | None) -> int | None:
    try:
        with os.scandir(session_records.sessions_dir(root)) as entries:
            return sum(1 for entry in entries if entry.is_dir())
    except FileNotFoundError:
        return 0
    except OSError:
        logger.warning('run_reap: could not count record dirs', exc_info=True)
        return None


def _a_reap_started_since(requested_at: datetime, root: Path | str | None) -> bool:
    try:
        last = json.loads(reap_report_path(root).read_text(encoding='utf-8'))
        return datetime.fromisoformat(last['started_at']) > requested_at
    except (OSError, ValueError, KeyError, TypeError):
        return False


def run_reap(root: Path | str | None = None) -> ReapReport | None:
    """Drain the registry: index unpointed t3 records, mark orphans/windowless-wm exited, THEN delete stale dirs and pointers.

    The ``reap`` verb, for the operator and for ``_run_launching``'s detached
    child alike. Concurrent calls queue on ``reap_lock``. A call whose turn
    comes after a reap that STARTED after this call was requested returns
    ``None`` without draining: that reap already covered everything this
    call could see, so a burst of spawns costs one drain, not one each.

    Mark-then-delete keeps a pass complete: a record either liveness sweep
    marks exited this pass must not simultaneously be eligible for stale_pid
    deletion (marking bumps its mtime, so it lands back within
    reap_stale_records' NON_TERMINAL_HEARTBEAT_TTL heartbeat grace) -- it
    only becomes reapable later, once it ages past TERMINAL_TTL as any other
    terminal record does. ``mark_windowless_wm_sessions_exited`` runs here
    alongside ``mark_orphaned_sessions_exited`` because it marks exactly the
    live-pid / age<1h wm-display zombies the pid/TTL sweep is forced to keep.
    ``index_unpointed_t3_records`` runs first; the sweeps after it only read
    the thread index. ``reap_stale_session_pointers`` runs last so pointers
    to records removed in this same pass go too.

    Each phase is timed and fail-soft on its own, so one phase's fault never
    skips the rest. The report is written atomically to
    ``reap_report_path``: it is the only window into a detached child whose
    stdio is DEVNULL. A failed report write is logged, not raised.
    """
    requested_at = datetime.now(UTC)
    with reap_lock(root):
        if _a_reap_started_since(requested_at, root):
            logger.info('run_reap: a reap started since this one was requested; skipping')
            return None
        started_at = datetime.now(UTC)
        phases = (
            _run_reap_phase('index_unpointed_t3_records', index_unpointed_t3_records, root),
            _run_reap_phase('mark_orphaned_sessions_exited', mark_orphaned_sessions_exited, root),
            _run_reap_phase(
                'mark_windowless_wm_sessions_exited', mark_windowless_wm_sessions_exited, root
            ),
            _run_reap_phase('reap_stale_records', reap_stale_records, root),
            _run_reap_phase('reap_stale_session_pointers', reap_stale_session_pointers, root),
        )
        report = ReapReport(
            requested_at=requested_at.isoformat(),
            started_at=started_at.isoformat(),
            finished_at=datetime.now(UTC).isoformat(),
            phases=phases,
            remaining_record_dirs=_count_record_dirs(root),
        )
        try:
            atomic_write_text(reap_report_path(root), report.to_json())
        except OSError:
            logger.error('run_reap: failed to write %s', reap_report_path(root), exc_info=True)
    return report


# ---------------------------------------------------------------------------
# Role leases: single-owner-per-role (Attention Rail T7; PRD T7 §4.1-4.2, §6 G5)
# ---------------------------------------------------------------------------

LEASE_NAME_SANITIZE_RE = re.compile(r'[^A-Za-z0-9._#-]')
"""Unlike session_records.sanitize_slug, this PRESERVES '#': task-scoped lease names
(build_lease_name('unblock', project, task_id)) encode their task with a
literal '#', which must survive into the .lease filename. Everything else
outside [A-Za-z0-9._#-] (notably '/') maps to '-', so a malformed name can
never escape leases_dir via a path separator."""


def leases_dir(root: Path | str | None = None) -> Path:
    return fleet_root(root) / 'leases'


def lease_path_for_name(name: str, root: Path | str | None = None) -> Path:
    """Resolve *name* (see build_lease_name) to its ``<leases_dir>/<name>.lease`` path.

    *name* is sanitized through LEASE_NAME_SANITIZE_RE first, so it always
    resolves to a single file directly inside leases_dir.
    """
    safe_name = LEASE_NAME_SANITIZE_RE.sub('-', name)
    return leases_dir(root) / f'{safe_name}.lease'


def build_lease_name(role: str, project: str, task_id: str | None = None) -> str:
    """Build the canonical lease name for *role* (PRD T7): ``<role>-<project>[#<task_id>]``.

    task_id is optional; when supplied, the lease is scoped to a single task
    (e.g. ``unblock-df#2085``), so two concurrent /unblock sessions on
    DIFFERENT tasks in the same project never contend the same lease.
    """
    name = f'{role}-{project}'
    if task_id:
        name = f'{name}#{task_id}'
    return name


def resolve_own_record_slug(
    env: Mapping[str, str] | None = None, *, root: Path | str | None = None
) -> str | None:
    """Return THIS claude session's registry record slug, via ``$CLAUDE_PID``'s pointer.

    A lease slug (``default_lease_slug``: ``<lease-name>-$CLAUDE_PID``) and a
    registry record slug (``session_records.build_session_slug``) are different namespaces,
    so neither can be looked up as the other. The pid pointer is the join:
    the session hooks stamp ``sessions-by-pid/<claude pid>`` with the record
    key, and ``session_records.resolve_session_slug_for_pid`` reads it back only while that
    record still names the pid as its owner. Its consumers are
    ``LeaseHolder.record_slug`` and ``decision_registry.py::DecisionRecord.record_slug``.

    None when ``CLAUDE_PID`` is unusable or no vouched-for pointer exists.
    A shared ``claude agents`` host has no session-scoped pid, so the result
    is None there too.
    Quiet by contract: this serves enrichment, so an unresolvable slug is a
    blank field in the caller's result, not a log line.
    """
    pid = session_identity.resolve_session_identity(env).session_scoped_pid
    return None if pid is None else session_records.resolve_session_slug_for_pid(pid, root=root)


def default_lease_slug(
    name: str,
    env: Mapping[str, str] | None = None,
    *,
    identity: session_identity.SessionIdentity | None = None,
) -> str | None:
    """Derive THIS session's lease slug for lease *name*: ``<name>-<session pid>``.

    Under a shared ``claude agents`` host it is
    ``<name>-<session pid>-<CLAUDE_CODE_SESSION_ID>``: every hosted session
    shares that pid, so the pid alone would hand them all one identity.

    WHY THIS EXISTS (task 4248). This is ``session_identity.resolve_session_pid``'s own argument,
    extended from the pid to the slug. 3994 moved the PID into code because "it
    depended on every skill doc getting one shell token right" -- and then, in
    the same change, made the SLUG load-bearing: ``--slug`` became required on
    ``lease-heartbeat``/``lease-release`` and a mismatch is REFUSED. So the
    token that decides whether every heartbeat and the final release ACT or are
    refused was still being assembled in shell, in three SKILL.md files, as
    ``<role>-<project>-${CLAUDE_PID:-$PPID}`` -- and ``$PPID`` is measurably NOT
    stable across Claude Code tool calls (1430433 then 1471645, the first
    already dead), so a session could drift into a slug its own later heartbeat
    would fail to match. Deriving it HERE makes the correct token structural.

    It must be RE-DERIVABLE rather than carried: each Claude Code Bash tool call
    is a fresh ``/bin/bash -c``, so a ``SLUG=$(...)`` captured during the claim
    is gone by the time the heartbeat runs -- the same property that makes ``$$``
    an unusable pid. Hence a pure function of (name, env), evaluated afresh on
    every verb, rather than a value threaded through the skills.

    RETURNS None WHEN THE SESSION PID IS UNRESOLVABLE (or, under a shared host,
    when the session id is) -- deliberately, and specifically NOT
    ``f'{name}-0'`` nor ``f'{name}-<host pid>'``, which every hosted session
    would share. ``session_identity.resolve_session_pid`` may degrade to 0
    because a pid is a LIVENESS probe and 0 is provably dead, which keeps a
    crashed holder's lease reapable. A slug is an IDENTITY, and ``{name}-0`` is
    not unique: two concurrently-degraded sessions contending the same lease
    name would compute an IDENTICAL slug, so each would pass the other's
    ``_is_lease_owner`` check and could heartbeat or release the other's lease
    -- exactly 3994 defect 1 ("any caller may evict/refresh any lease") reopened
    on the degraded path. Refusing to synthesize a token we cannot make unique
    is the only answer that keeps the ownership guard honest. The CLI turns the
    None into a loud ``parser.error`` (exit 2) naming the unusable variable and
    the ``--slug`` override -- the same exit-2 class a slug-less ``lease-claim``
    already produced pre-4248, so only the DERIVABLE case gains new behaviour.

    SILENT ON THE DEGRADED PATH, on purpose: the CLI fails loudly on the None,
    and ``session_identity.lease_liveness_pid`` reports a degraded liveness pid where one is
    written. The *name* is passed through verbatim -- a lease slug is never a
    filesystem path (``lease_path_for_name`` sanitizes only the NAME, and
    ``session_records.sanitize_slug`` applies only to session RECORD slugs), so the ``#`` in a
    task-scoped ``unblock-df#2085`` needs no escaping; see LeaseHolder.session_slug.

    *identity* ACCEPTS AN ALREADY-RESOLVED session_identity.SessionIdentity, so ``main()`` derives
    the lease body's liveness pid and this slug from ONE resolution; *env* is
    then ignored. Passing None resolves from *env* (defaulting to ``os.environ``).
    """
    return (
        identity if identity is not None else session_identity.resolve_session_identity(env)
    ).lease_slug(name)


LEASE_HEARTBEAT_TTL = timedelta(hours=2)
"""How long a lease survives with a dead holder pid and no fresh heartbeat
(mtime touch) before it is stale-reapable. Mirrors NON_TERMINAL_HEARTBEAT_TTL's
role for session records: staleness requires BOTH an aged heartbeat AND a
holder pid that cannot vouch for the lease -- see _lease_is_stale.

VALUE DERIVATION (task 2796 THREAD 1, RE-DERIVED by task 3994). The value is
unchanged at 7200s; its BASIS is not, because task 2796's basis is now false
and a false derivation is exactly what a later rotation reads and
mis-diagnoses itself with.

WITHDRAWN BASIS (2796): "watcher leases are effectively HEARTBEAT-ONLY, not
pid-guarded" -- true only because both watcher SKILLs claimed with
``--pid $$``, the ephemeral Bash-tool wrapper pid, so ``pid_alive`` was
~always False and the staleness AND-guard collapsed to a pure heartbeat-TTL
check. Task 3994's ``session_identity.resolve_session_pid`` restores a real, long-lived
session pid, so that premise no longer holds.

CURRENT BASIS (3994): with the pid guard actually live, a LIVE holder is
never reaped regardless of heartbeat age (a shared ``claude agents`` host pid
excepted, see _lease_is_stale), so this TTL now bounds only ONE
thing -- crash recovery for a genuinely-dead holder. It nonetheless stays at
2h rather than dropping back toward the old 5min, for two reasons:
(a) session_identity.resolve_session_pid still degrades -- to pid 0, which `pid_alive`
reports dead by construction -- whenever CLAUDE_PID is unresolvable on some
launch path or names a shared ``claude agents`` host, and that degradation is
logged but not fatal. On that path the
AND-guard collapses to its heartbeat half exactly as it did pre-3994, so
heartbeat-only behaviour must remain SURVIVABLE:
7200s = 2x the canonical 3600s watcher-rearm.sh ``--timeout`` slice, giving
a full quiet slice plus a full slice of margin, so a live-but-quiet holder
is never reaped-and-reclaimed by a duplicate (the duplicate-spawn incident);
(b) a crashed holder no longer COSTS 2h of silence -- ``lease-claim`` now
prints ``holder_liveness=orphaned`` as soon as the holder's pid is not
running (a SINGLE-signal pid probe; see _run_lease_claim), so the contender
surfaces the crash on the very next claim instead of waiting out the TTL.
Lowering it would
re-introduce live-holder-eviction risk to buy recovery latency that the
orphan signal already provides.

TRADE-OFF: a genuinely-crashed holder's lease still SURVIVES up to 2h before
it is stale-reapable. That is loud and recoverable, not silent: a contender
stands down with an explicit "pid <n> is not running, but its heartbeat is
FRESH ... NOT reclaimable for another <r>s" message plus the orphan signal,
an operator can reclaim immediately via ``lease-release --force``, and any
reap-and-reclaim of a supposedly-live holder emits a structured WARNING
(see claim_lease)."""


class LeasePolicy(StrEnum):
    """How claim_lease should respond when the lease is already held live.

    STAND_DOWN: the caller (e.g. a duplicate escalation-watcher) must exit.
    WARN_AND_PROCEED: the caller (e.g. a second /unblock on the same task)
        surfaces the current holder to the user but continues.
    """

    STAND_DOWN = 'stand-down'
    WARN_AND_PROCEED = 'warn-and-proceed'


class LeaseDecision(StrEnum):
    """The outcome of a claim_lease call."""

    ACQUIRED = 'acquired'
    STAND_DOWN = 'stand-down'
    PROCEED = 'proceed'


class LeaseMutation(StrEnum):
    """The outcome of a heartbeat_lease / release_lease call (task 3994).

    A lease is single-owner-per-role, so a MUTATION of one (refreshing its
    heartbeat, or removing it) is only legitimate from its actual holder.
    Before 3994 both verbs took only ``name`` and mutated unconditionally:
    any caller could evict a live holder or keep a stranger's lease
    structurally unreapable forever. These values report which of those an
    attempted mutation turned out to be.

    APPLIED: the caller IS the holder; the mutation was performed.
    FORCED: the caller is NOT the holder (or the body was unreadable) but
        passed ``force=True``; the mutation was performed anyway and logged
        at WARNING naming BOTH parties (operator recovery, attributable).
    ABSENT: there is no such lease. A no-op, and NOT a failure -- releasing
        an already-released lease is idempotent by design, so this is quiet.
    REFUSED: the caller is not the holder (or the body is unreadable, which
        fails TOWARD held); nothing was touched, logged at ERROR naming both
        parties.
    FAULTED: the mutation itself faulted (an OSError from utime/unlink).
        Logged at ERROR and returned, never raised -- a lease-substrate
        fault must not interrupt a watcher's main loop.
    """

    APPLIED = 'applied'
    FORCED = 'forced'
    ABSENT = 'absent'
    REFUSED = 'refused'
    FAULTED = 'faulted'


class HolderRecordState(StrEnum):
    """What a lease holder's ``record_slug`` says about its registry record.

    An axis INDEPENDENT of pid liveness (``holder_alive``): neither
    overrides the other, and reading them together is the caller's job.

    UNLINKED: the holder carries no ``record_slug`` (a body that predates the
        field, or a claimant whose pointer was unresolvable), so nothing was
        looked up. No evidence either way -- in particular NOT evidence that
        the holder is alive or gone.
    ABSENT: a ``record_slug`` WAS carried and no record exists under it: the
        record has been reaped. Evidence the holder's session is gone.
    UNREADABLE: the ``record_slug`` is not a record key, or its record
        exists but cannot be read or parsed.
    ACTIVE: the record's status is not terminal. For a hand-launched holder
        that tracks its terminal, not the claude process, so ACTIVE never
        overrides a dead pid.
    EXITED: the record's status is terminal (``session_records.TERMINAL_STATUSES``).
    """

    UNLINKED = 'unlinked'
    ABSENT = 'absent'
    UNREADABLE = 'unreadable'
    ACTIVE = 'active'
    EXITED = 'exited'


@dataclass(frozen=True)
class LeaseHolder:
    """Serialized identity of a lease's current holder -- the exact ``.lease`` file body.

    Every field is type-checked at construction, and from_dict also rejects
    a body that is not a JSON object, so a mistyped body is unreadable
    rather than a holder.

    session_slug: a CLAIMANT-CHOSEN ownership token, NOT a session-registry
        record key. The skills prescribe ``<lease-role>-<project>-<pid>``
        (escalation-watcher's and recon-escalation-watcher's "Claiming the
        ... Lease" sections, and unblock's "Claim the unblock lease"). Do NOT
        pass it to ``session_records.read_record``; the holder's record is reached through
        ``record_slug`` below.

        THIS DOCSTRING USED TO CLAIM THE OPPOSITE -- that it was "the
        holder's own session-registry slug (see session_records.build_session_slug), letting
        a contended claim point back at the current owner's
        session_registry record". It cannot be, and task 3994 built (then
        withdrew) a whole orphan-corroboration path on that false premise.
        The two namespaces differ in all three segments: a record slug is
        ``<role>-<project>[-<task>]-<uniqueness>`` where the uniqueness
        token is a session UUID for hand-launched sessions
        (``session_hooks.hook_session_slug``) rather than a pid, and the
        project token is the registry's project id (``dark_factory``) not
        the lease's short name (``df``). Measured on the live fleet root:
        ``watcher-df.lease`` holds ``session_slug=watcher-df-1894895``,
        while 0 of 46,624 session record dirs begin with ``watcher-``. So
        ``session_records.read_record(session_slug)`` ALWAYS raises FileNotFoundError.
        Pinned by tests/test_session_registry.py's
        TestLeaseSlugIsNotASessionRecordKey.

        Its ONLY two roles are (i) the ownership comparison in
        ``_check_lease_ownership`` and (ii) naming the holder in a
        human-readable contention/refusal message.
    pid: the holder process's pid; liveness is checked via pid_alive. Any int, bool
        excluded (0 is session_identity.resolve_session_pid's degraded provably-dead value).
    start_ts: ISO-8601 timestamp of when this holder claimed the lease.
    record_slug: the holder's session-registry record key, resolved at claim
        time from *pid*'s pointer (``session_records.resolve_session_slug_for_pid``). '' when
        it could not be resolved or the body predates the field -- an
        absence of evidence, never evidence of absence.
    """

    session_slug: str
    pid: int
    start_ts: str
    record_slug: str = ''

    def __post_init__(self) -> None:
        if isinstance(self.pid, bool) or not isinstance(self.pid, int):
            raise TypeError(f'LeaseHolder.pid must be an int (not a bool), got {self.pid!r}')
        for name in ('session_slug', 'start_ts', 'record_slug'):
            value = getattr(self, name)
            if not isinstance(value, str):
                raise TypeError(f'LeaseHolder.{name} must be a str, got {value!r}')

    def to_dict(self) -> dict[str, Any]:
        return {
            'session_slug': self.session_slug,
            'pid': self.pid,
            'start_ts': self.start_ts,
            'record_slug': self.record_slug,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> LeaseHolder:
        if not isinstance(data, dict):
            raise TypeError(f'LeaseHolder body must be a JSON object, got {data!r}')
        record_slug = data.get('record_slug')
        return cls(
            session_slug=data['session_slug'],
            pid=data['pid'],
            start_ts=data['start_ts'],
            record_slug='' if record_slug is None else record_slug,
        )

    def to_json(self) -> str:
        return json.dumps(self.to_dict())

    @classmethod
    def from_json(cls, raw: str) -> LeaseHolder:
        return cls.from_dict(json.loads(raw))


def holder_record_state(
    holder: LeaseHolder | None, *, root: Path | str | None = None
) -> HolderRecordState:
    """Classify *holder*'s registry record -- see ``HolderRecordState``. Never raises."""
    if holder is None or not holder.record_slug:
        return HolderRecordState.UNLINKED
    if not session_records.is_record_key(holder.record_slug):
        return HolderRecordState.UNREADABLE
    try:
        record = session_records.read_record(holder.record_slug, root=root)
    except FileNotFoundError:
        return HolderRecordState.ABSENT
    except (OSError, session_records.CorruptSessionRecord):
        return HolderRecordState.UNREADABLE
    if record.status in session_records.TERMINAL_STATUSES:
        return HolderRecordState.EXITED
    return HolderRecordState.ACTIVE


@dataclass(frozen=True)
class LeaseClaim:
    """The outcome of a claim_lease call.

    name: the lease name claimed (see build_lease_name).
    decision: ACQUIRED / STAND_DOWN / PROCEED.
    acquired: True iff this call became the lease's holder (decision is
        ACQUIRED); False if the lease was already held by someone else.
    holder: the lease's holder after this call -- the caller's own *holder*
        argument when acquired, otherwise the pre-existing holder found on
        disk (or None if that body was missing/unreadable).
    holder_alive: whether `holder`'s pid is currently alive (pid_alive).
    heartbeat_age_secs: seconds since the lease file's mtime; 0.0 for a
        freshly-acquired lease.
    message: fully-formatted, user-observable line -- callers print this
        verbatim rather than re-deriving it from the other fields.
    holder_record_state: `holder`'s registry record, via its ``record_slug``
        (see HolderRecordState) -- a second axis beside ``holder_alive``,
        which alone still drives ``decision``.

    Task 3994 withdrew an earlier ``holder_session_state`` that read
    ``session_records.read_record(holder.session_slug)``: a lease slug is not a record key,
    so that field was a structural constant. This one reads
    ``holder.record_slug``, a real record key, and spells the no-evidence
    case UNLINKED so it cannot be mistaken for a finding.
    """

    name: str
    decision: LeaseDecision
    acquired: bool
    holder: LeaseHolder | None
    holder_alive: bool
    heartbeat_age_secs: float
    message: str
    holder_record_state: HolderRecordState


def _render_contention_message(
    holder: LeaseHolder | None,
    *,
    holder_alive: bool,
    age_secs: float,
    policy: LeasePolicy,
    ttl: timedelta = LEASE_HEARTBEAT_TTL,
) -> str:
    """Build the exact user-observable contention line callers print verbatim.

    Centralizing this here (rather than in each skill) guarantees the signal
    string is identical everywhere it appears and is unit-testable with an
    injected clock. *holder* may be None (an unreadable/corrupt lease body);
    that is rendered as an explicit placeholder rather than raising.

    WHY THE OLD FORM WAS WITHDRAWN (task 3994, defect 3). This used to render
    ``lease held by {slug} ({alive|dead}, heartbeat {n}s ago)``, collapsing
    two INDEPENDENT axes -- whether the holder's PID is running, and how
    fresh its HEARTBEAT is -- into one parenthetical. On the branch that
    matters most that reads as a flat contradiction: "dead, heartbeat 42s
    ago" invites the reader to conclude the holder is gone and the lease is
    therefore reclaimable, when a fresh heartbeat means precisely the
    opposite (staleness requires BOTH a dead pid AND an aged heartbeat, see
    LEASE_HEARTBEAT_TTL). That exact string misled the 2026-08-08 rotation
    into force-releasing a live holder's lease.

    So each axis now names itself and the DECISION explains itself:
    - live holder -> ``(pid {n} alive, heartbeat {n}s ago)``;
    - dead pid, heartbeat within *ttl* -> states the pid is not running AND
      that the heartbeat is fresh AND that the lease is still held and NOT
      reclaimable, with the remaining TTL in seconds, so a contender cannot
      read "dead" and conclude "mine";
    - dead pid past *ttl* -> phrased as reclaim-eligible with the reclaim
      race lost (we are only here because someone else won it), never as a
      healthy holder;
    - unreadable body -> says so, rather than asserting a fake holder.

    *ttl* is a parameter rather than a module-global read so the
    remaining-TTL arithmetic is pinnable in a unit test. Always exactly ONE
    line, so the signal stays greppable in a transcript.
    """
    action = 'standing down' if policy is LeasePolicy.STAND_DOWN else 'proceeding anyway'
    if holder is None:
        return (
            f'lease held by <unknown> (lease body unreadable, '
            f'heartbeat {int(age_secs)}s ago) — {action}'
        )
    if holder_alive:
        state = f'pid {holder.pid} alive, heartbeat {int(age_secs)}s ago'
    elif age_secs <= ttl.total_seconds():
        remaining = int(ttl.total_seconds() - age_secs)
        state = (
            f'pid {holder.pid} is not running, but its heartbeat is FRESH — '
            f'{int(age_secs)}s ago; the lease is still held and is NOT reclaimable '
            f'for another {remaining}s'
        )
    else:
        state = (
            f'pid {holder.pid} is not running and its heartbeat is STALE — '
            f'{int(age_secs)}s ago, past the {int(ttl.total_seconds())}s TTL; the lease was '
            f'reclaim-eligible and someone else won the reclaim race'
        )
    return f'lease held by {holder.session_slug} ({state}) — {action}'


def _create_and_write_lease(path: Path, holder: LeaseHolder) -> None:
    """``os.open(O_CREAT|O_EXCL)`` + write *holder*'s body (mirrors ArtifactStore.lock_plan).

    Raises ``FileExistsError`` if the lease already exists (lost the race).
    On a write failure, cleans up the empty file it just created so a
    subsequent claim is not falsely blocked by a poison-pill empty file.
    """
    fd = os.open(str(path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    try:
        os.write(fd, holder.to_json().encode())
    except Exception:
        with contextlib.suppress(OSError):
            path.unlink(missing_ok=True)
        raise
    finally:
        os.close(fd)


def _acquired_claim(name: str, holder: LeaseHolder, root: Path | str | None) -> LeaseClaim:
    return LeaseClaim(
        name=name,
        decision=LeaseDecision.ACQUIRED,
        acquired=True,
        holder=holder,
        holder_alive=True,
        heartbeat_age_secs=0.0,
        message=f'lease {name} acquired by {holder.session_slug}',
        holder_record_state=holder_record_state(holder, root=root),
    )


def _read_lease_holder_state(
    path: Path, *, now: datetime
) -> tuple[LeaseHolder | None, bool, float]:
    """Read an existing lease file's ``(holder, holder_alive, heartbeat_age_secs)``.

    *holder* is None when the body is missing/corrupt/unparseable -- callers
    treat that as "held by an unknown holder" (fail toward held, not free)
    rather than raising; *holder_alive* is then False, so a corrupt body
    older than LEASE_HEARTBEAT_TTL is still stale-reapable.

    If *path* itself vanishes before it can be stat'd (e.g. a concurrent
    release/reap in the narrow window between our failed O_EXCL create and
    this read), reports a synthetic age just past LEASE_HEARTBEAT_TTL
    instead of raising -- mirrors the guarded stat() in reap_stale_leases,
    and makes the caller's existing is_stale check reclaim it exactly as it
    would a dead-and-expired holder rather than propagating an uncaught
    FileNotFoundError.
    """
    try:
        mtime = path.stat().st_mtime
    except OSError:
        return None, False, LEASE_HEARTBEAT_TTL.total_seconds() + 1.0
    age_secs = (now - datetime.fromtimestamp(mtime, tz=UTC)).total_seconds()
    try:
        holder = LeaseHolder.from_json(path.read_text(encoding='utf-8'))
    except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError):
        return None, False, age_secs
    return holder, pid_alive(holder.pid), age_secs


def _lease_is_stale(holder: LeaseHolder | None, *, holder_alive: bool, age_secs: float) -> bool:
    """Whether a lease is reapable: heartbeat past LEASE_HEARTBEAT_TTL and no pid vouching for it.

    A running shared-host pid cannot vouch for its session (see session_identity.is_shared_session_host).
    """
    if age_secs <= LEASE_HEARTBEAT_TTL.total_seconds():
        return False
    return holder is None or not holder_alive or session_identity.is_shared_session_host(holder.pid)


def claim_lease(
    name: str,
    *,
    holder: LeaseHolder,
    policy: LeasePolicy = LeasePolicy.STAND_DOWN,
    root: Path | str | None = None,
    now: datetime | None = None,
) -> LeaseClaim:
    """Atomically claim the *name* lease for *holder* (PRD T7 single-owner-per-role).

    Uses ``os.open(O_CREAT|O_EXCL|O_WRONLY)`` -- the identical atomic
    exclusive-create idiom as ``ArtifactStore.lock_plan`` (artifacts.py) --
    so the first caller to create the ``.lease`` file wins outright. When the
    lease is already held, the existing holder's liveness (pid_alive) and
    heartbeat age (file mtime) are checked: if _lease_is_stale, the stale
    lease is reaped and the claim retried once; otherwise the existing holder is
    reported under *policy* (STAND_DOWN/WARN_AND_PROCEED) and the on-disk
    body is never touched (no clobbering). *now* is injectable for
    deterministic tests; defaults to the real UTC clock.
    """
    if now is None:
        now = datetime.now(UTC)
    path = lease_path_for_name(name, root=root)
    path.parent.mkdir(parents=True, exist_ok=True)

    try:
        _create_and_write_lease(path, holder)
    except FileExistsError:
        pass
    else:
        return _acquired_claim(name, holder, root)

    existing_holder, holder_alive, age_secs = _read_lease_holder_state(path, now=now)
    is_stale = _lease_is_stale(existing_holder, holder_alive=holder_alive, age_secs=age_secs)

    if is_stale:
        # Re-verify staleness immediately before unlinking: a competitor
        # could have reaped-and-reclaimed this same stale lease in the gap
        # since the read above (their O_EXCL create winning the race). This
        # narrows -- POSIX has no atomic compare-and-unlink, so it cannot
        # fully eliminate -- that window: if the lease is no longer stale,
        # it must NOT be unlinked, or we would clobber the competitor's
        # brand-new, live lease and silently violate single-owner-per-role.
        existing_holder, holder_alive, age_secs = _read_lease_holder_state(path, now=now)
        is_stale = _lease_is_stale(existing_holder, holder_alive=holder_alive, age_secs=age_secs)

    if is_stale:
        path.unlink(missing_ok=True)
        try:
            _create_and_write_lease(path, holder)
        except FileExistsError:
            # Lost the retry race to a new competitor: report THEIR holder
            # under the normal held/policy path below (retry is attempted
            # at most once -- we do not loop).
            existing_holder, holder_alive, age_secs = _read_lease_holder_state(path, now=now)
        else:
            # Reaped-and-reclaimed a stale lease. The displaced holder was
            # SUPPOSED to be gone (_lease_is_stale), but this is exactly the
            # path that -- with a
            # mis-tuned TTL -- once let a duplicate steal a live-but-quiet
            # watcher's lease (task 2796, THREAD 1). Emit a structured, greppable
            # WARNING naming the displaced holder (slug + pid), the observed
            # heartbeat age, and the new holder, so any residual reclaim of a
            # supposedly-live holder is loud rather than silent. Pure logging --
            # no effect on the claim outcome (fail-safe); *existing_holder* may
            # be None (a corrupt/unreadable displaced body), rendered as an
            # explicit placeholder rather than raising.
            displaced_slug = existing_holder.session_slug if existing_holder is not None else '<unknown>'
            displaced_pid = existing_holder.pid if existing_holder is not None else -1
            logger.warning(
                'claim_lease: reaped stale lease %s (displaced holder=%s pid=%s '
                'heartbeat_age=%.0fs); acquired by %s',
                name,
                displaced_slug,
                displaced_pid,
                age_secs,
                holder.session_slug,
            )
            return _acquired_claim(name, holder, root)

    decision = LeaseDecision.STAND_DOWN if policy is LeasePolicy.STAND_DOWN else LeaseDecision.PROCEED
    return LeaseClaim(
        name=name,
        decision=decision,
        acquired=False,
        holder=existing_holder,
        holder_alive=holder_alive,
        heartbeat_age_secs=age_secs,
        message=_render_contention_message(
            existing_holder, holder_alive=holder_alive, age_secs=age_secs, policy=policy
        ),
        holder_record_state=holder_record_state(existing_holder, root=root),
    )


def heartbeat_lease(
    name: str, *, slug: str, force: bool = False, root: Path | str | None = None
) -> LeaseMutation:
    """Bump the *name* lease file's mtime -- but only if *slug* actually holds it.

    The file mtime remains the SINGLE heartbeat clock (the one
    ``claim_lease`` and ``reap_stale_leases`` both read); ``os.utime`` is
    still the only mutation, and no timestamp is written into the lease body
    where it could silently disagree with mtime.

    A REFUSED result is a deliberate no-op, not a fault: before task 3994
    any caller could refresh any lease, and because staleness requires BOTH
    a dead holder pid AND an aged heartbeat, a stranger beating a dead
    holder's lease kept the AND-guard from ever firing -- the lease became
    structurally unreapable and outlived its holder indefinitely.

    Fail-soft is unchanged: an absent lease is ABSENT -- including one a
    concurrent --force release or reap removes after the ownership gate but
    before the touch -- and any other OSError from ``os.utime`` is FAULTED
    after a logger.error -- never a raise, so a lease-substrate fault cannot
    interrupt a watcher's main loop.
    """
    path = lease_path_for_name(name, root=root)
    terminal, holder = _check_lease_ownership(path, slug=slug, force=force, verb='heartbeat_lease')
    if terminal is not None:
        return terminal
    try:
        os.utime(path, None)
    except FileNotFoundError:
        return LeaseMutation.ABSENT
    except OSError:
        logger.error('heartbeat_lease: failed to touch %s', path, exc_info=True)
        return LeaseMutation.FAULTED
    return LeaseMutation.APPLIED if _is_lease_owner(holder, slug) else LeaseMutation.FORCED


def _is_lease_owner(holder: LeaseHolder | None, slug: str) -> bool:
    """Whether *slug* is the lease's holder.

    THE single ownership predicate -- ``_check_lease_ownership`` gates on it
    and the mutating verbs re-use it to tell APPLIED (I am the owner) from
    FORCED (I overrode someone else), so the two can never drift apart.

    Compares ONLY ``session_slug``, never the pid: the slug is a session's
    stable identity key (see session_records.build_session_slug), while the pid is the
    INDEPENDENT liveness probe. Conflating them would make a legitimate
    re-heartbeat fail whenever pid resolution differed between two
    invocations of the same session. An unreadable body (*holder* is None)
    is owned by nobody -- fail toward held.
    """
    return holder is not None and holder.session_slug == slug


def _check_lease_ownership(
    path: Path,
    *,
    slug: str,
    force: bool,
    verb: str,
) -> tuple[LeaseMutation | None, LeaseHolder | None]:
    """Gate a lease mutation on ownership (task 3994).

    Returns ``(None, holder)`` when the caller MAY proceed, and
    ``(<terminal LeaseMutation>, holder)`` when it must not. Shared by
    ``heartbeat_lease`` and ``release_lease`` so there is exactly ONE
    ownership predicate for the two mutating verbs -- a second copy would
    be free to drift, and "who may touch this lease" is precisely the
    question that must have one answer.

    The (holder, alive, age) read goes through ``_read_lease_holder_state``,
    the same reader ``claim_lease`` decides on, so a refusal log carries the
    identical fields a contention message does -- the holder a refused
    caller is told about is the holder a contending claim would report.

    - no readable holder and no file -- absent from the start, or removed
      mid-check by a concurrent --force release or reap -> (ABSENT, None),
      QUIET: an idempotent release is not a failure and must not
      manufacture an ERROR.
    - body still present but unreadable/corrupt -> not owned by anybody, so
      REFUSED (fail TOWARD held, mirroring _read_lease_holder_state's own
      documented rule -- a corrupt lease is not stranded: reap_stale_leases
      still removes it under the ``corrupt`` rule once past
      LEASE_HEARTBEAT_TTL, and --force covers immediate operator recovery).
    - slug matches -> proceed.
    - slug mismatch -> REFUSED + logger.error naming the verb, the lease,
      the REAL holder (slug + pid), the heartbeat age and the requester, so
      the refusal is greppable and both parties are attributable (INV-2).
    - *force* overrides either refusal, at WARNING, naming both parties.
    """
    holder, _holder_alive, age_secs = _read_lease_holder_state(path, now=datetime.now(UTC))
    if holder is None and not path.exists():
        return LeaseMutation.ABSENT, None
    if _is_lease_owner(holder, slug):
        return None, holder

    holder_slug = holder.session_slug if holder is not None else '<unreadable lease body>'
    holder_pid = holder.pid if holder is not None else -1
    if force:
        logger.warning(
            '%s: FORCED on %s -- requester=%s is not the holder '
            '(holder=%s pid=%s heartbeat_age=%.0fs); overriding on explicit --force',
            verb,
            path.stem,
            slug,
            holder_slug,
            holder_pid,
            age_secs,
        )
        return None, holder
    logger.error(
        '%s: REFUSED on %s -- requester=%s is not the holder '
        '(holder=%s pid=%s heartbeat_age=%.0fs); nothing was touched '
        '(inspect with lease-show, override with --force)',
        verb,
        path.stem,
        slug,
        holder_slug,
        holder_pid,
        age_secs,
    )
    return LeaseMutation.REFUSED, holder


def release_lease(
    name: str, *, slug: str, force: bool = False, root: Path | str | None = None
) -> LeaseMutation:
    """Remove the *name* lease -- but only if *slug* actually holds it.

    *slug* is a REQUIRED keyword argument, not an optional check: an
    opt-in ownership test would leave "evict a live holder's lease" reachable
    from Python and would be the path of least resistance for any future
    caller (task 3994). ``force=True`` is the single, loudly-logged bypass.

    Idempotent and fail-soft, exactly as before: an absent lease is ABSENT
    (quiet -- a second release is not a failure), and an OSError from the
    unlink itself is FAULTED after a logger.error, never a raise.

    Returns the LeaseMutation describing what actually happened; see that
    enum for each value's meaning.
    """
    path = lease_path_for_name(name, root=root)
    terminal, holder = _check_lease_ownership(path, slug=slug, force=force, verb='release_lease')
    if terminal is not None:
        return terminal
    try:
        path.unlink(missing_ok=True)
    except OSError:
        logger.error('release_lease: failed to remove %s', path, exc_info=True)
        return LeaseMutation.FAULTED
    return LeaseMutation.APPLIED if _is_lease_owner(holder, slug) else LeaseMutation.FORCED


@dataclass(frozen=True)
class LeaseStatus:
    """A read-only snapshot of one lease -- what ``lease-show`` prints (task 3994).

    name: the lease name inspected (see build_lease_name).
    state: 'held' (a parseable body), 'unreadable' (the file exists but its
        body is missing/corrupt -- reported, never rendered as a fake
        holder) or 'absent' (no such lease). The discriminator: the holder_*
        identity fields are populated only for 'held'.
    holder_slug / holder_pid: the holder's identity from the body, or None.
    holder_pid_alive: whether that pid is currently running (pid_alive) --
        the LIVENESS axis, independent of freshness.
    heartbeat_ts: ISO-8601 timestamp of the last heartbeat, derived from the
        lease file's mtime (None when absent) -- the FRESHNESS axis.
    heartbeat_age_secs: seconds since that heartbeat.
    reclaimable: True iff a contender could legitimately reap this lease --
        i.e. exactly claim_lease/reap_stale_leases' own staleness predicate,
        _lease_is_stale.
    holder_record_slug: the body's ``record_slug``, or None when it is blank
        or the lease is not 'held'.
    holder_record_state: the holder's registry record (see
        HolderRecordState) -- the record axis, independent of liveness.

    Task 3994 withdrew an earlier ``holder_session`` field that printed a
    constant ``holder_session=absent`` (a lease slug is not a record key).
    ``holder_record_state`` reads the body's ``record_slug`` instead, and
    spells the no-evidence case UNLINKED.
    """

    name: str
    state: str
    holder_slug: str | None
    holder_pid: int | None
    holder_pid_alive: bool
    heartbeat_ts: str | None
    heartbeat_age_secs: float
    reclaimable: bool
    holder_record_slug: str | None
    holder_record_state: HolderRecordState


def lease_status(
    name: str, *, root: Path | str | None = None, now: datetime | None = None
) -> LeaseStatus:
    """Inspect the *name* lease without touching it (backs the ``lease-show`` verb).

    WHY THIS EXISTS (task 3994, defect 3). A lease's heartbeat lives ONLY in
    the file's mtime -- the body carries an immutable ``start_ts`` -- so
    ``cat <lease>`` shows you WHO holds it and WHEN THEY STARTED but not
    whether they are still beating. Diagnosing a contended lease therefore
    meant ``cat`` + ``stat -c %y`` + a TTL comparison done by hand, and
    getting that arithmetic wrong is what let the 2026-08-08 rotation
    conclude a live holder's lease was reclaimable. This replaces the
    archaeology with one command.

    Freshness and liveness are computed by ``_read_lease_holder_state`` --
    the SAME reader claim_lease decides on -- and ``heartbeat_ts`` is derived
    from the age that reader returns rather than from a second stat(), so
    there is exactly one clock and the display cannot disagree with the
    decision. Likewise *reclaimable* is claim_lease's own staleness
    predicate, _lease_is_stale.

    READ-ONLY and fail-soft: never mutates the lease (an inspection that
    bumped the mtime would keep a dead holder's lease unreapable), never
    raises -- an unreadable body is a reported *state*, not an exception.
    *now* is injectable for deterministic tests.
    """
    if now is None:
        now = datetime.now(UTC)
    path = lease_path_for_name(name, root=root)
    if not path.is_file():
        return LeaseStatus(
            name=name,
            state='absent',
            holder_slug=None,
            holder_pid=None,
            holder_pid_alive=False,
            heartbeat_ts=None,
            heartbeat_age_secs=0.0,
            reclaimable=False,
            holder_record_slug=None,
            holder_record_state=HolderRecordState.UNLINKED,
        )
    holder, holder_alive, age_secs = _read_lease_holder_state(path, now=now)
    return LeaseStatus(
        name=name,
        state='held' if holder is not None else 'unreadable',
        holder_slug=holder.session_slug if holder is not None else None,
        holder_pid=holder.pid if holder is not None else None,
        holder_pid_alive=holder_alive,
        heartbeat_ts=(now - timedelta(seconds=age_secs)).isoformat(),
        heartbeat_age_secs=age_secs,
        reclaimable=_lease_is_stale(holder, holder_alive=holder_alive, age_secs=age_secs),
        holder_record_slug=(holder.record_slug or None) if holder is not None else None,
        holder_record_state=holder_record_state(holder, root=root),
    )


@dataclass(frozen=True)
class ReapedLease:
    """One ``.lease`` file removed by reap_stale_leases.

    path: the removed ``<name>.lease`` file.
    lease_name: the lease's identity, derived from the filename itself
        (the stem of *path*) -- never from the body, so a corrupt/missing
        body is still reapable.
    reason: why it was reaped -- ``'stale_pid'`` (a parseable body whose
        pid is not running), ``'shared_host'`` (a parseable body whose pid
        is running but is a shared ``claude agents`` host, which cannot
        vouch for its session) or ``'corrupt'`` (body missing/unparseable),
        each once _lease_is_stale judges the lease stale.
    """

    path: Path
    lease_name: str
    reason: str


def reap_stale_leases(
    root: Path | str | None = None,
    *,
    now: datetime | None = None,
) -> list[ReapedLease]:
    """Sweep ``<root>/leases/*.lease`` and remove stale lease files.

    Structural copy of reap_stale_records (PRD §4.8): identity (which file
    to remove) is derived from the *path* -- the ``<name>.lease`` filename
    stem -- never from the body, so a corrupt or unreadable lease body is
    still reapable. Age is measured from the lease file's mtime, the same
    heartbeat clock claim_lease/heartbeat_lease bump.

    Rules (first match wins):
    - lease body unreadable (missing or corrupt) and age >
      LEASE_HEARTBEAT_TTL -> reaped, reason='corrupt'.
    - a parseable body that _lease_is_stale judges stale -> reaped,
      reason='shared_host' when its pid is running (only a shared host's
      running pid fails to vouch), else reason='stale_pid'.
    - otherwise -> kept -- in particular a LIVE holder pid that can vouch
      for its session is never reaped, regardless of heartbeat age (mirrors
      reap_stale_records' non-terminal live-pid rule).

    *now* is injectable for deterministic tests; defaults to the real UTC
    clock.
    """
    if now is None:
        now = datetime.now(UTC)
    base = leases_dir(root)
    reaped: list[ReapedLease] = []
    if not base.is_dir():
        return reaped

    for lease_path in sorted(base.glob('*.lease')):
        if not lease_path.is_file():
            continue
        lease_name = lease_path.stem
        try:
            mtime = lease_path.stat().st_mtime
        except OSError:
            continue  # vanished mid-sweep (e.g. concurrent reap) -- skip it
        age_secs = (now - datetime.fromtimestamp(mtime, tz=UTC)).total_seconds()

        try:
            holder = LeaseHolder.from_json(lease_path.read_text(encoding='utf-8'))
        except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError):
            holder = None
        holder_alive = holder is not None and pid_alive(holder.pid)
        if not _lease_is_stale(holder, holder_alive=holder_alive, age_secs=age_secs):
            reason = None
        elif holder is None:
            reason = 'corrupt'
        elif holder_alive:
            reason = 'shared_host'
        else:
            reason = 'stale_pid'

        if reason is not None:
            try:
                lease_path.unlink()
            except OSError:
                # A single unreapable lease (permission error, a concurrent
                # reap/release race) must not abort the sweep -- log and
                # move on to the next candidate.
                logger.error('reap_stale_leases: failed to remove %s', lease_path, exc_info=True)
                continue
            reaped.append(ReapedLease(path=lease_path, lease_name=lease_name, reason=reason))

    return reaped


# ---------------------------------------------------------------------------
# Project token from cwd: the enclosing git checkout's main working tree
# (reads git's on-disk layout, gitrepository-layout(5); never raises)
# ---------------------------------------------------------------------------


def _enclosing_checkout(path: Path) -> Path | None:
    """The nearest of *path* and its ancestors holding a ``.git`` entry."""
    for candidate in (path, *path.parents):
        if os.path.exists(candidate / '.git'):
            return candidate
    return None


def _read_git_path(pointer_file: Path, prefix: str = '') -> Path | None:
    """The path a git pointer file (gitfile, ``commondir``) holds on its first line.

    The line must start with *prefix*; a relative path is resolved against
    the pointer file's directory and lexically normalised. Unreadable,
    unprefixed or empty -> None.
    """
    try:
        lines = pointer_file.read_text(encoding='utf-8').splitlines()
    except (OSError, ValueError):
        return None
    if not lines or not lines[0].startswith(prefix):
        return None
    raw = lines[0][len(prefix):].strip()
    if not raw:
        return None
    return Path(os.path.normpath(pointer_file.parent / raw))


def _main_working_tree(checkout: Path) -> Path:
    """The main working tree *checkout* belongs to.

    *checkout* itself when its ``.git`` is a directory, or when its gitfile
    does not lead to a common dir named ``.git`` (submodule, bare-backed
    worktree, broken gitfile).
    """
    dotgit = checkout / '.git'
    if os.path.isdir(dotgit):
        return checkout
    gitdir = _read_git_path(dotgit, prefix='gitdir:')
    common_dir = _read_git_path(gitdir / 'commondir') if gitdir is not None else None
    if common_dir is not None and common_dir.name == '.git':
        return common_dir.parent
    return checkout


def _cwd_project_token(cwd: str) -> str:
    """``parse_spawn_identity``'s default project -- see its step 3."""
    checkout = _enclosing_checkout(Path(cwd)) if os.path.isabs(cwd) else None
    if checkout is not None:
        name = _main_working_tree(checkout).name
        if name:
            return name
    return os.path.basename(cwd.rstrip('/')) or 'unknown'


# ---------------------------------------------------------------------------
# CLI + fail-soft (PRD: a registry fault must never change the spawn's exit code)
# ---------------------------------------------------------------------------

_TITLE_RE = re.compile(r'^(?P<role>[^:]+):(?P<project>[^#\s]+)(?:#(?P<task_id>\S+))?')
"""Matches the documented spawn terminal-title convention (skills/spawn/SKILL.md):
``'<role>:<project>#<task-id> <short-slug>'``, task-id and short-slug optional."""


@dataclass(frozen=True)
class SpawnIdentity:
    """Resolved role/project/task_id/escalation_id for a `launching` write."""

    role: str
    project: str
    task_id: str | None
    escalation_id: str | None


def parse_spawn_identity(
    env: Mapping[str, str],
    title: str,
    prompt: str,
    cwd: str,
) -> SpawnIdentity:
    """Resolve spawn identity: env > title-parse > defaults.

    1. ``CLAUDE_SPAWN_ROLE``/``PROJECT``/``TASK_ID``/``ESCALATION_ID`` in *env*,
       when set to a non-empty value.
    2. For whichever of role/project/task_id env did not supply, a best-effort
       parse of the documented terminal-title convention
       ``'<role>:<project>#<task-id> <short-slug>'`` (skills/spawn/SKILL.md).
       The trailing short-slug is always discarded; a project-level title
       (no ``#``) yields ``task_id=None``.
    3. Defaults: ``role='session'``; ``project`` is the basename of the
       enclosing git checkout's MAIN working tree (a linked worktree resolves
       through its gitfile's ``gitdir`` and ``commondir`` to the checkout it
       belongs to), else ``basename(cwd.rstrip('/'))`` or ``'unknown'``. A
       relative *cwd* is not resolved against the process cwd. The token is
       written raw (unfolded); the cockpit folds at its read boundary.

    *prompt* takes no part in this resolution; it is accepted so callers have
    one function for the whole spawn-identity cascade.
    """
    role = env.get('CLAUDE_SPAWN_ROLE') or None
    project = env.get('CLAUDE_SPAWN_PROJECT') or None
    task_id = env.get('CLAUDE_SPAWN_TASK_ID') or None
    escalation_id = env.get('CLAUDE_SPAWN_ESCALATION_ID') or None

    if (role is None or project is None) and title:
        m = _TITLE_RE.match(title.strip())
        if m:
            role = role or m.group('role')
            project = project or m.group('project')
            if task_id is None:
                task_id = m.group('task_id')

    if role is None:
        role = 'session'
    if project is None:
        project = _cwd_project_token(cwd)

    return SpawnIdentity(role=role, project=project, task_id=task_id, escalation_id=escalation_id)


def start_detached_reap(root: Path | str | None = None) -> bool:
    """Start ``run_reap`` on *root*'s fleet in a detached ``reap`` child and return at once.

    The child gets its own session, no inherited fds, DEVNULL stdio and ``/``
    as its cwd, so it can never hold a caller's pipe open nor keep the
    caller's worktree in use while it runs or queues. It inherits
    ``os.environ`` (so ``DISPLAY`` carries over) with ``CLAUDE_FLEET_ROOT``
    pinned to the absolute fleet root. A start failure is logged to stderr
    only and returns False.
    """
    try:
        env = {**os.environ, 'CLAUDE_FLEET_ROOT': str(fleet_root(root).absolute())}
        subprocess.Popen(
            [sys.executable, str(Path(__file__).resolve()), 'reap'],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            cwd='/',
            env=env,
            start_new_session=True,
            close_fds=True,
        )
    except (OSError, ValueError, subprocess.SubprocessError):
        logger.warning('start_detached_reap: could not start the reap child', exc_info=True)
        return False
    return True


def _parse_end_of_work_inputs(env: Mapping[str, str]) -> tuple[bool, session_records.T3End | None]:
    """Parse (self_exit, t3_end) strictly; an act is set only on the T3 lane, where self-exit means exactly archive."""
    self_exit_raw = env.get('CLAUDE_SPAWN_SELF_EXIT', '') or ''
    t3_end_raw = env.get('CLAUDE_SPAWN_T3_END', '') or ''
    self_exit = self_exit_raw == '1'
    t3_end = session_records.parse_t3_end_input(t3_end_raw)
    if t3_end is not None and self_exit != (t3_end is session_records.T3End.ARCHIVE):
        raise ValueError(
            f'CLAUDE_SPAWN_SELF_EXIT={self_exit_raw!r} contradicts CLAUDE_SPAWN_T3_END={t3_end_raw!r}: '
            'a T3 spawn self-exits exactly when its act is archive'
        )
    return self_exit, t3_end


def _run_launching(env: Mapping[str, str]) -> str:
    """Build + write a LAUNCHING record from CLAUDE_SPAWN_* env; return its dir.

    Every spawn still drives the full mark-then-delete drain (``run_reap``),
    now in a detached ``reap`` child (``start_detached_reap``), so the
    launcher never waits on an O(N) registry pass. That child's stdio is
    DEVNULL and a start failure only logs to stderr, so the printed record dir
    stays the only stdout -- spawn-claude.sh captures it into
    ``SESSION_RECORD_DIR``. Concurrent spawns' reaps serialise on
    ``reap_lock``.
    """
    title = env.get('CLAUDE_SPAWN_TITLE', '') or ''
    prompt = env.get('CLAUDE_SPAWN_PROMPT', '') or ''
    cwd = env.get('CLAUDE_SPAWN_CWD') or os.getcwd()
    launcher_pid_raw = env.get('CLAUDE_SPAWN_LAUNCHER_PID')
    launcher_pid = int(launcher_pid_raw) if launcher_pid_raw else os.getpid()
    self_exit, t3_end = _parse_end_of_work_inputs(env)

    identity = parse_spawn_identity(env, title, prompt, cwd)
    slug = session_records.build_session_slug(identity.role, identity.project, identity.task_id, launcher_pid)
    record_dir = session_records.record_path_for_slug(slug).parent

    record = session_records.SessionRecord(
        session_slug=slug,
        status=session_records.Status.LAUNCHING,
        title=title,
        role=identity.role,
        project=identity.project,
        task_id=identity.task_id,
        escalation_id=identity.escalation_id,
        prompt=prompt,
        cwd=cwd,
        launcher_pid=launcher_pid,
        launcher_identity=capture_for_record(launcher_pid),
        start_ts=datetime.now(UTC).isoformat(),
        result_file=str(record_dir / session_records.RESULT_FILENAME),
        transcript_path=session_records.transcript_path_for_cwd(cwd),
        self_exit=self_exit,
        t3_end=t3_end,
    )
    session_records.write_record(record)
    start_detached_reap()
    return str(record_dir)


def slug_root_from_record_dir(record_dir: str) -> tuple[str, Path]:
    """Derive (slug, root) from a record dir printed by `launching`.

    The dir is ``<root>/sessions/<slug>``, so its name is the slug and its
    grandparent is the root -- this is how the `exit`/`refresh` CLI
    subcommands (which only ever receive this printed dir, not a root) find
    their way back to the same session_records.write_record/session_records.read_record key.
    """
    path = Path(record_dir)
    return path.name, path.parent.parent


def _run_exit(record_dir: str, code: int) -> None:
    slug, root = slug_root_from_record_dir(record_dir)
    record = session_records.update_status(slug, root=root, status=session_records.Status.EXITED, exit_code=code)
    _drop_t3_thread_pointer_if_stale(record, root)


def _run_refresh(record_dir: str, status_value: str) -> None:
    slug, root = slug_root_from_record_dir(record_dir)
    record = session_records.refresh_record(slug, root=root, status=session_records.Status(status_value))
    if record.status in session_records.TERMINAL_STATUSES:
        _drop_t3_thread_pointer_if_stale(record, root)


def _run_set_display(
    record_dir: str,
    kind: str,
    wm_title: str,
    wm_window_id: str | None,
    tmux_target: str | None,
    t3_thread_id: str | None = None,
) -> None:
    slug, root = slug_root_from_record_dir(record_dir)
    session_records.update_display(
        slug,
        root=root,
        kind=kind,
        wm_title=wm_title,
        wm_window_id=wm_window_id,
        tmux_target=tmux_target,
        t3_thread_id=t3_thread_id,
    )


def _run_lease_claim(
    name: str, slug: str, pid: int | None, policy_value: str, *, record_slug: str = ''
) -> None:
    """Run the ``lease-claim`` verb: ALWAYS prints a ``decision=<value>`` line + message.

    *pid* is the claimant's long-lived session pid. Resolving it in code
    rather than in the SKILL.md makes the correct pid STRUCTURAL: the
    ``--pid $$`` defect (task 3994) existed because it depended on every skill
    doc getting one shell token right.

    ON THE CLI PATH IT ARRIVES ALREADY RESOLVED (task 4248): ``main()`` resolves
    the session_identity.SessionIdentity ONCE and derives both the slug and this pid from it, so
    the identity token and the pid written into the lease body cannot
    desynchronise. The ``pid is None`` fallback below is
    therefore a defensive path for a DIRECT caller (a test, a future in-process
    caller), not the CLI's route -- and it stays, so this function remains
    correct standalone rather than depending on a caller it cannot see.
    *record_slug* likewise arrives resolved from ``main()``, from that same
    pid; it is never resolved here, so the fallback path records ''.

    This carries its OWN fail-open guard, independent of main()'s outer
    try/except: a fault raised by claim_lease itself (a corrupt lease body,
    an unwritable leases_dir, ...) must never surface as a silent failure or
    -- worse -- a false stand-down. On any exception here, the caller is
    still told to PROCEED (fail-open), exactly as if the lease were free,
    because a lease-substrate fault must never block a watcher cycle or a
    /unblock session.
    """
    try:
        if pid is None:
            pid = session_identity.resolve_session_pid()
        holder = LeaseHolder(
            session_slug=slug,
            pid=pid,
            start_ts=datetime.now(UTC).isoformat(),
            record_slug=record_slug,
        )
        claim = claim_lease(name, holder=holder, policy=LeasePolicy(policy_value))
    except Exception:
        logger.error('lease-claim %s failed', name, exc_info=True)
        print(f'decision={LeaseDecision.PROCEED.value}')
        print(f'lease-claim {name} faulted; proceeding (fail-open)')
        # `slug=` is emitted here too, while `holder_liveness=` deliberately is
        # not: the fault made the HOLDER unknown, not US. This is also where
        # the line is most useful -- the claim's outcome is least certain.
        print(f'slug={slug}')
        return
    print(f'decision={claim.decision.value}')
    print(claim.message)
    # ADDITIVE and LAST (task 3994 defect 4; extended by task 4248): `decision=`
    # stays line 1 and the message line 2, so a skill parser that reads only the
    # first two lines is unaffected -- which is why these are appended lines
    # rather than a fourth LeaseDecision token a not-yet-reloaded skill would
    # face as an unrecognised value. Lease SEMANTICS are unchanged: we never
    # auto-steal. This only lets a stand-down name a machine-readable owner, so
    # a watcher can file a DecisionRecord for a provably-gone holder instead of
    # exiting silently (INV-6/INV-7).
    #
    # TWO appended lines now: `holder_liveness=` (about the CONTENDING HOLDER)
    # and, last, `slug=` (about THIS CALLER). The second exists because the
    # skills no longer pass `--slug` at all: it makes the CLI-derived identity
    # LEGIBLE, so an agent whose later heartbeat returns `result=refused` can
    # cross-check what it claimed as against `lease-show`'s holder_slug. It is a
    # DIAGNOSTIC, explicitly NOT a value for the skills to thread into the next
    # verb -- a shell variable cannot survive to the next Claude Code Bash tool
    # call (each is a fresh `/bin/bash -c`), which is precisely why the
    # derivation had to move into the CLI rather than be printed for quoting.
    #
    # The MUTATING verbs deliberately gain no `slug=` line: `_run_lease_mutation`
    # already names the presented slug alongside the real holder in its refusal
    # message, so a second line would restate what the caller was just told.
    #
    # THREE values, not two: an ACQUIRED claim has no contending holder at all
    # -- the holder is this caller -- so it prints `none`. This first shipped
    # emitting `held` on a successful claim, which reads as "someone else has
    # it": an ambiguous machine-readable token in a change whose entire purpose
    # is removing ambiguous lease signals. The skills only branch on this line
    # under `decision=stand-down`, so `none` narrows the vocabulary rather than
    # breaking a reader. The line is absent only on the fail-open path above,
    # where a substrate fault means we genuinely know nothing about a holder
    # and must not assert one either way.
    #
    # `holder_record=` (task 4237), last of all, is a SECOND, INDEPENDENT orphan
    # axis: the holder's registry record, per HolderRecordState -- the
    # claimant's own on an acquired claim, the existing holder's on a contended
    # one. It deliberately does not change the `orphaned` predicate below: every
    # lease claimed before LeaseHolder.record_slug existed reads `unlinked`, so
    # requiring corroboration would silence today's orphan signal for that whole
    # population. Reading the two lines together is the skills' job; collapsing
    # them into one verdict is what task 3994 defect 3 forbade.
    if claim.acquired:
        print('holder_liveness=none')
        print(f'slug={slug}')
        print(f'holder_record={claim.holder_record_state.value}')
        return
    # A SINGLE signal, deliberately: `orphaned` means exactly "the pid
    # recorded in the lease body is not running". This predicate used to
    # AND in `claim.holder_session_state in ('absent', 'exited')`, which
    # WHENEVER THERE WAS A HOLDER measured as a structural constant 'absent'
    # -- a lease slug is not a session-registry record key (see
    # LeaseHolder.session_slug) -- so dropping it is an honesty fix, not a
    # behaviour change. The one case where that conjunct was NOT constant is
    # an unreadable body, whose holder is None and which therefore resolved
    # 'unknown' -> held; `claim.holder is not None` reproduces that
    # explicitly, on its real reason (no holder means no pid to probe, so
    # there is nothing to call orphaned) rather than as a side effect of a
    # withdrawn lookup. The emitted values are byte-identical before and
    # after, in every branch that HAS a contending holder.
    #
    # It is sound only because session_identity.resolve_session_pid now records the
    # long-lived `claude` pid; the old `--pid $$` was the transient Bash-tool
    # wrapper, always dead, which would have made this a tautology. Both
    # error directions are safe: pid REUSE yields a false `held` (silence,
    # the status quo, never a steal), and a false `orphaned` still triggers
    # only a NON-destructive response (file a DecisionRecord, then exit --
    # never auto-steal, which is the duplicate-spawn incident path). A holder
    # claimed on session_identity.resolve_session_pid's DEGRADED path (pid 0: an unresolvable
    # CLAUDE_PID, or a shared `claude agents` host) therefore reads `orphaned`
    # while possibly alive -- the
    # deliberate, bounded cost of keeping such a lease reapable at all.
    orphaned = claim.holder is not None and not claim.holder_alive
    print(f'holder_liveness={"orphaned" if orphaned else "held"}')
    # THIS CALLER's slug, never the holder's -- see the block comment above.
    print(f'slug={slug}')
    print(f'holder_record={claim.holder_record_state.value}')


def _run_lease_mutation(
    verb: str,
    name: str,
    slug: str,
    mutate: Callable[[], LeaseMutation],
) -> None:
    """Drive a mutating lease verb and print ``result=<value>`` + a human line.

    The machine-readable token is ALWAYS the first line, mirroring
    ``lease-claim``'s ``decision=`` so an agent parses all four lease verbs
    with one rule. On REFUSED/FORCED a second line names the lease's actual
    holder, so a caller told "no" also learns WHO owns it and can act
    (inspect with ``lease-show``) rather than guess.

    The holder is read BEFORE *mutate* runs: a FORCED ``lease-release``
    deletes the very file the holder's identity lives in, so reading after
    the fact would report ``<unreadable lease body>`` for exactly the case
    where naming the displaced holder matters most.

    ``--force`` reaches the mutation through *mutate*'s closure and nowhere
    else -- this helper takes no ``force`` parameter, deliberately. It had one
    briefly, unread by any line of the body, which invites the next reader to
    assume the wording below branches on it; the RESULT (FORCED vs REFUSED) is
    the only thing that decides what is printed.
    """
    holder, holder_alive, age_secs = _read_lease_holder_state(
        lease_path_for_name(name), now=datetime.now(UTC)
    )
    result = mutate()
    print(f'result={result.value}')
    if result not in (LeaseMutation.REFUSED, LeaseMutation.FORCED):
        return
    holder_slug = holder.session_slug if holder is not None else '<unreadable lease body>'
    holder_pid = holder.pid if holder is not None else -1
    alive_word = 'alive' if holder_alive else 'not running'
    holder_facts = f'{holder_slug} (pid {holder_pid} {alive_word}, heartbeat {int(age_secs)}s ago)'
    if result is LeaseMutation.REFUSED:
        print(
            f'{verb} {name} refused: held by {holder_facts}, not by {slug} '
            f'— inspect with `lease-show --name {name}`'
        )
    else:
        print(f'{verb} {name} FORCED by {slug} over holder {holder_facts}')


def _run_lease_heartbeat(name: str, slug: str, force: bool = False) -> None:
    _run_lease_mutation(
        'lease-heartbeat',
        name,
        slug,
        lambda: heartbeat_lease(name, slug=slug, force=force),
    )


def _run_lease_release(name: str, slug: str, force: bool = False) -> None:
    _run_lease_mutation(
        'lease-release',
        name,
        slug,
        lambda: release_lease(name, slug=slug, force=force),
    )


def _run_lease_show(name: str) -> None:
    """Run the ``lease-show`` verb: print one lease's full state as key=value lines.

    The replacement for ``cat <lease>`` + ``stat -c %y`` archaeology (task
    3994 defect 3), in the same machine-readable ``<key>=<value>`` form as
    ``decision=``/``result=`` so an agent parses every lease verb with one
    rule. ``state=`` is the discriminator: the holder_* identity lines are
    printed only when the body actually parsed, so an unreadable body reads
    as unreadable rather than as a holder named ``<unknown>``. Booleans are
    rendered lowercase (``true``/``false``) so a shell test is trivial.
    ``holder_record=`` and ``holder_record_slug=`` are ADDITIVE and LAST,
    the convention ``_run_lease_claim`` states.

    Read-only and fail-soft, like lease_status itself: it never mutates the
    lease and never raises.
    """
    status = lease_status(name)
    print(f'name={status.name}')
    print(f'state={status.state}')
    if status.state == 'absent':
        return
    if status.holder_slug is not None:
        print(f'holder_slug={status.holder_slug}')
    if status.holder_pid is not None:
        print(f'holder_pid={status.holder_pid}')
    print(f'holder_pid_alive={str(status.holder_pid_alive).lower()}')
    print(f'heartbeat_ts={status.heartbeat_ts}')
    print(f'heartbeat_age_secs={int(status.heartbeat_age_secs)}')
    print(f'reclaimable={str(status.reclaimable).lower()}')
    print(f'holder_record={status.holder_record_state.value}')
    if status.holder_record_slug is not None:
        print(f'holder_record_slug={status.holder_record_slug}')


def _run_lease_reap() -> list[ReapedLease]:
    return reap_stale_leases()


def _resolve_filing_record_slug(local_id: str, project: str) -> str:
    """Return the record_slug a write-decision filing of *local_id* stamps.

    That is resolve_own_record_slug's slug, else '' after a WARNING, because a
    blank one leaves that cockpit row unable to focus its filer's session for
    the record's life (task 7527).
    """
    slug = resolve_own_record_slug()
    if slug is not None:
        return slug
    logger.warning(
        'write-decision --id %s --project %s: $%s=%r resolves to no vouched session record '
        '(unset, unusable or a shared `claude agents` host pid; or no sessions-by-pid/<pid> '
        'pointer whose record still names that pid as its owner), so this decision gets a '
        'BLANK record_slug and its cockpit row cannot focus the filing session (Enter-to-focus) '
        'for the life of the record. This alone does not refuse the filing; an id on stdout is '
        'what says it landed.',
        local_id,
        project,
        session_identity.SESSION_PID_ENV,
        os.environ.get(session_identity.SESSION_PID_ENV),
    )
    return ''


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog='session_registry')
    sub = parser.add_subparsers(dest='verb', required=True)

    sub.add_parser('launching', help='write a LAUNCHING record from CLAUDE_SPAWN_* env')

    exit_p = sub.add_parser('exit', help='set status=exited + exit_code on a record')
    exit_p.add_argument('--record', required=True, help='record dir, as printed by `launching`')
    exit_p.add_argument('--code', required=True, type=int)

    refresh_p = sub.add_parser('refresh', help='upsert a record with a new status (hook seam)')
    refresh_p.add_argument('--record', required=True, help='record dir')
    refresh_p.add_argument('--status', required=True, choices=[s.value for s in session_records.Status])

    set_display_p = sub.add_parser(
        'set-display', help='stamp display (kind/title/target) on a record'
    )
    set_display_p.add_argument('--record', required=True, help='record dir, as printed by `launching`')
    set_display_p.add_argument('--kind', required=True, choices=[k.value for k in session_records.DisplayKind])
    set_display_p.add_argument('--tmux-target', default=None)
    set_display_p.add_argument('--wm-title', default='')
    set_display_p.add_argument('--wm-window-id', default=None)
    set_display_p.add_argument('--t3-thread-id', default=None)

    sub.add_parser('reap', help='sweep and remove stale session records')

    lease_claim_p = sub.add_parser('lease-claim', help='claim a single-owner-per-role lease (T7)')
    lease_claim_p.add_argument('--name', required=True, help='lease name, see build_lease_name')
    lease_claim_p.add_argument(
        '--slug',
        default=None,
        help="this claimant's own session_slug; defaults to "
        'default_lease_slug(--name) (`<name>-$CLAUDE_PID`, or '
        '`<name>-$CLAUDE_PID-$CLAUDE_CODE_SESSION_ID` under a shared `claude agents` '
        'host). Only pass this as '
        'an explicit operator override -- the derived value is what every '
        'later lease-heartbeat/lease-release re-derives and matches against',
    )
    lease_claim_p.add_argument(
        '--pid',
        type=int,
        default=None,
        help="this claimant's own long-lived session pid; defaults to "
        'resolve_session_pid() ($CLAUDE_PID, or 0 under a shared `claude agents` host, '
        'whose pid outlives this session). Only pass this as an explicit '
        'operator override -- never `$$`, which is the transient Bash-tool shell. '
        'This is the lease body\'s LIVENESS pid only: it does not supply the '
        "slug's identity (which always derives from $CLAUDE_PID), so --pid alone "
        'does not satisfy an underivable --slug -- lease-heartbeat/lease-release '
        'have no --pid, and a slug only one verb can derive is not an identity',
    )
    lease_claim_p.add_argument(
        '--policy',
        choices=[p.value for p in LeasePolicy],
        default=LeasePolicy.STAND_DOWN.value,
        help='how to respond when the lease is already held live',
    )

    lease_heartbeat_p = sub.add_parser('lease-heartbeat', help='touch a held lease (heartbeat)')
    lease_heartbeat_p.add_argument('--name', required=True)

    lease_release_p = sub.add_parser('lease-release', help='release a held lease')
    lease_release_p.add_argument('--name', required=True)

    # Ownership (task 3994): both MUTATING verbs act only for the lease's
    # actual holder, identified by slug. AMENDED BY TASK 4248 -- the slug is
    # now DERIVED (default_lease_slug, in main()) rather than required from the
    # caller. That does NOT retract 3994's guard. The defect 3994 fixed was that
    # both verbs mutated UNCONDITIONALLY, with no ownership check at all; the
    # check still runs, and it now compares against a slug derived from THIS
    # caller's own $CLAUDE_PID, so a stranger session derives a different slug
    # and is still `result=refused` with the lease body untouched. What is
    # retired is only the requirement that the token be SPELLED IN SHELL, which
    # is exactly what let it drift: 3994 left three SKILL.md files assembling
    # `<role>-<project>-${CLAUDE_PID:-$PPID}` by hand, and $PPID is measurably
    # not stable across Claude Code tool calls. `required=True` was a forcing
    # function to make the call sites pass a slug at all; the call sites have
    # now been updated, and the derivation is the stronger guarantee -- a caller
    # can no longer MIS-spell a token it never spells. An unresolvable
    # $CLAUDE_PID still exits 2 (see main()); it never falls back silently.
    for _lease_mutation_p in (lease_heartbeat_p, lease_release_p):
        _lease_mutation_p.add_argument(
            '--slug',
            default=None,
            help='the slug you claimed this lease with; a mismatch is refused. '
            'Defaults to default_lease_slug(--name) (`<name>-$CLAUDE_PID`, or '
            '`<name>-$CLAUDE_PID-$CLAUDE_CODE_SESSION_ID` under a shared `claude agents` '
            'host), the same derivation lease-claim uses -- pass it only as an operator '
            'override',
        )
        _lease_mutation_p.add_argument(
            '--force',
            action='store_true',
            help='operator recovery: act even when you are not the holder; logged loudly',
        )

    lease_show_p = sub.add_parser(
        'lease-show',
        help="inspect a lease (holder, pid liveness, heartbeat freshness); "
        'read-only. Use this instead of `cat`, which cannot show freshness',
    )
    lease_show_p.add_argument('--name', required=True, help='lease name, see build_lease_name')

    sub.add_parser('lease-reap', help='sweep and remove stale leases')

    decision_cli.add_verb_parsers(sub)
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entry point.

    ALWAYS returns 0 for a session or lease verb: a fault in its core logic
    is logged loudly (stderr, via the standard logging machinery) and
    swallowed here rather than raised, so a registry fault can never change
    the exit code of the bash caller (spawn-claude.sh) that invokes this
    script. That covers runtime FAULTS only: a malformed invocation exits 2
    from argparse's parse_args() below, before the try/except.

    A decision verb goes to decision_cli.run_verb, which owns the decision
    half of this policy, including why close-decision and reopen-decision
    exit non-zero on a refusal. The filer's record slug it needs is session
    identity, so this module hands it _resolve_filing_record_slug.
    """
    parser = _build_parser()
    args = parser.parse_args(argv)

    # ONE derivation point for the lease slug, shared by all three lease verbs
    # (task 4248). It lives HERE, in main(), and not inside _run_lease_claim /
    # _run_lease_mutation, because all three must derive a BYTE-IDENTICAL slug
    # or the whole scheme fails -- a claim the holder's own later heartbeat
    # cannot match is precisely the drift this change removes. One shared point
    # makes that structural instead of a three-way convention nobody enforces.
    # An explicit --slug always wins: it stays the operator's override.
    #
    # ONE IDENTITY RESOLUTION TOO. `lease-claim` both WRITES a liveness pid
    # (into the lease body) and EMBEDS an identity (the derived slug), so both
    # derive from the one session_identity.SessionIdentity resolved here: to default_lease_slug
    # via identity=, and to _run_lease_claim via args.pid
    # (session_identity.lease_liveness_pid). Pinned by
    # test_main_lease_claim_derives_the_body_pid_and_the_slug_pid_from_one_resolution.
    # The two legitimately differ under a shared `claude agents` host (body pid
    # 0, slug carrying the host pid and session id) and under an explicit --pid,
    # which overrides the BODY's liveness pid only, never the slug -- the
    # identity has to stay derivable by the mutating verbs, which have no --pid.
    # The body pid also keys the holder's `record_slug` lookup below.
    #
    # Resolution is DEMAND-DRIVEN, and only lease-claim's liveness pid can log:
    # a degradation WARNING is useful only when it is TRUE of the work being
    # done, and a heartbeat writes no pid.
    if args.verb in ('lease-claim', 'lease-heartbeat', 'lease-release'):
        derives_body_pid = args.verb == 'lease-claim' and args.pid is None
        identity = None
        if args.slug is None or derives_body_pid:
            identity = session_identity.resolve_session_identity()
            if derives_body_pid:
                args.pid = session_identity.lease_liveness_pid(identity)
            if args.slug is None:
                args.slug = default_lease_slug(args.name, identity=identity)
        # FAIL LOUDLY, never silently. default_lease_slug returns None only
        # when this session's identity is underivable, and it refuses to
        # synthesize `<name>-0` (or `<name>-<shared host pid>`) because that
        # token is IDENTICAL for two such sessions -- each would then pass the
        # other's _is_lease_owner check, reopening 3994 defect 1. So we stop here.
        #
        # This is NOT a breach of _run_lease_claim's fail-open contract.
        # Fail-open covers lease-SUBSTRATE faults (a corrupt lease body, an
        # unwritable leases_dir) -- conditions the caller cannot fix and must
        # not be blocked by. An underivable slug is a CALLER/ENVIRONMENT INPUT
        # error, exactly the class argparse already exits 2 for, and it is
        # fixable at the call site (`--slug`). Note also that pre-4248 a
        # slug-less lease verb exited 2 as well: the undeterminable case KEEPS
        # its current exit code, and only the derivable case gained a default.
        #
        # parser.error raises SystemExit -- a BaseException -- so it
        # deliberately escapes main()'s `except Exception`, which swallows
        # faults and returns 0. That escape is the point: this must reach the
        # shell as a nonzero exit, not as a silently-successful no-op.
        if not args.slug:
            cause = (
                f'${session_identity.SESSION_PID_ENV}={identity.claude_pid} names a shared '
                f'`claude agents` host, so this session\'s identity needs '
                f'${session_identity.SESSION_ID_ENV}, which is unset '
                'or unusable'
                if identity is not None and identity.shared_host
                else f'${session_identity.SESSION_PID_ENV} is unset or unusable'
            )
            parser.error(
                f'cannot derive the lease slug for --name {args.name}: {cause}, '
                'so this session has no stable identity to claim, '
                'heartbeat or release a lease with. Refusing to invent one -- a synthesized '
                'token would collide with every other degraded session and let each act on '
                "the other's lease. Fix the environment, or pass an explicit "
                '--slug <stable-token> that this session will re-use on every later lease verb. '
                'Note --pid does NOT substitute for --slug: it sets the lease body\'s liveness '
                'pid, not this session\'s identity, and lease-heartbeat/lease-release have no '
                '--pid at all -- only --slug is honoured by all three verbs.'
            )
        if args.verb == 'lease-claim':
            args.record_slug = ''
            if args.pid is not None:
                args.record_slug = session_records.resolve_session_slug_for_pid(args.pid) or ''

    if args.verb in decision_cli.DECISION_VERBS:
        return decision_cli.run_verb(args, filing_record_slug=_resolve_filing_record_slug)

    try:
        if args.verb == 'launching':
            print(_run_launching(os.environ))
        elif args.verb == 'exit':
            _run_exit(args.record, args.code)
        elif args.verb == 'refresh':
            _run_refresh(args.record, args.status)
        elif args.verb == 'set-display':
            _run_set_display(
                args.record,
                args.kind,
                args.wm_title,
                args.wm_window_id,
                args.tmux_target,
                args.t3_thread_id,
            )
        elif args.verb == 'reap':
            run_reap()
        elif args.verb == 'lease-claim':
            _run_lease_claim(
                args.name, args.slug, args.pid, args.policy, record_slug=args.record_slug
            )
        elif args.verb == 'lease-heartbeat':
            _run_lease_heartbeat(args.name, args.slug, args.force)
        elif args.verb == 'lease-release':
            _run_lease_release(args.name, args.slug, args.force)
        elif args.verb == 'lease-show':
            _run_lease_show(args.name)
        elif args.verb == 'lease-reap':
            _run_lease_reap()
        else:
            logger.error('session_registry %s: the parser accepts this verb but no runner handles it', args.verb)
    except Exception:
        logger.error('session_registry %s failed', args.verb, exc_info=True)
        return 0

    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
