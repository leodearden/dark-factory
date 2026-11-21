"""orchestrator.session_reaper — the fleet registry's liveness sweeps and reaps (Attention Rail T3, PRD §4.8).

A session that dies without writing ``exited`` leaves a non-terminal record
behind. The sweeps here mark such records EXITED: a launcher judged dead past
the heartbeat TTL, a wm-display session whose window is gone, a T3-hosted
session whose thread ended. The reaps then delete terminal records past their
TTL and the pointers no record vouches for. ``run_reap`` runs them in that
order, mark-then-delete, and is what the ``reap`` verb of
``session_registry.py`` calls.

It imports only the stdlib, this checkout's stdlib-only ``shared`` leaves and
its siblings ``fleet_files``, ``session_records`` and ``session_windows``. It
never imports ``session_registry``: in a bare-shell (tier-1) run
session_registry is ``__main__``, so an upward import would load a second copy
of it. The bare-shell contract is pinned by
``orchestrator/tests/test_session_registry.py::TestStdlibOnlySelfContainment``.
"""

from __future__ import annotations

import contextlib
import dataclasses
import json
import logging
import os
import shutil
import sys
import time
from collections.abc import Callable, Iterator, Sized
from collections.abc import Set as AbstractSet
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING

# Reach this checkout's stdlib-only shared leaves from a bare shell; rationale in
# test_session_registry.py::TestStdlibOnlySelfContainment.
if (
    _CHECKOUT_SHARED_SRC := str(Path(__file__).resolve().parents[3] / 'shared' / 'src')
) not in sys.path and Path(_CHECKOUT_SHARED_SRC).is_dir():
    sys.path.append(_CHECKOUT_SHARED_SRC)

from shared.proc_liveness import pid_alive
from shared.process_identity import Liveness, ProcessIdentity
from shared.session_liveness import judge

# Siblings are imported relative to orchestrator.session_reaper, or else from this
# file's own directory: loaded by path, as a tier-1 sibling or under a foreign dotless
# name, that directory is on sys.path only if Python or the importer put it there, so it
# is borrowed for these imports alone. Each mode is pinned in
# test_session_registry.py::TestStdlibOnlySelfContainment.
# TYPE_CHECKING points the type checker at the package arm, the only one it can resolve.
if TYPE_CHECKING or __package__:
    from . import session_records, session_windows
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
        import session_records
        import session_windows
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
    if path.name != str(pid) or not session_records.is_session_pointer_pid(pid):
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


def drop_t3_thread_pointer_if_stale(record: session_records.SessionRecord, root: Path | str | None) -> None:
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
        except FileNotFoundError:
            continue
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
    positive pid, it names a shared ``claude agents`` host, which no reader
    follows (``session_records.is_session_pointer_pid``), its content is not
    a record key, the record it names has no ``record.json``, or its pid is
    judged DEAD by
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
    paths this sweep removed; an entry a concurrent sweep already removed is
    skipped silently, and any other entry that cannot be removed is logged
    and skipped.
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
    drop_t3_thread_pointer_if_stale(record, root)
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
    spawn's ``session_registry.py::_run_launching`` starts, not just by an
    explicit CLI ``reap``. This is self-correcting
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


def _wm_window_id_of(record: session_records.SessionRecord) -> str | None:
    """Canonical window id of a wm-display *record*; ``None`` for other displays or a bad id."""
    if record.display is None or record.display.kind != session_records.DisplayKind.WM.value:
        return None
    return session_windows.canonical_window_id(record.display.wm_window_id)


def _names_window(window_id: str) -> Callable[[session_records.SessionRecord], bool]:
    return lambda record: _wm_window_id_of(record) == window_id


def mark_windowless_wm_sessions_exited(
    root: Path | str | None = None,
    *,
    windows: session_windows.WindowProbe | None = None,
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
    (``session_windows.canonical_window_id`` returns ``None``) is ALSO left untouched --
    we cannot prove the window is gone, so we fail toward keep and defer to
    the pid/TTL sweep.

    HARD fail-soft: if the first ``live_window_ids()`` probe is ``None``
    (wmctrl missing or failing, or the probe raising -- see
    ``session_windows.py::live_window_ids``), this returns ``[]`` immediately and
    marks NOTHING. "No window evidence" (a headless host, an X11 hiccup)
    must never be treated as "every window is gone" -- that would mass-mark
    every wm session on the fleet EXITED.

    Two-probe rule: a record is marked only on a probe taken AFTER it was
    read. ``session_registry.py::_run_launching``'s detached reap runs
    concurrently with the spawned session's SessionStart hook, which writes
    its ``wm_window_id``
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
    process is connected to -- whatever ``run_reap`` (in the detached child
    ``session_registry.py::_run_launching`` starts, or a directly-invoked
    CLI) happens to inherit as ``$DISPLAY`` -- never
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

    *windows* is the window probe both probes read; ``None`` reads wmctrl
    (``session_windows.py::wmctrl_window_probe``).
    """
    probe = windows or session_windows.wmctrl_window_probe()
    live_ids = probe.live_window_ids()
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
    confirmed_live_ids = probe.live_window_ids()
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

    The ``reap`` verb, for the operator and for
    ``session_registry.py::_run_launching``'s detached child alike.
    Concurrent calls queue on ``reap_lock``. A call whose turn
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
