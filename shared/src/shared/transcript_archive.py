"""Best-effort archival of per-task Claude Code agent transcripts.

The per-task ``CLAUDE_CONFIG_DIR`` (see :class:`shared.config_dir.TaskConfigDir`)
stores each agent session's raw JSONL transcript under
``<config_dir>/projects/<enc>/<session_id>.jsonl`` (plus subagent transcripts
at ``<config_dir>/projects/<enc>/<session_id>/subagents/agent-*.jsonl``). That
config dir lives *inside* the task worktree and is destroyed at teardown, so
this module copies the transcripts to a durable archive root outside the
worktree while each session is still born-complete.

Design properties (see plans/agent-transcript-archival-prd.md, task α):

* **Credential-safe by construction** — only ``*.jsonl`` files *under*
  ``projects/`` are ever globbed. The per-task OAuth ``.credentials.json``
  lives at the config-dir root and is structurally unreachable, as are any
  non-``.jsonl`` files.
* **Idempotent** — the source ``.jsonl`` mtime is mirrored onto the archived
  copy via :func:`os.utime`; an already-current archive is skipped. A resumed
  session only ever grows its transcript, so mtime strictly advances and the
  grown transcript is re-archived (last-write-wins).
* **Best-effort / never-raises** — any per-file I/O error is caught and
  logged once as a structured WARNING (``path``/``task_id``/``errno``) — the
  production signal an operator greps for a systemic breakage — while the
  module-level :data:`_ARCHIVAL_FAILURES` counter accrues the same failures as
  machine-readable substrate for a future health/digest consumer. The function
  is total so the producer can call it inside ``_invoke``'s ``finally`` during
  exception propagation.
* **Never publishes a partial transcript** — each copy is written to a
  staging sibling and ``os.replace``'d onto its final name, so an interrupted
  or ENOSPC-truncated copy is discarded rather than left at the canonical
  archive path (see :func:`_archive_one`).

The archive is a VERBATIM copy: one plain, greppable ``.jsonl`` corpus, so
``rg`` works across it and no reader needs a decompression branch (task 3618,
leaf α of plans/transcript-preservation-seam-prd.md).

The module also owns the **read** side of that archive:
:func:`durable_archive_path` is the single session-id-keyed locator into it
(task 3727 / plans/session-resume-eligibility-seam-prd.md §8), so every reader
resolves the layout from the same place the writer defines it and the two
cannot drift. It is strictly read-only — glob and ``stat``, nothing created,
moved, deleted or decompressed. Its glob stays FORMAT-AGNOSTIC (``.jsonl*``)
even now that the writer emits only plain ``.jsonl``: archives written BEFORE
task 3618's flag day are still ``.jsonl.gz`` on disk, and task 3618's own
gunzip migration is a separate pass, so a session-id lookup that matched only
``.jsonl`` would silently miss every pre-3618 archive. *Restoring* an archived
transcript back into a live config dir is task 3578's, under task 3619's
archive-before-delete guard; that restore reuses this locator rather than
growing a second glob.
"""

from __future__ import annotations

import errno
import logging
import os
import shutil
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Module-level count of per-file archival failures this process.
#
# Loud-over-silent (design-invariants INV-2/INV-4) is carried TODAY by the
# per-file structured WARNING emitted in _record_failure (path/task_id/errno):
# that log line is the production signal a systemic breakage (e.g. disk full →
# every archive fails) surfaces on. This counter is the machine-readable
# companion to that log — process-cumulative substrate a later health/digest
# consumer can sample to render a climbing failure rate. It is deliberately
# owned-but-not-yet-consumed by this α task, exactly as α lays down the
# retention.* config knobs here for the δ GC consumer (task 2731) to enforce;
# the small read/reset accessor pair below keeps tests isolated in the meantime.
_ARCHIVAL_FAILURES: int = 0

# Suffix of the staging file every archive copy is written to before being
# atomically moved onto its final name. Deliberately NOT ``.jsonl``-terminated,
# mirroring migrate_transcript_archive_gunzip.STAGING_SUFFIX: debris stranded by
# a hard kill matches no reader's ``*.jsonl`` glob, so it is inert rather than a
# transcript lookalike, and the next archival of the same session rewrites it.
_STAGING_SUFFIX = '.archive-tmp'

# The single notification seam onto the counter above. _record_failure is the
# one funnel BOTH producers (archive_task_transcripts' copy, and
# archive_before_delete's move) already pass through, so hanging one hook there
# reaches every archival failure with no second counter and no per-producer
# callback that could drift out of step with the first.
#
# It stays a bare mechanism deliberately. The POLICY that consumes it — a rate
# threshold, a window, an escalation — needs the live orchestrator config to
# honour a hot reload, and this module is on the PURE_STDLIB_LEAVES contract
# (shared/tests/test_pure_stdlib_leaves.py), so the policy lives in the
# consumer (Harness) and only the notification lives here.
_ON_ARCHIVAL_FAILURE: Callable[[dict[str, Any]], None] | None = None


def _archival_failures() -> int:
    """Return the current module-level archival failure count (test aid)."""
    return _ARCHIVAL_FAILURES


def _reset_archival_failures() -> None:
    """Reset the module-level archival failure count (test isolation).

    Clears the installed failure hook too. Both are process-global state that a
    test can leave behind, and this is the one call every test in the suite
    already makes for isolation — resetting only half of it would let one
    test's callback be fed another test's failures.
    """
    global _ARCHIVAL_FAILURES, _ON_ARCHIVAL_FAILURE
    _ARCHIVAL_FAILURES = 0
    _ON_ARCHIVAL_FAILURE = None


def set_archival_failure_hook(
    hook: Callable[[dict[str, Any]], None] | None,
) -> None:
    """Install (or with ``None``, remove) the archival-failure notification.

    *hook* is invoked once per failed file with the same structured payload the
    WARNING carries — ``{'task_id': ..., 'path': ..., 'errno': ...}`` — so a
    consumer sees exactly what an operator greps for, and the log line and the
    machine-readable signal cannot disagree.

    This is the seam the α task described as owned-but-not-yet-consumed:
    ``_ARCHIVAL_FAILURES`` is substrate a consumer must poll, which is no use
    to something that needs to react to a BURST. The consumer arrives in
    ``Harness``, which applies a live-configured rate threshold and files one
    deduped escalation per window.

    Contract: a hook that raises is swallowed and logged, never propagated —
    see :func:`_record_failure`. One hook at a time, last install wins; the
    seam has exactly one consumer by design, and a list would invite the
    fan-out this module has no reason to own.
    """
    global _ON_ARCHIVAL_FAILURE
    _ON_ARCHIVAL_FAILURE = hook


def _record_failure(
    src: Path,
    task_id: str,
    exc: BaseException,
    *,
    detail: str = 'failed to archive',
) -> None:
    """Count and loudly (but non-fatally) log a single per-file archive failure.

    Increments :data:`_ARCHIVAL_FAILURES` and emits one structured WARNING so a
    systemic breakage (e.g. disk full → every file fails) is visible as a
    climbing counter rather than failing silently (design-invariants
    INV-2/INV-4).

    *detail* names WHAT failed. It exists so the one other loud failure this
    module reports — a credential-isolation regression, where
    :func:`_purge_config_dir` left ``.credentials.json`` behind — accrues to the
    SAME counter and the same ``path``/``task_id``/``errno`` shape rather than
    growing a second, separately-monitored signal. One counter, one greppable
    shape, one thing a consumer has to watch.
    """
    global _ARCHIVAL_FAILURES
    _ARCHIVAL_FAILURES += 1
    payload = {
        'path': str(src),
        'task_id': task_id,
        'errno': getattr(exc, 'errno', None),
    }
    logger.warning(
        'transcript_archive: %s %s: %s',
        detail,
        src,
        exc,
        extra=payload,
    )
    if _ON_ARCHIVAL_FAILURE is None:
        return
    try:
        _ON_ARCHIVAL_FAILURE(payload)
    except Exception as hook_exc:
        # A broken consumer must not become an archival outage. _record_failure
        # runs inside a function whose contract is totality — called from
        # ``finally`` blocks and teardown paths that may already be unwinding —
        # so letting the hook's exception escape would turn "the consumer has a
        # bug" into "teardown raises", strictly worse than the failure being
        # reported. Loud, not silent: its own WARNING, and the original failure
        # is already counted above where the hook cannot suppress it.
        logger.warning(
            'transcript_archive: archival-failure hook raised for %s: %s',
            src,
            hook_exc,
            extra=payload,
        )


def _discard_staging(staging: Path) -> None:
    """Remove a partially-written staging file after a failed archive copy.

    Best-effort and never raises: the caller is re-raising the ORIGINAL failure,
    which is the one an operator needs to see, so a cleanup failure must not
    mask it. It still gets its own WARNING, because leftover debris that could
    not be removed is a fact worth one line.
    """
    try:
        staging.unlink(missing_ok=True)
    except OSError as exc:  # pragma: no cover - cleanup is itself best-effort
        logger.warning(
            'transcript_archive: failed to remove staging file %s: %s',
            staging,
            exc,
        )


def _archive_one(
    src: Path,
    projects_root: Path,
    archive_root: Path,
    task_id: str,
) -> bool:
    """Copy a single transcript *src* to its mirror under *archive_root*.

    The destination mirrors *src*'s path relative to ``projects/`` with NO
    added suffix: ``<archive_root>/<task_id>/<relpath-under-projects>``.
    The source mtime is mirrored onto the copy so an already-current
    archive is skipped: if the dest exists and its (int-truncated) mtime
    equals the source's, the file is left untouched and ``False`` is
    returned. A resumed session only ever grows its transcript, so its mtime
    strictly advances and the grown file is re-archived (last-write-wins).
    Returns ``True`` when a file was newly written.

    The copy goes to a STAGING sibling and is ``os.replace``'d onto *dest*, so
    the canonical archive path only ever holds a COMPLETE copy. Writing *dest*
    directly would publish a truncated transcript whenever a copy is
    interrupted (a hard kill, or ENOSPC part-way through), and a plain
    ``.jsonl`` truncation is SILENT: every reader's ``*.jsonl`` glob picks it
    up as an ordinary transcript and ``iter_json_lines`` skips the partial
    trailing line, so the corpus under-reports that session with nothing
    counting it (design-invariants INV-4, loud over silent). For the teardown
    backstop's final archival there is no later pass to correct it either —
    the worktree is destroyed immediately after. ``os.replace`` is a
    same-directory rename, hence atomic on one filesystem, and the mtime mirror
    is applied to the staging file BEFORE the rename so the published file is
    correctly stamped from the instant it appears (a ``now``-stamped transcript
    would read to ``gc_agent_transcripts`` as a reset retention age).
    """
    rel = src.relative_to(projects_root)
    dest = archive_root / task_id / rel.parent / rel.name
    st = src.stat()
    # Idempotency: skip when the archive is already current. int-truncate to
    # dodge FS mtime-granularity mismatch between source and the utime copy.
    if dest.exists() and int(dest.stat().st_mtime) == int(st.st_mtime):
        return False
    dest.parent.mkdir(parents=True, exist_ok=True)
    staging = dest.with_name(dest.name + _STAGING_SUFFIX)
    try:
        # shutil.copyfile uses the platform fast-copy path (copy_file_range /
        # sendfile) and is exactly as memory-bounded as an explicit chunked
        # loop: agent-session JSONL can be many MB and only grows on resume, so
        # peak RSS stays flat regardless of transcript size. This is a COPY,
        # not a rename — the session may still be live and reading its own
        # source file. Any I/O error here is an OSError, caught + counted by
        # the caller (best-effort).
        shutil.copyfile(src, staging)
        os.utime(staging, (st.st_atime, st.st_mtime))
        os.replace(staging, dest)
    except OSError:
        # Only the staging file is ever touched before the replace, so a
        # pre-existing archived copy is still whole — drop the partial and let
        # the caller count + log the original failure.
        _discard_staging(staging)
        raise
    return True


@dataclass(frozen=True)
class ArchiveBeforeDelete:
    """Structured outcome of one :func:`archive_before_delete` call.

    *held* names the transcripts that could NOT be made durable and were
    therefore deliberately left on disk. It is the caller's only handle on a
    partial teardown: an empty ``held`` means every transcript is durable and
    the config dir is gone, which is the whole point of the operation.
    """

    archived: int = 0
    already_current: int = 0
    held: tuple[Path, ...] = ()
    config_dir_removed: bool = False


def _move_to_archive(
    src: Path,
    projects_root: Path,
    archive_root: Path,
    task_id: str,
) -> bool:
    """MOVE a single transcript *src* to its mirror under *archive_root*.

    The sibling of :func:`_archive_one`, and deliberately the same destination
    layout (``<archive_root>/<task_id>/<relpath-under-projects>``) so the
    already-current skip is shared by the copier and the mover — all three
    producer sites converge on one archive, not two.

    Returns ``True`` when the file was newly moved into the archive, ``False``
    when the archive was ALREADY current and the source was merely dropped.
    Either way the source is gone on return; an OSError means it is NOT.

    Why a rename rather than :func:`_archive_one`'s staged copy:

    * **No staging sibling is needed.** ``os.rename`` is itself atomic within
      one filesystem, so a truncated transcript can never appear at the
      canonical archive path — the failure mode the staged copy exists to
      prevent is structurally impossible here.
    * **The mtime mirror is free.** A rename preserves the inode, so the
      source's mtime lands on the archive without ``os.utime``. That matters
      twice over: a ``now``-stamped archive would read to
      ``gc_agent_transcripts`` as a reset retention age, and it would defeat
      the already-current skip above.
    * **It is O(1) metadata, not O(size) I/O** — which is what makes doing it
      synchronously, inside a teardown path, cheap enough to be
      unconditional.

    The rename fast path requires source and destination on ONE filesystem.
    Re-measured on this host 2026-08-18: ``<project_root>/.worktrees`` (where
    the per-task config dir lives) and ``<project_root>/data/orchestrator``
    (where the archive root lives) report the SAME ``st_dev``, so the fast
    path is the live route here. That is a property of the deployment, not a
    guarantee, and a Linux ``st_dev`` is an ephemeral mount handle rather than
    a stable id, so the cross-device case is handled rather than assumed (see
    the EXDEV branch).
    """
    rel = src.relative_to(projects_root)
    dest = archive_root / task_id / rel.parent / rel.name
    st = src.stat()
    # Idempotency, shared verbatim with _archive_one: int-truncate to dodge FS
    # mtime-granularity mismatch. The archive is already current, so the
    # precondition for deleting the source is already satisfied — corroborate,
    # then delete, rather than re-archive then delete.
    if dest.exists() and int(dest.stat().st_mtime) == int(st.st_mtime):
        src.unlink()
        return False
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.rename(src, dest)
    except OSError as exc:
        if exc.errno != errno.EXDEV:
            raise
        # EXDEV — source and destination are on different filesystems, so a
        # rename is physically impossible and no retry of it will ever work.
        # Fall back to the COPY this module already has: _archive_one stages
        # to a sibling, mirrors the mtime with os.utime (which the rename got
        # for free) and os.replace's it into place, so the canonical archive
        # path still only ever holds a complete, correctly-stamped transcript.
        # Then unlink the source, because deletion-after-archival is the whole
        # contract and a cross-device host must not silently degrade into an
        # unbounded hold.
        #
        # Unreachable on the measured host: .worktrees and data/orchestrator
        # report the same st_dev, so every archive takes the rename above.
        # Retained anyway per PRD §9 Q4 — an st_dev is an ephemeral mount
        # handle, an operator can mount either path elsewhere tomorrow, and
        # "assume same-device forever" is exactly the assumption that turns a
        # remount into silent data loss. Its only exercise is the test suite.
        _archive_one(src, projects_root, archive_root, task_id)
        src.unlink()
    return True


def _remove_member(entry: Path) -> None:
    """Delete one config-dir member. Never raises, and never FOLLOWS a symlink.

    The ``is_symlink()`` test comes first and is load-bearing. A symlink to a
    directory answers ``is_dir()`` True, and the per-task config dir contains
    exactly that shape — ``TaskConfigDir._setup_symlinks`` links
    ``settings.json`` / ``settings.local.json`` into the operator's real
    ``~/.claude``. Handing such an entry to :func:`shutil.rmtree` raises
    (rmtree refuses a symlinked root), and any variant that resolved the link
    first would delete the operator's real settings. Unlinking the LINK is the
    only correct move, and testing for it first makes following one
    structurally impossible rather than incidentally avoided.
    """
    try:
        if entry.is_symlink() or entry.is_file():
            entry.unlink(missing_ok=True)
        else:
            shutil.rmtree(entry, ignore_errors=True)
    except OSError as exc:  # pragma: no cover - the purge is itself best-effort
        logger.warning(
            'transcript_archive: failed to purge config-dir member %s: %s',
            entry,
            exc,
            extra={'path': str(entry), 'errno': getattr(exc, 'errno', None)},
        )


def _warn_on_surviving_credential(config_dir: Path, task_id: str) -> None:
    """Count + log a ``.credentials.json`` that outlived the purge.

    The purge runs on best-effort primitives (``rmtree(ignore_errors=True)``,
    a swallowing :func:`_remove_member`), so a credential CAN survive it — and
    a surviving per-task OAuth credential is a strictly worse outcome than the
    transcript loss this task set out to fix. It accrues to the existing
    archival-failure counter, so the same consumer watching for a broken
    archive root sees this too and there is no second signal to wire up.
    """
    creds = config_dir / '.credentials.json'
    if creds.exists():
        _record_failure(
            creds,
            task_id,
            RuntimeError('config-dir purge left it in place'),
            detail='CREDENTIAL ISOLATION REGRESSION — failed to purge',
        )


def _purge_config_dir(config_dir: Path, held: list[Path], task_id: str) -> bool:
    """Remove *config_dir*, or everything in it except the *held* transcripts.

    Returns ``True`` when the whole directory is gone. Never raises.

    With nothing held this is equivalent to the ``TaskConfigDir.cleanup()``
    rmtree it replaces. With something held the purge is still UNCONDITIONAL
    for every member that is neither a held transcript nor a directory on the
    path to one — the per-task OAuth ``.credentials.json``, the ``~/.claude``
    settings symlinks, ``sessions/``, ``telemetry/``, all of it.

    That asymmetry is the D1/INV-7 decision, and it is deliberate. Refusing to
    tear down at all until archival succeeds would convert a bounded transcript
    loss into an unbounded credential exposure: a permanently-failing archive
    root (full disk, wrong permissions, unmounted volume) would strand every
    task's live OAuth credential on disk indefinitely, defeating the per-task
    isolation the config dir exists to provide. So the hold is SCOPED to the
    un-archivable ``.jsonl`` alone; it is OWNED by the next process start's
    ``Harness._sweep_orphaned_transcripts``, which re-archives whatever is
    still lying in a surviving worktree; and it is therefore BOUNDED by a
    restart rather than open-ended in time.
    """
    if not held:
        shutil.rmtree(config_dir, ignore_errors=True)
        # Read the outcome back rather than assuming it: rmtree was asked to
        # ignore its errors, so "we called it" and "the directory is gone" are
        # different claims and only the second licenses the True.
        removed = not config_dir.exists()
        _warn_on_surviving_credential(config_dir, task_id)
        return removed

    projects_root = config_dir / 'projects'
    keep_files = set(held)
    keep_dirs: set[Path] = {projects_root, config_dir}
    for f in keep_files:
        keep_dirs.update(f.parents)

    # Everything outside projects/ goes, whatever happened to the transcripts.
    for entry in config_dir.iterdir():
        if entry != projects_root:
            _remove_member(entry)

    # Inside projects/, keep the held files and only the directories that lead
    # to them. Deepest-first, so any directory reached here provably contains
    # no keeper (a directory that did would be in keep_dirs) and no rmtree can
    # take a held transcript down with it.
    for entry in sorted(
        projects_root.rglob('*'), key=lambda q: len(q.parts), reverse=True
    ):
        if entry in keep_files or entry in keep_dirs:
            continue
        _remove_member(entry)

    _warn_on_surviving_credential(config_dir, task_id)
    return False


def archive_before_delete(
    config_dir: Path,
    task_id: str,
    *,
    archive_root: Path,
) -> ArchiveBeforeDelete:
    """Make *config_dir*'s transcripts durable, THEN delete the directory.

    The fused replacement for "call :func:`archive_task_transcripts`, then
    call ``TaskConfigDir.cleanup()``" at every teardown site (task 3619, leaf 2
    of plans/transcript-preservation-seam-prd.md). Those were two steps with a
    gap between them, and the gap is where transcripts were measurably lost:
    the archival step is skippable (a cancellation landing on its ``await``, a
    teardown path that never had one) while the ``rmtree`` is not, so the copy
    could be dropped and the original destroyed anyway. Here deletion is not a
    step AFTER archival, it is a step archival LICENSES: a transcript is
    unlinked only once its durable copy provably exists.

    Every ``projects/**/*.jsonl`` is moved to
    ``<archive_root>/<task_id>/<relpath-under-projects>`` — the same layout
    :func:`_archive_one` writes and :func:`durable_archive_path` reads, so
    producer, mover and reader cannot drift (INV-5). Nothing outside
    ``projects/`` is ever read, so the per-task OAuth ``.credentials.json`` at
    the config-dir root is structurally unreachable, exactly as in
    :func:`archive_task_transcripts`.

    **Synchronous, and that is the fix, not an oversight.** The caller must
    NOT wrap this in ``asyncio.to_thread`` or otherwise make it awaitable. An
    ``await`` is a cancellation point, and the SIGTERM that triggers teardown
    is precisely what delivers the cancellation — the archival step would be
    the one part of teardown that a shutdown can skip. A rename is O(1)
    metadata; even the EXDEV copy fallback is a single-digit-millisecond
    operation on the measured corpus (n=7788 archived transcripts: median
    381 KB, p95 1.0 MB, max 2.06 MB), i.e. cheaper than the transcript it
    would otherwise lose.

    **Total by contract** — never raises, mirroring
    :func:`archive_task_transcripts`, because callers invoke it from teardown
    paths and ``finally`` blocks that may already be unwinding an exception. A
    per-file failure is counted through :func:`_record_failure` and the file is
    HELD: reported in the returned :class:`ArchiveBeforeDelete`, left on disk,
    never deleted.

    A hold is deliberately NARROW and deliberately TEMPORARY. It covers the
    un-archivable ``.jsonl`` alone — every other member of the config dir,
    ``.credentials.json`` included, is purged unconditionally (see
    :func:`_purge_config_dir` for why holding those instead would be the worse
    bug). It is owned by the next process start's
    ``Harness._sweep_orphaned_transcripts``, which re-archives whatever is
    still lying in a surviving worktree, so the hold is bounded by a restart
    rather than open-ended in time.
    """
    config_dir = Path(config_dir)
    archive_root = Path(archive_root)
    projects_root = config_dir / 'projects'

    if not config_dir.exists():
        # Teardown sites call this blind (a lane that never got a config dir,
        # a second cleanup pass). A cheap no-op, not a raise — and note no
        # archive tree is conjured for a task that has no transcripts.
        return ArchiveBeforeDelete()

    archived = 0
    already_current = 0
    held: list[Path] = []
    # sorted() so the WARNING stream and the reported `held` order are
    # reproducible across runs rather than filesystem-order-dependent.
    for src in sorted(projects_root.glob('**/*.jsonl')):
        try:
            if _move_to_archive(src, projects_root, archive_root, task_id):
                archived += 1
            else:
                already_current += 1
        except OSError as exc:
            # Counted + logged loudly, and the source is HELD. Deleting an
            # un-archived transcript is the one thing this function exists to
            # prevent, so a failure costs us the directory, never the data.
            _record_failure(src, task_id, exc)
            held.append(src)

    removed = _purge_config_dir(config_dir, held, task_id)
    return ArchiveBeforeDelete(
        archived=archived,
        already_current=already_current,
        held=tuple(held),
        config_dir_removed=removed,
    )


def archive_task_transcripts(
    config_dir: Path,
    task_id: str,
    session_id: str | None = None,
    *,
    archive_root: Path,
) -> int:
    """Archive a task's agent transcripts to a durable plain-.jsonl mirror.

    When *session_id* is given, only that session's main transcript
    (``projects/*/<session_id>.jsonl``) and its subagent transcripts
    (``projects/*/<session_id>/subagents/*.jsonl``) are archived. When
    *session_id* is ``None``, every ``projects/**/*.jsonl`` is archived.
    Returns the number of files newly archived.
    """
    config_dir = Path(config_dir)
    archive_root = Path(archive_root)
    projects_root = config_dir / 'projects'

    if session_id is None:
        sources = list(projects_root.glob('**/*.jsonl'))
    else:
        sources = list(projects_root.glob(f'*/{session_id}.jsonl'))
        sources += list(projects_root.glob(f'*/{session_id}/subagents/*.jsonl'))

    count = 0
    for src in sources:
        # Best-effort: a per-file OSError is logged + counted, never re-raised,
        # so one bad file cannot lose its siblings and the whole function stays
        # total — the property the producer relies on to call it in a finally.
        try:
            if _archive_one(src, projects_root, archive_root, task_id):
                count += 1
        except OSError as exc:
            _record_failure(src, task_id, exc)
    return count


def durable_archive_path(
    archive_root: Path,
    task_id: str,
    session_id: str,
) -> Path | None:
    """Locate *session_id*'s archived main transcript under *archive_root*.

    The read side of this module. Returns the archived transcript's path, or
    ``None`` when this session has no archive. The located layout is the one
    :func:`_archive_one` writes::

        <archive_root>/<task_id>/<encoded-cwd>/<session_id>.jsonl[.gz]

    Contract (plans/session-resume-eligibility-seam-prd.md §8):

    * **I-A total** — never raises, for any input. Modelled on
      :func:`shared.cli_invoke._resolve_transcript_path`: one blanket
      ``except Exception`` → ``None``, not an enumeration of ``OSError``
      subclasses. Callers on the dispatch path (the orchestrator's
      session-resume guard and its fallback emit) lean on that totality to
      stay total themselves. A plain *miss* is silent (it is the common case);
      only a genuine fault is swallowed, and that is logged loudly.
    * **I-B cwd-agnostic** — the encoded-cwd component is *globbed*, never
      assumed, so a session archived from one worktree lane is still found
      after re-dispatch into a different lane.
    * **I-C format-agnostic** — matches both the gzipped ``.jsonl.gz`` shape
      and the plain ``.jsonl`` shape task 3618 moves to, so the cutover needs
      no flag day and this locator carries no dependency on it.
    * **I-D read-only** — glob and ``stat`` only. Nothing is created, moved,
      deleted or decompressed here; *restoration* is task 3578's, under task
      3619's archive-before-delete guard.
    * **I-E sole locator** — this is the only session-id-keyed lookup into the
      archive. Consumers call it rather than re-deriving the layout, so the
      producer and every reader cannot drift (INV-5).
    * **I-F deterministic** — when several lanes archived the same session,
      the newest-mtime match wins, with a stable tiebreak.

    Returns ``Path | None``, never ``bool`` (PRD decision D6): existence is
    spelled ``is not None``, and task 3578's restore reuses the very path
    returned here instead of growing a second glob that would have to agree
    with this one byte-for-byte forever.
    """
    try:
        # Coerce inside the try: a caller can easily hold a numeric task_id or
        # a str archive_root, and neither may raise from path composition.
        archive_root = Path(archive_root)
        task_id = str(task_id)
        session_id = str(session_id)
        # Two load-bearing details in this one line:
        #   * the trailing `*` on `.jsonl*` spans task 3618's gzip drop — it
        #     matches today's `.jsonl.gz` AND tomorrow's plain `.jsonl`, so the
        #     cutover needs no flag day and this locator no dependency on it;
        #   * `is_file()` excludes the SUBAGENT directory, which _archive_one
        #     names for the session itself (`<enc>/<sid>/subagents/agent-*.gz`).
        #     That dir carries no `.jsonl` suffix so the pattern misses it by
        #     luck today; the filter makes returning a directory as "the
        #     transcript" structurally impossible, not incidentally avoided.
        matches = [
            p for p in archive_root.glob(f'{task_id}/*/{session_id}.jsonl*') if p.is_file()
        ]
        if not matches:
            return None
        # Newest mtime wins: a resumed session's transcript only ever grows, so
        # the newest archive is the most complete one. The `str(p)` tiebreak is
        # not decoration — mtime alone is NOT a total order here, because
        # _archive_one mirrors the SOURCE mtime onto the archived copy via
        # os.utime (:124), so one session archived from two lanes can tie
        # exactly. Under a tie max() would fall back to filesystem-dependent
        # glob order; appending the path string makes the answer reproducible,
        # which is what makes a later resume reproducible.
        #
        # p.stat() can raise (a file unlinked between glob and stat — e.g. task
        # 2731's GC sweep landing mid-lookup). That is deliberately NOT caught
        # per-file: the blanket guard below is the single place totality is
        # enforced, so a miss and a race degrade to the same answer.
        return max(matches, key=lambda p: (p.stat().st_mtime, str(p)))
    except Exception as exc:
        # I-A: one blanket swallow, mirroring cli_invoke._resolve_transcript_path
        # rather than enumerating OSError subclasses — the whole point of the
        # contract is that NOTHING escapes, which is what lets the orchestrator's
        # session-resume guard and its fallback emit stay total on the dispatch
        # path.
        #
        # Level: WARNING, and note what does NOT reach here. The COMMON case —
        # this session simply has no archive (the PRD §2 reference measurement
        # puts ~36% of sessions there) — returns None from the empty-`matches`
        # branch above and logs nothing at all, so the ~36% can never become log
        # noise. Only a genuine fault lands in this handler: a glob that raised
        # (archive root unreadable/unmounted), a stat that raised (task 2731's
        # GC unlinking mid-lookup), or a non-coercible id from a caller bug.
        # Each of those is a real breakage an operator should see, so silence or
        # DEBUG here would be exactly the silent degradation design-invariants
        # INV-2/INV-4 forbid — and shared/tests/test_silent_fallthrough_gate.py
        # enforces that for new code (its baseline allowlist is a burn-down
        # ratchet for pre-existing sites like _resolve_transcript_path, not a
        # hatch new code may add to). Structured extra=, matching the shape
        # _record_failure already emits, so both archive-side failure signals
        # are greppable the same way.
        logger.warning(
            'durable_archive_path: lookup failed for session %s (task %s) under %s: %s',
            session_id,
            task_id,
            archive_root,
            exc,
            extra={
                'path': str(archive_root),
                'task_id': str(task_id),
                'errno': getattr(exc, 'errno', None),
            },
        )
        return None
