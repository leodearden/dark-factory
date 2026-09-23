"""Sweep tool: relocate resolved/dismissed escalations from queue root to archive.

Relocates resolved/dismissed escalations from queue root to archive.
Run at the escalation server-start single-writer window (pre-serving) —
the one window with no concurrent queue writers.

Usage:
  python -m escalation.sweep --queue-dir <path>          # dry-run (default)
  python -m escalation.sweep --queue-dir <path> --apply  # actually moves files
"""

from __future__ import annotations

import argparse
import contextlib
import errno
import json
import logging
import os
import shutil
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from escalation import archive
from escalation.models import Escalation
from escalation.queue import SEQ_COUNTER_SUFFIX, escalation_id_lock, read_escalation_for_scan

logger = logging.getLogger(__name__)

#: Parse-failure exceptions caught by ``sweep.py``'s two scan sites, passed to
#: ``read_escalation_for_scan``.  Deliberately WIDER than queue.py's
#: ``_SCAN_PARSE_ERRORS``: the bare ``ValueError`` is what makes a sweep
#: tolerate a validation failure from ``Escalation.from_json`` — and, since
#: ``UnicodeDecodeError`` subclasses ``ValueError``, an undecodable file too —
#: by skipping that one record instead of aborting the whole pass.  Named once
#: rather than spelled out at each call site: the two modules' tuples must stay
#: independently controllable, which is the entire reason ``parse_errors`` is a
#: parameter, and a literal repeated at both sites invites a future widening
#: applied to one and not the other — splitting sweep()'s behaviour from
#: reap_loose_archive_files()' silently, the exact drift the parameter exists
#: to prevent.
_SWEEP_PARSE_ERRORS: tuple[type[BaseException], ...] = (
    json.JSONDecodeError, KeyError, TypeError, ValueError,
)


@dataclass
class SweepReport:
    archived: int = 0
    reconciled_root_wins: int = 0
    reconciled_archive_wins: int = 0
    untouched_pending: int = 0
    # Counts files left in root for any skip reason: unparsable JSON (which
    # includes bytes that would not DECODE as UTF-8), an UNREADABLE file (any
    # other OSError -- EACCES, EIO, fd exhaustion), missing resolved_at on a
    # resolved/dismissed file, or unknown status value.  Individual cases are
    # logged at WARNING level for operator review.
    #
    # The I/O fault is folded in here DELIBERATELY, despite the name: this
    # counter's meaning is "left IN root for an operator-actionable reason",
    # and a file that is present but unreadable is exactly that.  Its own
    # WARNING says 'failed to read', not 'Failed to parse', so the log
    # distinguishes the two even though the tally does not — do not read a
    # nonzero count as proof of corrupt JSON.
    skipped_unparsable: int = 0
    # Counts records relocated OUT of the root by a CONCURRENT actor before
    # this pass could act on them — a resolve() in another process, or a second
    # startup sweep during a fleet redeploy, when several orchestrators
    # restart together.  Covers BOTH halves of the exposure: between the glob
    # and the read, and between the read and this pass's own move/unlink (the
    # record is read outside the per-id lock, so taking that lock later does
    # not make the parsed record fresh).  Deliberately NOT skipped_unparsable:
    # that counter
    # means "left IN root for an operator-actionable reason", and a vanished
    # file is neither.  Logged at DEBUG, not WARNING — this is expected
    # concurrency, not corruption.
    skipped_vanished: int = 0
    root_before: int = 0
    root_after: int = 0


def _pick_richer(root_esc: Escalation, archive_esc: Escalation) -> bool:
    """Return True iff root is strictly richer than archive; ties go to archive.

    Richness is compared lexicographically by:
      1. resolved_by present
      2. resolution_turns present
      3. len(resolution text)  — guards against silent loss of free-form text
      4. dedupe_count          — guards against losing folded-duplicate metadata
    """
    def _key(e: Escalation) -> tuple:
        return (
            e.resolved_by is not None,
            e.resolution_turns is not None,
            len(e.resolution or ''),
            e.dedupe_count,
        )
    return _key(root_esc) > _key(archive_esc)


def _build_archive_index(archive_root: Path) -> dict[str, Path]:
    """Return {stem: path} index of all esc-*.json files under archive_root.

    If the same escalation id appears in multiple dated subdirs (e.g. from a
    prior partial run or a bad import), the last path discovered wins and a
    WARNING is logged.  The operator should investigate and deduplicate manually
    before running --apply; the sweep will still proceed with the indexed copy.
    """
    if not archive_root.exists():
        return {}
    index: dict[str, Path] = {}
    for p in archive_root.rglob('esc-*.json'):
        if p.stem in index:
            logger.warning(
                'archive duplicate: %s exists at %s AND %s; using latter for reconciliation',
                p.stem, index[p.stem], p,
            )
        index[p.stem] = p
    return index


def _archive_stems(archive_root: Path) -> set[str]:
    """Return the set of escalation ids that have a record anywhere under archive_root.

    A stems-only sibling of :func:`_build_archive_index` for callers that need
    mere PRESENCE rather than paths.  Two things it deliberately does not do:

    * allocate a ``Path`` value per archived record, and
    * re-emit ``_build_archive_index``'s duplicate-id WARNING — ``sweep()`` has
      already logged it once for the operator earlier in the same startup, and a
      second copy per multi-dated duplicate is pure noise for a caller that does
      not care which path won.
    """
    if not archive_root.exists():
        return set()
    return {p.stem for p in archive_root.rglob('esc-*.json')}


def _atomic_move(src: Path, dst: Path) -> None:
    """Move src to dst atomically where possible, with a cross-device fallback.

    os.replace is atomic on the same filesystem but raises OSError(EXDEV) when
    src and dst are on different devices (e.g. archive is a bind mount).  In
    that case we fall back to shutil.move (copy + unlink) and log a warning.
    """
    try:
        os.replace(src, dst)
    except OSError as exc:
        if exc.errno == errno.EXDEV:
            logger.warning(
                'cross-device move detected; falling back to shutil.move for %s → %s', src, dst
            )
            shutil.move(str(src), str(dst))
        else:
            raise


def _source_vanished(path: Path) -> bool:
    """True iff an ENOENT just raised by a move/unlink was about *path* itself.

    Guarding the unlocked READ closes only half of sweep()'s exposure: the
    record is read outside any lock and moved a few lines later, so a
    concurrent ``resolve()`` — which takes the SAME per-id sidecar lock around
    ``_archive_resolved``, and therefore need only complete and release before
    this pass acquires — can leave the source gone by the time
    ``os.replace``/``os.unlink`` runs.  ``_atomic_move`` re-raises every
    ``OSError`` whose errno is not EXDEV, ENOENT included, so that would abort
    the whole pass exactly as the read fault used to.

    The discrimination matters because ``os.replace`` raises the SAME
    ``FileNotFoundError`` for a missing DESTINATION parent (a dated subdir
    pruned between our ``mkdir`` and our ``replace``).  That is a different
    fault — the target tree is disappearing under us, not a routine record
    relocation — so callers re-raise it rather than counting it as a benign
    race.  Checked by probing the source, which is the only thing that
    distinguishes the two after the fact.
    """
    return not path.exists()


def _relocate_terminal(
    queue_dir: Path,
    path: Path,
    esc: Escalation,
    *,
    apply: bool = True,
) -> str:
    """Relocate a terminal esc file to its dated archive subdir.

    Shared by ``sweep()``'s archive-missing branch and ``reap_loose_archive_files()``
    to avoid the two copies drifting independently.

    Performs target path computation, collision guard (checked **inside** the
    per-id lock to close the TOCTOU window), and the locked atomic move.

    Pre-conditions (verified by caller):
        - ``esc.status`` is ``'resolved'`` or ``'dismissed'``
        - ``esc.resolved_at`` is not ``None``

    Args:
        queue_dir: Root queue directory (parent of ``archive/``).
        path: Source file path to relocate.
        esc: Parsed escalation record.
        apply: If True, actually move the file; if False, only check (dry-run).

    Returns:
        ``'moved'``     -- the file was (or would be) relocated.
        ``'collision'`` -- skipped; the existing target is left untouched.
        ``'vanished'``  -- a concurrent actor relocated the SOURCE between this
        pass's unlocked read and the move.  A tag rather than a bool for the
        same reason ``read_escalation_for_scan`` returns one: the caller must
        be able to count a benign race into a different report field than work
        it actually performed.  See :func:`_source_vanished`.
    """
    assert esc.resolved_at is not None  # pre-condition: caller ensures terminal status
    target_dir = archive.archive_dir_for_date(queue_dir, esc.resolved_at)
    target_path = target_dir / path.name

    if apply:
        with escalation_id_lock(queue_dir, path.stem):
            # Re-check inside the lock to close the TOCTOU window: a concurrent
            # writer could have created target_path between our pre-check and
            # acquiring the lock.
            if target_path.exists():
                logger.warning(
                    'skipping %s: target already exists at %s (not overwriting)',
                    path.name, target_path,
                )
                return 'collision'
            target_dir.mkdir(parents=True, exist_ok=True)
            try:
                _atomic_move(path, target_path)
            except FileNotFoundError:
                if not _source_vanished(path):
                    raise
                logger.debug(
                    '%s vanished before its move (relocated concurrently); skipping',
                    path.name,
                )
                return 'vanished'
    else:
        # Dry-run: check for collisions without acquiring the lock.
        if target_path.exists():
            return 'collision'

    return 'moved'


def sweep(queue_dir: Path, *, apply: bool = False) -> SweepReport:
    """Sweep resolved/dismissed escalations from queue root to archive.

    Args:
        queue_dir: Root queue directory (parent of archive/).
        apply: If False (default), classify and report without mutating disk.
               If True, actually move/remove files.

    Returns:
        SweepReport with counts for each action category.
    """
    queue_dir = Path(queue_dir)
    report = SweepReport()

    root_files = list(queue_dir.glob('esc-*.json'))
    report.root_before = len(root_files)

    archive_root = queue_dir / archive.ARCHIVE_SUBDIR
    archive_index = _build_archive_index(archive_root)

    for path in root_files:
        # root_files was materialized above; nothing holds a lock over these
        # reads (the per-id lock is taken later, at the move), so a concurrent
        # actor can relocate any of them in this window.  The helper logs each
        # outcome on its own channel — hence no local warning here.
        # parse_errors preserves sweep's own (wider) tuple exactly: it catches
        # bare ValueError where queue.py deliberately does not.
        esc, reason = read_escalation_for_scan(
            path, context='sweep', parse_errors=_SWEEP_PARSE_ERRORS,
        )
        if reason == 'vanished':
            report.skipped_vanished += 1
            continue
        if reason != 'ok' or esc is None:
            # 'unreadable' is routed here deliberately: before this guard a
            # PermissionError aborted the ENTIRE sweep, so counting it and
            # staying loud (the helper already warned) is strictly better.
            # Pinned by TestSweepSafety::
            # test_unreadable_file_is_counted_loud_and_does_not_abort_the_pass.
            report.skipped_unparsable += 1
            continue

        if esc.status == 'pending':
            report.untouched_pending += 1
            continue

        if esc.status in ('resolved', 'dismissed'):
            if not esc.resolved_at:
                logger.warning(
                    'skipping %s: status %s but resolved_at is missing', path.name, esc.status
                )
                report.skipped_unparsable += 1
                continue
            existing_archive = archive_index.get(path.stem)
            archive_esc: Escalation | None = None

            if existing_archive is not None:
                # The ARCHIVE copy is read just as unlocked as the root one:
                # archive_index is an rglob SNAPSHOT taken before this loop, so
                # a concurrent archive.prune_archive — run_startup_sweep's own
                # pass 3, hence a second orchestrator's startup sweep during a
                # fleet redeploy — can rmtree the dated subdir holding it in
                # between.  Its own context tag so a warning names this read
                # rather than the root one.
                archive_esc, archive_reason = read_escalation_for_scan(
                    existing_archive, context='sweep.archive_copy',
                    parse_errors=_SWEEP_PARSE_ERRORS,
                )
                if archive_reason == 'vanished':
                    # No archive copy after all — treat exactly as absent and
                    # fall through to the relocation below.  Not counted in
                    # skipped_vanished: that counter tracks ROOT files leaving
                    # the root, and this root file is about to be moved by us.
                    existing_archive = None
                elif archive_reason != 'ok' or archive_esc is None:
                    # Unparsable or unreadable archive copy: richness cannot be
                    # compared, and overwriting it blind could destroy the only
                    # good record.  Leave BOTH copies on disk for the operator
                    # — which is precisely what skipped_unparsable counts.
                    report.skipped_unparsable += 1
                    continue

            if existing_archive is None or archive_esc is None:
                # No archive copy — or it vanished under us just above, which
                # amounts to the same thing: move to the dated subdir via the
                # shared helper.
                # The helper re-checks for collisions inside the per-id lock so
                # a file that appeared between index-build and lock-acquire is
                # detected atomically (TOCTOU safe).
                outcome = _relocate_terminal(queue_dir, path, esc, apply=apply)
                if outcome == 'moved':
                    report.archived += 1
                elif outcome == 'vanished':
                    report.skipped_vanished += 1
            else:
                # Archive copy exists: compare richness
                if _pick_richer(esc, archive_esc):
                    # Root wins: atomically overwrite the archive slot
                    if apply:
                        with escalation_id_lock(queue_dir, path.stem):
                            try:
                                _atomic_move(path, existing_archive)
                            except FileNotFoundError:
                                # Same move-window race as _relocate_terminal's;
                                # see _source_vanished for why the destination
                                # case is re-raised instead.
                                if not _source_vanished(path):
                                    raise
                                logger.debug(
                                    '%s vanished before its reconciling move '
                                    '(relocated concurrently); skipping',
                                    path.name,
                                )
                                report.skipped_vanished += 1
                                continue
                    report.reconciled_root_wins += 1
                else:
                    # Archive wins: drop the duplicate root copy
                    if apply:
                        with escalation_id_lock(queue_dir, path.stem):
                            try:
                                os.unlink(path)
                            except FileNotFoundError:
                                # No destination to be ambiguous about here:
                                # an unlink ENOENT means the root copy is
                                # already gone, so this pass performed no
                                # reconciliation and must not claim one.
                                logger.debug(
                                    '%s vanished before its duplicate-drop '
                                    '(relocated concurrently); skipping',
                                    path.name,
                                )
                                report.skipped_vanished += 1
                                continue
                    report.reconciled_archive_wins += 1
            continue

        # Defensive: unknown status (should not occur given models.py constraints)
        logger.warning('skipping %s: unknown status %r', path.name, esc.status)
        report.skipped_unparsable += 1

    # root_before is the glob count, so a vanished file WAS counted there but
    # belongs to none of the other subtracted categories — omitting its term
    # would leave this operator-facing figure over-reporting by that many.
    report.root_after = (
        report.root_before
        - report.archived
        - report.reconciled_root_wins
        - report.reconciled_archive_wins
        - report.skipped_vanished
    )
    return report


@dataclass
class StartupSweepReport:
    """Aggregated report from a single run_startup_sweep call."""
    sweep: SweepReport
    loose_reaped: int = 0
    pruned_dirs: int = 0
    # Sidecar locks unlinked by pass 4.  Under apply=False this is a LOWER
    # BOUND rather than a projection; see run_startup_sweep's ``apply`` arg.
    orphan_locks_reaped: int = 0


def run_startup_sweep(
    queue_dir: Path,
    *,
    retention_days: int = archive.DEFAULT_RETENTION_DAYS,
    apply: bool = True,
    now: datetime | None = None,
) -> StartupSweepReport:
    """Orchestrate sweep + loose-reap + prune + lock-reap at escalation server start.

    Runs four passes in sequence:
      1. sweep(queue_dir, apply=apply)       — root→archive relocation
      2. reap_loose_archive_files(…)         — archive top-level→dated subdir
      3. archive.prune_archive(…)            — drop subdirs beyond retention
      4. reap_orphan_locks(…)                — unlink sidecar locks whose record
                                               is in neither tier

    Pass 4 runs LAST on purpose: when APPLYING, records dropped by retention in
    pass 3 are dead ids by the time it looks, so their sidecars are reaped in the
    SAME run rather than lingering until the next restart.  That coupling is what
    makes ``apply=False`` under-report — see the ``apply`` arg.

    Logs one INFO summary line on the ``escalation.sweep`` logger.

    Args:
        queue_dir: Root queue directory.
        retention_days: Retention threshold forwarded to prune_archive.
        apply: If False, dry-run all four passes (no disk mutations).
            CAVEAT — in dry-run ``orphan_locks_reaped`` is a LOWER BOUND, not a
            projection.  Pass 3 is skipped entirely (``archive.prune_archive``
            has no dry-run mode to forward ``apply`` to, and giving it one is out
            of this pass's scope), so the archive records an applying run would
            have evicted are still on disk when pass 4 takes its snapshot, and
            their sidecars are counted as KEEPS.  Passes 1 and 2 do not skew the
            count — they only move a record between two tiers pass 4 treats
            alike.  So an applying run reaps at least what the preceding dry-run
            promised and possibly more; it never reaps fewer.
        now: Reference datetime for prune_archive cutoff (defaults to live UTC).

    Returns:
        StartupSweepReport with nested SweepReport plus loose_reaped, pruned_dirs
        and orphan_locks_reaped.
    """
    queue_dir = Path(queue_dir)
    sweep_report = sweep(queue_dir, apply=apply)
    loose_reaped = reap_loose_archive_files(queue_dir, apply=apply)
    pruned_dirs = archive.prune_archive(queue_dir, retention_days, now=now) if apply else 0
    # AFTER prune: a record evicted by retention just above is a dead id now, so
    # its sidecar goes in this run.  It also means the archive index this pass
    # builds is post-prune and cannot mistake an evicted record for a live one.
    orphan_locks_reaped = reap_orphan_locks(queue_dir, apply=apply)

    report = StartupSweepReport(
        sweep=sweep_report,
        loose_reaped=loose_reaped,
        pruned_dirs=pruned_dirs,
        orphan_locks_reaped=orphan_locks_reaped,
    )
    logger.info(
        'Startup sweep %s: archived=%d reconciled(root=%d archive=%d) '
        'loose_reaped=%d pruned_dirs=%d orphan_locks=%d pending=%d skipped=%d '
        'vanished=%d; root: %d → %d',
        'APPLIED' if apply else 'DRY-RUN',
        sweep_report.archived,
        sweep_report.reconciled_root_wins,
        sweep_report.reconciled_archive_wins,
        loose_reaped,
        pruned_dirs,
        orphan_locks_reaped,
        sweep_report.untouched_pending,
        sweep_report.skipped_unparsable,
        sweep_report.skipped_vanished,
        sweep_report.root_before,
        sweep_report.root_after,
    )
    return report


def reap_loose_archive_files(queue_dir: Path, *, apply: bool = True) -> int:
    """Relocate loose archive-top-level esc-*.json files into dated subdirs.

    A "loose" file is one that sits directly under ``archive/`` (not inside
    a ``YYYY-MM-DD`` dated subdir).  These accumulate when the queue's
    ``_archive_resolved`` fast-path is bypassed or when a prior sweep run
    wrote files without the dated-subdir structure.

    Each relocation is serialized per-id by ``escalation_id_lock`` so it
    does not race concurrent queue writers that hold the same id lock.

    Args:
        queue_dir: Root queue directory (parent of ``archive/``).
        apply: If True (default), actually move files.  If False, only count
               how many would be moved (dry-run).

    Returns:
        Number of files moved (or would-move when apply=False).

    Skips (with WARNING):
        - Files whose JSON cannot be parsed
        - Files whose status is not resolved/dismissed
        - Files missing ``resolved_at``
        - Files where a same-name file already exists in the target dated dir
          (never overwrite / lose data)
    """
    queue_dir = Path(queue_dir)
    archive_root = queue_dir / archive.ARCHIVE_SUBDIR
    if not archive_root.exists():
        return 0

    moved = 0
    # Materialize the glob before iterating so the directory scan completes
    # before any moves occur — mutating a directory during os.scandir/readdir
    # has unspecified behaviour on some filesystems and can cause subsequent
    # sibling entries to be skipped (mirrors sweep()'s `root_files = list(...)` pattern).
    # Only glob the archive top level — non-recursive, so dated-subdir files
    # are excluded (the glob `esc-*.json` does not recurse into subdirs).
    for path in list(archive_root.glob('esc-*.json')):
        # Same unlocked glob-then-read exposure as sweep(): a concurrent sweep
        # can file this loose record into its dated subdir before the read.
        # No counter is needed here, unlike sweep() — `moved` tallies the
        # relocations THIS pass performed, and one already done by someone else
        # simply is not among them, so the tally stays truthful as-is.
        esc, reason = read_escalation_for_scan(
            path, context='reap_loose', parse_errors=_SWEEP_PARSE_ERRORS,
        )
        if reason != 'ok' or esc is None:
            continue

        if esc.status not in ('resolved', 'dismissed'):
            logger.warning(
                'reap_loose: skipping %s: non-terminal status %r', path.name, esc.status
            )
            continue

        if not esc.resolved_at:
            logger.warning(
                'reap_loose: skipping %s: resolved_at is missing', path.name
            )
            continue

        # _relocate_terminal handles the collision guard inside the per-id lock
        # (closing the TOCTOU window) and performs the atomic move.  Only
        # 'moved' counts: a 'vanished' outcome means a concurrent sweep already
        # filed this record, which is not a relocation THIS pass performed.
        if _relocate_terminal(queue_dir, path, esc, apply=apply) == 'moved':
            moved += 1

    return moved


def reap_orphan_locks(queue_dir: Path, *, apply: bool = True) -> int:
    """Unlink queue-root sidecar locks whose escalation record no longer exists.

    ``escalation_id_lock`` creates a stable ``{escalation_id}.json.lock`` sidecar
    on first use and never renames it (task 1609's stable-inode contract).  The
    sidecar therefore outlives its record: once a record is pruned by archive
    retention — or was never submitted after its id was minted — the lock file
    is left behind forever and the queue root accumulates them.

    Sidecars for ``esc-{task_id}.seq`` counters are never candidates at all; see
    the ``SEQ_COUNTER_SUFFIX`` guard on the candidate list.

    A sidecar is an ORPHAN when its record is absent from BOTH tiers — the queue
    root AND the archive — in which case no writer can ever take that lock again
    and unlinking it is safe.  An archived record still counts as live: it is
    readable via ``get()``/``get_by_task`` and its resolve/dismiss paths still
    take the sidecar lock.  Every unlink is serialized on the very lock being
    removed, so a concurrent queue writer that already holds it is never cut in
    on.

    The archive index is a SNAPSHOT taken when this pass runs — which, inside
    ``run_startup_sweep``, is after ``archive.prune_archive``, so records dropped
    by retention in the same run have their sidecars reaped in that run.  The
    residual: a resolve that archives a record between the snapshot and the
    unlink could have its (now live) sidecar reaped.  The cost is bounded to a
    re-created sidecar — the next ``escalation_id_lock`` O_CREATs it — and a
    momentary exclusion gap for that one id.  Closing it would cost one archive
    rglob per candidate (thousands of candidates x thousands of archived files
    on the first run), which is not worth paying for a window this narrow.

    Deliberately silent per KEPT sidecar: the common case is thousands of
    legitimate keeps, and the aggregate is already on the startup INFO line.

    Args:
        queue_dir: Root queue directory (the only place sidecars are created).
        apply: If True (default), actually unlink.  If False, only count how many
               would be unlinked (dry-run — mutates nothing).

    Returns:
        Number of sidecars unlinked (or would-unlink when apply=False).
    """
    queue_dir = Path(queue_dir)

    # The `esc-` prefix is an explicit ALLOW-LIST, not a convenience: it is the
    # D6 hard invariant (pinned by TestD6GlobInvariant) applied to this pass.
    # Only escalation record ids and `esc-{task_id}.seq` counter ids ever take
    # escalation_id_lock, so nothing outside the esc- namespace can be a sidecar
    # this pass owns — any other `.json.lock` in the root belongs to a different
    # subsystem and is none of our business, record or no record.
    #
    # Materialize the glob before iterating so the directory scan completes
    # before any unlink occurs — mutating a directory during os.scandir/readdir
    # has unspecified behaviour on some filesystems and can cause subsequent
    # sibling entries to be skipped (mirrors reap_loose_archive_files()).
    #
    # A make_id sequence counter has NO .json record by construction, so it
    # would look orphaned forever; filtering it out HERE (rather than inside the
    # loop) also keeps it out of the emptiness check below.  Deleting a counter
    # or its sidecar forces make_id down _recover_seq_from_disk — a
    # correctness-relevant repair path, not merely a slow one: it derives from
    # SUBMITTED records only and so can rewind the counter below an id already
    # minted but not yet filed.
    candidates = [
        path for path in queue_dir.glob('esc-*.json.lock')
        # Sliced, not Path.stem: `.stem` strips only `.lock`, leaving `esc-1-1.json`.
        if not path.name[: -len('.json.lock')].endswith(SEQ_COUNTER_SUFFIX)
    ]
    if not candidates:
        # Steady state once the backlog is drained: nothing to judge, so skip the
        # archive walk entirely.  It is the most expensive thing this pass does
        # (a full rglob over the largest tree in the queue dir) and it is the
        # SECOND such walk in a run_startup_sweep, after sweep()'s own.
        return 0

    # One pass over the archive tier for the whole run.  Presence is all this
    # pass needs, so it uses the stems-only helper rather than
    # _build_archive_index, which would allocate a discarded Path per archived
    # record and re-log every duplicate-id warning sweep() already emitted.
    archived_stems = _archive_stems(queue_dir / archive.ARCHIVE_SUBDIR)

    reaped = 0
    for path in candidates:
        stem = path.name[: -len('.json.lock')]
        record_path = queue_dir / f'{stem}.json'

        if stem in archived_stems:
            continue

        if record_path.exists():
            continue

        if not apply:
            # Dry-run: report the would-reap count without touching disk, the
            # same apply=False contract sweep() and reap_loose_archive_files()
            # honour.
            reaped += 1
            continue

        # The OSError guard wraps the WHOLE critical section, not just the
        # unlink: on a sidecar the reaper cannot open (root-owned, EIO, a full
        # inode table) the raise comes out of escalation_id_lock's own
        # os.open(O_CREAT|O_RDWR), never out of os.unlink — which needs write
        # permission on the DIRECTORY and so would fail on every candidate or
        # none.  Letting either escape would abort the pass for the remaining
        # thousands AND take run_startup_sweep down before it builds its report,
        # costing the operator the INFO summary of passes 1-3 that already
        # succeeded (server.py wraps the call in a non-fatal try/except, so all
        # they would see is a generic WARNING).
        try:
            # Unlink INSIDE the critical section so the flock is released on the
            # already-unlinked inode and the context manager never re-creates the
            # file on its way out.
            with escalation_id_lock(queue_dir, stem):
                # Re-check inside the lock to close the TOCTOU window: a
                # concurrent writer could have created the record between our
                # pre-check and acquiring the lock — mid-submit, holding this
                # very sidecar first.
                if record_path.exists():
                    continue
                # Another reaper or process removing it first IS the intended
                # end state, so suppress and fall through to COUNT it rather
                # than pretend the sidecar is still there.  Suppression is
                # scoped to the unlink on purpose: a FileNotFoundError out of
                # the lock acquisition instead means the queue dir itself went
                # away — a failure to report, not a reap to count.
                with contextlib.suppress(FileNotFoundError):
                    os.unlink(path)
        except OSError as e:
            # Not counted: nothing was removed.
            logger.warning('reap_orphan_locks: could not reap %s: %s', path.name, e)
            continue
        reaped += 1

    return reaped


def main(argv: list[str] | None = None) -> int:
    """CLI entry point: python -m escalation.sweep."""
    parser = argparse.ArgumentParser(
        description='Sweep resolved/dismissed escalations from queue root to archive.',
    )
    parser.add_argument(
        '--queue-dir',
        required=True,
        type=Path,
        help='Path to the escalation queue directory (parent of archive/).',
    )
    parser.add_argument(
        '--apply',
        action='store_true',
        default=False,
        help='Actually move files (default: dry-run only).',
    )
    args = parser.parse_args(argv)

    if not args.queue_dir.exists():
        logger.error('queue-dir does not exist: %s', args.queue_dir)
        return 2

    report = sweep(args.queue_dir, apply=args.apply)
    logger.info(
        'Sweep %s: archived=%d reconciled(root_wins=%d, archive_wins=%d) pending=%d '
        'skipped=%d vanished=%d; root: %d → %d',
        'APPLIED' if args.apply else 'DRY-RUN',
        report.archived,
        report.reconciled_root_wins,
        report.reconciled_archive_wins,
        report.untouched_pending,
        report.skipped_unparsable,
        report.skipped_vanished,
        report.root_before,
        report.root_after,
    )
    return 0


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    sys.exit(main())
