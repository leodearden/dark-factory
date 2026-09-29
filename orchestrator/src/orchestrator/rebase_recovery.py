"""Guarded recovery from a wedged rebase or merge (task 4797).

``git rebase --abort`` is the standard way out of a conflicted rebase, and on
this host it has two measured failure modes that leave an operator — or an
agent following a skill — with no way forward:

* **Dangling rerere ref.**  A ``MERGE_RR`` record naming a conflict id whose
  ``rr-cache/<id>`` directory is absent makes the abort die inside
  ``rerere_clear()``: measured ``Segmentation fault (core dumped)``, rc 139,
  with the rebase state fully intact and a fresh ``MERGE_RR.lock`` created.
* **Stale lock.**  A leftover ``MERGE_RR.lock`` whose creating process is long
  gone makes the abort fail rc 128 with git's "Another git process seems to be
  running" advice, which names no remedy the caller can act on.  This is an
  independent failure: it reproduces with a perfectly intact rr-cache.  The
  sweep's scope is deliberately wider than the file that motivated it — see
  :func:`sweep_stale_locks`.

This module is the preflight that makes the abort safe, plus the command
prefix the abort itself must carry.  Both measures are applied, and they do
DIFFERENT jobs — neither is redundant dead weight:

* ``RECOVERY_GIT`` disables rerere for the abort, so git never opens MERGE_RR
  at all.  That neutralises the crash and the stale-lock rc 128 in one stroke.
* The quarantine preserves MERGE_RR, which a SUCCESSFUL abort would otherwise
  delete (measured), and emits the WARNING that makes the repair observable.

Each is independently sufficient to make the abort return rc 0.  They are kept
together because the first stops the failure and the second keeps the
evidence; deleting either on the theory that the other covers it loses a
capability the remaining half never had.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import re
import subprocess
import sys
from collections.abc import Awaitable, Callable, Iterable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

logger = logging.getLogger(__name__)

#: Command prefix for every abort issued on a recovery path.  Disabling rerere
#: means git never opens MERGE_RR, which is what makes the abort survive both
#: a dangling rr-cache ref and a stale ``MERGE_RR.lock``.
RECOVERY_GIT = ('git', '-c', 'rerere.enabled=false')

#: Wall-clock bound on the one subprocess this module spawns.  The probe runs
#: on the critical path of every abort in the merge lane, against worktrees
#: that are damaged by construction — so an unbounded ``rev-parse`` would wedge
#: the lane the abort exists to unwedge, which is this module's own anti-goal.
#: Ten seconds is ~160x the slowest healthy answer measured here (2.4-61.4ms)
#: and still bounded; a timeout is answered exactly like any other failure to
#: resolve, so the bound costs no branch of its own.
_PROBE_TIMEOUT_SECONDS = 10.0

#: One MERGE_RR record: a conflict id and the path it belongs to.  The id is
#: hex with an OPTIONAL rerere variant suffix ``.<N>``; the suffix is part of
#: the rr-cache directory name, so the group must capture it.
#:
#: 40 hex OR 64, and nothing between or beyond.  rerere hashes with the
#: REPOSITORY's algorithm, so a ``--object-format=sha256`` repository writes
#: 64 — measured on git 2.43.0, with a matching ``rr-cache/<64-hex>/``.  The
#: orchestrator is documented as operating projects other than this one, so
#: SHA-1 is not a safe assumption; against a 40-only grammar every record of
#: such a repository is ``unparsable``, which makes the scan suspect and
#: quarantines a HEALTHY MERGE_RR on every guarded abort.  The two widths are
#: spelled as a pair rather than as ``{40,64}`` because the in-between lengths
#: no git can produce are exactly the truncation the corruption arm must keep
#: catching.
_RECORD = re.compile(
    rb'^([0-9a-f]{40}(?:[0-9a-f]{24})?(?:\.\d+)?)\t(.+)$', re.DOTALL,
)


@dataclass(frozen=True)
class MergeRrRecord:
    """A parsed MERGE_RR record.

    ``conflict_id`` is the LITERAL token from the file, variant suffix and
    all.  It is used verbatim as the ``rr-cache`` directory name — never
    stripped, split on ``.``, or otherwise normalized, because the bare and
    suffixed forms name DIFFERENT directories and only one of them exists.
    """

    conflict_id: str
    path: str


@dataclass(frozen=True)
class ParsedMergeRr:
    """Everything a MERGE_RR file yielded, including what it failed to yield."""

    records: tuple[MergeRrRecord, ...]
    unparsable: tuple[bytes, ...]


def parse_merge_rr(data: bytes) -> ParsedMergeRr:
    """Parse MERGE_RR bytes into records and un-parsable leftovers.

    Pure: bytes in, values out — no filesystem, no subprocess — so both the
    scan and the CLI can reuse it and neither needs a repository to test it.

    The grammar is NUL-TERMINATED records, not lines: git writes
    ``<id>\\t<path>\\0`` per conflict, so a path may legally contain anything
    except NUL, newlines included.  A record that does not match is returned in
    ``unparsable`` rather than raised — git ``die()``s with ``corrupt
    MERGE_RR`` on the same input, making it a second way recovery fails hard,
    and this module exists to keep a recovery path from failing hard.
    """
    records: list[MergeRrRecord] = []
    unparsable: list[bytes] = []
    for raw in data.split(b'\x00'):
        if not raw:
            continue
        match = _RECORD.match(raw)
        if match is None:
            unparsable.append(raw)
            continue
        conflict_id, path = match.groups()
        records.append(
            MergeRrRecord(
                conflict_id=conflict_id.decode(),
                path=path.decode(errors='surrogateescape'),
            ),
        )
    return ParsedMergeRr(records=tuple(records), unparsable=tuple(unparsable))


@dataclass(frozen=True)
class MergeRrScan:
    """What one worktree's MERGE_RR says, and which of it resolves.

    ``suspect`` is the single question a caller acts on: it is true when the
    file names a conflict id with no backing rr-cache directory, when it holds
    a record git itself would reject, or when it could not be READ at all.  The
    three have different symptoms and the same remedy, so they share one flag.

    ``unreadable`` is carried as a flag rather than folded into an empty parse
    because absent and unreadable are opposite answers: an absent MERGE_RR is
    the normal healthy state, while an unreadable one is a file this run never
    managed to inspect — unknown, and unknown is not healthy.  A rename needs
    write permission on the DIRECTORY rather than read on the file, so an
    unreadable MERGE_RR can still be moved aside and its evidence kept.

    ``common_dir`` is carried because it is where a quarantined MERGE_RR goes,
    and it is the one directory in play that outlives the worktree -- see
    :func:`_quarantine_dir`.
    """

    merge_rr_path: Path
    common_dir: Path
    records: tuple[MergeRrRecord, ...]
    dangling: tuple[MergeRrRecord, ...]
    unparsable: tuple[bytes, ...]
    unreadable: bool = False

    @property
    def suspect(self) -> bool:
        return bool(self.dangling or self.unparsable or self.unreadable)


def scan_merge_rr(*, git_dir: Path, common_dir: Path) -> MergeRrScan:
    """Classify a worktree's MERGE_RR records against the shared rr-cache.

    *git_dir* is the PER-WORKTREE git directory, where MERGE_RR lives.
    *common_dir* is the shared one, where ``rr-cache`` lives — in a linked
    worktree these differ, and resolving rr-cache under the per-worktree dir
    would find nothing for every record, reporting healthy state as dangling.

    The conflict id indexes ``rr-cache`` VERBATIM, variant suffix included:
    ``<hex>`` and ``<hex>.1`` name different directories, and only one of them
    exists in the case this guards.  An entry must be a directory, since that
    is what git stores the preimage inside.

    A missing MERGE_RR is the normal healthy state — most worktrees have none —
    and yields an empty scan rather than an error.  Any OTHER read failure is
    recorded as ``unreadable`` instead: it must not raise, because this runs on
    a recovery path, and it must not be folded into the healthy branch either,
    because that would report a file nothing inspected as clean.  Both
    arguments stay explicit so the classifier is pure filesystem work, testable
    without a repository; :func:`resolve_git_dirs` supplies them for real
    callers.
    """
    merge_rr_path = git_dir / 'MERGE_RR'
    unreadable = False
    try:
        data = merge_rr_path.read_bytes()
    except FileNotFoundError:
        data = b''
    except OSError as exc:
        logger.warning(
            'Could not read MERGE_RR at %s: %s. Treated as suspect — a file '
            'this run never inspected is unknown, not healthy.',
            merge_rr_path, exc,
        )
        data = b''
        unreadable = True

    parsed = parse_merge_rr(data)
    rr_cache = common_dir / 'rr-cache'
    dangling = tuple(
        record for record in parsed.records
        if not (rr_cache / record.conflict_id).is_dir()
    )
    return MergeRrScan(
        merge_rr_path=merge_rr_path,
        common_dir=common_dir,
        records=parsed.records,
        dangling=dangling,
        unparsable=parsed.unparsable,
        unreadable=unreadable,
    )


def _suspicion(
    dangling: tuple[MergeRrRecord, ...],
    unparsable: tuple[bytes, ...],
    *,
    unreadable: bool,
) -> str:
    """One rendering of why a MERGE_RR is suspect, for every place that says so.

    Three callers describe the same three facts — the quarantine's WARNING, the
    report-only WARNING, and :attr:`PreflightResult.unrepaired` — and a
    suspicion named at two of them and missed at the third reads as "suspect,
    for no reason given".
    """
    return (
        f'dangling rr-cache refs: '
        f'[{", ".join(record.conflict_id for record in dangling)}]; '
        f'unparsable records: {len(unparsable)}; unreadable: {unreadable}'
    )


#: Backup names are ``MERGE_RR.quarantined-<stamp>`` plus a counter when a
#: same-second repeat would collide.  The stamp orders the evidence for a
#: reader; the EXCLUSIVE CLAIM in :func:`quarantine_merge_rr` is what
#: guarantees nothing is ever overwritten — the counter only supplies the next
#: name to try.
_QUARANTINE_STAMP = '%Y%m%dT%H%M%S'

#: Directory under the COMMON git dir that holds quarantined MERGE_RR files.
#: Common-dir, not per-worktree, for the reason :func:`_quarantine_dir` gives.
_QUARANTINE_DIRNAME = 'rerere-specimens'

#: How many names to try before giving up and reporting the MERGE_RR
#: un-repaired.  Each failure means another run claimed that exact name in
#: between, so exhausting this needs a burst of concurrent preflights on one
#: worktree; the bound exists so the loop cannot spin, not because it is
#: expected to be reached.
_QUARANTINE_CLAIM_ATTEMPTS = 5


def quarantine_merge_rr(scan: MergeRrScan) -> Path | None:
    """Move a suspect MERGE_RR aside and return the backup path.

    A no-op returning ``None`` unless the scan is suspect: a healthy MERGE_RR
    is left exactly where git put it, untouched.

    The file is MOVED rather than deleted because it is the only record of
    which conflict ids the wedged worktree was carrying, and the abort that
    follows would otherwise destroy it — a successful ``git rebase --abort``
    deletes MERGE_RR outright.  It moves to the COMMON dir rather than beside
    the original, for the reason :func:`_quarantine_dir` gives.  Backup names
    never collide, so a worktree that wedges twice keeps both wedges' evidence
    rather than overwriting the first with the second — and that holds for
    CONCURRENT runs, not just ordered ones, because the name is CLAIMED rather
    than merely found free.  A counter
    that checks ``exists()`` and then renames answers only the sequential case:
    two preflights on the same worktree in the same second both see the base
    name free, and ``Path.rename`` on POSIX silently replaces its destination,
    so the second destroys the first backup while both report success.
    ``os.link`` fails with ``FileExistsError`` instead, which is the whole
    reason the move is a link-then-unlink rather than a rename.

    Logs at WARNING naming the dangling ids and the backup path: a repair that
    happens silently is indistinguishable from a repair that never ran.

    A move that FAILS is reported, never raised: ``None`` comes back exactly as
    for a healthy scan, and the caller distinguishes the two by the scan it
    already holds — a suspect scan with no backup is what
    :attr:`PreflightResult.unrepaired` renders and what turns the verdict
    ``blocked``.  Degrading this way is the whole point of the module: an abort
    that must still run cannot be blocked by a repair that could not, and it is
    the same shape :func:`sweep_stale_locks` uses for a lock it cannot unlink.

    A returned path means MOVED, never merely COPIED, and the link-then-unlink
    has two halves that fail separately.  A failed claim leaves nothing behind;
    a failed unlink leaves the evidence safe at the backup and the suspect file
    exactly where git put it.  Only the second is tempting to call a success,
    and it is not one: answering with the backup there would make ``unrepaired``
    skip the MERGE_RR arm and verdict a worktree ``repaired`` whose damage is
    still on disk, which is the one direction this module must never fail in.
    Both halves therefore answer ``None``, and both name the backup in the log
    so nothing this run wrote is lost to the reader.
    """
    if not scan.suspect:
        return None

    for _ in range(_QUARANTINE_CLAIM_ATTEMPTS):
        backup = _free_backup_path(scan)
        try:
            backup.parent.mkdir(parents=True, exist_ok=True)
            os.link(scan.merge_rr_path, backup)
        except FileExistsError:
            continue          # another run claimed it; take the next name
        except OSError as exc:
            logger.warning(
                'Could not quarantine suspect MERGE_RR %s: %s. Left in place; '
                'the abort still runs and the result reports it un-repaired.',
                scan.merge_rr_path, exc,
            )
            return None
        break
    else:
        logger.warning(
            'Could not claim a free backup name for %s after %d attempts. '
            'Left in place; the abort still runs and the result reports it '
            'un-repaired.',
            scan.merge_rr_path, _QUARANTINE_CLAIM_ATTEMPTS,
        )
        return None

    try:
        scan.merge_rr_path.unlink()
    except OSError as exc:
        logger.warning(
            'Copied suspect MERGE_RR %s to %s but could not remove the '
            'original: %s. The evidence is preserved at the copy; the suspect '
            'file is STILL IN PLACE, so this run reports it un-repaired and '
            'the next guarded abort will quarantine it again.',
            scan.merge_rr_path, backup, exc,
        )
        return None
    logger.warning(
        'Quarantined suspect MERGE_RR to %s — %s. Evidence preserved; the '
        'abort that follows would have deleted it.',
        backup,
        _suspicion(scan.dangling, scan.unparsable, unreadable=scan.unreadable),
    )
    return backup


def _quarantine_dir(scan: MergeRrScan) -> Path:
    """Where evidence is kept: a COMMON-dir directory, NOT beside the original.

    In a linked worktree MERGE_RR lives in ``.git/worktrees/<id>/``, a
    directory the worktree lifecycle OWNS: any successful landing, reaper sweep
    or ``cleanup_worktree`` removes it and everything inside it.  Evidence kept
    there survives exactly until the task succeeds -- which is the case nobody
    is looking at, so preservation that is a no-op on the common path is
    indistinguishable from the ``rm`` this module refuses to do.  A real
    specimen was lost exactly that way (task 5033, merge ``dcbff02f00``).

    The common dir is the natural home rather than merely a surviving one: it
    already holds ``rr-cache``, so a specimen sits beside the thing it is
    evidence ABOUT, and it outlives every worktree by construction.  It is also
    necessarily on the same filesystem as the MERGE_RR being moved, which the
    link-then-unlink in :func:`quarantine_merge_rr` requires.
    """
    return scan.common_dir / _QUARANTINE_DIRNAME


def _worktree_token(scan: MergeRrScan) -> str:
    """Which worktree a specimen came from, for a name that must now say so.

    Beside the original, the containing directory identified the worktree.  In
    the shared directory the NAME has to, or two worktrees' specimens are
    indistinguishable to a reader and collide for the claim.  The per-worktree
    git dir is named for the worktree; the main checkout's git dir IS the
    common dir and carries no such name.
    """
    git_dir = scan.merge_rr_path.parent
    try:
        if git_dir.resolve() == scan.common_dir.resolve():
            return 'main'
    except OSError:
        pass          # unresolvable is not the main checkout; name it as found
    return git_dir.name


def _free_backup_path(scan: MergeRrScan) -> Path:
    """First unused ``<common>/rerere-specimens/MERGE_RR.<worktree>-<stamp>``.

    A CANDIDATE, not a reservation: between this answer and the claim another
    process may take the name, which is why the caller claims it exclusively
    and asks again rather than trusting this.
    """
    stamp = datetime.now(UTC).strftime(_QUARANTINE_STAMP)
    base = _quarantine_dir(scan) / (
        f'{scan.merge_rr_path.name}.{_worktree_token(scan)}-{stamp}'
    )
    if not base.exists():
        return base
    counter = 2
    while (candidate := Path(f'{base}-{counter}')).exists():
        counter += 1
    return candidate


#: Age past which an UNHELD ``*.lock`` is treated as abandoned.  Conservative
#: by a wide margin: no legitimate git operation holds MERGE_RR.lock longer
#: than a single rerere write, while the incident's lock was 22.6 hours old.
DEFAULT_LOCK_STALE_AFTER_SECONDS = 3600.0


@dataclass(frozen=True)
class LockFinding:
    """One ``*.lock`` and the facts the removal decision turns on.

    ``holders_confirmed`` is the difference between "nothing holds this" and
    "this run could not tell".  Both render as an empty ``holder_pids``, and
    only the first may authorise a deletion.
    """

    path: Path
    age_seconds: float
    holder_pids: tuple[int, ...]
    holders_confirmed: bool = True


@dataclass(frozen=True)
class LockSweep:
    """Which locks were cleared and which were deliberately left alone.

    Retained locks are returned rather than dropped: a lock this refused to
    touch is the reason a subsequent abort may still fail, so the caller can
    say so instead of leaving the operator to rediscover it.
    """

    removed: tuple[LockFinding, ...]
    retained: tuple[LockFinding, ...]


#: Where the holder probe looks.  Injectable for the same reason
#: ``verify_cancel.lane_lock_holder_pids`` takes ``locks_path``: the answer is
#: read out of a pseudo-filesystem, and a test that cannot substitute one has
#: to reach into the implementation to say anything about it.
_PROC_ROOT = Path('/proc')


@dataclass(frozen=True)
class HolderScan:
    """Which pids hold which of the paths asked about — and whether we know.

    ``confirmed`` is false when the process table could not be enumerated at
    all, which makes every empty entry mean "unknown" rather than "unheld".
    The caller must not collapse the two: only one of them may delete a file.
    """

    pids_by_path: Mapping[Path, tuple[int, ...]]
    confirmed: bool


def scan_lock_holders(
    paths: Iterable[Path], *, proc_root: Path = _PROC_ROOT,
) -> HolderScan:
    """Find the pids holding *paths* open, in ONE pass over the process table.

    Deliberately NOT
    ``orchestrator/src/orchestrator/verify_cancel.py::lane_lock_holder_pids``,
    which reads ``/proc/locks``.  That file lists kernel FLOCK/POSIX locks, whereas git's
    ``*.lock`` files are plain ``O_CREAT|O_EXCL`` sentinels held open by file
    descriptor with no kernel lock at all — so it would report "no holder" for
    every live git lock, and this sweep would delete them.  The two probes
    answer different questions and neither substitutes for the other.

    ONE pass for every candidate, not one per lock: each pass readlinks every
    descriptor of every process, so per-lock repetition multiplies the syscalls
    by the number of locks — under precisely the contention this module exists
    to handle, and serialising a merge-lane abort behind it.

    THREE failures, and they are NOT the same answer:

    * ``proc_root`` cannot be enumerated — no ``/proc`` mounted at all.  The
      scan saw nothing, so ``confirmed`` is false and every path comes back
      unknown.  This measured as RAISING out of the preflight before this
      guard, which breached the module's totality invariant.
    * A pid's ``fd`` directory is not ours to read.  Tolerated, and it does NOT
      make the scan unconfirmed.  MEASURED on this host as the orchestrator's
      own uid: 860 of 1329 pids are unreadable, so vetoing on them would retain
      every lock on every run — silently deleting the stale-lock half of this
      task while its unit cases stayed green.
    * A pid vanishes mid-scan.  Its descriptors are gone because the process
      is, so it holds nothing; tolerated, and likewise not unconfirmed.

    KNOWN BOUND, stated because it cannot be closed here: under a ``hidepid``
    ``/proc``, other users' pids are not listed at all rather than denied, so a
    foreign holder is invisible and no counting of denials would reveal it.
    The conjunction in :func:`sweep_stale_locks` is the mitigation — a file is
    removed only if it ALSO has not been touched for an hour.

    ASKING ABOUT NOTHING COSTS NOTHING.  A worktree with no ``*.lock`` files is
    the overwhelmingly common case — every healthy git dir — and this runs
    inside every abort the orchestrator issues, including the ``abort_merge``
    its caller fires on any non-zero merge rc.  A pass costs ~0.18s measured,
    so answering the empty request without one is the same syscall-frugality
    argument that makes the scan per-sweep rather than per-lock.
    """
    wanted = {str(path.resolve()): path for path in paths}
    if not wanted:
        return HolderScan(pids_by_path={}, confirmed=True)

    holders: dict[Path, set[int]] = {path: set() for path in wanted.values()}
    try:
        entries = list(proc_root.iterdir())
    except OSError as exc:
        logger.warning(
            'Could not enumerate %s: %s. Lock holders are UNKNOWN for this '
            'run, so nothing is removed on the strength of finding none.',
            proc_root, exc,
        )
        return HolderScan(
            pids_by_path={path: () for path in holders}, confirmed=False,
        )

    for entry in entries:
        if not entry.name.isdigit():
            continue
        try:
            fds = list((entry / 'fd').iterdir())
        except OSError:
            continue
        for fd in fds:
            try:
                resolved = os.readlink(fd)
            except OSError:
                continue
            path = wanted.get(resolved)
            if path is not None:
                holders[path].add(int(entry.name))
    return HolderScan(
        pids_by_path={
            path: tuple(sorted(pids)) for path, pids in holders.items()
        },
        confirmed=True,
    )


def survey_locks(
    git_dir: Path, *, now: datetime | None = None,
    proc_root: Path = _PROC_ROOT,
) -> tuple[LockFinding, ...]:
    """Measure every ``*.lock`` directly under *git_dir*, removing nothing.

    The observation half of :func:`sweep_stale_locks`, which consumes it and
    then partitions on the conjunction.  Keeping the measurement in one place
    means the report-only path and the removing path cannot drift into
    disagreeing about what they saw.
    """
    moment = now if now is not None else datetime.now(UTC)
    aged: list[tuple[Path, float]] = []
    for lock in sorted(git_dir.glob('*.lock')):
        try:
            age = moment.timestamp() - lock.stat().st_mtime
        except OSError:
            continue
        aged.append((lock, age))

    scan = scan_lock_holders([lock for lock, _ in aged], proc_root=proc_root)
    return tuple(
        LockFinding(
            path=lock,
            age_seconds=age,
            holder_pids=scan.pids_by_path.get(lock, ()),
            holders_confirmed=scan.confirmed,
        )
        for lock, age in aged
    )


def _is_abandoned(finding: LockFinding, stale_after_seconds: float) -> bool:
    """Does *finding* satisfy the removal conjunction?

    THE single statement of "this lock may be cleared", so the removing path
    and every path that merely REPORTS cannot drift into disagreeing about
    which locks matter.  They did drift: the report-only preflight used to
    describe an abandoned lock as merely retained, so a worktree the repairing
    run would have fixed came back ``clean`` and a skill branching on that
    verdict walked into the rc 128 the module exists to prevent.
    """
    return (
        finding.holders_confirmed
        and not finding.holder_pids
        and finding.age_seconds > stale_after_seconds
    )


def sweep_stale_locks(
    *,
    git_dir: Path,
    now: datetime | None = None,
    stale_after_seconds: float = DEFAULT_LOCK_STALE_AFTER_SECONDS,
    proc_root: Path = _PROC_ROOT,
) -> LockSweep:
    """Remove abandoned ``*.lock`` files directly under *git_dir*.

    A lock is removed ONLY when the holder scan CONFIRMED that nothing holds it
    open AND its mtime is older than *stale_after_seconds*.  The conjunction is
    the whole contract: age alone would delete a lock a live process still
    depends on, and a holder check alone would clear a lock the instant its
    writer blinked.

    A scan that could not run answers "unknown", never "unheld" — see
    :func:`scan_lock_holders`.  Unknown retains, and the lock is reported, so
    the one direction this module must never fail in stays closed.

    The mtime of the operation the lock BLOCKS is never consulted — not as a
    tiebreak, not as a hint.  Incident 3517's lock was older than the
    rebase-merge directory it blocked, so "newer than the operation" clears a
    lock that must be kept and "older than the operation" keeps one that must
    be cleared; the relative comparison is wrong in both directions.

    Size is likewise not a liveness signal: git's lock files are empty for as
    long as they are held, so a 0-byte lock is exactly as likely to be live as
    it is to be abandoned.

    SCOPE IS EVERY ``*.lock`` DIRECTLY UNDER *git_dir*, not just the
    ``MERGE_RR.lock`` the incident left behind.  git answers rc 128 with the
    same unactionable advice whichever sentinel is stale, so a sweep that
    cleared only one of them would leave the identical failure reachable
    through ``index.lock``.  The breadth is asserted rather than left as a side
    effect of the glob, by
    ``test_an_abandoned_index_lock_is_swept_and_a_held_one_is_not``.

    THE BAR IS STRICTER THAN THE ONE ``index.lock`` ALREADY HAS, deliberately.
    ``orchestrator/src/orchestrator/git_ops.py::_INDEX_LOCK_STALE_FLOOR_S`` is
    300s and governs ``project_root``'s index lock at two collaborating sites
    that must never drift from EACH OTHER — one short-circuits a wait, the
    other renders ``rm -f`` ADVICE from the same bar.  This module neither
    reads nor moves those: at 3600s every file it removes is one both sites
    would already call stale, so the agreement they need is untouched.  What it
    does that they do not is perform the deletion, which is affordable only in
    this position — on a recovery path, for a worktree whose abort has already
    been asked for, behind a confirmed-unheld holder scan AND an hour of
    stillness.  The constant is not imported because git_ops imports THIS
    module; a reciprocal import is the cycle :data:`AbortRunner` documents.
    """
    removed: list[LockFinding] = []
    retained: list[LockFinding] = []

    for finding in survey_locks(git_dir, now=now, proc_root=proc_root):
        if not _is_abandoned(finding, stale_after_seconds):
            retained.append(finding)
            continue
        lock = finding.path
        try:
            lock.unlink()
        except OSError:
            retained.append(finding)
            continue
        removed.append(finding)
        logger.warning(
            'Removed stale git lock %s — age %.0fs (threshold %.0fs), '
            'no holder process found in a completed /proc scan.',
            finding.path, finding.age_seconds, stale_after_seconds,
        )

    return LockSweep(removed=tuple(removed), retained=tuple(retained))


#: Verdicts the CLI and the skills branch on.  ``blocked`` is the only one
#: that means "do not proceed", and it covers two shapes.  Damage this run did
#: not repair — typically a lock something still holds open, so the abort ahead
#: will hit git's "Another git process seems to be running" and a human has to
#: decide what that process is.  And a worktree that could not be RESOLVED, so
#: nothing was inspected at all: ``clean`` there would be a claim about files
#: no one read, and "nothing inspected" is strictly less known than the
#: unreadable MERGE_RR that already refuses to be called clean.
VERDICT_CLEAN = 'clean'   # nothing needs attention
VERDICT_REPAIRED = 'repaired'  # damage found and fixed; proceed
VERDICT_BLOCKED = 'blocked'  # damage this run did not fix; do not proceed


@dataclass(frozen=True)
class PreflightResult:
    """Everything the preflight found and everything it changed."""

    worktree: Path
    dangling: tuple[MergeRrRecord, ...]
    unparsable: tuple[bytes, ...]
    merge_rr_backup: Path | None
    locks_removed: tuple[LockFinding, ...]
    locks_retained: tuple[LockFinding, ...]
    resolved: bool = True
    merge_rr_unreadable: bool = False
    lock_stale_after_seconds: float = DEFAULT_LOCK_STALE_AFTER_SECONDS

    @property
    def unrepaired(self) -> tuple[str, ...]:
        """Findings this run did NOT fix — the reason a caller must not proceed.

        Three sources.  An UNRESOLVED worktree comes first because it subsumes
        the other two: nothing was inspected, so every other field is empty for
        want of a look rather than for want of damage, and a caller branching
        on the verdict alone would read that emptiness as a licence to proceed.
        A retained lock is judged by :func:`_retained_lock_reason`, against the
        same threshold that decided the sweep — carried on
        :attr:`lock_stale_after_seconds` precisely so the report-only path
        cannot answer a different question from the repairing one.  A suspect
        MERGE_RR with no backup means the damage is still in place — the
        ``report_only`` case, where leaving it is the whole point, and also a
        quarantine that failed.

        ``resolved`` is reported as a field too, but the ENUM is what every
        consumer branches on — both skills read ``verdict`` and neither reads
        ``resolved`` — so a distinction that lives only in the field is a
        distinction the consumer contract does not carry.
        """
        reasons = [
            f'worktree {self.worktree} could not be resolved — nothing was '
            f'inspected, so neither MERGE_RR nor the locks are known'
        ] if not self.resolved else []
        reasons += [
            reason
            for finding in self.locks_retained
            if (reason := _retained_lock_reason(
                finding, self.lock_stale_after_seconds,
            ))
        ]
        suspect = self.dangling or self.unparsable or self.merge_rr_unreadable
        if self.merge_rr_backup is None and suspect:
            reasons.append(
                'suspect MERGE_RR left in place — '
                + _suspicion(
                    self.dangling, self.unparsable,
                    unreadable=self.merge_rr_unreadable,
                ),
            )
        return tuple(reasons)

    @property
    def verdict(self) -> str:
        """``clean`` means NOTHING NEEDS ATTENTION, not "I changed nothing".

        The distinction only bites in ``report_only``, and that is exactly
        where a caller acts on the answer: a verdict derived from what was
        MUTATED would report ``clean`` for a worktree the same payload
        describes as dangling, and a skill branching on it would proceed
        unguarded into the state this exists to catch.
        """
        if self.unrepaired:
            return VERDICT_BLOCKED
        if self.merge_rr_backup is not None or self.locks_removed:
            return VERDICT_REPAIRED
        return VERDICT_CLEAN

    def as_json(self) -> dict:
        """The CLI payload — structured values, never a rendered sentence."""
        return {
            'verdict': self.verdict,
            'unrepaired': list(self.unrepaired),
            'worktree': str(self.worktree),
            'resolved': self.resolved,
            'dangling': [
                {'conflict_id': r.conflict_id, 'path': r.path} for r in self.dangling
            ],
            'unparsable_record_count': len(self.unparsable),
            'merge_rr_backup': (
                str(self.merge_rr_backup) if self.merge_rr_backup else None
            ),
            'locks_removed': [_lock_json(f) for f in self.locks_removed],
            'locks_retained': [_lock_json(f) for f in self.locks_retained],
        }


def _retained_lock_reason(
    finding: LockFinding, stale_after_seconds: float,
) -> str | None:
    """Why a retained lock still needs attention, or ``None`` if it does not.

    Three arms.  A lock with a live holder is a human's decision: something is
    using it, and this module will not guess what.  A lock whose holders could
    not be determined is the same decision with less evidence, and is named
    rather than passed over in silence — an unscannable process table is
    exactly when an operator most needs to be told why a lock survived.  The
    third arm is a lock that MET the removal conjunction and is on disk anyway:
    report-only, or an unlink that failed.  Why it survived does not change
    what the caller must do about it, so both render as one reason — the abort
    ahead will hit git's "Another git process seems to be running" either way.
    """
    if finding.holder_pids:
        return (
            f'lock {finding.path} held by pids '
            f'{", ".join(str(pid) for pid in finding.holder_pids)}'
        )
    if not finding.holders_confirmed:
        return (
            f'lock {finding.path} left in place — holders unknown, so finding '
            f'none is not evidence there are none'
        )
    if _is_abandoned(finding, stale_after_seconds):
        return (
            f'lock {finding.path} left in place — abandoned for '
            f'{finding.age_seconds:.0f}s with no holder (threshold '
            f'{stale_after_seconds:.0f}s), and this run did not remove it'
        )
    return None


def _lock_json(finding: LockFinding) -> dict:
    return {
        'path': str(finding.path),
        'age_seconds': round(finding.age_seconds, 3),
        'holder_pids': list(finding.holder_pids),
        'holders_confirmed': finding.holders_confirmed,
    }


def resolve_git_dirs(worktree: Path) -> tuple[Path, Path] | None:
    """Resolve *worktree*'s ``(git_dir, common_dir)``, or ``None`` if it cannot.

    The two differ in a linked worktree, which is the case that matters here:
    MERGE_RR is per-worktree while rr-cache is shared.  One ``git rev-parse``
    answers both, so this is the module's only subprocess.

    IDENTITY IS CHECKED, NOT ASSUMED, and that is what makes the preflight safe
    to point at a wedged path.  Git's repository discovery walks UP, so a cwd
    that exists but is not itself a worktree — one whose ``.git`` file was
    removed or corrupted, which is the very class of wedged state this module
    is invoked for, or a stale ``.worktrees/<id>`` recreated as a plain
    directory — resolves to the ENCLOSING repository.  The preflight's repairs
    are destructive (it quarantines MERGE_RR and unlinks locks), and in
    production that enclosing repository is ``project_root``, which is
    machine-operated.  The unguarded abort this decorates was harmless there
    ("no rebase in progress"), so discovery escape would hand a recovery helper
    blast radius the code it guards never had.  One measure closes it:
    ``--show-toplevel`` must name *worktree* itself, and a mismatch answers
    ``None``.  That covers every route to a foreign repository, not just the
    upward walk — ``GIT_DIR``/``GIT_WORK_TREE`` inherited from the orchestrator's
    own environment name one outright, and a ceiling on the ascent would not
    see them.

    A cwd git cannot even be spawned in — a worktree deleted out-of-band, or a
    path that is a file — raises ``OSError`` from the spawn itself, before any
    exit code exists, and a probe that never answers within
    :data:`_PROBE_TIMEOUT_SECONDS` raises ``TimeoutExpired``.  Both are
    answered with ``None``, the same as a non-zero exit and the same as a
    foreign repository, so the caller takes the one unresolved branch instead
    of branches that differ only in how the worktree failed to be the worktree.
    It also keeps the vanished-worktree case reaching ``git_ops._run``, whose
    own pre-flight raises the typed ``WorktreeMissing`` its consumers match on;
    raising that here instead is impossible without an import cycle (see
    :data:`AbortRunner`) and would duplicate the class besides.
    """
    try:
        resolved_worktree = worktree.resolve()
    except OSError as exc:
        logger.warning('Could not resolve %s: %s', worktree, exc)
        return None
    try:
        proc = subprocess.run(
            ['git', 'rev-parse', '--git-dir', '--git-common-dir', '--show-toplevel'],
            cwd=str(worktree), capture_output=True, text=True, check=False,
            timeout=_PROBE_TIMEOUT_SECONDS,
        )
    except OSError as exc:
        logger.warning('Could not spawn git in %s: %s', worktree, exc)
        return None
    except subprocess.TimeoutExpired:
        logger.warning(
            'git rev-parse in %s did not answer within %.0fs; treating the '
            'worktree as unresolved rather than holding the abort behind it.',
            worktree, _PROBE_TIMEOUT_SECONDS,
        )
        return None
    if proc.returncode != 0:
        return None
    lines = proc.stdout.splitlines()
    if len(lines) != 3:
        return None
    # Split on LINES, not whitespace: git prints one path per line and a git
    # dir may legally contain spaces.  Each path is relative to *worktree*
    # unless already absolute, which ``os.path.join`` handles either way.
    git_dir, common_dir, toplevel = (
        Path(os.path.join(worktree, line)) for line in lines
    )
    if toplevel.resolve() != resolved_worktree:
        logger.warning(
            'Rebase-recovery preflight was pointed at %s, but git resolved that '
            'to the repository rooted at %s. Refusing to repair a repository '
            'that is not the worktree asked about.',
            worktree, toplevel,
        )
        return None
    return git_dir, common_dir


def preflight_rebase_recovery(
    worktree: Path,
    *,
    now: datetime | None = None,
    lock_stale_after_seconds: float = DEFAULT_LOCK_STALE_AFTER_SECONDS,
    report_only: bool = False,
) -> PreflightResult:
    """Make a ``git rebase``/``merge --abort`` in *worktree* safe to run.

    Scans MERGE_RR, quarantines it when suspect, and sweeps abandoned locks.
    The abort itself is the CALLER's to issue, prefixed with
    :data:`RECOVERY_GIT` — see :func:`guarded_abort`, which pairs the two.

    *report_only* performs detection and reporting with no mutation, so an
    operator can inspect before authorising a repair.  It changes WHAT IS DONE
    about a finding, never WHETHER IT IS A FINDING: *lock_stale_after_seconds*
    still decides which locks count, and everything a repairing run would have
    fixed comes back under :attr:`PreflightResult.unrepaired` with the verdict
    ``blocked``.

    FAIL-SAFE, and TOTAL: no filesystem state makes this raise.  Every failure
    — a worktree git cannot be spawned in, a MERGE_RR that cannot be read, a
    quarantine or a lock removal that cannot be performed — is folded into the
    returned value, whose ``verdict`` then says what was left un-repaired.
    This decorates a RECOVERY path, so it must never itself become the reason
    recovery fails: an unguarded abort that might crash still beats no abort at
    all.  That is an enforced invariant, not an aspiration —
    ``orchestrator/tests/test_rebase_recovery.py::TestPreflightIsTotal``
    parametrizes it over the hostile states measured to have broken it.
    """
    worktree = Path(worktree)
    dirs = resolve_git_dirs(worktree)
    if dirs is None:
        logger.warning(
            'Rebase-recovery preflight could not resolve git dirs for %s; '
            'nothing was inspected, and the verdict says so.', worktree,
        )
        return PreflightResult(
            worktree=worktree, dangling=(), unparsable=(), merge_rr_backup=None,
            locks_removed=(), locks_retained=(), resolved=False,
        )

    git_dir, common_dir = dirs
    scan = scan_merge_rr(git_dir=git_dir, common_dir=common_dir)
    backup = None if report_only else quarantine_merge_rr(scan)
    if report_only and scan.suspect:
        logger.warning(
            'Rebase-recovery preflight (report-only) found a suspect MERGE_RR '
            'at %s — %s. Nothing was moved.',
            scan.merge_rr_path,
            _suspicion(scan.dangling, scan.unparsable, unreadable=scan.unreadable),
        )

    sweep = (
        LockSweep(removed=(), retained=survey_locks(git_dir, now=now))
        if report_only
        else sweep_stale_locks(
            git_dir=git_dir, now=now, stale_after_seconds=lock_stale_after_seconds,
        )
    )

    return PreflightResult(
        worktree=worktree,
        dangling=scan.dangling,
        unparsable=scan.unparsable,
        merge_rr_backup=backup,
        locks_removed=sweep.removed,
        locks_retained=sweep.retained,
        merge_rr_unreadable=scan.unreadable,
        lock_stale_after_seconds=lock_stale_after_seconds,
    )


#: The subprocess runner :func:`guarded_abort` delegates to.  Injected rather
#: than imported because the only caller is
#: ``orchestrator/src/orchestrator/git_ops.py``, which imports THIS module; a
#: reciprocal import would be a cycle.
AbortRunner = Callable[..., Awaitable[tuple[int, str, str]]]


async def guarded_abort(
    verb: str, cwd: Path, run: AbortRunner,
) -> tuple[int, str, str]:
    """Run ``git <verb> --abort`` in *cwd* behind the recovery guard.

    THE single abort path for the orchestrator.  Every ``rebase --abort`` and
    ``merge --abort`` routes through here, because the failures it guards are
    invisible at the call site: an operator reading any one of them would have
    no reason to suspect that aborting can segfault.

    Two things happen, in this ORDER, and the order is the contract:

    1. :func:`preflight_rebase_recovery` quarantines a MERGE_RR whose rr-cache
       refs dangle and clears abandoned ``*.lock`` files.  It must precede the
       abort — a successful abort DELETES MERGE_RR, so a preflight running
       afterwards would find nothing, report clean, and preserve no evidence.
    2. The abort itself runs prefixed with :data:`RECOVERY_GIT`, so git never
       opens MERGE_RR.  That neutralises both the dangling-ref crash and the
       stale-lock rc 128.

    The preflight is sync filesystem work, so it runs off-thread rather than
    blocking the event loop.  It never raises — see
    :func:`preflight_rebase_recovery` for the enforced totality invariant and
    the test that pins it — so a worktree it cannot inspect, or damage it
    cannot repair, degrades to an unguarded abort rather than to no abort.
    The abort's own errors still surface: ``run`` reports a non-zero exit as a
    value, and ``git_ops._run`` still raises ``WorktreeMissing`` for a cwd that
    has vanished, which is the typed exception its callers recover from.

    A :data:`VERDICT_BLOCKED` preflight does NOT stop the abort, and it is
    LOGGED before the abort runs.  Proceeding is the fail-safe choice;
    proceeding silently would throw the diagnosis away, because the sweep logs
    only REMOVALS — a lock retained because something still holds it appears in
    no log at all, and the abort then fails rc 128 with git's "Another git
    process seems to be running", whose remedy is exactly the holder pid this
    line carries.

    Returns *run*'s ``(rc, stdout, stderr)`` unchanged, so no call site's
    control flow or return value has to change.  A FAILED abort is logged
    here, at the one point that sees every abort's exit code, rather than at
    each of the four call sites: a caller that discards the rc would otherwise
    report "aborted" for an abort that did not happen, and the worktree it
    then hands on is half-applied rather than clean.  Logging cannot make such
    a caller correct — it makes the failure legible instead of silent, which
    is the fail-soft floor this module holds everywhere else.

    Only a WEDGED failure is logged, not every non-zero abort: git exits
    non-zero just as readily because there was nothing to abort, which is the
    ordinary outcome of a defensive abort and of a rebase that failed before
    starting one.  :func:`_operation_still_in_progress` separates the two, and
    it runs only on the failure path.
    """
    result = await asyncio.to_thread(preflight_rebase_recovery, cwd)
    if result.verdict == VERDICT_BLOCKED:
        logger.warning(
            'Proceeding with git %s --abort in %s despite findings this run '
            'did not repair: %s',
            verb, cwd, '; '.join(result.unrepaired),
        )
    rc, out, err = await run([*RECOVERY_GIT, verb, '--abort'], cwd=cwd)
    if rc != 0 and await asyncio.to_thread(_operation_still_in_progress, cwd):
        logger.warning(
            'git %s --abort FAILED in %s (rc %d) and the interrupted operation '
            'is STILL THERE: %s. The worktree is not clean, so any step that '
            'runs next sees a half-applied tree.',
            verb, cwd, rc, (err or out or '').strip() or '(no output)',
        )
    return rc, out, err


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

#: Files and directories whose PRESENCE means git has an interrupted operation
#: to undo.  Checked only AFTER a failed abort, where they are what separates
#: "the abort did not clean up" from "there was nothing to abort" -- two states
#: git reports with the same non-zero exit.
_IN_PROGRESS_MARKERS = (
    'rebase-merge', 'rebase-apply', 'MERGE_HEAD', 'CHERRY_PICK_HEAD', 'REVERT_HEAD',
)


def _operation_still_in_progress(worktree: Path) -> bool:
    """Does *worktree* still carry an interrupted git operation?

    Answers the question a failed abort cannot: ``git rebase --abort`` exits
    non-zero BOTH when it could not undo an interrupted rebase AND when there
    was no rebase to undo ("fatal: No rebase in progress?").  Only the first is
    a wedged worktree; the second is the ordinary outcome of aborting
    defensively, or of a rebase that failed BEFORE it started one (an unstaged
    change, a bad revision).  Warning about the second would cry wolf on a
    routine path and, worse, assert a half-applied tree that does not exist.

    Unresolvable answers TRUE: a worktree this cannot inspect is unknown, and
    an unknown one is reported rather than passed over in silence -- the same
    direction every other degraded answer in this module takes.
    """
    dirs = resolve_git_dirs(worktree)
    if dirs is None:
        return True
    git_dir, _ = dirs
    return any((git_dir / marker).exists() for marker in _IN_PROGRESS_MARKERS)


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog='python -m orchestrator.rebase_recovery',
        description=(
            'Make a git rebase/merge --abort safe to run: quarantine a MERGE_RR '
            'whose rr-cache refs dangle, and clear abandoned *.lock files.'
        ),
    )
    sub = p.add_subparsers(dest='verb', required=True)

    preflight = sub.add_parser(
        'preflight',
        help='inspect and repair a worktree, emitting a JSON verdict',
    )
    preflight.add_argument('--worktree', required=True,
                           help='path to the worktree to inspect')
    preflight.add_argument(
        '--lock-stale-after-seconds', type=float,
        default=DEFAULT_LOCK_STALE_AFTER_SECONDS,
        help=(
            'age past which an UNHELD *.lock is removed '
            f'(default: {DEFAULT_LOCK_STALE_AFTER_SECONDS:.0f}); a HELD lock is '
            'retained at any age'
        ),
    )
    preflight.add_argument(
        '--report-only', action='store_true',
        help='detect and report without moving or removing anything',
    )
    return p


def main(argv: list[str] | None = None) -> int:
    """Entry point; returns an exit code.

    Always 0 for a completed inspection, including the unresolved case: the
    VERDICT carries the outcome, and a non-zero exit would make a skill treat a
    guarded recovery as a failed one.
    """
    args = _build_parser().parse_args(argv)

    result = preflight_rebase_recovery(
        Path(args.worktree),
        lock_stale_after_seconds=args.lock_stale_after_seconds,
        report_only=args.report_only,
    )
    print(json.dumps(result.as_json()))
    return 0


if __name__ == '__main__':
    sys.exit(main())
