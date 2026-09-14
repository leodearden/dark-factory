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
  independent failure: it reproduces with a perfectly intact rr-cache.

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
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

logger = logging.getLogger(__name__)

#: Command prefix for every abort issued on a recovery path.  Disabling rerere
#: means git never opens MERGE_RR, which is what makes the abort survive both
#: a dangling rr-cache ref and a stale ``MERGE_RR.lock``.
RECOVERY_GIT = ('git', '-c', 'rerere.enabled=false')

#: One MERGE_RR record: a conflict id and the path it belongs to.  The id is
#: ``<40-hex>`` with an OPTIONAL rerere variant suffix ``.<N>``; the suffix is
#: part of the rr-cache directory name, so the group must capture it.
_RECORD = re.compile(rb'^([0-9a-f]{40}(?:\.\d+)?)\t(.+)$', re.DOTALL)


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
    file names a conflict id with no backing rr-cache directory, OR when it
    holds a record git itself would reject.  The two have different symptoms
    and the same remedy, so they share one flag.
    """

    merge_rr_path: Path
    records: tuple[MergeRrRecord, ...]
    dangling: tuple[MergeRrRecord, ...]
    unparsable: tuple[bytes, ...]

    @property
    def suspect(self) -> bool:
        return bool(self.dangling or self.unparsable)


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
    and yields an empty scan rather than an error.  Both arguments stay
    explicit so the classifier is pure filesystem work, testable without a
    repository; :func:`resolve_git_dirs` supplies them for real callers.
    """
    merge_rr_path = git_dir / 'MERGE_RR'
    try:
        data = merge_rr_path.read_bytes()
    except FileNotFoundError:
        data = b''

    parsed = parse_merge_rr(data)
    rr_cache = common_dir / 'rr-cache'
    dangling = tuple(
        record for record in parsed.records
        if not (rr_cache / record.conflict_id).is_dir()
    )
    return MergeRrScan(
        merge_rr_path=merge_rr_path,
        records=parsed.records,
        dangling=dangling,
        unparsable=parsed.unparsable,
    )


#: Backup names are ``MERGE_RR.quarantined-<stamp>`` plus a counter when a
#: same-second repeat would collide.  The stamp orders the evidence for a
#: reader; the counter is what guarantees nothing is ever overwritten.
_QUARANTINE_STAMP = '%Y%m%dT%H%M%S'


def quarantine_merge_rr(scan: MergeRrScan) -> Path | None:
    """Move a suspect MERGE_RR aside and return the backup path.

    A no-op returning ``None`` unless the scan is suspect: a healthy MERGE_RR
    is left exactly where git put it, untouched.

    The file is MOVED rather than deleted because it is the only record of
    which conflict ids the wedged worktree was carrying, and the abort that
    follows would otherwise destroy it — a successful ``git rebase --abort``
    deletes MERGE_RR outright.  Backup names never collide, so a worktree that
    wedges twice keeps both wedges' evidence rather than overwriting the first
    with the second.

    Logs at WARNING naming the dangling ids and the backup path: a repair that
    happens silently is indistinguishable from a repair that never ran.
    """
    if not scan.suspect:
        return None

    backup = _free_backup_path(scan.merge_rr_path)
    scan.merge_rr_path.rename(backup)
    logger.warning(
        'Quarantined suspect MERGE_RR to %s — dangling rr-cache refs: [%s]; '
        'unparsable records: %d. Evidence preserved; the abort that follows '
        'would have deleted it.',
        backup,
        ', '.join(record.conflict_id for record in scan.dangling),
        len(scan.unparsable),
    )
    return backup


def _free_backup_path(merge_rr_path: Path) -> Path:
    """First unused ``MERGE_RR.quarantined-<stamp>[-<n>]`` beside the original."""
    stamp = datetime.now(UTC).strftime(_QUARANTINE_STAMP)
    base = merge_rr_path.with_name(f'{merge_rr_path.name}.quarantined-{stamp}')
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
    """One ``*.lock`` and the two facts the removal decision turns on."""

    path: Path
    age_seconds: float
    holder_pids: tuple[int, ...]


@dataclass(frozen=True)
class LockSweep:
    """Which locks were cleared and which were deliberately left alone.

    Retained locks are returned rather than dropped: a lock this refused to
    touch is the reason a subsequent abort may still fail, so the caller can
    say so instead of leaving the operator to rediscover it.
    """

    removed: tuple[LockFinding, ...]
    retained: tuple[LockFinding, ...]


def lock_holder_pids(path: Path) -> tuple[int, ...]:
    """Pids holding *path* open, found by scanning ``/proc/<pid>/fd``.

    Deliberately NOT ``git_ops.lane_lock_holder_pids``, which reads
    ``/proc/locks``.  That file lists kernel FLOCK/POSIX locks, whereas git's
    ``*.lock`` files are plain ``O_CREAT|O_EXCL`` sentinels held open by file
    descriptor with no kernel lock at all — so it would report "no holder" for
    every live git lock, and this sweep would delete them.  The two probes
    answer different questions and neither substitutes for the other.

    Per-entry ``OSError`` is tolerated throughout: processes exit mid-scan and
    other users' fd directories are not ours to read.  Either way the answer
    for that pid is "cannot confirm it holds this", which is what skipping it
    records.
    """
    target = str(path.resolve())
    holders: list[int] = []
    for entry in Path('/proc').iterdir():
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
            if resolved == target:
                holders.append(int(entry.name))
                break
    return tuple(holders)


def survey_locks(
    git_dir: Path, *, now: datetime | None = None,
) -> tuple[LockFinding, ...]:
    """Measure every ``*.lock`` directly under *git_dir*, removing nothing.

    The observation half of :func:`sweep_stale_locks`, which consumes it and
    then partitions on the conjunction.  Keeping the measurement in one place
    means the report-only path and the removing path cannot drift into
    disagreeing about what they saw.
    """
    moment = now if now is not None else datetime.now(UTC)
    findings: list[LockFinding] = []
    for lock in sorted(git_dir.glob('*.lock')):
        try:
            age = moment.timestamp() - lock.stat().st_mtime
        except OSError:
            continue
        findings.append(
            LockFinding(
                path=lock, age_seconds=age, holder_pids=lock_holder_pids(lock),
            ),
        )
    return tuple(findings)


def sweep_stale_locks(
    *,
    git_dir: Path,
    now: datetime | None = None,
    stale_after_seconds: float = DEFAULT_LOCK_STALE_AFTER_SECONDS,
) -> LockSweep:
    """Remove abandoned ``*.lock`` files directly under *git_dir*.

    A lock is removed ONLY when nothing holds it open AND its mtime is older
    than *stale_after_seconds*.  The conjunction is the whole contract: age
    alone would delete a lock a live process still depends on, and a holder
    check alone would clear a lock the instant its writer blinked.

    The mtime of the operation the lock BLOCKS is never consulted — not as a
    tiebreak, not as a hint.  Incident 3517's lock was older than the
    rebase-merge directory it blocked, so "newer than the operation" clears a
    lock that must be kept and "older than the operation" keeps one that must
    be cleared; the relative comparison is wrong in both directions.

    Size is likewise not a liveness signal: git's lock files are empty for as
    long as they are held, so a 0-byte lock is exactly as likely to be live as
    it is to be abandoned.
    """
    removed: list[LockFinding] = []
    retained: list[LockFinding] = []

    for finding in survey_locks(git_dir, now=now):
        if finding.holder_pids or finding.age_seconds <= stale_after_seconds:
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
            'no holder process found in /proc.',
            finding.path, finding.age_seconds, stale_after_seconds,
        )

    return LockSweep(removed=tuple(removed), retained=tuple(retained))


#: Verdicts the CLI and the skills branch on.  ``blocked`` is the only one
#: that means "do not proceed": something still holds a lock open, so the abort
#: will hit git's "Another git process seems to be running" and a human has to
#: decide what that process is.
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

    @property
    def unrepaired(self) -> tuple[str, ...]:
        """Findings this run did NOT fix — the reason a caller must not proceed.

        Two arms.  A lock with a live holder is a human's decision: something
        is using it, and this module will not guess what.  A suspect MERGE_RR
        with no backup means the damage is still in place — the ``report_only``
        case, where leaving it is the whole point, and also a quarantine that
        failed.
        """
        reasons = [
            f'lock {finding.path} held by pids '
            f'{", ".join(str(pid) for pid in finding.holder_pids)}'
            for finding in self.locks_retained if finding.holder_pids
        ]
        if self.merge_rr_backup is None and (self.dangling or self.unparsable):
            reasons.append(
                'suspect MERGE_RR left in place — dangling rr-cache refs: ['
                + ', '.join(record.conflict_id for record in self.dangling)
                + f']; unparsable records: {len(self.unparsable)}',
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


def _lock_json(finding: LockFinding) -> dict:
    return {
        'path': str(finding.path),
        'age_seconds': round(finding.age_seconds, 3),
        'holder_pids': list(finding.holder_pids),
    }


def resolve_git_dirs(worktree: Path) -> tuple[Path, Path] | None:
    """Resolve *worktree*'s ``(git_dir, common_dir)``, or ``None`` if it cannot.

    The two differ in a linked worktree, which is the case that matters here:
    MERGE_RR is per-worktree while rr-cache is shared.  One ``git rev-parse``
    answers both, so this is the module's only subprocess.

    A cwd git cannot even be spawned in — a worktree deleted out-of-band, or a
    path that is a file — raises ``OSError`` from the spawn itself, before any
    exit code exists.  That is answered with ``None``, the same as a non-zero
    exit, so the caller takes the one unresolved-but-clean branch instead of
    two branches that differ only in how the worktree failed to exist.  It also
    keeps the vanished-worktree case reaching ``git_ops._run``, whose own
    pre-flight raises the typed ``WorktreeMissing`` its consumers match on;
    raising that here instead is impossible without an import cycle (see
    :data:`AbortRunner`) and would duplicate the class besides.
    """
    try:
        proc = subprocess.run(
            ['git', 'rev-parse', '--git-dir', '--git-common-dir'],
            cwd=str(worktree), capture_output=True, text=True, check=False,
        )
    except OSError as exc:
        logger.warning('Could not spawn git in %s: %s', worktree, exc)
        return None
    if proc.returncode != 0:
        return None
    lines = proc.stdout.splitlines()
    if len(lines) != 2:
        return None
    # Split on LINES, not whitespace: git prints one path per line and a git
    # dir may legally contain spaces.  Each path is relative to *worktree*
    # unless already absolute, which ``os.path.join`` handles either way.
    git_dir, common_dir = (Path(os.path.join(worktree, line)) for line in lines)
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
    operator can inspect before authorising a repair.

    FAIL-SAFE: a preflight that cannot resolve the git directories logs and
    returns an unresolved-but-clean result rather than raising.  This decorates
    a RECOVERY path, so it must never itself become the reason recovery fails —
    an unguarded abort that might crash still beats no abort at all.
    """
    worktree = Path(worktree)
    dirs = resolve_git_dirs(worktree)
    if dirs is None:
        logger.warning(
            'Rebase-recovery preflight could not resolve git dirs for %s; '
            'proceeding unguarded.', worktree,
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
            'at %s — dangling rr-cache refs: [%s]; unparsable records: %d. '
            'Nothing was moved.',
            scan.merge_rr_path,
            ', '.join(record.conflict_id for record in scan.dangling),
            len(scan.unparsable),
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
    blocking the event loop.  It never raises: a preflight that cannot resolve
    the worktree degrades to an unguarded abort, which is still strictly better
    than no abort at all.

    Returns *run*'s ``(rc, stdout, stderr)`` unchanged, so no call site's
    control flow, return value or logging has to change.
    """
    await asyncio.to_thread(preflight_rebase_recovery, cwd)
    return await run([*RECOVERY_GIT, verb, '--abort'], cwd=cwd)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

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
