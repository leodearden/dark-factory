#!/usr/bin/env python3
"""One-off migration: decompress the agent-transcript archive in place.

Task 3618 (leaf α of plans/transcript-preservation-seam-prd.md). The α producer
hook (:mod:`shared.transcript_archive`) historically gzipped each finished agent
session's transcripts into a durable archive root outside the per-task worktree,
laid out as ``<root>/<task_id>/<enc>/<session_id>.jsonl.gz`` (with nested
subagent transcripts at ``<sid>/subagents/agent-<hex>.jsonl.gz``). That bought
disk at the cost of making the corpus opaque to ``rg`` and forcing every reader
to carry a gzip branch. This sweep converts the existing archive to ONE plain,
greppable ``.jsonl`` corpus so those branches can be deleted.

The ordering contract (INV-3 corroborate-before-destroy)
--------------------------------------------------------
:func:`gunzip_one` executes STRICTLY in this order, and the order is the whole
point of the module:

1. stream-decompress ``<name>.jsonl.gz`` to a STAGING sibling
   ``<name>.jsonl.migrate-tmp`` (chunked, so peak RSS stays bounded over
   multi-MB transcripts);
2. close, then RE-OPEN the written file and read it back, corroborating that it
   is both fully written and decodable;
3. mirror the ``.gz`` mtime onto it via ``os.utime``;
4. ``os.replace`` the staging file over ``<name>.jsonl`` — atomic, same
   directory, therefore same filesystem;
5. ONLY THEN unlink the source ``.gz``.

A file that cannot be corroborated is NEVER destroyed. Step 2 is not
ceremonial: a short write (ENOSPC part-way through a ~1.4 GB expansion) yields
a plausible-looking but truncated transcript, and unlinking the authoritative
source on top of that would be unrecoverable data loss.

Step 1 stages rather than writing ``<name>.jsonl`` DIRECTLY, and that is the
destination half of the same invariant. Opening the final path ``'wb'`` would
truncate a PRE-EXISTING plain twin before a single source byte had been
validated, and the failure path would then unlink the remains — so a damaged
``.gz`` sitting beside a healthy plain twin would cost BOTH copies and lose the
session's transcript entirely. That is reachable on the documented operator
path: the runbook instructs a re-run after the fleet redeploy, exactly when
both forms coexist, and the resumed-session guard below cannot catch it because
a damaged stream has no trustworthy ISIZE trailer to compare against. Staging
makes the destination untouched until a complete, corroborated, correctly
stamped file exists to replace it atomically.

Step 3 is not cosmetic either. ``gc_agent_transcripts.scan_task_dirs`` derives
each task dir's retention age from its NEWEST descendant file mtime, and the
``.gz`` already carries the SOURCE transcript's mtime (the writer mirrors it).
Letting the gunzip stamp ``now`` on every file would silently reset the entire
90-day retention window in a single pass — nothing would age out for another
90 days.

The conflict rule (when the PLAIN file is the authoritative one)
----------------------------------------------------------------
The ``.gz`` is authoritative only until something longer turns up next to it.
Re-running this sweep after a fleet redeploy — which the runbook explicitly
instructs, to clear transition-window residue — reaches sessions that were in
flight ACROSS that redeploy and so have BOTH forms on disk: the pre-redeploy
process archived ``<sid>.jsonl.gz``, then the post-redeploy producer hook wrote
a plain, RESUMED and therefore LONGER ``<sid>.jsonl``. Gunzipping over that
would overwrite the live transcript with stale content and roll its mtime
backwards. So a twin that does not corroborate but is LONGER than the ``.gz`` is
a CONFLICT: both copies are retained untouched, it is counted and named, and
the run exits :data:`EXIT_CONFLICTS`.

SIZE is the signal, never mtime. Both shapes that reach that branch are NEWER
than the archived ``.gz`` — a killed write stamped ``now`` because its
``os.utime`` mirror had not run yet — so mtime cannot separate them. Direction
of size can: a killed write leaves a strict PREFIX (shorter, rebuild it), an
append-only resume leaves a strict SUPERSET (longer, keep it).

The ``.gz`` is deliberately NOT unlinked on that path. A size comparison
establishes only that the twin is longer, never that it CONTAINS the archived
content, and deleting on that basis would be the same unverified destruction
this module exists to prevent. One residual ``.gz`` and a line of operator
attention is the right price.

This module is STDLIB-ONLY by design, mirroring the deliberate no-repo-imports
posture of :mod:`gc_agent_transcripts` (a sweep over this very same tree), so it
runs standalone in any environment with no config load and no PYTHONPATH setup.
"""

from __future__ import annotations

import argparse
import gzip
import json
import logging
import os
import shutil
import sys
import zlib
from pathlib import Path

logger = logging.getLogger('migrate_transcript_archive_gunzip')

# Stable greppable prefix on every operator-facing log line.
_LOG_PREFIX = 'migrate_transcript_archive_gunzip:'

# Copy chunk size: bounds peak RSS independently of transcript size.
_CHUNK_BYTES = 1 << 20

# Mirrors gc_agent_transcripts' constants for the same archive tree — the
# archive root is a durable, git-ignored data dir under project_root, NOT the
# per-task worktree.
DEFAULT_PROJECT_ROOT = Path('/home/leo/src/dark-factory')
ARCHIVE_ROOT_RELATIVE = 'data/orchestrator/agent-transcripts'
DEFAULT_ARCHIVE_ROOT = DEFAULT_PROJECT_ROOT / ARCHIVE_ROOT_RELATIVE

GZ_SUFFIX = '.gz'

# Suffix for the staging file each decompression is written to before being
# atomically moved onto its final name. Deliberately NOT ``.jsonl``-terminated,
# so a leftover from a hard kill is invisible to both this sweep's
# ``*.jsonl.gz`` walk and every downstream reader's ``*.jsonl`` glob — inert
# debris, never mistakable for a transcript. A re-run rewrites it in place.
STAGING_SUFFIX = '.migrate-tmp'

# Exit codes. DIVERGES from gc_agent_transcripts' unconditional exit 0: that is
# a routine cron sweep whose per-dir hiccups must not alarm a watchdog, whereas
# this is a one-off migration whose failures an operator has to act on. A
# non-zero failure count is an exit code, not a warning buried in a log.
EXIT_OK = 0
EXIT_FAILURES = 1
# A DISTINCT code for "needs adjudication", because it is not "is damaged": a
# conflict means a plain twin that is LONGER than its .gz was left in place and
# both copies retained. It cannot stay 0 — a conflict leaves a .gz on disk,
# which contradicts the runbook's `expect 0` confirmation.
EXIT_CONFLICTS = 2

# Every way reading a real gzip stream (or its decoded text) can fail.
#
# Deliberately WIDER than the post-3618 reader-side
# ``inventory.UNREADABLE_FILE_ERRORS``: after this task the readers only ever
# open plain ``.jsonl``, so the two non-OSError gzip shapes below are
# unreachable there and that tuple narrows to ``(UnicodeDecodeError,)``. This
# script is the ONE component that still reads real gzip streams, which is
# exactly what licenses the narrowing — so the shapes have to be caught HERE.
#
# * ``OSError``      — covers ``gzip.BadGzipFile`` (a subclass) plus ordinary
#                      I/O failures: EACCES on the destination, EIO mid-read.
# * ``EOFError``     — a stream that ends before its end-of-stream marker (the
#                      interrupted-write shape). NOT an OSError.
# * ``zlib.error``   — an unparseable DEFLATE body. Also NOT an OSError.
# * ``UnicodeDecodeError`` — a structurally valid stream whose decompressed
#                      bytes are not UTF-8. A ``ValueError`` subclass, so an
#                      ``except OSError``-only handler would miss it entirely.
_UNREADABLE_SOURCE_ERRORS = (OSError, EOFError, zlib.error, UnicodeDecodeError)


def default_archive_root() -> Path:
    """Return the default archive root (``project_root / ARCHIVE_ROOT_RELATIVE``).

    The single resolution site for the CLI ``--root`` default, mirroring
    :func:`gc_agent_transcripts.default_archive_root`. ``--root`` overrides it;
    there is no env knob (the archive root is a fixed data-dir convention).
    """
    return DEFAULT_ARCHIVE_ROOT


def plain_sibling(gz_path: Path) -> Path:
    """Return the plain ``.jsonl`` path a ``.jsonl.gz`` migrates to.

    Strips exactly the trailing ``.gz``, leaving the rest of the name intact —
    so ``sess-a.jsonl.gz`` becomes ``sess-a.jsonl`` in the SAME directory. The
    archive layout is preserved verbatim; only the suffix is dropped.
    """
    if gz_path.name.endswith(GZ_SUFFIX):
        return gz_path.with_name(gz_path.name[: -len(GZ_SUFFIX)])
    return gz_path


def staging_path(dest: Path) -> Path:
    """Return the staging path a decompression is written to before *dest*.

    A sibling in the SAME directory, which is what makes the final
    :func:`os.replace` an atomic same-filesystem rename rather than a copy.
    """
    return dest.with_name(dest.name + STAGING_SUFFIX)


def _decompress_to(gz_path: Path, dest: Path) -> int:
    """Stream *gz_path* into *dest*, returning the byte count written.

    Chunked through :func:`shutil.copyfileobj` rather than a single ``read()``
    so peak RSS stays bounded regardless of transcript size. The returned count
    is what :func:`_corroborate` checks the on-disk size against, which is how a
    short write (ENOSPC) is caught before the source is destroyed.

    *dest* is always a STAGING path (see :func:`gunzip_one`) — this opens it
    ``'wb'``, which truncates, so it must never be handed a live transcript.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(gz_path, 'rb') as src, open(dest, 'wb') as out:
        shutil.copyfileobj(src, out, _CHUNK_BYTES)
    return dest.stat().st_size


def _corroborate(path: Path, expected_size: int | None = None) -> None:
    """Re-open *path* and read it back, raising if it is not trustworthy.

    Two independent failure modes, both of which must block the unlink:

    * **short/partial** — when *expected_size* is given, the on-disk size must
      match it exactly. Catches a write that ran out of disk part-way and left
      a plausible-looking prefix of a real transcript.
    * **undecodable** — the bytes must read back under STRICT ``utf-8``. The
      whole point of the migration is a corpus every reader can open as text;
      a file that cannot be decoded is not a successful migration.

    Raises ``OSError``/``UnicodeDecodeError`` (both handled by the caller's
    per-file guard) rather than returning a flag, so an un-corroborated file
    can never fall through to the unlink by accident.
    """
    actual = path.stat().st_size
    if expected_size is not None and actual != expected_size:
        raise OSError(
            f'short write: {path} is {actual} bytes, expected {expected_size}'
        )
    with open(path, encoding='utf-8') as handle:
        while handle.read(_CHUNK_BYTES):
            pass


def _warn_migrate_failure(path: Path, err: BaseException) -> None:
    """LOUD, greppable WARNING for a file that could not be migrated.

    Modelled on :func:`gc_agent_transcripts._warn_stat_skip`: lazy
    %-formatting, the greppable ``migrate_transcript_archive_gunzip:`` prefix
    first, and an explicit ``errno=`` where the exception carries one (the
    non-OSError gzip shapes do not). The path is named so an operator can act
    on the specific file rather than just seeing a count.
    """
    logger.warning(
        '%s failed to migrate %s (errno=%s): %s: %s — source RETAINED',
        _LOG_PREFIX,
        path,
        getattr(err, 'errno', None),
        type(err).__name__,
        err,
    )


def _discard_partial(dest: Path) -> None:
    """Remove a partially-written STAGING file after a failed migration.

    Not tidiness — correctness. :func:`migrate_archive` decides whether to
    trust an existing twin, and a half-written file left behind by a failure
    is exactly the debris a later resume run could mistake for a finished
    migration and then unlink the authoritative ``.gz`` on top of.

    Only ever called with a staging path, NEVER the final ``.jsonl``: the
    caller has not yet replaced the destination when this runs, so a
    pre-existing twin is still whole and must stay that way.
    """
    try:
        dest.unlink(missing_ok=True)
    except OSError as err:  # pragma: no cover - cleanup is itself best-effort
        logger.warning(
            '%s failed to remove partial %s (errno=%s): %s',
            _LOG_PREFIX,
            dest,
            err.errno,
            err,
        )


def gunzip_one(gz_path: Path) -> Path:
    """Migrate one ``.jsonl.gz`` to its plain sibling, returning the new path.

    Implements the module's ordering contract exactly: decompress to staging →
    corroborate → mirror mtime → atomically replace → unlink. The source ``.gz``
    is removed ONLY after the plain twin has been read back and stamped; any
    exception before that point leaves the authoritative source untouched, the
    destination untouched, and clears only the staging file.

    Nothing is ever written over ``dest`` in place. Until the ``os.replace``,
    an existing plain twin is whole — so a source that turns out to be damaged
    costs neither copy. This is the destination half of corroborate-before-
    destroy, and it is the reason the write goes to staging at all.

    Raises the underlying failure rather than swallowing it — classification
    and counting belong to :func:`migrate_archive`, which owns the report.
    """
    dest = plain_sibling(gz_path)
    staging = staging_path(dest)
    # Capture the source mtime BEFORE any mutation — it is the retention age
    # gc_agent_transcripts keys on, and the source is gone by the end.
    source_stat = gz_path.stat()

    try:
        written = _decompress_to(gz_path, staging)      # (1) decompress
        _corroborate(staging, expected_size=written)    # (2) read back
        os.utime(staging, (source_stat.st_atime, source_stat.st_mtime))  # (3) mtime
    except _UNREADABLE_SOURCE_ERRORS:
        _discard_partial(staging)
        raise
    # (4) atomic same-dir rename. The mtime mirror rides across it, so the
    # published file is stamped correctly from the instant it appears — there
    # is no window where a reader sees a `now`-stamped transcript.
    os.replace(staging, dest)
    gz_path.unlink()                                    # (5) only now destroy
    return dest


def _gz_uncompressed_size(gz_path: Path) -> int | None:
    """Read the gzip ISIZE trailer — the uncompressed length, modulo 2**32.

    Cheap (a 4-byte tail read, no decompression) and authoritative, which makes
    it the right yardstick for deciding whether a PRE-EXISTING twin is complete.
    Returns ``None`` when the file is too small to carry a trailer.

    The modulo is harmless here: transcripts are megabytes, and the only way the
    wrap could mislead is a twin whose size happens to be congruent mod 4 GiB.
    A spurious MISmatch simply costs a redundant re-decompress — the failure
    mode is extra work, never false trust.
    """
    with open(gz_path, 'rb') as handle:
        if handle.seek(0, os.SEEK_END) < 8:
            return None
        handle.seek(-4, os.SEEK_END)
        return int.from_bytes(handle.read(4), 'little')


def _twin_is_trustworthy(gz_path: Path, twin: Path) -> bool:
    """Is an EXISTING plain *twin* good enough to skip re-decompressing?

    Applies the SAME read-back corroboration a fresh write gets — never mere
    existence. A run killed between writing a twin and unlinking its source
    leaves a half-written file that looks exactly like a finished one from the
    outside; trusting it would silently truncate a transcript while deleting
    the only authoritative copy.
    """
    try:
        _corroborate(twin, expected_size=_gz_uncompressed_size(gz_path))
    except (OSError, UnicodeDecodeError):
        return False
    return True


def _twin_supersedes_gz(gz_path: Path, twin: Path) -> bool:
    """Does an UN-trustworthy *twin* carry MORE data than the ``.gz``?

    Applied ONLY after :func:`_twin_is_trustworthy` has already returned False,
    to separate the two very different shapes that land there:

    * a **half-written** twin — a run killed mid-write left a strict PREFIX of
      the payload. The ``.gz`` is authoritative; rebuild from it.
    * a **resumed-session** twin — the plain file is LONGER than the archive
      because a session resumed and appended to it. It is authoritative;
      overwriting it from the ``.gz`` is unrecoverable data loss.

    SIZE IS THE DISCRIMINATOR, NOT MTIME. Both shapes are NEWER than the
    archived ``.gz``: the killed write stamped ``now`` because its
    :func:`os.utime` mirror had not run yet, and a resumed session is genuinely
    newer. Only the DIRECTION of size separates them — a killed write is
    strictly shorter, an append-only resume strictly longer. Triggering on
    mtime would misclassify the half-written twin and strand a truncated
    transcript.

    A ``None`` ISIZE (a sub-8-byte ``.gz``, too small to carry a trailer) makes
    no comparison possible, so this does NOT claim a conflict: that file fails
    decompression and lands in ``failed`` where it belongs.
    """
    isize = _gz_uncompressed_size(gz_path)
    if isize is None:
        return False
    return twin.stat().st_size > isize


def _warn_migrate_conflict(gz_path: Path, twin: Path) -> None:
    """LOUD, greppable WARNING for a twin that supersedes its ``.gz``.

    Modelled on :func:`_warn_migrate_failure`: lazy %-formatting and the
    greppable prefix first. Names BOTH paths with their sizes and mtimes —
    mtime is reported as EVIDENCE for the operator, never as the decision — and
    states plainly that both copies were retained, so the line is actionable on
    its own rather than pointing at a counter.
    """
    gz_stat = gz_path.stat()
    twin_stat = twin.stat()
    logger.warning(
        '%s CONFLICT: %s (%d bytes, mtime=%s) is LONGER than the archived %s '
        '(%d bytes compressed, mtime=%s) — a session resumed after this .gz was '
        'written. BOTH RETAINED for adjudication; the plain file is the '
        'authoritative copy.',
        _LOG_PREFIX,
        twin,
        twin_stat.st_size,
        twin_stat.st_mtime,
        gz_path,
        gz_stat.st_size,
        gz_stat.st_mtime,
    )


def verify_source_readable(gz_path: Path) -> None:
    """Stream *gz_path* through decompression and discard it, raising on damage.

    The dry-run counterpart of :func:`gunzip_one`: it exercises the exact same
    read path (decompress + strict UTF-8 decode) while writing nothing at all.

    This is what makes ``--apply``-less runs REAL validation rather than a file
    count. A dry run that only counted ``.gz`` entries would report
    ``failed == 0`` structurally — just as happily over a tree of corrupt files
    as a healthy one — and the operator's only full-scale rehearsal before
    deleting thousands of live sources would prove nothing.
    """
    with gzip.open(gz_path, 'rt', encoding='utf-8') as handle:
        while handle.read(_CHUNK_BYTES):
            pass


def migrate_archive(root: Path, *, apply: bool) -> dict:
    """Sweep *root*, migrating every ``.jsonl.gz`` to its plain sibling.

    Walks ``root.rglob('*.jsonl.gz')`` in sorted order so a run is deterministic
    and an interrupted one resumes over the same sequence. Each source is
    classified before any work happens:

    * an existing twin that corroborates -> ``skipped`` (the crash-resume path);
      its now-redundant ``.gz`` is still unlinked, so the archive converges to
      zero ``.gz`` even across a killed run;
    * an un-trustworthy twin that is nevertheless LONGER than the ``.gz``
      (:func:`_twin_supersedes_gz`) -> ``conflicts``. Nothing is written, no
      mtime is touched and the ``.gz`` is NOT unlinked: both copies are left for
      the operator to adjudicate;
    * anything else -> :func:`gunzip_one`, counted ``migrated``.

    ``apply=False`` is a pure projection: it classifies and counts without
    touching a single byte, so an operator can see exactly what a real run
    would do first. It still READS every source through
    :func:`verify_source_readable`, so a projected ``failed`` count is real
    evidence about the archive rather than a structural zero. An absent or
    non-directory *root* is a clean empty sweep, not a crash.

    Returns a summary keeping CLASSIFICATION distinct from ACTION (mirroring
    :func:`gc_agent_transcripts.build_gc_report`), so "skipped 1, migrated 1" is
    never conflated with "migrated 2".
    """
    summary = {
        'root': str(root),
        'apply': apply,
        'scanned': 0,
        'migrated': 0,
        'skipped': 0,
        'conflicts': 0,
        'conflict_paths': [],
        'failed': 0,
        'failed_paths': [],
    }
    if not root.is_dir():
        return summary

    for gz_path in sorted(root.rglob('*.jsonl.gz')):
        summary['scanned'] += 1
        twin = plain_sibling(gz_path)

        try:
            if twin.exists():
                # Corroboration FIRST: a complete twin killed before its utime
                # mirror is newer than the .gz yet exactly right, and must still
                # resume cleanly.
                if _twin_is_trustworthy(gz_path, twin):
                    if apply:
                        gz_path.unlink()
                    summary['skipped'] += 1
                    continue

                # Classified on the dry-run path too, so a projection reports
                # conflicts without mutating anything.
                if _twin_supersedes_gz(gz_path, twin):
                    _warn_migrate_conflict(gz_path, twin)
                    summary['conflicts'] += 1
                    summary['conflict_paths'].append(str(gz_path))
                    continue

            if apply:
                gunzip_one(gz_path)
            else:
                verify_source_readable(gz_path)
            summary['migrated'] += 1
        except _UNREADABLE_SOURCE_ERRORS as err:
            # Never re-raise: one bad file must not abort a sweep over ~4,554
            # siblings. The source .gz is still on disk (gunzip_one unlinks it
            # only after corroborating), which is the whole point.
            _warn_migrate_failure(gz_path, err)
            summary['failed'] += 1
            summary['failed_paths'].append(str(gz_path))

    return summary


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI arg parser (unit-testable in isolation).

    ``--root`` defaults to :func:`default_archive_root`. ``--apply`` opts INTO
    mutation: this follows the MUTATOR house convention (dry-run by default,
    as ``repair_wiped_metadata_files.py`` and the fused-memory migrations do)
    rather than :mod:`gc_agent_transcripts`' ``--check`` sweep convention,
    because a mis-invocation here deletes thousands of source files.
    """
    parser = argparse.ArgumentParser(
        prog='migrate_transcript_archive_gunzip.py',
        description=(
            'Decompress the agent-transcript archive in place, leaving one '
            'plain, greppable .jsonl corpus (idempotent, LOUD, dry-run by '
            'default).'
        ),
    )
    parser.add_argument(
        '--root',
        default=str(default_archive_root()),
        help='Archive root to migrate (default: %(default)s).',
    )
    parser.add_argument(
        '--apply',
        action='store_true',
        help=(
            'Actually migrate. WITHOUT this flag the run is a dry-run: it '
            'classifies and reports what it would do, and changes nothing.'
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the migration: walk → migrate (or project) → report → exit.

    The JSON summary goes to stdout and the LOUD lines to stderr, so the report
    can be piped into ``jq`` while failures stay visible. Exits
    :data:`EXIT_FAILURES` if any file could not be migrated — every such source
    ``.gz`` is still on disk and named in ``failed_paths``, so the run is
    re-runnable after the operator investigates — or :data:`EXIT_CONFLICTS` if
    the only outstanding work is adjudication. ``failed`` takes precedence when
    both are present: damage is the more urgent of the two.
    """
    logging.basicConfig(level=logging.INFO)
    args = build_parser().parse_args(argv)

    root = Path(args.root)
    summary = migrate_archive(root, apply=args.apply)
    print(json.dumps(summary))

    logger.info(
        '%s migration complete — root=%s scanned=%d migrated=%d skipped=%d '
        'conflicts=%d failed=%d apply=%s',
        _LOG_PREFIX,
        root,
        summary['scanned'],
        summary['migrated'],
        summary['skipped'],
        summary['conflicts'],
        summary['failed'],
        args.apply,
    )
    if not args.apply:
        logger.info(
            '%s DRY-RUN — nothing was changed; re-run with --apply to migrate.',
            _LOG_PREFIX,
        )
    if summary['failed']:
        return EXIT_FAILURES
    return EXIT_CONFLICTS if summary['conflicts'] else EXIT_OK


if __name__ == '__main__':
    sys.exit(main())
