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

1. stream-decompress ``<name>.jsonl.gz`` to the sibling ``<name>.jsonl``
   (chunked, so peak RSS stays bounded over multi-MB transcripts);
2. close, then RE-OPEN the written file and read it back, corroborating that it
   is both fully written and decodable;
3. mirror the ``.gz`` mtime onto it via ``os.utime``;
4. ONLY THEN unlink the source ``.gz``.

A file that cannot be corroborated is NEVER destroyed. Step 2 is not
ceremonial: a short write (ENOSPC part-way through a ~1.4 GB expansion) yields
a plausible-looking but truncated transcript, and unlinking the authoritative
source on top of that would be unrecoverable data loss.

Step 3 is not cosmetic either. ``gc_agent_transcripts.scan_task_dirs`` derives
each task dir's retention age from its NEWEST descendant file mtime, and the
``.gz`` already carries the SOURCE transcript's mtime (the writer mirrors it).
Letting the gunzip stamp ``now`` on every file would silently reset the entire
90-day retention window in a single pass — nothing would age out for another
90 days.

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

# Exit codes. DIVERGES from gc_agent_transcripts' unconditional exit 0: that is
# a routine cron sweep whose per-dir hiccups must not alarm a watchdog, whereas
# this is a one-off migration whose failures an operator has to act on. A
# non-zero failure count is an exit code, not a warning buried in a log.
EXIT_OK = 0
EXIT_FAILURES = 1

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


def _decompress_to(gz_path: Path, dest: Path) -> int:
    """Stream *gz_path* into *dest*, returning the byte count written.

    Chunked through :func:`shutil.copyfileobj` rather than a single ``read()``
    so peak RSS stays bounded regardless of transcript size. The returned count
    is what :func:`_corroborate` checks the on-disk size against, which is how a
    short write (ENOSPC) is caught before the source is destroyed.
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
    """Remove a partially-written destination after a failed migration.

    Not tidiness — correctness. :func:`migrate_archive` decides whether to
    trust an existing twin, and a half-written file left behind by a failure
    is exactly the debris a later resume run could mistake for a finished
    migration and then unlink the authoritative ``.gz`` on top of.
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

    Implements the module's ordering contract exactly: decompress → corroborate
    → mirror mtime → unlink. The source ``.gz`` is removed ONLY after the plain
    twin has been read back and stamped; any exception before that point leaves
    the authoritative source untouched and clears the partial destination.

    Raises the underlying failure rather than swallowing it — classification
    and counting belong to :func:`migrate_archive`, which owns the report.
    """
    dest = plain_sibling(gz_path)
    # Capture the source mtime BEFORE any mutation — it is the retention age
    # gc_agent_transcripts keys on, and the source is gone by the end.
    source_stat = gz_path.stat()

    try:
        written = _decompress_to(gz_path, dest)      # (1) decompress
        _corroborate(dest, expected_size=written)    # (2) read back
        os.utime(dest, (source_stat.st_atime, source_stat.st_mtime))  # (3) mtime
    except _UNREADABLE_SOURCE_ERRORS:
        _discard_partial(dest)
        raise
    gz_path.unlink()                                 # (4) only now destroy
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
        'failed': 0,
        'failed_paths': [],
    }
    if not root.is_dir():
        return summary

    for gz_path in sorted(root.rglob('*.jsonl.gz')):
        summary['scanned'] += 1
        twin = plain_sibling(gz_path)

        try:
            if twin.exists() and _twin_is_trustworthy(gz_path, twin):
                if apply:
                    gz_path.unlink()
                summary['skipped'] += 1
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
    re-runnable after the operator investigates.
    """
    logging.basicConfig(level=logging.INFO)
    args = build_parser().parse_args(argv)

    root = Path(args.root)
    summary = migrate_archive(root, apply=args.apply)
    print(json.dumps(summary))

    logger.info(
        '%s migration complete — root=%s scanned=%d migrated=%d skipped=%d '
        'failed=%d apply=%s',
        _LOG_PREFIX,
        root,
        summary['scanned'],
        summary['migrated'],
        summary['skipped'],
        summary['failed'],
        args.apply,
    )
    if not args.apply:
        logger.info(
            '%s DRY-RUN — nothing was changed; re-run with --apply to migrate.',
            _LOG_PREFIX,
        )
    return EXIT_FAILURES if summary['failed'] else EXIT_OK


if __name__ == '__main__':
    sys.exit(main())
