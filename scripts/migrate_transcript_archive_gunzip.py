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

import gzip
import logging
import os
import shutil
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


def gunzip_one(gz_path: Path) -> Path:
    """Migrate one ``.jsonl.gz`` to its plain sibling, returning the new path.

    Implements the module's ordering contract exactly: decompress → corroborate
    → mirror mtime → unlink. The source ``.gz`` is removed ONLY after the plain
    twin has been read back and stamped; any exception before that point leaves
    the authoritative source untouched.
    """
    dest = plain_sibling(gz_path)
    # Capture the source mtime BEFORE any mutation — it is the retention age
    # gc_agent_transcripts keys on, and the source is gone by the end.
    source_stat = gz_path.stat()

    written = _decompress_to(gz_path, dest)          # (1) decompress
    _corroborate(dest, expected_size=written)        # (2) read back
    os.utime(dest, (source_stat.st_atime, source_stat.st_mtime))  # (3) mtime
    gz_path.unlink()                                 # (4) only now destroy
    return dest
