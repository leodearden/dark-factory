"""Tests for scripts/migrate_transcript_archive_gunzip.py — the one-off sweep
that decompresses the agent-transcript archive in place, leaving one plain,
greppable `.jsonl` corpus (task 3618, leaf α of
plans/transcript-preservation-seam-prd.md).

step-1: the CORROBORATE-BEFORE-DESTROY contract (INV-3). `gunzip_one(gz)`
decompresses to the sibling `.jsonl`, re-opens and reads the result back,
mirrors the `.gz` mtime, and only THEN unlinks the source. The mtime mirror is
a first-class assertion, not an incidental: `gc_agent_transcripts.scan_task_dirs`
derives each task dir's retention age from its NEWEST descendant mtime, so a
migration that stamped `now` on all 4,554 files would silently reset the whole
90-day retention window in a single pass.
"""
from __future__ import annotations

import gzip
import json
import os
from pathlib import Path

import migrate_transcript_archive_gunzip as mig

LOG_PREFIX = "migrate_transcript_archive_gunzip:"

DAY = 86_400
NOW = 1_000_000_000.0

# A distinctive PAST mtime, far enough from "now" that a migration which
# stamped the current clock instead of mirroring the source would be visibly
# wrong rather than merely imprecise.
ARCHIVED_MTIME = NOW - 30 * DAY


def _payload(n_records: int = 20) -> bytes:
    """Serialize a multi-record JSONL body — the exact bytes we compress and
    then expect back byte-for-byte after the round-trip."""
    return "".join(
        json.dumps({"type": "user", "seq": i, "pad": "x" * 50}) + "\n"
        for i in range(n_records)
    ).encode("utf-8")


def _write_gz(path: Path, payload: bytes, mtime: float = ARCHIVED_MTIME) -> Path:
    """Write *payload* as a valid gz at *path*, stamped with *mtime*.

    Mirrors the real writer (shared/src/shared/transcript_archive.py), which
    `os.utime`s each archived copy with the SOURCE transcript's mtime — which
    is why mirroring the `.gz` mtime onto the `.jsonl` is the correct thing for
    the migration to do.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wb") as handle:
        handle.write(payload)
    os.utime(path, (mtime, mtime))
    return path


# ---------------------------------------------------------------------------
# step-1: gunzip_one — round-trip exactness, mtime mirror, source removal
# ---------------------------------------------------------------------------

def test_gunzip_one_writes_byte_identical_plain_sibling(tmp_path):
    """(a) The produced `.jsonl` is BYTE-IDENTICAL to the pre-compression
    payload. DEFLATE is lossless by construction, so this is exact equality —
    not a tolerance."""
    payload = _payload()
    gz = _write_gz(tmp_path / "3618" / "enc" / "sess-a.jsonl.gz", payload)

    mig.gunzip_one(gz)

    plain = tmp_path / "3618" / "enc" / "sess-a.jsonl"
    assert plain.exists(), "the sibling .jsonl was not created"
    assert plain.read_bytes() == payload


def test_gunzip_one_mirrors_the_gz_mtime_onto_the_plain_file(tmp_path):
    """(b) The retention-age property `gc_agent_transcripts.scan_task_dirs`
    keys on survives the migration. Stamping `now` here would silently reset
    the entire 90-day retention window across the archive."""
    gz = _write_gz(tmp_path / "3618" / "enc" / "sess-b.jsonl.gz", _payload())
    gz_mtime = gz.stat().st_mtime

    mig.gunzip_one(gz)

    plain = tmp_path / "3618" / "enc" / "sess-b.jsonl"
    assert int(plain.stat().st_mtime) == int(gz_mtime)
    assert int(plain.stat().st_mtime) == int(ARCHIVED_MTIME)


def test_gunzip_one_unlinks_the_source_after_corroborating(tmp_path):
    """(c) The source `.gz` is gone once the plain twin is written, read back
    and stamped — the destroy half of corroborate-before-destroy."""
    gz = _write_gz(tmp_path / "3618" / "enc" / "sess-c.jsonl.gz", _payload())

    mig.gunzip_one(gz)

    assert not gz.exists(), "the source .gz should be unlinked after a good migration"
    assert (tmp_path / "3618" / "enc" / "sess-c.jsonl").exists()


def test_gunzip_one_migrates_the_nested_subagent_layout(tmp_path):
    """(d) The deeper `<sid>/subagents/agent-<hex>.jsonl.gz` layout migrates
    identically — the archive nests subagent transcripts one level below the
    main session file, and both shapes must be swept."""
    payload = _payload(5)
    gz = _write_gz(
        tmp_path / "3618" / "enc" / "sess-d" / "subagents" / "agent-1a2b.jsonl.gz",
        payload,
    )

    mig.gunzip_one(gz)

    plain = tmp_path / "3618" / "enc" / "sess-d" / "subagents" / "agent-1a2b.jsonl"
    assert plain.read_bytes() == payload
    assert int(plain.stat().st_mtime) == int(ARCHIVED_MTIME)
    assert not gz.exists()
