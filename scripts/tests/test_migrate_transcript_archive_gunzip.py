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


# ---------------------------------------------------------------------------
# step-3: migrate_archive — idempotency and crash-resume (INV-7)
#
# The sweep must be safe to re-run. It walks ~4,554 files and an operator can
# kill it part-way, so every intermediate state it can be interrupted in has to
# be a state a later run recovers from correctly — including the nastiest one,
# where a run died between writing a twin and unlinking its source, leaving a
# HALF-WRITTEN twin next to a perfectly good .gz.
# ---------------------------------------------------------------------------

def _mtimes(root: Path) -> dict[Path, float]:
    """Snapshot every file's mtime under *root* — for proving a re-run is inert."""
    return {p: p.stat().st_mtime for p in sorted(root.rglob("*")) if p.is_file()}


def test_existing_good_twin_is_skipped_and_its_gz_unlinked(tmp_path):
    """(a) The resume case: a run killed between writing the twin and unlinking
    the source. The twin reads back cleanly, so it is trusted — classified
    `skipped`, NOT re-decompressed — and the redundant .gz is unlinked so the
    archive still converges to zero .gz."""
    payload = _payload()
    gz = _write_gz(tmp_path / "3618" / "enc" / "sess-a.jsonl.gz", payload)
    twin = tmp_path / "3618" / "enc" / "sess-a.jsonl"
    twin.write_bytes(payload)
    os.utime(twin, (ARCHIVED_MTIME, ARCHIVED_MTIME))
    before = twin.stat().st_mtime_ns

    summary = mig.migrate_archive(tmp_path, apply=True)

    assert summary["skipped"] == 1
    assert summary["migrated"] == 0
    assert summary["failed"] == 0
    # Not re-decompressed: an untouched mtime proves the good twin was trusted
    # rather than silently rewritten.
    assert twin.stat().st_mtime_ns == before
    assert twin.read_bytes() == payload
    assert not gz.exists()


def test_existing_unreadable_twin_is_re_gunzipped_from_the_authoritative_gz(tmp_path):
    """(b) A half-written twin must NEVER be trusted on mere existence. The .gz
    is the authoritative copy until corroborated, so an undecodable/partial twin
    is overwritten from it and only then is the source unlinked."""
    payload = _payload()
    gz = _write_gz(tmp_path / "3618" / "enc" / "sess-b.jsonl.gz", payload)
    twin = tmp_path / "3618" / "enc" / "sess-b.jsonl"
    # A partial prefix of the real payload, ending in a raw 0xFF — the shape a
    # killed write leaves behind: plausible-looking, but neither complete nor
    # decodable.
    twin.write_bytes(payload[: len(payload) // 2] + b"\xff")

    summary = mig.migrate_archive(tmp_path, apply=True)

    assert summary["migrated"] == 1
    assert summary["skipped"] == 0
    assert summary["failed"] == 0
    assert twin.read_bytes() == payload, "the bad twin was not rebuilt from the .gz"
    assert int(twin.stat().st_mtime) == int(ARCHIVED_MTIME)
    assert not gz.exists()


def test_second_run_over_a_migrated_tree_is_a_clean_no_op(tmp_path):
    """(c) Idempotency proper: re-running over a fully migrated tree migrates
    nothing, fails nothing, and does not touch a single file's mtime."""
    _write_gz(tmp_path / "3618" / "enc" / "sess-c.jsonl.gz", _payload())
    _write_gz(
        tmp_path / "3618" / "enc" / "sess-c" / "subagents" / "agent-1.jsonl.gz",
        _payload(3),
    )

    first = mig.migrate_archive(tmp_path, apply=True)
    assert first["migrated"] == 2

    before = _mtimes(tmp_path)
    second = mig.migrate_archive(tmp_path, apply=True)

    assert second["scanned"] == 0
    assert second["migrated"] == 0
    assert second["skipped"] == 0
    assert second["failed"] == 0
    assert _mtimes(tmp_path) == before, "a no-op re-run must not touch any file"


def test_plain_jsonl_without_a_gz_is_left_untouched(tmp_path):
    """(d) An already-plain transcript with no .gz at all is not the sweep's
    business — it is neither scanned, rewritten, nor re-stamped."""
    plain = tmp_path / "3618" / "enc" / "sess-d.jsonl"
    plain.parent.mkdir(parents=True)
    plain.write_bytes(_payload())
    os.utime(plain, (ARCHIVED_MTIME, ARCHIVED_MTIME))
    before = _mtimes(tmp_path)

    summary = mig.migrate_archive(tmp_path, apply=True)

    assert summary["scanned"] == 0
    assert summary["migrated"] == 0
    assert summary["skipped"] == 0
    assert _mtimes(tmp_path) == before


def test_summary_carries_distinct_counters_not_one_conflated_total(tmp_path):
    """The report separates CLASSIFICATION from ACTION (mirroring
    gc_agent_transcripts.build_gc_report): a run that skipped 1 and migrated 1
    must be distinguishable from one that migrated 2."""
    payload = _payload()
    _write_gz(tmp_path / "3618" / "enc" / "fresh.jsonl.gz", payload)
    _write_gz(tmp_path / "3618" / "enc" / "resumed.jsonl.gz", payload)
    twin = tmp_path / "3618" / "enc" / "resumed.jsonl"
    twin.write_bytes(payload)

    summary = mig.migrate_archive(tmp_path, apply=True)

    assert summary["scanned"] == 2
    assert summary["migrated"] == 1
    assert summary["skipped"] == 1
    assert summary["failed"] == 0
    assert summary["failed_paths"] == []


def test_absent_root_is_a_clean_empty_no_op(tmp_path):
    """An absent root is an empty sweep, not a crash."""
    summary = mig.migrate_archive(tmp_path / "does-not-exist", apply=True)

    assert summary["scanned"] == 0
    assert summary["migrated"] == 0
    assert summary["failed"] == 0


def test_non_directory_root_is_a_clean_empty_no_op(tmp_path):
    """A root that exists but is a FILE is an empty sweep, not a crash."""
    root = tmp_path / "not-a-dir"
    root.write_text("x")

    summary = mig.migrate_archive(root, apply=True)

    assert summary["scanned"] == 0
    assert summary["migrated"] == 0
    assert summary["failed"] == 0
